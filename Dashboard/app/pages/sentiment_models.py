# -*- coding: utf-8 -*-
"""
Sentiment Models page – Financial NLP Dashboard

Provides three tabs:
  1. Metrics   – side-by-side performance comparison (confusion matrices,
                 classification reports, accuracy & F1 bar charts).
  2. BiLSTM    – interactive inference with a BiLSTM + GloVe model.
  3. BERT      – interactive inference with DistilBERT / FinBERT / BERT Base.
"""

import os
import json
import numpy as np
import h5py
import torch
import tensorflow as tf
import plotly.graph_objects as go
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig,
    BertTokenizer, BertForSequenceClassification, BertConfig,
)
import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/sentiment-models")

# ─────────────────────────────────────────────
# Configuration & paths
# ─────────────────────────────────────────────

# Resolve project root directories
DASHBOARD_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
TEXT_M_ROOT = os.path.dirname(DASHBOARD_ROOT)

# BERT / Transformer weight files
BERT_WEIGHTS_DIR = os.path.join(TEXT_M_ROOT, "Bert_weights", "bert_checkpoints_256token")
DISTILBERT_WEIGHTS = os.path.join(BERT_WEIGHTS_DIR, "bert_best_model_256token.pt")
FINBERT_WEIGHTS    = os.path.join(BERT_WEIGHTS_DIR, "finbert_best_model_256token.pt")
BERT_WEIGHTS       = os.path.join(BERT_WEIGHTS_DIR, "bert_base_best_model_256token.pt")

# BiLSTM artefacts (tokenizer, architecture config, trained weights)
BILSTM_CHECKPOINTS  = os.path.join(TEXT_M_ROOT, "bilstm_glove_checkpoints")
BILSTM_TOKENIZER    = os.path.join(BILSTM_CHECKPOINTS, "tokenizer.json")
BILSTM_CONFIG       = os.path.join(BILSTM_CHECKPOINTS, "config.json")
BILSTM_WEIGHTS      = os.path.join(BILSTM_CHECKPOINTS, "model.weights.h5")
BILSTM_BEST_CONFIG  = os.path.join(TEXT_M_ROOT, "bilstm_glove_gridsearch", "best_config.json")

# Pre-computed evaluation metrics (confusion matrices, classification reports)
BERT_RESULTS_PATH   = os.path.join(TEXT_M_ROOT, "model_results.json")
BILSTM_RESULTS_PATH = os.path.join(TEXT_M_ROOT, "bilstm_results.json")

MAX_LEN   = 256  # shared max token length for all BERT-family models
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {0: "Positive", 1: "Negative", 2: "Neutral"}
LABEL_NAMES = ["Positive", "Negative", "Neutral"]

# Dashboard colour palette (reused across all charts and result cards)
COLORS = {
    "positive": "#2ecc71",
    "neutral":  "#5dade2",
    "negative": "#e74c3c",
    "accent":   "#f39c12",
    "text":     "#1f2d3d",
}

# ─────────────────────────────────────────────
# Model loading (lazy-cached)
# ─────────────────────────────────────────────

# Shared cache so each model is loaded at most once per process lifetime
_model_cache = {}


def _get_bilstm():
    """Load (or retrieve from cache) the BiLSTM + GloVe model, its Keras
    tokenizer, and the expected input sequence length."""
    if "bilstm" in _model_cache:
        return _model_cache["bilstm"]

    # --- Keras tokenizer (word → index mapping) ---
    with open(BILSTM_TOKENIZER, "r", encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())

    # --- Architecture hyper-parameters from training config ---
    with open(BILSTM_CONFIG, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    max_seq_len = int(cfg["MAX_SEQUENCE_LEN"])
    vocab_size = int(cfg["VOCAB_SIZE"])

    # --- Best hyper-parameters found via grid search ---
    with open(BILSTM_BEST_CONFIG, "r", encoding="utf-8") as f:
        best = json.load(f)

    # Reconstruct the same Sequential architecture used during training
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 100, mask_zero=True, trainable=True),
        tf.keras.layers.SpatialDropout1D(best["dropout_rate"]),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(int(best["lstm_units"]), dropout=best["dropout_rate"])
        ),
        tf.keras.layers.Dense(int(best["dense_units"])),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(best["dropout_rate"]),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])
    model.build(input_shape=(None, max_seq_len))

    # Manually restore weights from the .h5 checkpoint layer-by-layer.
    # The Bidirectional wrapper stores forward/backward cells at different
    # paths, so it needs special handling.
    with h5py.File(BILSTM_WEIGHTS, "r") as f:
        for layer in model.layers:
            h5_path = f"layers/{layer.name}/vars"

            if layer.name.startswith("bidirectional"):
                fw = f"layers/{layer.name}/forward_layer/cell/vars"
                bw = f"layers/{layer.name}/backward_layer/cell/vars"
                if fw in f and bw in f:
                    weights = []
                    for path in [fw, bw]:
                        i = 0
                        while f"{path}/{i}" in f:
                            weights.append(np.array(f[f"{path}/{i}"]))
                            i += 1
                    if weights:
                        layer.set_weights(weights)
                continue

            if h5_path not in f:
                continue
            weights = []
            i = 0
            while f"{h5_path}/{i}" in f:
                weights.append(np.array(f[f"{h5_path}/{i}"]))
                i += 1
            if weights:
                layer.set_weights(weights)

    _model_cache["bilstm"] = (model, tokenizer, max_seq_len)
    return model, tokenizer, max_seq_len


def _predict_bilstm(text: str):
    """Tokenize, pad, and run inference with the BiLSTM model.
    Returns (predicted_label_index, probability_array)."""
    model, tokenizer, max_seq_len = _get_bilstm()
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(
        seq, maxlen=max_seq_len, padding="post", truncating="post"
    ).astype("int32")
    probs = model.predict(padded, verbose=0)[0]
    return int(np.argmax(probs)), probs


def _get_bert_model(name: str):
    """Load (or retrieve from cache) a BERT-family model by short name.
    Supported names: 'distilbert', 'finbert', 'bert'."""
    if name in _model_cache:
        return _model_cache[name]

    if name == "distilbert":
        config = DistilBertConfig.from_pretrained(
            "distilbert-base-uncased", num_labels=3,
            hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.1,
        )
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", config=config,
        )
        model.load_state_dict(torch.load(DISTILBERT_WEIGHTS, map_location=DEVICE))
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    elif name == "finbert":
        # ProsusAI/finbert is pre-trained on financial corpora
        model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
        model.load_state_dict(torch.load(FINBERT_WEIGHTS, map_location=DEVICE))
        tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")

    elif name == "bert":
        config = BertConfig.from_pretrained(
            "bert-base-uncased", num_labels=3,
            hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.1,
        )
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", config=config,
        )
        model.load_state_dict(torch.load(BERT_WEIGHTS, map_location=DEVICE))
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    else:
        raise ValueError(f"Unknown model: {name}")

    model.to(DEVICE).eval()
    _model_cache[name] = (model, tokenizer)
    return model, tokenizer


def _predict_bert(text: str, model, tokenizer):
    """Run inference with a BERT-family model.
    Returns (predicted_label_index, probability_array)."""
    inputs = tokenizer(
        text, padding="max_length", truncation=True,
        max_length=MAX_LEN, return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"].to(DEVICE),
            attention_mask=inputs["attention_mask"].to(DEVICE),
        ).logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return int(np.argmax(probs)), probs


# ─────────────────────────────────────────────
# Metrics tab – data loading & chart builders
# ─────────────────────────────────────────────

def _load_metrics() -> dict:
    """Merge BiLSTM and BERT evaluation results into a single dict
    keyed by model name."""
    all_results = {}
    if os.path.exists(BILSTM_RESULTS_PATH):
        with open(BILSTM_RESULTS_PATH, "r", encoding="utf-8") as f:
            all_results.update(json.load(f))
    if os.path.exists(BERT_RESULTS_PATH):
        with open(BERT_RESULTS_PATH, "r", encoding="utf-8") as f:
            all_results.update(json.load(f))
    return all_results


def _make_confusion_matrix(cm, model_name: str):
    """Build a Plotly heatmap showing both raw counts and row-normalised
    percentages for a single model's confusion matrix."""
    cm_array = np.array(cm)
    cm_pct = cm_array.astype("float") / cm_array.sum(axis=1)[:, np.newaxis]
    text = [[f"{cm_array[i][j]}<br>({cm_pct[i][j]:.1%})"
             for j in range(3)] for i in range(3)]

    fig = go.Figure(data=go.Heatmap(
        z=cm_pct, x=LABEL_NAMES, y=LABEL_NAMES,
        text=text, texttemplate="%{text}",
        colorscale=[[0, "#fef9f0"], [1, "#f39c12"]],
        showscale=False,
        hoverinfo="skip",
    ))
    fig.update_layout(
        title=dict(text=model_name, font=dict(size=14, color=COLORS["text"])),
        xaxis_title="Predicted", yaxis_title="True",
        yaxis=dict(autorange="reversed"),
        height=340, margin=dict(l=60, r=20, t=40, b=60),
        plot_bgcolor="white", font=dict(color=COLORS["text"]),
    )
    return fig


def _make_report_table(report: dict, model_name: str):
    """Render a sklearn-style classification report as an HTML table."""
    rows = []
    for label in LABEL_NAMES:
        if label in report:
            r = report[label]
            rows.append(html.Tr([
                html.Td(label),
                html.Td(f"{r['precision']:.3f}"),
                html.Td(f"{r['recall']:.3f}"),
                html.Td(f"{r['f1-score']:.3f}"),
                html.Td(f"{int(r['support'])}"),
            ]))

    # Accuracy row (single value, spans the F1 column)
    if "accuracy" in report:
        rows.append(html.Tr([
            html.Td("Accuracy", style={"fontWeight": "bold"}),
            html.Td(""), html.Td(""),
            html.Td(f"{report['accuracy']:.3f}", style={"fontWeight": "bold"}),
            html.Td(f"{int(report.get('macro avg', {}).get('support', 0))}"),
        ], style={"borderTop": "2px solid #dee2e6"}))

    # Macro-averaged row
    if "macro avg" in report:
        r = report["macro avg"]
        rows.append(html.Tr([
            html.Td("Macro Avg", style={"fontStyle": "italic"}),
            html.Td(f"{r['precision']:.3f}"),
            html.Td(f"{r['recall']:.3f}"),
            html.Td(f"{r['f1-score']:.3f}"),
            html.Td(f"{int(r['support'])}"),
        ]))

    return html.Div([
        html.H6(model_name, className="text-muted mt-3"),
        dbc.Table([
            html.Thead(html.Tr([
                html.Th("Class"), html.Th("Precision"),
                html.Th("Recall"), html.Th("F1-Score"), html.Th("Support"),
            ])),
            html.Tbody(rows),
        ], bordered=True, hover=True, size="sm", className="mt-2"),
    ])


def _make_accuracy_comparison(all_results: dict):
    """Bar chart comparing overall accuracy across all models."""
    models = []
    accuracies = []

    for name, data in all_results.items():
        models.append(name)
        accuracies.append(data["report"]["accuracy"])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models, y=accuracies,
        marker_color=COLORS["accent"],
        text=[f"{a:.3f}" for a in accuracies],
        textposition="outside",
        hoverinfo="skip",
    ))
    fig.update_layout(
        title="Accuracy Comparison Across Models",
        yaxis_title="Accuracy",
        height=420, margin=dict(l=60, r=20, t=50, b=40),
        plot_bgcolor="white", font=dict(color=COLORS["text"]),
        showlegend=False,
    )
    fig.update_yaxes(range=[0, 1], gridcolor="#eee")
    return fig


def _make_f1_comparison(all_results: dict):
    """Grouped bar chart of per-class F1 scores with a macro-F1 overlay."""
    models, f1_pos, f1_neg, f1_neu, f1_macro = [], [], [], [], []
    for name, data in all_results.items():
        r = data["report"]
        models.append(name)
        f1_pos.append(r["Positive"]["f1-score"])
        f1_neg.append(r["Negative"]["f1-score"])
        f1_neu.append(r["Neutral"]["f1-score"])
        f1_macro.append(r["macro avg"]["f1-score"])

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Positive", x=models, y=f1_pos, marker_color=COLORS["positive"]))
    fig.add_trace(go.Bar(name="Negative", x=models, y=f1_neg, marker_color=COLORS["negative"]))
    fig.add_trace(go.Bar(name="Neutral", x=models, y=f1_neu, marker_color=COLORS["neutral"]))
    fig.add_trace(go.Scatter(
        name="Macro F1", x=models, y=f1_macro,
        mode="markers+lines",
        marker=dict(size=10, color=COLORS["accent"]),
        line=dict(color=COLORS["accent"], width=2, dash="dash"),
    ))
    fig.update_layout(
        title="F1-Score Comparison Across Models",
        yaxis_title="F1-Score", barmode="group",
        height=420, margin=dict(l=60, r=20, t=50, b=40),
        plot_bgcolor="white", font=dict(color=COLORS["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig.update_yaxes(range=[0, 1], gridcolor="#eee")
    return fig


def _build_metrics_tab():
    """Assemble the full Metrics tab layout: confusion matrices,
    classification reports, accuracy chart, and F1 chart."""
    all_results = _load_metrics()

    if not all_results:
        return html.Div(
            dbc.Alert(
                "No results files found. Generate bilstm_results.json "
                "and/or model_results.json first.",
                color="warning",
            ),
            className="mt-3",
        )

    model_names = list(all_results.keys())

    # One confusion-matrix card per model
    cm_cards = []
    for name in model_names:
        fig = _make_confusion_matrix(all_results[name]["cm"], name)
        cm_cards.append(
            dbc.Col(dcc.Graph(figure=fig, config={"displayModeBar": False}), md=6, lg=3)
        )

    # One classification-report table per model
    report_tables = []
    for name in model_names:
        report_tables.append(
            dbc.Col(_make_report_table(all_results[name]["report"], name), md=6)
        )

    acc_fig = _make_accuracy_comparison(all_results)
    f1_fig = _make_f1_comparison(all_results)

    return html.Div([
        html.H5("Performance Comparison", className="mt-3"),
        html.P("Evaluation metrics across all trained models on the test set.",
               className="text-muted"),

        html.Hr(),
        html.H6("Confusion Matrices"),
        dbc.Row(cm_cards, className="g-3"),

        html.Hr(),
        html.H6("Classification Reports"),
        dbc.Row(report_tables, className="g-3"),

        html.Hr(),
        html.H6("Accuracy Comparison"),
        dcc.Graph(figure=acc_fig, config={"displayModeBar": False}),

        html.Hr(),
        html.H6("F1-Score Comparison"),
        dcc.Graph(figure=f1_fig, config={"displayModeBar": False}),
    ])


# ─────────────────────────────────────────────
# Shared result card (used by both inference tabs)
# ─────────────────────────────────────────────

def _build_result_card(label_idx: int, probs, model_display_name: str, device_str: str):
    """Render a prediction result as a Bootstrap card with horizontal
    probability bars for each class."""
    label_names = ["Positive", "Negative", "Neutral"]
    bar_colors = [COLORS["positive"], COLORS["negative"], COLORS["neutral"]]

    prob_bars = []
    for i, (name, prob) in enumerate(zip(label_names, probs)):
        prob_bars.append(
            html.Div([
                html.Span(name, style={
                    "display": "inline-block", "width": "80px",
                    "fontWeight": "bold", "color": COLORS["text"],
                }),
                html.Div(style={
                    "display": "inline-block",
                    "width": f"{prob * 100:.1f}%",
                    "maxWidth": "60%",
                    "height": "20px",
                    "backgroundColor": bar_colors[i],
                    "borderRadius": "4px",
                    "verticalAlign": "middle",
                    "marginRight": "10px",
                    "minWidth": "4px",
                }),
                html.Span(f"{prob * 100:.1f}%", style={"color": COLORS["text"]}),
            ], className="mb-2")
        )

    return dbc.Card([
        dbc.CardBody([
            html.H4("Prediction"),
            html.H3(
                LABEL_MAP[label_idx],
                style={"color": bar_colors[label_idx], "fontWeight": "bold"},
            ),
            html.Hr(),
            html.H6("Confidence", className="text-muted mb-3"),
            *prob_bars,
            html.Hr(),
            html.Small(
                f"Model: {model_display_name}  ·  Device: {device_str}",
                className="text-muted",
            ),
        ])
    ])


# ─────────────────────────────────────────────
# Tab layouts
# ─────────────────────────────────────────────

# Pre-build the metrics tab at import time (static content)
metrics_tab = _build_metrics_tab()

# BiLSTM inference tab
bilstm_tab = html.Div([
    html.H5("BiLSTM + GloVe (100d)", className="mt-3"),
    html.P(
        "Bidirectional LSTM trained on financial news descriptions "
        "with GloVe 6B 100d embeddings and Keras tokenizer.",
        className="text-muted",
    ),
    html.Hr(),
    html.H5("Financial news text"),
    dcc.Textarea(
        id="bilstm-input",
        placeholder="Paste a financial news description here…",
        style={"width": "100%", "height": "140px"},
    ),
    html.Br(),
    dbc.Button("Classify", id="bilstm-classify-btn", n_clicks=0, color="primary"),
    html.Br(), html.Br(),
    dcc.Loading(id="bilstm-loading", type="circle",
                children=html.Div(id="bilstm-output")),
])

# BERT-family inference tab (model selectable via radio buttons)
bert_tab = html.Div([
    html.H5("Select model", className="mt-3"),
    dbc.RadioItems(
        id="bert-model-selector",
        options=[
            {"label": "DistilBERT", "value": "distilbert"},
            {"label": "FinBERT", "value": "finbert"},
            {"label": "BERT Base", "value": "bert"},
        ],
        value="distilbert",
        inline=True,
    ),
    html.Hr(),
    html.H5("Financial news text"),
    dcc.Textarea(
        id="sentiment-input",
        placeholder="Paste a financial news description here…",
        style={"width": "100%", "height": "140px"},
    ),
    html.Br(),
    dbc.Button("Classify", id="sentiment-classify-btn", n_clicks=0, color="primary"),
    html.Br(), html.Br(),
    dcc.Loading(id="sentiment-loading", type="circle",
                children=html.Div(id="sentiment-output")),
])

# ─────────────────────────────────────────────
# Page layout (three-tab container)
# ─────────────────────────────────────────────

layout = html.Div([
    html.H2("Sentiment Models"),
    dbc.Tabs([
        dbc.Tab(metrics_tab, label="Metrics",  tab_id="metrics"),
        dbc.Tab(bilstm_tab,  label="BiLSTM",   tab_id="bilstm"),
        dbc.Tab(bert_tab,    label="BERT",      tab_id="bert"),
    ], id="tabs-models", active_tab="metrics"),
])

# ─────────────────────────────────────────────
# Dash callbacks (inference on button click)
# ─────────────────────────────────────────────

@callback(
    Output("bilstm-output", "children"),
    Input("bilstm-classify-btn", "n_clicks"),
    State("bilstm-input", "value"),
    prevent_initial_call=True,
)
def classify_bilstm(n_clicks, text):
    """Triggered when the user clicks 'Classify' on the BiLSTM tab."""
    if not text or not text.strip():
        return dbc.Alert("Please enter a news description.", color="warning")
    try:
        label_idx, probs = _predict_bilstm(text)
    except Exception as e:
        return dbc.Alert(f"Error: {e}", color="danger")
    device_str = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
    return _build_result_card(label_idx, probs, "BiLSTM + GloVe", device_str)


@callback(
    Output("sentiment-output", "children"),
    Input("sentiment-classify-btn", "n_clicks"),
    State("sentiment-input", "value"),
    State("bert-model-selector", "value"),
    prevent_initial_call=True,
)
def classify_bert(n_clicks, text, model_name):
    """Triggered when the user clicks 'Classify' on the BERT tab.
    Loads the selected transformer variant on first use."""
    if not text or not text.strip():
        return dbc.Alert("Please enter a news description.", color="warning")
    try:
        model, tokenizer = _get_bert_model(model_name)
        label_idx, probs = _predict_bert(text, model, tokenizer)
    except Exception as e:
        return dbc.Alert(f"Error: {e}", color="danger")
    model_display = {
        "distilbert": "DistilBERT",
        "finbert": "FinBERT",
        "bert": "BERT Base",
    }.get(model_name, model_name.upper())
    return _build_result_card(label_idx, probs, model_display, str(DEVICE).upper())