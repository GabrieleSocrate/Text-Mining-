# -*- coding: utf-8 -*-
"""
Financial News Sentiment Classification App
Requirements:
    pip install transformers dash dash-bootstrap-components torch tensorflow keras numpy
"""
import json
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig,
    BertTokenizer, BertForSequenceClassification, BertConfig
)
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

# ─────────────────────────────────────────────
# CONFIGURAZIONE PATH PESI
# ─────────────────────────────────────────────
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # risale a Text Mining\
"""
DISTILBERT_WEIGHTS = os.path.join(BASE_DIR, "Bert_weights", "bert_checkpoints_256token", "bert_best_model_256token.pt")
FINBERT_WEIGHTS    = os.path.join(BASE_DIR, "Bert_weights", "bert_checkpoints_256token", "finbert_best_model_256token.pt")
BERT_WEIGHTS       = os.path.join(BASE_DIR, "Bert_weights", "bert_checkpoints_256token", "bert_base_best_model_256token.pt")
"""
BILSTM_WEIGHTS     = os.path.join(BASE_DIR, "TextMiningProject", "best_final_model.keras") # aggiorna con il tuo path
BILSTM_WORD_INDEX  = os.path.join(BASE_DIR, "TextMiningProject", "tokenizer.json")
BILSTM_CONFIG    =  os.path.join(BASE_DIR, "TextMiningProject", "config.json")
DISTILBERT_WEIGHTS = os.path.join(BASE_DIR, "TextMiningProject", "bert_best_model_256token.pt")
FINBERT_WEIGHTS    = os.path.join(BASE_DIR, "TextMiningProject", "finbert_best_model_256token.pt")
BERT_WEIGHTS       = os.path.join(BASE_DIR, "TextMiningProject", "bert_base_best_model_256token.pt")



BILSTM_MAXLEN = 98
MAX_LEN    = 256
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP  = {0: "🟢 Positive", 1: "🔴 Negative", 2: "⚪ Neutral"}
LABEL_COLORS = {0: "#2ecc71", 1: "#e74c3c", 2: "#95a5a6"}

# ─────────────────────────────────────────────
# CARICAMENTO MODELLI
# ─────────────────────────────────────────────

def load_distilbert():
    config = DistilBertConfig.from_pretrained(
        "distilbert-base-uncased", num_labels=3,
        hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.1
    )
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", config=config
    )
    model.load_state_dict(torch.load(DISTILBERT_WEIGHTS, map_location=DEVICE))
    model.to(DEVICE).eval()
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    return model, tokenizer

def load_finbert():
    model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.load_state_dict(torch.load(FINBERT_WEIGHTS, map_location=DEVICE))
    model.to(DEVICE).eval()
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    return model, tokenizer

def load_bert_base():
    config = BertConfig.from_pretrained(
        "bert-base-uncased", num_labels=3,
        hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.1
    )
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", config=config
    )
    model.load_state_dict(torch.load(BERT_WEIGHTS, map_location=DEVICE))
    model.to(DEVICE).eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer


def load_bilstm():
    
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
        
    model = load_model(BILSTM_WEIGHTS)
        
    with open(BILSTM_WORD_INDEX, "r", encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())
    #tokenizer = Tokenizer(num_words=len(word_index) + 1, oov_token="<OOV>")
    #tokenizer.word_index = word_index
    #tokenizer.index_word = {v: k for k, v in word_index.items()}
    with open(BILSTM_CONFIG, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    max_seq_len = int(cfg["MAX_SEQUENCE_LEN"])
    return model, tokenizer , max_seq_len   #,tokenizer   delete word_index if tokenizer is active
    

# ─────────────────────────────────────────────
# FUNZIONI DI PREDIZIONE
# ─────────────────────────────────────────────

def predict_transformer(text, model, tokenizer):
    inputs = tokenizer(
        text, padding="max_length", truncation=True,
        max_length=MAX_LEN, return_tensors="pt"
    )
    input_ids      = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    probs     = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    label_idx = int(np.argmax(probs))
    return label_idx, probs

def predict_bilstm(text, model,tokenizer,max_seq_len): #swap word_index with tokenizer
    """from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=BILSTM_MAXLEN, padding="post", truncating="post")
    probs     = model.predict(seq)[0]
    label_idx = int(np.argmax(probs))
    return label_idx, probs"""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
 
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=max_seq_len, padding="post", truncating="post")
 
    probs     = model.predict(seq, verbose=0)[0]
    label_idx = int(np.argmax(probs))
    return label_idx, probs

# ─────────────────────────────────────────────
# LAYOUT APP
# ─────────────────────────────────────────────

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Financial Sentiment Classifier"

navbar = dbc.Navbar(
    dbc.Container([
        dbc.Row([
            dbc.Col(html.Span("📈", style={"fontSize": "28px"})),
            dbc.Col(dbc.NavbarBrand(
                "Financial News Sentiment Classifier",
                style={"color": "white", "fontSize": "22px", "fontFamily": "Georgia, serif", "fontWeight": "bold"}
            )),
        ], align="center"),
    ], fluid=True),
    color="#1a1a2e", dark=True, style={"borderBottom": "2px solid #e94560"}
)

body_app = dbc.Container([
    html.Br(),

    # Input testo
    dbc.Row([
        dbc.Col([
            html.H5("📰 Financial News Description", style={"color": "#e0e0e0", "fontFamily": "Georgia, serif"}),
            dcc.Textarea(
                id="news-input",
                value="",
                placeholder="Paste a financial news description here...",
                style={
                    "width": "100%", "height": "160px",
                    "backgroundColor": "#16213e", "color": "#e0e0e0",
                    "border": "1px solid #e94560", "borderRadius": "8px",
                    "padding": "12px", "fontSize": "14px", "resize": "vertical"
                }
            ),
        ], width=10),
    ], justify="center"),

    html.Br(),

    # Selezione modello
    dbc.Row([
        dbc.Col([
            html.H5("🤖 Select Model", style={"color": "#e0e0e0", "fontFamily": "Georgia, serif"}),
            dbc.RadioItems(
                id="model-selector",
                options=[
                    {"label": " DistilBERT",  "value": "distilbert"},
                    {"label": " FinBERT",     "value": "finbert"},
                    {"label": " BERT Base",   "value": "bert"},
                    {"label": " BiLSTM + GloVe", "value": "bilstm"},
                ],
                value="distilbert",
                inline=True,
                style={"color": "#e0e0e0"},
                inputStyle={"marginRight": "6px"},
                labelStyle={"marginRight": "24px", "fontSize": "15px"},
            ),
        ], width=10),
    ], justify="center"),

    html.Br(),

    # Bottone
    dbc.Row([
        dbc.Col(
            dbc.Button(
                "Classify", id="classify-btn", n_clicks=0,
                style={
                    "backgroundColor": "#e94560", "border": "none",
                    "borderRadius": "6px", "padding": "10px 36px",
                    "fontSize": "16px", "fontWeight": "bold", "color": "white"
                }
            ),
            width="auto"
        ),
    ], justify="center"),

    html.Br(),

    # Output
    dbc.Row([
        dbc.Col(
            dcc.Loading(
                id="loading-output",
                type="circle",
                color="#e94560",
                children=html.Div(id="output-div")
            ),
            width=10
        ),
    ], justify="center"),

    html.Br(),

], fluid=True, style={"backgroundColor": "#0f3460", "minHeight": "100vh", "paddingTop": "20px"})

app.layout = html.Div([navbar, body_app], style={"backgroundColor": "#0f3460"})

# ─────────────────────────────────────────────
# CALLBACK
# ─────────────────────────────────────────────

@app.callback(
    Output("output-div", "children"),
    Input("classify-btn", "n_clicks"),
    State("news-input", "value"),
    State("model-selector", "value"),
    prevent_initial_call=True
)
def classify(n_clicks, text, model_name):
    if not text or not text.strip():
        return dbc.Alert("⚠️ Please enter a news description.", color="warning")

    try:
        if model_name == "distilbert":
            model, tokenizer = load_distilbert()
            label_idx, probs = predict_transformer(text, model, tokenizer)
        elif model_name == "finbert":
            model, tokenizer = load_finbert()
            label_idx, probs = predict_transformer(text, model, tokenizer)
        elif model_name == "bert":
            model, tokenizer = load_bert_base()
            label_idx, probs = predict_transformer(text, model, tokenizer)
        
        elif model_name == "bilstm":
            model, tokenizer , seq_len= load_bilstm()
           
            label_idx, probs = predict_bilstm(text, model, tokenizer, seq_len)
        
        else:
            return dbc.Alert("Unknown model selected.", color="danger")

        label_text  = LABEL_MAP[label_idx]
        label_color = LABEL_COLORS[label_idx]
        label_names = ["Positive", "Negative", "Neutral"]

        # Barre probabilità
        prob_bars = []
        for i, (name, prob) in enumerate(zip(label_names, probs)):
            bar_color = LABEL_COLORS[i]
            prob_bars.append(
                html.Div([
                    html.Div([
                        html.Span(name, style={"color": "#e0e0e0", "width": "80px", "display": "inline-block", "fontSize": "13px"}),
                        html.Div(style={
                            "display": "inline-block",
                            "width": f"{prob * 100:.1f}%",
                            "backgroundColor": bar_color,
                            "height": "18px", "borderRadius": "4px",
                            "verticalAlign": "middle", "marginLeft": "8px",
                            "minWidth": "4px"
                        }),
                        html.Span(f"{prob * 100:.1f}%", style={"color": "#e0e0e0", "marginLeft": "10px", "fontSize": "13px"}),
                    ])
                ], style={"marginBottom": "8px"})
            )

        result_card = dbc.Card([
            dbc.CardBody([
                html.H4("Prediction", style={"color": "#e0e0e0", "fontFamily": "Georgia, serif"}),
                html.H2(label_text, style={"color": label_color, "fontWeight": "bold", "fontFamily": "Georgia, serif"}),
                html.Hr(style={"borderColor": "#e94560"}),
                html.H6("Confidence", style={"color": "#aaa", "marginBottom": "12px"}),
                *prob_bars,
                html.Hr(style={"borderColor": "#333"}),
                html.Small(
                    f"Model: {model_name.upper()}  |  Device: {str(DEVICE).upper()}",
                    style={"color": "#888"}
                ),
            ])
        ], style={
            "backgroundColor": "#16213e",
            "border": f"1px solid {label_color}",
            "borderRadius": "10px"
        })

        return result_card

    except Exception as e:
        return dbc.Alert(f"❌ Error: {str(e)}", color="danger")


if __name__ == "__main__":
    app.run(debug=True)