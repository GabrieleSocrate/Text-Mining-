import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from collections import Counter

from utils.data_loader import load_dataset

dash.register_page(__name__, path="/eda", name="EDA")

COLORS = {
    "positive": "#2ecc71",
    "neutral": "#5dade2",
    "negative": "#e74c3c",
    "accent": "#f39c12",
    "text": "#1f2d3d",
}

HOVER_STYLE = dict(
    bgcolor="white",
    font_size=13,
    font_family="Inter, sans-serif",
)

LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=60, l=40, r=40, b=40),
)


# =============================================================================
# Chart builders
# =============================================================================


def create_sentiment_pie(df):
    """
    Donut chart showing the distribution of sentiment labels.
    Displays total count in the center and percentage on each slice.
    """
    counts = df["sentiment_label"].value_counts().reset_index()
    counts.columns = ["sentiment_label", "count"]
    total = int(counts["count"].sum())

    # Enforce consistent ordering: positive → neutral → negative
    order = ["positive", "neutral", "negative"]
    counts["sentiment_label"] = pd.Categorical(
        counts["sentiment_label"], categories=order, ordered=True
    )
    counts = counts.sort_values("sentiment_label")

    fig = go.Figure(
        go.Pie(
            labels=counts["sentiment_label"],
            values=counts["count"],
            hole=0.55,
            marker=dict(
                colors=[COLORS[s] for s in counts["sentiment_label"]],
                line=dict(color="white", width=2),
            ),
            textinfo="percent",
            textfont=dict(size=15, color="white"),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Count: %{value:,}<br>"
                "Percentage: %{percent}"
                "<extra></extra>"
            ),
            sort=False,
        )
    )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Sentiment Label Distribution", x=0.5, font=dict(size=16)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
        hoverlabel=HOVER_STYLE,
        height=400,
    )

    # Central annotation with total count
    fig.add_annotation(
        text=f"<b>Total</b><br>{total:,}",
        x=0.5,
        y=0.5,
        font=dict(size=18, color=COLORS["text"]),
        showarrow=False,
    )

    return fig


def create_text_length_histograms(df):
    """
    Three side-by-side histograms showing word count distributions
    for title, description, and main text fields.
    """
    df = df.copy()
    df["title_len"] = df["title"].astype(str).apply(lambda x: len(x.split()))
    df["description_len"] = df["description"].astype(str).apply(lambda x: len(x.split()))
    df["maintext_len"] = df["maintext"].astype(str).apply(lambda x: len(x.split()))

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Title Length", "Description Length", "Main Text Length"),
        horizontal_spacing=0.08,
    )

    # Each field gets its own histogram with consistent styling
    fields = [
        ("title_len", 1),
        ("description_len", 2),
        ("maintext_len", 3),
    ]

    for field, col in fields:
        fig.add_trace(
            go.Histogram(
                x=df[field],
                nbinsx=50,
                marker=dict(
                    color=COLORS["accent"],
                    line=dict(color="rgba(255,255,255,0.3)", width=0.5),
                ),
                hovertemplate="Words: %{x}<br>Count: %{y:,}<extra></extra>",
            ),
            row=1,
            col=col,
        )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text="Text Length Distribution (Number of Words)",
            x=0.5,
            font=dict(size=16),
        ),
        showlegend=False,
        bargap=0.05,
        height=350,
        hoverlabel=HOVER_STYLE,
    )

    fig.update_xaxes(title_text="Words")
    fig.update_yaxes(title_text="Count", col=1)

    return fig


def create_temporal_distribution_chart(df):
    """
    Returns two separate figures:
      1. Bar chart of article volume per year
      2. Line/area chart of monthly sentiment evolution

    Splitting into two charts avoids cramped subplots and allows
    independent zoom and interaction on each.
    """
    df_time = df.copy()
    df_time["date_publish"] = pd.to_datetime(df_time["date_publish"], errors="coerce")
    df_time = df_time.dropna(subset=["date_publish"]).copy()

    # --- Figure 1: Articles per Year ---
    news_per_year = (
        df_time.groupby(df_time["date_publish"].dt.year)
        .size()
        .reset_index(name="count")
    )
    news_per_year.columns = ["year", "count"]

    fig_year = go.Figure(
        go.Bar(
            x=news_per_year["year"],
            y=news_per_year["count"],
            marker=dict(
                color=COLORS["accent"],
                line=dict(color="rgba(255,255,255,0.3)", width=0.5),
            ),
            hovertemplate="<b>%{x}</b><br>Articles: %{y:,}<extra></extra>",
        )
    )

    fig_year.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Articles per Year", x=0.5, font=dict(size=16)),
        xaxis=dict(title="Year", dtick=1),
        yaxis=dict(title="Number of Articles", gridcolor="rgba(0,0,0,0.06)"),
        hoverlabel=HOVER_STYLE,
        height=350,
        bargap=0.15,
    )

    # --- Figure 2: Sentiment Evolution over Time (monthly) ---
    sentiment_time = df_time.copy()
    sentiment_time["year_month"] = (
        sentiment_time["date_publish"].dt.to_period("M").dt.to_timestamp()
    )

    sentiment_counts = (
        sentiment_time.groupby(["year_month", "sentiment_label"])
        .size()
        .reset_index(name="count")
    )

    fig_sentiment = go.Figure()

    # Helper to convert hex color to rgba with transparency for fill
    def hex_to_rgba(hex_color, alpha):
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"

    for sentiment in ["positive", "neutral", "negative"]:
        temp = sentiment_counts[sentiment_counts["sentiment_label"] == sentiment]
        fig_sentiment.add_trace(
            go.Scatter(
                x=temp["year_month"],
                y=temp["count"],
                mode="lines",
                line=dict(width=2, color=COLORS[sentiment]),
                name=sentiment.capitalize(),
                hovertemplate=(
                    "<b>%{x|%b %Y}</b><br>"
                    f"{sentiment.capitalize()}: %{{y:,}}"
                    "<extra></extra>"
                ),
                fill="tozeroy",
                fillcolor=hex_to_rgba(COLORS[sentiment], 0.08),
            )
        )

    fig_sentiment.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text="Sentiment Evolution over Time",
            x=0.5,
            font=dict(size=16),
        ),
        xaxis=dict(title="Time", type="date"),
        yaxis=dict(title="Number of Articles", gridcolor="rgba(0,0,0,0.06)"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        hoverlabel=HOVER_STYLE,
        hovermode="x unified",
        height=400,
    )

    return fig_year, fig_sentiment


def create_top_mentioned_companies_chart(df, top_n=15):
    """
    Horizontal bar chart of the top N most frequently mentioned companies.
    Bar opacity scales with mention count to highlight dominant companies.
    """
    if "mentioned_companies" not in df.columns:
        return empty_figure("No company data available")

    # Count company mentions across all articles
    counter = Counter()
    for companies in df["mentioned_companies"]:
        if isinstance(companies, list):
            cleaned = [
                str(c).strip()
                for c in companies
                if str(c).strip() and str(c).strip().lower() != "none"
            ]
            counter.update(cleaned)

    if not counter:
        return empty_figure("No company data available")

    top_companies = counter.most_common(top_n)
    top_df = pd.DataFrame(top_companies, columns=["company", "count"]).sort_values(
        "count", ascending=True
    )

    # Gradient opacity: bars scale from 40% to 100% based on count
    max_count = top_df["count"].max()
    bar_colors = [
        f"rgba(243, 156, 18, {0.4 + 0.6 * (c / max_count)})"
        for c in top_df["count"]
    ]

    fig = go.Figure(
        go.Bar(
            x=top_df["count"],
            y=top_df["company"],
            orientation="h",
            marker=dict(
                color=bar_colors,
                line=dict(color="rgba(255,255,255,0.3)", width=0.5),
            ),
            hovertemplate="<b>%{y}</b><br>Mentions: %{x:,}<extra></extra>",
        )
    )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text=f"Top {top_n} Mentioned Companies",
            x=0.5,
            font=dict(size=16),
        ),
        xaxis=dict(title="Number of Mentions"),
        yaxis=dict(title=None),
        hoverlabel=HOVER_STYLE,
        height=500,
    )

    return fig


def empty_figure(title="No data available"):
    """Placeholder figure shown when data is missing or unavailable."""
    fig = go.Figure()
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=title, x=0.5),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text="Graph will appear here",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color="#adb5bd"),
            )
        ],
    )
    return fig


# =============================================================================
# Data loading and pre-computation
# =============================================================================

df_full = load_dataset("full")
df_balanced = load_dataset("balanced")

# Sentiment pie charts
full_sentiment_fig = create_sentiment_pie(df_full)
balanced_sentiment_fig = create_sentiment_pie(df_balanced)

# Text length histograms
full_length_fig = create_text_length_histograms(df_full)
balanced_length_fig = create_text_length_histograms(df_balanced)

# Temporal charts (each returns two figures: yearly bar + monthly sentiment)
full_year_fig, full_sentiment_time_fig = create_temporal_distribution_chart(df_full)
balanced_year_fig, balanced_sentiment_time_fig = create_temporal_distribution_chart(
    df_balanced
)

# Top mentioned companies
full_top_companies_fig = create_top_mentioned_companies_chart(df_full)
balanced_top_companies_fig = create_top_mentioned_companies_chart(df_balanced)


# =============================================================================
# Page layout
# =============================================================================

layout = dbc.Container(
    [
        # --- Header ---
        dbc.Row(
            dbc.Col(
                [
                    html.H1(
                        "Exploratory Data Analysis",
                        className="text-center mb-3",
                    ),
                    html.P(
                        "This section introduces the dataset used in the project, explains the "
                        "preprocessing choices adopted before training, and compares the original "
                        "and balanced versions of the data.",
                        className="lead text-center text-muted",
                    ),
                ],
                width=10,
                className="mx-auto mb-4",
            )
        ),
        # --- Section 1: Dataset Overview ---
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H3("1. Dataset Overview", className="mb-3"),
                            html.P(
                                "The dataset contains financial news articles enriched with textual, "
                                "temporal, and company-related information. Each observation includes "
                                "metadata such as publication date, title, description, full text, "
                                "mentioned companies, related companies, industries, sentiment scores, "
                                "emotion scores, and a final sentiment label."
                            ),
                        ]
                    ),
                    className="shadow-sm border-0",
                ),
                width=12,
                className="mb-4",
            )
        ),
        # --- Section 2: Preprocessing Rationale ---
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H3("2. Preprocessing Rationale", className="mb-3"),
                            html.P(
                                "Before training the models, a preprocessing step was performed to ensure "
                                "consistency between the datasets used for the BiLSTM and BERT-based models."
                            ),
                            html.P(
                                "Since transformer architectures such as DistilBERT have a maximum input "
                                "length constraint, a temporary tokenization step was applied to the article "
                                "descriptions."
                            ),
                            html.P(
                                "Articles whose tokenized description length exceeded 256 tokens, including "
                                "special tokens, were removed from the dataset. This avoids implicit truncation "
                                "in BERT-based models and ensures that both BiLSTM and transformer models are "
                                "trained on the same set of observations."
                            ),
                            html.P(
                                "This filtering step makes the comparison between the two architectures more consistent and fair.",
                            ),
                        ]
                    ),
                    className="shadow-sm border-0",
                ),
                width=12,
                className="mb-4",
            )
        ),
        # --- Section 3: Variables Description ---
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H3("3. Variables Description", className="mb-3"),
                            dbc.Table(
                                [
                                    html.Thead(
                                        html.Tr(
                                            [html.Th("Variable"), html.Th("Meaning")]
                                        )
                                    ),
                                    html.Tbody(
                                        [
                                            html.Tr([html.Td("date_publish"), html.Td("Publication date of the article")]),
                                            html.Tr([html.Td("description"), html.Td("Short summary of the article")]),
                                            html.Tr([html.Td("maintext"), html.Td("Full article text")]),
                                            html.Tr([html.Td("title"), html.Td("Headline of the article")]),
                                            html.Tr([html.Td("mentioned_companies"), html.Td("Companies explicitly mentioned in the article")]),
                                            html.Tr([html.Td("related_companies"), html.Td("Companies related to the mentioned firms")]),
                                            html.Tr([html.Td("industries"), html.Td("Industry identifiers associated with the article")]),
                                            html.Tr([html.Td("sentiment"), html.Td("Sentiment probability distribution")]),
                                            html.Tr([html.Td("emotion"), html.Td("Emotion probability distribution")]),
                                            html.Tr([html.Td("sentiment_label"), html.Td("Final sentiment class assigned to the article")]),
                                        ]
                                    ),
                                ],
                                bordered=True,
                                hover=True,
                                striped=True,
                                responsive=True,
                            ),
                        ]
                    ),
                    className="shadow-sm border-0",
                ),
                width=12,
                className="mb-4",
            )
        ),
        # --- Section 4: Original vs Balanced comparison header ---
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H3(
                                "4. Original vs Balanced Dataset", className="mb-3"
                            ),
                            html.P(
                                "The following charts compare the original dataset and the balanced dataset "
                                "to highlight differences in sentiment distribution, text length, temporal "
                                "coverage, and company representation."
                            ),
                        ]
                    ),
                    className="shadow-sm border-0 bg-light",
                ),
                width=12,
                className="mb-4",
            )
        ),
        # --- Side-by-side chart comparison ---
        dbc.Row(
            [
                # Left column: Original Dataset
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H3(
                                    "Original Dataset",
                                    className="text-center mb-4",
                                ),
                                html.H5("Sentiment Distribution", className="mb-3"),
                                dcc.Graph(
                                    figure=full_sentiment_fig,
                                    config={"displayModeBar": False},
                                ),
                                html.H5(
                                    "Text Length Distribution",
                                    className="mt-4 mb-3",
                                ),
                                dcc.Graph(
                                    figure=full_length_fig,
                                    config={"displayModeBar": False},
                                ),
                                html.H5("Articles per Year", className="mt-4 mb-3"),
                                dcc.Graph(
                                    figure=full_year_fig,
                                    config={"displayModeBar": False},
                                ),
                                html.H5(
                                    "Sentiment Evolution", className="mt-4 mb-3"
                                ),
                                dcc.Graph(
                                    figure=full_sentiment_time_fig,
                                    config={"displayModeBar": False},
                                ),
                                html.H5(
                                    "Top Mentioned Companies",
                                    className="mt-4 mb-3",
                                ),
                                dcc.Graph(
                                    figure=full_top_companies_fig,
                                    config={"displayModeBar": False},
                                ),
                            ]
                        ),
                        className="shadow-sm border-0 h-100",
                    ),
                    md=6,
                    className="mb-4",
                ),
                # Right column: Balanced Dataset
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H3(
                                    "Balanced Dataset",
                                    className="text-center mb-4",
                                ),
                                html.H5("Sentiment Distribution", className="mb-3"),
                                dcc.Graph(
                                    figure=balanced_sentiment_fig,
                                    config={"displayModeBar": False},
                                ),
                                html.H5(
                                    "Text Length Distribution",
                                    className="mt-4 mb-3",
                                ),
                                dcc.Graph(
                                    figure=balanced_length_fig,
                                    config={"displayModeBar": False},
                                ),
                                html.H5("Articles per Year", className="mt-4 mb-3"),
                                dcc.Graph(
                                    figure=balanced_year_fig,
                                    config={"displayModeBar": False},
                                ),
                                html.H5(
                                    "Sentiment Evolution", className="mt-4 mb-3"
                                ),
                                dcc.Graph(
                                    figure=balanced_sentiment_time_fig,
                                    config={"displayModeBar": False},
                                ),
                                html.H5(
                                    "Top Mentioned Companies",
                                    className="mt-4 mb-3",
                                ),
                                dcc.Graph(
                                    figure=balanced_top_companies_fig,
                                    config={"displayModeBar": False},
                                ),
                            ]
                        ),
                        className="shadow-sm border-0 h-100",
                    ),
                    md=6,
                    className="mb-4",
                ),
            ]
        ),
    ],
    fluid=True,
    className="py-4",
)