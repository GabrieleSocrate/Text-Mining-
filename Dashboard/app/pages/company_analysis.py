import dash
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, Input, Output, callback
import itertools
import dash_cytoscape as cyto
import numpy as np

from utils.data_loader import get_dataset_metadata, filter_dataset

import time

dash.register_page(__name__, path="/company-analysis")

# ── Page layout ──────────────────────────────────────────────────────────────

layout = html.Div(
    className="company-analysis-page",
    children=[
        # Page header
        html.Div(
            className="page-header",
            children=[
                html.H2("Company and Sector Analysis"),
                html.P(
                    "Filter by company, date range, and dataset to explore how sentiment "
                    "is distributed across companies and sectors, track article volume over "
                    "time, and visualise co-occurrence relationships between companies.",
                    style={"maxWidth": "800px", "opacity": "0.85", "marginTop": "4px"},
                ),
            ],
        ),

        # ── Top controls row (company selector, date range, dataset) ─────
        html.Div(
            className="top-controls-row",
            style={
                "display": "flex",
                "gap": "20px",
                "alignItems": "flex-start",
                "flexWrap": "wrap",
                "marginBottom": "30px",
            },
            children=[
                # Company multi-select dropdown
                html.Div(
                    className="control-card company-selector-card",
                    style={"flex": "2", "minWidth": "320px"},
                    children=[
                        html.H4("Select Companies"),
                        dcc.Dropdown(
                            id="company-selector",
                            options=[],
                            value=[],
                            multi=True,
                            searchable=True,
                            placeholder="Search company...",
                            closeOnSelect=False,
                        ),
                    ],
                ),

                # Date range picker
                html.Div(
                    className="control-card date-range-card",
                    style={"flex": "1", "minWidth": "260px"},
                    children=[
                        html.H4("Select Date Range"),
                        dcc.DatePickerRange(
                            id="date-range-picker",
                            start_date_placeholder_text="Start Date",
                            end_date_placeholder_text="End Date",
                            display_format="YYYY-MM-DD",
                            style={"width": "100%"},
                        ),
                    ],
                ),

                # Full / Balanced dataset toggle
                html.Div(
                    className="control-card dataset-selector-card",
                    style={"flex": "1", "minWidth": "220px"},
                    children=[
                        html.H4("Select Dataset"),
                        dcc.RadioItems(
                            id="dataset-selector",
                            options=[
                                {"label": "Full Dataset", "value": "full"},
                                {"label": "Balanced Dataset", "value": "balanced"},
                            ],
                            value="full",
                            inline=False,
                            labelStyle={"display": "block", "marginBottom": "8px"},
                        ),
                    ],
                ),
            ],
        ),

        # ── Two side-by-side sentiment charts ────────────────────────────
        html.Div(
            className="charts-row",
            style={
                "display": "flex",
                "gap": "20px",
                "alignItems": "stretch",
                "flexWrap": "wrap",
                "marginBottom": "30px",
            },
            children=[
                # Company sentiment stacked bar chart
                html.Div(
                    className="chart-card half-width",
                    style={"flex": "1", "minWidth": "450px"},
                    children=[
                        html.H4("Company Sentiment Distribution"),
                        html.P("Stacked breakdown of positive, neutral, and negative articles per company.",
                               style={"fontSize": "13px", "opacity": "0.7", "marginTop": "-4px", "marginBottom": "8px"}),
                        dcc.Graph(
                            id="company-sentiment-chart",
                            figure={},
                            config={"displayModeBar": False},
                        ),
                    ],
                ),
                # Sector sentiment stacked bar chart
                html.Div(
                    className="chart-card half-width",
                    style={"flex": "1", "minWidth": "450px"},
                    children=[
                        html.H4("Sector Sentiment Distribution"),
                        html.P("Sentiment composition by industry sector; hover to see which companies belong to each sector.",
                               style={"fontSize": "13px", "opacity": "0.7", "marginTop": "-4px", "marginBottom": "8px"}),
                        dcc.Graph(
                            id="sector-sentiment-chart",
                            figure={},
                            config={"displayModeBar": False},
                        ),
                    ],
                ),
            ],
        ),

        # ── Time distribution section ────────────────────────────────────
        html.Div(
            className="chart-card full-width time-distribution-section",
            style={"marginBottom": "30px"},
            children=[
                html.H4("Article Distribution Over Time"),
                html.P("Volume of articles over time split by sentiment; switch between daily, weekly, monthly, or yearly aggregation.",
                       style={"fontSize": "13px", "opacity": "0.7", "marginTop": "-4px", "marginBottom": "8px"}),
                dcc.RadioItems(
                    id="time-aggregation-selector",
                    options=[
                        {"label": "Daily", "value": "D"},
                        {"label": "Weekly", "value": "W"},
                        {"label": "Monthly", "value": "M"},
                        {"label": "Yearly", "value": "Y"},
                    ],
                    value="M",
                    inline=True,
                    labelStyle={"marginRight": "20px"},
                    style={"marginBottom": "15px"},
                ),
                dcc.Graph(
                    id="time-distribution-chart",
                    figure={},
                    config={"displayModeBar": False},
                ),
            ],
        ),

        # ── Company co-occurrence network ────────────────────────────────
        html.Div(
            className="chart-card full-width network-section",
            children=[
                html.H4("Company Connection Network"),
                html.P("Co-occurrence network: companies that appear together in the same article are linked; edge thickness reflects frequency.",
                       style={"fontSize": "13px", "opacity": "0.7", "marginTop": "-4px", "marginBottom": "8px"}),
                cyto.Cytoscape(
                    id="company-network-graph",
                    elements=[],
                    layout={"name": "cose"},
                    style={
                        "width": "100%",
                        "height": "500px",
                        "border": "1px solid #ddd",
                        "borderRadius": "10px",
                        "backgroundColor": "#fafafa",
                    },
                    stylesheet=[
                        # Default node style
                        {
                            "selector": "node",
                            "style": {
                                "label": "data(label)",
                                "text-valign": "center",
                                "text-halign": "center",
                                "font-size": "11px",
                                "font-family": "Inter, sans-serif",
                                "width": "mapData(size, 1, 50, 30, 70)",
                                "height": "mapData(size, 1, 50, 30, 70)",
                                "background-color": "#5dade2",
                                "color": "#1f2d3d",
                                "text-wrap": "wrap",
                                "text-max-width": "80px",
                                "tooltip": "data(tooltip)",
                            },
                        },
                        # Highlighted style for user-selected companies
                        {
                            "selector": ".selected-company",
                            "style": {
                                "background-color": "#2ecc71",
                                "border-width": 3,
                                "border-color": "#1e8449",
                                "font-weight": "bold",
                                "font-size": "13px",
                            },
                        },
                        # Edge style – width and colour mapped to co-occurrence weight
                        {
                            "selector": "edge",
                            "style": {
                                "curve-style": "bezier",
                                "width": "mapData(weight, 1, 20, 1, 8)",
                                "line-color": "data(edge_color)",
                                "opacity": 0.7,
                                "tooltip": "data(tooltip)",
                            },
                        },
                        # Reduce overlay flash on click
                        {
                            "selector": "node:active",
                            "style": {
                                "overlay-opacity": 0.1,
                            },
                        },
                    ],
                ),
            ],
        ),
    ],
    style={"padding": "20px"},
)


# ── Callbacks ────────────────────────────────────────────────────────────────


@callback(
    Output("company-selector", "options"),
    Output("company-selector", "value"),
    Output("date-range-picker", "min_date_allowed"),
    Output("date-range-picker", "max_date_allowed"),
    Output("date-range-picker", "start_date"),
    Output("date-range-picker", "end_date"),
    Input("dataset-selector", "value"),
)
def update_company_and_date_controls(selected_dataset):
    """Populate the company dropdown and date-range picker from dataset metadata."""
    t0 = time.time()

    metadata = get_dataset_metadata(selected_dataset)

    companies = metadata["companies"]
    min_date = metadata["min_date"]
    max_date = metadata["max_date"]

    company_options = [{"label": company, "value": company} for company in companies]

    print(f"Callback time: {time.time() - t0:.3f} seconds")

    return (
        company_options,
        companies,
        min_date,
        max_date,
        min_date,
        max_date,
    )


# ── Company sentiment stacked bar chart ──────────────────────────────────────


@callback(
    Output("company-sentiment-chart", "figure"),
    Input("dataset-selector", "value"),
    Input("company-selector", "value"),
    Input("date-range-picker", "start_date"),
    Input("date-range-picker", "end_date"),
)
def update_company_sentiment_chart(
    selected_dataset,
    selected_companies,
    start_date,
    end_date,
):
    """Build a stacked bar chart of sentiment counts per company."""
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="No data available for the selected filters",
        template="plotly_white",
        xaxis_title="Company",
        yaxis_title="Number of Articles",
    )

    df = filter_dataset(
        dataset_name=selected_dataset,
        selected_companies=selected_companies,
        start_date=start_date,
        end_date=end_date,
    )

    if df.empty or "mentioned_companies" not in df.columns or "sentiment_label" not in df.columns:
        return empty_fig

    df = df.dropna(subset=["mentioned_companies", "sentiment_label"]).copy()
    if df.empty:
        return empty_fig

    # A single article mentioning multiple companies counts once per company
    df = df.explode("mentioned_companies").copy()

    if df.empty:
        return empty_fig

    df["mentioned_companies"] = df["mentioned_companies"].astype(str).str.strip()
    df = df[df["mentioned_companies"].ne("")]
    df = df[df["mentioned_companies"].str.lower().ne("none")]

    # Keep only the companies that the user selected
    if selected_companies:
        selected_companies_set = {str(company).strip() for company in selected_companies}
        df = df[df["mentioned_companies"].isin(selected_companies_set)]

    if df.empty:
        return empty_fig

    # Aggregate counts by company × sentiment
    grouped = (
        df.groupby(["mentioned_companies", "sentiment_label"])
        .size()
        .reset_index(name="count")
    )

    pivot_df = grouped.pivot(
        index="mentioned_companies",
        columns="sentiment_label",
        values="count",
    ).fillna(0)

    for col in ["positive", "neutral", "negative"]:
        if col not in pivot_df.columns:
            pivot_df[col] = 0

    pivot_df = pivot_df[["positive", "neutral", "negative"]]
    pivot_df["total"] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values("total", ascending=False)

    if pivot_df.empty:
        return empty_fig

    # Percentage columns for the tooltip
    pivot_df["positive_pct"] = (pivot_df["positive"] / pivot_df["total"] * 100).fillna(0)
    pivot_df["neutral_pct"] = (pivot_df["neutral"] / pivot_df["total"] * 100).fillna(0)
    pivot_df["negative_pct"] = (pivot_df["negative"] / pivot_df["total"] * 100).fillna(0)

    x_values = pivot_df.index.tolist()

    colors = {
        "positive": "#2ecc71",
        "neutral": "#5dade2",
        "negative": "#e74c3c",
    }

    # Custom data array for the unified hover template
    customdata = list(zip(
        pivot_df["total"],
        pivot_df["positive"],
        pivot_df["positive_pct"].round(1),
        pivot_df["neutral"],
        pivot_df["neutral_pct"].round(1),
        pivot_df["negative"],
        pivot_df["negative_pct"].round(1),
    ))

    # Only the first (bottom) trace shows the full tooltip; others are hidden
    full_hover = (
        "<b>%{x}</b><br>"
        "─────────────────<br>"
        "<b>Total Articles:</b> %{customdata[0]:.0f}<br>"
        "🟢 Positive: %{customdata[1]:.0f} (%{customdata[2]:.1f}%)<br>"
        "🔵 Neutral: %{customdata[3]:.0f} (%{customdata[4]:.1f}%)<br>"
        "🔴 Negative: %{customdata[5]:.0f} (%{customdata[6]:.1f}%)"
        "<extra></extra>"
    )

    fig = go.Figure()

    # Trace order bottom → top: positive, neutral, negative
    traces = [
        ("Positive", pivot_df["positive"], colors["positive"], full_hover, customdata, False),
        ("Neutral", pivot_df["neutral"], colors["neutral"], None, None, True),
        ("Negative", pivot_df["negative"], colors["negative"], None, None, True),
    ]

    for name, y_vals, color, hover, cdata, skip in traces:
        trace_kwargs = dict(
            x=x_values,
            y=y_vals,
            name=name,
            marker_color=color,
            marker_line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
        )

        if skip:
            trace_kwargs["hoverinfo"] = "skip"
        else:
            trace_kwargs["hovertemplate"] = hover
            trace_kwargs["customdata"] = cdata

        fig.add_trace(go.Bar(**trace_kwargs))

    fig.update_layout(
        barmode="stack",
        template="plotly_white",
        title=None,
        xaxis=dict(
            title="Company",
            categoryorder="array",
            categoryarray=x_values,
            tickangle=-90,
        ),
        yaxis=dict(
            title="Number of Articles",
            gridcolor="rgba(0,0,0,0.06)",
        ),
        legend=dict(
            title="Sentiment",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            traceorder="normal",
        ),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_family="Inter, sans-serif",
        ),
        margin=dict(t=40, r=20, b=100, l=60),
        bargap=0.15,
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# ── Sector sentiment stacked bar chart ───────────────────────────────────────


@callback(
    Output("sector-sentiment-chart", "figure"),
    Input("dataset-selector", "value"),
    Input("company-selector", "value"),
    Input("date-range-picker", "start_date"),
    Input("date-range-picker", "end_date"),
)
def update_sector_sentiment_chart(
    selected_dataset,
    selected_companies,
    start_date,
    end_date,
):
    """Build a stacked bar chart of sentiment counts per sector."""
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="No data available for the selected filters",
        template="plotly_white",
        xaxis_title="Sector",
        yaxis_title="Number of Articles",
    )

    df = filter_dataset(
        dataset_name=selected_dataset,
        selected_companies=selected_companies,
        start_date=start_date,
        end_date=end_date,
    )

    if df.empty or "sector_group" not in df.columns or "sentiment_label" not in df.columns:
        return empty_fig

    df = df.dropna(subset=["sector_group", "sentiment_label"]).copy()
    if df.empty:
        return empty_fig

    # Ensure the column exists even if empty
    if "mentioned_companies" not in df.columns:
        df["mentioned_companies"] = [[] for _ in range(len(df))]

    # A single article with multiple sectors counts once per sector
    sector_df = df.explode("sector_group").copy()

    if sector_df.empty:
        return empty_fig

    sector_df["sector_group"] = sector_df["sector_group"].astype(str).str.strip()
    sector_df = sector_df[sector_df["sector_group"].ne("")]
    sector_df = sector_df[sector_df["sector_group"].str.lower().ne("none")]

    if sector_df.empty:
        return empty_fig

    # Aggregate sentiment counts per sector
    grouped = (
        sector_df.groupby(["sector_group", "sentiment_label"])
        .size()
        .reset_index(name="count")
    )

    pivot_df = grouped.pivot(
        index="sector_group",
        columns="sentiment_label",
        values="count",
    ).fillna(0)

    for col in ["positive", "neutral", "negative"]:
        if col not in pivot_df.columns:
            pivot_df[col] = 0

    pivot_df = pivot_df[["positive", "neutral", "negative"]]
    pivot_df["total"] = pivot_df.sum(axis=1)

    if pivot_df.empty:
        return empty_fig

    # Percentage columns for the tooltip
    pivot_df["positive_pct"] = (pivot_df["positive"] / pivot_df["total"] * 100).fillna(0)
    pivot_df["neutral_pct"] = (pivot_df["neutral"] / pivot_df["total"] * 100).fillna(0)
    pivot_df["negative_pct"] = (pivot_df["negative"] / pivot_df["total"] * 100).fillna(0)

    # Build a mapping: sector → comma-separated list of companies present in the filtered data
    company_sector_df = df[["sector_group", "mentioned_companies"]].copy()
    company_sector_df = company_sector_df.explode("sector_group")
    company_sector_df = company_sector_df.explode("mentioned_companies")

    if not company_sector_df.empty:
        company_sector_df["sector_group"] = company_sector_df["sector_group"].astype(str).str.strip()
        company_sector_df["mentioned_companies"] = company_sector_df["mentioned_companies"].astype(str).str.strip()

        company_sector_df = company_sector_df[company_sector_df["sector_group"].ne("")]
        company_sector_df = company_sector_df[company_sector_df["sector_group"].str.lower().ne("none")]
        company_sector_df = company_sector_df[company_sector_df["mentioned_companies"].ne("")]
        company_sector_df = company_sector_df[company_sector_df["mentioned_companies"].str.lower().ne("none")]

        # Format the company list for the tooltip: max 5 per row, max 15 shown
        MAX_SHOWN = 15
        PER_ROW = 5

        def _format_company_list(names):
            unique = sorted(set(names))
            total = len(unique)
            shown = unique[:MAX_SHOWN]
            # Split into rows of PER_ROW companies each
            rows = [
                ", ".join(shown[i : i + PER_ROW])
                for i in range(0, len(shown), PER_ROW)
            ]
            text = "<br>".join(rows)
            if total > MAX_SHOWN:
                text += f"<br>+ {total - MAX_SHOWN} more"
            return text

        sector_companies_map = (
            company_sector_df.groupby("sector_group")["mentioned_companies"]
            .apply(_format_company_list)
            .to_dict()
        )
    else:
        sector_companies_map = {}

    pivot_df["companies_in_sector"] = pivot_df.index.map(
        lambda sector: sector_companies_map.get(sector, "No companies available")
    )

    # Sort sectors by total article volume (descending)
    pivot_df = pivot_df.sort_values("total", ascending=False)

    x_values = pivot_df.index.tolist()

    colors = {
        "positive": "#2ecc71",
        "neutral": "#5dade2",
        "negative": "#e74c3c",
    }

    # Custom data array for the unified hover template
    customdata = list(zip(
        pivot_df["total"],
        pivot_df["positive"],
        pivot_df["positive_pct"].round(1),
        pivot_df["neutral"],
        pivot_df["neutral_pct"].round(1),
        pivot_df["negative"],
        pivot_df["negative_pct"].round(1),
        pivot_df["companies_in_sector"],
    ))

    full_hover = (
        "<b>%{x}</b><br>"
        "─────────────────<br>"
        "<b>Total Articles:</b> %{customdata[0]:.0f}<br>"
        "🟢 Positive: %{customdata[1]:.0f} (%{customdata[2]:.1f}%)<br>"
        "🔵 Neutral: %{customdata[3]:.0f} (%{customdata[4]:.1f}%)<br>"
        "🔴 Negative: %{customdata[5]:.0f} (%{customdata[6]:.1f}%)<br>"
        "<b>Companies:</b> %{customdata[7]}"
        "<extra></extra>"
    )

    fig = go.Figure()

    # Trace order bottom → top: positive, neutral, negative
    traces = [
        ("Positive", pivot_df["positive"], colors["positive"], full_hover, customdata, False),
        ("Neutral", pivot_df["neutral"], colors["neutral"], None, None, True),
        ("Negative", pivot_df["negative"], colors["negative"], None, None, True),
    ]

    for name, y_vals, color, hover, cdata, skip in traces:
        trace_kwargs = dict(
            x=x_values,
            y=y_vals,
            name=name,
            marker_color=color,
            marker_line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
        )

        if skip:
            trace_kwargs["hoverinfo"] = "skip"
        else:
            trace_kwargs["hovertemplate"] = hover
            trace_kwargs["customdata"] = cdata

        fig.add_trace(go.Bar(**trace_kwargs))

    fig.update_layout(
        barmode="stack",
        template="plotly_white",
        title=None,
        xaxis=dict(
            title="Sector",
            categoryorder="array",
            categoryarray=x_values,
            tickangle=-30,
        ),
        yaxis=dict(
            title="Number of Articles",
            gridcolor="rgba(0,0,0,0.06)",
        ),
        legend=dict(
            title="Sentiment",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            traceorder="normal",
        ),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_family="Inter, sans-serif",
        ),
        margin=dict(t=40, r=20, b=80, l=60),
        bargap=0.15,
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# ── Time distribution stacked bar chart ──────────────────────────────────────


@callback(
    Output("time-distribution-chart", "figure"),
    Input("dataset-selector", "value"),
    Input("company-selector", "value"),
    Input("date-range-picker", "start_date"),
    Input("date-range-picker", "end_date"),
    Input("time-aggregation-selector", "value"),
)
def update_time_distribution_chart(
    selected_dataset,
    selected_companies,
    start_date,
    end_date,
    time_aggregation,
):
    """Build a stacked bar chart of sentiment counts aggregated over time."""
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="No data available for the selected filters",
        template="plotly_white",
        xaxis_title="Time",
        yaxis_title="Number of Articles",
    )

    df = filter_dataset(
        dataset_name=selected_dataset,
        selected_companies=selected_companies,
        start_date=start_date,
        end_date=end_date,
    )

    if df.empty or "date_publish" not in df.columns or "sentiment_label" not in df.columns:
        return empty_fig

    df = df.dropna(subset=["date_publish", "sentiment_label"]).copy()
    if df.empty:
        return empty_fig

    # Map radio-button value to pandas period frequency
    freq_map = {"D": "D", "W": "W", "M": "M", "Y": "Y"}
    aggregation_labels = {"D": "Daily", "W": "Weekly", "M": "Monthly", "Y": "Yearly"}

    # Convert dates to period start timestamps for grouping
    df["time_period"] = df["date_publish"].dt.to_period(freq_map[time_aggregation]).dt.to_timestamp()

    grouped = (
        df.groupby(["time_period", "sentiment_label"])
        .size()
        .reset_index(name="count")
    )

    pivot_df = grouped.pivot(
        index="time_period", columns="sentiment_label", values="count"
    ).fillna(0)

    for col in ["positive", "neutral", "negative"]:
        if col not in pivot_df.columns:
            pivot_df[col] = 0

    pivot_df = pivot_df[["positive", "neutral", "negative"]].sort_index()
    pivot_df["total"] = pivot_df.sum(axis=1)

    # Percentage columns for the tooltip
    pivot_df["positive_pct"] = (pivot_df["positive"] / pivot_df["total"] * 100).fillna(0)
    pivot_df["neutral_pct"] = (pivot_df["neutral"] / pivot_df["total"] * 100).fillna(0)
    pivot_df["negative_pct"] = (pivot_df["negative"] / pivot_df["total"] * 100).fillna(0)

    x_values = pivot_df.index

    colors = {
        "positive": "#2ecc71",
        "neutral":  "#5dade2",
        "negative": "#e74c3c",
    }

    # Custom data array for the unified hover template
    customdata = list(zip(
        pivot_df["total"],
        pivot_df["positive"],
        pivot_df["positive_pct"].round(1),
        pivot_df["neutral"],
        pivot_df["neutral_pct"].round(1),
        pivot_df["negative"],
        pivot_df["negative_pct"].round(1),
    ))

    # Date format in the tooltip depends on the chosen aggregation level
    date_fmt_map = {"D": "%d %b %Y", "W": "Week of %d %b %Y", "M": "%b %Y", "Y": "%Y"}
    date_fmt = date_fmt_map[time_aggregation]

    full_hover = (
        f"<b>%{{x|{date_fmt}}}</b><br>"
        "─────────────────<br>"
        "<b>Total:</b> %{customdata[0]:.0f}<br>"
        "🟢 Positive: %{customdata[1]:.0f} (%{customdata[2]:.1f}%)<br>"
        "🔵 Neutral: %{customdata[3]:.0f} (%{customdata[4]:.1f}%)<br>"
        "🔴 Negative: %{customdata[5]:.0f} (%{customdata[6]:.1f}%)"
        "<extra></extra>"
    )

    fig = go.Figure()

    # Trace order bottom → top: positive, neutral, negative
    traces = [
        ("Positive", pivot_df["positive"], colors["positive"], full_hover, customdata, False),
        ("Neutral",  pivot_df["neutral"],  colors["neutral"],  None, None, True),
        ("Negative", pivot_df["negative"], colors["negative"], None, None, True),
    ]

    for name, y_vals, color, hover, cdata, skip in traces:
        trace_kwargs = dict(
            x=x_values,
            y=y_vals,
            name=name,
            marker_color=color,
            marker_line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
        )
        if skip:
            trace_kwargs["hoverinfo"] = "skip"
        else:
            trace_kwargs["hovertemplate"] = hover
            trace_kwargs["customdata"] = cdata
        fig.add_trace(go.Bar(**trace_kwargs))

    fig.update_layout(
        barmode="stack",
        template="plotly_white",
        title=None,
        xaxis=dict(
            title="Time",
            tickformat=date_fmt_map.get(time_aggregation, "%b %Y"),
            type="date",
        ),
        yaxis=dict(
            title="Number of Articles",
            gridcolor="rgba(0,0,0,0.06)",
        ),
        legend=dict(
            title="Sentiment",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            traceorder="normal",
        ),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_family="Inter, sans-serif",
        ),
        margin=dict(t=40, r=20, b=40, l=60),
        bargap=0.08,
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# ── Company co-occurrence network graph ──────────────────────────────────────


@callback(
    Output("company-network-graph", "elements"),
    Output("company-network-graph", "layout"),
    Input("dataset-selector", "value"),
    Input("company-selector", "value"),
    Input("date-range-picker", "start_date"),
    Input("date-range-picker", "end_date"),
)
def update_company_network_graph(
    selected_dataset,
    selected_companies,
    start_date,
    end_date,
):
    """Build Cytoscape elements for the company co-occurrence network."""
    empty_layout = {"name": "cose"}

    df = filter_dataset(
        dataset_name=selected_dataset,
        selected_companies=selected_companies,
        start_date=start_date,
        end_date=end_date,
    )

    if (
        df.empty
        or "mentioned_companies" not in df.columns
        or not selected_companies
    ):
        return [], empty_layout

    df = df.dropna(subset=["mentioned_companies"]).copy()
    if df.empty:
        return [], empty_layout

    selected_companies_set = {str(c).strip() for c in selected_companies}

    # Count co-occurrences (edges) and article mentions (node sizes)
    edge_weights = {}
    node_article_counts = {}

    for companies in df["mentioned_companies"]:
        if not isinstance(companies, list):
            continue

        cleaned = sorted({
            str(c).strip()
            for c in companies
            if str(c).strip() and str(c).strip().lower() != "none"
        })

        if len(cleaned) == 0:
            continue

        # Only consider articles that mention at least one selected company
        if not any(c in selected_companies_set for c in cleaned):
            continue

        for c in cleaned:
            node_article_counts[c] = node_article_counts.get(c, 0) + 1

        for source, target in itertools.combinations(cleaned, 2):
            edge_key = tuple(sorted((source, target)))
            edge_weights[edge_key] = edge_weights.get(edge_key, 0) + 1

    if not edge_weights and not node_article_counts:
        return [], empty_layout

    # ── Filter: keep top-N connections per selected company ──────────
    MAX_CONNECTIONS_PER_COMPANY = 15
    MIN_COOCCURRENCES = 2

    # Keep edges that involve at least one selected company and meet the minimum weight
    relevant_edges = {
        (s, t): w
        for (s, t), w in edge_weights.items()
        if (s in selected_companies_set or t in selected_companies_set)
        and w >= MIN_COOCCURRENCES
    }

    # For each selected company, retain only the strongest connections
    final_edges = {}
    for selected in selected_companies_set:
        company_edges = {
            (s, t): w
            for (s, t), w in relevant_edges.items()
            if s == selected or t == selected
        }
        top_edges = dict(
            sorted(company_edges.items(), key=lambda x: x[1], reverse=True)[
                :MAX_CONNECTIONS_PER_COMPANY
            ]
        )
        final_edges.update(top_edges)

    # Collect all nodes that are connected via at least one final edge
    connected_nodes = set(selected_companies_set)
    for s, t in final_edges:
        connected_nodes.add(s)
        connected_nodes.add(t)

    # ── Edge colour: light grey → dark blue based on weight ──────────
    max_weight = max(final_edges.values()) if final_edges else 1
    min_weight = min(final_edges.values()) if final_edges else 1

    def edge_color(weight):
        if max_weight == min_weight:
            t = 0.5
        else:
            t = (weight - min_weight) / (max_weight - min_weight)
        r = int(176 * (1 - t) + 41 * t)
        g = int(183 * (1 - t) + 128 * t)
        b = int(195 * (1 - t) + 185 * t)
        return f"rgb({r},{g},{b})"

    # ── Build Cytoscape elements ─────────────────────────────────────
    elements = []

    for company in sorted(connected_nodes):
        count = node_article_counts.get(company, 0)
        is_selected = company in selected_companies_set

        # Gather this node's connections for the tooltip
        connections = []
        for (s, t), w in final_edges.items():
            if s == company:
                connections.append((t, w))
            elif t == company:
                connections.append((s, w))
        connections.sort(key=lambda x: x[1], reverse=True)

        # Plain-text tooltip content
        tooltip_lines = [f"{company}", f"Articles: {count}"]
        if connections:
            tooltip_lines.append(f"Connections: {len(connections)}")
            for partner, w in connections[:3]:
                tooltip_lines.append(f"  ↔ {partner}: {w}")
            if len(connections) > 3:
                tooltip_lines.append(f"  + {len(connections) - 3} more")

        elements.append({
            "data": {
                "id": company,
                "label": company,
                "size": max(1, count),
                "tooltip": "\n".join(tooltip_lines),
            },
            "classes": "selected-company" if is_selected else "",
        })

    for (source, target), weight in final_edges.items():
        elements.append({
            "data": {
                "source": source,
                "target": target,
                "weight": weight,
                "tooltip": f"{source} ↔ {target}\nCo-occurrences: {weight}",
                "edge_color": edge_color(weight),
            }
        })

    # COSE layout parameters for a readable graph
    layout = {
        "name": "cose",
        "animate": True,
        "fit": True,
        "padding": 40,
        "nodeRepulsion": 8000,
        "idealEdgeLength": 120,
        "edgeElasticity": 100,
        "gravity": 0.3,
        "numIter": 1000,
    }

    return elements, layout