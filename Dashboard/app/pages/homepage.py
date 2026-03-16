import dash
from dash import html
import dash_bootstrap_components as dbc

# Register the homepage in the Dash multipage application
dash.register_page(__name__, path="/")

# =============================================================================
# Page layout
# =============================================================================

layout = dbc.Container(
    [
        # ---------------------------------------------------------------------
        # Project introduction
        # Short description of the dashboard and its main purpose
        # ---------------------------------------------------------------------
        dbc.Row(
            dbc.Col(
                [
                    html.P(
                        "An interactive dashboard for analyzing financial news and understanding how "
                        "article sentiment can influence the interpretation of market-related information. "
                        "The project focuses on classifying financial news into positive, negative, "
                        "and neutral categories using Natural Language Processing models, while also "
                        "providing visual tools to explore the data and interpret sentiment patterns.",
                        className="lead text-center",
                    ),
                ],
                width=10,
                className="mx-auto",
            )
        ),

        # ---------------------------------------------------------------------
        # Dashboard objective
        # High-level explanation of the goal of the platform
        # ---------------------------------------------------------------------
        dbc.Row(
            dbc.Col(
                dbc.Alert(
                    [
                        html.H5("Dashboard Objective", className="alert-heading text-center"),
                        html.P(
                            "The main goal of this dashboard is to provide a complete environment for "
                            "exploring financial news sentiment: from understanding the dataset, to "
                            "comparing NLP models, to analyzing company-specific sentiment trends.",
                            className="mb-0 text-center",
                        ),
                    ],
                    color="light",
                    className="shadow-sm border",
                ),
                width=10,
                className="mx-auto mb-4",
            )
        ),

        # ---------------------------------------------------------------------
        # Main dashboard sections (row 1)
        # Brief descriptions of the EDA and Sentiment Models pages
        # ---------------------------------------------------------------------
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Exploratory Data Analysis (EDA)", className="card-title"),
                                html.P(
                                    "The EDA section introduces the dataset behind the dashboard and "
                                    "shows the data on which the available tools are built."
                                ),
                                html.Ul(
                                    [
                                        html.Li("Dataset overview: structure, size, and feature descriptions."),
                                        html.Li("Sentiment distribution: class balance and label breakdowns."),
                                        html.Li("Text statistics: word counts, token lengths, and common terms."),
                                    ],
                                    className="mb-0",
                                ),
                            ]
                        ),
                        className="shadow-sm h-100 border-0",
                    ),
                    md=6,
                    className="mb-4",
                ),

                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Sentiment Models", className="card-title"),
                                html.P(
                                    "This section presents the sentiment analysis models used to classify "
                                    "financial news articles."
                                ),
                                html.Ul(
                                    [
                                        html.Li("Metrics: a comparison of the evaluation results of all implemented models."),
                                        html.Li("BiLSTM: a deep learning model based on a Bidirectional LSTM architecture."),
                                        html.Li("BERT Models: a comparison of BERT, FinBERT, and DistilBERT."),
                                    ],
                                    className="mb-0",
                                ),
                            ]
                        ),
                        className="shadow-sm h-100 border-0",
                    ),
                    md=6,
                    className="mb-4",
                ),
            ]
        ),

        # ---------------------------------------------------------------------
        # Main dashboard sections (row 2)
        # Overview of the RAG assistant and the company-level sentiment analysis
        # ---------------------------------------------------------------------
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("RAG", className="card-title"),
                                html.P(
                                    "This section explores Retrieval-Augmented Generation approaches "
                                    "applied to financial news analysis. It combines language models "
                                    "with information retrieval techniques to support richer and more "
                                    "context-aware exploration of financial content.",
                                    className="mb-0",
                                ),
                            ]
                        ),
                        className="shadow-sm h-100 border-0",
                    ),
                    md=6,
                    className="mb-4",
                ),

                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Company Analysis", className="card-title"),
                                html.P(
                                    "This section provides a deeper analysis of sentiment in articles "
                                    "related to specific companies. Interactive visualizations are used "
                                    "to highlight distributions, trends, and sentiment dynamics across "
                                    "different companies and news flows.",
                                    className="mb-0",
                                ),
                            ]
                        ),
                        className="shadow-sm h-100 border-0",
                    ),
                    md=6,
                    className="mb-4",
                ),
            ]
        ),

        # ---------------------------------------------------------------------
        # Navigation guidance
        # Suggested workflow for exploring the dashboard sections
        # ---------------------------------------------------------------------
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4("How to Navigate the Dashboard", className="text-center mb-3"),
                            html.P(
                                "Start with the EDA section to understand the dataset, then explore the "
                                "Sentiment Models page to compare model performance and architectures. "
                                "After that, use the RAG and Company Analysis sections to investigate "
                                "more advanced insights and company-level sentiment behavior.",
                                className="text-center mb-0",
                            ),
                        ]
                    ),
                    className="shadow-sm border-0 bg-light",
                ),
                width=10,
                className="mx-auto mt-2",
            )
        ),
    ],

    # Fluid container allows the layout to adapt to different screen sizes
    fluid=True,
    className="py-4",
)