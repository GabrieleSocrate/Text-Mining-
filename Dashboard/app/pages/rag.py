import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc

import sys
from pathlib import Path

# Add the project root directory to the Python path
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

# Import the RAG pipeline function
from RAG.RAG import rag_answer

# Register this page in the Dash multipage app
dash.register_page(__name__, path="/rag")

layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                [
                    # Page header
                    html.Div(
                        [
                            html.H2("RAG Assistant", className="mb-2"),
                            html.P(
                                "Ask questions about financial documents and retrieve AI-generated answers.",
                                className="text-muted mb-4",
                            ),
                        ],
                        className="text-center mt-4 mb-4",
                    ),

                    # Input card
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Ask a question", className="mb-3"),
                                dcc.Textarea(
                                    id="rag-question",
                                    placeholder="Ask about financial reports, company performance, market trends, or document content...",
                                    style={
                                        "width": "100%",
                                        "height": "140px",
                                        "padding": "12px",
                                        "borderRadius": "10px",
                                        "border": "1px solid #ced4da",
                                        "resize": "none",
                                        "fontSize": "16px",
                                    },
                                ),
                                html.Div(
                                    dbc.Button(
                                        "Generate Answer",
                                        id="rag-button",
                                        color="primary",
                                        className="mt-3 px-4",
                                    ),
                                    className="d-flex justify-content-end",
                                ),
                            ]
                        ),
                        className="shadow-sm border-0 mb-4",
                        style={"borderRadius": "14px"},
                    ),

                    # Output card
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Answer", className="mb-3"),
                                dcc.Loading(
                                    type="circle",
                                    children=html.Div(
                                        id="rag-answer-container",
                                        children=dcc.Markdown(
                                            "The generated answer will appear here.",
                                            id="rag-answer",
                                            style={
                                                "whiteSpace": "pre-wrap",
                                                "fontSize": "16px",
                                                "lineHeight": "1.7",
                                            },
                                        ),
                                        style={
                                            "minHeight": "160px",
                                            "padding": "10px",
                                            "backgroundColor": "#f8f9fa",
                                            "borderRadius": "10px",
                                            "border": "1px solid #e9ecef",
                                        },
                                    ),
                                ),
                            ]
                        ),
                        className="shadow-sm border-0 mb-4",
                        style={"borderRadius": "14px"},
                    ),
                ],
                xs=12,
                sm=11,
                md=10,
                lg=8,
                xl=8,
            ),
            justify="center",
        )
    ],
    fluid=True,
)

# Generate an answer from the RAG pipeline
@callback(
    Output("rag-answer", "children"),
    Input("rag-button", "n_clicks"),
    State("rag-question", "value"),
    prevent_initial_call=True,
)
def generate_answer(n_clicks, question):
    if not question or not question.strip():
        return "Please write a question before submitting."

    try:
        answer = rag_answer(question)

        if not answer:
            return "No answer was generated."

        return answer

    except Exception as e:
        return f"**Error:** {str(e)}"