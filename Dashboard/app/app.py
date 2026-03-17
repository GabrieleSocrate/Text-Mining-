import dash
from dash import html
import dash_bootstrap_components as dbc

# Initialize Dash app with multi-page support and Bootstrap styling
app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Navigation bar for dashboard pages
navbar = dbc.Nav(
    [
        dbc.NavLink("Homepage", href="/", active="exact"),
        dbc.NavLink("EDA", href="/eda", active="exact"),
        dbc.NavLink("Sentiment Models", href="/sentiment-models", active="exact"),
        dbc.NavLink("RAG", href="/rag", active="exact"),
        dbc.NavLink("Company Analysis", href="/company-analysis", active="exact"),
    ],
    pills=True,
    horizontal=True,
    className="justify-content-center",
)

# Main app layout
app.layout = dbc.Container(
    [
        html.H1(
            "Financial sentiment analysis",
            style={"textAlign": "center", "marginTop": "20px"}
        ),
        html.Hr(),
        navbar,
        html.Br(),
        dash.page_container,  # Loads selected page
    ],
    fluid=True,
)

# Run the application
if __name__ == "__main__":
    app.run(debug=True)