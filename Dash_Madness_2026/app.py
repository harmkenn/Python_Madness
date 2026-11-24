"""Dash entry point and navigation shell for Python Madness 2026."""
from dash import Dash, Input, Output, dcc, html

from subapps import (
    all_brackets,
    all_games,
    back_predict,
    bracket_maker,
    full_predict,
    new_bracket_maker,
    new_full_prediction,
    optimize_variation,
    seed_history,
    team_performance,
    team_rankings,
    team_wins,
    update_data,
)

PAGE_MODULES = [
    all_games, all_brackets, seed_history, team_performance, team_wins,
    team_rankings, back_predict, full_predict, update_data, bracket_maker,
    new_full_prediction, new_bracket_maker, optimize_variation,
]
PAGES = {f"/{module.TITLE.lower().replace(' ', '-').replace('·', '').strip('-')}": module for module in PAGE_MODULES}

app = Dash(__name__, title="Python Madness 2026", suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Aside([
        html.H2("Dash Madness 2026"),
        html.P("by Ken Harmon"),
        html.Nav([
            dcc.Link(module.TITLE, href=path, className="nav-link")
            for path, module in PAGES.items()
        ], style={"display": "grid", "gap": "8px"}),
        html.Hr(),
        html.Small("Dash edition · local data files"),
    ], style={"width": "255px", "padding": "22px", "background": "#f3f5f8", "minHeight": "100vh", "boxSizing": "border-box"}),
    html.Main([html.Div(id="page-content")], style={"padding": "28px", "flex": 1, "minWidth": 0}),
], style={"display": "flex", "fontFamily": "Arial, sans-serif", "color": "#202733"})


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def route_page(pathname):
    module = PAGES.get(pathname or "/all-games", all_games)
    return module.layout()


for page_module in PAGE_MODULES:
    register = getattr(page_module, "register_callbacks", None)
    if register:
        register(app)


if __name__ == "__main__":
    app.run(debug=True)
