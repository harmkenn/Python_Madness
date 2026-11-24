from dash import dcc, html
from .common import csv, table

TITLE = "All Brackets"

def layout():
    return html.Div([html.H1("All Tournament Brackets Since 2008"), html.Label("Tournament year"),
        dcc.Dropdown(list(range(2008, 2027)), 2026, id="brackets-year", clearable=False), html.Div(id="brackets-output", style={"marginTop":"16px"})])

def register_callbacks(app):
    from dash import Input, Output
    @app.callback(Output("brackets-output", "children"), Input("brackets-year", "value"))
    def show_bracket(year):
        if year == 2020:
            return html.P("No tournament was held in 2020.")
        df = csv("step05c_FUHistory.csv")
        df = df[(df.Year == year) & (df.Game >= 1)].sort_values(["Round", "Game"])
        if df.empty:
            return html.P(f"No bracket data is available for {year}.")
        return html.Div([html.H2(f"Bracket for {year}"), table(df, 64)])
