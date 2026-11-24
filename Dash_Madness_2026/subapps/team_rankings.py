from dash import Input, Output, dcc, html
from .common import csv, table, team_filter

TITLE = "Team Rankings"

def layout():
    return html.Div([html.H1("Team Rankings Since 2008"), html.Label("Filter by team"),
        dcc.Input(id="rank-team", type="text", debounce=True, placeholder="Any team"), html.Div(id="rank-output", style={"marginTop":"16px"})])

def register_callbacks(app):
    @app.callback(Output("rank-output", "children"), Input("rank-team", "value"))
    def show_rankings(team):
        df = csv("step05f_AllStats.csv")
        df["Year"] = df.Year.astype(str)
        return table(team_filter(df, team or "", ["Team"]))
