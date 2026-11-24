from dash import Input, Output, dcc, html
from .common import csv, table, team_filter

TITLE = "All Games"

def layout():
    return html.Div([html.H1("All Tournament Games Since 2008"), html.Label("Filter by team"),
                     dcc.Input(id="games-team", type="text", debounce=True, placeholder="Any team"),
                     html.Div(id="games-output", style={"marginTop": "16px"})])

def register_callbacks(app):
    @app.callback(Output("games-output", "children"), Input("games-team", "value"))
    def show_games(team):
        df = csv("step05c_FUHistory.csv")
        df = df[df.Year <= 2026]
        return table(team_filter(df, team or "", ["AFTeam", "AUTeam"]), sort_by=[{"column_id": "Year", "direction": "desc"}])
