import numpy as np
import pandas as pd
from dash import Input, Output, dcc, html
from .common import csv, table, team_filter

TITLE = "Team Wins"

def layout():
    return html.Div([html.H1("Team Wins Since 2008"), html.Label("Filter by team"),
        dcc.Input(id="wins-team", type="text", debounce=True, placeholder="Any team"), html.Div(id="wins-output", style={"marginTop":"16px"})])

def register_callbacks(app):
    @app.callback(Output("wins-output", "children"), Input("wins-team", "value"))
    def show_wins(team):
        df = csv("step05c_FUHistory.csv")
        df = df[df.Year <= 2026]
        champs = df[df.Round == 6].copy()
        champs["Champion"] = np.where(champs.AFScore > champs.AUScore, champs.AFTeam, champs.AUTeam)
        fw, uw = pd.crosstab(df.AFTeam, df.Year), pd.crosstab(df.AUTeam, df.Year)
        wins = fw.add(uw, fill_value=0).sub(1).clip(lower=0).astype(int)
        champ_wins = pd.crosstab(champs.Champion, champs.Year)
        wins = wins.add(champ_wins, fill_value=0).fillna(0).astype(int).reset_index(names="Team")
        return table(team_filter(wins, team or "", ["Team"]), 25)
