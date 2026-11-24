import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output, State, dcc, html
from .common import simulate_ensemble, table

TITLE = "Optimize Variation"

def layout():
    return html.Div([html.H1("Variation Optimization Tool"),
        html.P("Compare prediction randomness settings against historical tournament outcomes."),
        html.Label("Simulation runs per setting"), dcc.Slider(1, 20, 1, value=5, id="opt-runs", marks={1:"1",5:"5",10:"10",15:"15",20:"20"}),
        html.Button("Run Optimization", id="optimize", n_clicks=0), html.Div(id="optimization-results", style={"marginTop":"18px"})])

def register_callbacks(app):
    @app.callback(Output("optimization-results", "children"), Input("optimize", "n_clicks"), State("opt-runs", "value"), prevent_initial_call=True)
    def optimize(_n_clicks, runs):
        rows = []
        for variation in range(16):
            scores = []
            for i in range(min(int(runs or 1), 20)):
                for year in [y for y in range(2008, 2026) if y != 2020]:
                    try:
                        result, _, _ = simulate_ensemble(year, variation, seed=i + 7)
                        scores.append(result.ESPN_Score.sum())
                    except Exception:
                        continue
            rows.append({"Variation": variation, "Average ESPN score": round(float(np.mean(scores)), 1) if scores else 0, "Evaluations": len(scores)})
        df = pd.DataFrame(rows)
        return html.Div([html.H2("Historical Variation Analysis"), table(df, 16),
                         dcc.Graph(figure=px.line(df, x="Variation", y="Average ESPN score", markers=True))])
