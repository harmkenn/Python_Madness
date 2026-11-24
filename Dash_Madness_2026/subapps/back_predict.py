import numpy as np
from dash import Input, Output, dcc, html
from sklearn.linear_model import LinearRegression
from .common import csv, table

TITLE = "Back Predict"

def layout():
    return html.Div([html.H1("Predicting Each Game Independently"), html.Label("Tournament year"),
        dcc.Dropdown(list(range(2008, 2027)), 2025, id="back-year", clearable=False), html.Div(id="back-output", style={"marginTop":"16px"})])

def register_callbacks(app):
    @app.callback(Output("back-output", "children"), Input("back-year", "value"))
    def predict_games(year):
        if year == 2020:
            return html.P("No tournament was held in 2020.")
        source = csv("step05g_FUStats.csv")
        result = source[(source.Year == year) & (source.Game >= 1)].copy()
        if result.empty:
            return html.P(f"No tournament data is available for {year}.")
        numeric = source[source.Game >= 1].select_dtypes(exclude="object")
        features = numeric.drop(columns=[c for c in ["AFScore", "AUScore"] if c in numeric], errors="ignore")
        train_x, test_x = features[features.Year != year], features[features.Year == year]
        train_y = numeric[numeric.Year != year]
        model_f = LinearRegression().fit(train_x, train_y.AFScore)
        model_u = LinearRegression().fit(train_x, train_y.AUScore)
        result["PFScore"], result["PUScore"] = model_f.predict(test_x), model_u.predict(test_x)
        result["Actual Winner"] = np.where(result.AFScore >= result.AUScore, result.AFTeam, result.AUTeam)
        result["Predicted Winner"] = np.where(result.PFScore >= result.PUScore, result.AFTeam, result.AUTeam)
        result["ESPN"] = np.where(result["Actual Winner"] == result["Predicted Winner"], 10 * 2 ** (result.Round - 1), 0)
        return html.Div([html.H2(f"{year} · Total ESPN Score: {result.ESPN.sum():.0f}"), table(result)])
