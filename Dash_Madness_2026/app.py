"""Dash edition of Python Madness 2026.

Run from this directory with: python app.py
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, dash_table, dcc, html, no_update
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PAGES = [
    "All Games", "All Brackets", "Seed History", "Team Performance", "Team Wins",
    "Team Rankings", "Back Predict", "Full Predict", "Update Data", "Bracket Maker",
    "New Full Prediction", "New Bracket Maker", "Optimize Variation",
]
app = Dash(__name__, title="Python Madness 2026", suppress_callback_exceptions=True)
server = app.server


def csv(name):
    return pd.read_csv(DATA / name).fillna(0)


def table(df, page_size=25, height="620px"):
    if df is None or df.empty:
        return html.P("No rows match the selected options.")
    shown = df.copy()
    shown.columns = [str(c) for c in shown.columns]
    for col in shown.columns:
        dtype = shown[col].dtype
        if pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.StringDtype):
            shown[col] = shown[col].astype(str)
    return dash_table.DataTable(
        data=shown.to_dict("records"), columns=[{"name": c, "id": c} for c in shown.columns],
        page_size=page_size, sort_action="native", filter_action="native",
        style_table={"overflowX": "auto", "maxHeight": height, "overflowY": "auto"},
        style_cell={"fontFamily": "Arial", "fontSize": 13, "padding": "6px", "textAlign": "left", "minWidth": "70px", "maxWidth": "260px", "whiteSpace": "normal"},
        style_header={"fontWeight": "bold", "backgroundColor": "#edf2f7"},
        virtualization=len(shown) > 500,
    )


def team_filter(df, query, columns):
    if not query:
        return df
    pattern = re.escape(query)
    mask = pd.Series(False, index=df.index)
    for col in columns:
        if col in df:
            mask |= df[col].astype(str).str.contains(pattern, case=False, na=False)
    return df[mask]


def stats_pair(all_stats, year, favored, underdog):
    ys = all_stats[all_stats.Year == year].drop_duplicates("Team").set_index("Team")
    a = ys.loc[favored] if favored in ys.index else pd.Series(dtype=float)
    b = ys.loc[underdog] if underdog in ys.index else pd.Series(dtype=float)
    result = {}
    for col in ys.select_dtypes(include=[np.number]).columns:
        result[f"{col}_x"] = a.get(col, 0)
        result[f"{col}_y"] = b.get(col, 0)
    return result


def simulate_ensemble(year, variation=0, progressive=False, seed=42, ensemble=True):
    games = csv("step05g_FUStats.csv")
    stats = csv("step05f_AllStats.csv")
    games = games[(games.Year <= 2025) & (games.Game >= 1)].drop_duplicates(["Year", "Game"]).copy()
    games["PFSeed"], games["PUSeed"] = games.AFSeed, games.AUSeed
    games["PFTeam"], games["PUTeam"] = games.AFTeam, games.AUTeam
    games["PFScore"], games["PUScore"] = games.AFScore, games.AUScore
    games["Actual_Winner"] = np.where(games.AFScore >= games.AUScore, games.AFTeam, games.AUTeam)
    games["seed_diff"] = abs(games.PFSeed - games.PUSeed)
    games["higher_seed"] = np.minimum(games.PFSeed, games.PUSeed)
    train = games[games.Year != year].select_dtypes(exclude="object")
    drop = ["PFScore", "PUScore", "Year", "Round", "Game", "AFSeed", "AUSeed", "AFScore", "AUScore", "Fti", "Uti"]
    drop += [c for c in train if "Unnamed" in c or "Record" in c or "Team.1" in c]
    X = train.drop(columns=[c for c in set(drop) if c in train])
    if X.empty or len(train) < 10:
        raise ValueError("Not enough historical games are available to train the prediction model.")
    # The original model predicts both team scores with an LR/GBM voting ensemble.
    if ensemble:
        model_f = VotingRegressor([("lr", LinearRegression()), ("gbm", GradientBoostingRegressor(n_estimators=100, random_state=42))]).fit(X, train.PFScore)
        model_u = VotingRegressor([("lr", LinearRegression()), ("gbm", GradientBoostingRegressor(n_estimators=100, random_state=42))]).fit(X, train.PUScore)
    else:
        model_f = LinearRegression().fit(X, train.PFScore)
        model_u = LinearRegression().fit(X, train.PUScore)
    features = list(X.columns)
    actual = games[games.Year == year].set_index("Game").Actual_Winner.to_dict()
    first = games[(games.Year == year) & (games.Round == 1)].sort_values("Game")
    if first.empty:
        raise ValueError(f"No tournament games are available for {year}. Update the source data first.")
    # Bring the selected year’s team ratings into feature-compatible game rows.
    def add_features(row):
        row.update(stats_pair(stats, year, row["PFTeam"], row["PUTeam"]))
        row["scoring_margin"] = row.get("Pts_x", 0) - row.get("Pts_y", 0)
        row["seed_diff"] = abs(row["PFSeed"] - row["PUSeed"])
        row["higher_seed"] = min(row["PFSeed"], row["PUSeed"])
        return row
    rng = np.random.default_rng(seed)
    bracket = []
    current = first[["Game", "Round", "PFSeed", "PFTeam", "PUSeed", "PUTeam"]].copy()
    for round_no in range(1, 7):
        if current.empty:
            break
        current = current.sort_values("Game").copy()
        records = [add_features(r.to_dict()) for _, r in current.iterrows()]
        frame = pd.DataFrame(records)
        for feature in features:
            if feature not in frame:
                frame[feature] = 0
        pred_f = model_f.predict(frame[features])
        pred_u = model_u.predict(frame[features])
        jitter = variation + round_no - 1 if progressive else variation
        if jitter:
            pred_f = pred_f + rng.integers(-jitter, jitter + 1, len(pred_f))
        winners = []
        for i, (_, game) in enumerate(current.iterrows()):
            won_f = pred_f[i] >= pred_u[i]
            wt = game.PFTeam if won_f else game.PUTeam
            ws = game.PFSeed if won_f else game.PUSeed
            game_no = int(game.Game)
            bracket.append({"Year": year, "Round": round_no, "Game": game_no,
                            "PFTeam": game.PFTeam, "PFSeed": game.PFSeed, "PFScore": round(float(pred_f[i]), 1),
                            "PUTeam": game.PUTeam, "PUSeed": game.PUSeed, "PUScore": round(float(pred_u[i]), 1),
                            "PWTeam": wt, "PWSeed": ws,
                            "ESPN_Score": 10 * 2 ** (round_no - 1) if actual.get(game_no) == wt else 0})
            winners.append((ws, wt))
        if round_no == 6:
            break
        current = pd.DataFrame([{
            "Year": year, "Round": round_no + 1, "Game": {1: 33, 2: 49, 3: 57, 4: 61, 5: 63}[round_no] + i,
            "PFSeed": min(winners[i * 2][0], winners[i * 2 + 1][0]),
            "PFTeam": winners[i * 2][1] if winners[i * 2][0] <= winners[i * 2 + 1][0] else winners[i * 2 + 1][1],
            "PUSeed": max(winners[i * 2][0], winners[i * 2 + 1][0]),
            "PUTeam": winners[i * 2][1] if winners[i * 2][0] > winners[i * 2 + 1][0] else winners[i * 2 + 1][1],
        } for i in range(len(winners) // 2)])
    return pd.DataFrame(bracket), model_f, features


app.layout = html.Div([
    html.Aside([
        html.H2("Dash Madness 2026"), html.P("by Ken Harmon"),
        html.Label("Navigation"), dcc.Dropdown(PAGES, PAGES[0], id="page", clearable=False),
        html.Div([
            html.Div([html.Label("Team filter"), dcc.Input(id="team", type="text", debounce=True, placeholder="Any team", style={"width": "100%"})], id="team-controls", style={"display": "grid", "gap": "8px"}),
            html.Div([html.Label("Tournament year"), dcc.Dropdown(list(range(2008, 2027)), 2025, id="year", clearable=False)], id="year-controls", style={"display": "grid", "gap": "8px"}),
            html.Div([html.Label("Randomness (points)"), dcc.Slider(0, 25, 1, value=5, id="variation", marks={0:"0", 5:"5", 10:"10", 15:"15", 20:"20", 25:"25"}), dcc.Checklist([{"label":"Increase randomness each round","value":"progressive"}], [], id="progressive")], id="simulation-controls", style={"display": "grid", "gap": "8px"}),
            html.Div([html.Label("Simulation runs per setting"), dcc.Slider(1, 20, 1, value=5, id="runs", marks={1:"1",5:"5",10:"10",15:"15",20:"20"})], id="runs-controls", style={"display": "grid", "gap": "8px"}),
        ], id="page-controls", style={"marginTop": "22px", "display": "grid", "gap": "10px"}),
        html.Hr(), html.Small("Dash edition · local data files"),
    ], style={"width": "255px", "padding": "22px", "background": "#f3f5f8", "minHeight": "100vh", "boxSizing": "border-box"}),
    html.Main([html.Div(id="content")], style={"padding": "28px", "flex": 1, "minWidth": 0}),
], style={"display": "flex", "fontFamily": "Arial, sans-serif", "color": "#202733"})


@app.callback(Output("team-controls", "style"), Output("year-controls", "style"),
              Output("simulation-controls", "style"), Output("runs-controls", "style"),
              Input("page", "value"))
def controls(page):
    team_pages = {"All Games", "Team Wins", "Team Rankings"}
    year_pages = {"All Brackets", "Back Predict", "Full Predict", "Bracket Maker", "New Full Prediction", "New Bracket Maker"}
    simulation_pages = {"Bracket Maker", "New Bracket Maker"}
    visible, hidden = {"display": "grid", "gap": "8px"}, {"display": "none"}
    return (visible if page in team_pages else hidden,
            visible if page in year_pages else hidden,
            visible if page in simulation_pages else hidden,
            visible if page == "Optimize Variation" else hidden)


@app.callback(Output("content", "children"), Input("page", "value"),
              Input("team", "value"), Input("year", "value"), Input("variation", "value"),
              Input("progressive", "value"), Input("runs", "value"))
def render(page, team, year, variation, progressive, runs):
    team = team or ""
    if page == "All Games":
        df = csv("step05c_FUHistory.csv"); df = df[df.Year <= 2026]
        return html.Div([html.H1("All Tournament Games Since 2008"), table(team_filter(df, team, ["AFTeam", "AUTeam"]))])
    if page == "All Brackets":
        df = csv("step05c_FUHistory.csv"); df = df[(df.Year == year) & (df.Game >= 1)]
        if year == 2020: return html.Div([html.H1("All Tournament Brackets"), html.P("No tournament was held in 2020.")])
        if df.empty: return html.Div([html.H1(f"Bracket for {year}"), html.P("No bracket data is available for this year.")])
        return html.Div([html.H1(f"All Tournament Brackets · {year}"), table(df.sort_values(["Round", "Game"]), 64)])
    if page == "Seed History":
        df = csv("step05d_SeedHistory.csv"); df.insert(0, "Seed", np.arange(1, len(df)+1))
        return html.Div([html.H1("Seed Success History Since 2008"), table(df, 16)])
    if page == "Team Performance":
        return html.Div([html.H1("Team Performance Against Seed Expectation Since 2008"), table(csv("step05e_PASE.csv"))])
    if page == "Team Wins":
        df = csv("step05c_FUHistory.csv"); df = df[df.Year <= 2026]
        champs = df[df.Round == 6].copy()
        champs["Champion"] = np.where(champs.AFScore > champs.AUScore, champs.AFTeam, champs.AUTeam)
        fw = pd.crosstab(df.AFTeam, df.Year); uw = pd.crosstab(df.AUTeam, df.Year)
        wins = fw.add(uw, fill_value=0).sub(1).clip(lower=0).astype(int)
        cw = pd.crosstab(champs.Champion, champs.Year)
        wins = wins.add(cw, fill_value=0).fillna(0).astype(int).reset_index(names="Team")
        return html.Div([html.H1("Team Wins Since 2008"), table(team_filter(wins, team, ["Team"]), 25)])
    if page == "Team Rankings":
        df = csv("step05f_AllStats.csv"); df["Year"] = df.Year.astype(str)
        return html.Div([html.H1("Team Rankings Since 2008"), table(team_filter(df, team, ["Team"]))])
    if page == "Back Predict":
        if year == 2020: return html.Div([html.H1("Independent Game Predictions"), html.P("No tournament was held in 2020.")])
        df = csv("step05g_FUStats.csv"); df = df[(df.Year == year) & (df.Game >= 1)].copy()
        if df.empty: return html.Div([html.H1("Independent Game Predictions"), html.P(f"No tournament data is available for {year}.")])
        numeric = csv("step05g_FUStats.csv").query("Game >= 1").select_dtypes(exclude="object")
        feats = numeric.drop(columns=[c for c in ["AFScore", "AUScore"] if c in numeric], errors="ignore")
        X = feats[feats.Year != year]; target = numeric[numeric.Year != year]
        test = feats[feats.Year == year]
        mf, mu = LinearRegression().fit(X, target.AFScore), LinearRegression().fit(X, target.AUScore)
        df["PFScore"], df["PUScore"] = mf.predict(test), mu.predict(test)
        df["Actual Winner"] = np.where(df.AFScore >= df.AUScore, df.AFTeam, df.AUTeam)
        df["Predicted Winner"] = np.where(df.PFScore >= df.PUScore, df.AFTeam, df.AUTeam)
        df["ESPN"] = np.where(df["Actual Winner"] == df["Predicted Winner"], 10 * 2 ** (df.Round - 1), 0)
        return html.Div([html.H1("Predicting Each Game Independently"), html.H3(f"Total ESPN Score: {df.ESPN.sum():.0f}"), table(df)])
    if page in {"Full Predict", "New Full Prediction", "Bracket Maker", "New Bracket Maker"}:
        title = {"Full Predict":"Linear Regression · Full Bracket Prediction", "New Full Prediction":"NCAA Tournament Prediction Engine · Ensemble", "Bracket Maker":"Bracket Maker", "New Bracket Maker":"New Bracket Maker · Ensemble"}[page]
        try:
            bracket, model, features = simulate_ensemble(
                year,
                variation if page in {"Bracket Maker", "New Bracket Maker"} else 0,
                "progressive" in (progressive or []),
                ensemble=page != "Full Predict",
            )
        except Exception as exc:
            return html.Div([html.H1(title), html.P(str(exc))])
        fields = [c for c in ["Round", "Game", "PFTeam", "PFSeed", "PFScore", "PUTeam", "PUSeed", "PUScore", "PWTeam", "PWSeed", "ESPN_Score"] if c in bracket]
        blocks = [html.H1(title), html.H3(f"Prediction for {year} · ESPN score {int(bracket.ESPN_Score.sum())}"), table(bracket[fields], 64)]
        if page in {"New Full Prediction", "New Bracket Maker"} and not bracket.empty:
            blocks.append(dcc.Graph(figure=px.scatter(bracket, x="PFScore", y="PUScore", color="Round", hover_data=["PFTeam", "PUTeam"], title="Predicted matchup scores")))
        return html.Div(blocks)
    if page == "Update Data":
        return html.Div([html.H1("Update Data"), html.P("The Streamlit updater uses Selenium and remote sources, then rewrites the project CSVs. The Dash copy keeps the source data isolated in its own data folder."), html.P("Data update automation is not connected in this Dash edition yet. Use the original Streamlit app’s Update Data section to refresh its own files.")])
    if page == "Optimize Variation":
        return html.Div([html.H1("Variation Optimization Tool"), html.P("Compare prediction randomness settings against historical tournament outcomes."), html.Button("Run Optimization", id="optimize", n_clicks=0), html.Div(id="optimization-results", style={"marginTop":"18px"})])
    return html.P("Select a section from the navigation.")


@app.callback(Output("optimization-results", "children"), Input("optimize", "n_clicks"), State("runs", "value"), prevent_initial_call=True)
def optimize(n_clicks, runs):
    rows = []
    for variation in range(0, 16):
        scores = []
        for i in range(min(int(runs or 1), 20)):
            for year in [y for y in range(2008, 2026) if y != 2020]:
                try:
                    result, _, _ = simulate_ensemble(year, variation, seed=i + 7)
                    scores.append(result.ESPN_Score.sum())
                except Exception:
                    pass
        rows.append({"Variation": variation, "Average ESPN score": round(float(np.mean(scores)), 1) if scores else 0, "Evaluations": len(scores)})
    df = pd.DataFrame(rows)
    return html.Div([html.H3("Historical Variation Analysis"), table(df, 16), dcc.Graph(figure=px.line(df, x="Variation", y="Average ESPN score", markers=True))])


if __name__ == "__main__":
    app.run(debug=True)
