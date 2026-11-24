from pathlib import Path
import re
import numpy as np
import pandas as pd
import plotly.express as px
from dash import dash_table, dcc, html
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

def csv(name):
    return pd.read_csv(DATA / name).fillna(0)


def table(df, page_size=25, height="620px", sort_by=None):
    if df is None or df.empty:
        return html.P("No rows match the selected options.")
    shown = df.copy()
    shown.columns = [str(c) for c in shown.columns]
    for col in shown.columns:
        dtype = shown[col].dtype
        if pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.StringDtype):
            shown[col] = shown[col].astype(str)
    column_widths = []
    for col in shown.columns:
        longest = max(len(col), int(shown[col].astype(str).str.len().max() or 0))
        width = max(70, longest * 8 + 24)
        column_widths.append({
            "if": {"column_id": col},
            "minWidth": f"{width}px",
            "width": f"{width}px",
            "maxWidth": f"{width}px",
        })
    return dash_table.DataTable(
        data=shown.to_dict("records"), columns=[{"name": c, "id": c} for c in shown.columns],
        page_size=page_size, sort_action="native", sort_by=sort_by or [], filter_action="native",
        style_table={"overflowX": "auto", "maxHeight": height, "overflowY": "auto"},
        style_cell={"fontFamily": "Arial", "fontSize": 13, "padding": "6px", "textAlign": "left", "whiteSpace": "nowrap"},
        style_cell_conditional=column_widths,
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

