"""Dash port of the active local data standardization workflow."""
from pathlib import Path

import numpy as np
import pandas as pd
from dash import Input, Output, dcc, html

from .common import DATA

TITLE = "Update Data"
YEAR = 2027


def _read(name):
    return pd.read_csv(DATA / name)


def _standardize_years(df):
    df = df.copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    return df


def run_data_update():
    """Rebuild the standardized data files from the local source CSVs.

    Scraping calls in the original Streamlit button were commented out. This
    routine therefore uses the source CSVs already present in this Dash copy.
    All outputs are prepared before the CSV files are written.
    """
    repair = _read("step05b_repair.csv").dropna(subset=["tofix", "replacewith"])
    old_names = repair["tofix"].tolist()
    new_names = repair["replacewith"].tolist()
    official_names = set(_read("asn.csv")["Team"].dropna().astype(str))

    bpi = _read("step02b_espnbpi.csv").dropna().copy()
    bpi = _standardize_years(bpi)
    bpi["Team"] = bpi["Team"].replace({"State": "St."}, regex=True).replace(old_names, new_names)
    bpi = bpi[bpi["Team"] != "out"]
    bpi_unmatched = sorted(set(bpi.Team.astype(str).unique()) - official_names)

    # Remove unresolved Git conflict blocks from KenPom before parsing, as in
    # the Streamlit updater, without changing the source file during the run.
    kp_path = DATA / "step01b_kenpom.csv"
    clean_lines, skip = [], False
    for line in kp_path.read_text(encoding="utf-8-sig").splitlines(keepends=True):
        if line.startswith("<<<<<<<"):
            continue
        if line.startswith("======="):
            skip = True
            continue
        if line.startswith(">>>>>>>"):
            skip = False
            continue
        if not skip:
            clean_lines.append(line)
    from io import StringIO
    kp = pd.read_csv(StringIO("".join(clean_lines)))
    kp = kp.loc[:, ~kp.columns.duplicated()]
    kp = kp.dropna(subset=["Team"])
    kp = _standardize_years(kp)
    kp["Team"] = kp["Team"].replace(old_names, new_names)
    kp = kp[~kp["Team"].isin(["out", "Team"])]
    kp_unmatched = sorted(set(kp.Team.astype(str).unique()) - official_names)

    br = _read("step03b_br.csv").dropna(subset=["Team"]).copy()
    br = _standardize_years(br)
    br["Team"] = br["Team"].replace({"State": "St."}, regex=True).replace(old_names, new_names)
    br = br[br["Team"] != "out"]
    br_unmatched = sorted(set(br.Team.astype(str).unique()) - official_names)

    bart = _read("step04b_bart.csv")
    bart = bart.loc[:, ~bart.columns.astype(str).str.contains("^Unnamed")]
    bart = bart.dropna(subset=["Team"])
    bart = _standardize_years(bart)
    bart["Team"] = bart["Team"].replace({"State": "St."}, regex=True).replace(old_names, new_names)
    bart = bart[bart["Team"] != "out"]
    bart_unmatched = sorted(set(bart.Team.astype(str).unique()) - official_names)

    history = _read("step05c_FUHistory.csv").dropna().copy()
    history["AFTeam"] = history["AFTeam"].replace(old_names, new_names)
    history["AUTeam"] = history["AUTeam"].replace(old_names, new_names)
    history = history[(history["AFTeam"] != "out") & (history["AUTeam"] != "out")]
    history_unmatched = sorted((set(history.AFTeam.astype(str)) | set(history.AUTeam.astype(str))) - official_names)

    # Seed performance history uses completed tournaments before this season.
    past = history[history.Year < YEAR]
    champions = past[past.Round == 6]
    champion_seeds = []
    for row in champions.itertuples(index=False):
        seed = row.AFSeed if row.AFScore > row.AUScore else row.AUSeed
        champion_seeds.append({"Round": "W", "Seed": seed})
    champion_table = pd.DataFrame(champion_seeds, columns=["Round", "Seed"])
    champion_counts = pd.crosstab(champion_table["Seed"], champion_table["Round"])
    fav_counts = pd.crosstab(past["AFSeed"], past["Round"])
    dog_counts = pd.crosstab(past["AUSeed"], past["Round"])
    seed_history = fav_counts.add(dog_counts, fill_value=0).fillna(0).astype(int)
    if 0 in seed_history.columns:
        seed_history = seed_history.drop(columns=[0])
    seed_history = seed_history.add(champion_counts, fill_value=0).fillna(0).astype(int)
    max_round_count = seed_history.max().max()
    seed_history["Exp Wins"] = (seed_history.sum(axis=1) - max_round_count) / max_round_count
    seed_history.index.name = "Seed"
    seed_history = seed_history.reset_index()

    # Rebuild PASE (performance against seed expectation) from prior results.
    fw = pd.crosstab(past["AFTeam"], past["Year"])
    uw = pd.crosstab(past["AUTeam"], past["Year"])
    team_wins = fw.add(uw, fill_value=0).fillna(0).astype(int) - 1
    team_wins = team_wins.replace(-1, "")
    first_round = past[past.Round == 1]
    fav_seed = first_round.pivot(index="AFTeam", columns="Year", values="AFSeed").fillna(0).astype(int)
    dog_seed = first_round.pivot(index="AUTeam", columns="Year", values="AUSeed").fillna(0).astype(int)
    seeds = fav_seed.add(dog_seed, fill_value=0).fillna(0).astype(int).replace(0, "")
    expected = seed_history.set_index("Seed")["Exp Wins"].reindex(range(1, 17))
    expected_table = pd.DataFrame({"Seed": range(1, 17), "Exp Wins": expected.to_numpy()})
    expected_seeds = seeds.replace(expected_table.Seed.tolist(), expected_table["Exp Wins"].tolist())
    expected_seeds = expected_seeds.replace(r"^\s*$", np.nan, regex=True).fillna(0)
    team_wins = team_wins.replace(r"^\s*$", np.nan, regex=True).fillna(0)
    pase = team_wins.add(expected_seeds).replace(0, np.nan).mean(axis=1).rename("PASE").reset_index()
    pase = pase.rename(columns={pase.columns[0]: "Team"})
    pase["Team"] = pase["Team"].replace(old_names, new_names)
    pase = pase[pase["Team"] != "out"]
    pase_unmatched = sorted(set(pase.Team.astype(str).unique()) - official_names)

    # Merge the four ratings sources, attach PASE, then join both teams' stats
    # to each historical tournament game.
    all_stats = bpi.merge(kp, how="left", on=["Year", "Team"])
    all_stats = all_stats.merge(br, how="left", on=["Year", "Team"])
    all_stats = all_stats.merge(bart, how="left", on=["Year", "Team"])
    all_stats = all_stats.merge(pase, how="left", on="Team")
    games_with_stats = history[history.Year >= 2008].merge(
        all_stats, left_on=["Year", "AFTeam"], right_on=["Year", "Team"], how="left"
    ).merge(all_stats, left_on=["Year", "AUTeam"], right_on=["Year", "Team"], how="left")
    games_with_stats = games_with_stats.drop(columns=["Team_x", "Team_y"], errors="ignore")

    outputs = {
        "asn.csv": pd.DataFrame({"Team": sorted(official_names)}),
        "step02b_espnbpi.csv": bpi,
        "step01b_kenpom.csv": kp,
        "step03b_br.csv": br,
        "step04b_bart.csv": bart,
        "step05c_FUHistory.csv": history,
        "step05d_SeedHistory.csv": seed_history.drop(columns=["Seed"]),
        "step05e_PASE.csv": pase,
        "step05f_AllStats.csv": all_stats,
        "step05g_FUStats.csv": games_with_stats,
    }
    for filename, frame in outputs.items():
        frame.to_csv(DATA / filename, index=False)

    return {
        "rows": {name: len(frame) for name, frame in outputs.items()},
        "unmatched": {
            "ESPN BPI": bpi_unmatched,
            "KenPom": kp_unmatched,
            "Basketball Reference": br_unmatched,
            "Bart Torvik": bart_unmatched,
            "Tournament history": history_unmatched,
            "PASE": pase_unmatched,
        },
    }


def layout():
    return html.Div([
        html.H1("Update Data"),
        html.P("Standardize the local ratings and rebuild the derived tournament data files for this Dash copy."),
        html.P("The original scraping calls were commented out, so this action uses the source CSVs already in Dash_Madness_2026/data. It does not scrape websites or write to the Streamlit project."),
        html.Button("Run Data Update", id="run-data-update", n_clicks=0),
        dcc.Loading(html.Div(id="data-update-output", style={"marginTop": "18px"})),
    ])


def register_callbacks(app):
    @app.callback(Output("data-update-output", "children"), Input("run-data-update", "n_clicks"), prevent_initial_call=True)
    def update_data(_n_clicks):
        try:
            result = run_data_update()
        except Exception as exc:
            return html.Div([html.H3("Update failed"), html.Pre(str(exc))], style={"color": "#9b1c1c"})
        rows = html.Ul([html.Li(f"{name}: {count:,} rows") for name, count in result["rows"].items()])
        unmatched = []
        for source, teams in result["unmatched"].items():
            if teams:
                unmatched.append(html.Li([html.Strong(f"{source}: "), ", ".join(teams)]))
        report = [html.H3("Data update completed"), html.P("Rebuilt local data files:"), rows]
        if unmatched:
            report += [html.H4("Names outside the official team list"), html.Ul(unmatched)]
        else:
            report.append(html.P("All team names match the official name list."))
        return html.Div(report, style={"color": "#173b24"})
