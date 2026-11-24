from dash import Input, Output, dcc, html
from .common import csv

TITLE = "All Brackets"

REGIONS = [
    ("NW", "East Region", 1),
    ("SW", "South Region", 9),
    ("NE", "West Region", 17),
    ("SE", "Midwest Region", 25),
]
ROUND_NAMES = {1: "Round of 64", 2: "Round of 32", 3: "Sweet 16", 4: "Elite Eight"}

def layout():
    return html.Div([
        html.H1("Tournament Bracket"),
        html.Label("Tournament year"),
        dcc.Dropdown(list(range(2008, 2027)), 2026, id="brackets-year", clearable=False,
                     style={"maxWidth": "240px"}),
        html.Div(id="brackets-output", style={"marginTop": "20px"}),
    ])


def team_line(seed, team, score, won):
    return html.Div([
        html.Span(str(int(seed)) if seed else "", style={"color": "#64748b", "textAlign": "center"}),
        html.Span(str(team)),
        html.Span(str(int(score)) if score != "" else "", style={"textAlign": "right", "fontVariantNumeric": "tabular-nums"}),
    ], style={
        "display": "grid", "gridTemplateColumns": "18px minmax(0, 1fr) 24px", "gap": "4px",
        "alignItems": "center", "padding": "4px 6px", "fontSize": "11px",
        "fontWeight": "700" if won else "400",
        "background": "#e8f5e9" if won else "#fff",
        "color": "#173b24" if won else "#263445",
        "whiteSpace": "nowrap",
    })


def game_card(game):
    af_won = float(game.AFScore) > float(game.AUScore)
    au_won = float(game.AUScore) > float(game.AFScore)
    return html.Div([
        team_line(game.AFSeed, game.AFTeam, game.AFScore, af_won),
        html.Div(style={"height": "1px", "background": "#d9e0e8"}),
        team_line(game.AUSeed, game.AUTeam, game.AUScore, au_won),
    ], style={
        "border": "1px solid #cbd5e1", "borderRadius": "5px", "overflow": "hidden",
        "background": "white", "boxShadow": "0 1px 2px #0f172a12",
    })


def round_column(label, games):
    return html.Div([
        html.Div(label, style={"fontSize": "11px", "fontWeight": "700", "color": "#475569", "height": "26px", "display": "flex", "alignItems": "center"}),
        html.Div([game_card(g) for g in games], style={
            "height": "560px", "minWidth": "155px", "display": "flex", "flexDirection": "column",
            "justifyContent": "space-around", "gap": "5px",
        }),
    ], style={"flex": "0 0 155px"})


def region_bracket(df, region_code, region_name, first_game):
    by_game = {int(row.Game): row for row in df.itertuples(index=False)}
    region_index = [code for code, _, _ in REGIONS].index(region_code)
    rounds = [
        (1, range(first_game, first_game + 8)),
        (2, range(33 + region_index * 4, 37 + region_index * 4)),
        (3, range(49 + region_index * 2, 51 + region_index * 2)),
        (4, [57 + region_index]),
    ]
    columns = []
    for round_no, game_ids in rounds:
        games = [by_game[g] for g in game_ids if g in by_game]
        columns.append(round_column(ROUND_NAMES[round_no], games))
    return html.Section([
        html.H2(region_name, style={"fontSize": "16px", "margin": "0 0 6px", "color": "#15243a"}),
        html.Div(columns, style={"display": "flex", "gap": "9px", "overflowX": "auto", "paddingBottom": "6px"}),
    ], style={"marginBottom": "16px", "padding": "10px", "background": "#f5f7fa", "borderRadius": "8px", "border": "1px solid #e2e8f0"})


def national_bracket(df):
    by_game = {int(row.Game): row for row in df.itertuples(index=False)}
    semifinal_ids = [61, 62]
    semis = [game_card(by_game[g]) for g in semifinal_ids if g in by_game]
    final = game_card(by_game[63]) if 63 in by_game else html.Div("Championship not played", style={"padding": "14px"})
    return html.Section([
        html.H2("Final Four and Championship", style={"fontSize": "17px", "margin": "0 0 10px"}),
        html.Div([
            html.Div([html.Div("Semifinal 1", style={"fontWeight": "700", "fontSize": "12px", "marginBottom": "5px"}), semis[0] if len(semis) > 0 else html.Div("Not played")], style={"width": "190px"}),
            html.Div([html.Div("Semifinal 2", style={"fontWeight": "700", "fontSize": "12px", "marginBottom": "5px"}), semis[1] if len(semis) > 1 else html.Div("Not played")], style={"width": "190px"}),
            html.Div([html.Div("National Championship", style={"fontWeight": "700", "fontSize": "12px", "marginBottom": "5px"}), final], style={"width": "205px", "alignSelf": "center"}),
        ], style={"display": "flex", "gap": "14px", "alignItems": "center", "overflowX": "auto"}),
    ], style={"padding": "10px", "background": "#eef3fb", "borderRadius": "8px", "border": "1px solid #d7e2f1"})


def register_callbacks(app):
    @app.callback(Output("brackets-output", "children"), Input("brackets-year", "value"))
    def show_bracket(year):
        if year == 2020:
            return html.P("No tournament was held in 2020.")
        df = csv("step05c_FUHistory.csv")
        df = df[(df.Year == year) & (df.Game >= 1)]
        if df.empty:
            return html.P(f"No bracket data is available for {year}.")
        children = [html.H2(str(year), style={"marginTop": "0"})]
        children.extend(region_bracket(df, *region) for region in REGIONS)
        children.append(national_bracket(df))
        return html.Div(children)
