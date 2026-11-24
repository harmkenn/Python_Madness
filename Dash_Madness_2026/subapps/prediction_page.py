from dash import Input, Output, dcc, html
import plotly.express as px
from .common import simulate_ensemble, table

def layout(page_id, title, default_year=2025, simulation=False):
    controls = [html.Label("Tournament year"), dcc.Dropdown(list(range(2008, 2027)), default_year, id=f"{page_id}-year", clearable=False)]
    if simulation:
        controls += [html.Label("Randomness (points)"), dcc.Slider(0, 25, 1, value=5, id=f"{page_id}-variation", marks={0:"0",5:"5",10:"10",15:"15",20:"20",25:"25"}),
                     dcc.Checklist([{"label":"Increase randomness each round","value":"progressive"}], [], id=f"{page_id}-progressive")]
    return html.Div([html.H1(title), *controls, html.Div(id=f"{page_id}-output", style={"marginTop":"16px"})])

def register_callbacks(app, page_id, title, ensemble=True, simulation=False, chart=False):
    inputs = [Input(f"{page_id}-year", "value")]
    if simulation:
        inputs += [Input(f"{page_id}-variation", "value"), Input(f"{page_id}-progressive", "value")]
    @app.callback(Output(f"{page_id}-output", "children"), *inputs)
    def predict(year, *options):
        variation, progressive = (options[0] or 0, "progressive" in (options[1] or [])) if simulation else (0, False)
        try:
            bracket, _model, _features = simulate_ensemble(year, variation, progressive, ensemble=ensemble)
        except Exception as exc:
            return html.P(str(exc))
        fields = [c for c in ["Round", "Game", "PFTeam", "PFSeed", "PFScore", "PUTeam", "PUSeed", "PUScore", "PWTeam", "PWSeed", "ESPN_Score"] if c in bracket]
        children = [html.H2(f"Prediction for {year} · ESPN score {int(bracket.ESPN_Score.sum())}"), table(bracket[fields], 64)]
        if chart and not bracket.empty:
            children.append(dcc.Graph(figure=px.scatter(bracket, x="PFScore", y="PUScore", color="Round", hover_data=["PFTeam", "PUTeam"], title="Predicted matchup scores")))
        return html.Div(children)
