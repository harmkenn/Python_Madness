from .prediction_page import layout as _layout, register_callbacks as _register
TITLE = "New Bracket Maker"
def layout(): return _layout("new-bracket-maker", "New Bracket Maker · Ensemble", simulation=True, default_year=2025)
def register_callbacks(app): _register(app, "new-bracket-maker", TITLE, ensemble=True, simulation=True, chart=True)
