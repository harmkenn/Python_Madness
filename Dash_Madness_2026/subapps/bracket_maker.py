from .prediction_page import layout as _layout, register_callbacks as _register
TITLE = "Bracket Maker"
def layout(): return _layout("bracket-maker", "Bracket Maker", simulation=True)
def register_callbacks(app): _register(app, "bracket-maker", TITLE, ensemble=False, simulation=True)
