from .prediction_page import layout as _layout, register_callbacks as _register
TITLE = "New Full Prediction"
def layout(): return _layout("ensemble-predict", "NCAA Tournament Prediction Engine · Ensemble", chart=True)
def register_callbacks(app): _register(app, "ensemble-predict", TITLE, ensemble=True, chart=True)
