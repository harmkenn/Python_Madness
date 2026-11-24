from .prediction_page import layout as _layout, register_callbacks as _register
TITLE = "Full Predict"
def layout(): return _layout("linear-predict", "Linear Regression · Full Bracket Prediction")
def register_callbacks(app): _register(app, "linear-predict", TITLE, ensemble=False)
