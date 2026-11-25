from .prediction_page import layout as _layout, register_callbacks as _register
TITLE = "Full Predict"
def layout(): return _layout("ridge-predict", "Ridge Regression · Full Bracket Prediction", ridge=True)
def register_callbacks(app): _register(app, "ridge-predict", TITLE, ensemble=False, ridge=True)
