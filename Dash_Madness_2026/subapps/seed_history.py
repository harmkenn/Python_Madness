import numpy as np
from dash import html
from .common import csv, table

TITLE = "Seed History"

def layout():
    df = csv("step05d_SeedHistory.csv")
    df.insert(0, "Seed", np.arange(1, len(df) + 1))
    return html.Div([html.H1("Seed Success History Since 2008"), table(df, 16)])
