from dash import html
from .common import csv, table

TITLE = "Team Performance"

def layout():
    return html.Div([html.H1("Team Performance Against Seed Expectation Since 2008"), table(csv("step05e_PASE.csv"))])
