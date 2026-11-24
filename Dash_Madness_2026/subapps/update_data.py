from dash import html

TITLE = "Update Data"

def layout():
    return html.Div([html.H1("Update Data"),
        html.P("The original updater uses Selenium and remote sources, then rewrites project CSVs."),
        html.P("Data update automation is not connected in this Dash edition yet. Use the original Streamlit app's Update Data section to refresh its own files.")])
