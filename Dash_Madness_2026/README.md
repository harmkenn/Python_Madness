# Python Madness 2026 · Dash edition

This folder contains a separate Dash application and a copy of the CSV data. It does not write to `Python_Madness_2026/data`.

## Run

```powershell
cd Dash_Madness_2026
py -m pip install -r requirements.txt
py app.py
```

Open the local address printed by Dash (normally `http://127.0.0.1:8050`).

The navigation includes the 13 sections from the Streamlit app. Historical tables, per-game predictions, bracket predictions, ensemble simulations, and variation scoring use the copied CSV files. The Streamlit `Update Data` workflow is intentionally not wired into this app: it scrapes external sites and rewrites source data, so refreshes should currently be run from the original Streamlit application.
