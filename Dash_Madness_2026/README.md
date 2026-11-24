# Python Madness 2026 · Dash edition

This folder contains a separate Dash application and a copy of the CSV data. Each navigation item is implemented as a page module in `subapps/`; `app.py` contains only the shared shell and URL router. The Dash app does not write to `Python_Madness_2026/data`.

## Run

```powershell
cd Dash_Madness_2026
py -m pip install -r requirements.txt
py app.py
```

Open the local address printed by Dash (normally `http://127.0.0.1:8050`).

The navigation includes the 13 sections from the Streamlit app, each implemented in its own module under `subapps/`. Historical tables, predictions, bracket simulations, and variation scoring use the copied CSV files. The Dash `Update Data` page standardizes those local source files and rebuilds derived tables; it does not scrape sites or write into the Streamlit project.
