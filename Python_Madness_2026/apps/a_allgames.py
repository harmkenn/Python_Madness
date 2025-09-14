import streamlit as st
import polars as pl
from st_aggrid import AgGrid, GridOptionsBuilder

# Title of the app
st.markdown('All Tournament Games Since 2008')
team = st.text_input("Team:", '')
py = 2026

# Read CSV with Polars
AG = pl.read_csv('Python_Madness_2026/data/step05c_FUHistory.csv')

# Make Year numeric
AG = AG.with_columns(
    pl.col("Year").cast(pl.Int64, strict=False)
)

# Drop unused columns
AG = AG.drop(["Fti", "Uti"])

# Filter by year
AG = AG.filter(pl.col("Year") < py)

# Cast Year to string
AG = AG.with_columns(
    pl.col("Year").cast(pl.Utf8)
)

# filter by team if provided
if team != "":
    pattern = f"(?i){team}"  # (?i) makes regex case-insensitive
    AG = AG.filter(
        (pl.col("AFTeam").str.contains(pattern)) |
        (pl.col("AUTeam").str.contains(pattern))
    )

# Convert to pandas for AgGrid
df = AG.to_pandas()

# Configure AgGrid options
gb = GridOptionsBuilder.from_dataframe(df)
gb.configure_default_column(
    filter=True, sortable=True, resizable=True, editable=False
)
gb.configure_grid_options(domLayout='normal')
gridOptions = gb.build()

# Display AgGrid with Excel-like filters
AgGrid(
    df,
    gridOptions=gridOptions,
    height=600,
    width="100%",
    enable_enterprise_modules=False,
    fit_columns_on_grid_load=False
)
