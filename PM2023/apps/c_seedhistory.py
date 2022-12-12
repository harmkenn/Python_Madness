import streamlit as st
import pandas as pd

def app():
    # title of the app
    st.markdown('Seed Success History Since 1985')
    df = pd.read_csv('notebooks/step04_SeedHistory.csv')
          
    st.dataframe(df,height=5000,width=5000)