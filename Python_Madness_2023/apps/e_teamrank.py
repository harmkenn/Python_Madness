import streamlit as st
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format

def app():
    # title of the app
    st.markdown('Team Rankings since 2008')

    TRP=pd.read_csv('notebooks/step04_AllStats.csv') 
    
    team = st.text_input("Team: ",'')
    if team != "":
        TRP = TRP[TRP['Team'].str.contains(team)]
    
    TRP.round(decimals=2)
             
    st.dataframe(TRP)