import streamlit as st
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format

def app():
    # title of the app
    st.markdown('Team Rankings since 2008')

    TRP=pd.read_csv('Python_Madness_2024/notebooks/step06_AllStats.csv') 
    
    team = st.text_input("Team: ",'')
    if team != "":
        TRP = TRP[TRP['Team'].str.contains(team)]
    
    TRP.round(decimals=2)
    TRP['Year'] = TRP['Year'].astype('str')         
    st.dataframe(TRP)