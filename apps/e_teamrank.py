import streamlit as st
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format

def app():
    # title of the app
    st.markdown('Team Rankings since 2008')
    TR = pd.read_csv('data/B3_KPBPIBR.csv')
    pase = pd.read_csv('data/d3_PASE.csv')
    
    pase.columns =['Team', 'PASE']
    TRP = TR.merge(pase, on='Team', how = 'left')
    TRP.to_csv('data/e3_RankPASE.csv',index=False) 
    
    team = st.text_input("Team: ",'')
    if team != "":
        TRP = TRP[TRP['Team'].str.contains(team)]
    
    TRP.round(decimals=2)
             
    st.dataframe(TRP)