import streamlit as st
import pandas as pd

def app():
    # title of the app
    st.markdown('How the teams have faired since 1985')
    AG = pd.read_csv('data/B1_FavGames.csv')
    
    # table of wins
    fw = pd.crosstab(index=AG['AFTeam'],columns=AG['Year'])
    uw = pd.crosstab(index=AG['AUTeam'],columns=AG['Year'])
    cw = fw.add(uw, fill_value=0).fillna(0).astype(int) - 1
    cw = cw.replace(-1, '').astype(str)
    
    team = st.text_input("Team: ",'')
    if team != "":
        cw = cw[cw.index.str.contains(team)]
    
    cw = cw.style.set_properties(**{
                    'background-color': 'midnightblue',
                    'font-size': '8pt', 'width': '12px'
                    })
             
    st.dataframe(cw)

    
