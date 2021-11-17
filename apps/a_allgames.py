import streamlit as st
import pandas as pd

def app():
    # title of the app
    st.markdown('All Tournament Games Since 1985')
    team = st.text_input("Team: ",'')
    AG = pd.read_csv('data/B1_FavGames.csv')
    if team != "":
        AG = AG[(AG['AFTeam']==team)|(AG['AUTeam']==team)]
    st.dataframe(AG)
    
     
    