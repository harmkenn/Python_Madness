import streamlit as st
import pandas as pd

def app():
    # title of the app
    st.markdown('All Tournament Games Since 1985')
    team = st.text_input("Team: ",'')
    AG = pd.read_csv('data/B1_FavGames.csv')
    AG = AG.drop(['Fti','Uti'],axis=1)
    if team != "":
        AG = AG[(AG['AFTeam'].str.contains(team))|(AG['AUTeam'].str.contains(team))]
    st.dataframe(AG,height= 4000)
    
    
     
    