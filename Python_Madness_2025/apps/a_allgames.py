import streamlit as st
import pandas as pd


# title of the app
st.markdown('All Tournament Games Since 2008')
team = st.text_input("Team: ",'')
py=2025
AG = pd.read_csv('Python_Madness_2025/data/step05c_FUHistory.csv')
AG['Year'] = pd.to_numeric(AG['Year'], errors='coerce').astype('Int64')

AG = AG.drop(['Fti','Uti'],axis=1)
AG = AG[AG['Year'] < py]
AG['Year'] = AG['Year'].astype('str')

if team != "":
    AG = AG[(AG['AFTeam'].str.contains(team))|(AG['AUTeam'].str.contains(team))]
    
    
    
st.dataframe(AG,height= 500)



    
