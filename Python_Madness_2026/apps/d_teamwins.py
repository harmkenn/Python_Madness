import streamlit as st
import pandas as pd


# title of the app
st.markdown('Team Wins Since 2008')
AG = pd.read_csv('Python_Madness_2026/data/step05c_FUHistory.csv')
py = 2026
AG = AG[AG['Year']<py]
champs = AG[AG['Round']== 6].copy()
champs.index = champs["Year"]

for year in champs['Year']:
    if champs.loc[year,'AFScore'] > champs.loc[year,'AUScore']:
        champs.loc[year,6] = champs.loc[year,'AFTeam']
    else:
        champs.loc[year,6] = champs.loc[year,'AUTeam']

# table of wins
fw = pd.crosstab(index=AG['AFTeam'],columns=AG['Year'])
uw = pd.crosstab(index=AG['AUTeam'],columns=AG['Year'])
chw = pd.crosstab(index=champs[6],columns=champs['Year'])

cw = fw.add(uw, fill_value=0).fillna(0).astype(int) - 1
cw = cw.add(chw, fill_value=0).fillna(0).astype(int)
cw = cw.replace(-1, '').astype(str)

team = st.text_input("Team: ",'')
if team != "":
    cw = cw[cw.index.str.contains(team)]

        
st.dataframe(cw, height=600, width='content')
