
  
import streamlit as st
import pandas as pd
import numpy as np

def app():
    # title of the app
    st.markdown('How the teams have faired since 1985')
    AG = pd.read_csv('data/B1_FavGames.csv')
    AG = AG[AG['Year']<=2021]
    # table of wins
    fw = pd.crosstab(index=AG['AFTeam'],columns=AG['Year'])
    uw = pd.crosstab(index=AG['AUTeam'],columns=AG['Year'])
    cw = fw.add(uw, fill_value=0).fillna(0).astype(int) - 1
    cw = cw.replace(-1, '')
    
    # table of seeds
    AG1 = AG[AG['Round']=='1']
    fs = AG1.pivot(index='AFTeam', columns='Year', values = 'AFSeed').fillna(0).astype(int)
    us = AG1.pivot(index='AUTeam', columns='Year', values = 'AUSeed').fillna(0).astype(int)
    cs = fs.add(us, fill_value=0).fillna(0).astype(int)
    cs = cs.replace(0, '')

    # table of expected wins
    sh = pd.read_csv('data/c3_SeedHistory.csv')
    ew = pd.DataFrame({'Seed':np.arange(1,17),'Exp Wins':sh['Exp Wins']})

    ewty = cs.replace(list(ew['Seed']),list(ew['Exp Wins']))

    ewz = (ewty * -1).replace(r'^\s*$', np.nan, regex=True).fillna(0)
    cwz = cw.replace(r'^\s*$', np.nan, regex=True).fillna(0)
    pase = cwz.add(ewz).replace(0, np.nan)
    pase['PASE']=pase.mean(axis=1)
    pase = pase.iloc[:,-1:]
    
    pase.to_csv('data/d3_PASE.csv')   
    
    team = st.text_input("Team: ",'')
    if team != "":
        pase = pase[pase.index.str.contains(team)]
    

             
    st.dataframe(pase)

    

  