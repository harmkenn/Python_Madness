import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def app():
    # title of the app
    st.markdown('All Tournament Brackets Since 1985')
    AG = pd.read_csv('notebooks/step04_FUHistory.csv')
    p_year = st.slider('Year: ', 1985,2022)
    if p_year == 2020:
        st.markdown("No Bracket in 2020")
    if p_year != 2020:
        st.markdown(f'Bracket for {p_year}' )
        AGP = AG[AG['Year']==p_year][AG['Game']>=1]
        AGPF = AGP.drop(['AUSeed','AUTeam','AUScore','Uti'],axis=1)
        AGPU = AGP.drop(['AFSeed','AFTeam','AFScore','Fti'],axis=1)
        AGPF['STS'] = AGPF['AFSeed'].astype(str)+' '+AGPF['AFTeam']+' '+AGPF['AFScore'].astype(str)
        AGPF = AGPF.drop(['AFSeed','AFTeam','AFScore'],axis=1).rename(columns={"Fti": "ti"})
        AGPU['STS'] = AGPU['AUSeed'].astype(str)+' '+AGPU['AUTeam']+' '+AGPU['AUScore'].astype(str)
        AGPU = AGPU.drop(['AUSeed','AUTeam','AUScore'],axis=1).rename(columns={"Uti": "ti"})
        AGC = pd.concat([AGPF,AGPU]).sort_values(by=['Game','ti'])
        AGC.reset_index(inplace=True)
        br_headers = ['Round 1 W','Round 2 W','Round 3 W','Round 4 W','Final Four W','Champ',
              'Final Four E','Round 4 E','Round 3 E','Round 2 E','Round 1 E']
        df = pd.DataFrame(columns = br_headers)
        for x in br_headers:
            for y in range(32):
                df.loc[y,x]=''
        for y in range(32):
            df.loc[y,'Round 1 W']= AGC.loc[y,'STS']
        for y in range(0,31,2):
            df.loc[y,'Round 2 W']= AGC.loc[y/2+64,'STS']
        for y in range(0,31,4):
            df.loc[y+1,'Round 3 W']= AGC.loc[y/4+96,'STS']
        for y in range(0,31,8):
            df.loc[y+3,'Round 4 W']= AGC.loc[y/8+112,'STS']
        for y in range(0,31,16):
            df.loc[y+7,'Final Four W']= AGC.loc[y/16+120,'STS']
        for y in range(32):
            df.loc[y,'Round 1 E']= AGC.loc[y+32,'STS']
        for y in range(0,31,2):
            df.loc[y,'Round 2 E']= AGC.loc[y/2+80,'STS']
        for y in range(0,31,4):
            df.loc[y+1,'Round 3 E']= AGC.loc[y/4+104,'STS']
        for y in range(0,31,8):
            df.loc[y+3,'Round 4 E']= AGC.loc[y/8+116,'STS']
        for y in range(0,31,16):
            df.loc[y+7,'Final Four E']= AGC.loc[y/16+122,'STS']
        df.loc[15,'Champ']= AGC.loc[124,'STS']
        df.loc[16,'Champ']= AGC.loc[125,'STS']
        Champ = AGP.iloc[-1,]
        if Champ['AFScore']>Champ['AUScore']:
            winner = Champ['AFTeam']
        else:
            winner = Champ['AUTeam']
        df.loc[0,'Champ'] = winner
        
               
        st.dataframe(df,height=5000,width=5000)
        

        
        
        