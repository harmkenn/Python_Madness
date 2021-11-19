import streamlit as st
import pandas as pd

def app():
    # title of the app
    st.markdown('Seed Success History Since 1985')
    AG = pd.read_csv('data/B1_FavGames.csv')
    LG = AG[AG['Round']=='6']
    CS = pd.DataFrame({'Round','Seed'})
    for x in range(0,36):
        if LG.iloc[x]["AFScore"]>LG.iloc[x]["AUScore"]:
            ws = LG.iloc[x]["AFSeed"]
        else:
            ws = LG.iloc[x]["AUSeed"]
        CS.loc[x,'Round'] = 'W'
        CS.loc[x,'Seed'] = ws
    cs = pd.crosstab(index=CS['Seed'],columns=CS['Round'])
    fs = pd.crosstab(index=AG['AFSeed'],columns=AG['Round'])
    us = pd.crosstab(index=AG['AUSeed'],columns=AG['Round'])
    ufs = fs.add(us, fill_value=0).astype(int).drop(['PI'],axis=1)
    df = ufs.add(cs, fill_value=0).fillna(0).astype(int)
    nos = df.max().max()
    df['Exp Wins'] = (df.sum(axis=1)-nos)/nos
    
    df = df.style.set_properties(**{
                    'background-color': 'midnightblue',
                    'font-size': '12pt', 'width': '12px'
                    })
               
    st.dataframe(df,height=5000,width=5000)