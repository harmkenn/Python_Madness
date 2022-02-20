import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def app():
    # title of the app
    py = 2022
    st.markdown('Predicting ' + str(py))
        
    fup = pd.read_csv("data/B3_AllFandU.csv").fillna(0)
    fup = fup[fup['Game']>=1]
    fup['Round'] = fup['Round'].astype('int32')
    fup['PFSeed']=fup['AFSeed']
    fup['PFTeam']=fup['AFTeam']
    fup['PFScore']=fup['AFScore']
    fup['PUSeed']=fup['AUSeed']
    fup['PUTeam']=fup['AUTeam']
    fup['PUScore']=fup['AUScore']
    fup = fup.drop(['AFSeed','AFTeam','AFScore','AUSeed','AUTeam','AUScore','Fti','Uti'],axis=1)
    
    
    # Build the linear model
    fupn = fup.select_dtypes(exclude=['object'])
    MX = fupn[fupn['Year']<=py].drop(['PFScore','PUScore'],axis=1)
    xcol = MX.columns
    MFY = fupn[fupn['Year']<=py]['PFScore']
    MUY = fupn[fupn['Year']<=py]['PUScore']
    LRF = LinearRegression()
    LRF.fit(MX,MFY)
    RFU = LinearRegression()
    RFU.fit(MX,MUY)
    
    BB = pd.read_csv('data/B1_FavGames.csv')
    BB = BB[BB['Year']==py][BB['Game']>=1][BB['Game']<=32]
    BB['Round']=BB['Round'].astype('int32')
    BB.index = BB.Game
    BB = BB.iloc[:,0:10]
    BBcol = ['Year','Round','Region','Game','PFSeed','PFTeam','PFScore','PUSeed','PUTeam','PUScore']
    BB.columns = BBcol

    KBBP = pd.read_csv("data/B3_KPBPIBRP.csv").fillna(0)
    # Predict Round 1
    BBstats = BB.merge(KBBP, left_on=['Year','PFTeam'],right_on=['Year','Team'],how='left')
    BBstats = BBstats.merge(KBBP, left_on=['Year','PUTeam'],right_on=['Year','Team'],how='left')
        
    r1p = BBstats
    
    pfs = LRF.predict(r1p[xcol])
    pus = RFU.predict(r1p[xcol])
    
    for x in range(1,33):
        BB.loc[x,'PFScore']=pfs[x-1]
        BB.loc[x,'PUScore']=pus[x-1]
        BB.loc[x,'PWSeed'] = np.where(BB.loc[x,'PFScore']>=BB.loc[x,'PUScore'],BB.loc[x,'PFSeed'],BB.loc[x,'PUSeed'])
        BB.loc[x,'PWTeam'] = str(np.where(BB.loc[x,'PFScore']>=BB.loc[x,'PUScore'],BB.loc[x,'PFTeam'],BB.loc[x,'PUTeam']))
    
    for x in range(33,49):
        BB.loc[x,'Year'] = py
        BB.loc[x,'Round'] = 2
        BB.loc[x,'Region'] = BB.loc[(x-32)*2,'Region']
        BB.loc[x,'Game'] = x
        BB.loc[x,'PFSeed'] = np.where(BB.loc[(x-32)*2-1,'PWSeed']<BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWSeed'],BB.loc[(x-32)*2,'PWSeed'])
        BB.loc[x,'PUSeed'] = np.where(BB.loc[(x-32)*2-1,'PWSeed']>BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWSeed'],BB.loc[(x-32)*2,'PWSeed'])
        BB.loc[x,'PFTeam'] = str(np.where(BB.loc[(x-32)*2-1,'PWSeed']<BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWTeam'],BB.loc[(x-32)*2,'PWTeam']))
        BB.loc[x,'PUTeam'] = str(np.where(BB.loc[(x-32)*2-1,'PWSeed']>BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWTeam'],BB.loc[(x-32)*2,'PWTeam']))
       
    BBstats = BB[BB['Round']==2].merge(KBBP, left_on=['Year','PFTeam'],right_on=['Year','Team'],how='left')
    BBstats = BBstats.merge(KBBP, left_on=['Year','PUTeam'],right_on=['Year','Team'],how='left')
    
    pfs = LRF.predict(BBstats[xcol])
    pus = RFU.predict(BBstats[xcol])
    for x in range(33,49):
        BB.loc[x,'PFScore']=pfs[x-33]
        BB.loc[x,'PUScore']=pus[x-33]
        BB.loc[x,'PWSeed'] = np.where(BB.loc[x,'PFScore']>=BB.loc[x,'PUScore'],BB.loc[x,'PFSeed'],BB.loc[x,'PUSeed'])
        BB.loc[x,'PWTeam'] = str(np.where(BB.loc[x,'PFScore']>=BB.loc[x,'PUScore'],BB.loc[x,'PFTeam'],BB.loc[x,'PUTeam']))
        
    for x in range(49,57):
        BB.loc[x,'Year'] = py
        BB.loc[x,'Round'] = 3
        BB.loc[x,'Region'] = BB.loc[(x-32)*2,'Region']
        BB.loc[x,'Game'] = x
        BB.loc[x,'PFSeed'] = np.where(BB.loc[(x-32)*2-1,'PWSeed']<BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWSeed'],BB.loc[(x-32)*2,'PWSeed'])
        BB.loc[x,'PUSeed'] = np.where(BB.loc[(x-32)*2-1,'PWSeed']>BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWSeed'],BB.loc[(x-32)*2,'PWSeed'])
        BB.loc[x,'PFTeam'] = str(np.where(BB.loc[(x-32)*2-1,'PWSeed']<BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWTeam'],BB.loc[(x-32)*2,'PWTeam']))
        BB.loc[x,'PUTeam'] = str(np.where(BB.loc[(x-32)*2-1,'PWSeed']>BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWTeam'],BB.loc[(x-32)*2,'PWTeam']))
       
    BBstats = BB[BB['Round']==3].merge(KBBP, left_on=['Year','PFTeam'],right_on=['Year','Team'],how='left')
    BBstats = BBstats.merge(KBBP, left_on=['Year','PUTeam'],right_on=['Year','Team'],how='left')
    
    pfs = LRF.predict(BBstats[xcol])
    pus = RFU.predict(BBstats[xcol])
    for x in range(49,57):
        BB.loc[x,'PFScore']=pfs[x-49]
        BB.loc[x,'PUScore']=pus[x-49]
        BB.loc[x,'PWSeed'] = np.where(BB.loc[x,'PFScore']>=BB.loc[x,'PUScore'],BB.loc[x,'PFSeed'],BB.loc[x,'PUSeed'])
        BB.loc[x,'PWTeam'] = str(np.where(BB.loc[x,'PFScore']>=BB.loc[x,'PUScore'],BB.loc[x,'PFTeam'],BB.loc[x,'PUTeam']))
        
    for x in range(57,61):
        BB.loc[x,'Year'] = py
        BB.loc[x,'Round'] = 4
        BB.loc[x,'Region'] = BB.loc[(x-32)*2,'Region']
        BB.loc[x,'Game'] = x
        BB.loc[x,'PFSeed'] = np.where(BB.loc[(x-32)*2-1,'PWSeed']<BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWSeed'],BB.loc[(x-32)*2,'PWSeed'])
        BB.loc[x,'PUSeed'] = np.where(BB.loc[(x-32)*2-1,'PWSeed']>BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWSeed'],BB.loc[(x-32)*2,'PWSeed'])
        BB.loc[x,'PFTeam'] = str(np.where(BB.loc[(x-32)*2-1,'PWSeed']<BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWTeam'],BB.loc[(x-32)*2,'PWTeam']))
        BB.loc[x,'PUTeam'] = str(np.where(BB.loc[(x-32)*2-1,'PWSeed']>BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWTeam'],BB.loc[(x-32)*2,'PWTeam']))
       
    BBstats = BB[BB['Round']==4].merge(KBBP, left_on=['Year','PFTeam'],right_on=['Year','Team'],how='left')
    BBstats = BBstats.merge(KBBP, left_on=['Year','PUTeam'],right_on=['Year','Team'],how='left')
    
    pfs = LRF.predict(BBstats[xcol])
    pus = RFU.predict(BBstats[xcol])
    for x in range(57,61):
        BB.loc[x,'PFScore']=pfs[x-57]
        BB.loc[x,'PUScore']=pus[x-57]
        BB.loc[x,'PWSeed'] = np.where(BB.loc[x,'PFScore']>=BB.loc[x,'PUScore'],BB.loc[x,'PFSeed'],BB.loc[x,'PUSeed'])
        BB.loc[x,'PWTeam'] = str(np.where(BB.loc[x,'PFScore']>=BB.loc[x,'PUScore'],BB.loc[x,'PFTeam'],BB.loc[x,'PUTeam']))
        
    for x in range(61,63):
        BB.loc[x,'Year'] = py
        BB.loc[x,'Round'] = 5
        BB.loc[x,'Region'] = BB.loc[(x-32)*2,'Region']
        BB.loc[x,'Game'] = x
        BB.loc[x,'PFSeed'] = np.where(BB.loc[(x-32)*2-1,'PWSeed']<BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWSeed'],BB.loc[(x-32)*2,'PWSeed'])
        BB.loc[x,'PUSeed'] = np.where(BB.loc[(x-32)*2-1,'PWSeed']>BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWSeed'],BB.loc[(x-32)*2,'PWSeed'])
        BB.loc[x,'PFTeam'] = str(np.where(BB.loc[(x-32)*2-1,'PWSeed']<BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWTeam'],BB.loc[(x-32)*2,'PWTeam']))
        BB.loc[x,'PUTeam'] = str(np.where(BB.loc[(x-32)*2-1,'PWSeed']>BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWTeam'],BB.loc[(x-32)*2,'PWTeam']))
       
    BBstats = BB[BB['Round']==5].merge(KBBP, left_on=['Year','PFTeam'],right_on=['Year','Team'],how='left')
    BBstats = BBstats.merge(KBBP, left_on=['Year','PUTeam'],right_on=['Year','Team'],how='left')
    
    pfs = LRF.predict(BBstats[xcol])
    pus = RFU.predict(BBstats[xcol])
    for x in range(61,63):
        BB.loc[x,'PFScore']=pfs[x-61]
        BB.loc[x,'PUScore']=pus[x-61]
        BB.loc[x,'PWSeed'] = np.where(BB.loc[x,'PFScore']>=BB.loc[x,'PUScore'],BB.loc[x,'PFSeed'],BB.loc[x,'PUSeed'])
        BB.loc[x,'PWTeam'] = str(np.where(BB.loc[x,'PFScore']>=BB.loc[x,'PUScore'],BB.loc[x,'PFTeam'],BB.loc[x,'PUTeam']))
        
    for x in range(63,64):
        BB.loc[x,'Year'] = py
        BB.loc[x,'Round'] = 6
        BB.loc[x,'Region'] = BB.loc[(x-32)*2,'Region']
        BB.loc[x,'Game'] = x
        BB.loc[x,'PFSeed'] = np.where(BB.loc[(x-32)*2-1,'PWSeed']<BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWSeed'],BB.loc[(x-32)*2,'PWSeed'])
        BB.loc[x,'PUSeed'] = np.where(BB.loc[(x-32)*2-1,'PWSeed']>BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWSeed'],BB.loc[(x-32)*2,'PWSeed'])
        BB.loc[x,'PFTeam'] = str(np.where(BB.loc[(x-32)*2-1,'PWSeed']<BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWTeam'],BB.loc[(x-32)*2,'PWTeam']))
        BB.loc[x,'PUTeam'] = str(np.where(BB.loc[(x-32)*2-1,'PWSeed']>BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWTeam'],BB.loc[(x-32)*2,'PWTeam']))
       
    BBstats = BB[BB['Round']==6].merge(KBBP, left_on=['Year','PFTeam'],right_on=['Year','Team'],how='left')
    BBstats = BBstats.merge(KBBP, left_on=['Year','PUTeam'],right_on=['Year','Team'],how='left')
    
    pfs = LRF.predict(BBstats[xcol])
    pus = RFU.predict(BBstats[xcol])
    for x in range(63,64):
        BB.loc[x,'PFScore']=pfs[x-63]
        BB.loc[x,'PUScore']=pus[x-63]
        BB.loc[x,'PWSeed'] = np.where(BB.loc[x,'PFScore']>=BB.loc[x,'PUScore'],BB.loc[x,'PFSeed'],BB.loc[x,'PUSeed'])
        BB.loc[x,'PWTeam'] = str(np.where(BB.loc[x,'PFScore']>=BB.loc[x,'PUScore'],BB.loc[x,'PFTeam'],BB.loc[x,'PUTeam']))
        
        BB['Year'] = BB['Year'].astype(int)
        BB['Round'] = BB['Round'].astype(int)
        BB['Game'] = BB['Game'].astype(int)
        BB['PFSeed'] = BB['PFSeed'].astype(int)
        BB['PUSeed'] = BB['PUSeed'].astype(int)
        BB['PWSeed'] = BB['PWSeed'].astype(int)
       
    st.dataframe(BB[BB['Game']<=63],height=500)
    