import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
import numpy as np

def app():
    # title of the app
    st.markdown('Use Linear Regression to do a Full Bracket Prediction')
    
    fup = pd.read_csv("notebooks/step04_FUStats.csv").fillna(0)
    fup = fup[fup['Year']<=2022][fup['Game']>=1]
    fup['Round'] = fup['Round'].astype('int32')
    fup['PFSeed']=fup['AFSeed']
    fup['PFTeam']=fup['AFTeam']
    fup['PFScore']=fup['AFScore']
    fup['PUSeed']=fup['AUSeed']
    fup['PUTeam']=fup['AUTeam']
    fup['PUScore']=fup['AUScore']
    
     
    py = st.slider('Year: ', 2008,2022)
    if py == 2020:
        st.markdown("No Bracket in 2020")
    if py != 2020:
        # Build the linear model
        fupn = fup.select_dtypes(exclude=['object'])
        MX = fupn[fupn['Year']!=py].drop(['AFScore','AUScore','AFSeed','AUSeed','PFScore','PUScore','Fti','Uti'],axis=1)
        xcol = MX.columns
        
        MFY = fupn[fupn['Year']!=py]['PFScore']
        MUY = fupn[fupn['Year']!=py]['PUScore']
        LRF = LinearRegression()
        LRF.fit(MX,MFY)
        RFU = LinearRegression()
        RFU.fit(MX,MUY)
        
        BB = fup[fup['Year']==py]
        BB = BB.iloc[:,0:10]
        BB.index = BB.Game
        
        # Predict Round 1
        
        r1p = fup[fup['Year']==py][fup['Round']==1]
        
        pfs = LRF.predict(r1p[xcol])
        pus = RFU.predict(r1p[xcol])
        
        for x in range(1,33):
            BB.loc[x,'PFSeed']=BB.loc[x,'AFSeed']
            BB.loc[x,'PFTeam']=BB.loc[x,'AFTeam']
            BB.loc[x,'PFScore']=pfs[x-1]
            BB.loc[x,'PUSeed']=BB.loc[x,'AUSeed']
            BB.loc[x,'PUTeam']=BB.loc[x,'AUTeam']
            BB.loc[x,'PUScore']=pus[x-1]
            BB.loc[x,'AWSeed'] = np.where(BB.loc[x,'AFScore']>=BB.loc[x,'AUScore'],BB.loc[x,'AFSeed'],BB.loc[x,'AUSeed'])
            BB.loc[x,'AWTeam'] = str(np.where(BB.loc[x,'AFScore']>=BB.loc[x,'AUScore'],BB.loc[x,'AFTeam'],BB.loc[x,'AUTeam']))
            BB.loc[x,'PWSeed'] = np.where(BB.loc[x,'PFScore']>=BB.loc[x,'PUScore'],BB.loc[x,'PFSeed'],BB.loc[x,'PUSeed'])
            BB.loc[x,'PWTeam'] = str(np.where(BB.loc[x,'PFScore']>=BB.loc[x,'PUScore'],BB.loc[x,'PFTeam'],BB.loc[x,'PUTeam']))
            BB.loc[x,'ESPN'] = np.where(BB.loc[x,'AWTeam']==BB.loc[x,'PWTeam'],10,0)
            
        for x in range(33,49):
            BB.loc[x,'PFSeed'] = np.where(BB.loc[(x-32)*2-1,'PWSeed']<BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWSeed'],BB.loc[(x-32)*2,'PWSeed'])
            BB.loc[x,'PUSeed'] = np.where(BB.loc[(x-32)*2-1,'PWSeed']>BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWSeed'],BB.loc[(x-32)*2,'PWSeed'])
            BB.loc[x,'PFTeam'] = str(np.where(BB.loc[(x-32)*2-1,'PWSeed']<BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWTeam'],BB.loc[(x-32)*2,'PWTeam']))
            BB.loc[x,'PUTeam'] = str(np.where(BB.loc[(x-32)*2-1,'PWSeed']>BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWTeam'],BB.loc[(x-32)*2,'PWTeam']))
            BB.loc[x,'AWSeed'] = np.where(BB.loc[x,'AFScore']>=BB.loc[x,'AUScore'],BB.loc[x,'AFSeed'],BB.loc[x,'AUSeed'])
            BB.loc[x,'AWTeam'] = str(np.where(BB.loc[x,'AFScore']>=BB.loc[x,'AUScore'],BB.loc[x,'AFTeam'],BB.loc[x,'AUTeam']))
            
        
        KBBP = pd.read_csv("notebooks/step04_AllStats.csv").fillna(0)
        #Predict Round 2
        BBstats = BB[BB['Round']==2].merge(KBBP, left_on=['Year','PFTeam'],right_on=['Year','Team'],how='left')
        BBstats = BBstats.merge(KBBP, left_on=['Year','PUTeam'],right_on=['Year','Team'],how='left')
        
        
        pfs = LRF.predict(BBstats[xcol])
        pus = RFU.predict(BBstats[xcol])
        
        for x in range(33,49):
            BB.loc[x,'PFScore']=pfs[x-33]
            BB.loc[x,'PUScore']=pus[x-33]
            BB.loc[x,'AWSeed'] = np.where(BB.loc[x,'AFScore']>=BB.loc[x,'AUScore'],BB.loc[x,'AFSeed'],BB.loc[x,'AUSeed'])
            BB.loc[x,'AWTeam'] = str(np.where(BB.loc[x,'AFScore']>=BB.loc[x,'AUScore'],BB.loc[x,'AFTeam'],BB.loc[x,'AUTeam']))
            BB.loc[x,'PWSeed'] = np.where(BB.loc[x,'PFScore']>=BB.loc[x,'PUScore'],BB.loc[x,'PFSeed'],BB.loc[x,'PUSeed'])
            BB.loc[x,'PWTeam'] = str(np.where(BB.loc[x,'PFScore']>=BB.loc[x,'PUScore'],BB.loc[x,'PFTeam'],BB.loc[x,'PUTeam']))
            BB.loc[x,'ESPN'] = np.where(BB.loc[x,'AWTeam']==BB.loc[x,'PWTeam'],20,0)
        for x in range(49,57):
            BB.loc[x,'PFSeed'] = np.where(BB.loc[(x-32)*2-1,'PWSeed']<BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWSeed'],BB.loc[(x-32)*2,'PWSeed'])
            BB.loc[x,'PUSeed'] = np.where(BB.loc[(x-32)*2-1,'PWSeed']>BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWSeed'],BB.loc[(x-32)*2,'PWSeed'])
            BB.loc[x,'PFTeam'] = str(np.where(BB.loc[(x-32)*2-1,'PWSeed']<BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWTeam'],BB.loc[(x-32)*2,'PWTeam']))
            BB.loc[x,'PUTeam'] = str(np.where(BB.loc[(x-32)*2-1,'PWSeed']>BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWTeam'],BB.loc[(x-32)*2,'PWTeam']))
            BB.loc[x,'AWSeed'] = np.where(BB.loc[x,'AFScore']>=BB.loc[x,'AUScore'],BB.loc[x,'AFSeed'],BB.loc[x,'AUSeed'])
            BB.loc[x,'AWTeam'] = str(np.where(BB.loc[x,'AFScore']>=BB.loc[x,'AUScore'],BB.loc[x,'AFTeam'],BB.loc[x,'AUTeam']))
        
        #Predict Round 3
        BBstats = BB[BB['Round']==3].merge(KBBP, left_on=['Year','PFTeam'],right_on=['Year','Team'],how='left')
        BBstats = BBstats.merge(KBBP, left_on=['Year','PUTeam'],right_on=['Year','Team'],how='left')
        pfs = LRF.predict(BBstats[xcol])
        pus = RFU.predict(BBstats[xcol])
        for x in range(49,57):
            BB.loc[x,'PFScore']=pfs[x-49]
            BB.loc[x,'PUScore']=pus[x-49]
            BB.loc[x,'AWSeed'] = np.where(BB.loc[x,'AFScore']>=BB.loc[x,'AUScore'],BB.loc[x,'AFSeed'],BB.loc[x,'AUSeed'])
            BB.loc[x,'AWTeam'] = str(np.where(BB.loc[x,'AFScore']>=BB.loc[x,'AUScore'],BB.loc[x,'AFTeam'],BB.loc[x,'AUTeam']))
            BB.loc[x,'PWSeed'] = np.where(BB.loc[x,'PFScore']>=BB.loc[x,'PUScore'],BB.loc[x,'PFSeed'],BB.loc[x,'PUSeed'])
            BB.loc[x,'PWTeam'] = str(np.where(BB.loc[x,'PFScore']>=BB.loc[x,'PUScore'],BB.loc[x,'PFTeam'],BB.loc[x,'PUTeam']))
            BB.loc[x,'ESPN'] = np.where(BB.loc[x,'AWTeam']==BB.loc[x,'PWTeam'],40,0)
        for x in range(57,61):
            BB.loc[x,'PFSeed'] = np.where(BB.loc[(x-32)*2-1,'PWSeed']<BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWSeed'],BB.loc[(x-32)*2,'PWSeed'])
            BB.loc[x,'PUSeed'] = np.where(BB.loc[(x-32)*2-1,'PWSeed']>BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWSeed'],BB.loc[(x-32)*2,'PWSeed'])
            BB.loc[x,'PFTeam'] = str(np.where(BB.loc[(x-32)*2-1,'PWSeed']<BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWTeam'],BB.loc[(x-32)*2,'PWTeam']))
            BB.loc[x,'PUTeam'] = str(np.where(BB.loc[(x-32)*2-1,'PWSeed']>BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWTeam'],BB.loc[(x-32)*2,'PWTeam']))
            BB.loc[x,'AWSeed'] = np.where(BB.loc[x,'AFScore']>=BB.loc[x,'AUScore'],BB.loc[x,'AFSeed'],BB.loc[x,'AUSeed'])
            BB.loc[x,'AWTeam'] = str(np.where(BB.loc[x,'AFScore']>=BB.loc[x,'AUScore'],BB.loc[x,'AFTeam'],BB.loc[x,'AUTeam']))
        
        # Predict Round 4
        BBstats = BB[BB['Round']==4].merge(KBBP, left_on=['Year','PFTeam'],right_on=['Year','Team'],how='left')
        BBstats = BBstats.merge(KBBP, left_on=['Year','PUTeam'],right_on=['Year','Team'],how='left')
        pfs = LRF.predict(BBstats[xcol])
        pus = RFU.predict(BBstats[xcol])
        for x in range(57,61):
            BB.loc[x,'PFScore']=pfs[x-57]
            BB.loc[x,'PUScore']=pus[x-57]
            BB.loc[x,'AWSeed'] = np.where(BB.loc[x,'AFScore']>=BB.loc[x,'AUScore'],BB.loc[x,'AFSeed'],BB.loc[x,'AUSeed'])
            BB.loc[x,'AWTeam'] = str(np.where(BB.loc[x,'AFScore']>=BB.loc[x,'AUScore'],BB.loc[x,'AFTeam'],BB.loc[x,'AUTeam']))
            BB.loc[x,'PWSeed'] = np.where(BB.loc[x,'PFScore']>=BB.loc[x,'PUScore'],BB.loc[x,'PFSeed'],BB.loc[x,'PUSeed'])
            BB.loc[x,'PWTeam'] = str(np.where(BB.loc[x,'PFScore']>=BB.loc[x,'PUScore'],BB.loc[x,'PFTeam'],BB.loc[x,'PUTeam']))
            BB.loc[x,'ESPN'] = np.where(BB.loc[x,'AWTeam']==BB.loc[x,'PWTeam'],80,0)
        for x in range(61,63):
            BB.loc[x,'PFSeed'] = np.where(BB.loc[(x-32)*2-1,'PWSeed']<BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWSeed'],BB.loc[(x-32)*2,'PWSeed'])
            BB.loc[x,'PUSeed'] = np.where(BB.loc[(x-32)*2-1,'PWSeed']>BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWSeed'],BB.loc[(x-32)*2,'PWSeed'])
            BB.loc[x,'PFTeam'] = str(np.where(BB.loc[(x-32)*2-1,'PWSeed']<BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWTeam'],BB.loc[(x-32)*2,'PWTeam']))
            BB.loc[x,'PUTeam'] = str(np.where(BB.loc[(x-32)*2-1,'PWSeed']>BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWTeam'],BB.loc[(x-32)*2,'PWTeam']))
            BB.loc[x,'AWSeed'] = np.where(BB.loc[x,'AFScore']>=BB.loc[x,'AUScore'],BB.loc[x,'AFSeed'],BB.loc[x,'AUSeed'])
            BB.loc[x,'AWTeam'] = str(np.where(BB.loc[x,'AFScore']>=BB.loc[x,'AUScore'],BB.loc[x,'AFTeam'],BB.loc[x,'AUTeam']))
        
        # Predict Round 5
        BBstats = BB[BB['Round']==5].merge(KBBP, left_on=['Year','PFTeam'],right_on=['Year','Team'],how='left')
        BBstats = BBstats.merge(KBBP, left_on=['Year','PUTeam'],right_on=['Year','Team'],how='left')
        pfs = LRF.predict(BBstats[xcol])
        pus = RFU.predict(BBstats[xcol])
        for x in range(61,63):
            BB.loc[x,'PFScore']=pfs[x-61]
            BB.loc[x,'PUScore']=pus[x-61]
            BB.loc[x,'AWSeed'] = np.where(BB.loc[x,'AFScore']>=BB.loc[x,'AUScore'],BB.loc[x,'AFSeed'],BB.loc[x,'AUSeed'])
            BB.loc[x,'AWTeam'] = str(np.where(BB.loc[x,'AFScore']>=BB.loc[x,'AUScore'],BB.loc[x,'AFTeam'],BB.loc[x,'AUTeam']))
            BB.loc[x,'PWSeed'] = np.where(BB.loc[x,'PFScore']>=BB.loc[x,'PUScore'],BB.loc[x,'PFSeed'],BB.loc[x,'PUSeed'])
            BB.loc[x,'PWTeam'] = str(np.where(BB.loc[x,'PFScore']>=BB.loc[x,'PUScore'],BB.loc[x,'PFTeam'],BB.loc[x,'PUTeam']))
            BB.loc[x,'ESPN'] = np.where(BB.loc[x,'AWTeam']==BB.loc[x,'PWTeam'],160,0)
        for x in range(63,64):
            BB.loc[x,'PFSeed'] = np.where(BB.loc[(x-32)*2-1,'PWSeed']<BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWSeed'],BB.loc[(x-32)*2,'PWSeed'])
            BB.loc[x,'PUSeed'] = np.where(BB.loc[(x-32)*2-1,'PWSeed']>BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWSeed'],BB.loc[(x-32)*2,'PWSeed'])
            BB.loc[x,'PFTeam'] = str(np.where(BB.loc[(x-32)*2-1,'PWSeed']<BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWTeam'],BB.loc[(x-32)*2,'PWTeam']))
            BB.loc[x,'PUTeam'] = str(np.where(BB.loc[(x-32)*2-1,'PWSeed']>BB.loc[(x-32)*2,'PWSeed'],BB.loc[(x-32)*2-1,'PWTeam'],BB.loc[(x-32)*2,'PWTeam']))
            BB.loc[x,'AWSeed'] = np.where(BB.loc[x,'AFScore']>=BB.loc[x,'AUScore'],BB.loc[x,'AFSeed'],BB.loc[x,'AUSeed'])
            BB.loc[x,'AWTeam'] = str(np.where(BB.loc[x,'AFScore']>=BB.loc[x,'AUScore'],BB.loc[x,'AFTeam'],BB.loc[x,'AUTeam']))
        
        # Predict Round 6
        BBstats = BB[BB['Round']==6].merge(KBBP, left_on=['Year','PFTeam'],right_on=['Year','Team'],how='left')
        BBstats = BBstats.merge(KBBP, left_on=['Year','PUTeam'],right_on=['Year','Team'],how='left')
        pfs = LRF.predict(BBstats[xcol])
        pus = RFU.predict(BBstats[xcol])
        for x in range(63,64):
            BB.loc[x,'PFScore']=pfs[x-63]
            BB.loc[x,'PUScore']=pus[x-63]
            BB.loc[x,'AWSeed'] = np.where(BB.loc[x,'AFScore']>=BB.loc[x,'AUScore'],BB.loc[x,'AFSeed'],BB.loc[x,'AUSeed'])
            BB.loc[x,'AWTeam'] = str(np.where(BB.loc[x,'AFScore']>=BB.loc[x,'AUScore'],BB.loc[x,'AFTeam'],BB.loc[x,'AUTeam']))
            BB.loc[x,'PWSeed'] = np.where(BB.loc[x,'PFScore']>=BB.loc[x,'PUScore'],BB.loc[x,'PFSeed'],BB.loc[x,'PUSeed'])
            BB.loc[x,'PWTeam'] = str(np.where(BB.loc[x,'PFScore']>=BB.loc[x,'PUScore'],BB.loc[x,'PFTeam'],BB.loc[x,'PUTeam']))
            BB.loc[x,'ESPN'] = np.where(BB.loc[x,'AWTeam']==BB.loc[x,'PWTeam'],320,0)
         
        st.write(BB['ESPN'].sum(skipna=True))
        st.dataframe(BB,height=500)
        
        