import streamlit as st
import pandas as pd
import numpy as np


# title of the app
st.markdown('Predicticting Each Game Independently')
#https://www.youtube.com/watch?v=xxgOkAt8nMU

FUP = pd.read_csv('Python_Madness_2024/notebooks/step07_FUStats.csv').fillna(0)
FUP = FUP[FUP['Game']>=1]
FUP['Round'] = FUP['Round'].astype('int32')
p_year = st.slider('Year: ', 2008,2024)
if p_year == 2020:
    st.markdown("No Bracket in 2020")
if p_year != 2020:

    FUPN = FUP.select_dtypes(exclude=['object'])

    # Split the dataframe into x and y
    expl = FUPN.drop(['AFScore', 'AUScore'], axis=1)
    respF = FUPN[['Year','AFScore']]
    respU = FUPN[['Year','AUScore']]
    
    # Tran and Test
    X_train = expl[expl['Year'] != p_year]
    y_train_F = respF[respF['Year'] != p_year]['AFScore'].to_numpy() 
    y_train_U = respU[respU['Year'] != p_year]['AUScore'].to_numpy() 
    X_test = expl[expl['Year'] == p_year]
    y_test_F = respF[respF['Year'] == p_year]['AFScore'].to_numpy()
    y_test_U = respU[respU['Year'] == p_year]['AUScore'].to_numpy()
    
    # Make Model
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train,y_train_F)
    y_pred_F = model.predict(X_test)
    model.fit(X_train,y_train_U)
    y_pred_U = model.predict(X_test)
    
    # How did the model do
    FUT = FUP[FUP['Year']==p_year]
    FUT = FUT.iloc[:,0:10]
    FUT['PFScore'] = y_pred_F
    FUT['PUScore'] = y_pred_U
    FUT.index = FUT.Game

    for x in range(1,64):
        FUT.loc[x,'AWTeam']=str(np.where(FUT.loc[x,'AFScore']>=FUT.loc[x,'AUScore'],FUT.loc[x,'AFTeam'],FUT.loc[x,'AUTeam']))
        FUT.loc[x,'PWTeam']=str(np.where(FUT.loc[x,'PFScore']>=FUT.loc[x,'PUScore'],FUT.loc[x,'AFTeam'],FUT.loc[x,'AUTeam']))
        FUT.loc[x,'ESPN'] = np.where(FUT.loc[x,'AWTeam']==FUT.loc[x,'PWTeam'],10*2**(FUT.loc[x,'Round']-1),0)
        #FUT['ESPN'] = (((FUT['AFScore'])>=(FUT['AUScore']))==((FUT['PFScore'])>=(FUT['PUScore']))).astype('int')*10*2**(FUT['Round']-1)
    TESPN = FUT['ESPN'].sum()
    
    st.markdown('Total ESPN Score: '+str(TESPN))
    FUT['Year'] = FUT['Year'].astype('str')
    st.dataframe(FUT)