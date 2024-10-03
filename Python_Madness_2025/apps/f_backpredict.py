import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import numpy as np

def app():
    # Title of the app
    st.markdown('Predicting Each Game Independently')
    # https://www.youtube.com/watch?v=xxgOkAt8nMU
    
    FUP = pd.read_csv('Python_Madness_2024/notebooks/step07_FUStats.csv').fillna(0)
    FUP = FUP[FUP['Game'] >= 1]
    FUP['Round'] = FUP['Round'].astype('int32')
    p_year = st.slider('Year: ', 2008, 2023)
    
    if p_year == 2020:
        st.markdown("No Bracket in 2020")
    elif p_year != 2020:
    
        FUPN = FUP.select_dtypes(exclude=['object'])
        
        # Split the dataframe into features and targets
        expl = FUPN[FUPN['Year'] != p_year].drop(['AFScore', 'AUScore'], axis=1)

        respF = FUPN[FUPN['Year'] != p_year]['AFScore']
        respU = FUPN[FUPN['Year'] != p_year]['AUScore']
        
        # Create SVM models with adjustable parameters
        kernel_options = ['linear', 'poly', 'rbf', 'sigmoid']
        kernel = st.selectbox('Kernel (default: rbf)', options=kernel_options, index=3)  # Default to 'rbf'
        C = st.number_input('Regularization Parameter (C) (default: 1.0)', min_value=0.0, value=1.0)
        epsilon = st.number_input('Epsilon (default: 0.1)', min_value=0.0, value=0.1)

        model_F = SVR(kernel=kernel, C=C, epsilon=epsilon)
        model_U = SVR(kernel=kernel, C=C, epsilon=epsilon)

        
        # Train the models on the entire dataset
        model_F.fit(expl, respF)
        model_U.fit(expl, respU)
        
        # Use the trained model to predict scores for the selected year
        FUNY = FUPN[FUPN['Year'] == p_year].drop(['AFScore', 'AUScore'], axis=1)
        pfs = model_F.predict(FUNY)
        pus = model_U.predict(FUNY)
        
        FUT = FUP[FUP['Year']==p_year]
        FUT['PFScore'] = pfs
        FUT['PUScore'] = pus
        FUT.index = FUT.Game
    
        for x in range(1, 64):
            FUT.loc[x, 'AWTeam'] = str(np.where(FUT.loc[x, 'AFScore'] >= FUT.loc[x, 'AUScore'], FUT.loc[x, 'AFTeam'], FUT.loc[x, 'AUTeam']))
            FUT.loc[x, 'PWTeam'] = str(np.where(FUT.loc[x, 'PFScore'] >= FUT.loc[x, 'PUScore'], FUT.loc[x, 'AFTeam'], FUT.loc[x, 'AUTeam']))
            FUT.loc[x, 'ESPN'] = np.where(FUT.loc[x, 'AWTeam'] == FUT.loc[x, 'PWTeam'], 10 * 2**(FUT.loc[x, 'Round'] - 1), 0)
    
        TESPN = FUT['ESPN'].sum()
        
        st.markdown('Total ESPN Score: ' + str(TESPN))
        FUT['Year'] = FUT['Year'].astype('str')
        st.dataframe(FUT)