
  
import streamlit as st
import pandas as pd
import numpy as np


# title of the app
st.markdown('Team Performance Against Seed Expectation Since 2008')
pase = pd.read_csv('Python_Madness_2024/notebooks/step05_PASE.csv')


            
st.dataframe(pase)



