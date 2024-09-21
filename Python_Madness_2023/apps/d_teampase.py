
  
import streamlit as st
import pandas as pd
import numpy as np

def app():
    # title of the app
    st.markdown('How the teams have faired since 1985')
    pase = pd.read_csv('notebooks/step04_PASE.csv')
    

             
    st.dataframe(pase)

    

  