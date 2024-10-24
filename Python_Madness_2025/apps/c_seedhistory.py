import streamlit as st
import pandas as pd
import numpy as np


# title of the app
st.markdown('Seed Success History Since 2008')
df = pd.read_csv('Python_Madness_2024/notebooks/step04_SeedHistory.csv')

df.insert(loc= 0 , column= 'Seed', value= np.arange(1,17))
            
#st.dataframe(df, height=600)
styler = df.style.hide()

st.write(styler.to_html(), unsafe_allow_html=True)

#st.write(df.to_html(index=False), unsafe_allow_html=True)