import pandas as pd
import streamlit as st

def execute_code():
    # Code to be executed when the button is clicked
    # Load the CSV file
    df = pd.read_csv('Python_Madness_2025/data/Just2025.csv')
    df = df.iloc[:, 1:8]
    df2 = pd.read_csv('Python_Madness_2025/data/step01_kenpom0824.csv')
    df_concat = pd.concat([df2, df])
    df_concat.to_csv('Python_Madness_2025/data/step01_kenpom0825.csv', index=False)
    st.dataframe(df_concat) 

# Display the button
if st.button("Update Data"):
    execute_code()
