# Now Scrape and add 2025 
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import io

allkp = pd.read_csv('Python_Madness_2025/notebooks/step01_kenpom0824.csv').reset_index(drop=True)
b = 2025
allkp = allkp[allkp['Year']<b]

for y in range(b,2026):
    headers = {
        "User-Agent":
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.6261.70 Safari/537.36"
    }

    url = f"https://kenpom.com/index.php?y={y}"

    with requests.Session() as request:
        response = request.get(url, timeout=30,  headers=headers)
    if response.status_code != 200:
        st.write(response.raise_for_status())

    soup = BeautifulSoup(response.text, "html.parser")

    df = pd.concat(pd.read_html(io.StringIO(str(soup))))
    df.columns = ['RK','Team','Conf','Record','AdjEM','AdjO','','AdjD','','AdjT','','Luck','','SOSEM','','SOSO','','SOSD','','NCSOS','']
    keep = ['Year','Team','Conf','AdjEM','AdjO','AdjD','AdjT','SOSEM','SOSO','SOSD']
    df['Year'] = y
    df = df[keep].dropna()
    def remove_trailing_space_or_digit(text):

        while text and (text[-1].isspace() or text[-1].isdigit()):
            text = text[:-1]
        return text
    df['Team'] = df['Team'].str.replace('*', '')

    # Apply the lambda function to the 'A' column
    df['Team'] = df['Team'].apply(remove_trailing_space_or_digit)

    allkp = pd.concat([allkp, df])

allkp.to_csv('step01_kenpom0825.csv',index=False)