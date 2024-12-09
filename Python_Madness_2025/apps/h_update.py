import pandas as pd
import streamlit as st
from selenium import webdriver
import io
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
import time

def kenpom_code():
    # Code to be executed when the button is clicked
    # Load the CSV file
    df = pd.read_csv('Python_Madness_2025/data/Just2025.csv')
    df = df.iloc[:, 1:8]
    df2 = pd.read_csv('Python_Madness_2025/data/step01_kenpom0824.csv')
    df_concat = pd.concat([df2, df])
    df_concat = df_concat.reset_index(drop=True)
    df_concat.to_csv('Python_Madness_2025/data/step01_kenpom0825.csv', index=False)
    st.write("KenPom Data Updated")
    st.dataframe(df_concat) 

def espnbpi_code():
    service = Service(executable_path="chromedriver.exe")
    options = webdriver.ChromeOptions()
    #options.add_argument('--headless')
    driver = webdriver.Chrome(service=service,options=options)
    y = 2024
    driver.get(f'https://www.espn.com/mens-college-basketball/bpi/_/view/bpi/season/{y}')
    driver.maximize_window()
    time.sleep(10)
    element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "onetrust-accept-btn-handler"))).click()

    try:
        for x in range(8):
            element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.LINK_TEXT, "Show More"))).click() 
            time.sleep(5)
    except:
        try:
            element = WebDriverWait(driver, 1000).until(EC.presence_of_element_located((By.XPATH, "//table")))
            with io.StringIO(driver.page_source) as f:
                df = pd.read_html(f)[0]
                df2 = pd.read_html(f)[1]
        finally:
            
            bpi = pd.concat([df, df2], axis=1)
            bpi['Year']=y
            bpi = bpi.iloc[:,[12,0,6,7]]
            bpi.columns = ['Year','Team','BPI(O)','BPI(D)']
            print(y)
    espnBPI = pd.read_csv('step02_espnbpi0824.csv')
    espnBPI = espnBPI[espnBPI['Year']<y]   
    espnBPI = pd.concat([espnBPI,bpi]) 

    espnBPI.to_csv('step02_espnbpi0825.csv',index=False)      

# Display the button
if st.button("Update Data"):
    kenpom_code()
