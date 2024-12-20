from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import streamlit as st
import pandas as pd
import io


def kenpom_code():
    # Code to be executed when the button is clicked
    # Load the CSV file
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    def remove_trailing_space_or_digit(text):
        while text and (text[-1].isspace() or text[-1].isdigit()):
            text = text[:-1]
        return text

    b = 2024

    kenpom = pd.read_csv('data/step01c_kenpom0824.csv')
    kenpom = kenpom[kenpom['Year']<b]

    for y in range(b,2026):

        driver.get(f'https://kenpom.com/index.php?y={y}')
        driver.maximize_window()
        element = WebDriverWait(driver, 1000).until(EC.presence_of_element_located((By.XPATH, "//table")))
        with io.StringIO(driver.page_source) as f:
            df = pd.read_html(f)[0]
        df['Year'] = y
        #df.columns = df.columns.droplevel(list(range(18)))
        new_columns = ['Rank', 'Team', 'Conf', 'W-L', 'NetRtg', 'ORtg', 'rk','DRtg','rk','AdjT'
                        ,'rk','Luck','rk','Luck','rk','Luck','rk','Luck','rk','Luck','rk','Year']  # create a list of new column names
        df.columns = new_columns
        columns_to_keep = ['Year','Team', 'Conf', 'NetRtg', 'ORtg', 'DRtg','AdjT']  # create a list of column names to keep
        df = df.loc[:, columns_to_keep]

        # Apply the lambda function to the 'A' column
        df['Team'] = df['Team'].apply(remove_trailing_space_or_digit)

        # assuming kenpom is your main DataFrame
        kenpom = pd.concat([kenpom, df], ignore_index=True)

    kenpom.to_csv('step01d_kenpom0825.csv',index=False)

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
