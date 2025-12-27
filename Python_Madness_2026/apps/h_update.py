from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import streamlit as st
import pandas as pd
import io
import time
import requests
import numpy as np

b = 2025

def kenpom_code():
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)

    def clean_team(name):
        name = str(name)
        while name and (name[-1].isdigit() or name[-1].isspace()):
            name = name[:-1]
        return name

    # ================= READ EXISTING DATA =================
    kenpom = pd.read_csv('Python_Madness_2026/data/step01b_kenpom.csv')

    # Drop duplicate columns if any
    kenpom = kenpom.loc[:, ~kenpom.columns.duplicated()]

    # Keep only previous years
    kenpom = kenpom[kenpom['Year'] < b]

    driver.get(f'https://kenpom.com/index.php?y={b}')
    driver.maximize_window()

    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.XPATH, "//table"))
    )

    with io.StringIO(driver.page_source) as f:
        df = pd.read_html(f)[0]

    # ================= FLATTEN HEADERS =================
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in col if x and x != 'nan']).strip()
            for col in df.columns
        ]

    # Drop duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # ================= DYNAMIC COLUMN MAPPING =================
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if 'team' in cl or 'school' in cl:
            col_map[c] = 'Team'
        elif cl.startswith('conf'):
            col_map[c] = 'Conf'
        elif 'adjem' in cl:
            col_map[c] = 'NetRtg'
        elif 'adjo' in cl:
            col_map[c] = 'ORtg'
        elif 'adjd' in cl:
            col_map[c] = 'DRtg'
        elif 'adjt' in cl:
            col_map[c] = 'AdjT'

    df = df.rename(columns=col_map)

    # ================= ENSURE REQUIRED COLUMNS =================
    required = ['Team', 'Conf', 'NetRtg', 'ORtg', 'DRtg', 'AdjT']
    for col in required:
        if col not in df.columns:
            df[col] = pd.NA

    df['Year'] = b
    df['Team'] = df['Team'].apply(clean_team)

    df = df[['Year'] + required].dropna(subset=['Team'])

    # ================= CONCAT SAFELY =================
    # Before concatenating, make sure both DataFrames have unique columns
    kenpom = kenpom.loc[:, ~kenpom.columns.duplicated()]
    df = df.loc[:, ~df.columns.duplicated()]

    # Also align columns in case the CSV has extra columns
    common_cols = [col for col in df.columns if col in kenpom.columns] + \
                  [col for col in df.columns if col not in kenpom.columns]

    df = df[common_cols]

    kenpom = pd.concat([kenpom, df], ignore_index=True)

    kenpom.to_csv('Python_Madness_2026/data/step01b_kenpom.csv', index=False)

    driver.quit()

    st.write('KenPom updated!')
    st.dataframe(kenpom[kenpom['Year'] == b].head())


def espnbpi_code():
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)


    driver.get(f'https://www.espn.com/mens-college-basketball/bpi/_/season/{b}')
    driver.maximize_window()
    time.sleep(10)

    for _ in range(8):
        try:
            driver.find_element(By.CSS_SELECTOR, "a.loadMore__link").click()
            time.sleep(3)
        except:
            break

    with io.StringIO(driver.page_source) as f:
        tables = pd.read_html(f)
        bpi = pd.concat(tables[:2], axis=1)

    bpi['Year'] = b
    bpi = bpi.iloc[:, [12, 0, 6, 7]]
    bpi.columns = ['Year', 'Team', 'BPI(O)', 'BPI(D)']

    espnBPI = pd.read_csv('Python_Madness_2026/data/step02b_espnbpi.csv')
    espnBPI = espnBPI[espnBPI['Year'] < b]
    espnBPI = pd.concat([espnBPI, bpi])

    espnBPI.to_csv('Python_Madness_2026/data/step02b_espnbpi.csv', index=False)
    driver.quit()

    st.write('ESPN BPI updated!')
    st.dataframe(espnBPI[espnBPI['Year'] == b].head())


def scrapeBR():
    allbr = pd.read_csv('Python_Madness_2026/data/step03b_br.csv')

    allbr = allbr[allbr['Year'] < b]

    url = f'https://www.sports-reference.com/cbb/seasons/men/{b}-ratings.html'
    html = requests.get(url).content
    df = pd.read_html(html)[-1]

    # Clean column names
    df.columns = [c[1] if isinstance(c, tuple) else c for c in df.columns]

    df = df.rename(columns={'School': 'Team'})
    df['Year'] = b

    keep = ['Year', 'Team', 'W', 'L', 'Pts', 'Opp', 'MOV', 'SOS', 'OSRS', 'DSRS', 'SRS']
    df = df[keep]
    df = df[df['Team'] != 'School'].dropna()

    allbr = pd.concat([allbr, df])
    allbr.to_csv('Python_Madness_2026/data/step03b_br.csv', index=False)

    st.write('Basketball Reference updated!')
    st.dataframe(allbr[allbr['Year'] == b].head())


def bartdata():
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)


    bartdata = pd.read_csv('Python_Madness_2026/data/step04b_bart.csv')
    bartdata = bartdata[bartdata['Year'] < b]

    driver.get(f'https://barttorvik.com/team-tables_each.php?year={b}&top=0&conlimit=All')
    driver.maximize_window()

    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.XPATH, "//table"))
    )

    with io.StringIO(driver.page_source) as f:
        df = pd.read_html(f)[0]

    df['Year'] = b
    bartdata = pd.concat([bartdata, df])
    bartdata.to_csv('Python_Madness_2026/data/step04b_bart.csv', index=False)

    driver.quit()
    st.write('Bart Data updated!')
    st.dataframe(bartdata[bartdata['Year'] == b].head())


# ================= STREAMLIT =================

if st.button("Update Data"):
    kenpom_code()
    espnbpi_code()
    scrapeBR()
    bartdata()
