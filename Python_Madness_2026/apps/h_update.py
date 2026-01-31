from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import streamlit as st
import pandas as pd
<<<<<<< HEAD
import io
import time
import requests
import numpy as np
import re
=======
>>>>>>> 6b192ff (sync)

YEAR = 2026

def kenpom_code():

<<<<<<< HEAD
    CSV_PATH = "Python_Madness_2026/data/step01b_kenpom.csv"

    # ---------------- Chrome ----------------
    options = Options()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    def clean_team(name):
        """
        Clean team names by removing trailing numeric rankings like '1*', '3', etc.
        """
        name = str(name).strip()
        # Remove trailing number optionally followed by asterisk
        return re.sub(r'\s\d+\*?$', '', name)

    # ---------------- Existing CSV ----------------
    try:
        kenpom = pd.read_csv(CSV_PATH)
        kenpom = kenpom.loc[:, ~kenpom.columns.duplicated()]
        kenpom = kenpom[kenpom["Year"] != YEAR]  # Remove old entries for this year
    except FileNotFoundError:
        kenpom = pd.DataFrame()

    # ---------------- Load KenPom ----------------
    driver.get(f"https://kenpom.com/index.php?y={YEAR}")

    table = WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.ID, "ratings-table"))
    )

    df = pd.read_html(table.get_attribute("outerHTML"))[0]

    # ---------------- Flatten headers ----------------
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(x) for x in col if x not in ("", None, "nan")).strip()
            for col in df.columns
        ]

    df = df.loc[:, ~df.columns.duplicated()]
    cols = list(df.columns)

    # ---------------- Anchor columns ----------------
    col_team = next(c for c in cols if "team" in c.lower())
    col_conf = next(c for c in cols if "conf" in c.lower())
    col_adjt = next(c for c in cols if "adjt" in c.lower())

    i = cols.index(col_adjt)

    # Correct offsets
    col_net = cols[i - 5]  # AdjEM
    col_o   = cols[i - 4]  # AdjO
    col_d   = cols[i - 2]  # AdjD
=======
    def clean_team(name):
        """Remove trailing digits or spaces from team names"""
        name = str(name)
        while name and (name[-1].isdigit() or name[-1].isspace()):
            name = name[:-1]
        return name

    kenpom_path = 'Python_Madness_2026/data/step01b_kenpom.csv'

    # ================= READ EXISTING DATA =================
    kenpom = pd.read_csv(kenpom_path)
    kenpom = kenpom.loc[:, ~kenpom.columns.duplicated()]
    kenpom = kenpom[kenpom['Year'] < b]

    # ================= SCRAPE NEW YEAR =================
    driver.get(f'https://kenpom.com/index.php?y={b}')
    driver.maximize_window()

    # Wait until table rows are loaded
    WebDriverWait(driver, 30).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "table#ratings-table tbody tr"))
    )

    # Extract table rows
    rows = driver.find_elements(By.CSS_SELECTOR, "table#ratings-table tbody tr")
    data = []
    for row in rows:
        cells = row.find_elements(By.TAG_NAME, "td")
        data.append([c.text for c in cells])

    # Convert to DataFrame
    df = pd.DataFrame(data)
    st.write("Raw table shape:", df.shape)
    st.write("Raw table preview:", df.head())

    # ================= ADD COLUMN NAMES =================
    columns = [
        "Rk","Team","Conf","W-L","NetRtg","ORtg","DRtg","AdjT","Luck",
        "SOS_NetRtg","SOS_ORtg","SOS_DRtg","SOS_AdjT","NCSOS_NetRtg"
    ]
    # If table has extra columns, fill remaining names with dummy
    while len(columns) < df.shape[1]:
        columns.append(f"Extra_{len(columns)}")

    df.columns = columns
>>>>>>> 6b192ff (sync)

    df = df.rename(columns={
        col_team: "Team",
        col_conf: "Conf",
        col_net:  "NetRtg",
        col_o:    "ORtg",
        col_d:    "DRtg",
        col_adjt: "AdjT"
    })

<<<<<<< HEAD
    # ---------------- Keep only needed columns ----------------
    df = df[["Team", "Conf", "NetRtg", "ORtg", "DRtg", "AdjT"]]

    # ---------------- Final cleanup ----------------
    df["Team"] = df["Team"].apply(clean_team)
    df["Year"] = YEAR
    df = df.dropna(subset=["Team"])

    # ---------------- Append ----------------
    kenpom = pd.concat([kenpom, df], ignore_index=True) if not kenpom.empty else df.copy()

    # ---------------- Deduplicate + enforce column order ----------------
    kenpom = kenpom.loc[:, ~kenpom.columns.duplicated()]

    # ---------------- Save CSV with exact order ----------------
    SAVE_COLS = ["Year","Team", "Conf", "NetRtg", "ORtg", "DRtg", "AdjT"]
    kenpom.to_csv(CSV_PATH, index=False, columns=SAVE_COLS)


    driver.quit()

    # ---------------- Streamlit display ----------------
    st.success("KenPom updated successfully")
    st.dataframe(
        kenpom.loc[kenpom["Year"] == YEAR,
                   ["Year", "Team", "Conf", "NetRtg", "ORtg", "DRtg", "AdjT"]]
        .head(15)
    )
=======
    # ================= CONCAT =================
    # Align columns with existing kenpom
    for col in kenpom.columns:
        if col not in df.columns:
            df[col] = pd.NA
    for col in df.columns:
        if col not in kenpom.columns:
            kenpom[col] = pd.NA

    df = df[kenpom.columns]  # Reorder to match kenpom
    kenpom = pd.concat([kenpom, df], ignore_index=True)

    # ================= SAVE =================
    kenpom.to_csv(kenpom_path, index=False)

    driver.quit()

    st.write(f'KenPom updated for {b}!')
    st.dataframe(kenpom[kenpom['Year'] == b].head())
    st.write("Columns:", kenpom.columns)
>>>>>>> 6b192ff (sync)

def espnbpi_code():
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)


    driver.get(f'https://www.espn.com/mens-college-basketball/bpi/_/season/{YEAR}')
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

    bpi['Year'] = YEAR
    bpi = bpi.iloc[:, [12, 0, 6, 7]]
    bpi.columns = ['Year', 'Team', 'BPI(O)', 'BPI(D)']

    espnBPI = pd.read_csv('Python_Madness_2026/data/step02b_espnbpi.csv')
    espnBPI = espnBPI[espnBPI['Year'] < YEAR]
    espnBPI = pd.concat([espnBPI, bpi])

    espnBPI.to_csv('Python_Madness_2026/data/step02b_espnbpi.csv', index=False)
    driver.quit()

    st.write('ESPN BPI updated!')
    st.dataframe(espnBPI[espnBPI['Year'] == YEAR].head())


def scrapeBR():
    allbr = pd.read_csv('Python_Madness_2026/data/step03b_br.csv')

    allbr = allbr[allbr['Year'] < YEAR]

    url = f'https://www.sports-reference.com/cbb/seasons/men/{YEAR}-ratings.html'
    html = requests.get(url).content
    df = pd.read_html(html)[-1]

    # Clean column names
    df.columns = [c[1] if isinstance(c, tuple) else c for c in df.columns]

    df = df.rename(columns={'School': 'Team'})
    df['Year'] = YEAR

    keep = ['Year', 'Team', 'W', 'L', 'Pts', 'Opp', 'MOV', 'SOS', 'OSRS', 'DSRS', 'SRS']
    df = df[keep]
    df = df[df['Team'] != 'School'].dropna()

    allbr = pd.concat([allbr, df])
    allbr.to_csv('Python_Madness_2026/data/step03b_br.csv', index=False)

    st.write('Basketball Reference updated!')
    st.dataframe(allbr[allbr['Year'] == YEAR].head())


def bartdata():
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)


    bartdata = pd.read_csv('Python_Madness_2026/data/step04b_bart.csv')
    bartdata = bartdata[bartdata['Year'] < YEAR]

    driver.get(f'https://barttorvik.com/team-tables_each.php?year={YEAR}&top=0&conlimit=All')
    driver.maximize_window()

    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.XPATH, "//table"))
    )

    with io.StringIO(driver.page_source) as f:
        df = pd.read_html(f)[0]

    df['Year'] = YEAR
    bartdata = pd.concat([bartdata, df])
    bartdata.to_csv('Python_Madness_2026/data/step04b_bart.csv', index=False)

    driver.quit()
    st.write('Bart Data updated!')
    st.dataframe(bartdata[bartdata['Year'] == YEAR].head())

def combined():
    # Fix KenPom
    KP = pd.read_csv('Python_Madness_2026/data/step01b_kenpom.csv').dropna()
    KP['Year'] = pd.to_numeric(KP['Year'], errors='coerce').astype('Int32')
    repair = pd.read_csv('Python_Madness_2026/data/step05b_repair.csv', encoding='latin1')
    LF = list(repair['tofix'])
    LR = list(repair['replacewith'])
    KP['Team'] = KP['Team'].replace(LF,LR)
    KP = KP[KP['Team']!='out']
    KP = KP[KP['Team']!='Team']
    sn = KP[KP['Year']>2024].sort_values('Team')['Team'].unique()
    KPN = KP.sort_values('Team')['Team'].unique()
    pd.DataFrame(KPN).to_csv('Python_Madness_2026/data/kpn.csv', index=False)
    KP2fix = list(set(KPN) - set(sn))
    KP.to_csv('Python_Madness_2026/data/step01b_kenpom.csv',index=False)
    pd.DataFrame(sn).to_csv('Python_Madness_2026/data/asn.csv',index=False)
    st.write(KP2fix)
    st.write('KenPom Fixed!')

    # Make ESPN BPI Name match Kep Pom
    BPI = pd.read_csv('Python_Madness_2026/data/step02b_espnbpi.csv').dropna()
    BPI['Year'] = pd.to_numeric(BPI['Year'], errors='coerce').astype('Int32')
    LF = list(repair['tofix'])
    LR = list(repair['replacewith'])
    BPI['Team'] = BPI['Team'].replace({'State':'St.'}, regex=True)
    BPI['Team'] = BPI['Team'].replace(LF,LR)
    BPI = BPI[BPI['Team']!='out']
    BPIN = BPI['Team'].unique()
    BPI2fix = list(set(BPIN) - set(sn))

    BPI.to_csv('Python_Madness_2026/data/step02b_espnbpi.csv',index=False)
    st.write(BPI2fix)
    st.write('ESPNBPI Fixed!')

    # Make Baskeball Reference Team Names match Ken Pom
    BR = pd.read_csv("Python_Madness_2026/data/step03b_br.csv").dropna()
    BR['Year'] = pd.to_numeric(BR['Year'], errors='coerce').astype('Int32')
    LF = list(repair['tofix'])
    LR = list(repair['replacewith'])
    BR['Team'] = BR['Team'].replace({'State':'St.'}, regex=True)
    BR['Team'] = BR['Team'].replace(LF,LR)
    BR = BR[BR['Team']!='out']
    BRN = BR['Team'].unique()
    BR2fix = list(set(BRN) - set(sn))
    BR.to_csv('Python_Madness_2026/data/step03b_br.csv',index=False)
    st.write(BR2fix)
    st.write('Basketball Reference Fixed!')

    # Make Bart Reference Team Names match Ken Pom
    bartdata = pd.read_csv("Python_Madness_2026/data/step04b_bart.csv").dropna()
    LF = list(repair['tofix'])
    LR = list(repair['replacewith'])
    bartdata['Team'] = bartdata['Team'].replace({'State':'St.'}, regex=True)
    bartdata['Team'] = bartdata['Team'].replace(LF,LR)
    bartdata = bartdata[bartdata['Team']!='out']
    bartdataN = bartdata['Team'].unique()
    bart2fix = list(set(bartdataN) - set(sn))
    bartdata.to_csv('Python_Madness_2026/data/step04b_bart.csv',index=False)
    st.write(bart2fix)
    st.write('Bart Fixed!')

    # Compute Seed History
    AG = pd.read_csv("Python_Madness_2026/data/step05c_FUHistory.csv").dropna()
    AG = AG[AG['Year']<YEAR]

    LG = AG[AG['Round']==6]
    CS = pd.DataFrame({'Round','Seed'})

    for row in range(len(LG)):
        if LG.iloc[row]["AFScore"]>LG.iloc[row]["AUScore"]:
            ws = LG.iloc[row]["AFSeed"]

        else:
            ws = LG.iloc[row]["AUSeed"]
        CS.loc[row,'Round'] = 'W'
        CS.loc[row,'Seed'] = ws

    cs = pd.crosstab(index=CS['Seed'],columns=CS['Round'])
    fs = pd.crosstab(index=AG['AFSeed'],columns=AG['Round'])
    us = pd.crosstab(index=AG['AUSeed'],columns=AG['Round'])
    ufs = fs.add(us, fill_value=0).astype(int).drop([0],axis=1)
    sh = ufs.add(cs, fill_value=0).fillna(0).astype(int)
    nos = sh.max().max()
    sh['Exp Wins'] = (sh.sum(axis=1)-nos)/nos
    sh.to_csv('Python_Madness_2026/data/step05d_SeedHistory.csv',index=False) 
    
    st.write('Seed History Updated!')

    # Make the history Favored team names match Ken Pom
    AG = pd.read_csv("Python_Madness_2026/data/step05c_FUHistory.csv").dropna()
    AG['AFTeam'] = AG['AFTeam'].replace(LF,LR)
    AG = AG[AG['AFTeam']!='out']
    AGN = AG['AFTeam'].unique()
    AG2fix = list(set(AGN) - set(sn))
    st.write(AG2fix)
    st.write('Favored Checked!')
    # Make the history Underdog team names match Ken Pom
    AG['AUTeam'] = AG['AUTeam'].replace(LF,LR)
    AG = AG[AG['AUTeam']!='out']
    AGN = AG['AUTeam'].unique()
    AG2fix = list(set(AGN) - set(sn))
    st.write(AG2fix)
    st.write('Underdogs Checked!')

    AG.to_csv('Python_Madness_2026/data/step05c_FUHistory.csv',index=False) 
    

    #Build PASE

    # table of wins
    fw = pd.crosstab(index=AG['AFTeam'],columns=AG['Year'])
    uw = pd.crosstab(index=AG['AUTeam'],columns=AG['Year'])

    # How many wins did the team acutally get per year if they were in the tournament
    cw = fw.add(uw, fill_value=0).fillna(0).astype(int) - 1
    cw = cw.replace(-1, '')

    # table of seeds
    AG1 = AG[AG['Round']==1]
    fs = AG1.pivot(index='AFTeam', columns='Year', values = 'AFSeed').fillna(0).astype(int)
    us = AG1.pivot(index='AUTeam', columns='Year', values = 'AUSeed').fillna(0).astype(int)

    cs = fs.add(us, fill_value=0).fillna(0).astype(int)
    cs = cs.replace(0, '')

    # table of expected wins
    ew = pd.DataFrame({'Seed':np.arange(1,17),'Exp Wins':sh['Exp Wins']})

    # ewty is expected number of wins by team and by year
    ewty = cs.replace(list(ew['Seed']),list(ew['Exp Wins']))
    # fill in zeros into the blank spaces
    pd.set_option('future.no_silent_downcasting', True)

    ewz = (ewty * -1).replace(r'^\s*$', np.nan, regex=True).fillna(0)

    cwz = cw.replace(r'^\s*$', np.nan, regex=True).fillna(0)
    pase = cwz.add(ewz).replace(0, np.nan)
    pase['PASE']=pase.mean(axis=1)
    pase = pase.iloc[:,-1:]

    
    # Make sure PASE team names are matched to Ken Pom
    pase = pase.reset_index()
    pase = pase.rename(columns={'index': 'Team'})
    pase.columns = ['Team','PASE']
    pase['Team'] = pase['Team'].replace(LF,LR)
    pase = pase[pase['Team']!='out']
    paseN = pase['Team'].unique()
    pase2fix = list(set(paseN) - set(sn))

    pase.to_csv('Python_Madness_2026/data/step05e_PASE.csv', index=False)  
    pase2fix
    st.write('PASE Checked and Updated!')

    #Combine all the datframes
    KP = KP.merge(BPI,how='left', on=['Year','Team'])
    KP = KP.merge(BR, how = 'left', on=['Year','Team'])
    KP = KP.merge(bartdata, how = 'left', on=['Year','Team'])
    KPBPIBRP = KP.merge(pase, how = 'left', on = ['Team'])
    KPBPIBRP.to_csv('Python_Madness_2026/data/step05f_AllStats.csv',index=False)

    st.write('All Stats Updated!')

    #Attach stats to Tournament Games

    AG = AG[AG['Year']>=2008]
    AGstats = AG.merge(KPBPIBRP, left_on = ['Year','AFTeam'], right_on = ['Year','Team'], how = 'left')
    AGstatsandU = AGstats.merge(KPBPIBRP, left_on = ['Year','AUTeam'], right_on = ['Year','Team'], how = 'left')
    AGstatsandU = AGstatsandU.drop(['Team_x','Team_y'],axis=1)
    AGstatsandU.to_csv('Python_Madness_2026/data/step05g_FUStats.csv',index=False)
    st.write('Tournament Stats Updated!')


# ================= STREAMLIT =================

if st.button("Update Data"):
    #kenpom_code()
    #espnbpi_code()
    #scrapeBR()
    #bartdata()
    combined()
