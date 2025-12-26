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


def kenpom_code():
    # Code to be executed when the button is clicked
    # Load the CSV file
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    def remove_trailing_space_or_digit(text):
        while text and (text[-1].isspace() or text[-1].isdigit()):
            text = text[:-1]
        return text

    b = 2026

    kenpom = pd.read_csv('Python_Madness_2026/data/step01c_kenpom0825.csv')
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

    kenpom.to_csv('Python_Madness_2026/data/step01d_kenpom0825.csv',index=False)

    st.write('KenPom updated!')
    st.dataframe(kenpom[kenpom['Year']==2026].head(5))

def espnbpi_code():
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)

    y = 2026
    driver.get(f'https://www.espn.com/mens-college-basketball/bpi/_/season/{y}')
    driver.maximize_window()
    time.sleep(10)

    try:
        for _ in range(8):
            show_more_link = driver.find_element(By.CSS_SELECTOR, "a.loadMore__link")
            show_more_link.click()
            time.sleep(3)  # wait for the content to load
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
    espnBPI = pd.read_csv('Python_Madness_2026/data/step02b_espnbpi0825.csv')
    espnBPI = espnBPI[espnBPI['Year']<y]   
    espnBPI = pd.concat([espnBPI,bpi]) 

    espnBPI.to_csv('Python_Madness_2026/data/step02c_espnbpi0825.csv',index=False)      

    a = espnBPI['Team'].unique()
    b = pd.DataFrame({'tm':sorted(a)})
    b.to_csv('Python_Madness_2026/data/bbb.csv',index=False) 
    st.write('ESPN BPI updated!')
    st.dataframe(espnBPI[espnBPI['Year']==2026].head(5))

def scrapeBR():
    allbr = pd.read_csv('Python_Madness_2026/data/step03b_br0825.csv')
    y = 2026
    allbr = allbr[allbr['Year']<y]
    url = f'https://www.sports-reference.com/cbb/seasons/men/{y}-ratings.html'
    html = requests.get(url).content
    df_list = pd.read_html(html)
    df = df_list[-1]  # Assuming the relevant table is the last one
    df.columns = ['Rk','Team','Conf','','','W','L','Pts','Opp','MOV','','SOS','','OSRS','DSRS','SRS','','','']

    keep = ['Year','Team','W', 'L', 'Pts', 'Opp', 'MOV','SOS', 'OSRS', 'DSRS', 'SRS']

    df['Year'] = y

    df = df[keep].dropna()
    df = df[df['Team']!='School']

    allbr = pd.concat([allbr,df])

    allbr.to_csv('Python_Madness_2026/data/step03b_br0825.csv',index=False)
    st.write('Basketball Reference updated!')
    st.dataframe(allbr[allbr['Year']==2026].head(5))

def bartdata():
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)

    b = 2026

    bartdata = pd.read_csv('Python_Madness_2026/data/step04b_bart0825.csv')
    bartdata = bartdata[bartdata['Year']<b]

    for y in range(b,2026):

        driver.get(f'https://barttorvik.com/team-tables_each.php?year={y}&top=0&conlimit=All')
        driver.maximize_window()
        element = WebDriverWait(driver, 1000).until(EC.presence_of_element_located((By.XPATH, "//table")))
        with io.StringIO(driver.page_source) as f:
            df = pd.read_html(f)[0]
            
        df['Year'] = y
        keep = ['Year', 'Team', 'Adj OE', 'Adj DE', 'Barthag', 'Wins',
            'Games', 'eFG', 'eFG D.', 'FT Rate', 'FT Rate D', 'TOV%', 'TOV% D',
            'O Reb%', 'Op OReb%', 'Raw T', '2P %', '2P % D.', '3P %', '3P % D.',
            'Blk %', 'Blked %', 'Ast %', 'Op Ast %', '3P Rate', '3P Rate D',
            'Adj. T', 'Avg Hgt.', 'Eff. Hgt.', 'Exp.', 'PAKE', 'PASE',
            'Talent', 'FT%', 'Op. FT%', 'PPP Off.', 'PPP Def.',
            'Elite SOS']
        df = df[keep]
        bartdata = pd.concat([bartdata,df])

    bartdata.to_csv('Python_Madness_2026/data/step04b_bart0825.csv',index=False)
    st.write('Bart Data updated!')
    st.dataframe(bartdata[bartdata['Year']==2026].head(5))    

def combined():
    # Fix KenPom
    KP = pd.read_csv('Python_Madness_2026/data/step01d_kenpom0825.csv').dropna()
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
    KP.to_csv('Python_Madness_2026/data/step01d_kenpom0825.csv',index=False)
    pd.DataFrame(sn).to_csv('Python_Madness_2026/data/asn.csv',index=False)
    st.write(KP2fix)
    st.write('KenPom Fixed!')

    # Make ESPN BPI Name match Kep Pom
    BPI = pd.read_csv('Python_Madness_2026/data/step02c_espnbpi0825.csv').dropna()
    BPI['Year'] = pd.to_numeric(BPI['Year'], errors='coerce').astype('Int32')
    LF = list(repair['tofix'])
    LR = list(repair['replacewith'])
    BPI['Team'] = BPI['Team'].replace({'State':'St.'}, regex=True)
    BPI['Team'] = BPI['Team'].replace(LF,LR)
    BPI = BPI[BPI['Team']!='out']
    BPIN = BPI['Team'].unique()
    BPI2fix = list(set(BPIN) - set(sn))

    BPI.to_csv('Python_Madness_2026/data/step02c_espnbpi0825.csv',index=False)
    st.write(BPI2fix)
    st.write('ESPNBPI Fixed!')

    # Make Baskeball Reference Team Names match Ken Pom
    BR = pd.read_csv("Python_Madness_2026/data/step03b_br0825.csv").dropna()
    BR['Year'] = pd.to_numeric(BR['Year'], errors='coerce').astype('Int32')
    LF = list(repair['tofix'])
    LR = list(repair['replacewith'])
    BR['Team'] = BR['Team'].replace({'State':'St.'}, regex=True)
    BR['Team'] = BR['Team'].replace(LF,LR)
    BR = BR[BR['Team']!='out']
    BRN = BR['Team'].unique()
    BR2fix = list(set(BRN) - set(sn))
    BR.to_csv('Python_Madness_2026/data/step03b_br0825.csv',index=False)
    st.write(BR2fix)
    st.write('Basketball Reference Fixed!')

    # Make Bart Reference Team Names match Ken Pom
    bartdata = pd.read_csv("Python_Madness_2026/data/step04b_bart0825.csv").dropna()
    LF = list(repair['tofix'])
    LR = list(repair['replacewith'])
    bartdata['Team'] = bartdata['Team'].replace({'State':'St.'}, regex=True)
    bartdata['Team'] = bartdata['Team'].replace(LF,LR)
    bartdata = bartdata[bartdata['Team']!='out']
    bartdataN = bartdata['Team'].unique()
    bart2fix = list(set(bartdataN) - set(sn))
    bartdata.to_csv('Python_Madness_2026/data/step04b_bart0825.csv',index=False)
    st.write(bart2fix)
    st.write('Bart Fixed!')

    # Compute Seed History
    AG = pd.read_csv("Python_Madness_2026/data/step05c_FUHistory.csv").dropna()
    AG = AG[AG['Year']<2026]

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


# Display the button
if st.button("Update Data"):
    #kenpom_code()
    #espnbpi_code()
    #scrapeBR()
    #bartdata()
    combined()
