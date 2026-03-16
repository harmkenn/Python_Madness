import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import random


# ---------------------------------------------------------------------------
# Bracket HTML visualization  (color scheme from ensemble app)
# ---------------------------------------------------------------------------

def highlight_html(df):
    html = """
    <style>
        .bracket {
            display: flex;
            gap: 20px;
            overflow-x: auto;
            padding: 20px;
            background-color: #1a1a1a;
            border-radius: 5px;
        }
        .round {
            display: flex;
            flex-direction: column;
            gap: 40px;
            min-width: 150px;
            padding: 10px;
            background-color: #2a2a2a;
            border-radius: 5px;
        }
        .round.round1 {
            gap: 5px;
        }
        .round.round2plus {
            gap: 5px;
            align-content: flex-start;
        }
        .round-title {
            font-weight: bold;
            text-align: center;
            color: #FFC107;
            font-size: 14px;
            border-bottom: 2px solid #555;
            padding-bottom: 10px;
            margin-bottom: 10px;
        }
        .team {
            padding: 2px 4px;
            background-color: #333333;
            border: 1px solid #555;
            border-radius: 3px;
            font-size: 13px;
            white-space: nowrap;
            color: #ffffff;
        }
        .team.group1  { background-color: #1e3a8a; }
        .team.group2  { background-color: #7f1d1d; }
        .team.group3  { background-color: #1e3a8a; }
        .team.group4  { background-color: #7f1d1d; }
        .team.group5  { background-color: #1e3a8a; }
        .team.group6  { background-color: #7f1d1d; }
        .team.group7  { background-color: #1e5a3a; }
        .team.group8  { background-color: #3a1e5a; }
        .team.group9  { background-color: #1e3a8a; }
        .team.group10 { background-color: #7f1d1d; }
        .team.group11 { background-color: #1e5a3a; }
        .team.group12 { background-color: #3a1e5a; }
        .team.group13 { background-color: #1e3a8a; }
        .team.group14 { background-color: #7f1d1d; }
        .team.group15 { background-color: #1e5a3a; }
        .team.group16 { background-color: #3a1e5a; }
        .team.group17 { background-color: #1e5a3a; }
        .team.group18 { background-color: #3a1e5a; }
        .team:hover   { background-color: #444444; }
    </style>
    <div class="bracket">
    """

    round_names = {
        1: "Round 1",
        2: "Round 2",
        3: "Sweet 16",
        4: "Elite 8",
        5: "Final 4",
        6: "Championship",
    }

    for round_num in sorted(df['Round'].unique()):
        round_data = df[df['Round'] == round_num].sort_values('Game')

        if round_num == 1:
            # Split Round 1 into West (first 16) and East (second 16)
            html += "<div style='display: flex; gap: 20px;'>"

            html += "<div class='round round1'>"
            html += "<div class='round-title'>Round 1 West</div>"
            for idx in range(min(16, len(round_data))):
                row = round_data.iloc[idx]
                group = (idx // 8) % 4 + 1
                html += f"<div class='team group{group}'>({int(row['PWSeed'])}) {row['PWTeam']}</div>"
            html += "</div>"

            html += "<div class='round round1'>"
            html += "<div class='round-title'>Round 1 East</div>"
            for idx in range(16, min(32, len(round_data))):
                row = round_data.iloc[idx]
                group = ((idx - 16) // 8) % 2 + 17
                html += f"<div class='team group{group}'>({int(row['PWSeed'])}) {row['PWTeam']}</div>"
            html += "</div>"

            html += "</div>"

        else:
            round_class = "round round2plus" if round_num >= 2 else "round"
            html += f"<div class='{round_class}'>"
            html += f"<div class='round-title'>{round_names.get(round_num, f'Round {round_num}')}</div>"

            for idx, (_, row) in enumerate(round_data.iterrows()):
                if round_num == 2:
                    group = (idx // 4) % 4 + 5
                elif round_num == 3:
                    group = (idx // 2) % 4 + 9
                elif round_num == 4:
                    group = idx % 4 + 13
                else:
                    group = 1  # Final 4 / Championship use default

                html += f"<div class='team group{group}'>({int(row['PWSeed'])}) {row['PWTeam']}</div>"

            html += "</div>"

    html += "</div>"
    return html


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

py = 2026
st.markdown('Predicting ' + str(py))
variation = st.sidebar.slider('Randomness (Points)', 0, 25, 11)
st.sidebar.button('Run New Simulation')

fup = pd.read_csv("Python_Madness_2026/data/step05g_FUStats.csv").fillna(0)
fup['Year'] = pd.to_numeric(fup['Year'], errors='coerce').astype('Int32')
fup = fup[fup['Game'] >= 1]
fup['Round'] = fup['Round'].astype('int32')

fup = fup.rename(columns={
    'AFSeed': 'PFSeed', 'AFTeam': 'PFTeam', 'AFScore': 'PFScore',
    'AUSeed': 'PUSeed', 'AUTeam': 'PUTeam', 'AUScore': 'PUScore',
})
fup = fup.drop(columns=[c for c in ['AFSeed','AFTeam','AFScore','AUSeed','AUTeam','AUScore'] if c in fup.columns])

# ---------------------------------------------------------------------------
# Train linear models
# ---------------------------------------------------------------------------
fupn = fup.select_dtypes(exclude=['object'])
train = fupn[fupn['Year'] <= py]
xcol = train.drop(columns=['PFScore', 'PUScore']).columns

LRF = LinearRegression().fit(train[xcol], train['PFScore'])
LRU = LinearRegression().fit(train[xcol], train['PUScore'])

# ---------------------------------------------------------------------------
# Load Round 1 bracket
# ---------------------------------------------------------------------------
BB = pd.read_csv('Python_Madness_2026/data/step05c_FUHistory.csv')
BB = BB[(BB['Year'] == py) & (BB['Game'] >= 1) & (BB['Game'] <= 32)].copy()
BB['Round'] = BB['Round'].astype('int32')
BB.index = BB['Game']
BB = BB.iloc[:, 0:10]
BB.columns = ['Year', 'Round', 'Region', 'Game', 'PFSeed', 'PFTeam', 'PFScore', 'PUSeed', 'PUTeam', 'PUScore']
BB['PFScore'] = BB['PFScore'].astype(float)
BB['PUScore'] = BB['PUScore'].astype(float)
BB['Year'] = pd.to_numeric(BB['Year'], errors='coerce').astype('Int64')

KBBP = pd.read_csv("Python_Madness_2026/data/step05f_AllStats.csv").fillna(0)
KBBP = KBBP[KBBP['Year'] == py]

if BB.empty:
    st.warning(f"No Round 1 games found for {py}. Please ensure data is updated.")
    st.stop()


# ---------------------------------------------------------------------------
# Helper: predict + assign winners for a given set of games
# ---------------------------------------------------------------------------

def predict_and_assign(BB, game_range, BBstats, n_games):
    pfs = LRF.predict(BBstats[xcol]) + [random.randint(-variation, variation) for _ in range(n_games)]
    pus = LRU.predict(BBstats[xcol])
    offset = game_range[0]
    for i, x in enumerate(game_range):
        BB.loc[x, 'PFScore'] = pfs[i]
        BB.loc[x, 'PUScore'] = pus[i]
        BB.loc[x, 'PWSeed'] = np.where(pfs[i] >= pus[i], BB.loc[x, 'PFSeed'], BB.loc[x, 'PUSeed'])
        BB.loc[x, 'PWTeam'] = str(np.where(pfs[i] >= pus[i], BB.loc[x, 'PFTeam'], BB.loc[x, 'PUTeam']))
    return BB


def setup_next_round(BB, game_range, round_num):
    """Populate matchup rows for a new round using the previous round's winners."""
    for x in game_range:
        prev1 = (x - 32) * 2 - 1
        prev2 = (x - 32) * 2
        BB.loc[x, 'Year']   = py
        BB.loc[x, 'Round']  = round_num
        BB.loc[x, 'Game']   = x
        seed1 = BB.loc[prev1, 'PWSeed']
        seed2 = BB.loc[prev2, 'PWSeed']
        BB.loc[x, 'PFSeed']  = min(seed1, seed2)
        BB.loc[x, 'PUSeed']  = max(seed1, seed2)
        BB.loc[x, 'PFTeam']  = str(np.where(seed1 < seed2, BB.loc[prev1, 'PWTeam'], BB.loc[prev2, 'PWTeam']))
        BB.loc[x, 'PUTeam']  = str(np.where(seed1 > seed2, BB.loc[prev1, 'PWTeam'], BB.loc[prev2, 'PWTeam']))
        BB.loc[x, 'Region']  = BB.loc[prev2, 'Region']
    return BB


def merge_stats(BB, round_num):
    s = BB[BB['Round'] == round_num].merge(KBBP, left_on=['Year', 'PFTeam'], right_on=['Year', 'Team'], how='left')
    s = s.merge(KBBP, left_on=['Year', 'PUTeam'], right_on=['Year', 'Team'], how='left')
    return s


# ---------------------------------------------------------------------------
# Round 1
# ---------------------------------------------------------------------------
r1stats = BB.merge(KBBP, left_on=['Year', 'PFTeam'], right_on=['Year', 'Team'], how='left')
r1stats = r1stats.merge(KBBP, left_on=['Year', 'PUTeam'], right_on=['Year', 'Team'], how='left')
BB = predict_and_assign(BB, range(1, 33), r1stats, 32)

# ---------------------------------------------------------------------------
# Rounds 2–4  (game ranges follow the original index arithmetic)
# ---------------------------------------------------------------------------
round_config = [
    (range(33, 49), 2, 16),
    (range(49, 57), 3,  8),
    (range(57, 61), 4,  4),
]

for game_range, round_num, n in round_config:
    BB = setup_next_round(BB, game_range, round_num)
    BB = predict_and_assign(BB, game_range, merge_stats(BB, round_num), n)

# ---------------------------------------------------------------------------
# Round 5 – Final Four (seed assignment differs; no strict favorite/underdog)
# ---------------------------------------------------------------------------
for x in range(61, 63):
    prev1 = (x - 32) * 2 - 1
    prev2 = (x - 32) * 2
    BB.loc[x, 'Year']   = py
    BB.loc[x, 'Round']  = 5
    BB.loc[x, 'Game']   = x
    BB.loc[x, 'PFSeed'] = BB.loc[prev1, 'PWSeed']
    BB.loc[x, 'PUSeed'] = BB.loc[prev2, 'PWSeed']
    BB.loc[x, 'PFTeam'] = BB.loc[prev1, 'PWTeam']
    BB.loc[x, 'PUTeam'] = BB.loc[prev2, 'PWTeam']
BB.loc[61, 'Region'] = 'West'
BB.loc[62, 'Region'] = 'East'
BB = predict_and_assign(BB, range(61, 63), merge_stats(BB, 5), 2)

# ---------------------------------------------------------------------------
# Round 6 – Championship
# ---------------------------------------------------------------------------
x = 63
BB.loc[x, 'Year']   = py
BB.loc[x, 'Round']  = 6
BB.loc[x, 'Game']   = x
BB.loc[x, 'PFSeed'] = BB.loc[61, 'PWSeed']
BB.loc[x, 'PUSeed'] = BB.loc[62, 'PWSeed']
BB.loc[x, 'PFTeam'] = BB.loc[61, 'PWTeam']
BB.loc[x, 'PUTeam'] = BB.loc[62, 'PWTeam']
BB.loc[x, 'Region'] = 'Champ'
BB = predict_and_assign(BB, range(63, 64), merge_stats(BB, 6), 1)

# ---------------------------------------------------------------------------
# Final type cleanup
# ---------------------------------------------------------------------------
for col in ['Year', 'Round', 'Game', 'PFSeed', 'PUSeed', 'PWSeed']:
    BB[col] = BB[col].astype(int)
BB['Year'] = BB['Year'].astype(str)

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
# Reset index so 'Game' exists only as a column, not also as the index name
BB = BB.reset_index(drop=True)

st.markdown(highlight_html(BB), unsafe_allow_html=True)
st.dataframe(BB[BB['Game'] <= 63], height=500)