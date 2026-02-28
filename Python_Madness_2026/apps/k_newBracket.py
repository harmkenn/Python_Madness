import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
import os
import webbrowser

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
        .team.group1 {
            background-color: #1e3a8a;
        }
        .team.group2 {
            background-color: #7f1d1d;
        }
        .team.group3 {
            background-color: #1e3a8a;
        }
        .team.group4 {
            background-color: #7f1d1d;
        }
        .team.group5 {
            background-color: #1e3a8a;
        }
        .team.group6 {
            background-color: #7f1d1d;
        }
        .team.group7 {
            background-color: #1e5a3a;
        }
        .team.group8 {
            background-color: #3a1e5a;
        }
        .team.group9 {
            background-color: #1e3a8a;
        }
        .team.group10 {
            background-color: #7f1d1d;
        }
        .team.group11 {
            background-color: #1e5a3a;
        }
        .team.group12 {
            background-color: #3a1e5a;
        }
        .team.group13 {
            background-color: #1e3a8a;
        }
        .team.group14 {
            background-color: #7f1d1d;
        }
        .team.group15 {
            background-color: #1e5a3a;
        }
        .team.group16 {
            background-color: #3a1e5a;
        }
        .team.group17 {
            background-color: #1e5a3a;
        }
        .team.group18 {
            background-color: #3a1e5a;
        }
        .team:hover {
            background-color: #444444;
        }
    </style>
    <div class="bracket">
    """
    
    # Group by round
    for round_num in sorted(df['Round'].unique()):
        round_data = df[df['Round'] == round_num].sort_values('Game')
        
        # Round names
        round_names = {
            1: "Round 1",
            2: "Round 2", 
            3: "Sweet 16",
            4: "Elite 8",
            5: "Final 4",
            6: "Championship"
        }
        
        # Special handling for Round 1 - split into West and East
        if round_num == 1:
            html += f"<div style='display: flex; gap: 20px;'>"
            
            # Round 1 West (first 16 teams)
            html += f"<div class='round round1'>"
            html += f"<div class='round-title'>Round 1 West</div>"
            for idx in range(min(16, len(round_data))):
                row = round_data.iloc[idx]
                seed = int(row['PWSeed'])
                team = row['PWTeam']
                team_display = f"({seed}) {team}"
                group = (idx // 8) % 4
                team_class = f"team group{group + 1}"
                html += f"<div class='{team_class}'>{team_display}</div>"
            html += "</div>"
            
            # Round 1 East (second 16 teams)
            html += f"<div class='round round1'>"
            html += f"<div class='round-title'>Round 1 East</div>"
            for idx in range(16, min(32, len(round_data))):
                row = round_data.iloc[idx]
                seed = int(row['PWSeed'])
                team = row['PWTeam']
                team_display = f"({seed}) {team}"
                group = ((idx - 16) // 8) % 2
                team_class = f"team group{17 + group}"
                html += f"<div class='{team_class}'>{team_display}</div>"
            html += "</div>"
            
            html += "</div>"
        else:
            round_class = "round round2plus" if round_num >= 2 else "round"
            html += f"<div class='{round_class}'>"
            html += f"<div class='round-title'>{round_names.get(round_num, f'Round {round_num}')}</div>"
            
            for idx, (_, row) in enumerate(round_data.iterrows()):
                seed = int(row['PWSeed'])
                team = row['PWTeam']
                team_display = f"({seed}) {team}"
                
                # Alternate colors for each round
                team_class = "team"
                if round_num == 2:
                    group = (idx // 4) % 4
                    team_class = f"team group{group + 5}"
                elif round_num == 3:
                    group = (idx // 2) % 4
                    team_class = f"team group{group + 9}"
                elif round_num == 4:
                    group = idx % 4
                    team_class = f"team group{group + 13}"
                
                html += f"<div class='{team_class}'>{team_display}</div>"
            
            html += "</div>"
    
    html += "</div>"
    return html


def run():
    st.title('New Bracket Maker (Ensemble Model)')

    # --- 1. SECURE DATA LOADING (from j_newprediction.py) ---
    @st.cache_data
    def load_data():
        base_path = os.path.dirname(__file__)
        data_path = os.path.join(base_path, "..", "data", "step05g_FUStats.csv")
        stats_path = os.path.join(base_path, "..", "data", "step05f_AllStats.csv")
        
        if not os.path.exists(data_path) or not os.path.exists(stats_path):
            st.error(f"Error: Data files not found.")
            return pd.DataFrame(), pd.DataFrame()

        df = pd.read_csv(data_path).fillna(0)
        if 'Game' in df.columns and 'Year' in df.columns:
             df = df.drop_duplicates(subset=['Year', 'Game'])
        all_stats = pd.read_csv(stats_path).fillna(0)
        
        rename_map = {
            'AFSeed': 'PFSeed', 'AUSeed': 'PUSeed', 
            'AFTeam': 'PFTeam', 'AUTeam': 'PUTeam',
            'AFScore': 'PFScore', 'AUScore': 'PUScore'
        }
        df = df.rename(columns=rename_map)
        
        df['Year'] = df['Year'].astype(int)
        df['Round'] = df['Round'].astype(int)
        df['PFScore'] = df['PFScore'].astype(float)
        df['PUScore'] = df['PUScore'].astype(float)
        return df, all_stats

    fup, all_stats = load_data()
    if fup.empty:
        return

    # --- 2. FEATURE ENGINEERING (from j_newprediction.py) ---
    def create_advanced_features(df):
        if 'Pts_x' in df.columns and 'Pts_y' in df.columns:
            df['scoring_margin'] = df['Pts_x'] - df['Pts_y']
        else:
            df['scoring_margin'] = 0
            
        df['seed_diff'] = abs(df['PFSeed'] - df['PUSeed'])
        df['higher_seed'] = np.minimum(df['PFSeed'], df['PUSeed'])
        return df

    fup = create_advanced_features(fup)

    # --- 3. MODEL TRAINING (from j_newprediction.py) ---
    py = 2026
    st.sidebar.button("Run New Simulation")
    st.markdown('Predicting ' + str(py))
    variation = st.sidebar.slider('Randomness (Points)', 0, 25, 11)

    @st.cache_resource
    def train_ensemble_models(_df, year):
        train_df = _df[_df['Year'] != year].select_dtypes(exclude=['object'])
        
        drop_list = ['PFScore', 'PUScore', 'Year', 'Round', 'Game', 'AWTeam', 'Fti', 'Uti', 'Actual_Winner', 'ESPN_Score']
        garbage_cols = [c for c in train_df.columns if 'Unnamed' in c or 'Record' in c or 'Team.1' in c]
        
        X = train_df.drop(columns=[c for c in drop_list + garbage_cols if c in train_df.columns])
        features = X.columns
        y_fav = train_df['PFScore']
        y_und = train_df['PUScore']

        model_f = VotingRegressor([('lr', LinearRegression()), ('gbm', GradientBoostingRegressor(n_estimators=100, random_state=42))])
        model_f.fit(X, y_fav)
        
        model_u = VotingRegressor([('lr', LinearRegression()), ('gbm', GradientBoostingRegressor(n_estimators=100, random_state=42))])
        model_u.fit(X, y_und)
        
        return model_f, model_u, features

    if py == 2020:
        st.warning("2020 Tournament was cancelled.")
    else:
        with st.spinner("Training Ensemble Models..."):
            model_fav, model_und, xcol = train_ensemble_models(fup, py)

        # --- 4. BRACKET SIMULATION (Adapted from j_newprediction.py) ---
        base_path = os.path.dirname(__file__)
        history_path = os.path.join(base_path, "..", "data", "step05c_FUHistory.csv")
        
        if os.path.exists(history_path):
            hist_df = pd.read_csv(history_path)
            BB = hist_df[(hist_df['Year'] == py) & (hist_df['Round'] == 1) & (hist_df['Game'] >= 1) & (hist_df['Game'] <= 32)].copy()
            
            if BB.empty:
                 BB = fup[(fup['Year'] == py) & (fup['Round'] == 1)].copy()
            else:
                rename_map_hist = {
                    'AFSeed': 'PFSeed', 'AUSeed': 'PUSeed', 
                    'AFTeam': 'PFTeam', 'AUTeam': 'PUTeam',
                    'AFScore': 'PFScore', 'AUScore': 'PUScore'
                }
                BB = BB.rename(columns=rename_map_hist)
                BB['PFScore'] = BB['PFScore'].astype(float)
                BB['PUScore'] = BB['PUScore'].astype(float)
                
                year_stats = all_stats[all_stats['Year'] == py].copy()
                if 'Year' in year_stats.columns: year_stats = year_stats.drop(columns=['Year'])
                year_stats = year_stats.drop_duplicates(subset=['Team'])
                stat_cols = [c for c in year_stats.columns if c != 'Team']
                
                BB = BB.merge(year_stats, left_on='PFTeam', right_on='Team', how='left')
                rename_x = {c: f"{c}_x" for c in stat_cols}
                BB = BB.rename(columns=rename_x).drop(columns=['Team'], errors='ignore')
                
                BB = BB.merge(year_stats, left_on='PUTeam', right_on='Team', how='left')
                rename_y = {c: f"{c}_y" for c in stat_cols}
                BB = BB.rename(columns=rename_y).drop(columns=['Team'], errors='ignore')
                
                BB = create_advanced_features(BB)
                
                missing_cols = [c for c in xcol if c not in BB.columns]
                if missing_cols:
                    BB = pd.concat([BB, pd.DataFrame(0, index=BB.index, columns=missing_cols)], axis=1)
        else:
            BB = fup[(fup['Year'] == py) & (fup['Round'] == 1)].copy()
            
        BB = BB.sort_values('Game')

        def predict_round(round_num, bracket_df, model_f, model_u, features):
            mask = (bracket_df['Round'] == round_num)
            if not mask.any(): return bracket_df
            
            round_X = bracket_df.loc[mask, features]
            p_fav = model_f.predict(round_X)
            p_und = model_u.predict(round_X)
            
            # Add randomness to mimic i_bracketmaker.py (random integer between -11 and 11 added to favored score)
            p_fav += np.random.randint(-variation, variation + 1, size=len(p_fav))
            
            bracket_df.loc[mask, 'PFScore'] = p_fav
            bracket_df.loc[mask, 'PUScore'] = p_und
            
            winner_mask = (p_fav >= p_und)
            bracket_df.loc[mask, 'PWSeed'] = np.where(winner_mask, bracket_df.loc[mask, 'PFSeed'], bracket_df.loc[mask, 'PUSeed'])
            bracket_df.loc[mask, 'PWTeam'] = np.where(winner_mask, bracket_df.loc[mask, 'PFTeam'], bracket_df.loc[mask, 'PUTeam'])
            
            return bracket_df

        next_game_idx = 33
        for r in range(1, 7):
            BB = predict_round(r, BB, model_fav, model_und, xcol)
            
            if r < 6:
                winners = BB[BB['Round'] == r].sort_values('Game')
                next_gen = []
                for i in range(0, len(winners), 2):
                    if i + 1 < len(winners):
                        g1, g2 = winners.iloc[i], winners.iloc[i+1]
                        
                        if g1['PWSeed'] <= g2['PWSeed']:
                            pf_s, pf_t = g1['PWSeed'], g1['PWTeam']
                            pu_s, pu_t = g2['PWSeed'], g2['PWTeam']
                        else:
                            pf_s, pf_t = g2['PWSeed'], g2['PWTeam']
                            pu_s, pu_t = g1['PWSeed'], g1['PWTeam']
                            
                        next_gen.append({
                            'Year': py, 'Round': r+1,
                            'Game': next_game_idx,
                            'PFSeed': pf_s, 'PFTeam': pf_t,
                            'PUSeed': pu_s, 'PUTeam': pu_t
                        })
                        next_game_idx += 1
                
                if next_gen:
                    next_df = pd.DataFrame(next_gen)
                    year_stats = all_stats[all_stats['Year'] == py].copy()
                    if 'Year' in year_stats.columns: year_stats = year_stats.drop(columns=['Year'])
                    year_stats = year_stats.drop_duplicates(subset=['Team'])
                    stat_cols = [c for c in year_stats.columns if c != 'Team']
                    
                    next_df = next_df.merge(year_stats, left_on='PFTeam', right_on='Team', how='left')
                    rename_x = {c: f"{c}_x" for c in stat_cols}
                    next_df = next_df.rename(columns=rename_x).drop(columns=['Team'], errors='ignore')
                    
                    next_df = next_df.merge(year_stats, left_on='PUTeam', right_on='Team', how='left')
                    rename_y = {c: f"{c}_y" for c in stat_cols}
                    next_df = next_df.rename(columns=rename_y).drop(columns=['Team'], errors='ignore')
                    
                    next_df = create_advanced_features(next_df)
                    
                    missing_cols = [c for c in xcol if c not in next_df.columns]
                    if missing_cols:
                        next_df = pd.concat([next_df, pd.DataFrame(0, index=next_df.index, columns=missing_cols)], axis=1)
                        
                    BB = pd.concat([BB, next_df], ignore_index=True)

        # --- 5. VISUALIZATION (Mimicking i_bracketmaker.py) ---
        BB_display = BB.copy()
        BB_display['Game'] = BB_display['Game'].astype(int)
        BB_display = BB_display.sort_values('Game')
        
        # Ensure integer types for display
        cols_to_int = ['Round', 'PFSeed', 'PUSeed', 'PWSeed']
        for c in cols_to_int:
            if c in BB_display.columns:
                BB_display[c] = BB_display[c].fillna(0).astype(int)
        
        BB_display['Year'] = BB_display['Year'].astype(str)

        st.markdown(highlight_html(BB_display), unsafe_allow_html=True)
        st.dataframe(BB_display[BB_display.index <= 63], height=500)

run()