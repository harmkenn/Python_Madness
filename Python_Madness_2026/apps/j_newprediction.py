import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import os

# v2.0 - Random Forest & Full Stat Lookup
def run():
    st.title('NCAA Tournament Prediction Engine (Random Forest)')

    # --- 1. SECURE DATA LOADING ---
    @st.cache_data
    def load_data():
        # Get the path relative to THIS file's location
        base_path = os.path.dirname(__file__)
        data_path = os.path.join(base_path, "..", "data", "step05g_FUStats.csv")
        stats_path = os.path.join(base_path, "..", "data", "step05f_AllStats.csv")
        
        if not os.path.exists(data_path) or not os.path.exists(stats_path):
            st.error(f"Error: Data files not found.")
            return pd.DataFrame(), pd.DataFrame()

        df = pd.read_csv(data_path).fillna(0)
        all_stats = pd.read_csv(stats_path).fillna(0)
        
        # Create Actual Winner column for scoring before renaming
        if 'AFScore' in df.columns and 'AUScore' in df.columns:
            df['Actual_Winner'] = np.where(df['AFScore'] >= df['AUScore'], df['AFTeam'], df['AUTeam'])

        # Standardize names immediately to prevent KeyError: 'PFSeed'
        # We map "Actual" column names to "Predicted" column names
        rename_map = {
            'AFSeed': 'PFSeed', 'AUSeed': 'PUSeed', 
            'AFTeam': 'PFTeam', 'AUTeam': 'PUTeam',
            'AFScore': 'PFScore', 'AUScore': 'PUScore'
        }
        df = df.rename(columns=rename_map)
        
        # Ensure correct data types
        df['Year'] = df['Year'].astype(int)
        df['Round'] = df['Round'].astype(int)
        df['PFScore'] = df['PFScore'].astype(float)
        df['PUScore'] = df['PUScore'].astype(float)
        return df, all_stats

    fup, all_stats = load_data()
    if fup.empty:
        return

    # --- 2. FEATURE ENGINEERING ---
    def create_advanced_features(df):
        # Using .get() or checking existence to prevent crashes on generated next-rounds
        if 'Pts_x' in df.columns and 'Pts_y' in df.columns:
            df['scoring_margin'] = df['Pts_x'] - df['Pts_y']
        else:
            df['scoring_margin'] = 0
            
        df['seed_diff'] = abs(df['PFSeed'] - df['PUSeed'])
        df['higher_seed'] = np.minimum(df['PFSeed'], df['PUSeed'])
        return df

    fup = create_advanced_features(fup)

    # --- 3. MODEL TRAINING ---
    py = st.slider('Select Tournament Year: ', 2008, 2025, 2025)

    if py == 2020:
        st.warning("2020 Tournament was cancelled.")
    else:
        # Prepare training data (all years EXCEPT the one we are predicting)
        train_df = fup[fup['Year'] != py].select_dtypes(exclude=['object'])
        
        # Define features and targets
        # We drop anything that wouldn't be known BEFORE a game starts
        drop_list = ['PFScore', 'PUScore', 'Year', 'Round', 'Game', 'AWTeam', 'Fti', 'Uti', 'Actual_Winner', 'ESPN_Score']
        
        # Filter garbage columns
        garbage_cols = [c for c in train_df.columns if 'Unnamed' in c or 'Record' in c or 'Team.1' in c]
        
        X = train_df.drop(columns=[c for c in drop_list + garbage_cols if c in train_df.columns])
        xcol = X.columns
        y_fav = train_df['PFScore']
        y_und = train_df['PUScore']

        # Train Random Forest
        rf_fav = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_fav)
        rf_und = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_und)

        # --- 4. BRACKET SIMULATION & SCORING PREP ---
        # Create Actual Winners Lookup for Scoring
        actual_winners = fup[fup['Year'] == py].set_index('Game')['Actual_Winner'].to_dict()

        # Start with only Round 1 for the selected year
        BB = fup[(fup['Year'] == py) & (fup['Round'] == 1)].copy()
        BB = BB.sort_values('Game')

        def predict_round(round_num, bracket_df, model_f, model_u, features):
            mask = (bracket_df['Round'] == round_num)
            if not mask.any(): return bracket_df
            
            # Predict
            round_X = bracket_df.loc[mask, features]
            p_fav = model_f.predict(round_X)
            p_und = model_u.predict(round_X)
            
            # Assign results
            bracket_df.loc[mask, 'PFScore'] = p_fav
            bracket_df.loc[mask, 'PUScore'] = p_und
            
            # Determine winners
            winner_mask = (p_fav >= p_und)
            bracket_df.loc[mask, 'PWSeed'] = np.where(winner_mask, bracket_df.loc[mask, 'PFSeed'], bracket_df.loc[mask, 'PUSeed'])
            bracket_df.loc[mask, 'PWTeam'] = np.where(winner_mask, bracket_df.loc[mask, 'PFTeam'], bracket_df.loc[mask, 'PUTeam'])
            
            return bracket_df

        # Initialize Game Counter for next rounds
        next_game_idx = 33

        # Run 6 Rounds
        for r in range(1, 7):
            BB = predict_round(r, BB, rf_fav, rf_und, xcol)
            
            if r < 6:
                winners = BB[BB['Round'] == r].sort_values('Game')
                next_gen = []
                # Pair up winners for the next round
                for i in range(0, len(winners), 2):
                    if i + 1 < len(winners):
                        g1, g2 = winners.iloc[i], winners.iloc[i+1]
                        
                        # Set "Favored" as the better (lower) seed
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
                    
                    # --- CRITICAL FIX: Merge Stats for Next Round Teams ---
                    # Get stats for the current year
                    year_stats = all_stats[all_stats['Year'] == py].copy()
                    if 'Year' in year_stats.columns: year_stats = year_stats.drop(columns=['Year'])
                    
                    # Identify stat columns (everything except Team)
                    stat_cols = [c for c in year_stats.columns if c != 'Team']
                    
                    # Merge for PFTeam (Favored) -> suffix _x
                    next_df = next_df.merge(year_stats, left_on='PFTeam', right_on='Team', how='left')
                    rename_x = {c: f"{c}_x" for c in stat_cols}
                    next_df = next_df.rename(columns=rename_x).drop(columns=['Team'], errors='ignore')
                    
                    # Merge for PUTeam (Underdog) -> suffix _y
                    next_df = next_df.merge(year_stats, left_on='PUTeam', right_on='Team', how='left')
                    rename_y = {c: f"{c}_y" for c in stat_cols}
                    next_df = next_df.rename(columns=rename_y).drop(columns=['Team'], errors='ignore')
                    
                    next_df = create_advanced_features(next_df)
                    
                    # Fill missing feature columns with 0 for the model
                    missing_cols = [c for c in xcol if c not in next_df.columns]
                    if missing_cols:
                        next_df = pd.concat([next_df, pd.DataFrame(0, index=next_df.index, columns=missing_cols)], axis=1)
                        
                    BB = pd.concat([BB, next_df], ignore_index=True)

        # --- 5. SCORING ---
        def calculate_score(row):
            game_id = row.get('Game')
            predicted = row.get('PWTeam')
            actual = actual_winners.get(game_id)
            round_num = row.get('Round')
            
            if predicted == actual and actual:
                return 10 * (2 ** (round_num - 1))
            return 0

        BB['ESPN_Score'] = BB.apply(calculate_score, axis=1)
        total_score = BB['ESPN_Score'].sum()

        # --- 6. VISUALIZATION ---
        st.subheader(f"Final Prediction for {py}")
        st.metric("Total ESPN Bracket Score", f"{int(total_score)}")
        st.dataframe(BB[['Round', 'Game', 'PFTeam', 'PFSeed', 'PFScore', 'PUTeam', 'PUSeed', 'PUScore', 'PWTeam', 'ESPN_Score']], width="stretch")
        
        fig = px.scatter(BB, x="PFScore", y="PUScore", color="Round", hover_data=["PFTeam", "PUTeam"],
                         title="Matchup Intensity: Favored vs Underdog Predicted Scores")
        st.plotly_chart(fig)

# Streamlit apps often use a 'run' pattern when called from a main file
run()