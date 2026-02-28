import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
import plotly.express as px
import os

def run():
    st.title('Variation Optimization Tool')
    st.markdown("""
    This tool runs simulations across past tournaments to find the optimal **Randomness (Points)** setting.
    It trains the model on all other years, then simulates the selected year multiple times with different variation levels.
    """)

    # --- 1. Load Data ---
    @st.cache_data
    def load_data():
        base_path = os.path.dirname(__file__)
        data_path = os.path.join(base_path, "..", "data", "step05g_FUStats.csv")
        stats_path = os.path.join(base_path, "..", "data", "step05f_AllStats.csv")
        
        if not os.path.exists(data_path) or not os.path.exists(stats_path):
            st.error("Data files not found.")
            return pd.DataFrame(), pd.DataFrame()

        df = pd.read_csv(data_path).fillna(0)
        if 'Game' in df.columns and 'Year' in df.columns:
             df = df.drop_duplicates(subset=['Year', 'Game'])
        
        # Determine Actual Winner for scoring later
        # (Assuming AFScore/AUScore are the actual final scores in the CSV)
        if 'AFScore' in df.columns and 'AUScore' in df.columns:
            df['Actual_Winner'] = np.where(df['AFScore'] >= df['AUScore'], df['AFTeam'], df['AUTeam'])

        # Rename columns to match the prediction model's expected format
        rename_map = {
            'AFSeed': 'PFSeed', 'AUSeed': 'PUSeed', 
            'AFTeam': 'PFTeam', 'AUTeam': 'PUTeam',
            'AFScore': 'PFScore', 'AUScore': 'PUScore'
        }
        df = df.rename(columns=rename_map)
        
        all_stats = pd.read_csv(stats_path).fillna(0)
        return df, all_stats

    fup, all_stats = load_data()
    if fup.empty: return

    # --- 2. Feature Engineering ---
    def create_advanced_features(df):
        # Calculate scoring margin if stats are present
        if 'Pts_x' in df.columns and 'Pts_y' in df.columns:
            df['scoring_margin'] = df['Pts_x'] - df['Pts_y']
        else:
            df['scoring_margin'] = 0
            
        df['seed_diff'] = abs(df['PFSeed'] - df['PUSeed'])
        df['higher_seed'] = np.minimum(df['PFSeed'], df['PUSeed'])
        return df

    fup = create_advanced_features(fup)

    # --- 3. UI Controls ---
    # Get available years that have data
    years_available = sorted(fup[fup['PFScore'] > 0]['Year'].unique().tolist())
    # Filter out 2020
    years_available = [y for y in years_available if y != 2020]
    
    col1, col2 = st.columns(2)
    with col1:
        selected_years = st.multiselect(
            "Select Years to Backtest", 
            years_available, 
            default=[y for y in years_available if y >= 2021][-3:] # Default to last 3 played years
        )
    with col2:
        max_variation = st.slider("Max Variation to Test", 0, 30, 20)
        sims_per_setting = st.slider("Simulations per Setting", 1, 50, 10, help="More simulations = more accurate average, but slower.")
    
    if st.button("Run Optimization"):
        if not selected_years:
            st.warning("Please select at least one year.")
            return

        results = []
        
        # Progress bar setup
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_steps = len(selected_years) * (max_variation + 1)
        current_step = 0

        for year in selected_years:
            status_text.text(f"Processing {year}...")
            
            # --- Train Model (Leave-One-Out) ---
            # Train on all years EXCEPT the target year, and only on played games
            train_df = fup[(fup['Year'] != year) & (fup['PFScore'] + fup['PUScore'] > 0)].select_dtypes(exclude=['object'])
            
            drop_list = ['PFScore', 'PUScore', 'Year', 'Round', 'Game', 'AWTeam', 'Fti', 'Uti', 'Actual_Winner', 'ESPN_Score']
            garbage_cols = [c for c in train_df.columns if 'Unnamed' in c or 'Record' in c or 'Team.1' in c]
            
            X = train_df.drop(columns=[c for c in drop_list + garbage_cols if c in train_df.columns])
            features = X.columns
            y_fav = train_df['PFScore']
            y_und = train_df['PUScore']

            # Ensemble Model
            model_f = VotingRegressor([('lr', LinearRegression()), ('gbm', GradientBoostingRegressor(n_estimators=50, random_state=42))])
            model_f.fit(X, y_fav)
            
            model_u = VotingRegressor([('lr', LinearRegression()), ('gbm', GradientBoostingRegressor(n_estimators=50, random_state=42))])
            model_u.fit(X, y_und)

            # --- Prepare Simulation Data ---
            actual_winners = fup[fup['Year'] == year].set_index('Game')['Actual_Winner'].to_dict()
            base_bracket = fup[(fup['Year'] == year) & (fup['Round'] == 1)].copy()
            
            # Pre-filter stats for this year to speed up merging
            year_stats = all_stats[all_stats['Year'] == year].copy()
            if 'Year' in year_stats.columns: year_stats = year_stats.drop(columns=['Year'])
            year_stats = year_stats.drop_duplicates(subset=['Team'])
            stat_cols = [c for c in year_stats.columns if c != 'Team']

            # --- Loop Variations ---
            for v in range(max_variation + 1):
                scores = []
                
                for _ in range(sims_per_setting):
                    # --- Run Single Bracket Simulation ---
                    current_round_games = base_bracket.copy()
                    full_bracket_preds = []

                    # Predict Round 1
                    r1_X = current_round_games[features]
                    p_fav = model_f.predict(r1_X)
                    p_und = model_u.predict(r1_X)
                    
                    # Apply Variation (Randomness)
                    if v > 0:
                        p_fav += np.random.randint(-v, v + 1, size=len(p_fav))
                    
                    # Determine Winners
                    winner_mask = (p_fav >= p_und)
                    current_round_games['PWSeed'] = np.where(winner_mask, current_round_games['PFSeed'], current_round_games['PUSeed'])
                    current_round_games['PWTeam'] = np.where(winner_mask, current_round_games['PFTeam'], current_round_games['PUTeam'])
                    current_round_games['Round'] = 1 # Ensure round is set
                    
                    full_bracket_preds.append(current_round_games[['Game', 'Round', 'PWTeam']])
                    
                    prev_round_winners = current_round_games
                    
                    # Predict Rounds 2-6
                    for r in range(2, 7):
                        prev_round_winners = prev_round_winners.sort_values('Game')
                        winners_list = prev_round_winners.to_dict('records')
                        new_games = []
                        
                        # Map round to starting game index
                        start_game_map = {2:33, 3:49, 4:57, 5:61, 6:63}
                        game_idx = start_game_map[r]
                        
                        # Pair winners
                        for i in range(0, len(winners_list), 2):
                            if i+1 >= len(winners_list): break
                            g1, g2 = winners_list[i], winners_list[i+1]
                            
                            # Favored is lower seed
                            if g1['PWSeed'] <= g2['PWSeed']:
                                pf, pu = g1, g2
                            else:
                                pf, pu = g2, g1
                            
                            new_games.append({
                                'Year': year, 'Round': r, 'Game': game_idx,
                                'PFSeed': pf['PWSeed'], 'PFTeam': pf['PWTeam'],
                                'PUSeed': pu['PWSeed'], 'PUTeam': pu['PWTeam']
                            })
                            game_idx += 1
                        
                        if not new_games: break
                        
                        next_df = pd.DataFrame(new_games)
                        
                        # Merge Stats for new matchups
                        next_df = next_df.merge(year_stats, left_on='PFTeam', right_on='Team', how='left')
                        rename_x = {c: f"{c}_x" for c in stat_cols}
                        next_df = next_df.rename(columns=rename_x).drop(columns=['Team'], errors='ignore')
                        
                        next_df = next_df.merge(year_stats, left_on='PUTeam', right_on='Team', how='left')
                        rename_y = {c: f"{c}_y" for c in stat_cols}
                        next_df = next_df.rename(columns=rename_y).drop(columns=['Team'], errors='ignore')
                        
                        next_df = create_advanced_features(next_df)
                        
                        # Fill missing cols with 0
                        missing = [c for c in features if c not in next_df.columns]
                        if missing:
                            next_df = pd.concat([next_df, pd.DataFrame(0, index=next_df.index, columns=missing)], axis=1)
                            
                        # Predict
                        r_X = next_df[features]
                        p_f = model_f.predict(r_X)
                        p_u = model_u.predict(r_X)
                        
                        if v > 0:
                            p_f += np.random.randint(-v, v + 1, size=len(p_f))
                        
                        w_mask = (p_f >= p_u)
                        next_df['PWSeed'] = np.where(w_mask, next_df['PFSeed'], next_df['PUSeed'])
                        next_df['PWTeam'] = np.where(w_mask, next_df['PFTeam'], next_df['PUTeam'])
                        
                        full_bracket_preds.append(next_df[['Game', 'Round', 'PWTeam']])
                        prev_round_winners = next_df
                    
                    # --- Calculate Score ---
                    total_score = 0
                    all_preds = pd.concat(full_bracket_preds)
                    for _, row in all_preds.iterrows():
                        gid = row['Game']
                        pred_winner = row['PWTeam']
                        act_winner = actual_winners.get(gid)
                        rnd = row['Round']
                        
                        if act_winner and pred_winner == act_winner:
                            total_score += 10 * (2 ** (rnd - 1))
                    
                    scores.append(total_score)
                
                # Average score for this variation/year
                avg_score = sum(scores) / len(scores)
                results.append({'Year': year, 'Variation': v, 'AvgScore': avg_score})
                
                current_step += 1
                progress_bar.progress(current_step / total_steps)
        
        status_text.text("Optimization Complete!")
        
        # --- 4. Display Results ---
        res_df = pd.DataFrame(results)
        
        st.divider()
        st.subheader("Results Analysis")
        
        # Aggregate by Variation across all selected years
        agg_res = res_df.groupby('Variation')['AvgScore'].mean().reset_index()
        
        # Find optimal
        best_row = agg_res.loc[agg_res['AvgScore'].idxmax()]
        best_v = int(best_row['Variation'])
        best_score = best_row['AvgScore']
        
        st.success(f"**Optimal Variation:** {best_v} points (Avg Score: {best_score:.1f})")
        
        # Plot
        fig = px.line(agg_res, x='Variation', y='AvgScore', 
                      title=f"Average Bracket Score vs. Randomness (Years: {', '.join(map(str, selected_years))})",
                      markers=True, labels={'AvgScore': 'Average ESPN Score', 'Variation': 'Randomness (+/- Points)'})
        
        # Add a vertical line for the optimal
        fig.add_vline(x=best_v, line_dash="dash", line_color="green", annotation_text="Optimal")
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("View Detailed Data"):
            st.dataframe(res_df)

run()
