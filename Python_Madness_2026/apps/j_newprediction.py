import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import plotly.express as px

# Title of the app
st.markdown('Use Enhanced Machine Learning to Predict NCAA Tournament Outcomes')

# Load data
fup = pd.read_csv("Python_Madness_2026/data/step05g_FUStats.csv").fillna(0)
fup = fup[fup['Year'] <= 2025][fup['Game'] >= 1]
fup['Round'] = fup['Round'].astype('int32')

st.write(coulumns(fup))

# Feature engineering
def create_advanced_features(df):
    df['scoring_margin'] = df['points_per_game'] - df['opp_points_per_game']
    df['efficiency_ratio'] = df['field_goal_pct'] / df['opp_field_goal_pct']
    df['turnover_ratio'] = df['opp_turnovers'] / df['turnovers']
    df['strength_of_schedule_adj'] = df['sos'] * df['wins'] / df['games_played']
    return df

fup = create_advanced_features(fup)

# Add seed-based features
def add_seed_features(matchup_data):
    matchup_data['seed_diff'] = abs(matchup_data['PFSeed'] - matchup_data['PUSeed'])
    matchup_data['higher_seed'] = np.minimum(matchup_data['PFSeed'], matchup_data['PUSeed'])
    return matchup_data

fup = add_seed_features(fup)

# Year selection slider
py = st.slider('Year: ', 2008, 2025)

if py == 2020:
    st.markdown("No Bracket in 2020")
else:
    # Prepare training data
    fupn = fup.select_dtypes(exclude=['object'])
    MX = fupn[fupn['Year'] != py].drop(['AFScore', 'AUScore', 'AFSeed', 'AUSeed', 'PFScore', 'PUScore', 'Fti', 'Uti'], axis=1)
    xcol = MX.columns
    MFY = fupn[fupn['Year'] != py]['PFScore']
    MUY = fupn[fupn['Year'] != py]['PUScore']

    # Train Random Forest models
    LRF = RandomForestRegressor(n_estimators=100, random_state=42)
    RFU = RandomForestRegressor(n_estimators=100, random_state=42)
    LRF.fit(MX, MFY)
    RFU.fit(MX, MUY)

    # Cross-validation for model evaluation
    cv_scores = cross_val_score(LRF, MX, MFY, cv=5, scoring='neg_mean_squared_error')
    st.write(f"Cross-Validation RMSE: {np.sqrt(-cv_scores.mean()):.2f}")

    # Initialize bracket for the selected year
    BB = fup[fup['Year'] == py]
    BB = BB.iloc[:, 0:10]
    BB.index = BB.Game

    # Function to predict a single round
    def predict_round(round_num, BB, LRF, RFU, xcol):
        round_games = BB[BB['Round'] == round_num]
        pfs = LRF.predict(round_games[xcol])
        pus = RFU.predict(round_games[xcol])

        for x in round_games.index:
            BB.loc[x, 'PFScore'] = pfs[x - round_games.index[0]]
            BB.loc[x, 'PUScore'] = pus[x - round_games.index[0]]
            BB.loc[x, 'PWSeed'] = np.where(BB.loc[x, 'PFScore'] >= BB.loc[x, 'PUScore'], BB.loc[x, 'PFSeed'], BB.loc[x, 'PUSeed'])
            BB.loc[x, 'PWTeam'] = str(np.where(BB.loc[x, 'PFScore'] >= BB.loc[x, 'PUScore'], BB.loc[x, 'PFTeam'], BB.loc[x, 'PUTeam']))
            BB.loc[x, 'ESPN'] = np.where(BB.loc[x, 'AWTeam'] == BB.loc[x, 'PWTeam'], 10 * (2 ** (round_num - 1)), 0)

        return BB

    # Predict all rounds dynamically
    for round_num in range(1, 7):  # NCAA tournament has 6 rounds
        BB = predict_round(round_num, BB, LRF, RFU, xcol)

        # Prepare matchups for the next round
        if round_num < 6:  # Skip for the final round
            next_round_games = []
            for i in range(0, len(BB[BB['Round'] == round_num]), 2):
                game1 = BB.iloc[i]
                game2 = BB.iloc[i + 1]
                next_game = {
                    'Year': py,
                    'Round': round_num + 1,
                    'PFSeed': min(game1['PWSeed'], game2['PWSeed']),
                    'PUSeed': max(game1['PWSeed'], game2['PWSeed']),
                    'PFTeam': game1['PWTeam'] if game1['PWSeed'] < game2['PWSeed'] else game2['PWTeam'],
                    'PUTeam': game1['PWTeam'] if game1['PWSeed'] > game2['PWSeed'] else game2['PWTeam']
                }
                next_round_games.append(next_game)

            next_round_df = pd.DataFrame(next_round_games)
            BB = pd.concat([BB, next_round_df], ignore_index=True)

    # Visualization of predictions
    def plot_prediction_confidence(team_stats, predictions):
        fig = px.scatter(x=team_stats['seed_diff'], y=predictions, title='Predicted Performance by Seed Differential')
        st.plotly_chart(fig)

    plot_prediction_confidence(BB[BB['Round'] == 1], BB[BB['Round'] == 1]['PFScore'])

    # Validation metrics
    def calculate_bracket_metrics(predictions, actuals):
        metrics = {
            'correct_picks': (predictions == actuals).sum(),
            'total_games': len(predictions),
            'accuracy': (predictions == actuals).mean(),
            'upset_prediction_rate': ((predictions > actuals) & (actuals != predictions)).mean()
        }
        return metrics

    metrics = calculate_bracket_metrics(BB['PWTeam'], BB['AWTeam'])
    st.write(metrics)

    # Display the bracket
    BB['Year'] = BB['Year'].astype('str')
    st.dataframe(BB, height=500)
