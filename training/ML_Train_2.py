#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 16:12:36 2025

@author: damianesene
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os, json

os.makedirs("dist", exist_ok=True)
# Load the Excel file
file_path = "training/all-euro-data-2025-2026.xlsx"
xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names
df_list =[pd.read_excel(xls, sheet_name=sheet) for sheet in sheet_names]
df =  pd.concat(df_list, ignore_index=True)

# Sort Matches
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Create target variable: home win = 1, else = 0
#df['home_win'] = (df['FTR'] == 'H').astype(int)
# 3 way
ftr_map = {'A': 0, 'D': 1, 'H': 2}
df['result_3way'] = df['FTR'].map(ftr_map)

# Initialize rolling features
df['home_last5_points'] = 0.0
df['away_last5_points'] = 0.0
df['home_last5_goals_scored'] = 0.0
df['away_last5_goals_scored'] = 0.0
df['home_last5_goals_conceded'] = 0.0 # These are for Poission distribution calculations
df['away_last5_goals_conceded'] = 0.0


# =========================
# ELO RATING CALCULATION
# =========================

# Initialize ELO ratings
teams = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel())
elo = {team: 1500 for team in teams}

K = 30
HOME_ADV = 65

home_elo_list = []
away_elo_list = []

for _, row in df.iterrows():
    home = row['HomeTeam']
    away = row['AwayTeam']

    home_elo = elo[home]
    away_elo = elo[away]

    # Store PRE-match ELOs
    home_elo_list.append(home_elo)
    away_elo_list.append(away_elo)

    # Expected home result
    expected_home = 1 / (1 + 10 ** (((away_elo) - (home_elo + HOME_ADV)) / 400))

    # Actual result
    if row['FTR'] == 'H':
        score_home = 1
    elif row['FTR'] == 'D':
        score_home = 0.5
    else:
        score_home = 0

    # Update ELO ratings
    elo[home] += K * (score_home - expected_home)
    elo[away] += K * ((1 - score_home) - (1 - expected_home))

# Helper function: points from FTR
def get_points(result, is_home):
    if result == 'H':
        return 3 if is_home else 0
    elif result == 'A':
        return 0 if is_home else 3
    else:
        return 1

sorted_elo = sorted(elo.items(), key=lambda x: x[1], reverse=True)

for team, rating in sorted_elo:
    print(f"{team}: {round(rating, 1)}")
# Calculate rolling averages
teams = df['HomeTeam'].unique()

for team in teams:
    team_home_matches = df[df['HomeTeam'] == team]
    team_away_matches = df[df['AwayTeam'] == team]

    # Home rolling points
    team_home_matches = team_home_matches.sort_values('Date')
    home_points = team_home_matches['FTR'].apply(lambda x: get_points(x, True)).tolist()
    for idx in team_home_matches.index:
        loc = team_home_matches.index.get_loc(idx)
        start = max(0, loc - 5)
        end = loc
        last5_home = home_points[max(0, team_home_matches.index.get_loc(idx)-5):team_home_matches.index.get_loc(idx)]
        df.loc[idx, 'home_last5_points'] = np.mean(last5_home) if last5_home else 0
        #last5_home_goals = team_home_matches.iloc[max(0, team_home_matches.index.get_loc(idx)-5):team_home_matches.index.get_loc(idx)-1, 'FTHG']
        last5_home_goals = team_home_matches.iloc[
    max(0, team_home_matches.index.get_loc(idx)-5) : team_home_matches.index.get_loc(idx)
]['FTHG']
        df.loc[idx, 'home_last5_goals_scored'] = np.mean(last5_home_goals) if len(last5_home_goals) > 0 else 0

        # ✅ goals conceded at home
        last5_home_conceded = team_home_matches.iloc[start:end]['FTAG']
        df.loc[idx, 'home_last5_goals_conceded'] = (
            float(last5_home_conceded.mean()) if len(last5_home_conceded) > 0 else 0.0
        )

    # Away rolling points
    away_points = team_away_matches['FTR'].apply(lambda x: get_points(x, False)).tolist()
    for idx in team_away_matches.index:
        loc = team_away_matches.index.get_loc(idx)
        start = max(0, loc - 5)
        end = loc
        last5_away = away_points[max(0, team_away_matches.index.get_loc(idx)-5):team_away_matches.index.get_loc(idx)]
        df.loc[idx, 'away_last5_points'] = np.mean(last5_away) if last5_away else 0
        #last5_away_goals = team_away_matches.iloc[max(0, team_away_matches.index.get_loc(idx)-5):team_away_matches.index.get_loc(idx)-1, 'FTAG']
        last5_away_goals = team_away_matches.iloc[max(0, team_away_matches.index.get_loc(idx)-5):team_away_matches.index.get_loc(idx)]['FTAG']
        df.loc[idx, 'away_last5_goals_scored'] = np.mean(last5_away_goals) if len(last5_away_goals) > 0 else 0

        # ✅ goals conceded away
        last5_away_conceded = team_away_matches.iloc[start:end]['FTHG']
        df.loc[idx, 'away_last5_goals_conceded'] = (
            float(last5_away_conceded.mean()) if len(last5_away_conceded) > 0 else 0.0
        )


# add odds as features
df['home_win_prob'] = 1 / df['AvgH']
df['draw_prob'] = 1 / df['AvgD']
df['away_win_prob'] = 1 / df['AvgA']

# Derived odds-shape features (reduces domination)
df["odds_edge_home"] = df["home_win_prob"] - df["away_win_prob"]
df["odds_draw_strength"] = df["draw_prob"]

df['home_elo'] = home_elo_list
df['away_elo'] = away_elo_list
df['elo_diff'] = df['home_elo'] - df['away_elo']

# =========================
# POISSON LAMBDAS (expected goals)
# =========================

HOME_ADV_GOALS = 0.15  # small home boost (tweak 0.10–0.25)

# Expected home goals: home attack + away defensive weakness
df['lambda_home'] = (
    0.6 * df['home_last5_goals_scored'] +
    0.4 * df['away_last5_goals_conceded'] +
    HOME_ADV_GOALS
)

# Expected away goals: away attack + home defensive weakness
df['lambda_away'] = (
    0.6 * df['away_last5_goals_scored'] +
    0.4 * df['home_last5_goals_conceded']
)

# Clamp lambdas to keep them realistic/stable
df['lambda_home'] = df['lambda_home'].clip(lower=0.2, upper=3.5)
df['lambda_away'] = df['lambda_away'].clip(lower=0.2, upper=3.5)


import math

def poisson_pmf(k, lam):
    return math.exp(-lam) * (lam**k) / math.factorial(k)

def poisson_match_probs(lh, la, max_goals=6):
    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0

    for hg in range(max_goals + 1):
        ph = poisson_pmf(hg, lh)
        for ag in range(max_goals + 1):
            pa = poisson_pmf(ag, la)
            p = ph * pa

            if hg > ag:
                p_home += p
            elif hg == ag:
                p_draw += p
            else:
                p_away += p

    return p_home, p_draw, p_away

# Compute Poisson probabilities for every row
pois = df.apply(lambda r: poisson_match_probs(r['lambda_home'], r['lambda_away'], max_goals=6), axis=1)

df['pois_home_win_prob'] = [x[0] for x in pois]
df['pois_draw_prob']     = [x[1] for x in pois]
df['pois_away_win_prob'] = [x[2] for x in pois]


df['elo_abs_diff'] = df['elo_diff'].abs()
df['pois_total_goals'] = df['lambda_home'] + df['lambda_away']

# Select features for ML
features = ['home_last5_points', 'away_last5_points', 
            'home_last5_goals_scored', 'away_last5_goals_scored',
            'home_elo','away_elo','elo_diff',
            # ✅ Poisson features
    'pois_home_win_prob', 'pois_draw_prob', 'pois_away_win_prob','elo_abs_diff', 'pois_total_goals'
]

X = df[features].fillna(0)
#y = df['home_win']
y = df['result_3way']

# Drop any rows with missing values
X = X.fillna(0)

# Split chronologically (train on first 80%, test on last 20%)
split_idx = int(0.8 * len(df))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# Evaluate
#y_pred = model.predict(X_test)
#print("Accuracy:", (y_test == y_pred).mean())
#print(classification_report(y_test, y_pred))
y_pred = model.predict(X_test)

print("Accuracy:", (y_test == y_pred).mean())
print(classification_report(
    y_test,
    y_pred,
    target_names=['Away win', 'Draw', 'Home win']
))

#  confusion matrix
#import seaborn as sns
#import matplotlib.pyplot as plt
#cm = confusion_matrix(y_test, y_pred)
#sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#            xticklabels=['Pred Not Home Win','Pred Home Win'],
#            yticklabels=['Actual Not Home Win','Actual Home Win'])
#plt.show()

# Save model
joblib.dump(model, "dist/football_model.pkl")

# Save ELO ratings
joblib.dump(elo, "dist/elo_ratings.pkl")

metadata = {
    "model_type": "RandomForestClassifier",
    "features": features,
}
with open("dist/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

snapshot_cols = ["Date", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"]
df[snapshot_cols].to_parquet("dist/matches_snapshot.parquet", index=False)