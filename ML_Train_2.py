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

# Load the Excel file
file_path = "/Users/damianesene/Downloads/all-euro-data-2025-2026.xlsx"
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
df['home_last5_points'] = 0
df['away_last5_points'] = 0
df['home_last5_goals_scored'] = 0
df['away_last5_goals_scored'] = 0


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
        last5_home = home_points[max(0, team_home_matches.index.get_loc(idx)-5):team_home_matches.index.get_loc(idx)]
        df.loc[idx, 'home_last5_points'] = np.mean(last5_home) if last5_home else 0
        #last5_home_goals = team_home_matches.iloc[max(0, team_home_matches.index.get_loc(idx)-5):team_home_matches.index.get_loc(idx)-1, 'FTHG']
        last5_home_goals = team_home_matches.iloc[
    max(0, team_home_matches.index.get_loc(idx)-5) : team_home_matches.index.get_loc(idx)
]['FTHG']
        df.loc[idx, 'home_last5_goals_scored'] = np.mean(last5_home_goals) if len(last5_home_goals) > 0 else 0

    # Away rolling points
    away_points = team_away_matches['FTR'].apply(lambda x: get_points(x, False)).tolist()
    for idx in team_away_matches.index:
        last5_away = away_points[max(0, team_away_matches.index.get_loc(idx)-5):team_away_matches.index.get_loc(idx)]
        df.loc[idx, 'away_last5_points'] = np.mean(last5_away) if last5_away else 0
        #last5_away_goals = team_away_matches.iloc[max(0, team_away_matches.index.get_loc(idx)-5):team_away_matches.index.get_loc(idx)-1, 'FTAG']
        last5_away_goals = team_away_matches.iloc[max(0, team_away_matches.index.get_loc(idx)-5):team_away_matches.index.get_loc(idx)]['FTAG']
        df.loc[idx, 'away_last5_goals_scored'] = np.mean(last5_away_goals) if len(last5_away_goals) > 0 else 0


# add odds as features
df['home_win_prob'] = 1 / df['AvgH']
df['draw_prob'] = 1 / df['AvgD']
df['away_win_prob'] = 1 / df['AvgA']

df['home_elo'] = home_elo_list
df['away_elo'] = away_elo_list
df['elo_diff'] = df['home_elo'] - df['away_elo']

# Select features for ML
features = ['home_last5_points', 'away_last5_points', 
            'home_last5_goals_scored', 'away_last5_goals_scored',
            'home_win_prob', 'draw_prob', 'away_win_prob','home_elo','away_elo','elo_diff']

X = df[features]
#y = df['home_win']
y = df['result_3way']

# Drop any rows with missing values
X = X.fillna(0)

# Split chronologically (train on first 80%, test on last 20%)
split_idx = int(0.8 * len(df))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train Random Forest
model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
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
joblib.dump(model, "football_model_v3.pkl")

# Save ELO ratings
joblib.dump(elo, "elo_ratings.pkl")