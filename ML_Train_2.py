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
df['home_win'] = (df['FTR'] == 'H').astype(int)

# Initialize rolling features
df['home_last5_points'] = 0
df['away_last5_points'] = 0
df['home_last5_goals_scored'] = 0
df['away_last5_goals_scored'] = 0

# Helper function: points from FTR
def get_points(result, is_home):
    if result == 'H':
        return 3 if is_home else 0
    elif result == 'A':
        return 0 if is_home else 3
    else:
        return 1

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

# Select features for ML
features = ['home_last5_points', 'away_last5_points', 
            'home_last5_goals_scored', 'away_last5_goals_scored',
            'home_win_prob', 'draw_prob', 'away_win_prob']

X = df[features]
y = df['home_win']

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
y_pred = model.predict(X_test)
print("Accuracy:", (y_test == y_pred).mean())
print(classification_report(y_test, y_pred))

#  confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred Not Home Win','Pred Home Win'],
            yticklabels=['Actual Not Home Win','Actual Home Win'])
plt.show()

# Save model
joblib.dump(model, "football_model.pkl")