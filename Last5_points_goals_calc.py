#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 17:57:16 2025

@author: damianesene
"""

import pandas as pd
import numpy as np
import joblib

# =========================
# Load all past matches
# =========================
xls = pd.ExcelFile("/Users/damianesene/Downloads/all-euro-data-2025-2026.xlsx")
df_list = [pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names]
df = pd.concat(df_list, ignore_index=True)

# Sort by date
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# =========================
# 2️⃣ Function to compute last-5 points and goals
# =========================
def compute_last5_stats(team_name, match_date, is_home=True):
    """Compute last 5 points and goals for a given team before a given date."""
    
    if is_home:
        matches = df[(df['HomeTeam']==team_name) & (df['Date']<match_date)].sort_values('Date')
        points = matches['FTR'].apply(lambda x: 3 if x=='H' else (1 if x=='D' else 0)).tolist()
        goals = matches['FTHG'].tolist()
    else:
        matches = df[(df['AwayTeam']==team_name) & (df['Date']<match_date)].sort_values('Date')
        points = matches['FTR'].apply(lambda x: 3 if x=='A' else (1 if x=='D' else 0)).tolist()
        goals = matches['FTAG'].tolist()
    
    last5_points = np.mean(points[-5:]) if points else 0
    last5_goals = np.mean(goals[-5:]) if goals else 0
    
    return last5_points, last5_goals


home_team = "Newcastle"
away_team = "Chelsea"
match_date = pd.to_datetime("2025-12-14")

home_points, home_goals = compute_last5_stats(home_team, match_date, is_home=True)
away_points, away_goals = compute_last5_stats(away_team, match_date, is_home=False)


# Load Trained Model
model = joblib.load("football_model.pkl")

new_game = pd.DataFrame({
    'home_last5_points': [home_points],
    'away_last5_points': [away_points],
    'home_last5_goals_scored': [home_goals],
    'away_last5_goals_scored': [away_goals],
    'home_win_prob': [2.75],  # fanduel odds
    'draw_prob': [3.50],
    'away_win_prob': [2.50]
})

prediction = model.predict(new_game)
proba = model.predict_proba(new_game)

print("Predicted outcome:", "Home win" if prediction[0]==1 else "Not home win")
print("Probability Home win:", round(proba[0][1], 2))
print("Probability Not home win:", round(proba[0][0], 2))