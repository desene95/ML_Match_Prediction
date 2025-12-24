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


home_team = "Liverpool"
away_team = "Wolves"
match_date = pd.to_datetime("2025-12-14")

home_points, home_goals = compute_last5_stats(home_team, match_date, is_home=True)
away_points, away_goals = compute_last5_stats(away_team, match_date, is_home=False)



# Load Trained Model
model = joblib.load("football_model_v3.pkl")


elo = joblib.load("elo_ratings.pkl")

home_elo = elo.get(home_team, 1500)  # fallback if team missing
away_elo = elo.get(away_team, 1500)
elo_diff = home_elo - away_elo

new_game = pd.DataFrame({
    'home_last5_points': [home_points],
    'away_last5_points': [away_points],
    'home_last5_goals_scored': [home_goals],
    'away_last5_goals_scored': [away_goals],
    'home_win_prob': [1/1.25],  # fanduel odds
    'draw_prob': [1/7.00],
    'away_win_prob': [1/10.00],
    'home_elo': home_elo,
    'away_elo': away_elo,
    'elo_diff': elo_diff 
})

prediction = model.predict(new_game)
#proba = model.predict_proba(new_game)

proba = model.predict_proba(new_game)[0]

#labels = ['Away win', 'Draw', 'Home win']
labels = {0: "Away win", 1: "Draw", 2: "Home win"}
predicted_class = np.argmax(proba)

print("Predicted outcome:", labels[predicted_class])
print("Probabilities:")
for label, p in zip(labels, proba):
    print(f"  {label}: {round(p, 2)}")