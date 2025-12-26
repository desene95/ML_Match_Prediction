import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd

def get_points(ftr: str, is_home: bool) -> int:
    if ftr == "H":
        return 3 if is_home else 0
    if ftr == "A":
        return 0 if is_home else 3
    return 1

def last5_home_stats(df: pd.DataFrame, team: str, match_date: pd.Timestamp) -> tuple[float, float]:
    home_m = df[(df["HomeTeam"] == team) & (df["Date"] < match_date)].sort_values("Date")
    if len(m) == 0:
        return 0.0, 0.0
    home_pts = home_m["FTR"].apply(lambda r: get_points(r, True)).to_numpy()
    home_goals = home_m["FTHG"].to_numpy()
    return float(np.mean(home_pts[-5:])), float(np.mean(home_goals[-5:]))

def last5_away_stats(df: pd.DataFrame, team: str, match_date: pd.Timestamp) -> tuple[float, float]:
    away_m = df[(df["AwayTeam"] == team) & (df["Date"] < match_date)].sort_values("Date")
    if len(away_m) == 0:
        return 0.0, 0.0
    away_pts = away_m["FTR"].apply(lambda r: get_points(r, False)).to_numpy()
    away_goals = away_m["FTAG"].to_numpy()
    return float(np.mean(away_pts[-5:])), float(np.mean(away_goals[-5:]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home-team", required=True)
    ap.add_argument("--away-team", required=True)
    ap.add_argument("--home-odds", required=True)
    ap.add_argument("--draw-odds", required=True)
    ap.add_argument("--away-odds", required=True)
    ap.add_argument("--match-date", default="")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--elo-path", required=True)
    ap.add_argument("--metadata-path", required=True)
    ap.add_argument("--matches-path", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    home_team = args.home_team.strip()
    away_team = args.away_team.strip()

    # Parse match date (if empty, use "today" in UTC)
    if args.match_date.strip():
        match_date = pd.to_datetime(args.match_date.strip())
    else:
        match_date = pd.Timestamp.utcnow().normalize()

    # Load assets
    model = joblib.load(args.model_path)
    elo = joblib.load(args.elo_path)
    with open(args.metadata_path) as f:
        meta = json.load(f)

    df = pd.read_parquet(args.matches_path)
    df["Date"] = pd.to_datetime(df["Date"])

    # Compute last-5 stats
    home_points, home_goals = last5_home_stats(df, home_team, match_date)
    away_points, away_goals = last5_away_stats(df, away_team, match_date)

    # Compute bookmaker probabilities from decimal odds
    home_odds = float(args.home_odds)
    draw_odds = float(args.draw_odds)
    away_odds = float(args.away_odds)

    home_win_prob = 1.0 / home_odds
    draw_prob = 1.0 / draw_odds
    away_win_prob = 1.0 / away_odds

    # Pull ELO dynamically (fallback 1500 if team missing)
    home_elo = float(elo.get(home_team, 1500.0))
    away_elo = float(elo.get(away_team, 1500.0))
    elo_diff = home_elo - away_elo

    # Build feature row (must match training feature names)
    row = {
        "home_last5_points": home_points,
        "away_last5_points": away_points,
        "home_last5_goals_scored": home_goals,
        "away_last5_goals_scored": away_goals,
        "home_win_prob": home_win_prob,
        "draw_prob": draw_prob,
        "away_win_prob": away_win_prob,
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": elo_diff,
    }

    features = meta.get("features", list(row.keys()))
    new_game = pd.DataFrame([row], columns=features).fillna(0)

    # Predict
    proba = model.predict_proba(new_game)[0]
    classes = model.classes_.tolist()

    # Map class -> label (matches your 3-way encoding A=0 D=1 H=2)
    label_map = {0: "Away win", 1: "Draw", 2: "Home win"}
    prob_map = {label_map[int(c)]: float(p) for c, p in zip(classes, proba)}

    pred_class = int(classes[int(np.argmax(proba))])
    pred_label = label_map[pred_class]

    out = {
        "home_team": home_team,
        "away_team": away_team,
        "match_date": str(match_date.date()),
        "inputs": {
            "home_odds": home_odds,
            "draw_odds": draw_odds,
            "away_odds": away_odds,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": elo_diff,
            "home_last5_points": home_points,
            "away_last5_points": away_points,
            "home_last5_goals_scored": home_goals,
            "away_last5_goals_scored": away_goals,
        },
        "prediction": pred_label,
        "probabilities": prob_map,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # Nice summary in Actions log
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
