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
    if len(home_m) == 0:
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

import math

def last5_home_conceded(df: pd.DataFrame, team: str, match_date: pd.Timestamp) -> float:
    # Team is HOME -> conceded goals are FTAG (away goals)
    m = df[(df["HomeTeam"] == team) & (df["Date"] < match_date)].sort_values("Date")
    if len(m) == 0:
        return 0.0
    return float(np.mean(m["FTAG"].to_numpy()[-5:]))

def last5_away_conceded(df: pd.DataFrame, team: str, match_date: pd.Timestamp) -> float:
    # Team is AWAY -> conceded goals are FTHG (home goals)
    m = df[(df["AwayTeam"] == team) & (df["Date"] < match_date)].sort_values("Date")
    if len(m) == 0:
        return 0.0
    return float(np.mean(m["FTHG"].to_numpy()[-5:]))

def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam**k) / math.factorial(k)

def poisson_match_probs(lh: float, la: float, max_goals: int = 6) -> tuple[float, float, float]:
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

def elo_asof(df: pd.DataFrame, asof_date: pd.Timestamp, K: float = 30, HOME_ADV: float = 65) -> dict:
    # use only matches strictly before the match date
    hist = df[df["Date"] < asof_date].sort_values("Date")

    teams = pd.unique(hist[["HomeTeam", "AwayTeam"]].values.ravel())
    elo = {t: 1500.0 for t in teams}

    for _, r in hist.iterrows():
        home, away = r["HomeTeam"], r["AwayTeam"]
        home_elo, away_elo = elo.get(home, 1500.0), elo.get(away, 1500.0)

        expected_home = 1 / (1 + 10 ** (((away_elo) - (home_elo + HOME_ADV)) / 400))

        if r["FTR"] == "H":
            score_home = 1.0
        elif r["FTR"] == "D":
            score_home = 0.5
        else:
            score_home = 0.0

        elo[home] = home_elo + K * (score_home - expected_home)
        elo[away] = away_elo + K * ((1 - score_home) - (1 - expected_home))

    return elo

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

    # Last-5 goals conceded
    home_conceded = last5_home_conceded(df, home_team, match_date)
    away_conceded = last5_away_conceded(df, away_team, match_date)

    # Poisson lambdas (same formula you used in training)
    HOME_ADV_GOALS = 0.15

    lambda_home = (0.6 * home_goals + 0.4 * away_conceded + HOME_ADV_GOALS)
    lambda_away = (0.6 * away_goals + 0.4 * home_conceded)

    # Clamp for stability
    lambda_home = float(np.clip(lambda_home, 0.2, 3.5))
    lambda_away = float(np.clip(lambda_away, 0.2, 3.5))

    pois_home, pois_draw, pois_away = poisson_match_probs(lambda_home, lambda_away, max_goals=6)

    # Optional extras (if you trained with them)
    # elo_abs_diff = abs(elo_diff)
    # pois_total_goals = lambda_home + lambda_away


    # Compute bookmaker probabilities from decimal odds
    home_odds = float(args.home_odds)
    draw_odds = float(args.draw_odds)
    away_odds = float(args.away_odds)

    home_win_prob = 1.0 / home_odds
    draw_prob = 1.0 / draw_odds
    away_win_prob = 1.0 / away_odds

    odds_edge_home = home_win_prob - away_win_prob
    odds_draw_strength = draw_prob

    # Pull ELO dynamically (fallback 1500 if team missing)
    elo_now = elo_asof(df, match_date, K=30, HOME_ADV=65)
    home_elo = float(elo_now.get(home_team, 1500.0))
    away_elo = float(elo_now.get(away_team, 1500.0))
    elo_diff = home_elo - away_elo

    elo_abs_diff = abs(elo_diff)
    pois_total_goals = lambda_home + lambda_away

    # Build feature row (must match training feature names)
    row = {
        "home_last5_points": home_points,
        "away_last5_points": away_points,
        "home_last5_goals_scored": home_goals,
        "away_last5_goals_scored": away_goals,
        "home_last5_goals_conceded": home_conceded,
        "away_last5_goals_conceded": away_conceded,
        #"odds_edge_home": odds_edge_home,
        #"odds_draw_strength": odds_draw_strength,   
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": elo_diff,
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "pois_home_win_prob": float(pois_home),
        "pois_draw_prob": float(pois_draw),
        "pois_away_win_prob": float(pois_away),
        "elo_abs_diff": float(abs(elo_diff)),
        "pois_total_goals": float(lambda_home + lambda_away)
    }

    features = meta.get("features", list(row.keys()))
    new_game = pd.DataFrame([row], columns=features).fillna(0)

    #     # =========================
    # # DIAGNOSTIC: remove odds influence
    # # =========================
    # ng = new_game.copy()

    # if "home_win_prob" in ng.columns:
    #     ng["home_win_prob"] = 0
    # if "draw_prob" in ng.columns:
    #     ng["draw_prob"] = 0
    # if "away_win_prob" in ng.columns:
    #     ng["away_win_prob"] = 0

    # proba_no_odds = model.predict_proba(ng)[0]

    # print(
    #     "Without odds:",
    #     dict(zip(model.classes_, proba_no_odds))
    # )

    # Predict
    proba = model.predict_proba(new_game)[0]
    best_class = np.argmax(proba)
    confidence = proba[best_class]
    classes = model.classes_.tolist()

    # Map class -> label (matches your 3-way encoding A=0 D=1 H=2)
    label_map = {0: "Away win", 1: "Draw", 2: "Home win"}
    prob_map = {label_map[int(c)]: float(p) for c, p in zip(classes, proba)}

    pred_class = int(classes[int(np.argmax(proba))])
    pred_label = label_map[pred_class]
    confidence_label = label_map[int(classes[best_class])]

    # -----------------------------
# Draw-aware post processing
# -----------------------------

    p_away = prob_map["Away win"]
    p_draw = prob_map["Draw"]
    p_home = prob_map["Home win"]

    # Override with Draw when the game is tight + low scoring
    if (p_draw >= 0.33) and (abs(elo_diff) < 35) and (pois_total_goals < 2.6):
        final_label = "Draw"
    else:
        final_label = pred_label


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
            "home_last5_goals_conceded": home_conceded,
            "away_last5_goals_conceded": away_conceded,
            "lambda_home": lambda_home,
            "lambda_away": lambda_away,
            "pois_home_win_prob": float(pois_home),
            "pois_draw_prob": float(pois_draw),
            "pois_away_win_prob": float(pois_away),
            "pois_total_goals": float(lambda_home + lambda_away)
        },
        "prediction": final_label,
        "probabilities": prob_map,
    }
    # print("Expected features:", meta["features"])
    # print("Provided columns:", new_game.columns.tolist())
    # missing = [f for f in meta["features"] if f not in new_game.columns]
    # print("Missing:", missing)
    print("model.classes_ =", model.classes_)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # Nice summary in Actions log
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
