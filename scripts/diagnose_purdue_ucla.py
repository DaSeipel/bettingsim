#!/usr/bin/env python3
"""
Diagnose Purdue @ UCLA prediction: full feature row, XGB vs BARTHAG, form, tourney flags,
days_rest, and top feature importance. Run from repo root: python3 scripts/diagnose_purdue_ucla.py
"""
from __future__ import annotations

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import numpy as np
import pandas as pd
import joblib

# Game: UCLA home, Purdue away (March 14, 2026 odds)
HOME_TEAM = "UCLA"
AWAY_TEAM = "Purdue"
GAME_DATE = "2026-03-14"
BARTHAG_SCALE = 50.0

NCAAB_DIR = APP_ROOT / "data" / "ncaab"
MODELS_DIR = APP_ROOT / "data" / "models"
NCAAB_SPREAD_MODEL_PATH = MODELS_DIR / "xgboost_spread_ncaab.pkl"
CURRENT_FORM_PATH = NCAAB_DIR / "current_form_2026.csv"
HISTORICAL_GAMES_PATH = NCAAB_DIR / "historical_games.csv"
KENPOM_STAT_COLUMNS = ["ADJOE", "ADJDE", "BARTHAG", "ADJ_T", "EFG_O", "EFG_D", "TOR", "ORB", "FTR", "THREE_PT_RATE", "SEED"]


def load_kenpom() -> pd.DataFrame:
    for name in ["team_stats_2026.csv", "team_stats_combined.csv", "team_stats_2025.csv"]:
        path = NCAAB_DIR / name
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            if df.empty or "TEAM" not in df.columns:
                continue
            if "season" not in df.columns and "2026" in name:
                df["season"] = 2026
            return df
        except Exception:
            continue
    return pd.DataFrame()


def main() -> None:
    print("=" * 80)
    print("PURDUE @ UCLA DIAGNOSTIC (March 14, 2026)")
    print("=" * 80)

    # 1) current_form_2026.csv — Purdue and UCLA form values
    print("\n--- 1) current_form_2026.csv: Purdue and UCLA ---")
    if not CURRENT_FORM_PATH.exists():
        print("  File not found:", CURRENT_FORM_PATH)
    else:
        form_df = pd.read_csv(CURRENT_FORM_PATH)
        for team in [AWAY_TEAM, HOME_TEAM]:
            row = form_df[form_df["team"].astype(str).str.strip() == team]
            if row.empty:
                print(f"  {team}: NOT FOUND in form CSV")
            else:
                r = row.iloc[0]
                last5 = r.get("last5_margin")
                last10 = r.get("last10_margin")
                winpct = r.get("last5_winpct")
                print(f"  {team}: last5_margin={last5}, last10_margin={last10}, last5_winpct={winpct}")
                try:
                    lm, l10 = float(last5), float(last10)
                    if abs(lm) > 25 or abs(l10) > 25:
                        print(f"    WARNING: margin outside typical -20 to +25 range")
                except (TypeError, ValueError):
                    pass

    # 2) Load model and feature columns
    if not NCAAB_SPREAD_MODEL_PATH.exists():
        print("\nModel not found:", NCAAB_SPREAD_MODEL_PATH)
        sys.exit(1)
    payload = joblib.load(NCAAB_SPREAD_MODEL_PATH)
    model = payload["model"]
    feature_columns = list(payload.get("feature_columns", []))
    margin_home_minus_away = payload.get("margin_home_minus_away", False)
    margin_calibration_shift = payload.get("margin_calibration_shift", 0.0)
    print("\n--- 2) Model metadata ---")
    print(f"  feature_columns: {len(feature_columns)}")
    print(f"  margin_home_minus_away: {margin_home_minus_away}")
    print(f"  margin_calibration_shift: {margin_calibration_shift}")

    # 3) Build feature row same way predict_games does (KenPom + enrich)
    kenpom_df = load_kenpom()
    if kenpom_df.empty:
        print("\nKenPom data not found. Cannot build feature row.")
        sys.exit(1)

    effective_season = 2026
    sub = kenpom_df[kenpom_df["season"].astype(int) == effective_season]
    if sub.empty:
        sub = kenpom_df[kenpom_df["season"] == kenpom_df["season"].max()]

    home_row = sub[sub["TEAM"].astype(str).str.strip() == HOME_TEAM.strip()]
    away_row = sub[sub["TEAM"].astype(str).str.strip() == AWAY_TEAM.strip()]
    if home_row.empty or away_row.empty:
        print("\nUCLA or Purdue not found in KenPom. Check TEAM names.")
        sys.exit(1)
    home_row, away_row = home_row.iloc[0], away_row.iloc[0]

    def _float(val, default=0.0):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    row = {}
    for col in KENPOM_STAT_COLUMNS:
        if col == "SEED":
            row["home_seed"] = _float(home_row.get(col), 99.0) or 99.0
            row["away_seed"] = _float(away_row.get(col), 99.0) or 99.0
        else:
            row[f"home_{col}"] = _float(home_row.get(col))
            row[f"away_{col}"] = _float(away_row.get(col))
    row["home_SEED"] = row.get("home_seed", 99.0)
    row["away_SEED"] = row.get("away_seed", 99.0)

    # Diffs
    h_b = _float(row.get("home_BARTHAG"))
    a_b = _float(row.get("away_BARTHAG"))
    row["BARTHAG_diff"] = h_b - a_b
    row["ADJOE_diff"] = _float(row.get("home_ADJOE")) - _float(row.get("away_ADJOE"))
    row["ADJDE_diff"] = _float(row.get("home_ADJDE")) - _float(row.get("away_ADJDE"))
    row["tempo_diff"] = _float(row.get("home_ADJ_T")) - _float(row.get("away_ADJ_T"))

    # home_hca, is_neutral (neutral site per March 14 odds)
    row["home_hca"] = 0.0  # neutral site
    row["is_neutral"] = 1.0

    # Form from current_form_2026.csv
    form_df = pd.read_csv(CURRENT_FORM_PATH) if CURRENT_FORM_PATH.exists() else pd.DataFrame()
    def form_for(team):
        if form_df.empty:
            return 0.0, 0.0, 0.0
        r = form_df[form_df["team"].astype(str).str.strip() == team]
        if r.empty:
            return 0.0, 0.0, 0.0
        r = r.iloc[0]
        return _float(r.get("last5_margin")), _float(r.get("last10_margin")), _float(r.get("last5_winpct"))
    h5, h10, hw = form_for(HOME_TEAM)
    a5, a10, aw = form_for(AWAY_TEAM)
    row["home_last5_margin"] = h5
    row["home_last10_margin"] = h10
    row["home_last5_winpct"] = hw
    row["away_last5_margin"] = a5
    row["away_last10_margin"] = a10
    row["away_last5_winpct"] = aw
    row["last5_margin_diff"] = h5 - a5
    row["last10_margin_diff"] = h10 - a10
    row["last5_winpct_diff"] = hw - aw

    # seed_diff
    hs = row.get("home_seed")
    aes = row.get("away_seed")
    row["seed_diff"] = (_float(aes) - _float(hs)) if (hs is not None and aes is not None) else 0.0

    # is_conference_tourney, is_ncaa_tourney (March 14 = conference tournament week)
    month, day = 3, 14
    row["is_conference_tourney"] = 1.0 if (month == 3 and 1 <= day <= 15) else 0.0
    row["is_ncaa_tourney"] = 1.0 if ((month == 3 and day >= 16) or (month == 4 and day <= 10)) else 0.0

    # days_rest_home, days_rest_away (from historical_games or default)
    days_rest_home, days_rest_away = 0.0, 0.0
    if HISTORICAL_GAMES_PATH.exists():
        try:
            hg = pd.read_csv(HISTORICAL_GAMES_PATH)
            if "date" in hg.columns and "home_team" in hg.columns:
                hg["date"] = pd.to_datetime(hg["date"], errors="coerce")
                hg = hg.dropna(subset=["date"])
                from datetime import datetime
                game_dt = datetime.strptime(GAME_DATE, "%Y-%m-%d").date()
                hg = hg[hg["date"].dt.date < game_dt].sort_values("date")
                for _, r in hg.iterrows():
                    ht, at = str(r.get("home_team", "")).strip(), str(r.get("away_team", "")).strip()
                    d = r["date"]
                    if ht == HOME_TEAM:
                        delta = (game_dt - d.date()).days if hasattr(d, "date") else 7
                        days_rest_home = min(14, max(0, delta))
                    if at == AWAY_TEAM:
                        delta = (game_dt - d.date()).days if hasattr(d, "date") else 7
                        days_rest_away = min(14, max(0, delta))
        except Exception as e:
            print("  days_rest from historical_games failed:", e)
    row["days_rest_home"] = float(days_rest_home)
    row["days_rest_away"] = float(days_rest_away)

    # 4) Complete feature row — every feature name and value
    print("\n--- 3) Complete feature row fed to XGBoost (name, value) ---")
    for c in feature_columns:
        val = row.get(c, 0.0)
        try:
            v = float(val) if val is not None and not (isinstance(val, float) and pd.isna(val)) else 0.0
        except (TypeError, ValueError):
            v = 0.0
        flag = ""
        if c in ("home_BARTHAG", "away_BARTHAG") and (v < 0 or v > 1):
            flag = "  <-- BARTHAG outside 0-1!"
        if "ADJOE" in c or "ADJDE" in c:
            if v == 0:
                flag = "  <-- zero"
            elif v > 150 or v < 80:
                flag = "  <-- unusual range?"
        if "margin" in c or "winpct" in c:
            if abs(v) > 30 and "margin" in c:
                flag = "  <-- large margin (check sum vs avg)"
        if "days_rest" in c and (v < 0 or v > 14):
            flag = "  <-- days_rest unusual"
        if "conference_tourney" in c or "ncaa_tourney" in c:
            if v not in (0.0, 1.0):
                flag = "  <-- should be 0 or 1"
        print(f"  {c}: {v}{flag}")

    # 5) XGB raw predicted margin and BARTHAG-only
    X = np.array([[float(row.get(c, 0.0)) for c in feature_columns]], dtype=np.float64)
    xgb_raw = float(model.predict(X)[0])
    if str(payload.get("margin_home_minus_away", False)) != "True" and not margin_home_minus_away:
        xgb_raw = -xgb_raw
    xgb_margin = xgb_raw - margin_calibration_shift

    barthag_margin = (h_b - a_b) * BARTHAG_SCALE
    # Neutral: subtract ~3 from home margin
    barthag_margin -= 3.0

    print("\n--- 4) XGB raw vs BARTHAG-only predicted margin ---")
    print(f"  XGB raw (before calibration): {xgb_raw:.2f}")
    print(f"  XGB predicted margin (home - away): {xgb_margin:.2f}")
    print(f"  BARTHAG-only: (home_BARTHAG - away_BARTHAG) * 50 = ({h_b:.3f} - {a_b:.3f}) * 50 = {barthag_margin:.2f}")
    diff = abs(xgb_margin - barthag_margin)
    if diff > 10:
        print(f"  >>> LARGE DISCREPANCY: {diff:.1f} pts — feature row or model likely wrong")
    else:
        print(f"  Difference: {diff:.1f} pts")

    # 6) is_conference_tourney / is_ncaa_tourney for today
    print("\n--- 5) Tournament flags for game date (March 14) ---")
    print(f"  is_conference_tourney: {row.get('is_conference_tourney', 'MISSING')} (should be 1 for conf tourney week)")
    print(f"  is_ncaa_tourney: {row.get('is_ncaa_tourney', 'MISSING')}")

    # 7) days_rest
    print("\n--- 6) Days rest ---")
    print(f"  days_rest_home (UCLA): {row.get('days_rest_home', 'MISSING')}")
    print(f"  days_rest_away (Purdue): {row.get('days_rest_away', 'MISSING')}")

    # 8) Top 5 most important features and their values for this game
    print("\n--- 7) Top 5 feature importance and value for this game ---")
    try:
        booster = model.get_booster()
        # XGBoost can use f0, f1, ... or feature names; try weight (count) if gain empty
        imp = booster.get_score(importance_type="gain")
        if not imp:
            imp = booster.get_score(importance_type="weight")
        names = list(booster.feature_names) if hasattr(booster, "feature_names") and booster.feature_names else []
        if names and len(names) == len(feature_columns):
            name_to_imp = {names[i]: imp.get(f"f{i}", 0) + imp.get(names[i], 0) for i in range(len(names))}
        else:
            name_to_imp = {feature_columns[i]: imp.get(f"f{i}", 0) for i in range(len(feature_columns))}
        sorted_features = sorted(name_to_imp.items(), key=lambda x: -float(x[1]))[:5]
        for name, gain in sorted_features:
            val = row.get(name, 0.0)
            print(f"  {name}: importance={gain}, value={val}")
    except Exception as e:
        print("  Could not get booster importance:", e)
        try:
            fi = model.feature_importances_
            for i in np.argsort(fi)[::-1][:5]:
                c = feature_columns[i]
                print(f"  {c}: importance={fi[i]:.4f}, value={row.get(c, 0.0)}")
        except Exception as e2:
            print("  feature_importances_ failed:", e2)

    # Market spread for this game (UCLA +7.5 = home underdog)
    market_spread = 7.5  # home perspective: +7.5 = home underdog
    edge_xgb = xgb_margin - market_spread if xgb_margin is not None else None
    print("\n--- 8) Edge vs market (UCLA +7.5) ---")
    print(f"  Market spread (home): +7.5")
    print(f"  XGB pred margin: {xgb_margin:.2f}  => edge = pred - spread = {xgb_margin:.2f} - 7.5 = {edge_xgb:.2f}" if edge_xgb is not None else "  N/A")
    if edge_xgb is not None and abs(edge_xgb) > 20:
        print("  >>> Edge is impossibly large — indicates bad feature row or wrong model convention.")

    # 9) Simulate PIPELINE row: what value_plays_pipeline feeds (missing diffs/form/tourney/days_rest -> 0)
    print("\n--- 9) SIMULATED PIPELINE ROW (missing features filled with 0) ---")
    pipeline_row = {c: row.get(c, 0.0) for c in feature_columns}
    # Pipeline uses build_ncaab_feature_row_from_team_stats: has raw stats but NOT BARTHAG_diff, form, home_hca, is_neutral, is_conference_tourney, is_ncaa_tourney, days_rest_home, days_rest_away
    for key in ["BARTHAG_diff", "ADJOE_diff", "ADJDE_diff", "tempo_diff", "home_hca", "is_neutral",
                "home_last5_margin", "home_last10_margin", "home_last5_winpct",
                "away_last5_margin", "away_last10_margin", "away_last5_winpct",
                "last5_margin_diff", "last10_margin_diff", "last5_winpct_diff",
                "is_conference_tourney", "is_ncaa_tourney", "seed_diff", "days_rest_home", "days_rest_away"]:
        pipeline_row[key] = 0.0
    X_pipe = np.array([[float(pipeline_row.get(c, 0.0)) for c in feature_columns]], dtype=np.float64)
    xgb_pipe = float(model.predict(X_pipe)[0])
    if not margin_home_minus_away:
        xgb_pipe = -xgb_pipe
    xgb_pipe -= margin_calibration_shift
    edge_pipe = xgb_pipe - market_spread
    print(f"  With pipeline-style row (diffs/form/tourney/days_rest = 0): XGB pred margin = {xgb_pipe:.2f}, edge = {edge_pipe:.2f}")
    if abs(edge_pipe) > 20:
        print("  >>> This explains impossible edges: pipeline feeds incomplete row, model gets wrong prediction.")
    print()


if __name__ == "__main__":
    main()
