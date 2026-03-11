#!/usr/bin/env python3
"""
NCAAB 2026 pipeline diagnostic.

Checks:
  1) Data freshness for 2026 team stats, current form, and HCA.
  2) Model/feature alignment via a dummy Gonzaga vs Duke game.
  3) End-to-end run on the most recent odds CSV in data/odds/.

Does NOT write cache or change any data; prints diagnostics only.
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
import math

import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

NCAAB_DIR = APP_ROOT / "data" / "ncaab"
MODELS_DIR = APP_ROOT / "data" / "models"
ODDS_DIR = APP_ROOT / "data" / "odds"

TEAM_STATS_2026_PATH = NCAAB_DIR / "team_stats_2026.csv"
CURRENT_FORM_PATH = NCAAB_DIR / "current_form_2026.csv"
TEAM_HCA_PATH = NCAAB_DIR / "team_hca_by_season.csv"
CONF_HCA_PATH = NCAAB_DIR / "conf_hca_by_season.csv"
TEAM_CONF_PATH = NCAAB_DIR / "team_conf_by_season.csv"
HISTORICAL_GAMES_PATH = NCAAB_DIR / "historical_games.csv"
MODEL_PATH = MODELS_DIR / "xgboost_spread_ncaab.pkl"
COVER_PROB_PARAMS_PATH = MODELS_DIR / "cover_prob_params.json"

BARTHAG_TO_POINTS_SCALE = 50.0
NEUTRAL_SITE_ADJUSTMENT_PTS = 3.0

# Reuse the same odds-to-stats name mapping conceptually as predict_games.py
ODDS_TO_STATS_NAME = {
    "FGCU": "Florida Gulf Coast",
    "Central Ark": "Central Arkansas",
    "NW State": "Northwestern St.",
    "Nicholls": "Nicholls St.",
    "W. Carolina": "Western Carolina",
    "Queens Charlotte": "Queens",
    "Ga. Southern": "Georgia Southern",
    "Boston U": "Boston University",
    "Kansas State": "Kansas St.",
    "Penn State": "Penn St.",
    "Oklahoma State": "Oklahoma St.",
    "Wright State": "Wright St.",
    "Detroit Mercy": "Detroit Mercy",
    "Missouri State": "Missouri St.",
    "Jacksonville State": "Jacksonville St.",
    "Portland State": "Portland St.",
    "Jackson State": "Jackson St.",
}


def _normalize_team_name(s: str) -> str:
    if pd.isna(s):
        return ""
    return " ".join(str(s).strip().lower().split())


def _load_team_stats_2026() -> pd.DataFrame:
    if not TEAM_STATS_2026_PATH.exists():
        print(f"[ERROR] Missing {TEAM_STATS_2026_PATH}")
        return pd.DataFrame()
    df = pd.read_csv(TEAM_STATS_2026_PATH)
    if "TEAM" in df.columns:
        df["_team_norm"] = df["TEAM"].astype(str).map(_normalize_team_name)
    return df


def _lookup_team(stats_2026: pd.DataFrame, name: str) -> pd.Series | None:
    if stats_2026.empty or "TEAM" not in stats_2026.columns:
        return None
    # Apply alias map first
    mapped_name = ODDS_TO_STATS_NAME.get(name, name)
    norm = _normalize_team_name(mapped_name)
    if not norm:
        return None
    match = stats_2026[stats_2026.get("_team_norm", "").astype(str) == norm]
    if not match.empty:
        return match.iloc[0]
    return None


def _load_current_form() -> pd.DataFrame:
    if not CURRENT_FORM_PATH.exists():
        print(f"[WARN] Missing {CURRENT_FORM_PATH}")
        return pd.DataFrame(columns=["team", "last5_margin", "last10_margin", "last5_winpct"])
    try:
        return pd.read_csv(CURRENT_FORM_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to read {CURRENT_FORM_PATH}: {e}")
        return pd.DataFrame(columns=["team", "last5_margin", "last10_margin", "last5_winpct"])


def _team_form(form_df: pd.DataFrame, team: str) -> tuple[float, float, float]:
    if form_df.empty or "team" not in form_df.columns:
        return 0.0, 0.0, 0.0
    row = form_df[form_df["team"].astype(str).str.strip() == str(team).strip()]
    if row.empty:
        return 0.0, 0.0, 0.0
    r = row.iloc[0]
    def f(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0
    return f(r.get("last5_margin")), f(r.get("last10_margin")), f(r.get("last5_winpct"))


def _load_hca_lookups():
    team_hca = {}
    conf_hca = {}
    team_conf = {}
    if TEAM_HCA_PATH.exists():
        df = pd.read_csv(TEAM_HCA_PATH)
        for _, r in df.iterrows():
            s, t = int(r.get("season", 0)), str(r.get("team", "")).strip()
            if t:
                team_hca[(s, t)] = float(r.get("home_hca", 0) or 0)
    if CONF_HCA_PATH.exists():
        df = pd.read_csv(CONF_HCA_PATH)
        for _, r in df.iterrows():
            s, c = int(r.get("season", 0)), str(r.get("conf", "")).strip()
            if c:
                conf_hca[(s, c)] = float(r.get("home_hca", 0) or 0)
    if TEAM_CONF_PATH.exists():
        df = pd.read_csv(TEAM_CONF_PATH)
        for _, r in df.iterrows():
            s, t, c = int(r.get("season", 0)), str(r.get("team", "")).strip(), str(r.get("conf", "")).strip()
            if t:
                team_conf[(s, t)] = c
    return team_hca, conf_hca, team_conf


def _get_home_hca(season: int, home_team: str, is_neutral: bool,
                  team_hca: dict, conf_hca: dict, team_conf: dict) -> float:
    if is_neutral:
        return 0.0
    key = (season, str(home_team).strip())
    v = team_hca.get(key)
    if v is not None and (season != 2026 or v != 0.0):
        return float(v)
    # Fallback to 2025 team or 2025/2026 conf
    key_2025 = (2025, str(home_team).strip())
    v = team_hca.get(key_2025)
    if v is not None:
        return float(v)
    conf = team_conf.get(key) or team_conf.get(key_2025)
    if conf:
        v = conf_hca.get((season, conf)) or conf_hca.get((2025, conf))
        if v is not None:
            return float(v)
    return 0.0


def _load_model_and_features():
    import joblib
    if not MODEL_PATH.exists():
        print(f"[ERROR] Missing model at {MODEL_PATH}")
        return None, []
    payload = joblib.load(MODEL_PATH)
    model = payload.get("model")
    cols = payload.get("feature_columns") or []
    return model, list(cols)


def _load_cover_prob_params():
    if not COVER_PROB_PARAMS_PATH.exists():
        return None
    try:
        data = pd.read_json(COVER_PROB_PARAMS_PATH)
    except Exception:
        try:
            import json
            with open(COVER_PROB_PARAMS_PATH) as f:
                data = json.load(f)
        except Exception:
            return None
    k = data.get("k")
    intercept = data.get("intercept")
    if k is None or intercept is None:
        return None
    try:
        return {"k": float(k), "intercept": float(intercept)}
    except (TypeError, ValueError):
        return None


def _p_cover_from_edge(edge: float, params: dict | None) -> float | None:
    if params is None:
        return None
    k = params["k"]
    intercept = params["intercept"]
    z = k * edge + intercept
    try:
        p = 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        p = 1.0 if z > 0 else 0.0
    return float(max(0.0, min(1.0, p)))


def _dummy_feature_row(stats_2026: pd.DataFrame,
                       form_df: pd.DataFrame,
                       team_hca: dict,
                       conf_hca: dict,
                       team_conf: dict,
                       home: str,
                       away: str,
                       season: int,
                       is_neutral: bool) -> dict:
    """Build a single feature row dict for (home, away)."""
    home_row = _lookup_team(stats_2026, home)
    away_row = _lookup_team(stats_2026, away)
    feat: dict[str, float] = {}
    for col in ["ADJOE", "ADJDE", "BARTHAG", "ADJ_T", "EFG_O", "EFG_D", "TOR", "ORB", "FTR", "THREE_PT_RATE"]:
        def val(r):
            if r is None:
                return 0.0
            if col in r.index and pd.notna(r[col]):
                try:
                    return float(r[col])
                except (TypeError, ValueError):
                    return 0.0
            if col == "THREE_PT_RATE":
                for alt in ("3PR", "3pr"):
                    if alt in r.index and pd.notna(r[alt]):
                        try:
                            return float(r[alt])
                        except (TypeError, ValueError):
                            pass
            return 0.0
        feat[f"home_{col}"] = val(home_row)
        feat[f"away_{col}"] = val(away_row)
    feat["BARTHAG_diff"] = feat.get("home_BARTHAG", 0.0) - feat.get("away_BARTHAG", 0.0)
    feat["ADJOE_diff"] = feat.get("home_ADJOE", 0.0) - feat.get("away_ADJOE", 0.0)
    feat["ADJDE_diff"] = feat.get("home_ADJDE", 0.0) - feat.get("away_ADJDE", 0.0)
    feat["tempo_diff"] = feat.get("home_ADJ_T", 0.0) - feat.get("away_ADJ_T", 0.0)
    feat["home_hca"] = _get_home_hca(season, home, is_neutral, team_hca, conf_hca, team_conf)
    feat["is_neutral"] = 1 if is_neutral else 0

    h5_m, h10_m, h5_w = _team_form(form_df, home)
    a5_m, a10_m, a5_w = _team_form(form_df, away)
    feat["home_last5_margin"] = h5_m
    feat["home_last10_margin"] = h10_m
    feat["home_last5_winpct"] = h5_w
    feat["away_last5_margin"] = a5_m
    feat["away_last10_margin"] = a10_m
    feat["away_last5_winpct"] = a5_w
    feat["last5_margin_diff"] = h5_m - a5_m
    feat["last10_margin_diff"] = h10_m - a10_m
    feat["last5_winpct_diff"] = h5_w - a5_w

    # Tournament flags and simple defaults for dummy row
    feat["is_conference_tourney"] = 0
    feat["is_ncaa_tourney"] = 0
    # Seed diff from team stats if SEED exists
    def _seed(r):
        if r is None:
            return None
        s = r.get("SEED")
        try:
            v = float(s)
            return v if v != 0 and not pd.isna(v) else None
        except (TypeError, ValueError):
            return None
    hs = _seed(home_row)
    as_ = _seed(away_row)
    if hs is not None and as_ is not None:
        feat["seed_diff"] = as_ - hs
    else:
        feat["seed_diff"] = 0.0
    # Days rest defaults for dummy test
    feat["days_rest_home"] = 7.0
    feat["days_rest_away"] = 7.0
    return feat


def main() -> int:
    issues: list[str] = []

    print("=== 1) Data freshness checks ===")
    # Team stats 2026
    stats_2026 = _load_team_stats_2026()
    if stats_2026.empty:
        issues.append("team_stats_2026.csv missing or empty")
    else:
        print(f"team_stats_2026.csv: {len(stats_2026)} teams")
        sample = None
        for name in ["Gonzaga", "Duke"]:
            row = _lookup_team(stats_2026, name)
            if row is not None:
                sample = (name, row)
                break
        if sample:
            name, row = sample
            print(f"Sample team ({name}): BARTHAG={row.get('BARTHAG')}, ADJOE={row.get('ADJOE')}, ADJDE={row.get('ADJDE')}, ADJ_T={row.get('ADJ_T')}")
        else:
            issues.append("Could not find sample team (Gonzaga/Duke) in team_stats_2026.csv")

    # Current form 2026
    form_df = _load_current_form()
    if form_df.empty:
        issues.append("current_form_2026.csv missing or empty")
    else:
        print(f"current_form_2026.csv: {len(form_df)} teams with form")
        sample = form_df.iloc[0]
        print(f"Sample form team ({sample['team']}): last5_margin={sample.get('last5_margin')}, last10_margin={sample.get('last10_margin')}")

    # HCA 2026
    team_hca, conf_hca, team_conf = _load_hca_lookups()
    has_2026_hca = any(season == 2026 for (season, _team) in team_hca.keys())
    print(f"team_hca_by_season: has 2026 entries? {has_2026_hca}")
    for name in ["Gonzaga", "Duke"]:
        hca_val = _get_home_hca(2026, name, False, team_hca, conf_hca, team_conf)
        print(f"HCA lookup for {name} (2026): {hca_val:.3f}")
        if hca_val == 0.0:
            issues.append(f"HCA for {name} in 2026 resolved to 0.0 (check fallback logic)")
            break

    print("\n=== 2) Model alignment / dummy game ===")
    model, feature_cols = _load_model_and_features()
    if model is None or not feature_cols:
        issues.append("Model or feature_columns missing in xgboost_spread_ncaab.pkl")
    else:
        print(f"Model feature columns ({len(feature_cols)}):")
        print(", ".join(feature_cols))

        home = "Gonzaga"
        away = "Duke"
        dummy_feat = _dummy_feature_row(stats_2026, form_df, team_hca, conf_hca, team_conf, home, away, 2026, is_neutral=False)
        print(f"\nDummy feature row for {home} (home) vs {away} (away):")
        for k in feature_cols:
            print(f"  {k}: {dummy_feat.get(k)}")

        missing = [c for c in feature_cols if c not in dummy_feat]
        nulls = [c for c in feature_cols if c in dummy_feat and dummy_feat[c] is None]
        if missing or nulls:
            issues.append(f"Dummy feature row missing or null model features. missing={missing}, null={nulls}")
            print("Feature alignment: FAIL")
        else:
            print("Feature alignment: PASS (all model feature columns present and non-null)")
            # Run model
            import numpy as np
            X = np.array([[float(dummy_feat[c]) for c in feature_cols]], dtype=float)
            pred_margin = float(model.predict(X)[0])
            print(f"Predicted margin (home - away) for dummy game: {pred_margin:.2f}")

    print("\n=== 3) Latest odds file test ===")
    if not ODDS_DIR.exists():
        print(f"[WARN] Odds dir {ODDS_DIR} does not exist")
        issues.append("Odds directory missing")
    else:
        csvs = list(ODDS_DIR.glob("*.csv"))
        if not csvs:
            print("[WARN] No odds CSVs found in data/odds/")
            issues.append("No odds CSVs found")
        else:
            latest = max(csvs, key=lambda p: p.stat().st_mtime)
            print(f"Latest odds file: {latest.name}")
            try:
                odds = pd.read_csv(latest)
            except Exception as e:
                print(f"[ERROR] Failed to read odds file: {e}")
                issues.append("Failed to read latest odds CSV")
                odds = pd.DataFrame()
            if not odds.empty and model is not None and feature_cols:
                # Normalize headers
                ren = {}
                for c in odds.columns:
                    n = c.strip().lower().replace(" ", "_")
                    if n in ("home_team", "home"):
                        ren[c] = "Home_Team"
                    elif n in ("away_team", "away"):
                        ren[c] = "Away_Team"
                    elif n in ("spread", "line"):
                        ren[c] = "Spread"
                    elif n in ("over_under", "total", "ou"):
                        ren[c] = "Over_Under"
                    elif n in ("is_neutral", "neutral", "neutral_site"):
                        ren[c] = "Is_Neutral"
                odds = odds.rename(columns=ren)
                missing_cols = [c for c in ["Home_Team", "Away_Team", "Spread"] if c not in odds.columns]
                if missing_cols:
                    print(f"[ERROR] Latest odds file missing columns: {missing_cols}")
                    issues.append("Latest odds file missing required columns")
                else:
                    cover_params = _load_cover_prob_params()
                    print("\nHome, Away, Spread, Pred_XGB, Pred_BARTHAG, Edge, P(cover), home_hca, home_last5_margin, away_last5_margin")
                    for _, r in odds.iterrows():
                        home_raw = str(r.get("Home_Team", "")).strip()
                        away_raw = str(r.get("Away_Team", "")).strip()
                        try:
                            spread = float(r.get("Spread", 0.0))
                        except (TypeError, ValueError):
                            spread = 0.0
                        is_neutral = False
                        if "Is_Neutral" in r.index:
                            val = str(r.get("Is_Neutral", "")).strip().lower()
                            is_neutral = val in ("1", "true", "yes", "y")

                        feat = _dummy_feature_row(stats_2026, form_df, team_hca, conf_hca, team_conf,
                                                  home_raw, away_raw, 2026, is_neutral=is_neutral)
                        # Mark missing/defaulted features
                        defaults = []
                        for name in ["home_ADJOE", "away_ADJOE", "home_BARTHAG", "away_BARTHAG",
                                     "home_last5_margin", "away_last5_margin"]:
                            if name not in feat or feat[name] == 0.0:
                                defaults.append(name)

                        X = np.array([[float(feat.get(c, 0.0)) for c in feature_cols]], dtype=float)
                        pred_margin = float(model.predict(X)[0])
                        # BARTHAG-only margin
                        h_b = feat.get("home_BARTHAG", 0.0)
                        a_b = feat.get("away_BARTHAG", 0.0)
                        pred_barthag = (h_b - a_b) * BARTHAG_TO_POINTS_SCALE
                        if is_neutral:
                            pred_barthag -= NEUTRAL_SITE_ADJUSTMENT_PTS
                        edge = pred_margin + spread
                        p_cover = _p_cover_from_edge(pred_margin - spread, cover_params)
                        if p_cover is not None:
                            p_cover_str = f"{p_cover * 100:.1f}%"
                        else:
                            p_cover_str = "—"

                        print(
                            f"{home_raw}, {away_raw}, {spread:+.1f}, {pred_margin:+.2f}, "
                            f"{pred_barthag:+.2f}, {edge:+.2f}, {p_cover_str}, "
                            f"{feat.get('home_hca', 0.0):+.2f}, {feat.get('home_last5_margin', 0.0):+.2f}, "
                            f"{feat.get('away_last5_margin', 0.0):+.2f}"
                        )
                        if defaults:
                            issues.append(f"Defaults/zeros for {home_raw} vs {away_raw}: {defaults}")

    print("\n=== 4) Summary ===")
    if issues:
        print("PIPELINE ISSUES:")
        for msg in sorted(set(issues)):
            print(f"  - {msg}")
    else:
        print("PIPELINE OK")

    return 0


if __name__ == "__main__":
    sys.exit(main())

