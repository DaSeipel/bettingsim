#!/usr/bin/env python3
"""
Load the most recent NCAAB odds CSV from data/odds/, fuzzy-match team names to the team stats
database, and run spread + totals predictions. No fixed filename: always uses the latest file
in data/odds/ by modification time.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import pandas as pd

ODDS_DIR = APP_ROOT / "data" / "odds"
NCAAB_DIR = APP_ROOT / "data" / "ncaab"
REQUIRED_COLUMNS = ["Home_Team", "Away_Team", "Spread", "Over_Under"]
FUZZY_MIN_SCORE = 70
FUZZY_WARN_SCORE = 80  # Warn if best match is below this (suggest adding to mapping)
# KenPom stat columns used by xgboost_spread_ncaab.pkl (same as engine/betting_models.py)
KENPOM_STAT_COLUMNS = ["ADJOE", "ADJDE", "BARTHAG", "EFG_O", "EFG_D", "TOR", "TORD", "ORB", "DRB", "FTR", "FTRD", "ADJ_T", "SEED"]

# Odds CSV / display name -> name used in team stats (KenPom / team_stats_history)
ODDS_TO_STATS_NAME: dict[str, str] = {
    "FGCU": "Florida Gulf Coast",
    "Central Ark": "Central Arkansas",
}

# Value play: only when |edge| > this (pts). Pick Home if edge > 3, Away if edge < -3, skip if |edge| <= 3.
VALUE_PLAY_THRESHOLD = 3.0
# Confidence: High = |edge| > 7, Medium = 3 < |edge| <= 7, Low = |edge| <= 3
EDGE_HIGH_CONFIDENCE = 7.0
# BARTHAG-based margin: pred_margin = (home_BARTHAG - away_BARTHAG) * this (0.1 diff = 3 pts)
BARTHAG_TO_POINTS_SCALE = 30.0
# Rebounding: ⭐ when recommended side has ROff advantage over opponent > this
ROFF_ADVANTAGE_THRESHOLD = 3.0
# Reliability: if away BARTHAG > home BARTHAG + this, flag as low reliability and exclude from value plays
BARTHAG_MISMATCH_THRESHOLD = 0.25
LOW_RELIABILITY_FLAG = "Low Reliability - Large KenPom Mismatch"
# Diversity: max share of value plays that can be Home picks (remainder Away)
VALUE_PLAY_MAX_HOME_RATIO = 0.70
# Use 2026 in dates to match stats database
ODDS_YEAR = 2026


def _get_latest_odds_path() -> Path | None:
    """Return the most recent CSV in data/odds/ by modification time, or None."""
    if not ODDS_DIR.exists():
        return None
    csvs = list(ODDS_DIR.glob("*.csv"))
    if not csvs:
        return None
    return max(csvs, key=lambda p: p.stat().st_mtime)


def _game_date_from_odds_path(odds_path: Path) -> str:
    """Derive YYYY-MM-DD from odds filename (e.g. March_08_2026_Odds.csv -> 2026-03-08), else use today with ODDS_YEAR."""
    stem = odds_path.stem  # e.g. March_08_2026_Odds
    try:
        # Expect Month_DD_YYYY or Month_D_YYYY before _Odds
        base = stem.replace("_Odds", "").strip()
        parts = base.split("_")
        if len(parts) >= 3:
            month_str, day_str, year_str = parts[0], parts[1], parts[2]
            month_map = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
                         "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}
            mo = month_map.get(month_str)
            if mo is not None:
                day = int(day_str)
                yr = int(year_str)
                if 1 <= mo <= 12 and 1 <= day <= 31 and 2020 <= yr <= 2030:
                    return f"{yr}-{mo:02d}-{day:02d}"
    except (ValueError, IndexError, AttributeError):
        pass
    today_md = datetime.now().strftime("%m-%d")
    return f"{ODDS_YEAR}-{today_md}"


def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Allow case-insensitive headers; map to exact Home_Team, Away_Team, Spread, Over_Under."""
    rename = {}
    for col in df.columns:
        n = col.strip().lower().replace(" ", "_")
        if n in ("home_team", "home"):
            rename[col] = "Home_Team"
        elif n in ("away_team", "away"):
            rename[col] = "Away_Team"
        elif n in ("spread", "line"):
            rename[col] = "Spread"
        elif n in ("over_under", "total", "ou"):
            rename[col] = "Over_Under"
    return df.rename(columns=rename)


def _load_kenpom_from_csv() -> pd.DataFrame:
    """Load KenPom stats from data/ncaab/ (same CSVs used by engine/betting_models.py via merge script).
    Prefer team_stats_2026.csv (current season); else team_stats_combined.csv, team_stats_2025.csv, etc. Returns DataFrame with
    season, TEAM, and KENPOM_STAT_COLUMNS; empty if no file found."""
    for name in ["team_stats_2026.csv", "team_stats_combined.csv", "team_stats_2025.csv", "team_stats_2024.csv", "team_stats_2023.csv"]:
        path = NCAAB_DIR / name
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty or "TEAM" not in df.columns:
            continue
        if "season" not in df.columns and name != "team_stats_combined.csv":
            # Per-season file: infer from filename (e.g. team_stats_2025 -> 2025)
            try:
                year = int(name.replace("team_stats_", "").replace(".csv", ""))
                df["season"] = year
            except ValueError:
                continue
        df["season"] = df["season"].astype(int)
        keep = ["season", "TEAM"] + [c for c in KENPOM_STAT_COLUMNS if c in df.columns]
        return df[[c for c in keep if c in df.columns]].copy()
    return pd.DataFrame()


def _get_candidate_teams(kenpom_df: pd.DataFrame | None = None) -> list[str]:
    """Collect team names from team_stats_history, ncaab_team_season_stats, and KenPom CSV."""
    candidates = []
    try:
        from engine.team_stats_history import load_team_stats_history
        hist = load_team_stats_history()
        if not hist.empty and "Team" in hist.columns:
            candidates.extend(hist["Team"].astype(str).str.strip().dropna().unique().tolist())
    except Exception:
        pass
    try:
        import sqlite3
        db = APP_ROOT / "data" / "espn.db"
        if db.exists():
            conn = sqlite3.connect(db)
            try:
                df = pd.read_sql_query("SELECT TEAM FROM ncaab_team_season_stats", conn)
                if not df.empty:
                    candidates.extend(df["TEAM"].astype(str).str.strip().dropna().unique().tolist())
            finally:
                conn.close()
    except Exception:
        pass
    if kenpom_df is not None and not kenpom_df.empty and "TEAM" in kenpom_df.columns:
        candidates.extend(kenpom_df["TEAM"].astype(str).str.strip().dropna().unique().tolist())
    return list(dict.fromkeys(candidates))


def _resolve_team_for_lookup(raw_name: str) -> str:
    """Apply static mapping from odds name to stats DB name."""
    name = str(raw_name).strip()
    return ODDS_TO_STATS_NAME.get(name, name)


def _fuzzy_match_team(name: str, candidates: list[str], raw_name_for_warning: str | None = None) -> tuple[str | None, int]:
    """Return (best_match, score) or (None, 0). Uses thefuzz (fuzzywuzzy-compatible)."""
    if not name or not candidates:
        return None, 0
    try:
        from thefuzz import fuzz
        from thefuzz import process as fuzz_process
    except ImportError:
        return None, 0
    name = str(name).strip()
    display_name = (raw_name_for_warning or name).strip()
    match = fuzz_process.extractOne(name, candidates, scorer=fuzz.ratio)
    if not match or match[1] < FUZZY_MIN_SCORE:
        match2 = fuzz_process.extractOne(name, candidates, scorer=fuzz.token_set_ratio)
        if match2 and match2[1] >= FUZZY_MIN_SCORE:
            if match2[1] < FUZZY_WARN_SCORE:
                print(f"MISSING DATA FOR: [{display_name}] (fuzzy score {match2[1]}, consider adding to ODDS_TO_STATS_NAME)", file=sys.stderr)
            return match2[0], match2[1]
        if match and match[1] >= FUZZY_MIN_SCORE:
            if match[1] < FUZZY_WARN_SCORE:
                print(f"MISSING DATA FOR: [{display_name}] (fuzzy score {match[1]}, consider adding to ODDS_TO_STATS_NAME)", file=sys.stderr)
            return match[0], match[1]
        if match:
            print(f"MISSING DATA FOR: [{display_name}] (best match: {match[0]} @ {match[1]}, below threshold)", file=sys.stderr)
        return (match[0], match[1]) if match and match[1] >= FUZZY_MIN_SCORE else (None, 0)
    if match[1] < FUZZY_WARN_SCORE:
        print(f"MISSING DATA FOR: [{display_name}] (fuzzy score {match[1]}, consider adding to ODDS_TO_STATS_NAME)", file=sys.stderr)
    return match[0], match[1]


def _build_feature_row_from_kenpom(
    kenpom_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    effective_season: int,
) -> pd.Series | None:
    """Build a feature row for NCAAB XGBoost spread model from KenPom DataFrame.
    home_team/away_team must match TEAM in kenpom_df (use fuzzy-matched names). Returns None if either team missing."""
    if kenpom_df.empty or "TEAM" not in kenpom_df.columns or "season" not in kenpom_df.columns:
        return None
    sub = kenpom_df[kenpom_df["season"] == effective_season]
    if sub.empty:
        # Fallback to latest season in dataframe
        sub = kenpom_df[kenpom_df["season"] == kenpom_df["season"].max()]
    if sub.empty:
        return None
    home_row = sub[sub["TEAM"].astype(str).str.strip() == str(home_team).strip()]
    away_row = sub[sub["TEAM"].astype(str).str.strip() == str(away_team).strip()]
    if home_row.empty or away_row.empty:
        return None
    home_row = home_row.iloc[0]
    away_row = away_row.iloc[0]
    _seed_default = 99.0

    def _float(val, default: float = 0.0) -> float:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    row = {"league": "ncaab", "home_team_name": home_team, "away_team_name": away_team}
    for col in KENPOM_STAT_COLUMNS:
        if col == "SEED":
            row["home_seed"] = _float(home_row.get(col), _seed_default) or _seed_default
            row["away_seed"] = _float(away_row.get(col), _seed_default) or _seed_default
        else:
            if col in home_row.index:
                row[f"home_{col}"] = _float(home_row[col])
            if col in away_row.index:
                row[f"away_{col}"] = _float(away_row[col])
    row["home_SEED"] = row.get("home_seed", _seed_default)
    row["away_SEED"] = row.get("away_seed", _seed_default)
    row["home_days_rest"] = 0.0
    row["away_days_rest"] = 0.0
    row["home_games_in_last_5_days"] = 0
    row["away_games_in_last_5_days"] = 0
    row["home_is_b2b"] = 0
    row["away_is_b2b"] = 0
    return pd.Series(row)


def main() -> None:
    path = _get_latest_odds_path()
    if path is None:
        print(f"No CSV found in {ODDS_DIR}. Add a file (e.g. March_07_2026_Odds.csv) and run again.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    df = _normalize_headers(df)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"Missing columns {missing}. Required: {REQUIRED_COLUMNS}. File: {path}", file=sys.stderr)
        sys.exit(1)
    from engine.utils import game_season_from_date, effective_kenpom_season
    kenpom_df = _load_kenpom_from_csv()
    candidates = _get_candidate_teams(kenpom_df)
    if not candidates:
        print("No team stats found (team_stats_history, ncaab_team_season_stats, or KenPom CSV). Fuzzy match will be limited.", file=sys.stderr)
    from engine.betting_models import (
        build_ncaab_feature_row_from_team_stats,
        consensus_spread,
        consensus_totals,
    )
    from engine.engine import get_team_power_ratings, get_nba_team_pace_stats, NBA_LEAGUE_AVG_PACE, NBA_LEAGUE_AVG_OFF_RATING
    NCAAB_DEFAULT_RATING = (NBA_LEAGUE_AVG_PACE * NBA_LEAGUE_AVG_OFF_RATING) / 100.0
    from strategies.strategies import implied_probability_no_vig
    # Use date from odds filename (e.g. March_08_2026_Odds.csv -> 2026-03-08) so saved plays match the slate
    game_date = _game_date_from_odds_path(path)
    season = game_season_from_date(game_date) or ODDS_YEAR
    effective_season = effective_kenpom_season(game_date, season) or season
    pace_stats = get_nba_team_pace_stats()
    power_ratings = get_team_power_ratings(pace_stats, NBA_LEAGUE_AVG_PACE, NBA_LEAGUE_AVG_OFF_RATING)
    default_rating = NCAAB_DEFAULT_RATING
    results = []
    used_kenpom = 0
    used_fallback = 0
    for _, row in df.iterrows():
        home_raw = str(row.get("Home_Team", "")).strip()
        away_raw = str(row.get("Away_Team", "")).strip()
        try:
            spread = float(row.get("Spread", 0))
        except (TypeError, ValueError):
            spread = 0.0
        try:
            total = float(row.get("Over_Under", 0))
        except (TypeError, ValueError):
            total = 0.0
        home_lookup = _resolve_team_for_lookup(home_raw)
        away_lookup = _resolve_team_for_lookup(away_raw)
        home_matched, home_score = _fuzzy_match_team(home_lookup, candidates, raw_name_for_warning=home_raw)
        away_matched, away_score = _fuzzy_match_team(away_lookup, candidates, raw_name_for_warning=away_raw)
        home = home_matched or home_lookup
        away = away_matched or away_lookup
        # Prefer KenPom from data/ncaab/ to run XGBoost spread model; else fall back to espn.db / simple model
        feature_row = None
        used_kenpom_this_game = False
        if not kenpom_df.empty:
            feature_row = _build_feature_row_from_kenpom(kenpom_df, home, away, effective_season)
            if feature_row is not None:
                used_kenpom += 1
                used_kenpom_this_game = True
        if feature_row is None:
            feature_row = build_ncaab_feature_row_from_team_stats(home, away, game_date=game_date)
            if feature_row is None:
                used_fallback += 1
        if feature_row is not None:
            if "home_seed" in feature_row.index and "home_SEED" not in feature_row.index:
                feature_row["home_SEED"] = feature_row.get("home_seed", 0)
            if "away_seed" in feature_row.index and "away_SEED" not in feature_row.index:
                feature_row["away_SEED"] = feature_row.get("away_seed", 0)
        # BARTHAG: extract for margin (must be before we use them for pred_margin)
        h_b, a_b = None, None
        if feature_row is not None:
            idx = getattr(feature_row, "index", None)
            if idx is not None and hasattr(idx, "__contains__"):
                h_b = feature_row["home_BARTHAG"] if "home_BARTHAG" in idx else feature_row.get("home_BARTHAG")
                a_b = feature_row["away_BARTHAG"] if "away_BARTHAG" in idx else feature_row.get("away_BARTHAG")
            else:
                h_b = feature_row.get("home_BARTHAG")
                a_b = feature_row.get("away_BARTHAG")
        # Pure BARTHAG-based prediction (no XGBoost): pred_margin = (home_BARTHAG - away_BARTHAG) * 30
        market_spread = float(spread)
        pred_margin = None
        if h_b is not None and pd.notna(h_b) and a_b is not None and pd.notna(a_b):
            try:
                pred_margin = (float(h_b) - float(a_b)) * BARTHAG_TO_POINTS_SCALE
            except (TypeError, ValueError):
                pass
        # Edge = pred_margin - market_spread. Positive = Home covers, negative = Away covers.
        edge_pts = (pred_margin - market_spread) if pred_margin is not None else None
        # Pick: Home if edge > 3, Away if edge < -3, Skip if |edge| <= 3
        if edge_pts is not None:
            if edge_pts > VALUE_PLAY_THRESHOLD:
                pick_spread = "Home"
            elif edge_pts < -VALUE_PLAY_THRESHOLD:
                pick_spread = "Away"
            else:
                pick_spread = "Skip"
        else:
            pick_spread = "Skip"
        is_value_play = edge_pts is not None and abs(edge_pts) > VALUE_PLAY_THRESHOLD
        we_cover_favorite = spread <= 0
        in_house_spread = 0.0
        if feature_row is not None and power_ratings:
            home_r = power_ratings.get(home, default_rating)
            away_r = power_ratings.get(away, default_rating)
            in_house_spread = (home_r - away_r) / 100.0
        spread_prob, spread_ok = consensus_spread(feature_row, spread, we_cover_favorite, in_house_spread, 0.5, league="ncaab") if feature_row is not None else (0.5, True)
        over_prob, tot_ok = consensus_totals(feature_row, total, True, total, "ncaab", 0.5) if feature_row is not None and total else (0.5, True)
        under_prob = 1.0 - over_prob if total else 0.5
        if edge_pts is not None:
            abs_edge = abs(edge_pts)
            if abs_edge > EDGE_HIGH_CONFIDENCE:
                confidence = "High"
            elif abs_edge > VALUE_PLAY_THRESHOLD:
                confidence = "Medium"
            else:
                confidence = "Low"
        else:
            confidence = "Low"
        home_roff_raw = feature_row.get("home_ROff") if feature_row is not None else None
        away_roff_raw = feature_row.get("away_ROff") if feature_row is not None else None
        roff_available = (
            home_roff_raw is not None
            and away_roff_raw is not None
            and not pd.isna(home_roff_raw)
            and not pd.isna(away_roff_raw)
        )
        if roff_available:
            try:
                home_roff = float(home_roff_raw)
                away_roff = float(away_roff_raw)
            except (TypeError, ValueError):
                roff_available = False
                home_roff = away_roff = 0.0
        else:
            home_roff = away_roff = 0.0
        # Advantage = picked side's ROff minus opponent's (only star when picked side is actually higher)
        roff_advantage = (home_roff - away_roff) if pick_spread == "Home" else (away_roff - home_roff) if pick_spread == "Away" and roff_available else 0.0
        rebounding_star = roff_available and confidence == "High" and roff_advantage > ROFF_ADVANTAGE_THRESHOLD
        # Reliability: flag when away BARTHAG > home BARTHAG + 0.25 (exclude from value plays)
        reliability_flag = None
        if h_b is not None and a_b is not None:
            try:
                h_val = float(h_b) if pd.notna(h_b) else None
                a_val = float(a_b) if pd.notna(a_b) else None
                if h_val is not None and a_val is not None and (a_val - h_val) > BARTHAG_MISMATCH_THRESHOLD:
                    reliability_flag = LOW_RELIABILITY_FLAG
            except (TypeError, ValueError):
                pass
        results.append({
            "Home": home,
            "Away": away,
            "Home_BARTHAG": float(h_b) if h_b is not None and pd.notna(h_b) else None,
            "Away_BARTHAG": float(a_b) if a_b is not None and pd.notna(a_b) else None,
            "Market_Spread": market_spread,
            "Over_Under": total,
            "Pred_Margin": pred_margin,
            "Edge": edge_pts,
            "Confidence": confidence,
            "Is_Value_Play": is_value_play,
            "Rebounding_Star": rebounding_star,
            "ROff_Stats_Available": roff_available,
            "Used_KenPom": used_kenpom_this_game,
            "Spread_Prob": spread_prob,
            "Over_Prob": over_prob,
            "Pick_Spread": pick_spread,
            "Pick_Total": "Over" if over_prob > 0.5 else "Under",
            "Reliability_Flag": reliability_flag,
        })
    value_plays_raw = [r for r in results if r["Is_Value_Play"]]
    value_plays = [r for r in value_plays_raw if not r.get("Reliability_Flag")]
    filtered_mismatch = len(value_plays_raw) - len(value_plays)
    n_after_reliability = len(value_plays)
    # Diversity: cap Home picks at 70%; sort by edge and take top Home up to cap + all Away
    if value_plays:
        home_plays = sorted([r for r in value_plays if r["Pick_Spread"] == "Home"], key=lambda x: abs(x.get("Edge") or 0), reverse=True)
        away_plays = sorted([r for r in value_plays if r["Pick_Spread"] == "Away"], key=lambda x: abs(x.get("Edge") or 0), reverse=True)
        n_total = len(value_plays)
        max_home = int(VALUE_PLAY_MAX_HOME_RATIO * n_total)
        if len(home_plays) > max_home:
            value_plays = home_plays[:max_home] + away_plays
            value_plays = sorted(value_plays, key=lambda x: abs(x.get("Edge") or 0), reverse=True)
        else:
            value_plays = sorted(value_plays, key=lambda x: abs(x.get("Edge") or 0), reverse=True)
        n_home = sum(1 for r in value_plays if r["Pick_Spread"] == "Home")
        n_away = len(value_plays) - n_home
        pct_home = (100.0 * n_home / len(value_plays)) if value_plays else 0
        print(f"Value plays after diversity (max {int(VALUE_PLAY_MAX_HOME_RATIO*100)}% Home): {n_home} Home / {n_away} Away  ({pct_home:.0f}% Home).")
        print(f"Final Home vs Away ratio: {n_home} / {n_away}  ({pct_home:.0f}% Home).")
    value_plays_set = {(r["Home"], r["Away"]) for r in value_plays} if value_plays else set()
    high_confidence = [r for r in results if r.get("Confidence") == "High" and not r.get("Reliability_Flag")]
    print(f"Loaded: {path.name} ({len(results)} games)  |  Value plays (|edge| > {VALUE_PLAY_THRESHOLD} pts): {len(value_plays_raw)}  |  After reliability filter: {n_after_reliability}  |  After diversity: {len(value_plays)}  |  High confidence: {len(high_confidence)}")
    if filtered_mismatch:
        print(f"Filtered out {filtered_mismatch} game(s) as '{LOW_RELIABILITY_FLAG}' (away BARTHAG > home BARTHAG + {BARTHAG_MISMATCH_THRESHOLD}).")
    print(f"Games using KenPom (BARTHAG): {used_kenpom}  |  Fallback (Stats N/A): {used_fallback}\n")
    # BARTHAG by game: Home vs Away so we can see if home is genuinely better in most games
    print("BARTHAG by game (today's slate)")
    print("-" * 80)
    print(f"{'Game':<45} | {'Home BARTHAG':>12} | {'Away BARTHAG':>12} | Who is better")
    print("-" * 80)
    for r in results:
        game = f"{r['Away']} @ {r['Home']}"
        h_b = r.get("Home_BARTHAG")
        a_b = r.get("Away_BARTHAG")
        h_str = f"{h_b:.3f}" if h_b is not None else "—"
        a_str = f"{a_b:.3f}" if a_b is not None else "—"
        if h_b is not None and a_b is not None:
            if h_b > a_b:
                who = "Home"
            elif a_b > h_b:
                who = "Away"
            else:
                who = "Tie"
        else:
            who = "N/A"
        print(f"{game:<45} | {h_str:>12} | {a_str:>12} | {who}")
    print("-" * 80)
    # Debug: games where away has better BARTHAG — show pred_margin and pick
    _away_better_home_pick_games = [
        ("Colgate", "Lehigh"),
        ("Memphis", "Tulane"),
        ("San Francisco", "Oregon St."),
    ]
    print("\nAway better (BARTHAG) — pred_margin and pick:")
    print("-" * 80)
    for r in results:
        away_s = str(r.get("Away", "")).strip()
        home_s = str(r.get("Home", "")).strip()
        for away_key, home_key in _away_better_home_pick_games:
            if away_key in away_s and home_key in home_s:
                raw = r.get("Pred_Margin")
                edge = r.get("Edge")
                pick = r.get("Pick_Spread", "")
                raw_s = f"{raw:+.2f}" if raw is not None else "—"
                edge_s = f"{edge:+.2f}" if edge is not None else "—"
                print(f"  {away_s} @ {home_s}")
                print(f"    pred_margin (home): {raw_s}   Edge: {edge_s}   Pick_Spread: {pick}")
                break
    print("-" * 80)
    print()
    for r in results:
        pm = f"{r['Pred_Margin']:+.1f}" if r.get("Pred_Margin") is not None else "—"
        edge_str = f"{r['Edge']:+.1f}" if r["Edge"] is not None else "—"
        vp_tag = "  *** VALUE PLAY ***" if ((r["Home"], r["Away"]) in value_plays_set) else ""
        star = " ⭐" if r.get("Rebounding_Star") else ""
        stats_na = "  (Stats N/A)" if not r.get("Used_KenPom", False) else ""
        print(f"{r['Away']} @ {r['Home']}{vp_tag}{star}{stats_na}")
        reliability_note = f"  [{r.get('Reliability_Flag')}]" if r.get("Reliability_Flag") else ""
        pick_str = r["Pick_Spread"] if r["Pick_Spread"] != "Skip" else "Skip (no pick)"
        print(f"  Market (home): {r['Market_Spread']:+.1f}   Pred (home): {pm}   Edge: {edge_str} pts   Confidence: {r['Confidence']}   P(cover): {r['Spread_Prob']:.2f}  → {pick_str}{reliability_note}")
        print(f"  Total: {r['Over_Under']:.1f}  P(Over): {r['Over_Prob']:.2f}  → {r['Pick_Total']}\n")
    if value_plays:
        header = "--- Value Plays Summary (|edge| > 3 pts)" + (" — remaining after reliability filter ---" if filtered_mismatch else " ---")
        print(header)
        for r in value_plays:
            star = " ⭐" if r.get("Rebounding_Star") else ""
            print(f"  [{r['Confidence']}] {r['Away']} @ {r['Home']}: Edge {r['Edge']:+.1f} pts  → {r['Pick_Spread']}{star}")

    print("\n" + "=" * 60)
    if high_confidence:
        high_sorted = sorted(high_confidence, key=lambda x: abs(x["Edge"] or 0), reverse=True)
        top5 = high_sorted[:5]
        print("HIGH CONFIDENCE ONLY (Edge > 7.0 pts) — Top 5 by Edge")
        print("=" * 60)
        for r in top5:
            star = " ⭐ Rebounding edge" if r.get("Rebounding_Star") else ""
            stats_na = " (Stats N/A)" if not r.get("Used_KenPom", False) else ""
            print(f"  {r['Away']} @ {r['Home']}{stats_na}")
            pred_s = f"{r['Pred_Margin']:+.1f}" if r.get("Pred_Margin") is not None else "—"
            print(f"    Pick: {r['Pick_Spread']}   Edge: {r['Edge']:+.1f} pts   Market (home): {r['Market_Spread']:+.1f}   Pred (home): {pred_s}{star}")
        if len(high_sorted) > 5:
            print(f"  ... and {len(high_sorted) - 5} more high-confidence plays (see full list above)")
        print("=" * 60)

    # Save value plays to historical performance CSV (2026 date, with new columns)
    n_saved = _save_historical_performance(results, value_plays, path, game_date)
    n_home_final = sum(1 for r in value_plays if r.get("Pick_Spread") == "Home")
    n_away_final = len(value_plays) - n_home_final
    pct_home_final = (100.0 * n_home_final / len(value_plays)) if value_plays else 0
    print(f"\nSUCCESS: [{n_saved}] picks saved to historical_betting_performance.csv for {game_date}.")
    print(f"FINAL HOME/AWAY RATIO: {n_home_final} Home / {n_away_final} Away  ({pct_home_final:.0f}% Home).")

    # Write value plays to dashboard cache so "All Value Plays" section populates from manual odds
    _write_value_plays_cache(value_plays, path, game_date)


def _write_value_plays_cache(
    value_plays: list[dict],
    odds_path: Path,
    game_date: str,
) -> None:
    """Write value plays to data/cache/value_plays_cache.json so the dashboard 'All Value Plays' section populates from manual odds.
    Always overwrites the entire file with this run's plays only (never appends)."""
    cache_dir = APP_ROOT / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "value_plays_cache.json"

    value_plays_list = []
    for r in value_plays:
        home = str(r.get("Home", "")).strip()
        away = str(r.get("Away", "")).strip()
        pick = str(r.get("Pick_Spread", "Home")).strip()
        selection = home if pick == "Home" else away
        edge_pts = r.get("Edge")
        try:
            edge_pts_f = float(edge_pts) if edge_pts is not None else 0.0
        except (TypeError, ValueError):
            edge_pts_f = 0.0
        # Value % = edge magnitude for all picks (Away edges are negative; use abs so Away shows non-zero)
        value_pct = min(15.0, abs(edge_pts_f) / 3.0)
        market_spread = r.get("Market_Spread")
        try:
            spread_f = float(market_spread) if market_spread is not None else None
        except (TypeError, ValueError):
            spread_f = None
        # Spread from picked team's perspective: Home = market (home line), Away = -market (away line)
        point_picked = spread_f if pick == "Home" else (-spread_f if spread_f is not None else None)
        point_str = f"{point_picked:+.1f}" if point_picked is not None else "—"
        # Dynamic reasoning: edge pts, BARTHAG, which team favored by KenPom
        h_b = r.get("Home_BARTHAG")
        a_b = r.get("Away_BARTHAG")
        edge_str = f"{edge_pts_f:+.1f}" if edge_pts_f is not None else "—"
        if h_b is not None and a_b is not None and not (isinstance(h_b, float) and (h_b != h_b)) and not (isinstance(a_b, float) and (a_b != a_b)):
            try:
                h_val, a_val = float(h_b), float(a_b)
                diff = abs(h_val - a_val)
                if h_val > a_val:
                    favored = f"{home} (BARTHAG {h_val:.3f})"
                    underdog = f"{away} ({a_val:.3f})"
                else:
                    favored = f"{away} (BARTHAG {a_val:.3f})"
                    underdog = f"{home} ({h_val:.3f})"
                reason = f"Model favors {selection} ({point_str}) — Edge {edge_str} pts. KenPom: {favored} vs {underdog}. Stronger profile by {diff:.3f}."
            except (TypeError, ValueError):
                reason = f"Model favors {selection} ({point_str}) — Edge {edge_str} pts. Strong edge based on scoring margin and rebounding."
        else:
            reason = f"Model favors {selection} ({point_str}) — Edge {edge_str} pts. Strong edge based on scoring margin and rebounding."
        value_plays_list.append({
            "League": "NCAAB",
            "Event": f"{away} @ {home}",
            "Selection": selection,
            "Market": "Spread",
            "Odds": -110,
            "Value (%)": round(value_pct, 2),
            "point": point_picked,
            "Point": point_picked,
            "Recommended Stake": None,
            "Injury Alert": "—",
            "Start Time": "Today",
            "home_team": home,
            "away_team": away,
            "confidence_tier": str(r.get("Confidence", "Medium")).strip() or "Medium",
            "reason": reason,
            "reasoning_summary": reason,
        })
    payload = {
        "value_plays": value_plays_list,
        "potd_picks": {"NCAAB Pick 1": None, "NCAAB Pick 2": None},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "odds_source_meta": {"source": "manual_odds", "odds_file": odds_path.name},
        "value_plays_flagged_count": 0,
    }
    with open(cache_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {len(value_plays_list)} value plays to {cache_path.relative_to(APP_ROOT)} (cache overwritten with this run only).")


def _save_historical_performance(
    results: list[dict],
    value_plays: list[dict],
    odds_path: Path,
    game_date: str,
) -> int:
    """Append value plays to data/historical_betting_performance.csv with Confidence_Level, Edge_Points, Has_Rebound_Advantage. Returns count saved."""
    hist_path = APP_ROOT / "data" / "historical_betting_performance.csv"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "Date", "Odds_File", "Home", "Away", "Home_BARTHAG", "Away_BARTHAG", "Market_Spread", "Pred_Margin", "Pick_Spread",
        "Confidence_Level", "Edge_Points", "Has_Rebound_Advantage", "Spread_Prob", "Over_Under", "Pick_Total",
    ]
    # Use 2026 in Odds_File display name to match stats year (e.g. March_07_2026_Odds.csv)
    try:
        _dt = datetime.strptime(game_date, "%Y-%m-%d")
        odds_file_display = f"{_dt.strftime('%B_%d')}_{ODDS_YEAR}_Odds.csv"
    except Exception:
        odds_file_display = odds_path.name
    rows = []
    for r in value_plays:
        rows.append({
            "Date": game_date,
            "Odds_File": odds_file_display,
            "Home": r["Home"],
            "Away": r["Away"],
            "Home_BARTHAG": r.get("Home_BARTHAG"),
            "Away_BARTHAG": r.get("Away_BARTHAG"),
            "Market_Spread": r["Market_Spread"],
            "Pred_Margin": r["Pred_Margin"],
            "Pick_Spread": r["Pick_Spread"],
            "Confidence_Level": r["Confidence"],
            "Edge_Points": r["Edge"],
            "Has_Rebound_Advantage": bool(r.get("Rebounding_Star", False)),
            "Spread_Prob": r["Spread_Prob"],
            "Over_Under": r["Over_Under"],
            "Pick_Total": r["Pick_Total"],
        })
    if not rows:
        return 0
    df_new = pd.DataFrame(rows)
    if hist_path.exists():
        existing = pd.read_csv(hist_path)
        for c in columns:
            if c not in existing.columns:
                existing[c] = None
        # Keep only one set of plays per date: drop any existing rows with today's date before appending
        if "Date" in existing.columns:
            existing["Date"] = existing["Date"].astype(str).str.strip()
            existing = existing[existing["Date"] != game_date]
        df_new = df_new[[c for c in columns if c in df_new.columns]]
        combined = pd.concat([existing, df_new], ignore_index=True)
    else:
        combined = df_new
    combined.to_csv(hist_path, index=False)
    return len(rows)


if __name__ == "__main__":
    main()
