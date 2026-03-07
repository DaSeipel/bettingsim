"""
Value plays pipeline: fetch odds, build value plays, select POTD, write cache.
Runs outside Streamlit (e.g. from scripts/run_pipeline_to_cache.py) to avoid UI memory issues.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from engine.betting_models import (
    build_feature_row_for_upcoming_game,
    build_ncaab_feature_row_from_team_stats,
    consensus_moneyline,
    consensus_spread,
    consensus_totals,
    get_feature_based_reasoning,
    get_feature_row_for_game,
    get_spread_predicted_margin,
    get_top_shap_reasoning,
    load_feature_matrix_for_inference,
    NCAAB_KENPOM_SPREAD_FEATURE_COLUMNS,
)
from engine.clv_tracker import (
    record_recommendations as clv_record_recommendations,
    update_closing_odds as clv_update_closing_odds,
)
from engine.engine import (
    BASKETBALL_NCAAB,
    NBA_LEAGUE_AVG_OFF_RATING,
    NBA_LEAGUE_AVG_PACE,
    get_live_odds,
    get_nba_team_pace_stats,
    get_nba_teams_back_to_back,
    get_schedule_fatigue_penalty,
    get_team_power_ratings,
)
from engine.espn_odds import get_espn_live_odds_with_stats
from engine.rundown_odds import get_rundown_live_odds_with_stats
from engine.injury_scraper import add_injury_features
from engine.ncaab_march_context import add_ncaab_march_context_to_df
from engine.odds_quota import get_quota_status
from engine.play_history import archive_value_plays
from strategies.strategies import (
    HALF_UNIT_PCT,
    MAX_KELLY_PCT_WALTERS,
    american_to_decimal,
    damp_probability,
    fractional_kelly_half_units,
    implied_probability_fair_two_sided,
    implied_probability_no_vig,
    in_house_spread_from_ratings,
    kelly_fraction,
    key_number_value_adjustment,
    model_prob_from_in_house_spread,
    model_prob_from_in_house_total,
    model_prob_from_ratings_moneyline,
    predict_nba_total,
)

# -----------------------------------------------------------------------------
# Constants (mirror app.py)
# -----------------------------------------------------------------------------
LIVE_ODDS_SPORT_KEYS = [BASKETBALL_NCAAB]
EV_EPSILON_MIN_PCT = 2.0
EV_EPSILON_MAX_PCT = 15.0
MIN_BOOKMAKERS_VALUE_PLAY = 3
HIGH_RISK_ODDS_AMERICAN = 500
POTD_MIN_EDGE_PCT = 4.0
POTD_HIGH_CONFIDENCE_EDGE_PCT = 8.0
POTD_LARGE_SPREAD_POINTS = 12.0
POTD_LARGE_SPREAD_EDGE_PENALTY = 0.30
ARCHIVE_MIN_EDGE_PCT = 6.0
ARCHIVE_MAX_PLAYS_PER_DAY = 10
# NCAAB: min edge 2%. Spreads only: require 3 <= |line_error| <= 20 (moneylines have no line error filter)
NCAAB_SPREAD_MIN_EDGE_PCT = 2.0
SPREAD_LINE_ERROR_MIN_PTS = 3.0
SPREAD_LINE_ERROR_MAX_PTS = 20.0
MAX_VALUE_PLAYS_CACHE = 10
# Diversity cap: reserve slots for underdog vs favorite so dashboard shows a mix
DIVERSITY_UNDERDOG_SLOTS = 5
DIVERSITY_FAVORITE_SLOTS = 5
DIVERSITY_FLEX_SLOTS = 5
MARKET_LABELS = {"h2h": "Winner", "spreads": "Spread", "totals": "Over/Under"}
BANKROLL_FOR_STAKES = 1000.0

# Mascot suffixes to strip for NCAAB game matching (ESPN vs Rundown)
_NCAAB_CORE_SUFFIXES = (
    " cyclones", " wildcats", " volunteers", " commodores", " bulldogs", " tigers", " crimson tide",
    " tar heels", " blue devils", " jayhawks", " cardinals", " hurricanes", " seminoles",
    " gators", " gamecocks", " razorbacks", " rebels", " aggies", " horned frogs", " sooners",
    " longhorns", " red raiders", " cougars", " bearcats", " mountaineers", " buckeyes",
    " wolverines", " spartans", " hoosiers", " boilermakers", " fighting illini", " hawkeyes",
    " sun devils", " bruins", " trojans", " huskies", " shockers", " panthers", " orange",
    " demon deacons", " eagles", " ramblers", " revolutionaries", " sharks", " seahawks",
    " catamounts", " big green", " big red", " crimson", " highlanders", " wolfpack",
    " mustangs", " miners", " owls", " bearkats", " flames", " hilltoppers", " roadrunners",
    " mavericks", " lakers", " skyhawks", " spiders", " rams", " dukes", " bonnies",
    " blue demons", " musketeers", " fighting irish", " yellow jackets", " blue hens",
    " great danes", " river hawks", " retrievers", " explorers", " hawks", " billikens",
    " patriots", " black bears", " lobos", " broncos", " golden bears", " pirates",
    " paladins", " braves", " red foxes", " bobcats", " golden eagles", " mountaineers",
    " lancers", " thunderbirds", " pilots", " dones", " chanticleers", " tritons",
    " gauchos", " wolverines", " trailblazers", " hornets", " vandals", " falcons",
    " wolf pack", " redhawks", " rainbow warriors", " beach",
)


def _ncaab_team_core(s: str) -> str:
    """Normalize team name for matching: lowercase, strip, drop common mascot."""
    if not s or not isinstance(s, str):
        return ""
    n = " ".join(s.lower().strip().split())
    for suffix in _NCAAB_CORE_SUFFIXES:
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
            break
    return n


def _ncaab_game_key(away: str, home: str) -> tuple[str, str]:
    """Canonical key for matching: (core_away, core_home)."""
    return (_ncaab_team_core(away), _ncaab_team_core(home))


def _get_ncaab_odds_espn_primary_rundown_supplement(
    app_root: Path,
    commence_on_date: date | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    ESPN = primary NCAAB odds (full slate). Rundown = supplement; merge by team match.
    Returns (combined_df, meta) with meta keys: ncaab_espn_games, ncaab_rundown_games, ncaab_merged_games, ncaab_odds_rows.
    """
    from engine.espn_odds import get_espn_live_odds
    from engine.rundown_odds import get_rundown_live_odds

    empty_df = pd.DataFrame(columns=[
        "sport_key", "league", "event_id", "commence_time", "home_team", "away_team",
        "event_name", "market_type", "selection", "point", "odds",
    ])
    meta: dict = {}

    espn_df = get_espn_live_odds(
        sport_keys=[BASKETBALL_NCAAB],
        commence_on_date=commence_on_date,
        ncaab_include_all_today=True,
    )
    if espn_df.empty:
        espn_df = empty_df.copy()
    elif "league" in espn_df.columns:
        espn_df = espn_df[espn_df["league"].astype(str).str.strip().str.upper() == "NCAAB"].copy()
    n_espn = espn_df["event_id"].nunique() if not espn_df.empty and "event_id" in espn_df.columns else 0
    meta["ncaab_espn_games"] = n_espn
    meta["ncaab_espn_rows"] = len(espn_df)

    rundown_df = get_rundown_live_odds(
        sport_keys=[BASKETBALL_NCAAB],
        commence_on_date=commence_on_date,
    )
    if rundown_df.empty:
        live_odds_df = espn_df
        meta["ncaab_rundown_games"] = 0
        meta["ncaab_rundown_rows"] = 0
        meta["ncaab_merged_games"] = n_espn
        meta["ncaab_odds_rows"] = len(espn_df)
        return (live_odds_df, meta)

    rundown_df = rundown_df[rundown_df["league"].astype(str).str.strip().str.upper() == "NCAAB"].copy()
    n_rundown_raw = rundown_df["event_id"].nunique() if not rundown_df.empty else 0
    meta["ncaab_rundown_games"] = n_rundown_raw
    meta["ncaab_rundown_rows"] = len(rundown_df)

    # Build ESPN canonical lookup: game_key -> first row's event_id, event_name, home_team, away_team, commence_time
    espn_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in espn_df.drop_duplicates(subset=["event_id"]).iterrows():
        key = _ncaab_game_key(str(row.get("away_team", "")), str(row.get("home_team", "")))
        if key not in espn_lookup:
            espn_lookup[key] = {
                "event_id": row.get("event_id"),
                "event_name": row.get("event_name"),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "commence_time": row.get("commence_time"),
            }
        # Also add reversed key (home/away swap) so we match regardless of order
        key_rev = (key[1], key[0])
        if key_rev not in espn_lookup:
            espn_lookup[key_rev] = espn_lookup[key]

    # Relabel Rundown rows with ESPN canonical when match found
    def _canonical(row: pd.Series) -> dict[str, Any]:
        key = _ncaab_game_key(str(row.get("away_team", "")), str(row.get("home_team", "")))
        canonical = espn_lookup.get(key)
        if canonical:
            return {**row.to_dict(), **canonical}
        return row.to_dict()

    rundown_relabeled = pd.DataFrame([
        _canonical(rundown_df.iloc[i]) for i in range(len(rundown_df))
    ])
    live_odds_df = pd.concat([espn_df, rundown_relabeled], ignore_index=True)
    meta["ncaab_merged_games"] = live_odds_df["event_id"].nunique() if not live_odds_df.empty else 0
    meta["ncaab_odds_rows"] = len(live_odds_df)
    return (live_odds_df, meta)


def _format_start_time(commence_time: str) -> str:
    """Format ISO commence_time for display in Eastern Time."""
    if not commence_time or not str(commence_time).strip():
        return "—"
    try:
        from zoneinfo import ZoneInfo
        s = str(commence_time).replace("Z", "+00:00")
        dt_utc = datetime.fromisoformat(s)
        if dt_utc.tzinfo is None:
            return dt_utc.strftime("%b %d, %I:%M %p")
        dt_et = dt_utc.astimezone(ZoneInfo("America/New_York"))
        return dt_et.strftime("%b %d, %I:%M %p ET")
    except (ValueError, TypeError):
        return "—"


def _parse_event_teams(event_str: str) -> tuple[str, str]:
    """Parse 'Away @ Home' or 'Away vs Home' into (away_team, home_team)."""
    s = (event_str or "").strip()
    for sep in (" @ ", " vs ", " at "):
        if sep in s:
            parts = s.split(sep, 1)
            return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else (s, "")
    return (s, "")


def _bookmaker_counts(odds_df: pd.DataFrame) -> pd.DataFrame:
    """Count distinct bookmakers per (event_name, market_type, selection, point)."""
    if odds_df.empty or not all(c in odds_df.columns for c in ["event_name", "market_type", "selection", "point"]):
        return pd.DataFrame(columns=["event_name", "market_type", "selection", "point", "bookmaker_count"])
    grouped = odds_df.groupby(["event_name", "market_type", "selection", "point"], dropna=False).size().reset_index(name="bookmaker_count")
    return grouped


def _aggregate_odds_best_line_avg_implied(odds_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate odds across bookmakers: one row per (event, market, selection, point). NCAAB only."""
    if odds_df.empty or "odds" not in odds_df.columns:
        return odds_df
    if "league" in odds_df.columns:
        odds_df = odds_df[odds_df["league"].astype(str).str.strip().str.upper() == "NCAAB"].copy()
    if odds_df.empty:
        return odds_df
    odds_df = odds_df.loc[odds_df["odds"].notna()].copy()
    if odds_df.empty:
        return odds_df
    key_cols = ["event_id", "sport_key", "league", "commence_time", "home_team", "away_team", "event_name", "market_type", "selection", "point"]
    missing = [c for c in key_cols if c not in odds_df.columns]
    if missing:
        return odds_df

    def _mean_implied(ser: pd.Series) -> float:
        return float(np.mean([implied_probability_no_vig(float(o)) for o in ser]))

    agg = odds_df.groupby(key_cols, dropna=False).agg(
        odds=("odds", "max"),
        avg_implied_prob=("odds", _mean_implied),
        bookmaker_count=("odds", "count"),
    ).reset_index()
    agg["fair_implied_prob"] = agg["avg_implied_prob"].copy()
    spread_mask = agg["market_type"].astype(str).str.strip().str.lower() == "spreads"
    if spread_mask.any():
        spread_df = agg.loc[spread_mask]
        for (eid, league, ct, mt), g in spread_df.groupby(["event_id", "league", "commence_time", "market_type"]):
            if len(g) == 2:
                idx = g.index.tolist()
                o1, o2 = float(agg.loc[idx[0], "odds"]), float(agg.loc[idx[1], "odds"])
                f1, f2 = implied_probability_fair_two_sided(o1, o2)
                agg.loc[idx[0], "fair_implied_prob"] = f1
                agg.loc[idx[1], "fair_implied_prob"] = f2
    return agg


def _live_odds_to_value_plays(
    odds_df: pd.DataFrame,
    bankroll: float,
    kelly_frac: float = 0.25,
    min_ev_pct: float = EV_EPSILON_MIN_PCT,
    max_ev_pct: float = EV_EPSILON_MAX_PCT,
    include_high_risk: bool = False,
    pace_stats: Optional[dict] = None,
    b2b_teams: Optional[set] = None,
    as_of_date: Optional[date] = None,
    seed: int = 44,
    feature_matrix: Optional[pd.DataFrame] = None,
    debug: bool = False,
    min_bookmakers_override: Optional[int] = None,
    debug_failed_underdogs: Optional[list] = None,
    verbose_stats: Optional[dict] = None,
) -> tuple[pd.DataFrame, int]:
    """In-house line vs market; key-number adjustment; fractional Kelly. Returns (value_plays_df, n_flagged)."""
    if odds_df.empty:
        return (
            pd.DataFrame(
                columns=["League", "Event", "Selection", "Market", "Odds", "Value (%)", "Recommended Stake", "Injury Alert", "Start Time"]
            ),
            0,
        )
    if verbose_stats is not None and "event_id" in odds_df.columns:
        verbose_stats["ncaab_events_in_agg"] = odds_df["event_id"].nunique()
        verbose_stats["ncaab_events_with_features_set"] = set()
    np.random.seed(seed)
    pace_stats = pace_stats or {}
    b2b_teams = b2b_teams or set()
    as_of_date = as_of_date or date.today()
    power_ratings = get_team_power_ratings(pace_stats, NBA_LEAGUE_AVG_PACE, NBA_LEAGUE_AVG_OFF_RATING)
    default_rating = (NBA_LEAGUE_AVG_PACE * NBA_LEAGUE_AVG_OFF_RATING) / 100.0
    rows = []
    n_flagged = 0
    _debug_relax = os.environ.get("DEBUG_RELAX_FILTERS", "") == "1"
    _min_books = min_bookmakers_override if min_bookmakers_override is not None else (1 if (debug and _debug_relax) else MIN_BOOKMAKERS_VALUE_PLAY)

    _cur_pred_margin: Optional[float] = None
    _cur_line_error: Optional[float] = None
    for _, r in odds_df.iterrows():
        _cur_pred_margin, _cur_line_error = None, None
        if "bookmaker_count" in r.index and r.get("bookmaker_count") is not None and not pd.isna(r.get("bookmaker_count")):
            if int(r.get("bookmaker_count", 0)) < _min_books:
                continue
        odds_val = r.get("odds")
        if pd.isna(odds_val) or abs(float(odds_val)) < 100:
            continue
        odds_val = float(odds_val)
        if odds_val > HIGH_RISK_ODDS_AMERICAN and not include_high_risk:
            continue
        home_team = r.get("home_team", "")
        away_team = r.get("away_team", "")
        market_type = r.get("market_type", "")
        point = r.get("point")
        selection = str(r.get("selection", ""))
        league = str(r.get("league", ""))
        league_lookup = league.strip().lower() if league else ""
        commence_time = r.get("commence_time")
        game_date_str = None
        if commence_time:
            try:
                dt = datetime.fromisoformat(str(commence_time).replace("Z", "+00:00"))
                game_date_str = dt.strftime("%Y-%m-%d")
            except Exception:
                pass
        feature_row = (
            get_feature_row_for_game(feature_matrix, home_team, away_team, league_lookup, game_date=game_date_str)
            if feature_matrix is not None and not feature_matrix.empty
            else None
        )
        if feature_row is None and (league_lookup or "").strip().lower() == "ncaab":
            feature_row = build_ncaab_feature_row_from_team_stats(
                home_team, away_team, game_date=game_date_str
            )
        if verbose_stats is not None and league_lookup == "ncaab" and feature_row is not None:
            eid = r.get("event_id")
            if eid is not None:
                verbose_stats.setdefault("ncaab_events_with_features_set", set()).add(eid)

        model_prob = None
        consensus_ok = True
        if market_type == "totals" and point is not None:
            market_total = float(point)
            in_house_total = predict_nba_total(home_team, away_team, pace_stats, b2b_teams=b2b_teams)
            prefer_over = "Over" in selection or "over" in selection.lower()
            fallback = model_prob_from_in_house_total(in_house_total, market_total, prefer_over)
            model_prob, consensus_ok = consensus_totals(
                feature_row, market_total, prefer_over, in_house_total, league, fallback
            )
            if _debug_relax:
                consensus_ok = True
            if not consensus_ok:
                continue
        elif market_type == "spreads" and point is not None:
            market_spread = -float(point)
            if league_lookup == "ncaab":
                try:
                    pred_m = get_spread_predicted_margin(
                        feature_row, market_spread, league_lookup,
                        home_team=home_team, away_team=away_team,
                    )
                    line_err = (pred_m - market_spread) if pred_m is not None else None
                    _cur_pred_margin, _cur_line_error = pred_m, line_err
                except (TypeError, ValueError):
                    pass
                if verbose_stats is not None:
                    verbose_stats.setdefault("ncaab_spread_considered", 0)
                    verbose_stats["ncaab_spread_considered"] += 1
                    verbose_stats.setdefault("all_spread_line_errors", []).append({
                        "event": r.get("event_name", ""),
                        "selection": selection,
                        "point": float(point),
                        "market_spread": market_spread,
                        "pred_margin": round(_cur_pred_margin, 2) if _cur_pred_margin is not None else None,
                        "line_error": round(_cur_line_error, 2) if _cur_line_error is not None else None,
                    })
            # Debug: capture KenPom features for Duke/UNC to investigate extreme line errors
            if verbose_stats is not None and feature_row is not None and "debug_duke_unc_features" not in verbose_stats:
                evt = (r.get("event_name") or "").lower()
                ht = (home_team or "").lower()
                at = (away_team or "").lower()
                if ("duke" in evt or "duke" in ht or "duke" in at) and ("north carolina" in evt or "unc" in evt or "north carolina" in ht or "north carolina" in at):
                    raw = {}
                    for col in NCAAB_KENPOM_SPREAD_FEATURE_COLUMNS:
                        if col in feature_row.index:
                            v = feature_row[col]
                            raw[col] = float(v) if v is not None and not (isinstance(v, float) and pd.isna(v)) else None
                    verbose_stats["debug_duke_unc_features"] = {
                        "event": r.get("event_name", ""),
                        "home_team": home_team,
                        "away_team": away_team,
                        "market_spread": market_spread,
                        "pred_margin": _cur_pred_margin,
                        "line_error": _cur_line_error,
                        "kenpom_features": raw,
                    }
            home_rating = power_ratings.get(home_team, default_rating) - get_schedule_fatigue_penalty(home_team, as_of_date)
            away_rating = power_ratings.get(away_team, default_rating) - get_schedule_fatigue_penalty(away_team, as_of_date)
            in_house_spread = in_house_spread_from_ratings(home_rating, away_rating)
            we_cover_favorite = "-" in selection
            fallback = model_prob_from_in_house_spread(in_house_spread, market_spread, we_cover_favorite)
            model_prob, consensus_ok = consensus_spread(
                feature_row, market_spread, we_cover_favorite, in_house_spread, fallback, league=league_lookup
            )
            # NCAAB: only one spread model — skip consensus and pass on edge % alone
            if league_lookup == "ncaab":
                consensus_ok = True
            elif _debug_relax:
                consensus_ok = True
            if not consensus_ok:
                continue
        elif market_type == "h2h":
            home_rating = power_ratings.get(home_team, default_rating) - get_schedule_fatigue_penalty(home_team, as_of_date)
            away_rating = power_ratings.get(away_team, default_rating) - get_schedule_fatigue_penalty(away_team, as_of_date)
            selection_is_home = (
                str(selection).strip().lower() == str(home_team).strip().lower()
                or str(home_team).strip().lower() in str(selection).strip().lower()
                or str(selection).strip().lower() in str(home_team).strip().lower()
            )
            fallback = model_prob_from_ratings_moneyline(home_rating, away_rating, selection_is_home)
            model_prob, consensus_ok = consensus_moneyline(
                feature_row, selection_is_home, home_rating, away_rating, fallback, league=league_lookup
            )
            if _debug_relax:
                consensus_ok = True
            if not consensus_ok:
                continue
        else:
            implied = implied_probability_no_vig(odds_val)
            edge = np.random.uniform(0, 0.08)
            model_prob = damp_probability(min(0.92, implied + edge))

        implied_prob = r.get("fair_implied_prob")
        if implied_prob is None or pd.isna(implied_prob):
            implied_prob = r.get("avg_implied_prob")
        if implied_prob is None or pd.isna(implied_prob):
            implied_prob = implied_probability_no_vig(odds_val)
        implied_prob = float(implied_prob)
        is_underdog_side = market_type == "spreads" and "+" in selection
        ev_decimal = (model_prob * american_to_decimal(odds_val)) - 1.0
        ev_pct = ev_decimal * 100.0
        line_for_key = float(point) if point is not None else 0.0
        ev_pct += key_number_value_adjustment(line_for_key, league, market_type)
        if verbose_stats is not None and league_lookup == "ncaab" and market_type == "h2h":
            verbose_stats.setdefault("ncaab_h2h_raw_edges", []).append({
                "event": r.get("event_name", ""),
                "selection": selection,
                "model_prob": round(float(model_prob), 4),
                "implied_prob": round(implied_prob, 4),
                "raw_edge_pct": round(ev_pct, 2),
            })
        if ev_pct < min_ev_pct:
            if verbose_stats is not None and market_type == "spreads" and league_lookup == "ncaab":
                verbose_stats.setdefault("near_misses", []).append({
                    "event": r.get("event_name", ""),
                    "selection": selection,
                    "point": float(point) if point is not None else None,
                    "edge_pct": round(ev_pct, 2),
                    "pred_margin": _cur_pred_margin,
                    "line_error": round(_cur_line_error, 2) if _cur_line_error is not None else None,
                })
            if debug_failed_underdogs is not None and market_type == "spreads" and point is not None:
                try:
                    pt_float = float(point)
                    if pt_float > 0:  # underdog spread
                        market_spread = -pt_float
                        pred_margin = get_spread_predicted_margin(
                            feature_row, market_spread, league_lookup,
                            home_team=home_team, away_team=away_team,
                        )
                        line_error = (pred_margin - market_spread) if pred_margin is not None else None
                        debug_failed_underdogs.append({
                            "event": r.get("event_name", ""),
                            "selection": selection,
                            "point": pt_float,
                            "pred_margin": round(pred_margin, 2) if pred_margin is not None else None,
                            "edge_pct": round(ev_pct, 2),
                            "line_error": round(line_error, 2) if line_error is not None else None,
                        })
                except (TypeError, ValueError):
                    pass
            continue
        if ev_pct >= max_ev_pct:
            n_flagged += 1
            continue
        full_kelly = kelly_fraction(odds_val, model_prob, fraction=1.0)
        stake = fractional_kelly_half_units(bankroll, full_kelly, HALF_UNIT_PCT, MAX_KELLY_PCT_WALTERS)
        if stake <= 0:
            continue
        is_home_b2b = home_team in b2b_teams
        is_away_b2b = away_team in b2b_teams
        if verbose_stats is not None and league_lookup == "ncaab" and market_type == "spreads":
            verbose_stats.setdefault("ncaab_spread_passed", 0)
            verbose_stats["ncaab_spread_passed"] += 1
        line_error_for_row: Optional[float] = None
        if league_lookup == "ncaab" and market_type == "spreads":
            line_error_for_row = round(_cur_line_error, 2) if _cur_line_error is not None else None
        rows.append({
            "League": league,
            "Event": r.get("event_name", ""),
            "Selection": selection,
            "Market": market_type,
            "Odds": int(round(odds_val)),
            "Value (%)": round(ev_pct, 2),
            "Recommended Stake": stake,
            "Injury Alert": "—",
            "Start Time": _format_start_time(r.get("commence_time", "")),
            "commence_time": r.get("commence_time", ""),
            "model_prob": model_prob,
            "implied_prob": implied_prob,
            "point": point,
            "home_team": home_team,
            "away_team": away_team,
            "is_home_b2b": is_home_b2b,
            "is_away_b2b": is_away_b2b,
            "underdog_value": is_underdog_side and (model_prob > implied_prob),
            "line_error": line_error_for_row,
        })
    return pd.DataFrame(rows), n_flagged


def _apply_diversity_cap(df: pd.DataFrame, max_plays: int = MAX_VALUE_PLAYS_CACHE) -> pd.DataFrame:
    """
    Cap to max_plays by reserving at least 5 slots for underdog spreads (point > 0),
    5 for favorite spreads (point < 0), then filling the rest with next best regardless of side.
    If there aren't enough underdogs/favorites with sufficient edge, take as many as available.
    """
    if df.empty or max_plays <= 0:
        return df
    df = df.sort_values("Value (%)", ascending=False).reset_index(drop=True)
    pt = pd.to_numeric(df.get("point"), errors="coerce")
    is_spread = (df.get("Market", pd.Series(dtype=object)).astype(str).str.strip().str.lower() == "spreads")
    underdog_mask = is_spread & (pt > 0)
    favorite_mask = is_spread & (pt < 0)
    underdogs = df.loc[underdog_mask].sort_values("Value (%)", ascending=False).head(DIVERSITY_UNDERDOG_SLOTS)
    favorites = df.loc[favorite_mask].sort_values("Value (%)", ascending=False).head(DIVERSITY_FAVORITE_SLOTS)
    taken_idx = set(underdogs.index) | set(favorites.index)
    remaining = df.loc[~df.index.isin(taken_idx)].sort_values("Value (%)", ascending=False)
    flex_n = max(0, max_plays - len(underdogs) - len(favorites))
    flex = remaining.head(flex_n)
    out = pd.concat([underdogs, favorites, flex], ignore_index=True)
    out = out.sort_values("Value (%)", ascending=False).head(max_plays).reset_index(drop=True)
    return out


def _get_injury_alerts(_sport: str) -> dict[str, str]:
    """Return event_name -> injury alert text. No external API; returns empty."""
    return {}


def add_injury_alerts_to_value_plays(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    """Add 'Injury Alert' column. '—' when no alert."""
    if df.empty:
        df["Injury Alert"] = []
        return df
    alerts_map = _get_injury_alerts(sport)
    df = df.copy()
    df["Injury Alert"] = df["Event"].map(lambda e: alerts_map.get(e, "") or "—")
    cols = list(df.columns)
    cols.remove("Injury Alert")
    idx = cols.index("Event") + 1 if "Event" in cols else 0
    cols.insert(idx, "Injury Alert")
    return df[cols]


def select_play_of_the_day(
    value_plays_df: pd.DataFrame,
    live_odds_df: pd.DataFrame,
    min_edge_pct: float = POTD_MIN_EDGE_PCT,
) -> dict[str, Optional[dict]]:
    """Select two NCAAB POTD picks. Pick 1 = highest edge moneyline. Pick 2 = highest edge spread (3–20 pt line error); if none, 2nd moneyline."""
    result: dict[str, Optional[dict]] = {"NCAAB Pick 1": None, "NCAAB Pick 2": None}
    if value_plays_df.empty or "League" not in value_plays_df.columns:
        return result
    eligible = value_plays_df[
        (value_plays_df["League"].astype(str).str.strip().str.upper() == "NCAAB")
        & (value_plays_df["Value (%)"] > min_edge_pct)
    ].copy()
    if eligible.empty:
        return result
    market_str = eligible["Market"].astype(str).str.strip().str.lower()
    moneylines = eligible.loc[market_str == "h2h"].sort_values("Value (%)", ascending=False).reset_index(drop=True)
    spreads = eligible.loc[market_str == "spreads"].sort_values("Value (%)", ascending=False).reset_index(drop=True)

    def _is_large_spread(row: pd.Series) -> bool:
        if str(row.get("Market", "")).strip().lower() != "spreads":
            return False
        pt = row.get("point_x") if "point_x" in row.index else row.get("point")
        try:
            return abs(float(pt)) > POTD_LARGE_SPREAD_POINTS
        except (TypeError, ValueError):
            return False

    def _row_to_pick(top: pd.Series, label: str) -> dict:
        point_val = top.get("point_x") if "point_x" in top.index else top.get("point")
        odds_raw = top.get("Odds", 0)
        odds_for_pick = None
        if odds_raw is not None and not pd.isna(odds_raw):
            try:
                odds_for_pick = int(round(float(odds_raw)))
            except (TypeError, ValueError):
                pass
        commence_time = top.get("commence_time") or top.get("commence_time_x") or ""
        return {
            "League": label,
            "Event": top.get("Event", ""),
            "Selection": top.get("Selection", ""),
            "Market": top.get("Market", ""),
            "Odds": odds_for_pick,
            "Value (%)": float(top.get("Value (%)", 0)),
            "point": point_val,
            "Recommended Stake": top.get("Recommended Stake", 0),
            "Start Time": top.get("Start Time", "—"),
            "commence_time": commence_time,
            "Injury Alert": top.get("Injury Alert", "—"),
            "high_variance": _is_large_spread(top),
            "home_team": str(top.get("home_team", "")).strip() or "—",
            "away_team": str(top.get("away_team", "")).strip() or "—",
            "model_prob": float(top.get("model_prob", 0)),
            "confidence_tier": str(top.get("confidence_tier", "Medium")).strip() or "Medium",
            "reasoning_summary": top.get("reasoning_summary") if pd.notna(top.get("reasoning_summary")) else None,
            "line_error": top.get("line_error"),
        }

    if not moneylines.empty:
        result["NCAAB Pick 1"] = _row_to_pick(moneylines.iloc[0], "NCAAB Pick 1")
    if not spreads.empty:
        result["NCAAB Pick 2"] = _row_to_pick(spreads.iloc[0], "NCAAB Pick 2")
    elif len(moneylines) >= 2:
        result["NCAAB Pick 2"] = _row_to_pick(moneylines.iloc[1], "NCAAB Pick 2")
    return result


def _potd_reason(
    row: Any,
    feature_matrix: Optional[pd.DataFrame] = None,
    b2b_teams: Optional[set[str]] = None,
) -> str:
    """Build 3–4 sentence explanation for POTD (used when writing cache)."""
    market = str(row.get("Market", ""))
    market_label = MARKET_LABELS.get(market, market)
    selection = str(row.get("Selection", ""))
    event = str(row.get("Event", ""))
    edge_pct = float(row.get("Value (%)", 0))
    league_raw = str(row.get("League", ""))
    league = "NCAAB" if league_raw and "NCAAB" in league_raw.strip().upper() else league_raw.strip()
    league_shap = "ncaab" if league_raw and "ncaab" in league_raw.strip().lower() else (league_raw.strip().lower() if league_raw else "")
    point = row.get("point")
    away, home = _parse_event_teams(event)
    point_str = ""
    if point is not None:
        try:
            pt = float(point)
            if market == "spreads":
                point_str = f" {pt:+.1f}"
            elif market == "totals" and f"{pt:.1f}" not in selection:
                point_str = f" {pt:.1f}"
        except (TypeError, ValueError):
            pass
    opening = (
        f"Our model sees {edge_pct:.1f}% expected value on {selection}{point_str} in this {market_label}—"
        f"the best {league} play we're highlighting today."
    )
    why_sentences: list[str] = []
    feature_row = None
    if feature_matrix is not None and not feature_matrix.empty and home and away:
        feature_row = get_feature_row_for_game(feature_matrix, home_team=home, away_team=away, league=league)
    if feature_row is None and home and away:
        game_date_str = None
        if row.get("commence_time"):
            try:
                dt = datetime.fromisoformat(str(row["commence_time"]).replace("Z", "+00:00"))
                game_date_str = dt.strftime("%Y-%m-%d")
            except Exception:
                pass
        feature_row = build_feature_row_for_upcoming_game(
            home_team=home,
            away_team=away,
            league=league_shap or league.strip().lower(),
            game_date=game_date_str,
            b2b_teams=b2b_teams,
        )
    if feature_row is not None:
        shap_reason = get_top_shap_reasoning(
            feature_row, market, home_team=home, away_team=away, top_k=4, league=league_shap or league.strip().lower()
        )
        if shap_reason:
            why_sentences.append(shap_reason)
        if not shap_reason:
            feat_reason = get_feature_based_reasoning(
                feature_row, market, home_team=home, away_team=away, top_k=4
            )
            if feat_reason:
                why_sentences.append(feat_reason)
    if not why_sentences:
        opponent = away if (selection.strip() == home.strip()) else home
        why_sentences.append(
            f"Our model gives {selection} a higher win probability than the odds imply against {opponent}."
        )
    return " ".join([opening] + why_sentences)


def _load_tournament_eligible_teams(app_root: Path) -> set[str]:
    """Load team names from data/ncaab_seeds.csv. Returns set of lowercased names."""
    p = app_root / "data" / "ncaab_seeds.csv"
    if not p.exists():
        return set()
    try:
        df = pd.read_csv(p)
        col = "team" if "team" in df.columns else df.columns[0]
        s = df[col].astype(str).str.strip().str.lower()
        return set(s[(s != "") & s.notna()].tolist())
    except Exception:
        return set()


def _json_sanitize(obj: Any) -> Any:
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(x) for x in obj]
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if pd.isna(obj):
        return None
    return obj


def _print_verbose_ncaab_h2h_edges(verbose_stats: dict) -> None:
    """Print raw edge for every NCAAB moneyline (h2h) before any threshold filtering. Highlight Vandy/Tennessee and Texas Tech/BYU."""
    edges = verbose_stats.get("ncaab_h2h_raw_edges", [])
    if not edges:
        print("\n--- NCAAB moneyline (h2h) raw edges: none in today's data ---")
        return
    print("\n--- NCAAB moneyline (h2h) raw edge calculations (before 2% threshold) ---")
    print(f"{'Event':<45} | {'Selection':<28} | {'model_prob':>10} | {'implied_prob':>12} | {'raw_edge_%':>10}")
    print("-" * 115)
    for rec in edges:
        evt = (rec.get("event") or "")[:45]
        sel = (rec.get("selection") or "")[:28]
        mp = rec.get("model_prob", 0)
        ip = rec.get("implied_prob", 0)
        edge = rec.get("raw_edge_pct", 0)
        print(f"  {evt:<45} | {sel:<28} | {mp:>10.4f} | {ip:>12.4f} | {edge:>+9.2f}%")
    # Highlight Vanderbilt @ Tennessee and Texas Tech vs BYU
    evt_lower = [((rec.get("event") or "").lower(), rec) for rec in edges]
    for evt_str, rec in evt_lower:
        if "vanderbilt" in evt_str and "tennessee" in evt_str:
            print(f"\n  >>> Vanderbilt @ Tennessee: {rec.get('selection')}  model_prob={rec.get('model_prob')}  implied_prob={rec.get('implied_prob')}  raw_edge_%={rec.get('raw_edge_pct')}%")
        if "texas tech" in evt_str and "byu" in evt_str:
            print(f"  >>> Texas Tech vs BYU: {rec.get('selection')}  model_prob={rec.get('model_prob')}  implied_prob={rec.get('implied_prob')}  raw_edge_%={rec.get('raw_edge_pct')}%")
    print("---\n")


def _print_verbose_spread_stats(verbose_stats: dict) -> None:
    """Print NCAAB spread considered/passed, top 5 near-misses, and line errors for every spread."""
    considered = verbose_stats.get("ncaab_spread_considered", 0)
    passed = verbose_stats.get("ncaab_spread_passed", 0)
    print("\n--- NCAAB spread pipeline (verbose) ---")
    print(f"NCAAB spread plays considered (before {NCAAB_SPREAD_MIN_EDGE_PCT}% edge filter): {considered}")
    print(f"NCAAB spread plays passed (>= {NCAAB_SPREAD_MIN_EDGE_PCT}% edge): {passed}")

    near = verbose_stats.get("near_misses", [])
    if near:
        by_edge = sorted(near, key=lambda x: x.get("edge_pct", -999), reverse=True)
        top5 = by_edge[:5]
        print(f"\nTop 5 NCAAB spread plays closest to passing (passed consensus, edge < {NCAAB_SPREAD_MIN_EDGE_PCT}%):")
        for i, rec in enumerate(top5, 1):
            le = rec.get("line_error")
            le_str = f"{le:+.2f}" if le is not None else "N/A"
            pt = rec.get("point")
            pt_str = f"{pt:+.1f}" if pt is not None else "?"
            print(f"  {i}. {rec.get('selection', '')} ({pt_str})  |  {rec.get('event', '')[:45]}  |  Edge {rec.get('edge_pct', 0):.2f}%  |  Line error {le_str} pts")
    else:
        print(f"\nTop 5 near-misses: none (all NCAAB spread plays had edge >= {NCAAB_SPREAD_MIN_EDGE_PCT}% or failed other filters).")

    all_err = verbose_stats.get("all_spread_line_errors", [])
    if all_err:
        above_3 = [e for e in all_err if e.get("line_error") is not None and abs(e["line_error"]) > 3]
        print(f"\nLine error for every NCAAB spread game today ({len(all_err)} rows):")
        for e in all_err:
            le = e.get("line_error")
            le_str = f"{le:+.2f}" if le is not None else "N/A"
            print(f"  {e.get('event', '')[:40]}  |  {e.get('selection', '')} {e.get('point', 0):+.1f}  |  pred_margin={e.get('pred_margin')}  |  line_error={le_str} pts")
        print(f"\nGames with |line_error| > 3 pts: {len(above_3)}")
        alabama_75 = [e for e in all_err if "alabama" in (e.get("event") or "").lower() and (e.get("selection") or "").strip() == "Alabama" and e.get("point") == -7.5]
        if alabama_75:
            e = alabama_75[0]
            le = e.get("line_error")
            le_str = f"{le:+.2f} pts" if le is not None else "N/A"
            print(f"\n--- Alabama -7.5 (from spread line errors) ---  line_error = {le_str}  |  pred_margin = {e.get('pred_margin')} ---")
        # Top 5 spread plays today by |line_error| (largest disagreement with market first)
        with_abs = [(e, abs(e["line_error"]) if e.get("line_error") is not None else 0.0) for e in all_err]
        with_abs.sort(key=lambda x: x[1], reverse=True)
        top5_by_error = [e for e, _ in with_abs[:5]]
        if top5_by_error:
            print(f"\nTop 5 NCAAB spread plays today by |line_error|:")
            for i, e in enumerate(top5_by_error, 1):
                le = e.get("line_error")
                le_str = f"{le:+.2f}" if le is not None else "N/A"
                print(f"  {i}. {e.get('event', '')[:42]}  |  {e.get('selection', '')} {e.get('point', 0):+.1f}  |  pred_margin={e.get('pred_margin')}  |  line_error={le_str} pts")
    duke_unc = verbose_stats.get("debug_duke_unc_features")
    if duke_unc:
        print("\n--- Duke / North Carolina: raw KenPom features fed to spread model ---")
        print(f"Event: {duke_unc.get('event', '')}  |  home: {duke_unc.get('home_team')}  |  away: {duke_unc.get('away_team')}")
        print(f"market_spread={duke_unc.get('market_spread')}  |  pred_margin={duke_unc.get('pred_margin')}  |  line_error={duke_unc.get('line_error')}")
        kp = duke_unc.get("kenpom_features") or {}
        for k in sorted(kp.keys()):
            print(f"  {k}: {kp[k]}")
        print("---")
        print(f"\n>>> NCAAB spread plays passed (>= {NCAAB_SPREAD_MIN_EDGE_PCT}% edge): {passed} <<<\n")


def _print_verbose_auburn_alabama(
    live_odds_df: pd.DataFrame,
    verbose_stats: dict,
) -> None:
    """Print (1) Is Auburn @ Alabama in the 25 after future filter? (2) If yes: pred margin, closing spread, line error. (3) Exact exclusion reason."""
    if live_odds_df.empty or "event_name" not in live_odds_df.columns or "league" not in live_odds_df.columns:
        return
    ncaab = live_odds_df["league"].astype(str).str.strip().str.upper() == "NCAAB"
    sub = live_odds_df.loc[ncaab]
    if sub.empty:
        return
    event_names = sub["event_name"].astype(str).str.strip().drop_duplicates()
    auburn_alabama_events = [
        e for e in event_names
        if "auburn" in e.lower() and "alabama" in e.lower()
    ]
    in_future_25 = bool(auburn_alabama_events)
    event_str = auburn_alabama_events[0] if auburn_alabama_events else ""

    print("\n--- Auburn @ Alabama (requested debug) ---")
    print(f"  (1) Is Auburn @ Alabama in the games that pass the future filter?  {'Yes' if in_future_25 else 'No'}")
    if not in_future_25:
        print("  (2) N/A — game not in the 25.")
        print("  (3) N/A.")
        print("---\n")
        return

    all_err = verbose_stats.get("all_spread_line_errors", [])
    spread_rows = [
        e for e in all_err
        if "auburn" in (e.get("event") or "").lower() and "alabama" in (e.get("event") or "").lower()
    ]
    with_features = spread_rows and any(
        e.get("pred_margin") is not None or e.get("line_error") is not None
        for e in spread_rows
    )
    # Prefer a row that has pred_margin/line_error
    row = None
    for e in spread_rows:
        if e.get("pred_margin") is not None or e.get("line_error") is not None:
            row = e
            break
    if row is None and spread_rows:
        row = spread_rows[0]

    if row is not None:
        pred = row.get("pred_margin")
        spread_val = row.get("market_spread")
        le = row.get("line_error")
        print(f"  (2) Event: {event_str}")
        print(f"      Predicted margin: {pred if pred is not None else 'N/A'}")
        print(f"      Closing spread (home): {spread_val if spread_val is not None else 'N/A'}")
        print(f"      Line error (pts): {le if le is not None else 'N/A'}")
    else:
        print(f"  (2) Event: {event_str}")
        print(f"      Predicted margin: N/A (spread not evaluated)")
        print(f"      Closing spread: N/A")
        print(f"      Line error: N/A")

    # (3) Exact reason
    features_set = verbose_stats.get("ncaab_events_with_features_set") or set()
    # We don't have event_id in all_spread_line_errors; match by event name. Check if any event_id in sub for this event is in features_set.
    event_ids_for_game = sub.loc[sub["event_name"].astype(str).str.strip() == event_str, "event_id"].drop_duplicates().tolist()
    has_features = any(eid in features_set for eid in event_ids_for_game)

    if not has_features:
        print(f"  (3) Excluded: feature matrix match failed — no KenPom/team stats for this game (Auburn / Alabama).")
    elif row is None or (row.get("pred_margin") is None and row.get("line_error") is None):
        print(f"  (3) Excluded: spread not evaluated (no predicted margin / line error).")
    else:
        le_val = row.get("line_error")
        abs_le = abs(le_val) if le_val is not None else None
        in_line_range = (
            abs_le is not None
            and SPREAD_LINE_ERROR_MIN_PTS <= abs_le <= SPREAD_LINE_ERROR_MAX_PTS
        )
        near = verbose_stats.get("near_misses", [])
        auburn_alabama_near = [
            n for n in near
            if "auburn" in (n.get("event") or "").lower() and "alabama" in (n.get("event") or "").lower()
        ]
        failed_edge = bool(auburn_alabama_near)

        if not in_line_range and failed_edge:
            print(f"  (3) Excluded: (a) line error filter — |line_error| = {abs_le} pts not in {SPREAD_LINE_ERROR_MIN_PTS}–{SPREAD_LINE_ERROR_MAX_PTS} pt range; (b) edge filter — edge {auburn_alabama_near[0].get('edge_pct')}% below 2% threshold.")
        elif not in_line_range:
            print(f"  (3) Excluded: line error filter — |line_error| = {abs_le} pts not in {SPREAD_LINE_ERROR_MIN_PTS}–{SPREAD_LINE_ERROR_MAX_PTS} pt range.")
        elif failed_edge:
            print(f"  (3) Excluded: edge filter — edge {auburn_alabama_near[0].get('edge_pct')}% below 2% threshold.")
        else:
            print(f"  (3) Excluded: other (e.g. diversity cap or not in top value plays).")
    print("---\n")


def run_pipeline_to_cache(
    api_key: str,
    cache_path: Path,
    app_root: Optional[Path] = None,
    bankroll: float = BANKROLL_FOR_STAKES,
    kelly_frac: float = 0.25,
    include_high_risk: bool = False,
    march_madness_mode: bool = False,
    verbose: bool = False,
) -> None:
    """
    Run the full value-plays pipeline and write results to cache_path (JSON).
    Creates data/cache dir if needed. On error, writes a minimal cache with error field.
    """
    if app_root is None:
        app_root = Path(__file__).resolve().parent.parent
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_fallback(error_message: str) -> None:
        payload = {
            "value_plays": [],
            "potd_picks": {"NCAAB Pick 1": None, "NCAAB Pick 2": None},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "odds_source_meta": {},
            "value_plays_flagged_count": 0,
            "error": error_message,
        }
        with open(cache_path, "w") as f:
            json.dump(payload, f, indent=2)

    try:
        # NCAAB: ESPN primary, Rundown supplement (merge by team match for multiple bookmakers)
        meta: dict = {}
        as_of_date = date.today()
        live_odds_df, meta = _get_ncaab_odds_espn_primary_rundown_supplement(
            app_root=app_root,
            commence_on_date=as_of_date,
        )
        meta["used_ncaab_espn_primary"] = True
        n_ncaab_before_future = (
            live_odds_df.loc[live_odds_df["league"] == "NCAAB", "event_id"].nunique()
            if not live_odds_df.empty and "league" in live_odds_df.columns and "event_id" in live_odds_df.columns
            else 0
        )
        odds_event_ids_before_future: set[str] = set()
        if not live_odds_df.empty and "league" in live_odds_df.columns and "event_id" in live_odds_df.columns:
            ncaab_pre = live_odds_df[live_odds_df["league"].astype(str).str.strip().str.upper() == "NCAAB"]
            odds_event_ids_before_future = set(ncaab_pre["event_id"].astype(str).dropna().unique())
        if not live_odds_df.empty and "commence_time" in live_odds_df.columns:
            now_utc = datetime.now(timezone.utc)
            # Include games starting within the next 12 hours (e.g. Alabama 8:30pm ET)
            cutoff_end = now_utc + timedelta(hours=12)

            def _in_window(ct: Any) -> bool:
                if pd.isna(ct) or ct is None:
                    return False
                try:
                    dt = datetime.fromisoformat(str(ct).replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return now_utc <= dt <= cutoff_end
                except (ValueError, TypeError):
                    return False

            mask = live_odds_df["commence_time"].apply(_in_window)
            live_odds_df = live_odds_df.loc[mask].copy()

        n_ncaab_after_future = (
            live_odds_df.loc[live_odds_df["league"] == "NCAAB", "event_id"].nunique()
            if not live_odds_df.empty and "league" in live_odds_df.columns and "event_id" in live_odds_df.columns
            else 0
        )

        as_of_date = date.today()
        b2b_teams = set(get_nba_teams_back_to_back(api_key.strip(), as_of_date))
        feature_matrix = load_feature_matrix_for_inference(league=None)
        pace_stats = get_nba_team_pace_stats()
        use_fallback = True  # NCAAB uses ESPN + Rundown (no Odds API)
        min_ev = 1.5 if use_fallback else EV_EPSILON_MIN_PCT
        min_books_override = 1 if use_fallback else None

        if verbose:
            n_espn = meta.get("ncaab_espn_games", 0)
            n_rundown = meta.get("ncaab_rundown_games", 0)
            n_merged = meta.get("ncaab_merged_games", 0)
            n_rows = meta.get("ncaab_odds_rows", 0)
            print(f"NCAAB odds: ESPN primary ({n_espn} games), Rundown supplement ({n_rundown} games). Merged: {n_merged} games, {n_rows} odds rows.")
            # Master schedule as source of truth: flag games with no odds
            master_path = app_root / "data" / "cache" / "espn_master_schedule.json"
            if master_path.exists():
                try:
                    with open(master_path) as f:
                        master_data = json.load(f)
                    matchups = master_data.get("matchups") or []
                    odds_event_ids = odds_event_ids_before_future
                    no_odds: list[str] = []
                    for m in matchups:
                        if not isinstance(m, dict):
                            continue
                        gid = str(m.get("game_id", "")).strip()
                        away = (m.get("away_team") or "").strip()
                        home = (m.get("home_team") or "").strip()
                        if gid in odds_event_ids:
                            continue
                        no_odds.append(f"  {away} @ {home}")
                    if no_odds:
                        print("\n--- ESPN master schedule: No odds available (from ESPN or Rundown) ---")
                        for line in no_odds:
                            print(line)
                        print(f"--- {len(no_odds)} game(s) with no odds ---\n")
                except Exception as e:
                    if verbose:
                        print(f"[verbose] Could not load master schedule for no-odds check: {e}")

        vp_frames: list[pd.DataFrame] = []
        value_plays_flagged_count = 0
        debug_failed_underdogs: list = []
        verbose_stats: Optional[dict] = {} if verbose else None
        for league_name in ("NCAAB",):
            ev_for_league = NCAAB_SPREAD_MIN_EDGE_PCT if league_name == "NCAAB" else min_ev
            sub = (
                live_odds_df[live_odds_df["league"] == league_name].copy()
                if "league" in live_odds_df.columns and not live_odds_df.empty
                else pd.DataFrame()
            )
            if sub.empty:
                continue
            agg = _aggregate_odds_best_line_avg_implied(sub)
            if verbose and verbose_stats is not None and not agg.empty and "event_id" in agg.columns:
                verbose_stats["ncaab_events_after_odds_parsing"] = agg["event_id"].nunique()
            vp, flagged = _live_odds_to_value_plays(
                agg if not agg.empty else sub,
                bankroll=bankroll,
                kelly_frac=kelly_frac,
                min_ev_pct=ev_for_league,
                max_ev_pct=EV_EPSILON_MAX_PCT,
                include_high_risk=include_high_risk,
                pace_stats=pace_stats,
                b2b_teams=b2b_teams,
                as_of_date=as_of_date,
                feature_matrix=feature_matrix,
                min_bookmakers_override=min_books_override,
                debug_failed_underdogs=debug_failed_underdogs,
                verbose_stats=verbose_stats if verbose else None,
            )
            if not vp.empty:
                vp_frames.append(vp)
            value_plays_flagged_count += flagged

        if verbose and verbose_stats:
            n_before = n_ncaab_before_future
            n_after_future = n_ncaab_after_future
            n_after_odds = verbose_stats.get("ncaab_events_after_odds_parsing") or verbose_stats.get("ncaab_events_in_agg") or 0
            n_with_features = len(verbose_stats.get("ncaab_events_with_features_set") or set())
            print("\n--- NCAAB filter steps (Rundown API 87 → pipeline) ---")
            print(f"  (0) NCAAB events in odds df (Rundown; before pipeline filters): {n_before}")
            print(f"  (1) After future-only filter (start within next 12h):           {n_after_future}  (dropped {n_before - n_after_future})")
            print(f"  (2) After odds parsing (aggregated lines):                      {n_after_odds}  (dropped {n_after_future - n_after_odds})")
            print(f"  (3) After feature matrix match:                                {n_with_features}  (dropped {n_after_odds - n_with_features})")
            print("---\n")
            _print_verbose_ncaab_h2h_edges(verbose_stats)
            _print_verbose_spread_stats(verbose_stats)
            _print_verbose_auburn_alabama(live_odds_df, verbose_stats)

        value_plays_df = (
            pd.concat(vp_frames, ignore_index=True)
            if vp_frames
            else pd.DataFrame(
                columns=["League", "Event", "Selection", "Market", "Odds", "Value (%)", "Recommended Stake", "Injury Alert", "Start Time"]
            )
        )

        value_plays_df = add_injury_alerts_to_value_plays(value_plays_df, "Basketball")
        if not value_plays_df.empty:
            def _confidence_tier(row: pd.Series) -> str:
                edge = float(row.get("Value (%)", 0))
                return "High" if edge >= POTD_HIGH_CONFIDENCE_EDGE_PCT else "Medium"

            value_plays_df = value_plays_df.copy()
            value_plays_df["confidence_tier"] = value_plays_df.apply(_confidence_tier, axis=1)
            value_plays_df["reasoning_summary"] = value_plays_df.apply(
                lambda r: _potd_reason(r, feature_matrix=feature_matrix, b2b_teams=b2b_teams),
                axis=1,
            )

        # Moneylines: min edge 2%, no line error filter. Spreads: min edge 2% and 3 <= |line_error| <= 20
        if not value_plays_df.empty and "League" in value_plays_df.columns and "Market" in value_plays_df.columns:
            ncaab = value_plays_df["League"].astype(str).str.strip().str.upper() == "NCAAB"
            spread = value_plays_df["Market"].astype(str).str.strip().str.lower() == "spreads"
            ncaab_spread = ncaab & spread
            other = ~ncaab_spread
            spread_ok_edge = value_plays_df["Value (%)"] >= NCAAB_SPREAD_MIN_EDGE_PCT
            le = value_plays_df.get("line_error")
            if le is not None:
                abs_le = le.abs()
                spread_ok_line = le.notna() & (abs_le >= SPREAD_LINE_ERROR_MIN_PTS) & (abs_le <= SPREAD_LINE_ERROR_MAX_PTS)
            else:
                spread_ok_line = pd.Series(False, index=value_plays_df.index)
            if verbose:
                alabama = value_plays_df.loc[
                    ncaab_spread
                    & value_plays_df["Event"].astype(str).str.contains("Alabama", case=False, na=False)
                    & (value_plays_df["Selection"].astype(str).str.strip() == "Alabama")
                    & (pd.to_numeric(value_plays_df.get("point"), errors="coerce") == -7.5)
                ]
                if not alabama.empty:
                    row = alabama.iloc[0]
                    le_val = row.get("line_error")
                    le_str = f"{le_val:+.2f} pts" if le_val is not None else "N/A"
                    passes = bool(spread_ok_line.loc[alabama.index[0]] and spread_ok_edge.loc[alabama.index[0]])
                    print(f"\n--- Alabama -7.5 (requested debug) ---\n  line_error = {le_str}  |  passes 3–20 pt filter = {passes}\n---")
            keep = other | (ncaab_spread & spread_ok_edge & spread_ok_line)
            value_plays_df = value_plays_df.loc[keep].copy()
        value_plays_df = value_plays_df.sort_values("Value (%)", ascending=False).head(MAX_VALUE_PLAYS_CACHE).reset_index(drop=True)

        if verbose:
            print(f"Plays passed all filters: {len(value_plays_df)}.")
            if verbose:
                n_with_odds = n_ncaab_before_future
                n_after_12h = n_ncaab_after_future
                print(f"NCAAB games with odds (ESPN + Rundown merge): {n_with_odds}. After 12h window: {n_after_12h}. Value plays generated: {len(value_plays_df)}.")

        try:
            to_archive = value_plays_df[value_plays_df["Value (%)"] >= ARCHIVE_MIN_EDGE_PCT].copy()
            to_archive = to_archive.sort_values("Value (%)", ascending=False).head(ARCHIVE_MAX_PLAYS_PER_DAY)
            archive_value_plays(to_archive, as_of_date=as_of_date)
        except Exception:
            pass

        potd_picks = select_play_of_the_day(value_plays_df, live_odds_df, min_edge_pct=POTD_MIN_EDGE_PCT)
        for label in ("NCAAB Pick 1", "NCAAB Pick 2"):
            p = potd_picks.get(label)
            if not p:
                continue
            row_like = pd.Series(p) if not isinstance(p, pd.Series) else p
            p["reason"] = _potd_reason(row_like, feature_matrix=feature_matrix, b2b_teams=b2b_teams)

        try:
            potd_rows = []
            for label in ("NCAAB Pick 1", "NCAAB Pick 2"):
                p = potd_picks.get(label)
                if not p:
                    continue
                potd_rows.append({
                    "League": label,
                    "Event": p.get("Event", ""),
                    "Selection": p.get("Selection", ""),
                    "Market": p.get("Market", ""),
                    "Odds": p.get("Odds", 0) or 0,
                    "Value (%)": p.get("Value (%)", 0),
                    "point": p.get("point"),
                    "Recommended Stake": p.get("Recommended Stake"),
                    "home_team": p.get("home_team", "—"),
                    "away_team": p.get("away_team", "—"),
                    "model_prob": p.get("model_prob", 0),
                    "confidence_tier": p.get("confidence_tier", "Medium"),
                    "reasoning_summary": p.get("reasoning_summary"),
                })
            if potd_rows:
                archive_value_plays(pd.DataFrame(potd_rows), as_of_date=as_of_date)
        except Exception:
            pass

        try:
            clv_record_recommendations(value_plays_df)
            clv_update_closing_odds()
        except Exception:
            pass

        if not live_odds_df.empty and "league" in live_odds_df.columns:
            nba_games = (
                live_odds_df[live_odds_df["league"] == "NBA"][["event_name", "home_team", "away_team"]]
                .drop_duplicates()
                .rename(columns={"home_team": "home_team_name", "away_team": "away_team_name"})
            )
            if not nba_games.empty:
                try:
                    injury_enriched = add_injury_features(nba_games, league="nba")
                    inj_cols = ["injury_impact_score", "top5_out_or_doubtful_home", "top5_out_or_doubtful_away"]
                    if all(c in injury_enriched.columns for c in inj_cols):
                        value_plays_df = value_plays_df.merge(
                            injury_enriched[["event_name"] + inj_cols],
                            left_on="Event",
                            right_on="event_name",
                            how="left",
                        ).drop(columns=["event_name"], errors="ignore")
                        value_plays_df["injury_impact_score"] = value_plays_df["injury_impact_score"].fillna(0.0)
                        value_plays_df["top5_out_or_doubtful_home"] = value_plays_df["top5_out_or_doubtful_home"].fillna(False)
                        value_plays_df["top5_out_or_doubtful_away"] = value_plays_df["top5_out_or_doubtful_away"].fillna(False)
                except Exception:
                    pass

        value_plays_df = add_ncaab_march_context_to_df(value_plays_df)
        if march_madness_mode and not value_plays_df.empty and "League" in value_plays_df.columns:
            tournament_eligible = _load_tournament_eligible_teams(app_root)
            if tournament_eligible:
                ncaab_mask = value_plays_df["League"].astype(str).str.strip().str.upper() == "NCAAB"
                home_ok = value_plays_df["home_team"].astype(str).str.strip().str.lower().isin(tournament_eligible)
                away_ok = value_plays_df["away_team"].astype(str).str.strip().str.lower().isin(tournament_eligible)
                keep = ~ncaab_mask | (ncaab_mask & home_ok & away_ok)
                value_plays_df = value_plays_df.loc[keep].copy()

        # Ensure point/Point from odds is in every play row for cache (coerce to JSON-serializable)
        if not value_plays_df.empty:
            if "point" not in value_plays_df.columns:
                value_plays_df["point"] = None
            value_plays_df["point"] = value_plays_df["point"].apply(
                lambda x: float(x) if x is not None and pd.notna(x) else None
            )
            value_plays_df["Point"] = value_plays_df["point"]

        value_plays_list = value_plays_df.to_dict("records")
        potd_serializable: dict[str, Optional[dict]] = {}
        for k, v in potd_picks.items():
            if v is None:
                potd_serializable[k] = None
            else:
                d = dict(v)
                d["Point"] = d.get("point")
                potd_serializable[k] = _json_sanitize(d)

        payload = {
            "value_plays": _json_sanitize(value_plays_list),
            "potd_picks": potd_serializable,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "odds_source_meta": meta,
            "value_plays_flagged_count": int(value_plays_flagged_count),
        }
        with open(cache_path, "w") as f:
            json.dump(payload, f, indent=2)

        print("\n--- Today's value plays (type and line errors) ---")
        for i, play in enumerate(value_plays_list, 1):
            market = play.get("Market", "—")
            le = play.get("line_error")
            le_str = f"{le:+.2f} pts" if le is not None else "—"
            print(f"  {i}. [{market}] {str(play.get('Event', ''))[:45]}  |  {play.get('Selection', '')}  |  Edge {play.get('Value (%)', 0)}%  |  line_error {le_str}")
        print("---\n")

    except Exception as e:
        import traceback
        _write_fallback(str(e))
        raise
