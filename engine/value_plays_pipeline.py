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
EV_EPSILON_MIN_PCT = 3.0
EV_EPSILON_MAX_PCT = 15.0
MIN_BOOKMAKERS_VALUE_PLAY = 3
HIGH_RISK_ODDS_AMERICAN = 500
POTD_MIN_EDGE_PCT = 4.0
POTD_HIGH_CONFIDENCE_EDGE_PCT = 8.0
POTD_LARGE_SPREAD_POINTS = 12.0
POTD_LARGE_SPREAD_EDGE_PENALTY = 0.30
ARCHIVE_MIN_EDGE_PCT = 6.0
ARCHIVE_MAX_PLAYS_PER_DAY = 10
# NCAAB spreads: require at least 3% edge; only show when 8 <= |line_error| <= 20 (filter out bad model predictions)
NCAAB_SPREAD_MIN_EDGE_PCT = 3.0
SPREAD_LINE_ERROR_MIN_PTS = 8.0
SPREAD_LINE_ERROR_MAX_PTS = 20.0
MAX_VALUE_PLAYS_CACHE = 10
# Diversity cap: reserve slots for underdog vs favorite so dashboard shows a mix
DIVERSITY_UNDERDOG_SLOTS = 5
DIVERSITY_FAVORITE_SLOTS = 5
DIVERSITY_FLEX_SLOTS = 5
MARKET_LABELS = {"h2h": "Winner", "spreads": "Spread", "totals": "Over/Under"}
BANKROLL_FOR_STAKES = 1000.0


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
    """Select two NCAAB POTD picks. Pick 1 = highest edge moneyline. Pick 2 = highest edge spread (8–20 pt line error); if none, 2nd moneyline."""
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
        # Fetch odds
        live_odds_df = get_live_odds(
            api_key=api_key.strip(),
            sport_keys=LIVE_ODDS_SPORT_KEYS,
            commence_on_date=None,
        )
        meta: dict = {}
        quota = get_quota_status()
        remaining = quota.get("requests_remaining")
        if verbose:
            print(f"[DEBUG_RUNDOWN] Pipeline: live_odds_df.empty={live_odds_df.empty}, remaining={remaining!r}.")
        if live_odds_df.empty and remaining is not None:
            try:
                r = int(remaining)
            except (TypeError, ValueError):
                r = 999
            if verbose:
                print(f"[DEBUG_RUNDOWN] Pipeline: credits remaining={r}, r<10={r < 10}. Will try Rundown then ESPN when r<10.")
            if r < 10:
                # Primary fallback: The Rundown (RapidAPI); then ESPN
                if verbose:
                    os.environ["DEBUG_RUNDOWN_ODDS"] = "1"
                live_odds_df, n_games, n_odds_rows = get_rundown_live_odds_with_stats(
                    sport_keys=LIVE_ODDS_SPORT_KEYS,
                    commence_on_date=None,
                )
                if verbose:
                    os.environ.pop("DEBUG_RUNDOWN_ODDS", None)
                if not live_odds_df.empty:
                    meta = {"used_rundown_fallback": True, "rundown_games": n_games, "rundown_odds_rows": n_odds_rows}
                else:
                    if verbose:
                        os.environ["DEBUG_ESPN_ODDS"] = "1"
                    live_odds_df, n_games, n_odds_rows = get_espn_live_odds_with_stats(
                        sport_keys=LIVE_ODDS_SPORT_KEYS,
                        commence_on_date=None,
                    )
                    if verbose:
                        os.environ.pop("DEBUG_ESPN_ODDS", None)
                    meta = {"used_espn_fallback": True, "espn_games": n_games, "espn_odds_rows": n_odds_rows}
        if not live_odds_df.empty and "commence_time" in live_odds_df.columns:
            now_utc = datetime.now(timezone.utc)
            cutoff_end = now_utc + timedelta(hours=24)

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

        as_of_date = date.today()
        b2b_teams = set(get_nba_teams_back_to_back(api_key.strip(), as_of_date))
        feature_matrix = load_feature_matrix_for_inference(league=None)
        pace_stats = get_nba_team_pace_stats()
        use_rundown_fallback = bool(meta.get("used_rundown_fallback"))
        use_espn_fallback = bool(meta.get("used_espn_fallback"))
        use_fallback = use_rundown_fallback or use_espn_fallback
        min_ev = 1.5 if use_fallback else EV_EPSILON_MIN_PCT
        min_books_override = 1 if use_fallback else None

        if verbose:
            if use_rundown_fallback:
                n_g = meta.get("rundown_games", 0)
                n_o = meta.get("rundown_odds_rows", 0)
                ncaab_g = (live_odds_df.loc[live_odds_df["league"] == "NCAAB", "event_id"].nunique()
                    if not live_odds_df.empty and "league" in live_odds_df.columns else 0)
                print(f"The Rundown fallback: Yes. NCAAB games: {ncaab_g}. Total games: {n_g}. Odds rows: {n_o}.")
            elif use_espn_fallback:
                n_espn = meta.get("espn_games", 0)
                n_odds = meta.get("espn_odds_rows", 0)
                print(f"ESPN fallback: Yes. NCAAB games from ESPN: {n_espn}. Odds rows: {n_odds}.")
            else:
                ncaab_games = (live_odds_df.loc[live_odds_df["league"] == "NCAAB", "event_id"].nunique()
                    if not live_odds_df.empty and "league" in live_odds_df.columns else 0)
                print(f"ESPN fallback: No. NCAAB games from Odds API: {ncaab_games}.")

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
            _print_verbose_spread_stats(verbose_stats)

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

        # Moneylines: keep min edge 3%. Spreads: min edge 3% and 8 <= |line_error| <= 20 (filter out bad model predictions)
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
            keep = other | (ncaab_spread & spread_ok_edge & spread_ok_line)
            value_plays_df = value_plays_df.loc[keep].copy()
        value_plays_df = value_plays_df.sort_values("Value (%)", ascending=False).head(MAX_VALUE_PLAYS_CACHE).reset_index(drop=True)

        if verbose:
            print(f"Plays passed all filters: {len(value_plays_df)}.")

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
