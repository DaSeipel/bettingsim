"""
Daily 9am archive job: fetch odds, compute value plays (minimal pipeline), archive to play_history.
Runs without Streamlit so APScheduler can call it. Uses ODDS_API_KEY from environment.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .engine import (
    BASKETBALL_NBA,
    BASKETBALL_NCAAB,
    NBA_LEAGUE_AVG_OFF_RATING,
    NBA_LEAGUE_AVG_PACE,
    get_live_odds,
    get_nba_team_pace_stats,
    get_nba_teams_back_to_back,
    get_schedule_fatigue_penalty,
    get_team_power_ratings,
)
from .betting_models import (
    get_feature_row_for_game,
    load_feature_matrix_for_inference,
    predict_moneyline_prob,
    predict_spread_prob,
    predict_totals_prob,
    consensus_spread,
    consensus_totals,
    consensus_moneyline,
)
from .play_history import archive_value_plays
from strategies.strategies import (
    american_to_decimal,
    damp_probability,
    fractional_kelly_half_units,
    implied_probability_no_vig,
    in_house_spread_from_ratings,
    key_number_value_adjustment,
    kelly_fraction,
    model_prob_from_in_house_spread,
    model_prob_from_in_house_total,
    model_prob_from_ratings_moneyline,
    predict_nba_total,
)

BANKROLL = 1000.0
MIN_EDGE_PCT = 6.0
MAX_EDGE_PCT = 15.0
MAX_PLAYS_PER_DAY = 10
HIGH_RISK_ODDS = 500
HALF_UNIT_PCT = 0.005
MAX_KELLY_PCT = 0.03
HIGH_CONFIDENCE_EDGE = 8.0
MIN_BOOKMAKERS = 3


def _aggregate_odds(odds_df: pd.DataFrame) -> pd.DataFrame:
    if odds_df.empty or "odds" not in odds_df.columns:
        return odds_df
    key_cols = ["event_id", "sport_key", "league", "commence_time", "home_team", "away_team", "event_name", "market_type", "selection", "point"]
    if not all(c in odds_df.columns for c in key_cols):
        return odds_df

    def _mean_implied(ser: pd.Series) -> float:
        return float(np.mean([implied_probability_no_vig(float(o)) for o in ser]))

    return odds_df.groupby(key_cols, dropna=False).agg(
        odds=("odds", "max"),
        avg_implied_prob=("odds", _mean_implied),
        bookmaker_count=("odds", "count"),
    ).reset_index()


def _format_start_time(commence_time: str) -> str:
    if not commence_time or not str(commence_time).strip():
        return "—"
    try:
        s = str(commence_time).replace("Z", "+00:00")
        dt_utc = datetime.fromisoformat(s)
        if dt_utc.tzinfo is None:
            return dt_utc.strftime("%b %d, %I:%M %p")
        dt_et = dt_utc.astimezone(ZoneInfo("America/New_York"))
        return dt_et.strftime("%b %d, %I:%M %p ET")
    except (ValueError, TypeError):
        return "—"


def run_daily_archive(as_of_date: date | None = None) -> int:
    """
    Fetch today's odds, compute value plays (same logic as app pipeline), archive to play_history.
    Call from APScheduler at 9am ET. Requires ODDS_API_KEY in environment.
    Returns number of plays archived.
    """
    api_key = (os.environ.get("ODDS_API_KEY") or "").strip()
    if not api_key:
        return 0
    as_of_date = as_of_date or date.today()
    try:
        odds_df = get_live_odds(
            api_key,
            sport_keys=[BASKETBALL_NBA, BASKETBALL_NCAAB],
            display_timezone="America/New_York",
        )
    except Exception:
        return 0
    if odds_df.empty:
        return 0
    aggregated = _aggregate_odds(odds_df)
    if aggregated.empty:
        return 0
    pace_stats = get_nba_team_pace_stats()
    try:
        b2b_teams = get_nba_teams_back_to_back(api_key, as_of_date)
    except Exception:
        b2b_teams = set()
    power_ratings = get_team_power_ratings(pace_stats, NBA_LEAGUE_AVG_PACE, NBA_LEAGUE_AVG_OFF_RATING)
    default_rating = (NBA_LEAGUE_AVG_PACE * NBA_LEAGUE_AVG_OFF_RATING) / 100.0
    try:
        feature_matrix = load_feature_matrix_for_inference(league=None)
    except Exception:
        feature_matrix = pd.DataFrame()
    rows = []
    for _, r in aggregated.iterrows():
        if r.get("bookmaker_count", 0) < MIN_BOOKMAKERS:
            continue
        odds_val = float(r.get("odds", 0))
        if pd.isna(odds_val) or abs(odds_val) < 100 or odds_val > HIGH_RISK_ODDS:
            continue
        home_team = str(r.get("home_team", ""))
        away_team = str(r.get("away_team", ""))
        market_type = str(r.get("market_type", "")).strip().lower() or "h2h"
        point = r.get("point")
        selection = str(r.get("selection", ""))
        league = str(r.get("league", ""))
        league_lookup = league.strip().lower()
        commence_time = r.get("commence_time")
        game_date_str = None
        if commence_time:
            try:
                dt = datetime.fromisoformat(str(commence_time).replace("Z", "+00:00"))
                game_date_str = dt.strftime("%Y-%m-%d")
            except Exception:
                pass
        feature_row = get_feature_row_for_game(feature_matrix, home_team, away_team, league_lookup, game_date=game_date_str) if feature_matrix is not None and not feature_matrix.empty else None
        if market_type == "totals" and point is not None:
            market_total = float(point)
            in_house_total = predict_nba_total(home_team, away_team, pace_stats, b2b_teams=b2b_teams)
            prefer_over = "Over" in selection or "over" in selection.lower()
            fallback = model_prob_from_in_house_total(in_house_total, market_total, prefer_over)
            model_prob, consensus_ok = consensus_totals(feature_row, market_total, prefer_over, in_house_total, league, fallback)
            if not consensus_ok:
                continue
        elif market_type == "spreads" and point is not None:
            market_spread = -float(point)
            home_rating = power_ratings.get(home_team, default_rating) - get_schedule_fatigue_penalty(home_team, as_of_date)
            away_rating = power_ratings.get(away_team, default_rating) - get_schedule_fatigue_penalty(away_team, as_of_date)
            in_house_spread = in_house_spread_from_ratings(home_rating, away_rating)
            we_cover_favorite = "-" in selection
            fallback = model_prob_from_in_house_spread(in_house_spread, market_spread, we_cover_favorite)
            model_prob, consensus_ok = consensus_spread(feature_row, market_spread, we_cover_favorite, in_house_spread, fallback)
            if not consensus_ok:
                continue
        elif market_type == "h2h":
            home_rating = power_ratings.get(home_team, default_rating) - get_schedule_fatigue_penalty(home_team, as_of_date)
            away_rating = power_ratings.get(away_team, default_rating) - get_schedule_fatigue_penalty(away_team, as_of_date)
            selection_is_home = (str(selection).strip().lower() == str(home_team).strip().lower() or str(home_team).strip().lower() in str(selection).strip().lower())
            fallback = model_prob_from_ratings_moneyline(home_rating, away_rating, selection_is_home)
            model_prob, consensus_ok = consensus_moneyline(feature_row, selection_is_home, home_rating, away_rating, fallback)
            if not consensus_ok:
                continue
        else:
            implied = implied_probability_no_vig(odds_val)
            model_prob = damp_probability(min(0.92, implied + 0.05))
        implied_prob = float(r.get("avg_implied_prob")) if r.get("avg_implied_prob") is not None and not pd.isna(r.get("avg_implied_prob")) else implied_probability_no_vig(odds_val)
        ev_decimal = (model_prob * american_to_decimal(odds_val)) - 1.0
        ev_pct = ev_decimal * 100.0
        line_for_key = float(point) if point is not None else 0.0
        ev_pct += key_number_value_adjustment(line_for_key, league, market_type)
        if ev_pct < MIN_EDGE_PCT or ev_pct >= MAX_EDGE_PCT:
            continue
        full_kelly = kelly_fraction(odds_val, model_prob, fraction=1.0)
        stake = fractional_kelly_half_units(BANKROLL, full_kelly, HALF_UNIT_PCT, MAX_KELLY_PCT)
        if stake <= 0:
            continue
        confidence_tier = "High" if ev_pct >= HIGH_CONFIDENCE_EDGE else "Medium"
        rows.append({
            "League": league,
            "Event": r.get("event_name", ""),
            "Selection": selection,
            "Market": market_type,
            "Odds": int(round(odds_val)),
            "Value (%)": round(ev_pct, 2),
            "Recommended Stake": stake,
            "Injury Alert": "—",
            "Start Time": _format_start_time(str(commence_time) if commence_time else ""),
            "commence_time": commence_time,
            "model_prob": model_prob,
            "implied_prob": implied_prob,
            "point": point,
            "home_team": home_team,
            "away_team": away_team,
            "confidence_tier": confidence_tier,
            "reasoning_summary": None,
        })
    value_plays_df = pd.DataFrame(rows)
    if value_plays_df.empty:
        return 0
    value_plays_df = value_plays_df.sort_values("Value (%)", ascending=False).head(MAX_PLAYS_PER_DAY)
    return archive_value_plays(value_plays_df, as_of_date=as_of_date)
