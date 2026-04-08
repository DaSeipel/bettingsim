"""
Bobby Bottle's Betting Model - Streamlit Dashboard
Daily picks sheet: Play of the Day (NCAAB) and All Value Plays. Dark theme, NCAAB-focused (NBA coming soon).
Live odds from The Odds API; stakes from fractional Kelly.
"""

import gc
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo
import plotly.graph_objects as go
from engine.engine import (
    BettingEngine,
    get_live_odds,
    get_nba_team_pace_stats,
    get_nba_teams_back_to_back,
    get_team_power_ratings,
    get_schedule_fatigue_penalty,
    BASKETBALL_NBA,
    BASKETBALL_NCAAB,
    NBA_LEAGUE_AVG_PACE,
    NBA_LEAGUE_AVG_OFF_RATING,
)
from engine import get_espn_live_odds_with_stats
from engine.odds_quota import get_quota_status
from strategies.strategies import (
    strategy_kelly,
    kelly_fraction,
    implied_probability,
    implied_probability_no_vig,
    implied_probability_fair_two_sided,
    american_to_decimal,
    damp_probability,
    predict_nba_total,
    get_totals_value,
    key_number_value_adjustment,
    fractional_kelly_half_units,
    in_house_spread_from_ratings,
    model_prob_from_in_house_total,
    model_prob_from_in_house_spread,
    model_prob_from_ratings_moneyline,
    spread_cover_prob_from_margins,
    HALF_UNIT_PCT,
    MAX_KELLY_PCT_WALTERS,
)
from engine.injury_scraper import add_injury_features
from engine.clv_tracker import (
    record_recommendations as clv_record_recommendations,
    update_closing_odds as clv_update_closing_odds,
    get_clv_row_for_play,
    mark_bet_result,
)
from engine.play_history import archive_value_plays, delete_play, load_play_history, update_play_result
from engine.auto_result_job import run_auto_result
from engine.ncaab_march_context import add_ncaab_march_context_to_df, is_after_selection_sunday
from engine.betting_models import (
    load_feature_matrix_for_inference,
    get_feature_row_for_game,
    get_top_shap_reasoning,
    get_feature_based_reasoning,
    build_feature_row_for_upcoming_game,
    predict_spread_prob,
    predict_totals_prob,
    predict_moneyline_prob,
    consensus_spread,
    consensus_totals,
    consensus_moneyline,
)

def format_american(odds: float) -> str:
    """Format American odds for display: +150, -110. Returns "—" for missing/NaN/invalid."""
    if odds is None or pd.isna(odds):
        return "—"
    try:
        x = int(round(float(odds)))
        return f"{x:+d}" if x != 0 else "—"
    except (TypeError, ValueError, OverflowError):
        return "—"


def format_currency(val: float) -> str:
    """Format as $0.00 for display."""
    try:
        return f"${float(val):,.2f}"
    except (TypeError, ValueError):
        return "—"


# Strip-down mode: disable pipeline and DB loads so app loads stably. Set to False to enable value plays pipeline.
STRIP_DOWN_MODE = False
# Scheduler and startup thread: kept disabled for stability; set True to re-enable.
ENABLE_SCHEDULER = False
ENABLE_STARTUP_THREAD = False

# Human-readable market labels (replaces h2h, spreads, totals in UI)
MARKET_LABELS = {"h2h": "Winner", "spreads": "Spread", "totals": "Over/Under", "moneyline": "Moneyline"}

# NBA team full name -> abbreviation
NBA_TEAM_ABBREV: dict[str, str] = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}


def _team_abbrev(team_name: str, league: str) -> str:
    """Return abbreviation for team. NBA uses standard 3-letter; NCAAB derives from name."""
    s = (team_name or "").strip()
    if not s:
        return ""
    if str(league).strip().upper() == "NBA":
        return NBA_TEAM_ABBREV.get(s, s[:8])  # fallback: first 8 chars
    # NCAAB: use first 2 letters of key words (e.g. "SE Missouri St Redhawks" -> "SEMO", "Morehead St Eagles" -> "MOR")
    words = s.replace(".", "").split()
    skip = {"St", "State", "St.", "University", "U", "of", "The", "Eagles", "Redhawks", "Wildcats", "Bearcats", "etc"}
    key = [w for w in words if w not in skip][:3]
    if len(key) >= 2:
        return "".join(w[:2] for w in key).upper()[:6]
    return (key[0][:4].upper() if key else s[:6])[:6]


def _build_pick_explanation(row: pd.Series) -> str:
    """Build a human-readable explanation for why this is the model's best value play."""
    market = str(row.get("Market", ""))
    market_label = MARKET_LABELS.get(market, market)
    selection = str(row.get("Selection", ""))
    value_pct = row.get("Value (%)", 0)
    odds = row.get("Odds", 0)
    stake = row.get("Recommended Stake", 0)

    parts = []
    parts.append(f"Our model identifies **{value_pct:.1f}% expected value** on this play—the highest among today's NBA slate.")
    if market == "spreads":
        parts.append("Power ratings and schedule fatigue (back-to-backs) suggest the market is mispricing this spread.")
    elif market == "totals":
        parts.append("Pace-adjusted projections and key-number analysis (3, 7, 14) indicate the total is off.")
    else:
        parts.append("Implied probability vs our model probability shows meaningful edge on the moneyline.")
    parts.append(f"Recommended stake: **{format_currency(stake)}** ({format_american(odds)} odds).")
    return " ".join(parts)

# Odds > +500 (6.0 decimal) flagged as high-risk; excluded from Best Value unless toggled
HIGH_RISK_ODDS_AMERICAN = 500

# Confidence tier: edge >= this is High, else Medium (unless large spread → High Variance)
POTD_HIGH_CONFIDENCE_EDGE_PCT = 8.0
# Minimum edge % to qualify as Play of the Day (and for archive)
POTD_MIN_EDGE_PCT = 6.0
# Archive: save all generated value plays so they appear on Mark Results (no min edge, no cap)
ARCHIVE_MIN_EDGE_PCT = 0.0
ARCHIVE_MAX_PLAYS_PER_DAY = 999
# Large spread filter: |spread| > this gets 30% edge penalty for POTD selection and "High Variance" badge
POTD_LARGE_SPREAD_POINTS = 14.0
POTD_LARGE_SPREAD_EDGE_PENALTY = 0.30


def _bookmaker_counts(odds_df: pd.DataFrame) -> pd.DataFrame:
    """Count distinct bookmakers per (event_name, market_type, selection, point). Returns DataFrame with those cols + bookmaker_count."""
    if odds_df.empty or not all(c in odds_df.columns for c in ["event_name", "market_type", "selection", "point"]):
        return pd.DataFrame(columns=["event_name", "market_type", "selection", "point", "bookmaker_count"])
    # Odds API flattening: one row per bookmaker per outcome; no bookmaker id in flattened df, so count rows as proxy for "books agreeing"
    grouped = odds_df.groupby(["event_name", "market_type", "selection", "point"], dropna=False).size().reset_index(name="bookmaker_count")
    return grouped


def select_play_of_the_day(
    value_plays_df: pd.DataFrame,
    live_odds_df: pd.DataFrame,
    min_edge_pct: float = POTD_MIN_EDGE_PCT,
) -> dict[str, Optional[dict]]:
    """
    Select two NCAAB Plays of the Day: top 2 highest-conviction plays by edge % from different games.
    Tie-breaks (in order): (1) more bookmakers, (2) model prob gap from implied, (3) no major injury flags.
    Returns {"NCAAB Pick 1": pick_dict or None, "NCAAB Pick 2": pick_dict or None}. Each pick_dict
    includes Event, Selection, Market, Odds, Value (%), point, home_team, away_team, model_prob,
    confidence_tier, reasoning_summary for card display and archive.
    """
    result: dict[str, Optional[dict]] = {"NCAAB Pick 1": None, "NCAAB Pick 2": None}
    if value_plays_df.empty or "League" not in value_plays_df.columns:
        return result

    eligible = value_plays_df[
        (value_plays_df["League"].astype(str).str.strip().str.upper() == "NCAAB") &
        (value_plays_df["Value (%)"] > min_edge_pct)
    ].copy()
    if eligible.empty:
        return result

    bc = _bookmaker_counts(live_odds_df)
    if not bc.empty:
        eligible = eligible.merge(
            bc,
            left_on=["Event", "Market", "Selection", "point"],
            right_on=["event_name", "market_type", "selection", "point"],
            how="left",
        )
        eligible["bookmaker_count"] = eligible["bookmaker_count"].fillna(0).astype(int)
    else:
        eligible["bookmaker_count"] = 0

    if "model_prob" in eligible.columns and "implied_prob" in eligible.columns:
        eligible["model_prob_gap"] = (eligible["model_prob"] - eligible["implied_prob"]).abs()
    else:
        eligible["model_prob_gap"] = 0.0

    def _has_injury_flag(row: pd.Series) -> bool:
        if row.get("top5_out_or_doubtful_home") or row.get("top5_out_or_doubtful_away"):
            return True
        score = row.get("injury_impact_score")
        if pd.isna(score):
            return False
        return float(score) > 0.5

    eligible["has_injury_flag"] = eligible.apply(_has_injury_flag, axis=1)

    def _adjusted_edge(row: pd.Series) -> float:
        if str(row.get("Market", "")).strip().lower() != "spreads":
            return float(row.get("Value (%)", 0))
        pt = row.get("point_x") if "point_x" in row.index else row.get("point")
        try:
            abs_spread = abs(float(pt))
        except (TypeError, ValueError):
            return float(row.get("Value (%)", 0))
        if abs_spread > POTD_LARGE_SPREAD_POINTS:
            return float(row.get("Value (%)", 0)) * (1.0 - POTD_LARGE_SPREAD_EDGE_PENALTY)
        return float(row.get("Value (%)", 0))

    eligible["_adjusted_edge"] = eligible.apply(_adjusted_edge, axis=1)

    def _is_large_spread(row: pd.Series) -> bool:
        if str(row.get("Market", "")).strip().lower() != "spreads":
            return False
        pt = row.get("point_x") if "point_x" in row.index else row.get("point")
        try:
            return abs(float(pt)) > POTD_LARGE_SPREAD_POINTS
        except (TypeError, ValueError):
            return False

    sorted_df = eligible.sort_values(
        by=["_adjusted_edge", "bookmaker_count", "model_prob_gap", "has_injury_flag"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    def _row_to_pick(top: pd.Series, label: str) -> dict:
        point_val = top.get("point_x") if "point_x" in top.index else top.get("point")
        odds_raw = top.get("Odds", 0)
        if odds_raw is None or pd.isna(odds_raw):
            odds_for_pick = None
        else:
            try:
                odds_for_pick = int(round(float(odds_raw)))
            except (TypeError, ValueError):
                odds_for_pick = None
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
        }

    # Pick 1: top by edge
    result["NCAAB Pick 1"] = _row_to_pick(sorted_df.iloc[0], "NCAAB Pick 1")
    pick1_event = str(sorted_df.iloc[0].get("Event", ""))

    # Pick 2: top from a different game (Event is a Series — use .str.strip())
    other_games = sorted_df[sorted_df["Event"].astype(str).str.strip() != pick1_event.strip()]
    if not other_games.empty:
        result["NCAAB Pick 2"] = _row_to_pick(other_games.iloc[0], "NCAAB Pick 2")

    return result


# #region agent log
def _debug_log(location: str, message: str, data: dict, hypothesis_id: str = "") -> None:
    try:
        payload = {"sessionId": "a60dbe", "location": location, "message": message, "data": data, "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000)}
        if hypothesis_id:
            payload["hypothesisId"] = hypothesis_id
        with open("/Users/robertseipel/Desktop/bettingsim/bettingsim/.cursor/debug-a60dbe.log", "a") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass


# #endregion


def _get_yesterday_potd_results() -> list[dict]:
    """
    Load yesterday's play history and return the two POTD picks (NCAAB Pick 1, NCAAB Pick 2).
    Each item: {"sport": "NCAAB Pick 1"|"NCAAB Pick 2", "side": "Team Name", "result": "W"|"L"|"P"|None}.
    """
    if STRIP_DOWN_MODE:
        return []
    yesterday = date.today() - timedelta(days=1)
    hist = _load_play_history_cached(from_date_iso=yesterday.isoformat(), to_date_iso=yesterday.isoformat())
    if hist.empty or "my_edge_pct" not in hist.columns:
        return []
    hist = hist.copy()
    hist["result_clean"] = hist["result"].apply(
        lambda x: str(x).strip().upper() if x is not None and not pd.isna(x) else None
    )
    top = hist[hist["sport"].astype(str).str.strip().isin(("NCAAB Pick 1", "NCAAB Pick 2"))].copy()
    if not top.empty:
        top = top.loc[top.groupby("sport")["my_edge_pct"].idxmax()].reset_index(drop=True)
        top = top.sort_values("sport", ascending=True)
    rows = []
    for _, r in top.iterrows():
        # #region agent log
        _debug_log("app.py:_get_yesterday_potd_results", "top row", {"sport": str(r.get("sport")), "recommended_side": str(r.get("recommended_side")), "result_raw": r.get("result"), "result_clean": r.get("result_clean"), "bet_type": str(r.get("bet_type"))}, "A")
        # #endregion
        sport = str(r.get("sport", "")).strip() or "—"
        side = str(r.get("recommended_side", "")).strip() or "—"
        res = r.get("result_clean")
        if res is not None and not pd.isna(res) and str(res).upper() in ("W", "L", "P"):
            res = str(res).upper()
        else:
            res = None
        rows.append({"sport": sport, "side": side, "result": res})
    return rows


def _html_escape(s: str) -> str:
    """Escape for safe use inside HTML."""
    if not s:
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _parse_event_teams(event_str: str) -> tuple[str, str]:
    """Parse 'Away @ Home' or 'Away vs Home' into (away_team, home_team)."""
    s = (event_str or "").strip()
    for sep in (" @ ", " vs ", " at "):
        if sep in s:
            parts = s.split(sep, 1)
            return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else (s, "")
    return (s, "")


# Correlated play filter: same team on same day = correlated; only surface higher-edge play per team.
MAX_TOP_PLAYS_NO_CORRELATION = 10


def _filter_correlated_plays(
    plays_df: pd.DataFrame,
    max_plays: int = MAX_TOP_PLAYS_NO_CORRELATION,
    edge_col: str = "Value (%)",
    home_col: str = "home_team",
    away_col: str = "away_team",
) -> pd.DataFrame:
    """
    Among plays (one per game), keep at most max_plays and ensure no team appears in more than one play.
    When a team appears in multiple plays (e.g. Purdue in a spread and in a total), keep only the play with higher edge.
    Returns subset of plays_df with same columns.
    """
    if plays_df.empty or edge_col not in plays_df.columns:
        return plays_df
    has_teams = home_col in plays_df.columns and away_col in plays_df.columns
    if not has_teams:
        return plays_df.sort_values(edge_col, ascending=False).head(max_plays).reset_index(drop=True)
    sorted_df = plays_df.sort_values(edge_col, ascending=False).reset_index(drop=True)
    used_teams: set[str] = set()
    indices: list[int] = []
    for i, row in sorted_df.iterrows():
        if len(indices) >= max_plays:
            break
        h = str(row.get(home_col, "")).strip()
        a = str(row.get(away_col, "")).strip()
        if not h and not a:
            indices.append(i)
            continue
        if h in used_teams or a in used_teams:
            continue
        used_teams.add(h)
        used_teams.add(a)
        indices.append(i)
    if not indices:
        return pd.DataFrame()
    return sorted_df.loc[indices].reset_index(drop=True)


def _potd_reason(
    row,
    feature_matrix: Optional[pd.DataFrame] = None,
    b2b_teams: Optional[set[str]] = None,
) -> str:
    """Build a 3–4 sentence explanation for why this play was chosen (POTD card)."""
    market = str(row.get("Market", ""))
    market_label = MARKET_LABELS.get(market, market)
    selection = str(row.get("Selection", ""))
    event = str(row.get("Event", ""))
    edge_pct = float(row.get("Value (%)", 0))
    league_raw = str(row.get("League", ""))
    # Normalize so "NCAAB Pick 1" / "NCAAB Pick 2" use NCAAB feature lookup and SHAP model
    league = "NCAAB" if league_raw and "NCAAB" in league_raw.strip().upper() else league_raw.strip()
    league_shap = "ncaab" if league_raw and "ncaab" in league_raw.strip().lower() else (league_raw.strip().lower() if league_raw else "")
    point = row.get("point")
    away, home = _parse_event_teams(event)
    # #region agent log
    try:
        import time
        fm_shape = (len(feature_matrix), len(feature_matrix.columns)) if feature_matrix is not None and not feature_matrix.empty else None
        with open("/Users/robertseipel/Desktop/bettingsim/bettingsim/.cursor/debug-e8fe0d.log", "a") as f:
            f.write(json.dumps({"sessionId":"e8fe0d","hypothesisId":"H4,H5","location":"app.py:_potd_reason","message":"potd_reason_entry","data":{"fm_shape":fm_shape,"home":home,"away":away,"league":league,"event":event},"timestamp":int(time.time()*1000)}) + "\n")
    except Exception:
        pass
    # #endregion

    # Sentence 1: what we're betting and the edge
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

    # Sentences 2–4: why (SHAP drivers if available, else feature-based, else minimal fallback)
    why_sentences: list[str] = []
    feature_row: Optional[pd.Series] = None
    if feature_matrix is not None and not feature_matrix.empty and home and away:
        feature_row = get_feature_row_for_game(
            feature_matrix,
            home_team=home,
            away_team=away,
            league=league,
        )
        # #region agent log
        if feature_row is None:
            try:
                import time
                with open("/Users/robertseipel/Desktop/bettingsim/bettingsim/.cursor/debug-e8fe0d.log", "a") as f:
                    f.write(json.dumps({"sessionId":"e8fe0d","hypothesisId":"H1","location":"app.py:_potd_reason","message":"feature_row_none","data":{"home":home,"away":away,"league":league},"timestamp":int(time.time()*1000)}) + "\n")
            except Exception:
                pass
        # #endregion
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
        # #region agent log
        try:
            import time
            with open("/Users/robertseipel/Desktop/bettingsim/bettingsim/.cursor/debug-e8fe0d.log", "a") as f:
                f.write(json.dumps({"sessionId":"e8fe0d","hypothesisId":"H1","location":"app.py:_potd_reason","message":"feature_row_found","data":{"home":home,"away":away},"timestamp":int(time.time()*1000)}) + "\n")
        except Exception:
            pass
        # #endregion
        shap_reason = get_top_shap_reasoning(
            feature_row, market, home_team=home, away_team=away, top_k=4, league=league_shap or league.strip().lower()
        )
        feat_reason = None
        if shap_reason:
            why_sentences.append(shap_reason)
        if not shap_reason:
            feat_reason = get_feature_based_reasoning(
                feature_row, market, home_team=home, away_team=away, top_k=4
            )
            if feat_reason:
                why_sentences.append(feat_reason)
        # #region agent log
        try:
            import time
            with open("/Users/robertseipel/Desktop/bettingsim/bettingsim/.cursor/debug-e8fe0d.log", "a") as f:
                f.write(json.dumps({"sessionId":"e8fe0d","hypothesisId":"H2,H3","location":"app.py:_potd_reason","message":"reasoning_results","data":{"shap_ok":bool(shap_reason),"feat_ok":bool(feat_reason),"why_count":len(why_sentences)},"timestamp":int(time.time()*1000)}) + "\n")
        except Exception:
            pass
        # #endregion
    if not why_sentences:
        opponent = away if (selection.strip() == home.strip()) else home
        why_sentences.append(
            f"Our model gives {selection} a higher win probability than the odds imply against {opponent}."
        )

    return " ".join([opening] + why_sentences)


def _potd_badge_text(row: pd.Series) -> str:
    """Recommended bet label for badge: e.g. 'Spread · Lakers -3.5', 'Over 220.5', 'Moneyline · Lakers'."""
    market = str(row.get("Market", ""))
    selection = str(row.get("Selection", ""))
    # Avoid showing "+nan" when odds were missing (e.g. API had spread but no moneyline)
    for suffix in (" +nan", " +NaN", " +NAN", " -nan", " -NaN"):
        if selection.endswith(suffix):
            selection = selection[: -len(suffix)].strip()
            break
    market_label = MARKET_LABELS.get(market, market)
    point = row.get("point")
    if point is not None:
        try:
            pt = float(point)
            # For h2h (moneyline), point is NaN; do not append it (would show "+nan")
            if pd.isna(pt) or (isinstance(pt, float) and pt != pt):
                pass  # fall through to no-point return below
            elif market == "totals":
                return f"{market_label} · {selection} {pt:.1f}" if f"{pt:.1f}" not in selection else f"{market_label} · {selection}"
            else:
                return f"{market_label} · {selection} {pt:+.1f}"
        except (TypeError, ValueError):
            pass
    return f"{market_label} · {selection}"


# Normalize HTML so no line has leading spaces (avoids Markdown code-block → <pre> wrapping)
def _html_one_line_per_block(html: str) -> str:
    return "\n".join(line.strip() for line in html.strip().splitlines() if line.strip())

# Play of the Day card: custom CSS + HTML via st.markdown(unsafe_allow_html=True)
POTD_CARD_CSS = """
<style>
.potd-card {
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin: 0 0 1rem 0;
    font-family: inherit;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    border-left: 5px solid;
}
.potd-card--blue  { border-left-color: #1e88e5; background: linear-gradient(135deg, rgba(30,136,229,0.12) 0%, rgba(0,0,0,0.2) 100%); }
.potd-card--orange { border-left-color: #f57c00; background: linear-gradient(135deg, rgba(245,124,0,0.12) 0%, rgba(0,0,0,0.2) 100%); }
.potd-card--grey  { border-left-color: #78909c; background: linear-gradient(135deg, rgba(96,125,139,0.15) 0%, rgba(0,0,0,0.2) 100%); }
.potd-league {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
    opacity: 0.9;
}
.potd-card--blue .potd-league  { color: #42a5f5; }
.potd-card--orange .potd-league { color: #ffb74d; }
.potd-card--grey .potd-league  { color: #90a4ae; }
.potd-matchup {
    font-size: 1.35rem;
    font-weight: 800;
    line-height: 1.3;
    margin-bottom: 0.75rem;
    color: #fafafa;
}
.potd-matchup .potd-vs { font-weight: 400; opacity: 0.7; font-size: 1rem; }
.potd-matchup .potd-abbrev { font-size: 0.75rem; font-weight: 500; opacity: 0.7; margin-left: 0.25rem; }
.potd-tipoff { font-size: 0.85rem; color: rgba(255,255,255,0.8); margin-bottom: 0.5rem; }
.potd-current-line { font-size: 0.8rem; color: rgba(255,255,255,0.7); margin-top: -0.35rem; margin-bottom: 0.35rem; }
.potd-line-as-of { font-size: 0.7rem; color: rgba(255,255,255,0.5); margin-top: 0.5rem; }
.potd-badge-row { display: flex; align-items: center; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 0.75rem; }
.potd-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    line-height: 1.3;
}
.potd-card--blue .potd-badge  { background: rgba(30,136,229,0.4); color: #90caf9; }
.potd-card--orange .potd-badge { background: rgba(245,124,0,0.4); color: #ffe0b2; }
.potd-odds-inline { font-size: 1.2rem; font-weight: 700; color: rgba(255,255,255,0.95); }
.potd-edge {
    font-size: 1.75rem;
    font-weight: 800;
    color: #66bb6a;
    margin: 0.5rem 0;
}
.potd-confidence {
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    opacity: 0.95;
}
.potd-card--blue .potd-confidence  { color: #90caf9; }
.potd-card--orange .potd-confidence { color: #ffb74d; }
.potd-confidence.potd-confidence--warning { color: #ffb74d; }
.potd-march-context { font-size: 0.8rem; color: #ffb74d; margin-top: 0.35rem; margin-bottom: 0.35rem; }
.potd-reason {
    font-size: 0.9rem;
    line-height: 1.4;
    color: rgba(255,255,255,0.85);
    margin-bottom: 0.6rem;
}
.potd-empty-msg {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.6);
    text-align: center;
    padding: 1.5rem 0;
}

/* All Value Plays card list */
.vp-card {
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    font-family: inherit;
}
.vp-card-league { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #90a4ae; margin-bottom: 0.25rem; }
.vp-card-league.nba { color: #42a5f5; }
.vp-card-league.ncaab { color: #ffb74d; }
.vp-matchup { font-size: 1.05rem; font-weight: 700; color: #fafafa; margin-bottom: 0.35rem; }
.vp-meta { display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem; font-size: 0.9rem; }
.vp-bet-type { background: rgba(76, 175, 80, 0.35); color: #a5d6a7; padding: 0.2rem 0.5rem; border-radius: 6px; font-weight: 600; }
.vp-edge { color: #66bb6a; font-weight: 700; }
.vp-odds { color: rgba(255,255,255,0.9); font-weight: 600; }
.vp-underdog-value { background: rgba(255, 193, 7, 0.3); color: #ffc107; padding: 0.2rem 0.5rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }
.vp-tournament-context { background: rgba(156, 39, 176, 0.35); color: #ce93d8; padding: 0.2rem 0.5rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600; margin-left: 0.35rem; }
.potd-tournament-context { background: rgba(156, 39, 176, 0.35); color: #ce93d8; padding: 0.2rem 0.4rem; border-radius: 6px; font-size: 0.7rem; font-weight: 600; margin-top: 0.25rem; }
.vp-march-context { font-size: 0.75rem; color: #ffb74d; margin-top: 0.35rem; }
.vp-confidence-wrap { margin-top: 0.5rem; }
.vp-confidence-bar { height: 6px; border-radius: 3px; background: rgba(255,255,255,0.15); overflow: hidden; }
.vp-confidence-fill { height: 100%; border-radius: 3px; background: linear-gradient(90deg, #43a047, #66bb6a); transition: width 0.2s ease; }
.vp-reasoning { font-size: 0.8rem; color: rgba(255,255,255,0.65); font-style: italic; margin-top: 0.5rem; line-height: 1.35; }
/* Yesterday's Results strip (slim badges above Play of the Day) */
.yesterday-strip { display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem; margin-bottom: 1rem; padding: 0.5rem 0; font-family: inherit; }
.yesterday-strip-label { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: rgba(255,255,255,0.7); margin-right: 0.25rem; }
.yesterday-badge { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.35rem 0.75rem; border-radius: 8px; font-size: 0.9rem; font-weight: 700; }
.yesterday-badge--win { background: rgba(46,125,50,0.5); color: #a5d6a7; }
.yesterday-badge--loss { background: rgba(198,40,40,0.5); color: #ef9a9a; }
.yesterday-badge--pending { background: rgba(96,125,139,0.4); color: #b0bec5; }
/* POTD streak tracker (dots below Yesterday's Results) */
.streak-row { display: flex; align-items: center; gap: 0.35rem; flex-wrap: wrap; margin-bottom: 0.5rem; font-size: 0.8rem; color: rgba(255,255,255,0.7); }
.streak-label { min-width: 4rem; font-weight: 600; color: rgba(255,255,255,0.85); }
.streak-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.streak-dot--w { background: #66bb6a; }
.streak-dot--l { background: #e57373; }
.streak-dot--p { background: #90a4ae; }
.streak-flame { margin-left: 0.25rem; }
</style>
"""


def _render_yesterday_strip_html(rows: list[dict]) -> str:
    """Render slim 'Yesterday's Results' strip: sport, team, W/L badge. Green=win, red=loss, grey=push/unresolved."""
    if not rows:
        return ""
    parts = ['<div class="yesterday-strip">', '<span class="yesterday-strip-label">Yesterday\'s Results</span>']
    for r in rows:
        sport = _html_escape(str(r.get("sport", "")))
        side = _html_escape(str(r.get("side", "")))
        res = r.get("result")
        if res == "W":
            cls = "yesterday-badge yesterday-badge--win"
            label = "W"
        elif res == "L":
            cls = "yesterday-badge yesterday-badge--loss"
            label = "L"
        else:
            cls = "yesterday-badge yesterday-badge--pending"
            label = "P" if res == "P" else "—"
        parts.append(f'<span class="{cls}">{sport} · {side} · {label}</span>')
    parts.append("</div>")
    return "\n".join(parts)


def _get_last_10_potd_results() -> tuple[list[str], list[str]]:
    """
    Return (pick1_results, pick2_results): last 10 days of NCAAB Pick 1 and Pick 2 results (W/L/P) each.
    """
    if STRIP_DOWN_MODE:
        return ([], [])
    from_date = date.today() - timedelta(days=31)
    to_date = date.today() - timedelta(days=1)
    hist = _load_play_history_cached(from_date_iso=from_date.isoformat(), to_date_iso=to_date.isoformat())
    if hist.empty or "my_edge_pct" not in hist.columns:
        return ([], [])
    hist = hist.copy()
    hist["result_clean"] = hist["result"].apply(
        lambda x: str(x).strip().upper() if x is not None and not pd.isna(x) else None
    )
    top = hist[hist["sport"].astype(str).str.strip().isin(("NCAAB Pick 1", "NCAAB Pick 2"))].copy()
    if top.empty:
        return ([], [])
    top = top.loc[top.groupby(["date_generated", "sport"])["my_edge_pct"].idxmax()].reset_index(drop=True)
    top = top.sort_values(["date_generated", "sport"], ascending=[True, True])
    dates_desc = sorted(top["date_generated"].unique(), reverse=True)[:10]
    dates_asc = list(reversed(dates_desc))
    codes_1: list[str] = []
    codes_2: list[str] = []
    for d in dates_asc:
        day = top[top["date_generated"] == d]
        r1 = day[day["sport"].astype(str).str.strip() == "NCAAB Pick 1"]
        r2 = day[day["sport"].astype(str).str.strip() == "NCAAB Pick 2"]
        res1 = r1.iloc[0].get("result_clean") if not r1.empty else None
        res2 = r2.iloc[0].get("result_clean") if not r2.empty else None
        codes_1.append(res1 if res1 in ("W", "L", "P") else "P")
        codes_2.append(res2 if res2 in ("W", "L", "P") else "P")
    return (codes_1, codes_2)


def _render_streak_html(pick1_results: list[str], pick2_results: list[str]) -> str:
    """Render two rows of POTD streak dots (Pick 1, Pick 2) with 🔥 if 3+ wins in a row."""
    def _one_row(label: str, results: list[str]) -> str:
        if not results:
            return ""
        current_streak = 0
        for r in reversed(results):
            if r == "W":
                current_streak += 1
            else:
                break
        parts = [f'<div class="streak-row"><span class="streak-label">{_html_escape(label)}</span>']
        for r in results:
            cls = f"streak-dot streak-dot--{r.lower()}"
            parts.append(f'<span class="{cls}" title="{r}"></span>')
        if current_streak >= 3:
            parts.append(f'<span class="streak-flame" title="Current win streak: {current_streak}">🔥</span>')
        parts.append("</div>")
        return "\n".join(parts)
    out = []
    if pick1_results:
        out.append(_one_row("Pick 1", pick1_results))
    if pick2_results:
        out.append(_one_row("Pick 2", pick2_results))
    return "\n".join(out) if out else ""


def _value_play_reasoning(row: pd.Series) -> str:
    """
    One or two sentence plain-English reasoning from model feature context: rest/B2B, home court, market type, injury.
    Uses home_team, away_team, is_home_b2b, is_away_b2b, Market, Selection, and optional injury flags.
    If the row has reason or reasoning_summary (e.g. from NCAAB manual-odds cache), use that so dynamic KenPom/BARTHAG text is shown.
    """
    reason = row.get("reason") or row.get("reasoning_summary")
    if reason is not None and str(reason).strip():
        return str(reason).strip()
    home = str(row.get("home_team", "")).strip()
    away = str(row.get("away_team", "")).strip()
    selection = str(row.get("Selection", "")).strip()
    market = str(row.get("Market", ""))
    is_home_b2b = bool(row.get("is_home_b2b", False))
    is_away_b2b = bool(row.get("is_away_b2b", False))
    selection_is_home = (
        selection.lower() == home.lower()
        or home.lower() in selection.lower()
        or selection.lower() in home.lower()
    )
    parts: list[str] = []
    # Rest / B2B
    if is_away_b2b and not is_home_b2b:
        if selection_is_home:
            parts.append("Model favors the home team with the road side on a back-to-back.")
        else:
            parts.append("Road team is on a back-to-back; the model sees fatigue discount in the line.")
    elif is_home_b2b and not is_away_b2b:
        if not selection_is_home:
            parts.append("Model favors the road team with the home side on a back-to-back.")
        else:
            parts.append("Home team is on a back-to-back; the model sees fatigue priced in.")
    elif is_home_b2b and is_away_b2b:
        parts.append("Both teams on a back-to-back; power ratings and pace drive the edge.")
    # Home court
    if selection_is_home and "back-to-back" not in " ".join(parts).lower():
        parts.append("Home court edge is amplified in this matchup.")
    elif not selection_is_home and not parts:
        parts.append("Road side offers value against the market’s home bias.")
    # Market-specific
    if market == "totals":
        if not parts:
            parts.append("Pace-adjusted total and key-number analysis suggest the line is off.")
        else:
            parts.append("Pace and key numbers support this total.")
    elif market == "spreads" and not parts:
        parts.append("Power ratings and schedule context suggest the spread is mispriced.")
    elif market == "h2h" and not parts:
        parts.append("Model probability vs. implied odds shows edge on the moneyline.")
    # Injury
    if row.get("top5_out_or_doubtful_home") or row.get("top5_out_or_doubtful_away"):
        parts.append("Injury context is reflected in the model’s adjustment.")
    if not parts:
        return "Model identifies value from power ratings and market line."
    return " ".join(parts)


def _vp_bet_type_label(row: pd.Series) -> str:
    """Short bet label for value-play card: e.g. 'Spread · Lakers -3.5', 'Over 220.5'."""
    return _potd_badge_text(row)


def _march_context_badges(row: pd.Series) -> str:
    """Compact March context line for NCAAB: conf. tourney, seeds, bubble vs clinched. Returns '—' for non-NCAAB or no flags."""
    if str(row.get("League", "")).strip().upper() != "NCAAB":
        return "—"
    parts = []
    if row.get("is_conference_tournament"):
        parts.append("Conf. tourney")
    h_seed, a_seed = row.get("home_seed"), row.get("away_seed")
    if pd.notna(h_seed) and pd.notna(a_seed) and h_seed is not None and a_seed is not None:
        try:
            parts.append(f"#{int(h_seed)} vs #{int(a_seed)}")
        except (TypeError, ValueError):
            pass
    elif pd.notna(h_seed) and h_seed is not None:
        try:
            parts.append(f"#{int(h_seed)} vs ?")
        except (TypeError, ValueError):
            pass
    elif pd.notna(a_seed) and a_seed is not None:
        try:
            parts.append(f"? vs #{int(a_seed)}")
        except (TypeError, ValueError):
            pass
    home_clinched = row.get("home_clinched_ncaa_bid") in (True, 1)
    away_clinched = row.get("away_clinched_ncaa_bid") in (True, 1)
    home_bubble = row.get("home_is_bubble_team") in (True, 1)
    away_bubble = row.get("away_is_bubble_team") in (True, 1)
    if home_bubble or away_bubble or home_clinched or away_clinched:
        labels = []
        if away_bubble:
            labels.append("Bubble")
        elif away_clinched:
            labels.append("Clinched")
        else:
            labels.append("—")
        labels.append(" vs ")
        if home_bubble:
            labels.append("Bubble")
        elif home_clinched:
            labels.append("Clinched")
        else:
            labels.append("—")
        badge = "".join(labels).replace(" — vs —", "").strip()
        if badge and badge != "— vs —":
            parts.append(badge)
    return " · ".join(parts) if parts else "—"


def _is_top_seed_play(row: pd.Series, top_seed_max: int = 6) -> bool:
    """True if play involves a top seed (e.g. 1–6 = top ~25 teams). Used for Tournament Context badge."""
    try:
        h, a = row.get("home_seed"), row.get("away_seed")
        if pd.notna(h) and h is not None and int(h) <= top_seed_max:
            return True
        if pd.notna(a) and a is not None and int(a) <= top_seed_max:
            return True
    except (TypeError, ValueError):
        pass
    return False


def _render_value_play_card_html(row: pd.Series, edge_max_pct: float = 15.0, march_madness_mode: bool = False) -> str:
    """One All Value Plays card: matchup, bet type, edge %, odds, confidence bar, reasoning. Pick = Home/Away [Team]; edge green (Home) or blue (Away)."""
    league = str(row.get("League", ""))
    league_class = "nba" if league == "NBA" else "ncaab"
    matchup = _html_escape(str(row.get("Event", "")))
    bet_type = _html_escape(_vp_bet_type_label(row))
    edge_pct = float(row.get("Value (%)", 0))
    odds_str = format_american(row.get("Odds", 0))
    bar_pct = min(100.0, max(0.0, (edge_pct / edge_max_pct) * 100.0))
    reasoning = _html_escape(_value_play_reasoning(row))
    underdog_badge = '<span class="vp-underdog-value">Underdog value</span>' if row.get("underdog_value") else ""
    tournament_context_badge = ""
    if march_madness_mode and league == "NCAAB" and _is_top_seed_play(row):
        tournament_context_badge = '<span class="vp-tournament-context">Tournament Context</span>'
    # Pick label: "Home [Team]" or "Away [Team]"
    sel = str(row.get("Selection", "")).strip()
    h, a = str(row.get("home_team", "")).strip(), str(row.get("away_team", "")).strip()
    if sel and h and (sel.lower() == h.lower() or sel.lower() in h.lower() or h.lower() in sel.lower()):
        pick_label = _html_escape(f"Home {h}")
        edge_color = "#22c55e"
    else:
        pick_label = _html_escape(f"Away {a}") if a else _html_escape(sel or "—")
        edge_color = "#3b82f6"
    march_line = ""
    if league == "NCAAB":
        march_badges = _march_context_badges(row)
        if march_badges and march_badges != "—":
            march_line = f'<div class="vp-march-context">{_html_escape(march_badges)}</div>'
    html = f"""
    <div class="vp-card">
        <div class="vp-card-league {league_class}">{_html_escape(league)}</div>
        <div class="vp-matchup">{matchup}</div>
        <div class="vp-meta">
            <span class="vp-bet-type">{bet_type}</span>
            <span class="vp-pick">Pick: {pick_label}</span>
            <span class="vp-edge" style="color:{edge_color}">{edge_pct:.1f}% edge</span>
            <span class="vp-odds">{odds_str}</span>
            {underdog_badge}
            {tournament_context_badge}
        </div>
        {march_line}
        <div class="vp-confidence-wrap">
            <div class="vp-confidence-bar"><div class="vp-confidence-fill" style="width:{bar_pct:.1f}%"></div></div>
        </div>
        <div class="vp-reasoning">{reasoning}</div>
    </div>
    """
    return _html_one_line_per_block(html)


def _render_potd_card_html(
    league: str,
    row: Optional[pd.Series],
    accent: str,
    feature_matrix: Optional[pd.DataFrame] = None,
    odds_as_of: Optional[datetime] = None,
    b2b_teams: Optional[set[str]] = None,
    march_madness_mode: bool = False,
    mlb_park_line: Optional[str] = None,
    mlb_away_display: Optional[str] = None,
    mlb_home_display: Optional[str] = None,
) -> str:
    """Return HTML for one Play of the Day card. accent: 'blue' | 'orange' | 'grey'. Grey when no play.
    Optional MLB Overview: park line + matchup strings with recent-form tags."""
    if row is None or (isinstance(row, pd.Series) and (row.empty or len(row) == 0)) or (isinstance(row, dict) and len(row) == 0):
        html = f"""
        <div class="potd-card potd-card--grey">
            <div class="potd-league">{_html_escape(league)}</div>
            <div class="potd-empty-msg">No Play Today</div>
        </div>
        """
        return _html_one_line_per_block(html)
    r = row
    edge_pct = float(r.get("Value (%)", 0))
    high_variance = bool(r.get("high_variance", False))
    if high_variance:
        confidence = "High Variance"
    elif edge_pct >= POTD_HIGH_CONFIDENCE_EDGE_PCT:
        confidence = "High"
    else:
        confidence = "Medium"
    if (
        mlb_away_display
        and mlb_home_display
        and league
        and "MLB" in league
    ):
        matchup_html = f'{_html_escape(mlb_away_display)} <span class="potd-vs">@</span> {_html_escape(mlb_home_display)}'
    else:
        away, home = _parse_event_teams(str(r.get("Event", "")))
        away_abbrev = _team_abbrev(away, league)
        home_abbrev = _team_abbrev(home, league)
        matchup_html = f"{_html_escape(away)}"
        if away_abbrev:
            matchup_html += f' <span class="potd-abbrev">({away_abbrev})</span>'
        matchup_html += f' <span class="potd-vs">@</span> {_html_escape(home)}'
        if home_abbrev:
            matchup_html += f' <span class="potd-abbrev">({home_abbrev})</span>'
    commence = str(r.get("commence_time", "") or "").strip()
    tipoff = format_start_time(commence) if commence else (str(r.get("Start Time", "") or "").strip() or "—")
    _start_lbl = "First pitch" if league and "MLB" in league else "Tip-off"
    badge_text = _potd_badge_text(r)
    reason = r.get("reason")
    if not reason and (feature_matrix is not None or (b2b_teams and len(b2b_teams) > 0)):
        reason = _potd_reason(r, feature_matrix=feature_matrix, b2b_teams=b2b_teams)
    if not reason:
        reason = "Our model sees value on this play."
    odds_str = format_american(r.get("Odds", 0))
    market = str(r.get("Market", "")).strip().lower()
    point_val = r.get("point_x") or r.get("point")
    current_line = ""
    if point_val is not None and not pd.isna(point_val):
        try:
            pt = float(point_val)
            if market == "spreads":
                current_line = f"Current line: {pt:+.1f}"
            elif market == "totals":
                current_line = f"Current O/U: {pt:.1f}"
        except (TypeError, ValueError):
            pass
    line_as_of = ""
    if odds_as_of:
        try:
            et = odds_as_of.astimezone(ZoneInfo("America/New_York"))
            line_as_of = f"Line as of {et.strftime('%b %d, %I:%M %p ET')}"
        except Exception:
            line_as_of = f"Line as of {odds_as_of.strftime('%b %d, %I:%M %p')}"
    potd_march = ""
    if league and "NCAAB" in league:
        march_badges = _march_context_badges(r)
        if march_badges and march_badges != "—":
            potd_march = f'<div class="potd-march-context">{_html_escape(march_badges)}</div>'
    potd_tournament_badge = ""
    if march_madness_mode and league and "NCAAB" in league and _is_top_seed_play(r):
        potd_tournament_badge = '<div class="potd-tournament-context">Tournament Context</div>'
    _mlb_sp = str(r.get("mlb_sp_line", "") or "").strip() if (league and "MLB" in league) else ""
    _park_html = f'<div class="potd-current-line">{_html_escape(mlb_park_line)}</div>' if mlb_park_line else ""
    _sp_html = f'<div class="potd-current-line">{_html_escape(_mlb_sp)}</div>' if _mlb_sp else ""
    html = f"""
    <div class="potd-card potd-card--{accent}">
        <div class="potd-league">{_html_escape(league)}</div>
        <div class="potd-tipoff">{_html_escape(_start_lbl)}: {_html_escape(tipoff)}</div>
        <div class="potd-matchup">{matchup_html}</div>
        {_park_html}
        {_sp_html}
        <div class="potd-badge-row"><div class="potd-badge">{_html_escape(badge_text)}</div><span class="potd-odds-inline">{odds_str}</span></div>
        {f'<div class="potd-current-line">{_html_escape(current_line)}</div>' if current_line else ''}
        <div class="potd-edge">{edge_pct:.1f}% Edge</div>
        {potd_tournament_badge}
        {potd_march}
        <div class="potd-confidence{f' potd-confidence--warning' if high_variance else ''}">Confidence: {_html_escape(confidence)}</div>
        <div class="potd-reason">{_html_escape(reason)}</div>
        {f'<div class="potd-line-as-of">{_html_escape(line_as_of)}</div>' if line_as_of else ''}
    </div>
    """
    return _html_one_line_per_block(html)


def format_start_time(commence_time: str) -> str:
    """Format ISO commence_time for display in Eastern Time (e.g. Feb 28, 7:30 PM ET)."""
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


st.set_page_config(
    page_title="Bobby Bottle's Betting Model",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Project root for logs and scripts
_APP_ROOT = Path(__file__).resolve().parent


def _run_startup_snapshot_and_merge() -> None:
    """Background: fetch NCAAB odds snapshot, merge closing into games, write snapshot log timestamp.
    In March (on or after Selection Sunday), also fetch tournament seeds from ESPN for March Madness mode."""
    try:
        subprocess.run(
            [sys.executable, str(_APP_ROOT / "scripts" / "fetch_ncaab_odds_snapshot.py")],
            cwd=str(_APP_ROOT),
            capture_output=True,
            timeout=120,
            shell=False,
        )
        from engine.historical_odds import merge_historical_closing_into_games
        merge_historical_closing_into_games(
            espn_db_path=_APP_ROOT / "data" / "espn.db",
            odds_db_path=_APP_ROOT / "data" / "odds.db",
        )
        log_dir = _APP_ROOT / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "snapshot_log.txt", "a", encoding="utf-8") as f:
            f.write(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") + "\n")
        # March: after Selection Sunday, refresh tournament seeds for March Madness mode
        if is_after_selection_sunday(date.today()):
            subprocess.run(
                [sys.executable, str(_APP_ROOT / "scripts" / "fetch_ncaab_tournament_seeds.py")],
                cwd=str(_APP_ROOT),
                capture_output=True,
                timeout=30,
                shell=False,
            )
    except Exception:
        pass


def _start_scheduler_once() -> None:
    """Start APScheduler (8am auto-result, 9am archive, Monday 6am NCAAB retrain) once per session."""
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
        from engine.archive_job import run_daily_archive
        from engine.auto_result_job import run_auto_result
        from engine.retrain_job import run_weekly_retrain
        scheduler = BackgroundScheduler(timezone="America/New_York")
        scheduler.add_job(run_auto_result, CronTrigger(hour=8, minute=0), id="play_auto_result")
        scheduler.add_job(run_daily_archive, CronTrigger(hour=9, minute=0), id="play_history_archive")
        scheduler.add_job(run_weekly_retrain, CronTrigger(day_of_week="mon", hour=6, minute=0), id="ncaab_weekly_retrain")
        scheduler.start()
    except Exception:
        pass


if ENABLE_SCHEDULER and "scheduler_started" not in st.session_state:
    _start_scheduler_once()
    st.session_state["scheduler_started"] = True

if ENABLE_STARTUP_THREAD and "startup_snapshot_started" not in st.session_state:
    st.session_state["startup_snapshot_started"] = True
    t = threading.Thread(target=_run_startup_snapshot_and_merge, daemon=True)
    t.start()


def _read_last_snapshot_time() -> Optional[str]:
    """Last line of data/logs/snapshot_log.txt as display timestamp, or None."""
    p = _APP_ROOT / "data" / "logs" / "snapshot_log.txt"
    if not p.exists():
        return None
    try:
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            return None
        raw = lines[-1].strip()
        if not raw:
            return None
        # Show as-is if it looks like ISO; otherwise show raw
        return raw
    except Exception:
        return None


def _read_last_retrain_status() -> Optional[str]:
    """Parse last line of retrain_log.txt -> 'YYYY-MM-DD — X games, XX.XX%' or None."""
    p = _APP_ROOT / "data" / "logs" / "retrain_log.txt"
    if not p.exists():
        return None
    try:
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            return None
        line = lines[-1].strip()
        # Format: "2026-03-06 games=597 val_accuracy=0.8119"
        m = re.match(r"^(\d{4}-\d{2}-\d{2})\s+games=(\d+)\s+val_accuracy=([\d.]+)", line)
        if m:
            date_str, games, acc = m.group(1), m.group(2), m.group(3)
            pct = float(acc) * 100
            return f"{date_str} — {games} games, {pct:.2f}%"
        m = re.match(r"^(\d{4}-\d{2}-\d{2})\s+games=\?\s+val_accuracy=skipped", line)
        if m:
            return f"{m.group(1)} — skipped (no train)"
        return line
    except Exception:
        return None


def _load_tournament_eligible_teams() -> set[str]:
    """Load team names from data/ncaab_seeds.csv (tournament field). Returns set of lowercased names for matching."""
    p = _APP_ROOT / "data" / "ncaab_seeds.csv"
    if not p.exists():
        return set()
    try:
        df = pd.read_csv(p)
        col = "team" if "team" in df.columns else df.columns[0]
        s = df[col].astype(str).str.strip().str.lower()
        return set(s[(s != "") & s.notna()].tolist())
    except Exception:
        return set()


def get_injury_alerts(sport: str) -> dict[str, str]:
    """
    Return a map of event_name -> injury alert text. Uses real injury data when available.
    Until a real API is connected, returns no alerts.
    """
    return {}


def add_injury_alerts_to_value_plays(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    """Add 'Injury Alert' column to value plays DataFrame. '—' when no alert. Placed after Event."""
    if df.empty:
        df["Injury Alert"] = []
        return df
    alerts_map = get_injury_alerts(sport)
    df = df.copy()
    df["Injury Alert"] = df["Event"].map(lambda e: alerts_map.get(e, "") or "—")
    cols = list(df.columns)
    cols.remove("Injury Alert")
    idx = cols.index("Event") + 1 if "Event" in cols else 0
    cols.insert(idx, "Injury Alert")
    return df[cols]


def _fixtures_to_value_plays(
    fixtures: pd.DataFrame,
    bankroll: float,
    kelly_frac: float = 0.25,
    min_ev_pct: float = 5.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    From a fixtures DataFrame (event_name, odds_home, odds_draw, odds_away),
    build one row per selection with synthetic model_prob, compute EV% and
    Kelly stake; return rows where EV% > min_ev_pct.
    """
    np.random.seed(seed)
    rows = []
    for _, r in fixtures.iterrows():
        event = r.get("event_name") or f"{r.get('home_team', '')} vs {r.get('away_team', '')}"
        for label, odds_col in [("Home", "odds_home"), ("Draw", "odds_draw"), ("Away", "odds_away")]:
            odds_val = r.get(odds_col)
            if pd.isna(odds_val) or abs(float(odds_val)) < 100:
                continue
            odds_val = float(odds_val)  # American
            implied = implied_probability(odds_val)
            edge = np.random.uniform(0, 0.12)
            model_prob = min(0.92, implied + edge)
            ev_decimal = (model_prob * american_to_decimal(odds_val)) - 1.0
            ev_pct = ev_decimal * 100.0
            if ev_pct < min_ev_pct:
                continue
            frac = kelly_fraction(odds_val, model_prob, fraction=kelly_frac)
            stake = round(bankroll * frac, 2)
            rows.append({
                "Event": event,
                "Selection": label,
                "Odds": int(round(odds_val)),
                "Value (%)": round(ev_pct, 2),
                "Recommended Stake": stake,
            })
    return pd.DataFrame(rows)


def _basketball_to_value_plays(
    bb_df: pd.DataFrame,
    bankroll: float,
    kelly_frac: float = 0.25,
    min_ev_pct: float = 5.0,
    seed: int = 43,
) -> pd.DataFrame:
    """
    From basketball odds DataFrame (event_name, market_type, selection, odds),
    add synthetic model_prob, compute EV% and Kelly stake; return rows where EV% > min_ev_pct.
    """
    if bb_df.empty:
        return pd.DataFrame(columns=["Event", "Selection", "Market", "Odds", "Value (%)", "Recommended Stake"])
    np.random.seed(seed)
    rows = []
    for _, r in bb_df.iterrows():
        odds_val = r.get("odds")
        if pd.isna(odds_val) or abs(float(odds_val)) < 100:
            continue
        odds_val = float(odds_val)  # American
        implied = implied_probability(odds_val)
        edge = np.random.uniform(0, 0.12)
        model_prob = min(0.92, implied + edge)
        ev_decimal = (model_prob * american_to_decimal(odds_val)) - 1.0
        ev_pct = ev_decimal * 100.0
        if ev_pct < min_ev_pct:
            continue
        frac = kelly_fraction(odds_val, model_prob, fraction=kelly_frac)
        stake = round(bankroll * frac, 2)
        event_name = r.get("event_name", "")
        selection = r.get("selection", "")
        market_type = r.get("market_type", "")
        rows.append({
            "Event": event_name,
            "Selection": selection,
            "Market": market_type,
            "Odds": int(round(odds_val)),
            "Value (%)": round(ev_pct, 2),
            "Recommended Stake": stake,
        })
    return pd.DataFrame(rows)


# Sport keys for live Best Value Plays (The Odds API). NCAAB only for now; NBA code kept in backend.
LIVE_ODDS_SPORT_KEYS = [BASKETBALL_NCAAB]


# Edge thresholds (tuned for NCAAB): Value % in [3%, 15%); >= 15% flagged as Potential Data Error
EV_EPSILON_MIN_PCT = 3.0
EV_EPSILON_MAX_PCT = 15.0

# Minimum number of bookmakers offering a line for an outcome to be eligible as a value play
# Temporarily 1 when DEBUG_RELAX_FILTERS=1 (see _live_odds_to_value_plays)
MIN_BOOKMAKERS_VALUE_PLAY = 3


def _aggregate_odds_best_line_avg_implied(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate odds across bookmakers: one row per (event, market, selection, point).
    Limited to NCAAB only to reduce memory; NBA rows are dropped before aggregation.
    """
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
    # Standard two-outcome vig removal: normalize both sides so they sum to 100%
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
) -> tuple[pd.DataFrame, int]:
    """
    In-house line (power ratings) vs market; key-number adjustment; fractional Kelly rounded to half-units (1–3%).
    Only includes Value % in [min_ev_pct, max_ev_pct). Excludes odds > +500 unless include_high_risk.
    When min_bookmakers_override is set (e.g. 1 for ESPN fallback), that overrides MIN_BOOKMAKERS_VALUE_PLAY.
    """
    if odds_df.empty:
        if debug:
            _log_dir = Path(__file__).resolve().parent / "data" / "logs"
            _log_path = _log_dir / "debug_value_plays.log"
            _log_dir.mkdir(parents=True, exist_ok=True)
            try:
                f = open(_log_path, "w")
            except Exception:
                f = open("/tmp/debug_value_plays.log", "w")
            f.write("[DEBUG_VALUE_PLAYS] aggregated_rows=0 (empty odds_df)\n")
            f.close()
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
    _debug_relax = os.environ.get("DEBUG_RELAX_FILTERS", "") == "1"  # min_books=1, consensus disabled
    if min_bookmakers_override is not None:
        _min_books = min_bookmakers_override
    else:
        _min_books = 1 if (debug and _debug_relax) else MIN_BOOKMAKERS_VALUE_PLAY
    if debug:
        n_skip_bookmakers = n_skip_odds = n_skip_high_risk = n_skip_no_consensus = n_below_min = n_above_max = n_skip_stake = 0
        _log_dir = Path(__file__).resolve().parent / "data" / "logs"
        _log_path = _log_dir / "debug_value_plays.log"
        _log_dir.mkdir(parents=True, exist_ok=True)
        try:
            _dbg_file = open(_log_path, "w")
        except Exception:
            _dbg_file = open("/tmp/debug_value_plays.log", "w")
        _debug_log_path_used = getattr(_dbg_file, "name", str(_log_path))
        def _dbg(s: str):
            print(s)
            _dbg_file.write(s + "\n")
            _dbg_file.flush()
        _dbg("\n[DEBUG_VALUE_PLAYS] Thresholds: min_ev_pct={}%, max_ev_pct={}%, MIN_BOOKMAKERS_VALUE_PLAY={} (effective={})".format(
            min_ev_pct, max_ev_pct, MIN_BOOKMAKERS_VALUE_PLAY, _min_books))
        _dbg("[DEBUG_VALUE_PLAYS] DEBUG_RELAX_FILTERS={} (consensus disabled={})".format(
            os.environ.get("DEBUG_RELAX_FILTERS"), _debug_relax))
    for _, r in odds_df.iterrows():
        # When using aggregated odds, require at least _min_books books on this line
        if "bookmaker_count" in r.index and r.get("bookmaker_count") is not None and not pd.isna(r.get("bookmaker_count")):
            if int(r.get("bookmaker_count", 0)) < _min_books:
                if debug:
                    n_skip_bookmakers += 1
                    _dbg("[DEBUG] SKIP bookmakers<{}: {} | {} | {} | {} (books={})".format(
                        _min_books, r.get("league"), r.get("event_name"), r.get("market_type"), r.get("selection"), int(r.get("bookmaker_count", 0))))
                continue
        odds_val = r.get("odds")
        if pd.isna(odds_val) or abs(float(odds_val)) < 100:
            if debug:
                n_skip_odds += 1
                _dbg("[DEBUG] SKIP odds invalid: {} | {} | {} | {}".format(
                    r.get("league"), r.get("event_name"), r.get("market_type"), r.get("selection")))
            continue
        odds_val = float(odds_val)
        if odds_val > HIGH_RISK_ODDS_AMERICAN and not include_high_risk:
            if debug:
                n_skip_high_risk += 1
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
        feature_row = get_feature_row_for_game(
            feature_matrix, home_team, away_team, league_lookup, game_date=game_date_str
        ) if feature_matrix is not None and not feature_matrix.empty else None

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
                if debug:
                    n_skip_no_consensus += 1
                    _dbg("[DEBUG] SKIP no_consensus: {} | {} | {} | {} (model_prob={:.3f})".format(
                        league, r.get("event_name"), market_type, selection, model_prob))
                continue
        elif market_type == "spreads" and point is not None:
            market_spread = -float(point)
            home_rating = power_ratings.get(home_team, default_rating) - get_schedule_fatigue_penalty(home_team, as_of_date)
            away_rating = power_ratings.get(away_team, default_rating) - get_schedule_fatigue_penalty(away_team, as_of_date)
            in_house_spread = in_house_spread_from_ratings(home_rating, away_rating)
            we_cover_favorite = "-" in selection
            fallback = model_prob_from_in_house_spread(in_house_spread, market_spread, we_cover_favorite)
            model_prob, consensus_ok = consensus_spread(
                feature_row, market_spread, we_cover_favorite, in_house_spread, fallback, league=league_lookup
            )
            if _debug_relax:
                consensus_ok = True
            if not consensus_ok:
                if debug:
                    n_skip_no_consensus += 1
                    _dbg("[DEBUG] SKIP no_consensus: {} | {} | {} | {} (model_prob={:.3f})".format(
                        league, r.get("event_name"), market_type, selection, model_prob))
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
                if debug:
                    n_skip_no_consensus += 1
                    _dbg("[DEBUG] SKIP no_consensus: {} | {} | {} | {} (model_prob={:.3f})".format(
                        league, r.get("event_name"), market_type, selection, model_prob))
                continue
        else:
            implied = implied_probability_no_vig(odds_val)
            edge = np.random.uniform(0, 0.08)
            model_prob = damp_probability(min(0.92, implied + edge))

        # Edge at best line; use fair implied (two-sided vig removal) when available, else avg or single-outcome no-vig
        implied_prob = r.get("fair_implied_prob")
        if implied_prob is None or pd.isna(implied_prob):
            implied_prob = r.get("avg_implied_prob")
        if implied_prob is None or pd.isna(implied_prob):
            implied_prob = implied_probability_no_vig(odds_val)
        implied_prob = float(implied_prob)
        # Underdog value: model thinks underdog covers more often than market implies (sharp target)
        is_underdog_side = market_type == "spreads" and "+" in selection
        underdog_value = is_underdog_side and (model_prob > implied_prob)
        ev_decimal = (model_prob * american_to_decimal(odds_val)) - 1.0
        ev_pct = ev_decimal * 100.0
        line_for_key = float(point) if point is not None else 0.0
        ev_pct += key_number_value_adjustment(line_for_key, league, market_type)
        if debug:
            _dbg("[DEBUG] RAW_EDGE: {} | {} | {} | {} | model_prob={:.3f} implied_prob={:.3f} raw_edge_pct={:.2f}%".format(
                league, r.get("event_name"), market_type, selection, model_prob, implied_prob, ev_pct))
        if ev_pct < min_ev_pct:
            if debug:
                n_below_min += 1
            continue
        if ev_pct >= max_ev_pct:
            n_flagged += 1
            if debug:
                n_above_max += 1
            continue
        full_kelly = kelly_fraction(odds_val, model_prob, fraction=1.0)
        stake = fractional_kelly_half_units(bankroll, full_kelly, HALF_UNIT_PCT, MAX_KELLY_PCT_WALTERS)
        if stake <= 0:
            if debug:
                n_skip_stake += 1
            continue
        if debug:
            _dbg("[DEBUG] PASS: {} | {} | {} | {} | edge={:.2f}% stake={}".format(
                league, r.get("event_name"), market_type, selection, ev_pct, stake))
        is_home_b2b = home_team in b2b_teams
        is_away_b2b = away_team in b2b_teams
        rows.append({
            "League": league,
            "Event": r.get("event_name", ""),
            "Selection": selection,
            "Market": market_type,
            "Odds": int(round(odds_val)),
            "Value (%)": round(ev_pct, 2),
            "Recommended Stake": stake,
            "Injury Alert": "—",
            "Start Time": format_start_time(r.get("commence_time", "")),
            "commence_time": r.get("commence_time", ""),
            "model_prob": model_prob,
            "implied_prob": implied_prob,
            "point": point,
            "home_team": home_team,
            "away_team": away_team,
            "is_home_b2b": is_home_b2b,
            "is_away_b2b": is_away_b2b,
            "underdog_value": underdog_value,
        })
    if debug:
        _dbg("[DEBUG_VALUE_PLAYS] Summary: aggregated_rows={} | skip_bookmakers={} | skip_odds={} | skip_high_risk={} | skip_no_consensus={} | below_min_ev={} | above_max_ev={} | skip_stake={} | PASSED={}".format(
            len(odds_df), n_skip_bookmakers, n_skip_odds, n_skip_high_risk, n_skip_no_consensus, n_below_min, n_above_max, n_skip_stake, len(rows)))
        _dbg_file.close()
    return pd.DataFrame(rows), n_flagged


# Empty historical dataset columns for BettingEngine (no mock data; use real data or upload)
BASKETBALL_HISTORICAL_COLUMNS = ["event_id", "event_name", "odds", "model_prob", "result"]


# -----------------------------------------------------------------------------
# Sidebar — single combined model (no strategy dropdown)
# -----------------------------------------------------------------------------

st.sidebar.header("Settings")

# Fixed bankroll used internally for stake sizing (Kelly); not shown in UI
BANKROLL_FOR_STAKES = 1000.0

# Single model: value + Kelly (quarter Kelly) for stakes
kelly_frac = 0.25
strategy_fn = strategy_kelly(kelly_fraction_param=kelly_frac)

# API key: .streamlit/secrets.toml ([the_odds_api] api_key = "...") → env ODDS_API_KEY → sidebar
# Fallback: read .streamlit/secrets.toml from app dir (same as scripts/run_value_plays_debug.py) so dashboard finds key even if st.secrets not loaded
def _get_odds_api_key() -> str:
    key = ""
    try:
        if hasattr(st, "secrets") and st.secrets:
            # Prefer [the_odds_api] api_key from secrets.toml
            section = st.secrets.get("the_odds_api")
            if section is not None:
                key = getattr(section, "api_key", None) or (section.get("api_key") if hasattr(section, "get") else None) or ""
            key = (key or st.secrets.get("ODDS_API_KEY") or "").strip()
    except Exception:
        pass
    if not key:
        key = (os.environ.get("ODDS_API_KEY") or "").strip()
    if not key:
        try:
            secrets_path = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
            if secrets_path.exists():
                text = secrets_path.read_text()
                m = re.search(r'api_key\s*=\s*["\']([^"\']+)["\']', text)
                if m:
                    key = m.group(1).strip()
        except Exception:
            pass
    return key


odds_api_key = _get_odds_api_key()
if not odds_api_key:
    odds_api_key = st.sidebar.text_input(
        "Odds API key (for live Best Value Plays)",
        type="password",
        help="Get a key at the-odds-api.com. Or add to .streamlit/secrets.toml as [the_odds_api] api_key = \"...\"",
    )
    if odds_api_key:
        odds_api_key = odds_api_key.strip()
else:
    st.sidebar.caption("Odds API key loaded from secrets")

include_high_risk_odds = st.sidebar.checkbox(
    "Include high-risk odds (+500 or higher)",
    value=False,
    help="By default, odds greater than +500 are excluded from Best Value (flagged as High Risk).",
)

# March Madness mode: tournament-eligible NCAAB only, 5% min edge, Tournament Context badge for top seeds
march_madness_mode = st.sidebar.checkbox(
    "March Madness mode",
    value=is_after_selection_sunday(date.today()),
    help="Filter NCAAB to tournament-eligible teams only, 5% min edge, and show Tournament Context badge for top-25 seeds. Auto-on after Selection Sunday.",
    key="march_madness_mode",
)

# Status: last NCAAB snapshot and last retrain (from log files)
if STRIP_DOWN_MODE:
    st.sidebar.warning("**Strip-down mode:** Pipeline, scheduler, and DB loads disabled. Set `STRIP_DOWN_MODE = False` in app.py to restore.")
st.sidebar.caption("**Status**")
last_snapshot = _read_last_snapshot_time()
st.sidebar.caption(f"Last snapshot: {last_snapshot or '—'}")
last_retrain = _read_last_retrain_status()
st.sidebar.caption(f"Last retrain: {last_retrain or '—'}")

# -----------------------------------------------------------------------------
# Engine run — basketball (real historical data only; empty until user provides data)
# -----------------------------------------------------------------------------

basketball_historical_df = pd.DataFrame(columns=BASKETBALL_HISTORICAL_COLUMNS)
engine_basketball = BettingEngine(basketball_historical_df, strategy_fn, BANKROLL_FOR_STAKES)
results_basketball = engine_basketball.run()

# Value plays: live NBA + NCAAB from The Odds API; cached 15 min to stay under 500 requests/month
kelly_frac_val = kelly_frac
ODDS_CACHE_TTL_SECONDS = 900  # 15 minutes


def _df_mb(df: pd.DataFrame) -> float:
    """Return approximate memory size of DataFrame in MB (for pipeline memory logging)."""
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return 0.0
    try:
        return float(df.memory_usage(deep=True).sum() / (1024 * 1024))
    except Exception:
        return 0.0


def _log_df_mb(name: str, df: pd.DataFrame) -> None:
    """Print DataFrame memory in MB to help find largest allocations."""
    mb = _df_mb(df)
    print(f"[pipeline memory] {name}: {mb:.2f} MB")


@st.cache_data(ttl=ODDS_CACHE_TTL_SECONDS)
def _fetch_live_odds_cached(commence_date_iso: str, refresh_key: int = 0, api_key_available: bool = True) -> tuple[pd.DataFrame, dict]:
    """Fetch live odds from The Odds API (or ESPN fallback when credits < 10). Cached 15 min.
    Returns (live_odds_df, meta). meta may contain used_espn_fallback, espn_games, espn_odds_rows."""
    empty_df = pd.DataFrame()
    empty_meta: dict = {}
    # #region agent log
    try:
        import json, time
        p = Path(__file__).resolve().parent / ".cursor" / "debug-874db2.log"
        with open(p, "a") as f:
            f.write(json.dumps({"sessionId":"874db2","hypothesisId":"H2","location":"app.py:_fetch_live_odds_cached","message":"cache_fn_entered","data":{"api_key_available":api_key_available},"timestamp":int(time.time()*1000)}) + "\n")
    except Exception:
        pass
    # #endregion
    if not api_key_available:
        return (empty_df, empty_meta)
    api_key = _get_odds_api_key()
    # #region agent log
    try:
        import json, time
        p = Path(__file__).resolve().parent / ".cursor" / "debug-874db2.log"
        with open(p, "a") as f:
            f.write(json.dumps({"sessionId":"874db2","hypothesisId":"H2","location":"app.py:_fetch_live_odds_cached","message":"after_get_key","data":{"key_len_inside":len((api_key or "").strip())},"timestamp":int(time.time()*1000)}) + "\n")
    except Exception:
        pass
    # #endregion
    if not (api_key or "").strip():
        return (empty_df, empty_meta)
    df = get_live_odds(
        api_key=api_key.strip(),
        sport_keys=LIVE_ODDS_SPORT_KEYS,
        commence_on_date=None,
    )
    meta: dict = {}
    # If Odds API returned empty and we're low on credits, fall back to ESPN free odds
    quota = get_quota_status()
    remaining = quota.get("requests_remaining")
    if df.empty and remaining is not None:
        try:
            r = int(remaining)
        except (TypeError, ValueError):
            r = 999
        if r < 10:
            df, n_games, n_odds_rows = get_espn_live_odds_with_stats(
                sport_keys=LIVE_ODDS_SPORT_KEYS,
                commence_on_date=None,
            )
            meta = {"used_espn_fallback": True, "espn_games": n_games, "espn_odds_rows": n_odds_rows}
            print(f"ESPN odds fallback: {n_games} games, {n_odds_rows} odds rows.")
    # Memory: keep only today's games with commence_time within next 24 hours to limit DataFrame size
    if not df.empty and "commence_time" in df.columns:
        try:
            now_utc = datetime.now(timezone.utc)
            cutoff_end = now_utc + timedelta(hours=24)
            def _in_window(ct):
                if pd.isna(ct) or ct is None:
                    return False
                try:
                    dt = datetime.fromisoformat(str(ct).replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return now_utc <= dt <= cutoff_end
                except (ValueError, TypeError):
                    return False
            mask = df["commence_time"].apply(_in_window)
            df = df.loc[mask].copy()
        except Exception:
            pass
    # #region agent log
    try:
        import json, time
        p = Path(__file__).resolve().parent / ".cursor" / "debug-874db2.log"
        with open(p, "a") as f:
            f.write(json.dumps({"sessionId":"874db2","hypothesisId":"H2","location":"app.py:_fetch_live_odds_cached","message":"after_get_live_odds","data":{"rows":len(df)},"timestamp":int(time.time()*1000)}) + "\n")
    except Exception:
        pass
    # #endregion
    return (df, meta)


@st.cache_data(ttl=1800)
def _load_feature_matrix_cached(league=None) -> pd.DataFrame:
    """Cached load of games_with_team_stats + situational for inference (30 min TTL)."""
    return load_feature_matrix_for_inference(league=league)


@st.cache_data(ttl=1800)
def _load_play_history_cached(
    league: Optional[str] = None,
    from_date_iso: Optional[str] = None,
    to_date_iso: Optional[str] = None,
) -> pd.DataFrame:
    """Cached load of play_history from SQLite (30 min TTL). Pass date as ISO string or None."""
    from_date = date.fromisoformat(from_date_iso) if from_date_iso else None
    to_date = date.fromisoformat(to_date_iso) if to_date_iso else None
    return load_play_history(league=league, from_date=from_date, to_date=to_date)


@st.cache_data(ttl=ODDS_CACHE_TTL_SECONDS)
def _get_b2b_teams_cached(as_of_date_iso: str, refresh_key: int = 0) -> frozenset:
    """Teams that played last night (B2B). Cached 15 min with same key as odds."""
    api_key = _get_odds_api_key()
    if not (api_key or "").strip():
        return frozenset()
    return frozenset(get_nba_teams_back_to_back(api_key.strip(), date.fromisoformat(as_of_date_iso)))


# Value plays pipeline — SAME function as scripts/run_value_plays_debug.py (_live_odds_to_value_plays).
# Call chain: (1) Gate (odds_api_key) → (2) _fetch_live_odds_cached → (3) b2b + feature_matrix →
# (4) _aggregate_odds_best_line_avg_implied → (5) _live_odds_to_value_plays → (6) injury/march steps.
# Breaks: (A) odds_api_key empty → else branch, empty df. (B) Any exception in try → except sets empty df + st.session_state["value_plays_pipeline_error"] (shown in UI).
if "odds_refresh_key" not in st.session_state:
    st.session_state["odds_refresh_key"] = 0
if "value_plays_pipeline_error" not in st.session_state:
    st.session_state["value_plays_pipeline_error"] = None

b2b_teams: set[str] = set()
# #region agent log
def _vp_log(msg: str, data: dict):
    try:
        import json, time
        p = Path(__file__).resolve().parent / ".cursor" / "debug-874db2.log"
        with open(p, "a") as f:
            f.write(json.dumps({"sessionId":"874db2","hypothesisId":"H1","location":"app.py:value_plays","message":msg,"data":data,"timestamp":int(time.time()*1000)}) + "\n")
    except Exception:
        pass
# #endregion

# Value plays and POTD: always load from cache. NCAAB written by predict_games.py; run_pipeline_to_cache.py preserves it (no duplicate NCAAB prediction). No pipeline in-app.
def _load_value_plays_cache() -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, int, dict, Optional[str]]:
    """Load from data/cache/value_plays_cache.json. Returns (value_plays_df, totals_plays_df, potd_picks, live_odds_df, value_plays_flagged_count, odds_source_meta, error_message or None)."""
    # #region agent log
    import time
    _log_path = Path(__file__).resolve().parent / ".cursor" / "debug-874db2.log"
    def _dbg(msg: str, data: dict, hypothesis_id: str = "H0"):
        try:
            with open(_log_path, "a") as _f:
                _f.write(json.dumps({"sessionId": "874db2", "hypothesisId": hypothesis_id, "location": "app.py:_load_value_plays_cache", "message": msg, "data": data, "timestamp": int(time.time() * 1000)}) + "\n")
        except Exception:
            pass
    # #endregion
    empty_df = pd.DataFrame(columns=["League", "Event", "Selection", "Market", "Odds", "Value (%)", "Recommended Stake", "Injury Alert", "Start Time"])
    empty_live = pd.DataFrame(columns=["sport_key", "league", "event_id", "commence_time", "home_team", "away_team", "event_name", "market_type", "selection", "point", "odds"])
    cache_path = Path(__file__).resolve().parent / "data" / "cache" / "value_plays_cache.json"
    if not cache_path.exists():
        _dbg("cache file missing", {"cache_path": str(cache_path), "exists": False}, "H1")
        return empty_df, empty_df.copy(), {"NCAAB Pick 1": None, "NCAAB Pick 2": None}, empty_live, 0, {}, None
    _dbg("cache file exists", {"cache_path": str(cache_path), "exists": True}, "H1")
    try:
        with open(cache_path) as f:
            data = json.load(f)
    except Exception as e:
        _dbg("cache read exception", {"error": str(e), "type": type(e).__name__}, "H4")
        return empty_df, empty_df.copy(), {"NCAAB Pick 1": None, "NCAAB Pick 2": None}, empty_live, 0, {}, None
    value_plays_list = data.get("value_plays") or []
    value_plays_df = pd.DataFrame(value_plays_list) if value_plays_list else empty_df
    totals_plays_list = data.get("totals_plays") or []
    totals_plays_df = pd.DataFrame(totals_plays_list) if totals_plays_list else pd.DataFrame()
    _dbg("cache parsed", {"top_level_keys": list(data.keys()), "value_plays_len": len(value_plays_list), "totals_plays_len": len(totals_plays_list), "df_empty": value_plays_df.empty, "cache_error": data.get("error")}, "H2")
    potd_picks = data.get("potd_picks") or {"NCAAB Pick 1": None, "NCAAB Pick 2": None}
    value_plays_flagged_count = int(data.get("value_plays_flagged_count", 0))
    odds_source_meta = data.get("odds_source_meta") or {}
    cache_error = data.get("error")
    return value_plays_df, totals_plays_df, potd_picks, empty_live, value_plays_flagged_count, odds_source_meta, cache_error


def _filter_value_plays_not_started(df: pd.DataFrame, debug: bool = False) -> tuple[pd.DataFrame, int]:
    """Filter out plays where start_time (or commence_time) is in the past (UTC). Returns (filtered_df, n_already_started)."""
    _log = lambda msg: print(msg, file=sys.stderr)
    if df.empty:
        if debug:
            _log("[_filter_value_plays_not_started] df is empty")
        return df, 0
    # Pipeline writes commence_time; predict_games writes start_time. Use either.
    has_start = "start_time" in df.columns
    has_commence = "commence_time" in df.columns
    now_utc = datetime.now(timezone.utc)
    if debug:
        _log(f"[_filter_value_plays_not_started] current UTC time: {now_utc.isoformat()}")
        _log(f"[_filter_value_plays_not_started] columns: start_time={has_start}, commence_time={has_commence}")
    if not has_start and not has_commence:
        if debug:
            _log("[_filter_value_plays_not_started] No start_time or commence_time column — skipping filter")
        return df, 0
    kept_indices = []
    n_started = 0
    for idx, row in df.iterrows():
        st_val = row.get("start_time") if has_start else None
        if st_val is None or (isinstance(st_val, float) and pd.isna(st_val)) or not str(st_val).strip():
            st_val = row.get("commence_time") if has_commence else None
        event = row.get("Event", "")
        if st_val is None or (isinstance(st_val, float) and pd.isna(st_val)) or not str(st_val).strip():
            if debug:
                _log(f"  KEEP (no time) idx={idx} Event={event!r} start_time={row.get('start_time')!r} commence_time={row.get('commence_time')!r}")
            kept_indices.append(idx)
            continue
        # Parse ISO string: normalize Z to +00:00 so fromisoformat gets UTC
        raw = str(st_val).strip()
        s = raw.replace("Z", "+00:00")
        parsed_ok = False
        dt = None
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            parsed_ok = True
        except (ValueError, TypeError) as e:
            if debug:
                _log(f"  KEEP (parse error) idx={idx} Event={event!r} start_time={raw!r} error={e}")
            kept_indices.append(idx)
            continue
        if not parsed_ok or dt is None:
            kept_indices.append(idx)
            continue
        if dt <= now_utc:
            n_started += 1
            if debug:
                _log(f"  FILTER (started) idx={idx} Event={event!r} start_time={raw!r} parsed={dt.isoformat()} (<= now_utc)")
            continue
        if debug:
            _log(f"  KEEP (future) idx={idx} Event={event!r} start_time={raw!r} parsed={dt.isoformat()}")
        kept_indices.append(idx)
    filtered = df.loc[kept_indices].copy() if kept_indices else pd.DataFrame()
    if debug:
        _log(f"[_filter_value_plays_not_started] result: n_started={n_started} kept={len(kept_indices)} total={len(df)}")
    return filtered, n_started
value_plays_df, totals_plays_df, potd_picks, live_odds_df, value_plays_flagged_count, _odds_meta, _cache_err = _load_value_plays_cache()
_filter_debug = os.environ.get("DEBUG_VALUE_PLAYS_FILTER", "").strip() == "1"
value_plays_df, _n_value_plays_already_started = _filter_value_plays_not_started(value_plays_df, debug=_filter_debug)
st.session_state["odds_source_meta"] = _odds_meta
st.session_state["value_plays_pipeline_error"] = (_cache_err, "") if _cache_err else None

# Pathlib: project root for all data paths (works regardless of where Streamlit is launched)
_APP_ROOT = Path(__file__).resolve().parent
HISTORICAL_BETTING_CSV_PATH = _APP_ROOT / "data" / "historical_betting_performance.csv"
# Match prediction script: use 2026 for "today" when filtering CSV
HISTORICAL_PICKS_YEAR = 2026
MLB_VALUE_PLAYS_JSON_PATH = _APP_ROOT / "data" / "cache" / "mlb_value_plays.json"
MLB_PARK_FACTORS_JSON_PATH = _APP_ROOT / "data" / "mlb" / "mlb_park_factors.json"
# Odds/API home name -> key in mlb_park_factors.json (when it differs).
MLB_PARK_HOME_NAME_ALIASES: dict[str, str] = {"Oakland Athletics": "Athletics"}


def _load_mlb_park_runs_lookup() -> dict[str, int]:
    """team_name -> raw runs factor (100 = league average). Empty if file missing."""
    if not MLB_PARK_FACTORS_JSON_PATH.exists():
        return {}
    try:
        with open(MLB_PARK_FACTORS_JSON_PATH, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, int] = {}
    for k, v in raw.items():
        if k.startswith("_") or not isinstance(v, dict):
            continue
        try:
            out[k] = int(v.get("runs", 100))
        except (TypeError, ValueError):
            continue
    return out


def _mlb_park_runs_for_home_team(home_team: str, lookup: dict[str, int]) -> Optional[int]:
    h = (home_team or "").strip()
    if not h:
        return None
    if h in lookup:
        return lookup[h]
    alt = MLB_PARK_HOME_NAME_ALIASES.get(h)
    if alt and alt in lookup:
        return lookup[alt]
    return None


def _mlb_park_emoji_for_runs(runs: Optional[int]) -> str:
    if runs is None:
        return ""
    if runs >= 108:
        return "🔴 "
    if runs <= 94:
        return "🔵 "
    return ""


def _mlb_home_display_with_park(home_team: str, lookup: dict[str, int]) -> str:
    r = _mlb_park_runs_for_home_team(home_team, lookup)
    em = _mlb_park_emoji_for_runs(r)
    return f"{em}{home_team.strip()}"


def _mlb_park_text_label_for_runs(runs: Optional[int]) -> str:
    """Overview POTD line: hitter / pitcher / neutral park copy."""
    if runs is None:
        return "Neutral Park"
    if runs >= 108:
        return "🔴 Hitter's Park"
    if runs <= 94:
        return "🔵 Pitcher's Park"
    return "Neutral Park"


MLB_RECENT_FORM_CSV_PATH = _APP_ROOT / "data" / "mlb" / "recent_form.csv"


def _load_mlb_recent_form_by_team_name() -> dict[str, dict]:
    """recent_form.csv rows keyed by team_name (and Oakland → Athletics alias)."""
    if not MLB_RECENT_FORM_CSV_PATH.exists():
        return {}
    try:
        df = pd.read_csv(MLB_RECENT_FORM_CSV_PATH)
    except (OSError, ValueError):
        return {}
    if df.empty or "team_name" not in df.columns:
        return {}
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        tn = str(row.get("team_name", "")).strip()
        if tn:
            out[tn] = row.to_dict()
    if "Athletics" in out:
        out["Oakland Athletics"] = out["Athletics"]
    return out


def _mlb_recent_form_suffix_next_to_team(team_name: str, form_by_name: dict[str, dict]) -> str:
    """Append ' 🔥 Hot (5-2)' / ' ❄️ Cold (2-5)' when win% is extreme; else ''."""
    if not form_by_name or not team_name:
        return ""
    key = team_name.strip()
    rec = form_by_name.get(key)
    if rec is None and key == "Oakland Athletics":
        rec = form_by_name.get("Athletics")
    if not rec:
        return ""
    try:
        rwp = float(rec.get("recent_win_pct", 0.5))
        n = int(rec.get("games_counted", 0) or 0)
    except (TypeError, ValueError):
        return ""
    if n <= 0 or rwp != rwp:
        return ""
    wins = int(round(rwp * n))
    wins = max(0, min(n, wins))
    losses = n - wins
    if rwp > 0.650:
        return f" 🔥 Hot ({wins}-{losses})"
    if rwp < 0.350:
        return f" ❄️ Cold ({wins}-{losses})"
    return ""


def _mlb_confidence_from_edge_pct(edge_pct) -> str:
    if edge_pct is None or (isinstance(edge_pct, float) and pd.isna(edge_pct)):
        return "—"
    try:
        v = float(edge_pct)
    except (TypeError, ValueError):
        return "—"
    if v >= 5.0:
        return "Strong"
    if v >= 3.0:
        return "Lean"
    return "—"


def _mlb_edge_tier_row_style(row: pd.Series) -> list[str]:
    """Green ≥5% edge, amber 3–5%, else no fill."""
    n = len(row)
    try:
        ev = row.get("Edge %")
        if ev is None or pd.isna(ev):
            return [""] * n
        v = float(ev)
        if v >= 5.0:
            return ["background-color: rgba(46, 125, 50, 0.42); color: #e8f5e9"] * n
        if v >= 3.0:
            return ["background-color: rgba(255, 193, 7, 0.25); color: #fff8e1"] * n
    except (TypeError, ValueError):
        pass
    return [""] * n


def _render_mlb_top_play_card_html(
    play_row: Optional[pd.Series],
    park_lookup: dict[str, int],
) -> str:
    """POTD-style card for highest-edge moneyline play; grey empty state when None."""
    if play_row is None or (isinstance(play_row, pd.Series) and play_row.empty):
        html = """
        <div class="potd-card potd-card--grey">
            <div class="potd-league">MLB Top Play</div>
            <div class="potd-empty-msg">No moneyline play today</div>
        </div>
        """
        return _html_one_line_per_block(html)
    r = play_row
    away = str(r.get("away_team", "") or "").strip()
    home = str(r.get("home_team", "") or "").strip()
    home_disp = _mlb_home_display_with_park(home, park_lookup)
    matchup_html = f'{_html_escape(away)} <span class="potd-vs">@</span> {_html_escape(home_disp)}'
    commence = str(r.get("commence_time", "") or "").strip()
    tipoff = format_start_time(commence) if commence else "—"
    pick = str(r.get("selection", "") or "").strip() or "—"
    try:
        odds_am = float(r.get("odds_american", 0))
    except (TypeError, ValueError):
        odds_am = -110.0
    odds_str = format_american(odds_am)
    try:
        edge_dec = float(r.get("edge", 0))
    except (TypeError, ValueError):
        edge_dec = 0.0
    edge_pct = edge_dec * 100.0
    try:
        mp = float(r.get("model_prob", 0.5))
    except (TypeError, ValueError):
        mp = 0.5
    ap = str(r.get("away_pitcher", "") or "").strip() or "TBD"
    hp = str(r.get("home_pitcher", "") or "").strip() or "TBD"
    expl = (
        f"Model win probability {mp * 100.0:.1f}% vs. implied odds — "
        f"about {edge_pct:.1f}% expected value."
    )
    html = f"""
    <div class="potd-card potd-card--blue">
        <div class="potd-league">MLB Top Play</div>
        <div class="potd-tipoff">First pitch: {_html_escape(tipoff)}</div>
        <div class="potd-matchup">{matchup_html}</div>
        <div class="potd-badge-row"><div class="potd-badge">{_html_escape(pick)}</div><span class="potd-odds-inline">{_html_escape(odds_str)}</span></div>
        <div class="potd-edge">{edge_pct:.1f}% Edge</div>
        <div class="potd-confidence">Model probability: {mp * 100.0:.1f}%</div>
        <div class="potd-current-line">SP: {_html_escape(ap)} @ {_html_escape(hp)}</div>
        <div class="potd-reason">{_html_escape(expl)}</div>
    </div>
    """
    return _html_one_line_per_block(html)


def _load_mlb_value_plays_for_today() -> tuple[pd.DataFrame, list[dict]]:
    """Load MLB value plays from cache JSON; rows for today's date (card_date or per-play card_date)."""
    today = date.today().isoformat()
    if not MLB_VALUE_PLAYS_JSON_PATH.exists():
        return pd.DataFrame(), []
    try:
        with open(MLB_VALUE_PLAYS_JSON_PATH, encoding="utf-8") as f:
            blob = json.load(f)
    except (json.JSONDecodeError, OSError):
        return pd.DataFrame(), []
    if isinstance(blob, list):
        raw_plays = blob
        top_date = ""
    else:
        raw_plays = blob.get("plays") or []
        top_date = str(blob.get("card_date", "")).strip()
    filtered: list[dict] = []
    for p in raw_plays:
        if not isinstance(p, dict):
            continue
        pd_ = str(p.get("card_date", "")).strip()
        if pd_ == today or (not pd_ and top_date == today):
            filtered.append(p)
    if not filtered:
        return pd.DataFrame(), []
    return pd.DataFrame(filtered), filtered


def _ncaab_season_includes_date(d: date) -> bool:
    """Regular season + tournament: early November through the first week of April (inclusive through Apr 7)."""
    m, day = d.month, d.day
    if m in (11, 12, 1, 2, 3):
        return True
    if m == 4 and day <= 7:
        return True
    return False


def _mlb_season_includes_date(d: date) -> bool:
    """MLB Play of the Day window: April through October (regular season / playoffs)."""
    return d.month in (4, 5, 6, 7, 8, 9, 10)


def _mlb_row_to_potd_dict(row: pd.Series) -> dict:
    """Shape one MLB value-play row like NCAAB POTD dicts for _render_potd_card_html."""
    away = str(row.get("away_team", "")).strip()
    home = str(row.get("home_team", "")).strip()
    event = f"{away} @ {home}" if away and home else ""
    market_raw = str(row.get("market", "")).strip().lower()
    if market_raw in ("moneyline", "h2h"):
        market_key = "moneyline"
    elif market_raw in ("spreads", "spread"):
        market_key = "spreads"
    elif market_raw in ("totals", "total"):
        market_key = "totals"
    else:
        market_key = market_raw if market_raw else "moneyline"
    edge_dec = row.get("edge")
    edge_pct = 0.0
    if edge_dec is not None and pd.notna(edge_dec):
        try:
            edge_pct = float(edge_dec) * 100.0
        except (TypeError, ValueError):
            edge_pct = 0.0
    else:
        ep = row.get("edge_pct")
        if ep is not None and pd.notna(ep):
            try:
                fv = float(ep)
                edge_pct = fv if abs(fv) >= 1.0 else fv * 100.0
            except (TypeError, ValueError):
                pass
    point_val = None
    if market_key == "totals":
        tl = row.get("total_line")
        if tl is not None and pd.notna(tl):
            try:
                point_val = float(tl)
            except (TypeError, ValueError):
                point_val = None
    elif market_key == "spreads":
        for k in ("spread", "point", "spread_line"):
            v = row.get(k)
            if v is not None and pd.notna(v):
                try:
                    point_val = float(v)
                    break
                except (TypeError, ValueError):
                    pass
    sel = str(row.get("selection", "")).strip()
    mp = row.get("model_prob")
    try:
        mp_f = float(mp) if mp is not None and pd.notna(mp) else None
    except (TypeError, ValueError):
        mp_f = None
    if mp_f is not None:
        reason = f"Model win probability {mp_f:.1%} vs. implied odds — about {max(0.0, edge_pct):.1f}% expected value."
    else:
        reason = f"Model edge about {max(0.0, edge_pct):.1f}% vs. the posted line."
    ap_away = str(row.get("away_pitcher", "")).strip()
    ap_home = str(row.get("home_pitcher", "")).strip()
    mlb_sp_line = ""
    if ap_away or ap_home:
        mlb_sp_line = f"SP: {ap_away or 'TBD'} @ {ap_home or 'TBD'}"
    return {
        "Event": event,
        "home_team": home,
        "away_team": away,
        "Selection": sel,
        "Market": market_key,
        "Odds": row.get("odds_american"),
        "Value (%)": max(0.0, min(99.0, edge_pct)),
        "point": point_val,
        "commence_time": str(row.get("commence_time", "") or ""),
        "Start Time": "",
        "reason": reason,
        "high_variance": False,
        "Recommended Stake": 0,
        "confidence_tier": "Medium",
        "Injury Alert": "—",
        "model_prob": mp_f if mp_f is not None else 0.5,
        "mlb_sp_line": mlb_sp_line,
    }


def _mlb_overview_sort_key_series(df: pd.DataFrame) -> pd.Series:
    if "edge" in df.columns:
        return pd.to_numeric(df["edge"], errors="coerce").fillna(0.0).abs()
    ep = pd.to_numeric(df["edge_pct"], errors="coerce").fillna(0.0)
    arr = np.where(np.abs(ep) >= 1.0, np.abs(ep), np.abs(ep) * 100.0)
    return pd.Series(arr, index=df.index)


def _mlb_dedupe_by_event_sort(df: pd.DataFrame, sort_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.sort_values(sort_col, ascending=False)
    if "event_id" in d.columns:
        return d.groupby(d["event_id"].astype(str), sort=False).head(1).reset_index(drop=True).sort_values(
            sort_col, ascending=False
        )
    subset = [c for c in ("home_team", "away_team", "commence_time") if c in d.columns]
    return d.drop_duplicates(subset=subset if subset else None, keep="first").reset_index(drop=True).sort_values(
        sort_col, ascending=False
    )


def _mlb_overview_potd_picks(
    mlb_df: pd.DataFrame,
) -> tuple[Optional[dict], Optional[dict], Optional[dict]]:
    """
    Overview MLB cards: Pick 1 & 2 = top two moneyline edges (one row per game);
    Total pick = best O/U edge if any.
    """
    if mlb_df is None or mlb_df.empty:
        return None, None, None
    if "edge" not in mlb_df.columns and "edge_pct" not in mlb_df.columns:
        return None, None, None
    df = mlb_df.copy()
    df["_potd_sort_edge"] = _mlb_overview_sort_key_series(df)

    if "market" in df.columns:
        mkt = df["market"].astype(str).str.strip().str.lower()
        ml_df = df[mkt.isin(("moneyline", "h2h"))].copy()
        tot_df = df[mkt.isin(("total", "totals"))].copy()
    else:
        ml_df = df.copy()
        tot_df = pd.DataFrame()

    ml_dedup = _mlb_dedupe_by_event_sort(ml_df, "_potd_sort_edge")
    tot_dedup = _mlb_dedupe_by_event_sort(tot_df, "_potd_sort_edge")

    d1 = _mlb_row_to_potd_dict(ml_dedup.iloc[0]) if len(ml_dedup) >= 1 else None
    d2 = _mlb_row_to_potd_dict(ml_dedup.iloc[1]) if len(ml_dedup) >= 2 else None
    d3 = _mlb_row_to_potd_dict(tot_dedup.iloc[0]) if len(tot_dedup) >= 1 else None
    return d1, d2, d3


def _mlb_dataframe_for_play_history(mlb_df: pd.DataFrame) -> pd.DataFrame:
    """Build rows for archive_value_plays: League='MLB', Market in h2h/spreads/totals for bet_type mapping."""
    if mlb_df is None or mlb_df.empty:
        return pd.DataFrame()
    out_rows: list[dict] = []
    for _, r in mlb_df.iterrows():
        away = str(r.get("away_team", "")).strip()
        home = str(r.get("home_team", "")).strip()
        if not away or not home:
            continue
        sel = str(r.get("selection", "")).strip()
        market_raw = str(r.get("market", "")).strip().lower()
        point_val = None
        if market_raw in ("moneyline", "h2h"):
            market = "h2h"
        elif market_raw in ("spreads", "spread"):
            market = "spreads"
            for k in ("spread", "point", "spread_line"):
                v = r.get(k)
                if v is not None and pd.notna(v):
                    try:
                        point_val = float(v)
                        break
                    except (TypeError, ValueError):
                        pass
        elif market_raw in ("totals", "total"):
            market = "totals"
            tl = r.get("total_line")
            if tl is not None and pd.notna(tl):
                try:
                    point_val = float(tl)
                except (TypeError, ValueError):
                    point_val = None
        else:
            market = "h2h"
        edge_dec = r.get("edge")
        if edge_dec is not None and pd.notna(edge_dec):
            try:
                val_pct = float(edge_dec) * 100.0
            except (TypeError, ValueError):
                val_pct = 0.0
        else:
            val_pct = 0.0
            ep = r.get("edge_pct")
            if ep is not None and pd.notna(ep):
                try:
                    fv = float(ep)
                    val_pct = fv if abs(fv) >= 1.0 else fv * 100.0
                except (TypeError, ValueError):
                    pass
        try:
            odds = float(r.get("odds_american", -110) or -110)
        except (TypeError, ValueError):
            odds = -110.0
        try:
            mp = float(r.get("model_prob", 0.5) or 0.5)
        except (TypeError, ValueError):
            mp = 0.5
        try:
            frac = kelly_fraction(odds, mp, fraction=kelly_frac)
            rec_stake = round(BANKROLL_FOR_STAKES * frac, 2)
        except (TypeError, ValueError):
            rec_stake = None
        parts: list[str] = []
        ap_a = str(r.get("away_pitcher", "")).strip()
        ap_h = str(r.get("home_pitcher", "")).strip()
        if ap_a or ap_h:
            parts.append(f"SP: {ap_a or 'TBD'} @ {ap_h or 'TBD'}")
        eid = r.get("event_id")
        if eid is not None and str(eid).strip():
            parts.append(f"event_id={eid}")
        reasoning = " · ".join(parts) if parts else None
        out_rows.append(
            {
                "League": "MLB",
                "Event": f"{away} @ {home}",
                "Selection": sel or "—",
                "Market": market,
                "Odds": odds,
                "Value (%)": max(0.0, min(99.0, val_pct)),
                "model_prob": mp,
                "home_team": home,
                "away_team": away,
                "point": point_val,
                "Recommended Stake": rec_stake,
                "confidence_tier": "Medium",
                "reasoning_summary": reasoning,
            }
        )
    return pd.DataFrame(out_rows)


def _normalize_date_to_iso(raw: str) -> Optional[str]:
    """Parse date from YYYY-MM-DD or MM-DD-YYYY into YYYY-MM-DD; return None if unparseable."""
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "nat", "none"):
        return None
    for fmt in ("%Y-%m-%d", "%m-%d-%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            d = datetime.strptime(s, fmt)
            return d.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    try:
        parsed = pd.to_datetime(s, errors="coerce")
        if pd.notna(parsed):
            return parsed.strftime("%Y-%m-%d")
    except Exception:
        pass
    return None


def _load_historical_betting_performance_all() -> pd.DataFrame:
    """Load full data/historical_betting_performance.csv (all dates). Returns empty DataFrame if missing or no Date column."""
    if not HISTORICAL_BETTING_CSV_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(HISTORICAL_BETTING_CSV_PATH)
    except Exception:
        return pd.DataFrame()
    if df.empty or "Date" not in df.columns:
        return df
    return df.copy()


def _remove_play_from_historical_csv(date_generated: str, home_team: str, away_team: str) -> int:
    """Remove matching row(s) from historical_betting_performance.csv. Returns number of rows removed."""
    if not HISTORICAL_BETTING_CSV_PATH.exists():
        return 0
    try:
        df = pd.read_csv(HISTORICAL_BETTING_CSV_PATH)
    except Exception:
        return 0
    if df.empty or "Date" not in df.columns or "Home" not in df.columns or "Away" not in df.columns:
        return 0
    date_norm = _normalize_date_to_iso(date_generated) or str(date_generated).strip()
    home = str(home_team or "").strip()
    away = str(away_team or "").strip()
    mask = (
        (df["Date"].astype(str).str.strip() == date_norm)
        & (df["Home"].astype(str).str.strip() == home)
        & (df["Away"].astype(str).str.strip() == away)
    )
    n = int(mask.sum())
    if n > 0:
        df = df[~mask]
        df.to_csv(HISTORICAL_BETTING_CSV_PATH, index=False)
    return n


def _sync_result_to_historical_csv(date_generated: str, home_team: str, away_team: str, result: str) -> bool:
    """Write ATS_Result back to historical_betting_performance.csv so CSV and SQLite stay in sync."""
    if not HISTORICAL_BETTING_CSV_PATH.exists():
        return False
    result_map = {"W": "Win", "L": "Loss", "P": "Push"}
    csv_result = result_map.get(result)
    if csv_result is None:
        return False
    try:
        df = pd.read_csv(HISTORICAL_BETTING_CSV_PATH)
    except Exception:
        return False
    if df.empty or "Date" not in df.columns:
        return False
    if "ATS_Result" not in df.columns:
        df["ATS_Result"] = ""
    date_norm = _normalize_date_to_iso(date_generated) or str(date_generated).strip()
    home = str(home_team or "").strip()
    away = str(away_team or "").strip()
    mask = (
        (df["Date"].astype(str).str.strip() == date_norm)
        & (df["Home"].astype(str).str.strip() == home)
        & (df["Away"].astype(str).str.strip() == away)
    )
    if mask.any():
        df.loc[mask, "ATS_Result"] = csv_result
        df.to_csv(HISTORICAL_BETTING_CSV_PATH, index=False)
        return True
    return False


# ----- Game Lookup tab helpers -----
def _load_latest_odds_slate() -> pd.DataFrame:
    """Load the most recent CSV in data/odds/ (by mtime). Returns DataFrame with home_team, away_team, market_spread for Game Lookup."""
    odds_dir = _APP_ROOT / "data" / "odds"
    if not odds_dir.exists():
        return pd.DataFrame()
    csvs = list(odds_dir.glob("*.csv"))
    if not csvs:
        return pd.DataFrame()
    latest = max(csvs, key=lambda p: p.stat().st_mtime)
    try:
        df = pd.read_csv(latest)
    except Exception:
        return pd.DataFrame()
    # Normalize columns: Home_Team, Away_Team, Spread -> home_team, away_team, market_spread
    if "Home_Team" not in df.columns or "Away_Team" not in df.columns:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["home_team"] = df["Home_Team"].astype(str).str.strip()
    out["away_team"] = df["Away_Team"].astype(str).str.strip()
    out["market_spread"] = pd.to_numeric(df.get("Spread"), errors="coerce") if "Spread" in df.columns else None
    return out


def _load_team_stats_2026() -> pd.DataFrame:
    """Load data/ncaab/team_stats_2026.csv. Returns empty DataFrame if missing."""
    path = _APP_ROOT / "data" / "ncaab" / "team_stats_2026.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _team_match_query(team_name: str, query: str) -> bool:
    """True if query is contained in team_name (case-insensitive) or exact match."""
    t = (team_name or "").strip().lower()
    q = (query or "").strip().lower()
    if not q:
        return False
    return q in t or t == q


# Game Lookup: fuzzy match threshold (0-100). Matches Event, home_team, away_team.
GAME_LOOKUP_FUZZY_THRESHOLD = 60


def _value_play_matches_query(row: pd.Series, query: str) -> bool:
    """True if query matches this value-play row via exact/substring or fuzzy (Event, home_team, away_team) >= GAME_LOOKUP_FUZZY_THRESHOLD."""
    q = (query or "").strip()
    if not q:
        return False
    home = str(row.get("home_team", "")).strip()
    away = str(row.get("away_team", "")).strip()
    event = str(row.get("Event", "")).strip()
    if _team_match_query(home, query) or _team_match_query(away, query):
        return True
    try:
        from thefuzz import fuzz
        q_lower = q.lower()
        if fuzz.partial_ratio(q_lower, event.lower()) >= GAME_LOOKUP_FUZZY_THRESHOLD:
            return True
        if fuzz.partial_ratio(q_lower, home.lower()) >= GAME_LOOKUP_FUZZY_THRESHOLD:
            return True
        if fuzz.partial_ratio(q_lower, away.lower()) >= GAME_LOOKUP_FUZZY_THRESHOLD:
            return True
    except ImportError:
        pass
    return False


def _normalize_team_name_for_lookup(name: str) -> str:
    """Normalize so 'Wichita St.' and 'Wichita State' match (and similar St. / State)."""
    s = (name or "").strip().lower()
    s = s.replace(" st.", " state").replace(" st ", " state ")
    return s


def _find_team_in_stats(team_stats_df: pd.DataFrame, query: str) -> Optional[pd.Series]:
    """Return first row from team_stats_2026 where TEAM matches or contains query (case-insensitive). Normalizes St./State so 'Wichita State' matches 'Wichita St.'"""
    if team_stats_df.empty or "TEAM" not in team_stats_df.columns:
        return None
    q = _normalize_team_name_for_lookup(query)
    if not q:
        return None
    for _, row in team_stats_df.iterrows():
        team = _normalize_team_name_for_lookup(str(row.get("TEAM", "")))
        if not team:
            continue
        if q == team or q in team or team in q:
            return row
    return None


def _find_game_for_team(
    team_query: str,
    value_plays_df: pd.DataFrame,
    historical_picks_df: pd.DataFrame,
    today_odds_df: Optional[pd.DataFrame] = None,
) -> Optional[tuple[str, dict]]:
    """
    Find today's (or selected slate) game involving the given team.
    Returns ('cache', row_dict) or ('historical', row_dict) or ('odds', row_dict) or None.
    row_dict has home_team, away_team, market_spread, pred_margin, edge, confidence, pick, etc.
    """
    q = (team_query or "").strip().lower()
    if not q:
        return None
    # 1) Value plays cache: search Event, home_team, away_team (exact/substring or fuzzy >= GAME_LOOKUP_FUZZY_THRESHOLD)
    if not value_plays_df.empty and "home_team" in value_plays_df.columns and "away_team" in value_plays_df.columns:
        for _, row in value_plays_df.iterrows():
            home = str(row.get("home_team", "")).strip()
            away = str(row.get("away_team", "")).strip()
            if _value_play_matches_query(row, team_query):
                point = row.get("point") or row.get("Point")
                try:
                    pt = float(point) if point is not None and pd.notna(point) else None
                except (TypeError, ValueError):
                    pt = None
                # point is from picked team's perspective; market spread (home) = point if home picked else -point
                selection = str(row.get("Selection", "")).strip()
                spread = pt if selection == home else (-pt if pt is not None else None)
                # Edge: from reason string or Value (%) * 3 as proxy
                value_pct = row.get("Value (%)") or 0
                try:
                    edge_pts = float(value_pct) * 3.0 if value_pct is not None else None
                except (TypeError, ValueError):
                    edge_pts = None
                reason = str(row.get("reason") or row.get("reasoning_summary") or "").strip()
                return ("cache", {
                    "home_team": home,
                    "away_team": away,
                    "market_spread": spread,
                    "pred_margin": None,  # cache may not have it
                    "edge_points": edge_pts,
                    "confidence_tier": str(row.get("confidence_tier", "Medium")).strip() or "Medium",
                    "pick_spread": str(row.get("Selection", "")).strip(),
                    "recommended_pick": str(row.get("Selection", "")).strip(),
                    "value_pct": value_pct,
                    "reason": reason or None,
                })
    # 2) Historical CSV for selected slate date
    if not historical_picks_df.empty and "Home" in historical_picks_df.columns and "Away" in historical_picks_df.columns:
        for _, row in historical_picks_df.iterrows():
            home = str(row.get("Home", "")).strip()
            away = str(row.get("Away", "")).strip()
            if _team_match_query(home, team_query) or _team_match_query(away, team_query):
                try:
                    spread = float(row.get("Market_Spread")) if row.get("Market_Spread") is not None and pd.notna(row.get("Market_Spread")) else None
                except (TypeError, ValueError):
                    spread = None
                try:
                    pred = float(row.get("Pred_Margin")) if row.get("Pred_Margin") is not None and pd.notna(row.get("Pred_Margin")) else None
                except (TypeError, ValueError):
                    pred = None
                try:
                    edge_pts = float(row.get("Edge_Points")) if row.get("Edge_Points") is not None and pd.notna(row.get("Edge_Points")) else None
                except (TypeError, ValueError):
                    edge_pts = None
                pick = str(row.get("Pick_Spread", "Home")).strip()
                selection = home if pick == "Home" else away
                pred_margin = row.get("Pred_Margin")
                try:
                    pred_f = float(pred_margin) if pred_margin is not None and pd.notna(pred_margin) else None
                except (TypeError, ValueError):
                    pred_f = None
                if pred_f is not None and spread is not None:
                    reason = _human_readable_potd_reason(home, away, pred_f, spread, pick)
                else:
                    reason = f"Model favors {selection}." + (f" Edge of {edge_pts:+.1f} pts." if edge_pts is not None else "")
                return ("historical", {
                    "home_team": home,
                    "away_team": away,
                    "market_spread": spread,
                    "pred_margin": pred,
                    "edge_points": edge_pts,
                    "confidence_tier": str(row.get("Confidence_Level", "Medium")).strip() or "Medium",
                    "pick_spread": pick,
                    "recommended_pick": selection,
                    "home_barthag": row.get("home_barthag") if "home_barthag" in row.index else None,
                    "away_barthag": row.get("away_barthag") if "away_barthag" in row.index else None,
                    "reason": reason,
                })
    # 3) Today's odds (latest data/odds/*.csv): so any team in the slate can be found
    if today_odds_df is not None and not today_odds_df.empty and "home_team" in today_odds_df.columns and "away_team" in today_odds_df.columns:
        for _, row in today_odds_df.iterrows():
            home = str(row.get("home_team", "")).strip()
            away = str(row.get("away_team", "")).strip()
            if _team_match_query(home, team_query) or _team_match_query(away, team_query):
                try:
                    spread = float(row.get("market_spread")) if row.get("market_spread") is not None and pd.notna(row.get("market_spread")) else None
                except (TypeError, ValueError):
                    spread = None
                return ("odds", {
                    "home_team": home,
                    "away_team": away,
                    "market_spread": spread,
                    "pred_margin": None,
                    "edge_points": None,
                    "confidence_tier": "—",
                    "pick_spread": "—",
                    "recommended_pick": "—",
                    "reason": "Game from today's odds slate; no model pick for this matchup.",
                })
    return None


def _get_head_to_head(home_team: str, away_team: str) -> pd.DataFrame:
    """Return past games between these two teams this season from data/espn.db games_with_team_stats."""
    db_path = _APP_ROOT / "data" / "espn.db"
    if not db_path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        # Table may use home_team_name / away_team_name
        try:
            df = pd.read_sql_query(
                """
                SELECT game_date, home_team_name, away_team_name
                FROM games_with_team_stats
                WHERE league = 'ncaab'
                  AND (
                    (LOWER(TRIM(home_team_name)) = LOWER(?) AND LOWER(TRIM(away_team_name)) = LOWER(?))
                    OR (LOWER(TRIM(home_team_name)) = LOWER(?) AND LOWER(TRIM(away_team_name)) = LOWER(?))
                  )
                ORDER BY game_date DESC
                """,
                conn,
                params=(home_team.strip(), away_team.strip(), away_team.strip(), home_team.strip()),
            )
        except sqlite3.OperationalError:
            # Fallback if column names differ
            df = pd.read_sql_query("SELECT * FROM games_with_team_stats WHERE league = 'ncaab' LIMIT 1", conn)
            if df.empty or "home_team_name" not in df.columns:
                return pd.DataFrame()
            df = pd.read_sql_query(
                """
                SELECT game_date, home_team_name, away_team_name
                FROM games_with_team_stats
                WHERE league = 'ncaab'
                  AND (
                    (LOWER(TRIM(home_team_name)) = LOWER(?) AND LOWER(TRIM(away_team_name)) = LOWER(?))
                    OR (LOWER(TRIM(home_team_name)) = LOWER(?) AND LOWER(TRIM(away_team_name)) = LOWER(?))
                  )
                ORDER BY game_date DESC
                """,
                conn,
                params=(home_team.strip(), away_team.strip(), away_team.strip(), home_team.strip()),
            )
        return df
    finally:
        conn.close()


def _historical_unique_dates(df: pd.DataFrame) -> list[str]:
    """Return unique dates from CSV Date column (YYYY-MM-DD), sorted descending (most recent first)."""
    if df.empty or "Date" not in df.columns:
        return []
    normalized = df["Date"].astype(str).apply(_normalize_date_to_iso)
    unique = sorted(normalized.dropna().unique(), reverse=True)
    return [d for d in unique if d]


def _load_historical_betting_performance() -> pd.DataFrame:
    """Load data/historical_betting_performance.csv. Returns rows for today's date, accepting YYYY-MM-DD or MM-DD-YYYY in the CSV."""
    if not HISTORICAL_BETTING_CSV_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(HISTORICAL_BETTING_CSV_PATH)
    except Exception:
        return pd.DataFrame()
    if df.empty or "Date" not in df.columns:
        return df
    today_iso = f"{HISTORICAL_PICKS_YEAR}-{date.today().strftime('%m-%d')}"
    normalized = df["Date"].astype(str).apply(_normalize_date_to_iso)
    return df[normalized == today_iso].copy()


def _potd_from_historical_csv(df_today: pd.DataFrame) -> dict:
    """Build potd_picks from historical_betting_performance.csv: top 2 different games by abs(Edge_Points) descending."""
    result = {"NCAAB Pick 1": None, "NCAAB Pick 2": None}
    if df_today.empty or "Edge_Points" not in df_today.columns:
        return result
    # Dedupe: one row per game (keep row with max abs(Edge_Points) per Home/Away pair)
    df_today = df_today.copy()
    df_today["_game_key"] = df_today.apply(lambda r: (str(r.get("Home", "")).strip(), str(r.get("Away", "")).strip()), axis=1)
    df_today["_abs_edge"] = df_today["Edge_Points"].abs()
    best_per_game = df_today.loc[df_today.groupby("_game_key")["_abs_edge"].idxmax()].copy()
    best_per_game = best_per_game.sort_values("_abs_edge", ascending=False).reset_index(drop=True)

    if best_per_game.empty:
        return result

    # Pick 1: top by abs(Edge_Points)
    row1 = best_per_game.iloc[0]
    game_key_1 = (str(row1.get("Home", "")).strip(), str(row1.get("Away", "")).strip())
    result["NCAAB Pick 1"] = _historical_row_to_potd(row1, "NCAAB Pick 1")

    # Pick 2: next best *different* game by abs(Edge_Points)
    other = best_per_game[best_per_game["_game_key"].apply(lambda k: k != game_key_1)]
    if not other.empty:
        result["NCAAB Pick 2"] = _historical_row_to_potd(other.iloc[0], "NCAAB Pick 2")
    return result


def _potd_from_value_plays(
    value_df: pd.DataFrame,
    existing: Optional[dict] = None,
) -> dict:
    """Ensure NCAAB Pick 1 / Pick 2 are populated from today's value plays (top 2 by Value %, different games).

    This is used when cache only has one POTD or none; we always want two cards
    when there are at least two NCAAB value plays available.
    """
    picks = dict(existing or {"NCAAB Pick 1": None, "NCAAB Pick 2": None})
    if value_df is None or value_df.empty:
        return picks
    df = value_df.copy()
    # NCAAB only
    df = df[df["League"].astype(str).str.upper() == "NCAAB"]
    if df.empty or "Value (%)" not in df.columns:
        return picks
    # One row per Event, highest Value (%) per game
    df["_abs_edge"] = df["Value (%)"].abs()
    best_per_game = (
        df.sort_values("_abs_edge", ascending=False)
        .groupby("Event")
        .head(1)
        .reset_index(drop=True)
    )
    if best_per_game.empty:
        return picks

    def _row_to_potd(row: pd.Series, label: str) -> dict:
        selection = str(row.get("Selection", "")).strip()
        try:
            edge_pct = float(row.get("Value (%)", 0) or 0.0)
        except (TypeError, ValueError):
            edge_pct = 0.0
        edge_pct = max(0.0, min(15.0, edge_pct))
        point_val = row.get("point")
        if point_val is None or (isinstance(point_val, float) and pd.isna(point_val)):
            point_val = row.get("Point")
        reason = (
            row.get("reason")
            or row.get("reasoning_summary")
            or f"Our model sees {edge_pct:.1f}% expected value on {selection}."
        )
        return {
            "League": label,
            "Event": str(row.get("Event", "")).strip(),
            "Selection": selection,
            "Market": str(row.get("Market", "")).strip(),
            "Odds": row.get("Odds"),
            "Value (%)": edge_pct,
            "point": point_val,
            "Recommended Stake": row.get("Recommended Stake", 0),
            "Start Time": row.get("Start Time", ""),
            "commence_time": row.get("commence_time", ""),
            "Injury Alert": row.get("Injury Alert", "—"),
            "high_variance": bool(row.get("high_variance", False)),
            "home_team": row.get("home_team", "—"),
            "away_team": row.get("away_team", "—"),
            "model_prob": row.get("model_prob", 0.5),
            "confidence_tier": row.get("confidence_tier", "Medium"),
            "reason": reason,
        }

    # Helper: get event string safely from pick dict
    def _event_of(p: Optional[dict]) -> Optional[str]:
        return str(p.get("Event")).strip() if isinstance(p, dict) and p.get("Event") else None

    event_p1_existing = _event_of(picks.get("NCAAB Pick 1"))
    event_p2_existing = _event_of(picks.get("NCAAB Pick 2"))

    # Fill Pick 1 if missing, avoiding any existing Pick 2 event
    if picks.get("NCAAB Pick 1") is None and len(best_per_game) >= 1:
        if event_p2_existing:
            candidates = best_per_game[best_per_game["Event"].astype(str).str.strip() != event_p2_existing]
            if not candidates.empty:
                picks["NCAAB Pick 1"] = _row_to_potd(candidates.iloc[0], "NCAAB Pick 1")
        if picks.get("NCAAB Pick 1") is None:
            picks["NCAAB Pick 1"] = _row_to_potd(best_per_game.iloc[0], "NCAAB Pick 1")

    # Fill Pick 2 if missing, using next-best different game from Pick 1
    if picks.get("NCAAB Pick 2") is None and len(best_per_game) >= 2:
        event_p1 = _event_of(picks.get("NCAAB Pick 1"))
        remaining = (
            best_per_game[best_per_game["Event"].astype(str).str.strip() != event_p1]
            if event_p1
            else best_per_game
        )
        if not remaining.empty:
            picks["NCAAB Pick 2"] = _row_to_potd(remaining.iloc[0], "NCAAB Pick 2")

    # If both picks exist but happen to reference the same game (e.g. cache + backfill collision),
    # force Pick 2 to use the next-best different game when available.
    if (
        isinstance(picks.get("NCAAB Pick 1"), dict)
        and isinstance(picks.get("NCAAB Pick 2"), dict)
        and _event_of(picks["NCAAB Pick 1"]) == _event_of(picks["NCAAB Pick 2"])
        and len(best_per_game) >= 2
    ):
        event_p1 = _event_of(picks["NCAAB Pick 1"])
        remaining = best_per_game[best_per_game["Event"].astype(str).str.strip() != event_p1]
        if not remaining.empty:
            picks["NCAAB Pick 2"] = _row_to_potd(remaining.iloc[0], "NCAAB Pick 2")
    return picks


def _human_readable_potd_reason(
    home: str, away: str, pred_margin: Optional[float], market_spread: Optional[float], pick_side: str
) -> str:
    """Human-readable POTD reason: 'Model projects X to win by ~N pts. Market needs Y — take Z ±Y.' When we pick the underdog, avoid saying the favorite wins by more than the spread."""
    try:
        pred = float(pred_margin) if pred_margin is not None else None
    except (TypeError, ValueError):
        pred = None
    try:
        ms = float(market_spread) if market_spread is not None else None
    except (TypeError, ValueError):
        ms = None
    if pred is None or ms is None:
        return f"Model favors {home if pick_side == 'Home' else away}."
    abs_spread = abs(ms)
    pick_home = pick_side.strip().upper() == "HOME"
    selection = home if pick_home else away
    home_favored = ms < 0
    # Take line: from picked team's perspective
    if pick_home and home_favored:
        take = f"take {selection} -{abs_spread:.1f}"
    elif pick_home and not home_favored:
        take = f"take {selection} +{abs_spread:.1f}"
    elif not pick_home and home_favored:
        take = f"take {selection} +{abs_spread:.1f}"
    else:
        take = f"take {selection} -{abs_spread:.1f}"
    # Picking underdog = we take the +spread side (home +spread when home underdog, away +spread when away underdog)
    picking_underdog = (pick_home and not home_favored) or (not pick_home and home_favored)
    if picking_underdog and pred is not None:
        # If model projects favorite to win by more than the spread, don't say that — it contradicts taking the underdog
        if pred >= 0 and home_favored and pred > abs_spread:
            proj = f"Model projects a closer game than the spread — take {selection} +{abs_spread:.1f}."
            return proj
        if pred < 0 and not home_favored and abs(pred) > abs_spread:
            proj = f"Model projects a closer game than the spread — take {selection} +{abs_spread:.1f}."
            return proj
    # Projection: who does model say wins and by how much?
    if pred >= 0:
        proj = f"Model projects {home} to win by ~{int(round(pred))} pts."
    else:
        proj = f"Model projects {away} to win by ~{int(round(abs(pred)))} pts."
    return f"{proj} Market needs {abs_spread:.1f} — {take}."


def _generic_potd_reason(selection: str, market_spread: Optional[float], pick_side: str) -> str:
    """Fallback POTD reason when pred_margin/market_spread missing."""
    if market_spread is not None and pick_side == "Away":
        spread_from_pick = -market_spread
    else:
        spread_from_pick = market_spread
    spread_str = f"{spread_from_pick:+.1f}" if spread_from_pick is not None else None
    if spread_str:
        return f"Model favors {selection} to cover the {spread_str} spread."
    return f"Model favors {selection}."


def _historical_row_to_potd(row: pd.Series, label: str) -> dict:
    """Convert one historical_betting_performance.csv row to POTD card dict.
    Edge = Edge_Points / 3 (capped at 15%). Builds dynamic reasoning from BARTHAG, edge, and market spread when available.
    Spread (point) must be from the picked team's perspective: Home = Market_Spread, Away = -Market_Spread."""
    home = str(row.get("Home", "")).strip()
    away = str(row.get("Away", "")).strip()
    pick_side = str(row.get("Pick_Spread", "Home")).strip()
    selection = home if pick_side == "Home" else away
    edge_pts = row.get("Edge_Points")
    try:
        edge_pct = abs(float(edge_pts)) / 3.0 if edge_pts is not None and pd.notna(edge_pts) else 0.0
    except (TypeError, ValueError):
        edge_pct = 0.0
    edge_pct = min(15.0, edge_pct)
    try:
        ms = float(row.get("Market_Spread")) if row.get("Market_Spread") is not None and pd.notna(row.get("Market_Spread")) else None
    except (TypeError, ValueError):
        ms = None
    # Spread from picked team's perspective: market spread is home perspective; away pick gets -ms
    point_val = ms if pick_side == "Home" else (-float(ms) if ms is not None else None)
    # Human-readable reasoning from pred_margin and market_spread
    pred_margin = row.get("Pred_Margin")
    try:
        pred_f = float(pred_margin) if pred_margin is not None and pd.notna(pred_margin) else None
    except (TypeError, ValueError):
        pred_f = None
    if pred_f is not None and ms is not None:
        reason = _human_readable_potd_reason(home, away, pred_f, ms, pick_side)
    else:
        reason = _generic_potd_reason(selection, ms, pick_side)
    return {
        "League": label,
        "Event": f"{away} @ {home}",
        "Selection": selection,
        "Market": "Spread",
        "Odds": None,
        "Value (%)": edge_pct,
        "point": point_val,
        "Recommended Stake": 0,
        "Start Time": "Today",
        "commence_time": "",
        "Injury Alert": "—",
        "high_variance": False,
        "home_team": home,
        "away_team": away,
        "model_prob": 0.5,
        "confidence_tier": str(row.get("Confidence_Level", "Medium")).strip() or "Medium",
        "reason": reason,
    }


# Load full historical CSV; sidebar "Select Slate Date" (default = most recent); filter to that date for Overview
_historical_full_df = _load_historical_betting_performance_all()
_unique_slate_dates = _historical_unique_dates(_historical_full_df)
if not _unique_slate_dates:
    _unique_slate_dates = [date.today().strftime("%Y-%m-%d")]
_selected_slate_date = st.sidebar.selectbox(
    "Select Slate Date",
    options=_unique_slate_dates,
    index=0,
    key="select_slate_date",
    help="Slate to show on Overview (POTD and value plays). Default is most recent date in your record.",
)
_normalized_slate = _historical_full_df["Date"].astype(str).apply(_normalize_date_to_iso) if not _historical_full_df.empty else pd.Series(dtype=object)
_historical_picks_df = _historical_full_df[_normalized_slate == _selected_slate_date].copy() if not _historical_full_df.empty else pd.DataFrame()
if potd_picks.get("NCAAB Pick 1") is None and potd_picks.get("NCAAB Pick 2") is None and not _historical_picks_df.empty:
    potd_picks = _potd_from_historical_csv(_historical_picks_df)

# Always backfill missing NCAAB Pick 1 / 2 from today's value plays (top 2 by edge).
potd_picks = _potd_from_value_plays(value_plays_df, existing=potd_picks)

# Archive today's plays to play_history so Mark Results and Play of the Day History can show them tomorrow.
# This covers: (1) all value plays from cache (e.g. Vanderbilt Moneyline), (2) POTD from cache or historical CSV (Iowa St, Alabama).
# Only archive once per session to avoid re-inserting plays that were deliberately deleted.
_archive_key = f"_archived_{date.today().isoformat()}"
if not STRIP_DOWN_MODE and not st.session_state.get(_archive_key):
    try:
        _req = ["League", "Event", "Selection", "Market", "Odds", "Value (%)", "model_prob", "home_team", "away_team"]
        if not value_plays_df.empty and all(c in value_plays_df.columns for c in _req):
            _to_archive = value_plays_df.copy()
            _to_archive = _to_archive.sort_values("Value (%)", ascending=False)
            if not _to_archive.empty:
                archive_value_plays(_to_archive, as_of_date=date.today())
        _potd_rows = []
        for _label in ("NCAAB Pick 1", "NCAAB Pick 2"):
            _p = potd_picks.get(_label)
            if not _p or not isinstance(_p, dict):
                continue
            _potd_rows.append({
                "League": _label,
                "Event": _p.get("Event", ""),
                "Selection": _p.get("Selection", ""),
                "Market": _p.get("Market", "Spread"),
                "Odds": _p.get("Odds") if _p.get("Odds") is not None else -110,
                "Value (%)": _p.get("Value (%)", 0),
                "point": _p.get("point"),
                "Recommended Stake": _p.get("Recommended Stake"),
                "home_team": _p.get("home_team", "—"),
                "away_team": _p.get("away_team", "—"),
                "model_prob": _p.get("model_prob", 0.5),
                "confidence_tier": _p.get("confidence_tier", "Medium"),
                "reasoning_summary": _p.get("reason") or _p.get("reasoning_summary"),
            })
        if _potd_rows:
            archive_value_plays(pd.DataFrame(_potd_rows), as_of_date=date.today())
        st.session_state[_archive_key] = True
    except Exception:
        pass

# -----------------------------------------------------------------------------
# Main layout — tabbed (Overview = daily picks sheet, NCAAB, NBA)
# -----------------------------------------------------------------------------

def _today_str() -> str:
    """Today's date as 'March 1' (no zero-pad)."""
    d = date.today()
    return d.strftime("%B ") + str(d.day)


def _summary_bar_values(value_plays_df: pd.DataFrame) -> dict:
    """From value_plays_df (today's data), compute: date_str, total_plays, n_nba, n_ncaab, top_edge_label.
    Uses correlated-play filter: at most 10 plays and no team appears twice."""
    # #region agent log
    import time
    _log_path = Path(__file__).resolve().parent / ".cursor" / "debug-874db2.log"
    def _dbg(msg: str, data: dict, hypothesis_id: str = "H0"):
        try:
            with open(_log_path, "a") as _f:
                _f.write(json.dumps({"sessionId": "874db2", "hypothesisId": hypothesis_id, "location": "app.py:_summary_bar_values", "message": msg, "data": data, "timestamp": int(time.time() * 1000)}) + "\n")
        except Exception:
            pass
    # #endregion
    date_str = _today_str()
    has_league = "League" in value_plays_df.columns
    has_value_pct = "Value (%)" in value_plays_df.columns
    if value_plays_df.empty or not has_league or not has_value_pct:
        _dbg("summary early return", {"df_empty": value_plays_df.empty, "has_League": has_league, "has_Value_pct": has_value_pct, "columns": list(value_plays_df.columns)}, "H3")
        return {
            "date_str": date_str,
            "total_plays": 0,
            "n_nba": 0,
            "n_ncaab": 0,
            "top_edge_label": "—",
        }
    best_per_game = (
        value_plays_df.sort_values("Value (%)", key=lambda x: x.abs(), ascending=False)
        .groupby("Event")
        .head(1)
        .reset_index(drop=True)
    )
    diversified = _filter_correlated_plays(best_per_game, max_plays=MAX_TOP_PLAYS_NO_CORRELATION)
    total_plays = len(diversified)
    n_nba = int((diversified["League"] == "NBA").sum()) if not diversified.empty else 0
    n_ncaab = int((diversified["League"] == "NCAAB").sum()) if not diversified.empty else 0
    _dbg("summary computed", {"best_per_game_len": len(best_per_game), "total_plays": total_plays, "n_nba": n_nba, "n_ncaab": n_ncaab}, "H3")
    if diversified.empty:
        top_edge_label = "—"
    else:
        top_row = diversified.loc[diversified["Value (%)"].abs().idxmax()]
        sel = str(top_row.get("Selection", ""))
        pt = top_row.get("point")
        if pt is not None:
            try:
                pt_f = float(pt)
                if str(top_row.get("Market", "")) == "totals":
                    sel = f"{sel} {pt_f:.1f}"
                else:
                    sel = f"{sel} {pt_f:+.1f}"
            except (TypeError, ValueError):
                pass
        edge_pct = abs(float(top_row.get("Value (%)", 0)))
        top_edge_label = f"{sel} at {edge_pct:.1f}%"
    return {
        "date_str": date_str,
        "total_plays": total_plays,
        "n_nba": n_nba,
        "n_ncaab": n_ncaab,
        "top_edge_label": top_edge_label,
    }


# Cross-platform "March 1" style date (no zero-padded day)
def _today_str() -> str:
    d = date.today()
    return d.strftime("%B ") + str(d.day)  # "March 1"


st.title("Bobby Bottle's Betting Model")
st.caption("NCAAB & MLB — college basketball value plays and Play of the Day on Overview; MLB value plays on the **MLB** tab.")

# Stat bar: date | plays found | NCAAB | top edge. Use selected slate date and _historical_picks_df when available.
if not _historical_picks_df.empty:
    try:
        _slate_d = datetime.strptime(_selected_slate_date, "%Y-%m-%d")
        _date_str = _slate_d.strftime("%B ") + str(_slate_d.day)
    except (ValueError, TypeError):
        _date_str = _selected_slate_date
    _n_slate = len(_historical_picks_df)
    _top_ep = _historical_picks_df["Edge_Points"].abs().max() if "Edge_Points" in _historical_picks_df.columns else None
    _top_edge_slate = f"{float(_top_ep) / 3:.1f}%" if _top_ep is not None and pd.notna(_top_ep) else "—"
    _summary = {"date_str": _date_str, "total_plays": _n_slate, "n_nba": 0, "n_ncaab": _n_slate, "top_edge_label": _top_edge_slate}
else:
    _summary = _summary_bar_values(value_plays_df)
    # Top edge must reflect the true max over all value plays and POTD (not just diversified subset)
    _vp_edges = []
    if not value_plays_df.empty and "Value (%)" in value_plays_df.columns:
        _vp_edges = value_plays_df["Value (%)"].abs().tolist()
    _potd_edges = []
    for _k in ("NCAAB Pick 1", "NCAAB Pick 2"):
        _c = potd_picks.get(_k)
        if _c and isinstance(_c, dict):
            _v = _c.get("Value (%)")
            if _v is not None:
                try:
                    _potd_edges.append(abs(float(_v)))
                except (TypeError, ValueError):
                    pass
    _all_edges = _vp_edges + _potd_edges
    if _all_edges:
        _true_top_pct = max(_all_edges)
        _summary = {**_summary, "top_edge_label": f"{_true_top_pct:.1f}%"}
if _summary["total_plays"] == 0 and (potd_picks.get("NCAAB Pick 1") or potd_picks.get("NCAAB Pick 2")):
    _n_potd = (1 if potd_picks.get("NCAAB Pick 1") else 0) + (1 if potd_picks.get("NCAAB Pick 2") else 0)
    _potd_edges = []
    for _k in ("NCAAB Pick 1", "NCAAB Pick 2"):
        _c = potd_picks.get(_k)
        if _c and isinstance(_c, dict):
            _v = _c.get("Value (%)")
            if _v is not None:
                try:
                    _potd_edges.append(float(_v))
                except (TypeError, ValueError):
                    pass
    _top_edge = f"{max(abs(e) for e in _potd_edges):.1f}%" if _potd_edges else "—"
    _summary = {**_summary, "total_plays": _n_potd, "n_ncaab": _n_potd, "top_edge_label": _top_edge}
_plays_text = f"{_summary['total_plays']} play{'s' if _summary['total_plays'] != 1 else ''} found"
_nba_ncaab = f"{_summary['n_ncaab']} NCAAB"
_stat_bar_html = f"""
<style>
.dashboard-stat-bar {{ display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap;
  font-size: 0.875rem; color: rgba(255,255,255,0.8); margin-top: -0.4rem; margin-bottom: 0.5rem; }}
.dashboard-stat-bar span {{ display: inline-flex; align-items: center; }}
.dashboard-stat-bar .stat-divider {{ color: rgba(255,255,255,0.35); user-select: none; }}
.dashboard-stat-bar .stat-label {{ color: rgba(255,255,255,0.55); margin-right: 0.25rem; }}
.dashboard-slim-alert {{ font-size: 0.8rem; color: rgba(255,255,255,0.85); margin-bottom: 0.5rem;
  display: flex; align-items: center; gap: 0.4rem; }}
.dashboard-slim-alert a {{ color: #f0c14b; text-decoration: none; }}
.dashboard-slim-alert a:hover {{ text-decoration: underline; }}
</style>
<div class="dashboard-stat-bar">
  <span>{_html_escape(_summary['date_str'])}</span>
  <span class="stat-divider">|</span>
  <span><span class="stat-label">Plays</span>{_html_escape(_plays_text)}</span>
  <span class="stat-divider">|</span>
  <span>{_html_escape(_nba_ncaab)}</span>
  <span class="stat-divider">|</span>
  <span><span class="stat-label">Top edge</span>{_html_escape(_summary['top_edge_label'])}</span>
</div>
"""
st.markdown(_stat_bar_html, unsafe_allow_html=True)
_odds_meta = st.session_state.get("odds_source_meta") or {}
if _odds_meta.get("used_espn_fallback"):
    _n_plays = _summary.get("total_plays", 0)
    st.caption(
        f"Odds source: **ESPN (fallback)** — {_odds_meta.get('espn_games', 0)} games, {_odds_meta.get('espn_odds_rows', 0)} odds rows. "
        f"**{_n_plays} play{'s' if _n_plays != 1 else ''}** generated (min edge 1.5%, min 1 book)."
    )
if st.session_state.get("value_plays_pipeline_error"):
    err_msg, err_tb = st.session_state["value_plays_pipeline_error"]
    st.error("**Value plays pipeline failed** (showing 0 plays). Error: " + str(err_msg))
    with st.expander("Traceback"):
        st.code(err_tb)

def _bet_history_table(records: list) -> None:
    """Render bet history dataframe; records are dicts with step, event_name, odds, etc."""
    if not records:
        st.info("No bets placed for this sport with current strategy and data.")
        return
    bet_df = pd.DataFrame(records)
    bet_df = bet_df.rename(columns={
        "step": "Step",
        "event_name": "Event",
        "odds": "Odds",
        "model_prob": "Model prob",
        "stake": "Stake",
        "result": "Result",
        "profit": "P/L",
    })
    bet_display = bet_df.copy()
    if "Odds" in bet_display.columns:
        bet_display["Odds"] = bet_display["Odds"].apply(format_american)
    for col in ("Stake", "P/L"):
        if col in bet_display.columns:
            bet_display[col] = bet_display[col].apply(lambda x: format_currency(x) if pd.notna(x) else "—")
    cols = ["Step", "Event", "Odds", "Model prob", "Stake", "Result", "P/L"]
    st.dataframe(
        bet_display[[c for c in cols if c in bet_display.columns]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Step": st.column_config.NumberColumn("Step", width="small"),
            "Event": st.column_config.TextColumn("Event", width="medium"),
            "Odds": st.column_config.TextColumn("Odds (American)", width="small"),
            "Model prob": st.column_config.NumberColumn("Model prob", format="%.2f", width="small"),
            "Stake": st.column_config.TextColumn("Stake", width="small"),
            "Result": st.column_config.TextColumn("Result", width="small"),
            "P/L": st.column_config.TextColumn("P/L", width="small"),
        },
    )

# Score yesterday's plays from ESPN so Win/Loss show correctly when opening the next day
if not STRIP_DOWN_MODE and "auto_result_run" not in st.session_state:
    try:
        run_auto_result()
    except Exception:
        pass
    st.session_state["auto_result_run"] = True

# Persistent banner: unresolved plays older than 24 hours (skip DB load in strip-down mode)
_ph_all = pd.DataFrame() if STRIP_DOWN_MODE else _load_play_history_cached()
_cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)
_unresolved_stale = 0
if not _ph_all.empty:
    _ph_all["result_clean"] = _ph_all["result"].apply(
        lambda x: str(x).strip().upper() if x is not None and not pd.isna(x) else None
    )
    _ph_all["date_parsed"] = pd.to_datetime(_ph_all["date_generated"], errors="coerce", utc=True)
    _unresolved = _ph_all[_ph_all["result_clean"].isna()]
    _unresolved_stale = (_unresolved["date_parsed"] < _cutoff).sum()
if _unresolved_stale > 0:
    _n_plays = int(_unresolved_stale)
    _plays_word = "play" if _n_plays == 1 else "plays"
    _slim_alert_html = (
        f'<div class="dashboard-slim-alert">'
        f'⚠️ {_n_plays} {_plays_word} need results marked. '
        f'<a href="#play-history">Go to NCAAB Record</a>'
        f'</div>'
    )
    st.markdown(_slim_alert_html, unsafe_allow_html=True)
st.markdown('<div id="play-history"></div>', unsafe_allow_html=True)
tab_overview, tab_ncaab, tab_mlb, tab_nba, tab_mark_results, tab_play_history, tab_manual_odds, tab_game_lookup = st.tabs(
    ["Overview", "NCAAB", "MLB", "NBA (Coming Soon)", "Mark Results", "NCAAB Record", "Manual Odds", "Game Lookup"]
)

with tab_overview:
    st.markdown(POTD_CARD_CSS, unsafe_allow_html=True)
    # ——— Yesterday's Results (slim strip: two POTD badges W/L) ———
    yesterday_rows = _get_yesterday_potd_results()
    if yesterday_rows:
        st.markdown(_render_yesterday_strip_html(yesterday_rows), unsafe_allow_html=True)
    # ——— POTD streak (last 10 days: Pick 1 and Pick 2 as dots; 🔥 if 3+ wins in a row) ———
    streak_pick1, streak_pick2 = _get_last_10_potd_results()
    if streak_pick1 or streak_pick2:
        st.markdown(_render_streak_html(streak_pick1, streak_pick2), unsafe_allow_html=True)
    # ——— Play of the Day (NCAAB: Nov–early Apr; MLB: Apr–Oct) ———
    st.subheader("Play of the Day")
    _overview_cal_day = date.today()
    _ncaab_potd_season = _ncaab_season_includes_date(_overview_cal_day)
    _mlb_potd_season = _mlb_season_includes_date(_overview_cal_day)
    if _ncaab_potd_season:
        pod_1 = potd_picks.get("NCAAB Pick 1")
        pod_2 = potd_picks.get("NCAAB Pick 2")
        need_reason_fallback = (pod_1 and not pod_1.get("reason")) or (pod_2 and not pod_2.get("reason"))
        feature_matrix_potd = _load_feature_matrix_cached(league=None) if need_reason_fallback else None
        b2b_teams = _get_b2b_teams_cached(datetime.now(ZoneInfo("America/New_York")).date().isoformat(), st.session_state.get("odds_refresh_key", 0)) if need_reason_fallback else frozenset()
        st.caption("NCAAB — top two value plays from cache / pipeline.")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown(
                _render_potd_card_html("NCAAB Pick 1", pod_1, "orange", feature_matrix=feature_matrix_potd, odds_as_of=datetime.now(timezone.utc), b2b_teams=b2b_teams, march_madness_mode=march_madness_mode),
                unsafe_allow_html=True,
            )
        with col_p2:
            st.markdown(
                _render_potd_card_html("NCAAB Pick 2", pod_2, "orange", feature_matrix=feature_matrix_potd, odds_as_of=datetime.now(timezone.utc), b2b_teams=b2b_teams, march_madness_mode=march_madness_mode),
                unsafe_allow_html=True,
            )
        if value_plays_df.empty and pod_1 is None and pod_2 is None:
            if (odds_api_key or "").strip():
                st.caption("No NCAAB Play of the Day: no games or value plays loaded for today. Click **Refresh** below to run the pipeline, or check back later.")
            else:
                st.caption("No NCAAB Play of the Day: set your **Odds API key** in the sidebar and click **Refresh** to run the pipeline.")
    else:
        st.info("**NCAAB — Season complete.** Picks return in early November through the first week of April.")

    if _mlb_potd_season:
        st.markdown("##### MLB")
        st.caption(
            "Top moneyline plays (by edge) and best over/under play from `data/cache/mlb_value_plays.json` — same card style as NCAAB."
        )
        _mlb_df_overview, _ = _load_mlb_value_plays_for_today()
        mlb_pod_1, mlb_pod_2, mlb_pod_total = _mlb_overview_potd_picks(_mlb_df_overview)
        _mlb_odds_asof = datetime.now(timezone.utc)
        _park_ov = _load_mlb_park_runs_lookup()
        _form_nm = _load_mlb_recent_form_by_team_name()

        def _mlb_overview_card_decorations(pod: Optional[dict]) -> tuple[Optional[str], Optional[str], Optional[str]]:
            if not pod:
                return None, None, None
            a = str(pod.get("away_team", "")).strip()
            h = str(pod.get("home_team", "")).strip()
            ad = a + _mlb_recent_form_suffix_next_to_team(a, _form_nm)
            hd = h + _mlb_recent_form_suffix_next_to_team(h, _form_nm)
            pruns = _mlb_park_runs_for_home_team(h, _park_ov)
            pline = _mlb_park_text_label_for_runs(pruns)
            return ad, hd, pline

        a1, h1, p1 = _mlb_overview_card_decorations(mlb_pod_1)
        a2, h2, p2 = _mlb_overview_card_decorations(mlb_pod_2)
        at, ht, pt = _mlb_overview_card_decorations(mlb_pod_total)

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(
                _render_potd_card_html(
                    "MLB Pick 1",
                    mlb_pod_1,
                    "blue",
                    feature_matrix=None,
                    odds_as_of=_mlb_odds_asof,
                    b2b_teams=frozenset(),
                    march_madness_mode=False,
                    mlb_park_line=p1,
                    mlb_away_display=a1,
                    mlb_home_display=h1,
                ),
                unsafe_allow_html=True,
            )
        with col_m2:
            st.markdown(
                _render_potd_card_html(
                    "MLB Pick 2",
                    mlb_pod_2,
                    "blue",
                    feature_matrix=None,
                    odds_as_of=_mlb_odds_asof,
                    b2b_teams=frozenset(),
                    march_madness_mode=False,
                    mlb_park_line=p2,
                    mlb_away_display=a2,
                    mlb_home_display=h2,
                ),
                unsafe_allow_html=True,
            )
        with col_m3:
            st.markdown(
                _render_potd_card_html(
                    "MLB Total Pick",
                    mlb_pod_total,
                    "blue",
                    feature_matrix=None,
                    odds_as_of=_mlb_odds_asof,
                    b2b_teams=frozenset(),
                    march_madness_mode=False,
                    mlb_park_line=pt,
                    mlb_away_display=at,
                    mlb_home_display=ht,
                ),
                unsafe_allow_html=True,
            )
        if mlb_pod_1 is None and mlb_pod_2 is None and mlb_pod_total is None:
            st.caption("No MLB plays in cache for today. Open the **MLB** tab to run the model or update the cache.")

    st.divider()

    # ——— All Value Plays (spread only; totals picks hidden pending further model calibration) ———
    st.subheader("All Value Plays")
    st.caption("Today's best NCAAB value plays sorted by edge. Powered by XGBoost multi-feature model.")
    if _n_value_plays_already_started > 0:
        n_total = len(value_plays_df) if not value_plays_df.empty else 0
        st.caption(f"**{_n_value_plays_already_started}** play(s) already started — showing **{n_total}** remaining.")
    if value_plays_flagged_count > 0:
        st.warning(f"**Potential Data Error:** {value_plays_flagged_count} play(s) had Value % ≥ 15% and were excluded.")

    # Spread only; totals picks are hidden pending further model calibration (see predict_games.py SHOW_TOTALS_PICKS).
    def _row_event(r):
        return str(r.get("Event", "")).strip() or "—"
    def _row_pick(r, market: str):
        sel = str(r.get("Selection", "")).strip()
        pt = r.get("point") if "point" in r else r.get("Point") or r.get("Over_Under")
        if pt is None or (isinstance(pt, float) and pd.isna(pt)):
            return sel or "—"
        try:
            p = float(pt)
        except (TypeError, ValueError):
            return f"{sel} {pt}" if sel else str(pt)
        if market == "Total":
            return f"{sel} {p:.1f}"
        return f"{sel} {p:+.1f}"
    def _row_spread_line(r, market: str):
        pt = r.get("point") if "point" in r else r.get("Point") or r.get("Over_Under")
        if pt is None or (isinstance(pt, float) and pd.isna(pt)):
            return "—"
        try:
            p = float(pt)
        except (TypeError, ValueError):
            return str(pt)
        return f"{p:+.1f}" if market == "Spread" else f"{p:.1f}"
    def _row_edge(r, market: str):
        if market == "Spread":
            e = r.get("Edge_pts")
            if e is not None and not (isinstance(e, float) and pd.isna(e)):
                try:
                    return round(float(e), 1)
                except (TypeError, ValueError):
                    pass
            v = r.get("Value (%)")
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                try:
                    return round(float(v) * 3.0, 1)  # approximate from Value %
                except (TypeError, ValueError):
                    pass
            return "—"
        e = r.get("Totals_Edge")
        if e is not None and not (isinstance(e, float) and pd.isna(e)):
            try:
                return round(float(e), 1)
            except (TypeError, ValueError):
                pass
        return "—"
    def _row_confidence(r):
        return str(r.get("confidence_tier", "")).strip() or "—"

    rows = []
    if not value_plays_df.empty and "Event" in value_plays_df.columns:
        for _, r in value_plays_df.iterrows():
            if str(r.get("League", "")).strip().upper() != "NCAAB":
                continue
            edge_val = _row_edge(r, "Spread")
            abs_edge = abs(edge_val) if isinstance(edge_val, (int, float)) else 0.0
            rows.append({
                "Event": _row_event(r),
                "Pick": _row_pick(r, "Spread"),
                "Spread/Line": _row_spread_line(r, "Spread"),
                "Edge (pts)": edge_val,
                "Confidence": _row_confidence(r),
                "_abs_edge": abs_edge,
            })
    # Totals plays are not shown (hidden pending further model calibration).

    if rows:
        combined = pd.DataFrame(rows).sort_values("_abs_edge", ascending=False).drop(columns=["_abs_edge"])
        st.dataframe(combined, use_container_width=True, hide_index=True)
    else:
        if (odds_api_key or "").strip():
            st.info("No value plays from cache today. Run **scripts/predict_games.py** (or the pipeline) to generate plays.")
        else:
            st.info("Set Odds API key in the sidebar or run **scripts/predict_games.py** with an odds CSV to load value plays.")

with tab_ncaab:
    st.subheader("NCAAB")
    st.caption("Today's NCAAB value plays ranked by edge. Picks powered by XGBoost multi-feature model.")
    if _n_value_plays_already_started > 0:
        remaining = len(value_plays_df[value_plays_df["League"] == "NCAAB"]) if not value_plays_df.empty and "League" in value_plays_df.columns else 0
        st.caption(f"**{_n_value_plays_already_started}** play(s) already started — showing **{remaining}** remaining.")
    if (odds_api_key or "").strip():
        if st.button("Refresh", key="ncaab_refresh", help="Run the value-plays pipeline and reload cache"):
            _script = Path(__file__).resolve().parent / "scripts" / "run_pipeline_to_cache.py"
            env = os.environ.copy()
            env["ODDS_API_KEY"] = (odds_api_key or "").strip()
            try:
                subprocess.run(
                    [sys.executable, str(_script)],
                    cwd=str(Path(__file__).resolve().parent),
                    env=env,
                    timeout=300,
                    check=False,
                    shell=False,
                )
            except subprocess.TimeoutExpired:
                st.error("Pipeline timed out after 5 minutes.")
            except Exception as e:
                st.error(f"Pipeline error: {e}")
            st.rerun()
    ncaab_value = value_plays_df[value_plays_df["League"] == "NCAAB"] if not value_plays_df.empty and "League" in value_plays_df.columns else pd.DataFrame()
    if not ncaab_value.empty:
        ncaab_best = ncaab_value.sort_values("Value (%)", ascending=False).groupby("Event").head(1).reset_index(drop=True)
        if march_madness_mode:
            ncaab_best = ncaab_best[ncaab_best["Value (%)"] >= 5.0]
        st.caption("**Best value plays (NCAAB)** — same model as main table. March: conf. tourney, seeds, bubble vs clinched." + (" **March Madness mode:** 5% min edge, tournament-eligible only." if march_madness_mode else ""))
        _vp = ncaab_best.copy()
        _vp["Odds"] = _vp["Odds"].apply(format_american)
        if "Market" in _vp.columns:
            _vp["Market"] = _vp["Market"].map(lambda x: MARKET_LABELS.get(x, x) if pd.notna(x) else x)
        if "Recommended Stake" in _vp.columns:
            _vp = _vp.drop(columns=["Recommended Stake"])
        # Pick: "Home [Team]" or "Away [Team]" for clarity
        if "Selection" in _vp.columns and "home_team" in _vp.columns and "away_team" in _vp.columns:
            def _pick_label_vp(r):
                sel = str(r.get("Selection", "")).strip()
                h, a = str(r.get("home_team", "")).strip(), str(r.get("away_team", "")).strip()
                if sel and (sel.lower() == h.lower() or h and sel.lower() in h.lower() or h and h.lower() in sel.lower()):
                    return f"Home {h}"
                return f"Away {a}" if a else sel or "—"
            _vp["Pick"] = _vp.apply(_pick_label_vp, axis=1)
        # Market (home): spread line from home perspective with explicit +/-
        if "point" in _vp.columns and "Selection" in _vp.columns and "home_team" in _vp.columns:
            def _market_home_signed(r):
                pt = r.get("point")
                if pt is None or pd.isna(pt):
                    return "—"
                try:
                    p = float(pt)
                except (TypeError, ValueError):
                    return "—"
                sel = str(r.get("Selection", "")).strip()
                h = str(r.get("home_team", "")).strip()
                # Home spread: negative = home favored. If Selection is home, point is often home line; else away line = -home
                is_home_sel = sel and h and (sel.lower() == h.lower() or sel.lower() in h.lower() or h.lower() in sel.lower())
                home_spread = p if is_home_sel else (-p if p != 0 else 0)
                return f"{home_spread:+.1f}"
            _vp["Market (home)"] = _vp.apply(_market_home_signed, axis=1)
        _vp["March context"] = _vp.apply(_march_context_badges, axis=1)
        if march_madness_mode:
            _vp["Tournament Context"] = _vp.apply(lambda r: "Yes" if _is_top_seed_play(r) else "", axis=1)
        # Display table: only Point (spread from picked team), drop duplicate point and internal columns
        _cols_drop = ["point", "League", "Injury Alert", "home_team", "away_team", "March context", "Tournament Context", "Market (home)"]
        _vp_display = _vp.drop(columns=[c for c in _cols_drop if c in _vp.columns])
        st.dataframe(_vp_display.rename(columns={"Event": "Game", "confidence_tier": "Confidence"}), use_container_width=True, hide_index=True)
    else:
        if (odds_api_key or "").strip():
            st.info("No NCAAB games for today from the Odds API, or no value plays with 3% ≤ Value % < 15%. Try again later or use Refresh.")
        else:
            st.info("Set Odds API key in the sidebar to load today's NCAAB games and value plays.")

with tab_mlb:
    st.subheader("MLB")
    st.caption("Today's MLB value plays from the local cache (data/cache/mlb_value_plays.json). Edge = (model prob × decimal odds) − 1, same as NCAAB Value % basis.")
    if st.session_state.pop("mlb_model_success_msg", None):
        st.success(
            "MLB pipeline finished: `data/odds/live_mlb_odds.json` is updated and `predict_mlb.py` has run. "
            "Reloaded picks below."
        )
    _mlb_live_odds_path = _APP_ROOT / "data" / "odds" / "live_mlb_odds.json"
    if st.button("🚀 Run MLB Model", key="mlb_run_model", help="Run fetch_mlb_odds.py then predict_mlb.py (refreshes odds and value-play cache)"):
        with st.spinner("Running scripts/fetch_mlb_odds.py… then scripts/predict_mlb.py…"):
            _root = str(_APP_ROOT)
            _fetch = _APP_ROOT / "scripts" / "fetch_mlb_odds.py"
            _predict = _APP_ROOT / "scripts" / "predict_mlb.py"
            try:
                r1 = subprocess.run(
                    [sys.executable, str(_fetch)],
                    cwd=_root,
                    capture_output=True,
                    text=True,
                    timeout=180,
                    shell=False,
                )
                if r1.returncode != 0:
                    err = (r1.stderr or r1.stdout or "").strip() or f"exit {r1.returncode}"
                    st.error(f"fetch_mlb_odds.py failed: {err}")
                else:
                    r2 = subprocess.run(
                        [sys.executable, str(_predict)],
                        cwd=_root,
                        capture_output=True,
                        text=True,
                        timeout=180,
                        shell=False,
                    )
                    if r2.returncode != 0:
                        err = (r2.stderr or r2.stdout or "").strip() or f"exit {r2.returncode}"
                        st.error(f"predict_mlb.py failed: {err}")
                    elif _mlb_live_odds_path.is_file():
                        st.session_state["mlb_model_success_msg"] = True
                        st.rerun()
                    else:
                        st.error("Pipeline reported success but `data/odds/live_mlb_odds.json` was not found.")
            except subprocess.TimeoutExpired:
                st.error("MLB pipeline timed out after 3 minutes.")
            except Exception as e:
                st.error(f"MLB pipeline error: {e}")
    mlb_df, mlb_plays_raw = _load_mlb_value_plays_for_today()
    if mlb_df.empty:
        st.info(
            f"No MLB value plays for **{date.today().isoformat()}**. Add or update **{MLB_VALUE_PLAYS_JSON_PATH.name}** "
            "(set `card_date` to today or give each play a `card_date`)."
        )
    else:
        st.markdown(POTD_CARD_CSS, unsafe_allow_html=True)
        _park_lu = _load_mlb_park_runs_lookup()
        disp = mlb_df.copy()
        if "odds_american" in disp.columns:
            disp["Odds"] = disp["odds_american"].apply(format_american)
        if "edge" in disp.columns:
            disp["Edge %"] = disp["edge"].apply(lambda x: round(float(x) * 100.0, 2) if pd.notna(x) else None)
        elif "edge_pct" in disp.columns:
            disp["Edge %"] = disp["edge_pct"].apply(
                lambda x: (
                    round(float(x), 2)
                    if pd.notna(x) and abs(float(x)) >= 1.0
                    else round(float(x) * 100.0, 2)
                )
                if pd.notna(x)
                else None
            )
        if "model_prob" in disp.columns:
            disp["Model prob"] = disp["model_prob"]
        stake_col = []
        for _, r in disp.iterrows():
            try:
                o = float(r.get("odds_american", 0))
                p = float(r.get("model_prob", 0))
                frac = kelly_fraction(o, p, fraction=kelly_frac)
                stake_col.append(round(BANKROLL_FOR_STAKES * frac, 2))
            except (TypeError, ValueError):
                stake_col.append(None)
        disp["Rec. stake ($)"] = stake_col
        disp["Confidence"] = disp["Edge %"].apply(_mlb_confidence_from_edge_pct)
        if "away_team" in disp.columns:
            disp["Away"] = disp["away_team"].astype(str).str.strip()
        if "home_team" in disp.columns:
            disp["Home"] = disp["home_team"].apply(lambda h: _mlb_home_display_with_park(str(h).strip(), _park_lu))
        if "away_pitcher" in disp.columns:
            disp["SP (away)"] = disp["away_pitcher"]
        if "home_pitcher" in disp.columns:
            disp["SP (home)"] = disp["home_pitcher"]
        if "selection" in disp.columns:
            disp["Pick"] = disp["selection"]

        if "market" in disp.columns:
            _mkt = disp["market"].astype(str).str.strip().str.lower()
            ml_only = disp[_mkt.eq("moneyline")].copy()
            tot_only = disp[_mkt.eq("total")].copy()
        else:
            ml_only = disp.copy()
            tot_only = pd.DataFrame()

        top_ml_row = None
        if not ml_only.empty and "edge" in ml_only.columns:
            try:
                _ei = ml_only["edge"].astype(float)
                if _ei.notna().any():
                    top_ml_row = ml_only.loc[_ei.idxmax()]
            except (TypeError, ValueError):
                top_ml_row = None

        st.markdown(_render_mlb_top_play_card_html(top_ml_row, _park_lu), unsafe_allow_html=True)

        st.markdown("##### Moneyline Plays")
        if ml_only.empty:
            st.caption("No plays today")
        else:
            _ml_show = [
                c
                for c in [
                    "Away",
                    "Home",
                    "SP (away)",
                    "SP (home)",
                    "Pick",
                    "Odds",
                    "Model prob",
                    "Edge %",
                    "Confidence",
                    "Rec. stake ($)",
                ]
                if c in ml_only.columns
            ]
            _ml_df = ml_only[_ml_show].copy()
            _ml_fmt: dict[str, str] = {}
            if "Model prob" in _ml_df.columns:
                _ml_fmt["Model prob"] = "{:.3f}"
            if "Edge %" in _ml_df.columns:
                _ml_fmt["Edge %"] = "{:.2f}"
            if "Rec. stake ($)" in _ml_df.columns:
                _ml_fmt["Rec. stake ($)"] = "${:.2f}"
            _ml_styled = (
                _ml_df.style.apply(_mlb_edge_tier_row_style, axis=1)
                .format(_ml_fmt, na_rep="—")
                .hide(axis="index")
            )
            st.dataframe(_ml_styled, use_container_width=True)

        st.markdown("##### Over/Under Plays")
        if tot_only.empty:
            st.caption("No plays today")
        else:
            if "predicted_total" in tot_only.columns:
                tot_only["Pred. Total"] = tot_only["predicted_total"].apply(
                    lambda x: round(float(x), 2) if x is not None and pd.notna(x) else None
                )
            _ou_show = [
                c
                for c in [
                    "Away",
                    "Home",
                    "SP (away)",
                    "SP (home)",
                    "Pick",
                    "Odds",
                    "Pred. Total",
                    "Edge %",
                    "Confidence",
                    "Rec. stake ($)",
                ]
                if c in tot_only.columns
            ]
            _ou_df = tot_only[_ou_show].copy()
            _ou_fmt: dict[str, str] = {}
            if "Pred. Total" in _ou_df.columns:
                _ou_fmt["Pred. Total"] = "{:.2f}"
            if "Edge %" in _ou_df.columns:
                _ou_fmt["Edge %"] = "{:.2f}"
            if "Rec. stake ($)" in _ou_df.columns:
                _ou_fmt["Rec. stake ($)"] = "${:.2f}"
            _ou_styled = (
                _ou_df.style.apply(_mlb_edge_tier_row_style, axis=1)
                .format(_ou_fmt, na_rep="—")
                .hide(axis="index")
            )
            st.dataframe(_ou_styled, use_container_width=True)

        if st.button(
            "Save Picks",
            key="mlb_save_play_history",
            help="Archive today's MLB rows to play_history in data/espn.db (sport=MLB). Re-save updates edges/odds for the same date.",
        ):
            try:
                _mlb_archive_df = _mlb_dataframe_for_play_history(mlb_df)
                if _mlb_archive_df.empty:
                    st.warning("Nothing to save — no valid rows on today's card.")
                else:
                    n_saved = archive_value_plays(_mlb_archive_df, as_of_date=date.today())
                    _load_play_history_cached.clear()
                    st.success(f"Saved **{n_saved}** MLB pick(s) to local **play_history** (`data/espn.db`, sport=**MLB**).")
            except Exception as e:
                st.error(f"Could not save picks — {e}")

with tab_mark_results:
    st.subheader("Mark Results")
    st.caption("Mark Win / Loss / Push for plays with **Pending** results only (ignores slate date). You can see Saturday pending games while viewing Sunday's slate.")
    if st.button("🔄 Refresh pending plays", help="Reload play history from the database (clears 30‑min cache so new or updated plays appear)"):
        _load_play_history_cached.clear()
        st.rerun()
    _today = date.today()
    _yesterday = _today - timedelta(days=1)
    _mr_from = _today - timedelta(days=90)  # load last 90 days to find pending
    _mr_history = _load_play_history_cached(from_date_iso=_mr_from.isoformat(), to_date_iso=_yesterday.isoformat())
    if not _mr_history.empty:
        _mr_history = _mr_history.copy()
        _mr_history = _mr_history[_mr_history["sport"].astype(str).str.strip().str.upper().str.startswith("NCAAB")]
        _mr_history["result_clean"] = _mr_history["result"].apply(
            lambda x: str(x).strip().upper() if x is not None and not pd.isna(x) else None
        )
        # Show only Pending rows (ignore sidebar date) so user can mark Saturday games while viewing Sunday
        _mr_history = _mr_history[_mr_history["result_clean"].isna()]

        def _mr_matchup_key(row: pd.Series) -> str:
            d = row["date_generated"]
            s = row["sport"]
            h = str(row["home_team"] or "").strip()
            a = str(row["away_team"] or "").strip()
            bt = str(row["bet_type"] or "").strip().lower()
            side = str(row["recommended_side"] or "").strip()
            line = row.get("spread_or_total")
            if bt == "spread" and line is not None and not pd.isna(line) and line != -999:
                try:
                    line_f = float(line)
                    is_away_side = side.lower() == a.lower()
                    home_spread = -line_f if is_away_side else line_f
                    return f"{d}|{s}|{h}|{a}|Spread|{home_spread:.2f}"
                except (TypeError, ValueError):
                    pass
            return f"{d}|{s}|{h}|{a}|{bt}|{side}"
        _mr_history["_matchup_key"] = _mr_history.apply(_mr_matchup_key, axis=1)
        _mr_list = _mr_history.drop_duplicates(subset=["_matchup_key"], keep="last").sort_values(
            ["date_generated", "my_edge_pct"], ascending=[False, False]
        )
        if _mr_list.empty:
            st.info("No pending plays. All plays in the last 90 days have been marked Win / Loss / Push.")
        else:
            st.markdown("""
        <style>
        .mr-card { border-radius: 12px; padding: 1.1rem 1.25rem; margin-bottom: 1rem; border-left: 5px solid; box-shadow: 0 2px 8px rgba(0,0,0,0.2); }
        .mr-card--pending { border-left-color: #78909c; background: rgba(96,125,139,0.12); }
        .mr-card--win { border-left-color: #2e7d32; background: rgba(46,125,50,0.22); }
        .mr-card--loss { border-left-color: #c62828; background: rgba(198,40,40,0.22); }
        .mr-card--push { border-left-color: #616161; background: rgba(97,97,97,0.22); }
        .mr-matchup { font-size: 1.1rem; font-weight: 700; color: #fafafa; margin-bottom: 0.3rem; }
        .mr-meta { font-size: 0.9rem; color: rgba(255,255,255,0.85); margin-bottom: 0.4rem; }
        .mr-badge { display: inline-block; font-size: 1.1rem; font-weight: 700; padding: 0.25rem 0.6rem; border-radius: 6px; }
        .mr-badge--win { background: #2e7d32; color: #fff; }
        .mr-badge--loss { background: #c62828; color: #fff; }
        .mr-badge--push { background: #616161; color: #fff; }
        div:has(.mr-card) + div .stButton button { min-width: 5.5rem; box-sizing: border-box; }
        </style>
        """, unsafe_allow_html=True)

            def _mr_bet_str(r: pd.Series) -> str:
                bt = str(r.get("bet_type", ""))
                side = str(r.get("recommended_side", "")).strip()
                line = r.get("spread_or_total")
                home, away = str(r.get("home_team", "")).strip(), str(r.get("away_team", "")).strip()
                if line is None or pd.isna(line) or line == -999:
                    return f"{bt} · {side}".strip()
                line = float(line)
                if "Over" in bt or "Under" in bt or "total" in bt.lower():
                    return f"{bt} · {side} {line:.1f}".strip()
                # Spread: show picked team and that team's line (e.g. "Pennsylvania +9.5" not "YALE +9.5" when pick was away underdog)
                if "Spread" in bt or "spread" in bt.lower():
                    def _side_matches(s: str, team: str) -> bool:
                        if not s or not team:
                            return False
                        return s.lower() == team.lower() or s.lower() in team.lower() or team.lower() in s.lower()
                    if _side_matches(side, home):
                        display_line = line
                        team = away if display_line > 0 else home
                    elif _side_matches(side, away):
                        team, display_line = away, (-line if line < 0 else line)
                    elif side.lower() in ("home", "away"):
                        team = home if side.lower() == "home" else away
                        display_line = (-line if side.lower() == "away" and line < 0 else line)
                    else:
                        team, display_line = side, line
                    # When DB stored "Home" but line is underdog (+): show away underdog (e.g. Pennsylvania +9.5)
                    if display_line > 0 and away and _side_matches(team, home):
                        team, display_line = away, display_line
                    # Only when we're showing away favorite (-) but pick was home underdog: show home +line (do not overwrite away +line)
                    if display_line < 0 and home and _side_matches(team, away) and side.lower() == "home" and line > 0:
                        team, display_line = home, line
                    return f"{bt} · {team} {display_line:+.1f}".strip()
                return f"{bt} · {side} {line:+.1f}".strip()

            for _, row in _mr_list.iterrows():
                result = row.get("result_clean")
                card_class = "mr-card--pending"
                if result == "W":
                    card_class = "mr-card--win"
                elif result == "L":
                    card_class = "mr-card--loss"
                elif result == "P":
                    card_class = "mr-card--push"
                matchup = f"{row.get('away_team', '')} @ {row.get('home_team', '')}"
                bet_str = _mr_bet_str(row)
                edge_str = f"{float(row.get('my_edge_pct', 0)):.1f}%"
                odds_str = format_american(row.get("market_odds_at_time", 0))
                sport_label = str(row.get("sport", ""))
                if result == "W":
                    badge = '<span class="mr-badge mr-badge--win">Win</span>'
                elif result == "L":
                    badge = '<span class="mr-badge mr-badge--loss">Loss</span>'
                elif result == "P":
                    badge = '<span class="mr-badge mr-badge--push">Push</span>'
                else:
                    badge = ""
                st.markdown(
                    f'<div class="mr-card {card_class}">'
                    f'<div class="mr-matchup">{_html_escape(sport_label)} · {_html_escape(matchup)}</div>'
                    f'<div class="mr-meta">{_html_escape(bet_str)} · Edge {edge_str} · Odds {odds_str}</div>'
                    + (f'<div class="mr-meta">{badge}</div>' if badge else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )
                if result is None or pd.isna(result):
                    mk = row.get("_matchup_key", "")
                    pids_in_group = _mr_history.loc[_mr_history["_matchup_key"] == mk, "play_id"].astype(int).tolist()
                    c1, c2, c3, c4, _ = st.columns([1, 1, 1, 1, 2])
                    with c1:
                        if st.button("✅ Win", key=f"mr_w_{row.get('play_id')}", help="Mark win"):
                            for _pid in pids_in_group:
                                update_play_result(_pid, "W")
                            _load_play_history_cached.clear()
                            st.rerun()
                    with c2:
                        if st.button("❌ Loss", key=f"mr_l_{row.get('play_id')}", help="Mark loss"):
                            for _pid in pids_in_group:
                                update_play_result(_pid, "L")
                            _load_play_history_cached.clear()
                            st.rerun()
                    with c3:
                        if st.button("➖ Push", key=f"mr_p_{row.get('play_id')}", help="Mark push"):
                            for _pid in pids_in_group:
                                update_play_result(_pid, "P")
                            _load_play_history_cached.clear()
                            st.rerun()
                    with c4:
                        if st.button("🗑️ Delete", key=f"mr_del_{row.get('play_id')}", help="Permanently remove this play from play_history and historical_betting_performance.csv"):
                            first_row_data = None
                            for _pid in pids_in_group:
                                deleted = delete_play(_pid)
                                if deleted and first_row_data is None:
                                    first_row_data = deleted
                            if first_row_data:
                                _remove_play_from_historical_csv(
                                    first_row_data["date_generated"],
                                    first_row_data["home_team"],
                                    first_row_data["away_team"],
                                )
                            _load_play_history_cached.clear()
                            st.rerun()
    else:
        st.info("No past plays in the last 90 days to mark. Plays from yesterday and earlier appear here once they are archived.")

with tab_play_history:
    st.subheader("NCAAB Record")
    _record_view = st.radio("View", ["All Picks", "Regular Season", "March Madness"], horizontal=True, label_visibility="collapsed")
    st.caption("NCAAB daily picks & NCAA Tournament value plays (last 30 days).")
    _from = date.today() - timedelta(days=30)
    # Load fresh from DB on every render (no cache) so Mark Results / Delete updates appear immediately
    _history = load_play_history(from_date=_from, to_date=date.today())
    if not _history.empty:
        _history = _history.copy()
        _history["result_clean"] = _history["result"].apply(
            lambda x: str(x).strip().upper() if x is not None and not pd.isna(x) else None
        )

        def _canonical_matchup_key(row: pd.Series) -> str:
            d = row["date_generated"]
            s = row["sport"]
            h = str(row["home_team"] or "").strip()
            a = str(row["away_team"] or "").strip()
            bt = str(row["bet_type"] or "").strip().lower()
            side = str(row["recommended_side"] or "").strip()
            line = row.get("spread_or_total")
            if bt == "spread" and line is not None and not pd.isna(line) and line != -999:
                try:
                    line_f = float(line)
                    is_away_side = side.lower() == a.lower()
                    home_spread = -line_f if is_away_side else line_f
                    return f"{d}|{s}|{h}|{a}|Spread|{home_spread:.2f}"
                except (TypeError, ValueError):
                    pass
            return f"{d}|{s}|{h}|{a}|{bt}|{side}"
        _history["_matchup_key"] = _history.apply(_canonical_matchup_key, axis=1)
        _history = _history.drop_duplicates(subset=["_matchup_key"], keep="last")
        _sport_col = _history["sport"].astype(str).str.strip()

        _potd_picks = _history.loc[_history.groupby(["date_generated", "sport"])["my_edge_pct"].idxmax()].reset_index(drop=True)
        _potd_picks = _potd_picks[_potd_picks["sport"].astype(str).str.strip().isin(("NCAAB Pick 1", "NCAAB Pick 2"))]
        _tourney_picks = _history[_sport_col == "NCAAB Tournament"].copy()

        if _record_view == "Regular Season":
            top_per_day = _potd_picks
        elif _record_view == "March Madness":
            top_per_day = _tourney_picks
        else:
            top_per_day = pd.concat([_potd_picks, _tourney_picks]).sort_values("date_generated", ascending=False).reset_index(drop=True)

        _all_ncaab = _history[_history["result_clean"].notna() & _history["sport"].astype(str).str.strip().str.upper().str.startswith("NCAAB")]
        if _record_view == "March Madness":
            resolved = _all_ncaab[_all_ncaab["sport"].astype(str).str.strip() == "NCAAB Tournament"]
        elif _record_view == "Regular Season":
            resolved = _all_ncaab[_all_ncaab["sport"].astype(str).str.strip() != "NCAAB Tournament"]
        else:
            resolved = _all_ncaab
        resolved_potd = top_per_day[top_per_day["result_clean"].notna()]

        def _roi_and_units(df: pd.DataFrame) -> tuple[float, float, float, float, float]:
            if df.empty:
                return 0.0, 0.0, 0, 0, 0, 0.0
            stake_vals = pd.to_numeric(df["recommended_stake"], errors="coerce").fillna(0)
            staked = float(stake_vals.sum())
            staked = 0.0 if staked == 0 else staked
            payout_vals = pd.to_numeric(df["actual_payout"], errors="coerce").fillna(0)
            payout = float(payout_vals.sum())
            roi = (payout / staked * 100) if staked else 0.0
            units = 0.0
            for _, r in df.iterrows():
                s, pa = r.get("recommended_stake"), r.get("actual_payout")
                try:
                    s_f = float(pd.to_numeric(s, errors="coerce") or 0)
                    if s_f == 0 or pa is None or pd.isna(pa):
                        continue
                    pa_f = float(pd.to_numeric(pa, errors="coerce"))
                    if not pd.isna(pa_f):
                        units += pa_f / s_f
                except (TypeError, ValueError):
                    pass
            w = (df["result_clean"] == "W").sum()
            l = (df["result_clean"] == "L").sum()
            p = (df["result_clean"] == "P").sum()
            win_rate = (w / (w + l) * 100) if (w + l) > 0 else 0.0
            return roi, units, w, l, p, win_rate

        roi_pct, profit_units, w, l, p, win_rate = _roi_and_units(resolved)
        roi_potd, profit_units_potd, w_potd, l_potd, p_potd, win_rate_potd = _roi_and_units(resolved_potd)
        st.markdown("""
        <style>
        .ph-kpi-hero { text-align: center; padding: 1.25rem 1.5rem; margin-bottom: 1rem; }
        .ph-kpi-record { font-size: 3.25rem; font-weight: 900; letter-spacing: 0.04em; line-height: 1.2; }
        .ph-kpi-record .ph-w { color: #4caf50; }
        .ph-kpi-record .ph-l { color: #f44336; }
        .ph-kpi-record .ph-p { color: #9e9e9e; }
        .ph-kpi-second { text-align: center; display: flex; flex-wrap: wrap; justify-content: center; gap: 2rem; margin-bottom: 1.25rem; font-size: 1.35rem; font-weight: 700; }
        .ph-kpi-second .ph-roi { font-size: 1.5rem; }
        .ph-kpi-second .ph-roi--pos { color: #4caf50; }
        .ph-kpi-second .ph-roi--neg { color: #f44336; }
        .ph-kpi-second .ph-roi--zero { color: rgba(255,255,255,0.8); }
        .ph-kpi-breakdown { display: flex; flex-direction: column; align-items: center; gap: 0.5rem; margin-top: 0.75rem; padding: 0.75rem 1.5rem; background: rgba(255,255,255,0.05); border-radius: 10px; font-size: 0.95rem; }
        .ph-kpi-breakdown-row { display: flex; align-items: center; justify-content: center; gap: 0.75rem; width: 100%; }
        .ph-kpi-breakdown-label { color: rgba(255,255,255,0.7); font-size: 0.85rem; min-width: 7rem; text-align: right; }
        .ph-kpi-breakdown-record { font-weight: 700; }
        .ph-kpi-breakdown span { color: rgba(255,255,255,0.85); }
        .ph-kpi-breakdown strong { color: #fafafa; }
        .ph-feed-card { border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem; border-left: 5px solid; box-shadow: 0 2px 8px rgba(0,0,0,0.2); }
        .ph-feed-card--pending { border-left-color: #78909c; background: rgba(96,125,139,0.12); }
        .ph-feed-card--win { border-left-color: #2e7d32; background: rgba(46,125,50,0.22); }
        .ph-feed-card--loss { border-left-color: #c62828; background: rgba(198,40,40,0.22); }
        .ph-feed-card--push { border-left-color: #616161; background: rgba(97,97,97,0.22); }
        .ph-feed-date { font-size: 0.8rem; color: rgba(255,255,255,0.65); margin-bottom: 0.35rem; }
        .ph-feed-matchup { font-size: 1.1rem; font-weight: 700; color: #fafafa; margin-bottom: 0.4rem; }
        .ph-feed-meta { font-size: 0.9rem; color: rgba(255,255,255,0.85); margin-bottom: 0.6rem; }
        .ph-feed-badge { display: inline-block; font-size: 1.4rem; font-weight: 800; padding: 0.35rem 0.75rem; border-radius: 8px; letter-spacing: 0.05em; }
        .ph-feed-badge--win { background: #2e7d32; color: #fff; }
        .ph-feed-badge--loss { background: #c62828; color: #fff; }
        .ph-feed-badge--push { background: #616161; color: #fff; }
        .ph-feed-badge--pending { background: rgba(255,255,255,0.15); color: rgba(255,255,255,0.7); }
        .ph-feed-empty { border: 1px dashed rgba(255,255,255,0.25); border-radius: 12px; padding: 1.25rem; color: rgba(255,255,255,0.5); font-size: 0.9rem; text-align: center; }
        /* Equal width for Win / Loss / Push buttons under each feed card */
        div:has(.ph-feed-card) + div .stButton button { min-width: 5.5rem; box-sizing: border-box; }
        </style>
        """, unsafe_allow_html=True)
        roi_class = "ph-roi--pos" if roi_pct > 0 else ("ph-roi--neg" if roi_pct < 0 else "ph-roi--zero")
        roi_sign = "+" if roi_pct > 0 else ""
        units_sign = "+" if profit_units > 0 else ""
        st.markdown(
            f'<div class="ph-kpi-hero">'
            f'<div class="ph-kpi-record"><span class="ph-w">{int(w)}</span>–<span class="ph-l">{int(l)}</span>–<span class="ph-p">{int(p)}</span></div>'
            f'<div class="ph-kpi-second">'
            f'<span class="ph-roi {roi_class}">{roi_sign}{roi_pct:.1f}% Total ROI</span>'
            f'<span>{units_sign}{profit_units:.2f} units</span>'
            f'<span>{win_rate:.0f}% win rate</span>'
            f'</div>'
            f'<div class="ph-kpi-breakdown">'
            f'<div class="ph-kpi-breakdown-row"><span class="ph-kpi-breakdown-label">Play of the Day</span><span class="ph-kpi-breakdown-record"><span class="ph-w">{int(w_potd)}</span>–<span class="ph-l">{int(l_potd)}</span>–<span class="ph-p">{int(p_potd)}</span></span></div>'
            f'<div class="ph-kpi-breakdown-row"><span class="ph-kpi-breakdown-label">All value plays</span><span class="ph-kpi-breakdown-record"><span class="ph-w">{int(w)}</span>–<span class="ph-l">{int(l)}</span>–<span class="ph-p">{int(p)}</span></span></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # ── Quick-entry results form ──
        _pending = _history[_history["result_clean"].isna() & _history["sport"].astype(str).str.strip().str.upper().str.startswith("NCAAB")]
        if not _pending.empty:
            _pending = _pending.drop_duplicates(subset=["play_id"]).sort_values(["date_generated", "play_id"], ascending=[False, False])

            def _fmt_pending(r):
                d = pd.to_datetime(r["date_generated"]).strftime("%b %d")
                a = str(r.get("away_team", "")).strip()
                h = str(r.get("home_team", "")).strip()
                side = str(r.get("recommended_side", "")).strip()
                line = r.get("spread_or_total")
                line_str = f" {float(line):+.1f}" if line is not None and not pd.isna(line) and line != -999 else ""
                tag = " [T]" if str(r.get("sport", "")).strip() == "NCAAB Tournament" else ""
                return f"{d} | {a} @ {h} | {side}{line_str}{tag}"

            _pending_options = {_fmt_pending(row): int(row["play_id"]) for _, row in _pending.iterrows()}

            with st.expander("Mark Results", expanded=bool(len(_pending_options) > 0)):
                _sel = st.selectbox("Pending picks", list(_pending_options.keys()), label_visibility="collapsed")
                if _sel:
                    _sel_pid = _pending_options[_sel]
                    _sel_row = _pending[_pending["play_id"] == _sel_pid].iloc[0]
                    c_w, c_l, c_p, _ = st.columns([1, 1, 1, 3])
                    with c_w:
                        if st.button("Win", key="qr_w", type="primary"):
                            update_play_result(_sel_pid, "W")
                            _sync_result_to_historical_csv(
                                _sel_row["date_generated"], _sel_row["home_team"], _sel_row["away_team"], "W"
                            )
                            _load_play_history_cached.clear()
                            st.rerun()
                    with c_l:
                        if st.button("Loss", key="qr_l"):
                            update_play_result(_sel_pid, "L")
                            _sync_result_to_historical_csv(
                                _sel_row["date_generated"], _sel_row["home_team"], _sel_row["away_team"], "L"
                            )
                            _load_play_history_cached.clear()
                            st.rerun()
                    with c_p:
                        if st.button("Push", key="qr_p"):
                            update_play_result(_sel_pid, "P")
                            _sync_result_to_historical_csv(
                                _sel_row["date_generated"], _sel_row["home_team"], _sel_row["away_team"], "P"
                            )
                            _load_play_history_cached.clear()
                            st.rerun()

        # Monthly performance chart: W-L by date (green wins, red losses)
        if not resolved.empty and "date_generated" in resolved.columns:
            def _daily_wl(g):
                return pd.Series({
                    "Wins": (g["result_clean"] == "W").sum(),
                    "Losses": (g["result_clean"] == "L").sum(),
                })
            daily = resolved.groupby("date_generated").apply(_daily_wl).reset_index()
            daily["date_generated"] = pd.to_datetime(daily["date_generated"])
            daily = daily.sort_values("date_generated")
            # Limit to last 30 days for readability
            cutoff = date.today() - timedelta(days=30)
            daily = daily[daily["date_generated"].dt.date >= cutoff]
            if not daily.empty:
                st.subheader("Performance by Date")
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=daily["date_generated"].dt.strftime("%b %d"),
                        y=daily["Wins"],
                        name="Wins",
                        marker_color="#4caf50",
                    )
                )
                fig.add_trace(
                    go.Bar(
                        x=daily["date_generated"].dt.strftime("%b %d"),
                        y=daily["Losses"],
                        name="Losses",
                        marker_color="#f44336",
                    )
                )
                fig.update_layout(
                    barmode="group",
                    xaxis_title="Date",
                    yaxis_title="Count",
                    margin=dict(t=20, b=40, l=50, r=20),
                    height=280,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(255,255,255,0.05)",
                    font=dict(color="rgba(255,255,255,0.9)", size=12),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                )
                st.plotly_chart(fig, use_container_width=True)

        def _bet_str(r: pd.Series) -> str:
            bt = str(r.get("bet_type", ""))
            side = str(r.get("recommended_side", "")).strip()
            line = r.get("spread_or_total")
            home, away = str(r.get("home_team", "")).strip(), str(r.get("away_team", "")).strip()
            if line is None or pd.isna(line) or line == -999:
                return f"{bt} · {side}".strip()
            line = float(line)
            if "Over" in bt or "Under" in bt or "total" in bt.lower():
                return f"{bt} · {side} {line:.1f}".strip()
            # Spread: show picked team and that team's line (e.g. Pennsylvania +9.5)
            if "Spread" in bt or "spread" in bt.lower():
                def _ph_side_matches(s: str, team: str) -> bool:
                    if not s or not team:
                        return False
                    return s.lower() == team.lower() or s.lower() in team.lower() or team.lower() in s.lower()
                if _ph_side_matches(side, home):
                    display_line = line
                    team = away if display_line > 0 else home
                elif _ph_side_matches(side, away):
                    team, display_line = away, (-line if line < 0 else line)
                elif side.lower() in ("home", "away"):
                    team = home if side.lower() == "home" else away
                    display_line = (-line if side.lower() == "away" and line < 0 else line)
                else:
                    team, display_line = side, line
                if display_line > 0 and away and _ph_side_matches(team, home):
                    team, display_line = away, display_line
                if display_line < 0 and home and _ph_side_matches(team, away) and side.lower() == "home" and line > 0:
                    team, display_line = home, line
                return f"{bt} · {team} {display_line:+.1f}".strip()
            return f"{bt} · {side} {line:+.1f}".strip()

        def _render_pick_card(row, sport_label, _history_ref):
            """Render a single pick card with result badge and action buttons."""
            result = row.get("result_clean")
            is_tourney = str(row.get("sport", "")).strip() == "NCAAB Tournament"
            card_class = "ph-feed-card--pending"
            if result == "W":
                card_class = "ph-feed-card--win"
            elif result == "L":
                card_class = "ph-feed-card--loss"
            elif result == "P":
                card_class = "ph-feed-card--push"
            matchup = f"{row.get('away_team', '')} @ {row.get('home_team', '')}"
            bet_str = _bet_str(row)
            edge_str = f"{float(row.get('my_edge_pct', 0)):.1f}%"
            odds_str = format_american(row.get("market_odds_at_time", 0))
            tag_html = ' <span style="background:#ff6f00;color:#fff;font-size:0.65rem;padding:2px 6px;border-radius:4px;font-weight:700;vertical-align:middle;margin-left:4px;">MARCH MADNESS</span>' if is_tourney else ""
            if result == "W":
                badge = '<span class="ph-feed-badge ph-feed-badge--win">WIN</span>'
            elif result == "L":
                badge = '<span class="ph-feed-badge ph-feed-badge--loss">LOSS</span>'
            elif result == "P":
                badge = '<span class="ph-feed-badge ph-feed-badge--push">PUSH</span>'
            else:
                badge = '<span class="ph-feed-badge ph-feed-badge--pending">—</span>'
            d_display = pd.to_datetime(row.get("date_generated", "")).strftime("%a, %b %d") if row.get("date_generated") else ""
            st.markdown(
                f'<div class="ph-feed-card {card_class}">'
                f'<div class="ph-feed-date">{_html_escape(d_display)} · {sport_label}{tag_html}</div>'
                f'<div class="ph-feed-matchup">{_html_escape(matchup)}</div>'
                f'<div class="ph-feed-meta">{_html_escape(bet_str)} · Edge {edge_str} · Odds {odds_str}</div>'
                f'<div>{badge}</div></div>',
                unsafe_allow_html=True,
            )
            mk = row.get("_matchup_key", "")
            pids_in_group = _history_ref.loc[_history_ref["_matchup_key"] == mk, "play_id"].astype(int).tolist() if mk else [int(row.get("play_id", 0))]
            if result is None or pd.isna(result):
                c1, c2, c3, _gap, c4, _ = st.columns([1, 1, 1, 1.5, 0.7, 1.8])
                with c1:
                    if st.button("Win", key=f"ph_w_{row.get('play_id')}_{sport_label}", help="Mark win"):
                        for _pid in pids_in_group:
                            update_play_result(_pid, "W")
                        _load_play_history_cached.clear()
                        st.rerun()
                with c2:
                    if st.button("Loss", key=f"ph_l_{row.get('play_id')}_{sport_label}", help="Mark loss"):
                        for _pid in pids_in_group:
                            update_play_result(_pid, "L")
                        _load_play_history_cached.clear()
                        st.rerun()
                with c3:
                    if st.button("Push", key=f"ph_p_{row.get('play_id')}_{sport_label}", help="Mark push"):
                        for _pid in pids_in_group:
                            update_play_result(_pid, "P")
                        _load_play_history_cached.clear()
                        st.rerun()
                with c4:
                    if st.button("Del", key=f"ph_del_{row.get('play_id')}_{sport_label}", help="Delete this play"):
                        first_row_data = None
                        for _pid in pids_in_group:
                            deleted = delete_play(_pid)
                            if deleted and first_row_data is None:
                                first_row_data = deleted
                        if first_row_data:
                            _remove_play_from_historical_csv(
                                first_row_data["date_generated"],
                                first_row_data["home_team"],
                                first_row_data["away_team"],
                            )
                        _load_play_history_cached.clear()
                        st.rerun()
            else:
                _, c_del, _ = st.columns([5, 1, 0.5])
                with c_del:
                    if st.button("Delete", key=f"ph_del_{row.get('play_id')}_{sport_label}", help="Remove this play and update record"):
                        first_row_data = None
                        for _pid in pids_in_group:
                            deleted = delete_play(_pid)
                            if deleted and first_row_data is None:
                                first_row_data = deleted
                        if first_row_data:
                            _remove_play_from_historical_csv(
                                first_row_data["date_generated"],
                                first_row_data["home_team"],
                                first_row_data["away_team"],
                            )
                        _load_play_history_cached.clear()
                        st.rerun()

        dates_desc = sorted(top_per_day["date_generated"].unique(), reverse=True)
        for d in dates_desc:
            day_rows = top_per_day[top_per_day["date_generated"] == d]
            _day_sport = day_rows["sport"].astype(str).str.strip()
            pick1_row = day_rows[_day_sport == "NCAAB Pick 1"]
            pick2_row = day_rows[_day_sport == "NCAAB Pick 2"]
            tourney_rows = day_rows[_day_sport == "NCAAB Tournament"]

            if not pick1_row.empty or not pick2_row.empty:
                col_p1, col_p2 = st.columns(2)
                for col, sport_label, row_df in [(col_p1, "NCAAB Pick 1", pick1_row), (col_p2, "NCAAB Pick 2", pick2_row)]:
                    with col:
                        if row_df.empty:
                            date_display = pd.to_datetime(d).strftime("%a, %b %d") if hasattr(pd.to_datetime(d), "strftime") else str(d)
                            st.markdown(f'<div class="ph-feed-empty">{sport_label}: No pick</div>', unsafe_allow_html=True)
                            continue
                        _render_pick_card(row_df.iloc[0], sport_label, _history)

            if not tourney_rows.empty:
                tourney_list = tourney_rows.sort_values("my_edge_pct", ascending=False).reset_index(drop=True)
                for i in range(0, len(tourney_list), 2):
                    cols = st.columns(2)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(tourney_list):
                            with col:
                                _render_pick_card(tourney_list.iloc[idx], "NCAA Tournament", _history)
    else:
        st.info("No archived plays yet. Plays are saved when you load the dashboard (and at 9am ET). Open the app before games to capture today's top picks.")

with tab_manual_odds:
    st.subheader("Manual Odds")
    st.caption("Paste lines from any sportsbook (DraftKings, FanDuel, etc.) when free APIs don't have the game. Entries are saved to the cache and the pipeline picks them up as an extra odds source alongside ESPN and Rundown.")
    _app_root = Path(__file__).resolve().parent
    _manual_cache_dir = _app_root / "data" / "cache"
    _manual_odds_path = _manual_cache_dir / "manual_odds.json"

    with st.form("manual_odds_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            _home = st.text_input("Home team", placeholder="e.g. Alabama")
        with c2:
            _away = st.text_input("Away team", placeholder="e.g. Tennessee")
        _sport = st.selectbox("Sport", options=["NCAAB", "NBA"], index=0)
        c3, c4, c5 = st.columns(3)
        with c3:
            _spread = st.number_input("Spread (home)", value=None, placeholder="-7.5", step=0.5, format="%.1f")
        with c4:
            _ml_home = st.number_input("Moneyline (home)", value=None, placeholder="-300", step=1, format="%d")
        with c5:
            _ml_away = st.number_input("Moneyline (away)", value=None, placeholder="250", step=1, format="%d")
        _total = st.number_input("Total", value=None, placeholder="145.5", step=0.5, format="%.1f")
        submitted = st.form_submit_button("Save to cache")
        if submitted:
            if not (_home and _away):
                st.error("Home and away team are required.")
            else:
                _manual_cache_dir.mkdir(parents=True, exist_ok=True)
                existing = []
                if _manual_odds_path.exists():
                    try:
                        existing = json.loads(_manual_odds_path.read_text())
                        if not isinstance(existing, list):
                            existing = []
                    except (json.JSONDecodeError, OSError):
                        existing = []
                def _num_or_none(x, cast=int):
                    if x is None:
                        return None
                    try:
                        return cast(x)
                    except (TypeError, ValueError):
                        return None
                entry = {
                    "home_team": _home.strip(),
                    "away_team": _away.strip(),
                    "sport": _sport,
                    "spread": _num_or_none(_spread, float),
                    "moneyline_home": _num_or_none(_ml_home, int),
                    "moneyline_away": _num_or_none(_ml_away, int),
                    "total": _num_or_none(_total, float),
                }
                existing.append(entry)
                _manual_odds_path.write_text(json.dumps(existing, indent=2))
                st.success(f"Saved **{_away} @ {_home}** ({_sport}). Run **Refresh** on the Overview tab to include this game in value plays.")

    if _manual_odds_path.exists():
        try:
            _list = json.loads(_manual_odds_path.read_text())
            if isinstance(_list, list) and _list:
                st.caption(f"**{len(_list)}** manual game(s) in cache (pipeline uses today's date; duplicate matchups overwrite by last entry).")
        except (json.JSONDecodeError, OSError):
            pass

with tab_game_lookup:
    st.subheader("Game Lookup")
    st.caption("Search by team name to find today's matchup (from value plays or historical slate), view a detailed matchup card, BARTHAG comparison, and head-to-head history.")
    st.caption(f"**Search fields:** Event, home_team, away_team. **Match:** exact/substring or fuzzy score ≥ {GAME_LOOKUP_FUZZY_THRESHOLD}.")
    team_stats_2026 = _load_team_stats_2026()
    # BARTHAG rank: 1 = best (highest BARTHAG) in team_stats_2026
    if not team_stats_2026.empty and "BARTHAG" in team_stats_2026.columns:
        _barthag_sorted = team_stats_2026.sort_values("BARTHAG", ascending=False).reset_index(drop=True)
        _barthag_sorted["_barthag_rank"] = _barthag_sorted.index + 1
        _rank_map = _barthag_sorted.set_index("TEAM")["_barthag_rank"].to_dict()
    else:
        _rank_map = {}
    search_query = st.text_input("Team name", placeholder="e.g. Duke or Michigan", key="game_lookup_team")
    _today_odds_df = _load_latest_odds_slate()
    found_game = None
    if search_query and search_query.strip():
        found_game = _find_game_for_team(search_query.strip(), value_plays_df, _historical_picks_df, _today_odds_df)
    if found_game:
        source, game = found_game
        home_team = game["home_team"]
        away_team = game["away_team"]
        home_stats = _find_team_in_stats(team_stats_2026, home_team)
        away_stats = _find_team_in_stats(team_stats_2026, away_team)
        home_barthag = None
        away_barthag = None
        home_adjoe = None
        home_adjde = None
        away_adjoe = None
        away_adjde = None
        home_rank = None
        away_rank = None
        if home_stats is not None:
            home_barthag = home_stats.get("BARTHAG")
            home_adjoe = home_stats.get("ADJOE")
            home_adjde = home_stats.get("ADJDE")
            home_rank = _rank_map.get(str(home_stats.get("TEAM", "")).strip())
        if away_stats is not None:
            away_barthag = away_stats.get("BARTHAG")
            away_adjoe = away_stats.get("ADJOE")
            away_adjde = away_stats.get("ADJDE")
            away_rank = _rank_map.get(str(away_stats.get("TEAM", "")).strip())
        if home_barthag is None and game.get("home_barthag") is not None:
            home_barthag = game["home_barthag"]
        if away_barthag is None and game.get("away_barthag") is not None:
            away_barthag = game["away_barthag"]
        def _num_fmt(x, decimals=2):
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return "—"
            try:
                return f"{float(x):.{decimals}f}"
            except (TypeError, ValueError):
                return "—"
        def _record_str(stats_row: Optional[pd.Series]) -> str:
            if stats_row is None or not isinstance(stats_row, pd.Series):
                return ""
            try:
                g = stats_row.get("G")
                w = stats_row.get("W")
                if g is not None and w is not None and not (pd.isna(g) or pd.isna(w)):
                    gi, wi = int(g), int(w)
                    return f" ({wi}-{gi - wi})"
            except (TypeError, ValueError):
                pass
            return ""
        away_record = _record_str(away_stats)
        home_record = _record_str(home_stats)
        st.markdown(f"**Matchup:** {away_team} @ {home_team}")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("**Away**")
            st.markdown(f"**{away_team}**{away_record}")
            st.caption(f"BARTHAG rank: {away_rank if away_rank is not None else '—'}")
            st.caption(f"BARTHAG: {_num_fmt(away_barthag, 3)}")
            st.caption(f"ADJOE: {_num_fmt(away_adjoe)} | ADJDE: {_num_fmt(away_adjde)}")
        with col_b:
            st.markdown("**vs**")
            spread = game.get("market_spread")
            st.caption(f"Market spread (home): {_num_fmt(spread, 1) if spread is not None else '—'}")
            # Always recalculate pred_margin and edge from BARTHAG when available (ignore stale cache/CSV)
            pred = None
            if home_barthag is not None and away_barthag is not None:
                try:
                    hb = float(home_barthag)
                    ab = float(away_barthag)
                    if not (pd.isna(hb) or pd.isna(ab)):
                        pred = (hb - ab) * 50
                except (TypeError, ValueError):
                    pass
            if pred is None:
                pred = game.get("pred_margin")
            st.caption(f"Model predicted margin: {_num_fmt(pred, 1) if pred is not None else '—'}")
            if pred is not None and spread is not None:
                abs_spread = abs(spread)
                edge = (pred - abs_spread) if spread < 0 else (pred + abs_spread)
            else:
                edge = game.get("edge_points")
            st.caption(f"Edge (pts): {_num_fmt(edge, 1) if edge is not None else '—'}")
            st.caption(f"Confidence: **{game.get('confidence_tier', '—')}**")
            st.caption(f"Recommended pick: **{game.get('recommended_pick', '—')}**")
        with col_c:
            st.markdown("**Home**")
            st.markdown(f"**{home_team}**{home_record}")
            st.caption(f"BARTHAG rank: {home_rank if home_rank is not None else '—'}")
            st.caption(f"BARTHAG: {_num_fmt(home_barthag, 3)}")
            st.caption(f"ADJOE: {_num_fmt(home_adjoe)} | ADJDE: {_num_fmt(home_adjde)}")
        # Model Verdict: green if we like home, orange if we like away; pick + reasoning in large text
        pick_team = game.get("recommended_pick") or ""
        like_home = pick_team.strip().lower() == home_team.strip().lower() or (home_team and pick_team and home_team.strip().lower() in pick_team.strip().lower())
        verdict_bg = "#1b5e20" if like_home else "#e65100"
        verdict_label = "Home" if like_home else "Away"
        # Use human-readable reason when pred and spread available
        pick_side = "Home" if (pick_team and pick_team.strip().lower() == home_team.strip().lower()) else "Away"
        if pred is not None and spread is not None:
            reason_text = _human_readable_potd_reason(home_team, away_team, pred, spread, pick_side)
        else:
            reason_text = (game.get("reason") or "Model favors this side.").strip()
        reason_escaped = reason_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        pick_escaped = (pick_team or "—").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        verdict_html = (
            f'<div style="background:{verdict_bg}; color:white; padding:1rem 1.25rem; border-radius:8px; margin:1rem 0;">'
            f'<div style="font-size:0.85rem; opacity:0.9; margin-bottom:0.35rem;">Model Verdict — {verdict_label}</div>'
            f'<div style="font-size:1.5rem; font-weight:700; margin-bottom:0.5rem;">{pick_escaped}</div>'
            f'<div style="font-size:1.05rem; line-height:1.45; opacity:0.95;">{reason_escaped}</div>'
            f'</div>'
        )
        st.markdown(verdict_html, unsafe_allow_html=True)
        h_b = float(home_barthag) if home_barthag is not None and not (isinstance(home_barthag, float) and pd.isna(home_barthag)) else 0.0
        a_b = float(away_barthag) if away_barthag is not None and not (isinstance(away_barthag, float) and pd.isna(away_barthag)) else 0.0
        if (h_b > 0 or a_b > 0) and (home_team or away_team):
            away_label = f"{away_team} (Away)"
            home_label = f"{home_team} (Home)"
            x_away = round(a_b * 100, 1)
            x_home = round(h_b * 100, 1)
            fig = go.Figure()
            # Use numeric y (0, 1) so both bars always render; label via ticktext
            fig.add_trace(go.Bar(
                y=[0, 1],
                x=[x_away, x_home],
                orientation="h",
                marker_color=["steelblue", "darkorange"],
                text=[f"{away_team} ({x_away})", f"{home_team} ({x_home})"],
                textposition="outside",
            ))
            fig.update_layout(
                title="BARTHAG comparison (longer = stronger team)",
                xaxis_title="BARTHAG (%)",
                height=140,
                margin=dict(l=100, t=40, b=40),
                xaxis=dict(range=[0, 105]),
                yaxis=dict(
                    tickvals=[0, 1],
                    ticktext=[away_label, home_label],
                    range=[-0.6, 1.6],
                ),
            )
            st.plotly_chart(fig, use_container_width=True)
        # KenPom Stats comparison table: Home and Away as row headers, ADJOE, ADJDE, BARTHAG, ADJ_T as columns
        st.markdown("**KenPom Stats**")
        kp_cols = ["ADJOE", "ADJDE", "BARTHAG", "ADJ_T"]
        kp_data = []
        for label, stats_row in [("Away", away_stats), ("Home", home_stats)]:
            if stats_row is not None:
                row = {"": f"{away_team} (Away)" if label == "Away" else f"{home_team} (Home)"}
                for c in kp_cols:
                    if c in stats_row.index:
                        v = stats_row.get(c)
                        if v is not None and not (isinstance(v, float) and pd.isna(v)):
                            row[c] = f"{float(v):.2f}"
                        else:
                            row[c] = "—"
                    else:
                        row[c] = "—"
                kp_data.append(row)
        if kp_data:
            kp_df = pd.DataFrame(kp_data)
            st.dataframe(kp_df.set_index(""), use_container_width=True, hide_index=True)
        st.markdown("---")
        st.markdown("**Head to Head**")
        h2h = _get_head_to_head(home_team, away_team)
        if not h2h.empty:
            st.caption("Meetings this season (from espn.db):")
            st.dataframe(h2h, use_container_width=True, hide_index=True)
        else:
            st.caption("No prior meetings this season in our database, or espn.db not available.")
    else:
        if search_query and search_query.strip():
            team_row = _find_team_in_stats(team_stats_2026, search_query.strip())
            if team_row is not None:
                st.info(f"No game today for **{search_query.strip()}**. Showing season stats from team_stats_2026.")
                st.markdown(f"**{team_row.get('TEAM', '—')}** (season {team_row.get('season', '—')})")
                disp_cols = ["CONF", "G", "W", "ADJOE", "ADJDE", "BARTHAG", "ADJ_T", "WAB"]
                show = [c for c in disp_cols if c in team_row.index]
                if show:
                    st.dataframe(pd.DataFrame([team_row[show]]), use_container_width=True, hide_index=True)
                rank = _rank_map.get(str(team_row.get("TEAM", "")).strip())
                if rank is not None:
                    st.caption(f"BARTHAG rank: {rank}")
            else:
                st.warning(f"No game today and no team matching **{search_query.strip()}** in team_stats_2026.")
        else:
            st.caption("Enter a team name above to find today's game or view season stats.")

with tab_nba:
    st.subheader("NBA — Coming Soon")
    st.info("NBA value plays and totals will return here. Backend is still in place; we're focused on NCAAB for now.")
    # NBA tab content kept below (commented) for easy re-enable:
    # nba_value = value_plays_df[value_plays_df["League"] == "NBA"] if not value_plays_df.empty and "League" in value_plays_df.columns else pd.DataFrame()
    # ... (value plays table, totals tracker)
    nba_totals_df = pd.DataFrame()
    if False and not live_odds_df.empty and "sport_key" in live_odds_df.columns:
        totals_only = live_odds_df[
            (live_odds_df["sport_key"] == BASKETBALL_NBA) &
            (live_odds_df["market_type"] == "totals") &
            (live_odds_df["point"].notna())
        ]
        if not totals_only.empty:
            pace_stats = get_nba_team_pace_stats()
            b2b_teams = _get_b2b_teams_cached(
                date.today().isoformat(),
                refresh_key=st.session_state.get("odds_refresh_key", 0),
            )
            seen = set()
            rows = []
            for _, r in totals_only.iterrows():
                eid = r.get("event_id", "")
                if eid in seen:
                    continue
                seen.add(eid)
                home_team = r.get("home_team", "")
                away_team = r.get("away_team", "")
                market_total = float(r.get("point", 0))
                event_name = r.get("event_name", f"{away_team} @ {home_team}")
                predicted_total = predict_nba_total(home_team, away_team, pace_stats, b2b_teams=b2b_teams)
                edge = predicted_total - market_total
                value = get_totals_value(predicted_total, market_total)
                value_label = "Value Over" if value == "over" else ("Value Under" if value == "under" else "—")
                rows.append({
                    "Game": event_name,
                    "Market Total (O/U Line)": round(market_total, 1),
                    "My Projected Total": round(predicted_total, 1),
                    "Edge": round(edge, 1),
                    "Value": value_label,
                })
            nba_totals_df = pd.DataFrame(rows)
    # Original NBA totals/value-plays UI kept below for re-enable (run when LIVE_ODDS_SPORT_KEYS includes NBA again)
    if False and not nba_totals_df.empty:
        def _edge_color(edge_series: pd.Series) -> list[str]:
            styles = []
            for idx in edge_series.index:
                v = nba_totals_df.loc[idx, "Value"]
                if v == "Value Over":
                    styles.append("background-color: rgba(46, 125, 50, 0.45); color: #c8e6c9;")
                elif v == "Value Under":
                    styles.append("background-color: rgba(198, 40, 40, 0.45); color: #ffcdd2;")
                else:
                    styles.append("")
            return styles
        display_totals = nba_totals_df.copy()
        styled_totals = display_totals.style.apply(_edge_color, subset=["Edge"])
        st.dataframe(
            styled_totals,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Game": st.column_config.TextColumn("Game", width="medium"),
                "Market Total (O/U Line)": st.column_config.NumberColumn("Market Total (O/U Line)", format="%.1f", width="small"),
                "My Projected Total": st.column_config.NumberColumn("My Projected Total", format="%.1f", width="small"),
                "Edge": st.column_config.NumberColumn("Edge", format="%+.1f", width="small"),
                "Value": st.column_config.TextColumn("Value", width="small"),
            },
        )

