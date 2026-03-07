"""
Bobby Bottle's Betting Model - Streamlit Dashboard
Daily picks sheet: Play of the Day (NBA + NCAAB) and All Value Plays. Dark theme, NCAAB/NBA tabs.
Live odds from The Odds API; stakes from fractional Kelly.
"""

import json
import os
import re
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
from engine.play_history import archive_value_plays, load_play_history, update_play_result
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


# Human-readable market labels (replaces h2h, spreads, totals in UI)
MARKET_LABELS = {"h2h": "Winner", "spreads": "Spread", "totals": "Over/Under"}

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
# Archive: only save plays with edge >= this and cap total per day
ARCHIVE_MIN_EDGE_PCT = 6.0
ARCHIVE_MAX_PLAYS_PER_DAY = 10
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
    Select the Play of the Day for NBA and NCAAB.

    Criteria: highest edge % among today's plays with edge > min_edge_pct.
    Tie-breaks (in order): (1) more bookmakers agreeing on the line, (2) model prob
    furthest from implied (largest |model_prob - implied_prob|), (3) no major injury flags.

    Returns {"NBA": pick_dict or None, "NCAAB": pick_dict or None}. Each pick_dict has
    all fields needed for the Play of the Day card (Event, Selection, Market, Odds,
    Value (%), point, Recommended Stake, Start Time, League, etc.).
    """
    result: dict[str, Optional[dict]] = {"NBA": None, "NCAAB": None}
    if value_plays_df.empty or "League" not in value_plays_df.columns:
        return result

    # Only consider plays above minimum confidence
    eligible = value_plays_df[value_plays_df["Value (%)"] > min_edge_pct].copy()
    if eligible.empty:
        return result

    # Attach bookmaker count from live odds (same event/market/selection/point)
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

    # Model prob gap (tie-break 2): prefer larger |model_prob - implied_prob|
    if "model_prob" in eligible.columns and "implied_prob" in eligible.columns:
        eligible["model_prob_gap"] = (eligible["model_prob"] - eligible["implied_prob"]).abs()
    else:
        eligible["model_prob_gap"] = 0.0

    # Injury flag (tie-break 3): prefer no major injury
    def _has_injury_flag(row: pd.Series) -> bool:
        if row.get("top5_out_or_doubtful_home") or row.get("top5_out_or_doubtful_away"):
            return True
        score = row.get("injury_impact_score")
        if pd.isna(score):
            return False
        return float(score) > 0.5

    eligible["has_injury_flag"] = eligible.apply(_has_injury_flag, axis=1)

    # Large-spread penalty: for spreads with |point| > 14, use adjusted edge (30% penalty) for sorting
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

    for league in ("NBA", "NCAAB"):
        league_plays = eligible[eligible["League"] == league]
        if league_plays.empty:
            continue
        # Sort: adjusted edge desc (large spreads penalized), then bookmaker_count, model_prob_gap, has_injury_flag
        sorted_df = league_plays.sort_values(
            by=["_adjusted_edge", "bookmaker_count", "model_prob_gap", "has_injury_flag"],
            ascending=[False, False, False, True],
        )
        top = sorted_df.iloc[0]
        # After merge, point may appear as point_x (left) or point
        point_val = top.get("point_x") if "point_x" in top.index else top.get("point")
        # Odds: store None when missing/NaN (e.g. API returned spread but no moneyline) so card shows "—"
        odds_raw = top.get("Odds", 0)
        if odds_raw is None or pd.isna(odds_raw):
            odds_for_pick = None
        else:
            try:
                odds_for_pick = int(round(float(odds_raw)))
            except (TypeError, ValueError):
                odds_for_pick = None
        commence_time = top.get("commence_time") or top.get("commence_time_x") or ""
        pick: dict = {
            "League": league,
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
        }
        result[league] = pick

    return result


# #region agent log
def _debug_log(location: str, message: str, data: dict, hypothesis_id: str = "") -> None:
    try:
        payload = {"sessionId": "a60dbe", "location": location, "message": message, "data": data, "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000)}
        if hypothesis_id:
            payload["hypothesisId"] = hypothesis_id
        with open("/Users/robertseipel/Desktop/bettingsim/.cursor/debug-a60dbe.log", "a") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass
# #endregion


def _get_yesterday_potd_results() -> list[dict]:
    """
    Load yesterday's play history and return the two POTD picks (one per sport, max edge that day).
    Each item: {"sport": "NBA"|"NCAAB", "side": "Team Name", "result": "W"|"L"|"P"|None}.
    """
    yesterday = date.today() - timedelta(days=1)
    hist = load_play_history(from_date=yesterday, to_date=yesterday)
    if hist.empty or "my_edge_pct" not in hist.columns:
        return []
    hist = hist.copy()
    hist["result_clean"] = hist["result"].apply(
        lambda x: str(x).strip().upper() if x is not None and not pd.isna(x) else None
    )
    # One top play per (date, sport) by edge
    top = hist.loc[hist.groupby(["date_generated", "sport"])["my_edge_pct"].idxmax()].reset_index(drop=True)
    # Order: NBA then NCAAB (match Overview columns)
    top = top.sort_values("sport", ascending=True)  # NBA then NCAAB
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
    # #region agent log
    ncaab_all = hist[hist["sport"].astype(str).str.strip().str.upper() == "NCAAB"] if not hist.empty else pd.DataFrame()
    _debug_log("app.py:_get_yesterday_potd_results", "all NCAAB yesterday", {"n_rows": len(ncaab_all), "results": ncaab_all["result"].tolist() if not ncaab_all.empty else [], "sides": ncaab_all["recommended_side"].tolist() if not ncaab_all.empty else []}, "B")
    # #endregion
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
    league = str(row.get("League", ""))
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
            league=league.strip().lower(),
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
            feature_row, market, home_team=home, away_team=away, top_k=4, league=league.strip().lower()
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


def _value_play_reasoning(row: pd.Series) -> str:
    """
    One or two sentence plain-English reasoning from model feature context: rest/B2B, home court, market type, injury.
    Uses home_team, away_team, is_home_b2b, is_away_b2b, Market, Selection, and optional injury flags.
    """
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
    """One All Value Plays card: matchup, bet type, edge %, odds, confidence bar, reasoning."""
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
            <span class="vp-edge">{edge_pct:.1f}% edge</span>
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
) -> str:
    """Return HTML for one Play of the Day card. accent: 'blue' | 'orange' | 'grey'. Grey when no play."""
    if row is None or (isinstance(row, pd.Series) and (row.empty or len(row) == 0)):
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
    away, home = _parse_event_teams(str(r.get("Event", "")))
    away_abbrev = _team_abbrev(away, league)
    home_abbrev = _team_abbrev(home, league)
    matchup_html = f"{_html_escape(away)}"
    if away_abbrev:
        matchup_html += f' <span class="potd-abbrev">({away_abbrev})</span>'
    matchup_html += f' <span class="potd-vs">@</span> {_html_escape(home)}'
    if home_abbrev:
        matchup_html += f' <span class="potd-abbrev">({home_abbrev})</span>'
    tipoff = format_start_time(str(r.get("commence_time", "") or ""))
    badge_text = _potd_badge_text(r)
    reason = _potd_reason(r, feature_matrix=feature_matrix, b2b_teams=b2b_teams)
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
    if league == "NCAAB":
        march_badges = _march_context_badges(r)
        if march_badges and march_badges != "—":
            potd_march = f'<div class="potd-march-context">{_html_escape(march_badges)}</div>'
    potd_tournament_badge = ""
    if march_madness_mode and league == "NCAAB" and _is_top_seed_play(r):
        potd_tournament_badge = '<div class="potd-tournament-context">Tournament Context</div>'
    html = f"""
    <div class="potd-card potd-card--{accent}">
        <div class="potd-league">{_html_escape(league)}</div>
        <div class="potd-tipoff">Tip-off: {_html_escape(tipoff)}</div>
        <div class="potd-matchup">{matchup_html}</div>
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
    """Background: fetch NCAAB odds snapshot, merge closing into games, write snapshot log timestamp."""
    try:
        subprocess.run(
            [sys.executable, str(_APP_ROOT / "scripts" / "fetch_ncaab_odds_snapshot.py")],
            cwd=str(_APP_ROOT),
            capture_output=True,
            timeout=120,
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


if "scheduler_started" not in st.session_state:
    _start_scheduler_once()
    st.session_state["scheduler_started"] = True

if "startup_snapshot_started" not in st.session_state:
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


# Sport keys for live Best Value Plays (The Odds API)
LIVE_ODDS_SPORT_KEYS = [BASKETBALL_NBA, BASKETBALL_NCAAB]


# Edge threshold (epsilon): only show Value % in [3%, 15%); >= 15% flagged as Potential Data Error
EV_EPSILON_MIN_PCT = 3.0
EV_EPSILON_MAX_PCT = 15.0

# Minimum number of bookmakers offering a line for an outcome to be eligible as a value play
MIN_BOOKMAKERS_VALUE_PLAY = 3


def _aggregate_odds_best_line_avg_implied(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate odds across bookmakers: one row per (event, market, selection, point).
    - best_odds: best available line (max American odds = best price for bettor).
    - avg_implied_prob: mean of implied probability (no-vig) across all bookmakers.
    - bookmaker_count: number of books offering this line (for min-threshold filter).
    Drops rows with missing/NaN odds so we never surface moneyline when API returned spread but no moneyline.
    """
    if odds_df.empty or "odds" not in odds_df.columns:
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
) -> tuple[pd.DataFrame, int]:
    """
    In-house line (power ratings) vs market; key-number adjustment; fractional Kelly rounded to half-units (1–3%).
    Only includes Value % in [min_ev_pct, max_ev_pct). Excludes odds > +500 unless include_high_risk.
    """
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
    for _, r in odds_df.iterrows():
        # When using aggregated odds, require at least MIN_BOOKMAKERS_VALUE_PLAY books on this line
        if "bookmaker_count" in r.index and r.get("bookmaker_count") is not None and not pd.isna(r.get("bookmaker_count")):
            if int(r.get("bookmaker_count", 0)) < MIN_BOOKMAKERS_VALUE_PLAY:
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
        feature_row = get_feature_row_for_game(
            feature_matrix, home_team, away_team, league_lookup, game_date=game_date_str
        ) if feature_matrix is not None and not feature_matrix.empty else None

        if market_type == "totals" and point is not None:
            market_total = float(point)
            in_house_total = predict_nba_total(home_team, away_team, pace_stats, b2b_teams=b2b_teams)
            prefer_over = "Over" in selection or "over" in selection.lower()
            fallback = model_prob_from_in_house_total(in_house_total, market_total, prefer_over)
            model_prob, consensus_ok = consensus_totals(
                feature_row, market_total, prefer_over, in_house_total, league, fallback
            )
            if not consensus_ok:
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
            if not consensus_ok:
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
        if ev_pct < min_ev_pct:
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
st.sidebar.caption("---")
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


@st.cache_data(ttl=ODDS_CACHE_TTL_SECONDS)
def _fetch_live_odds_cached(commence_date_iso: str, refresh_key: int = 0) -> pd.DataFrame:
    """Fetch live odds from The Odds API. Cached 15 min; refresh_key forces refetch when changed.
    Pass today_et.isoformat() as commence_date_iso so cache key is ET date; engine uses ET for 'today' filter."""
    api_key = _get_odds_api_key()
    if not (api_key or "").strip():
        return pd.DataFrame()
    return get_live_odds(
        api_key=api_key.strip(),
        sport_keys=LIVE_ODDS_SPORT_KEYS,
        commence_on_date=None,
    )


@st.cache_data(ttl=ODDS_CACHE_TTL_SECONDS)
def _get_b2b_teams_cached(as_of_date_iso: str, refresh_key: int = 0) -> frozenset:
    """Teams that played last night (B2B). Cached 15 min with same key as odds."""
    api_key = _get_odds_api_key()
    if not (api_key or "").strip():
        return frozenset()
    return frozenset(get_nba_teams_back_to_back(api_key.strip(), date.fromisoformat(as_of_date_iso)))


if "odds_refresh_key" not in st.session_state:
    st.session_state["odds_refresh_key"] = 0

b2b_teams: set[str] = set()
if (odds_api_key or "").strip():
    try:
        today_et = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
        live_odds_df = _fetch_live_odds_cached(
            today_et,
            refresh_key=st.session_state["odds_refresh_key"],
        )
        b2b_teams = _get_b2b_teams_cached(
            datetime.now(ZoneInfo("America/New_York")).date().isoformat(),
            refresh_key=st.session_state.get("odds_refresh_key", 0),
        )
        feature_matrix_inference = load_feature_matrix_for_inference(league=None)
        # Aggregate: best line per outcome + average implied prob across bookmakers
        aggregated_odds_df = _aggregate_odds_best_line_avg_implied(live_odds_df)
        value_plays_df, value_plays_flagged_count = _live_odds_to_value_plays(
            aggregated_odds_df if not aggregated_odds_df.empty else live_odds_df,
            bankroll=BANKROLL_FOR_STAKES,
            kelly_frac=kelly_frac_val,
            min_ev_pct=EV_EPSILON_MIN_PCT,
            max_ev_pct=EV_EPSILON_MAX_PCT,
            include_high_risk=include_high_risk_odds,
            pace_stats=get_nba_team_pace_stats(),
            b2b_teams=b2b_teams,
            as_of_date=date.today(),
            feature_matrix=feature_matrix_inference,
        )
        value_plays_df = add_injury_alerts_to_value_plays(value_plays_df, "Basketball")
        # Add confidence_tier and reasoning_summary for play_history archive
        if not value_plays_df.empty:
            def _confidence_tier(row: pd.Series) -> str:
                edge = float(row.get("Value (%)", 0))
                return "High" if edge >= POTD_HIGH_CONFIDENCE_EDGE_PCT else "Medium"
            value_plays_df = value_plays_df.copy()
            value_plays_df["confidence_tier"] = value_plays_df.apply(_confidence_tier, axis=1)
            value_plays_df["reasoning_summary"] = value_plays_df.apply(
                lambda r: _potd_reason(r, feature_matrix=feature_matrix_inference, b2b_teams=b2b_teams), axis=1
            )
        # Archive today's plays to play_history (stricter: min 6% edge, max 10 per day by edge)
        try:
            to_archive = value_plays_df[value_plays_df["Value (%)"] >= ARCHIVE_MIN_EDGE_PCT].copy()
            to_archive = to_archive.sort_values("Value (%)", ascending=False).head(ARCHIVE_MAX_PLAYS_PER_DAY)
            archive_value_plays(to_archive, as_of_date=date.today())
        except Exception:
            pass
        # Record each recommended play for CLV (odds at recommendation; closing filled later)
        try:
            clv_record_recommendations(value_plays_df)
            clv_update_closing_odds()
        except Exception:
            pass
        # Add injury_impact_score and top5-out flags for NBA (feature matrix)
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
        # March Madness mode: keep only NCAAB plays where both teams are tournament-eligible (in ncaab_seeds.csv)
        if march_madness_mode and not value_plays_df.empty and "League" in value_plays_df.columns:
            tournament_eligible = _load_tournament_eligible_teams()
            if tournament_eligible:
                ncaab_mask = value_plays_df["League"].astype(str).str.strip().str.upper() == "NCAAB"
                home_ok = value_plays_df["home_team"].astype(str).str.strip().str.lower().isin(tournament_eligible)
                away_ok = value_plays_df["away_team"].astype(str).str.strip().str.lower().isin(tournament_eligible)
                keep = ~ncaab_mask | (ncaab_mask & home_ok & away_ok)
                value_plays_df = value_plays_df.loc[keep].copy()
    except Exception:
        value_plays_flagged_count = 0
        live_odds_df = pd.DataFrame(
            columns=[
                "sport_key", "league", "event_id", "commence_time", "home_team", "away_team",
                "event_name", "market_type", "selection", "point", "odds",
            ]
        )
        value_plays_df = pd.DataFrame(
            columns=["League", "Event", "Selection", "Market", "Odds", "Value (%)", "Recommended Stake", "Injury Alert", "Start Time"]
        )
else:
    value_plays_flagged_count = 0
    live_odds_df = pd.DataFrame(
        columns=[
            "sport_key", "league", "event_id", "commence_time", "home_team", "away_team",
            "event_name", "market_type", "selection", "point", "odds",
        ]
    )
    value_plays_df = pd.DataFrame(
        columns=["League", "Event", "Selection", "Market", "Odds", "Value (%)", "Recommended Stake", "Injury Alert", "Start Time"]
    )

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
    date_str = _today_str()
    if value_plays_df.empty or "League" not in value_plays_df.columns or "Value (%)" not in value_plays_df.columns:
        return {
            "date_str": date_str,
            "total_plays": 0,
            "n_nba": 0,
            "n_ncaab": 0,
            "top_edge_label": "—",
        }
    best_per_game = (
        value_plays_df.sort_values("Value (%)", ascending=False)
        .groupby("Event")
        .head(1)
        .reset_index(drop=True)
    )
    diversified = _filter_correlated_plays(best_per_game, max_plays=MAX_TOP_PLAYS_NO_CORRELATION)
    total_plays = len(diversified)
    n_nba = int((diversified["League"] == "NBA").sum()) if not diversified.empty else 0
    n_ncaab = int((diversified["League"] == "NCAAB").sum()) if not diversified.empty else 0
    if diversified.empty:
        top_edge_label = "—"
    else:
        top_row = diversified.iloc[0]
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
        edge_pct = float(top_row.get("Value (%)", 0))
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
st.caption("Daily picks • NBA & NCAAB • All odds American (+150, -110)")

# Slim summary bar: date • total plays • NBA / NCAAB counts • top edge (from value_plays_df)
_summary = _summary_bar_values(value_plays_df)
_summary_line = (
    f"{_summary['date_str']} • {_summary['total_plays']} play{'s' if _summary['total_plays'] != 1 else ''} found • "
    f"{_summary['n_nba']} NBA • {_summary['n_ncaab']} NCAAB • Top edge: {_summary['top_edge_label']}"
)
st.markdown(
    f'<div style="font-size:0.9rem; color:rgba(255,255,255,0.75); margin-top:-0.5rem; margin-bottom:0.75rem;">{_html_escape(_summary_line)}</div>',
    unsafe_allow_html=True,
)

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
if "auto_result_run" not in st.session_state:
    try:
        run_auto_result()
    except Exception:
        pass
    st.session_state["auto_result_run"] = True

# Persistent banner: unresolved plays older than 24 hours
_ph_all = load_play_history()
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
    st.warning(
        f"**{int(_unresolved_stale)} play{'s' if _unresolved_stale != 1 else ''} need results marked.** "
        "Go to the Play of the Day History tab below to keep your record accurate.",
        icon="⚠️",
    )
    st.markdown("[→ Go to Play of the Day History](#play-history)")
st.markdown('<div id="play-history"></div>', unsafe_allow_html=True)
tab_overview, tab_ncaab, tab_nba, tab_mark_results, tab_play_history = st.tabs(["Overview", "NCAAB", "NBA", "Mark Results", "Play of the Day History"])

with tab_overview:
    st.markdown(POTD_CARD_CSS, unsafe_allow_html=True)
    # ——— Yesterday's Results (slim strip: two POTD badges W/L) ———
    yesterday_rows = _get_yesterday_potd_results()
    if yesterday_rows:
        st.markdown(_render_yesterday_strip_html(yesterday_rows), unsafe_allow_html=True)
    # ——— Play of the Day (backend selection: edge > 4%, tie-break bookmakers / model gap / no injury) ———
    st.subheader("Play of the Day")
    potd_picks = select_play_of_the_day(value_plays_df, live_odds_df, min_edge_pct=POTD_MIN_EDGE_PCT)
    pod_nba = potd_picks["NBA"]
    pod_ncaab = potd_picks["NCAAB"]
    feature_matrix_potd = (
        load_feature_matrix_for_inference(league=None) if (pod_nba or pod_ncaab) else None
    )
    col_nba, col_ncaab = st.columns(2)
    with col_nba:
        st.markdown(
            _render_potd_card_html("NBA", pod_nba, "blue", feature_matrix=feature_matrix_potd, odds_as_of=datetime.now(timezone.utc), b2b_teams=b2b_teams, march_madness_mode=march_madness_mode),
            unsafe_allow_html=True,
        )
    with col_ncaab:
        st.markdown(
            _render_potd_card_html("NCAAB", pod_ncaab, "orange", feature_matrix=feature_matrix_potd, odds_as_of=datetime.now(timezone.utc), b2b_teams=b2b_teams, march_madness_mode=march_madness_mode),
            unsafe_allow_html=True,
        )
    if value_plays_df.empty:
        if (odds_api_key or "").strip():
            st.caption("No Play of the Day: no games or value plays loaded for today. Try **Refresh odds** below or check back later.")
        else:
            st.caption("No Play of the Day: set your **Odds API key** in the sidebar to load today's NBA & NCAAB games and value plays.")

    st.divider()

    # ——— All Value Plays (card list, filters, low-volume warning) ———
    st.subheader("All Value Plays")
    st.caption("Every game today where the model found edge. One best play per game. Sorted by edge %.")
    if (odds_api_key or "").strip():
        if st.button("Refresh odds", key="overview_refresh", help="Pull latest odds from The Odds API"):
            st.session_state["odds_refresh_key"] = st.session_state.get("odds_refresh_key", 0) + 1
            st.rerun()
    if value_plays_flagged_count > 0:
        st.warning(f"**Potential Data Error:** {value_plays_flagged_count} play(s) had Value % ≥ 15% and were excluded.")

    # Filter row: league (NBA / NCAAB / Both), min edge slider (2–15%, default 2)
    filter_col1, filter_col2, _ = st.columns([1, 1, 2])
    with filter_col1:
        league_filter = st.radio(
            "League",
            options=["Both", "NBA", "NCAAB"],
            index=0,
            horizontal=True,
            key="vp_league_filter",
        )
    with filter_col2:
        min_edge = st.slider(
            "Min edge %",
            min_value=2.0,
            max_value=15.0,
            value=2.0,
            step=0.5,
            key="vp_min_edge",
            help="Only show plays with edge above this threshold.",
        )

    if not value_plays_df.empty:
        # Filter by min edge and league; one best play per game; then correlated filter (max 10, no team twice)
        # March Madness mode: NCAAB plays must have at least 5% edge
        base_edge_ok = value_plays_df["Value (%)"] > min_edge
        if march_madness_mode:
            ncaab_min_ok = (value_plays_df["League"].astype(str).str.strip().str.upper() != "NCAAB") | (value_plays_df["Value (%)"] >= 5.0)
            eligible = value_plays_df[base_edge_ok & ncaab_min_ok].copy()
        else:
            eligible = value_plays_df[base_edge_ok].copy()
        if league_filter != "Both":
            eligible = eligible[eligible["League"] == league_filter]
        best_per_game = (
            eligible.sort_values("Value (%)", ascending=False)
            .groupby("Event")
            .head(1)
            .reset_index(drop=True)
        )
        value_plays_list = _filter_correlated_plays(best_per_game, max_plays=MAX_TOP_PLAYS_NO_CORRELATION)
        if len(value_plays_list) < 3:
            st.warning("**Low volume day** — fewer than 3 qualifying plays. Consider lowering the min edge or check back later.")
        st.caption("**Consensus filter:** Only plays where ≥2 of 3 sub-models agree on direction are shown. **Correlated plays:** Same team in multiple games today is de-duplicated (highest edge per team). Max 10 plays, no team repeated.")
        for _, row in value_plays_list.iterrows():
            st.markdown(_render_value_play_card_html(row, march_madness_mode=march_madness_mode), unsafe_allow_html=True)
            # Bet outcome: look up clv_tracker row and show Mark W / Mark L or current result
            _commence = row.get("commence_time") or ""
            _pt = row.get("point")
            if _pt is not None and pd.notna(_pt):
                try:
                    _pt = float(_pt)
                except (TypeError, ValueError):
                    _pt = None
            _clv = get_clv_row_for_play(
                home_team=str(row.get("home_team", "")),
                away_team=str(row.get("away_team", "")),
                commence_time=str(_commence),
                market_type=str(row.get("Market", "")).strip().lower() or "h2h",
                selection=str(row.get("Selection", "")),
                point=_pt,
            )
            if _clv and _clv.get("id"):
                _cid = _clv["id"]
                _result = _clv.get("result")
                if _result in ("W", "L"):
                    st.caption(f"**Result:** {_result}")
                else:
                    _col_w, _col_l, _ = st.columns([1, 1, 3])
                    with _col_w:
                        if st.button("Mark W", key=f"mark_w_{_cid}", help="Record this play as a win"):
                            mark_bet_result(_cid, "W")
                            st.rerun()
                    with _col_l:
                        if st.button("Mark L", key=f"mark_l_{_cid}", help="Record this play as a loss"):
                            mark_bet_result(_cid, "L")
                            st.rerun()
            st.divider()
    else:
        if (odds_api_key or "").strip():
            st.info("No value plays from live odds today, or no events. Try again later or use Refresh.")
        else:
            st.info("Set Odds API key in the sidebar to load today's NBA & NCAAB value plays.")

with tab_ncaab:
    st.subheader("NCAAB")
    st.caption("College basketball value plays from the same model: in-house line (power ratings), key numbers, fractional Kelly half-units.")
    if (odds_api_key or "").strip():
        if st.button("Refresh", key="ncaab_refresh", help="Pull latest NCAAB odds from The Odds API (also refreshes every 15 min automatically)"):
            st.session_state["odds_refresh_key"] = st.session_state.get("odds_refresh_key", 0) + 1
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
        _vp["March context"] = _vp.apply(_march_context_badges, axis=1)
        if march_madness_mode:
            _vp["Tournament Context"] = _vp.apply(lambda r: "Yes" if _is_top_seed_play(r) else "", axis=1)
        st.dataframe(_vp.rename(columns={"Event": "Game"}), use_container_width=True, hide_index=True)
    else:
        if (odds_api_key or "").strip():
            st.info("No NCAAB games for today from the Odds API, or no value plays with 3% ≤ Value % < 15%. Try again later or use Refresh.")
        else:
            st.info("Set Odds API key in the sidebar to load today's NCAAB games and value plays.")

with tab_mark_results:
    st.subheader("Mark Results")
    st.caption("Mark Win / Loss / Push for past plays (yesterday and earlier). Today's plays are not shown here. Results are saved to your record and used for ROI.")
    _today = date.today()
    _yesterday = _today - timedelta(days=1)
    _mr_from = _today - timedelta(days=90)  # show last 90 days of past plays
    _mr_history = load_play_history(from_date=_mr_from, to_date=_yesterday)
    if not _mr_history.empty:
        _mr_history = _mr_history.copy()
        _mr_history["result_clean"] = _mr_history["result"].apply(
            lambda x: str(x).strip().upper() if x is not None and not pd.isna(x) else None
        )

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
            side = str(r.get("recommended_side", ""))
            line = r.get("spread_or_total")
            if line is None or pd.isna(line) or line == -999:
                return f"{bt} · {side}".strip()
            line = float(line)
            if "Over" in bt or "Under" in bt or "total" in bt.lower():
                return f"{bt} · {side} {line:.1f}".strip()
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
                c1, c2, c3, _ = st.columns([1, 1, 1, 3])
                with c1:
                    if st.button("✅ Win", key=f"mr_w_{row.get('play_id')}", help="Mark win"):
                        for _pid in pids_in_group:
                            update_play_result(_pid, "W")
                        st.rerun()
                with c2:
                    if st.button("❌ Loss", key=f"mr_l_{row.get('play_id')}", help="Mark loss"):
                        for _pid in pids_in_group:
                            update_play_result(_pid, "L")
                        st.rerun()
                with c3:
                    if st.button("➖ Push", key=f"mr_p_{row.get('play_id')}", help="Mark push"):
                        for _pid in pids_in_group:
                            update_play_result(_pid, "P")
                        st.rerun()
    else:
        st.info("No past plays in the last 90 days to mark. Plays from yesterday and earlier appear here once they are archived.")

with tab_play_history:
    st.subheader("Play of the Day History")
    st.caption("One top NBA and one top NCAAB pick per day (last 30 days). Results auto-filled at 8am ET from ESPN.")
    _from = date.today() - timedelta(days=30)
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
        top_per_day = _history.loc[_history.groupby(["date_generated", "sport"])["my_edge_pct"].idxmax()].reset_index(drop=True)
        resolved = _history[_history["result_clean"].notna()]
        resolved_potd = top_per_day[top_per_day["result_clean"].notna()]

        def _roi_and_units(df: pd.DataFrame) -> tuple[float, float, float, float, float]:
            if df.empty:
                return 0.0, 0.0, 0, 0, 0, 0.0
            staked = df["recommended_stake"].replace([None, 0], float("nan")).sum()
            staked = 0.0 if pd.isna(staked) or staked == 0 else float(staked)
            payout = df["actual_payout"].fillna(0).sum()
            roi = (payout / staked * 100) if staked else 0.0
            units = 0.0
            for _, r in df.iterrows():
                s, pa = r.get("recommended_stake"), r.get("actual_payout")
                if s and pd.notna(s) and float(s) != 0 and pa is not None and pd.notna(pa):
                    units += float(pa) / float(s)
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
        .ph-kpi-breakdown { display: flex; justify-content: center; gap: 3rem; margin-top: 0.75rem; padding: 0.75rem; background: rgba(255,255,255,0.05); border-radius: 10px; font-size: 0.95rem; }
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
            f'<span class="ph-roi {roi_class}">{roi_sign}{roi_pct:.1f}% ROI</span>'
            f'<span>{units_sign}{profit_units:.2f} units</span>'
            f'<span>{win_rate:.0f}% win rate</span>'
            f'</div>'
            f'<div class="ph-kpi-breakdown">'
            f'<span><strong>Play of the Day:</strong> <span class="ph-w">{int(w_potd)}</span>–<span class="ph-l">{int(l_potd)}</span>–<span class="ph-p">{int(p_potd)}</span></span>'
            f'<span><strong>All value plays:</strong> <span class="ph-w">{int(w)}</span>–<span class="ph-l">{int(l)}</span>–<span class="ph-p">{int(p)}</span></span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        def _bet_str(r: pd.Series) -> str:
            bt = str(r.get("bet_type", ""))
            side = str(r.get("recommended_side", ""))
            line = r.get("spread_or_total")
            if line is None or pd.isna(line) or line == -999:
                return f"{bt} · {side}".strip()
            line = float(line)
            if "Over" in bt or "Under" in bt or "total" in bt.lower():
                return f"{bt} · {side} {line:.1f}".strip()
            return f"{bt} · {side} {line:+.1f}".strip()

        dates_desc = sorted(top_per_day["date_generated"].unique(), reverse=True)
        for d in dates_desc:
            day_rows = top_per_day[top_per_day["date_generated"] == d]
            date_display = pd.to_datetime(d).strftime("%a, %b %d") if hasattr(pd.to_datetime(d), "strftime") else str(d)
            nba_row = day_rows[day_rows["sport"].astype(str).str.upper() == "NBA"]
            ncaab_row = day_rows[day_rows["sport"].astype(str).str.upper() == "NCAAB"]
            col_nba, col_ncaab = st.columns(2)
            for col, sport_label, row_df in [(col_nba, "NBA", nba_row), (col_ncaab, "NCAAB", ncaab_row)]:
                with col:
                    if row_df.empty:
                        st.markdown(f'<div class="ph-feed-empty">{sport_label}: No pick</div>', unsafe_allow_html=True)
                        continue
                    row = row_df.iloc[0]
                    result = row.get("result_clean")
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
                    if result == "W":
                        badge = '<span class="ph-feed-badge ph-feed-badge--win">WIN</span>'
                    elif result == "L":
                        badge = '<span class="ph-feed-badge ph-feed-badge--loss">LOSS</span>'
                    elif result == "P":
                        badge = '<span class="ph-feed-badge ph-feed-badge--push">PUSH</span>'
                    else:
                        badge = '<span class="ph-feed-badge ph-feed-badge--pending">—</span>'
                    st.markdown(
                        f'<div class="ph-feed-card {card_class}">'
                        f'<div class="ph-feed-date">{_html_escape(date_display)} · {sport_label}</div>'
                        f'<div class="ph-feed-matchup">{_html_escape(matchup)}</div>'
                        f'<div class="ph-feed-meta">{_html_escape(bet_str)} · Edge {edge_str} · Odds {odds_str}</div>'
                        f'<div>{badge}</div></div>',
                        unsafe_allow_html=True,
                    )
                    if result is None or pd.isna(result):
                        mk = row.get("_matchup_key", "")
                        pids_in_group = _history.loc[_history["_matchup_key"] == mk, "play_id"].astype(int).tolist()
                        c1, c2, c3, _ = st.columns([1, 1, 1, 3])
                        with c1:
                            if st.button("✅ Win", key=f"ph_w_{row.get('play_id')}_{sport_label}", help="Mark win"):
                                for _pid in pids_in_group:
                                    update_play_result(_pid, "W")
                                st.rerun()
                        with c2:
                            if st.button("❌ Loss", key=f"ph_l_{row.get('play_id')}_{sport_label}", help="Mark loss"):
                                for _pid in pids_in_group:
                                    update_play_result(_pid, "L")
                                st.rerun()
                        with c3:
                            if st.button("➖ Push", key=f"ph_p_{row.get('play_id')}_{sport_label}", help="Mark push"):
                                for _pid in pids_in_group:
                                    update_play_result(_pid, "P")
                                st.rerun()
    else:
        st.info("No archived plays yet. Plays are saved when you load the dashboard (and at 9am ET). Open the app before games to capture today's top picks.")

with tab_nba:
    st.subheader("NBA")
    st.caption("All NBA value plays and totals. Today's top NBA pick is on the **Overview** tab.")
    if (odds_api_key or "").strip():
        if st.button("Refresh", key="nba_refresh", help="Pull latest NBA odds from The Odds API (also refreshes every 15 min automatically)"):
            st.session_state["odds_refresh_key"] = st.session_state.get("odds_refresh_key", 0) + 1
            st.rerun()

    nba_value = value_plays_df[value_plays_df["League"] == "NBA"] if not value_plays_df.empty and "League" in value_plays_df.columns else pd.DataFrame()
    if not nba_value.empty:
        nba_best = nba_value.sort_values("Value (%)", ascending=False).groupby("Event").head(1).reset_index(drop=True)
        st.caption("**Best value plays (NBA)** — same model as main table")
        _vp = nba_best.copy()
        _vp["Odds"] = _vp["Odds"].apply(format_american)
        if "Market" in _vp.columns:
            _vp["Market"] = _vp["Market"].map(lambda x: MARKET_LABELS.get(x, x) if pd.notna(x) else x)
        if "Recommended Stake" in _vp.columns:
            _vp = _vp.drop(columns=["Recommended Stake"])
        st.dataframe(_vp.rename(columns={"Event": "Game"}), use_container_width=True, hide_index=True)
    st.caption("**Totals tracker** — in-house projected total vs market O/U")
    nba_totals_df = pd.DataFrame()
    if not live_odds_df.empty and "sport_key" in live_odds_df.columns:
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
            # One row per game: take first occurrence per event_id (same point for Over/Under)
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
    if not nba_totals_df.empty:
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
    else:
        if (odds_api_key or "").strip():
            st.info("No NBA totals for today from the Odds API, or no games with O/U lines. Try again later or use Refresh.")
        else:
            st.info("Set Odds API key in the sidebar to load today's NBA games and totals.")

