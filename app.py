"""
Bobby Bottle's Betting Model - Streamlit Dashboard
Daily picks sheet: Play of the Day (NBA + NCAAB) and All Value Plays. Dark theme, NCAAB/NBA tabs.
Live odds from The Odds API; stakes from fractional Kelly.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import date, datetime, timezone
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
    get_clv_summary_last_30_days,
)
from engine.betting_models import (
    load_feature_matrix_for_inference,
    get_feature_row_for_game,
    predict_spread_prob,
    predict_totals_prob,
    predict_moneyline_prob,
)


def format_american(odds: float) -> str:
    """Format American odds for display: +150, -110."""
    try:
        x = int(round(float(odds)))
        return f"{x:+d}" if x != 0 else "—"
    except (TypeError, ValueError):
        return "—"


def format_currency(val: float) -> str:
    """Format as $0.00 for display."""
    try:
        return f"${float(val):,.2f}"
    except (TypeError, ValueError):
        return "—"


# Human-readable market labels (replaces h2h, spreads, totals in UI)
MARKET_LABELS = {"h2h": "Winner", "spreads": "Spread", "totals": "Over/Under"}


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

# Confidence tier: edge >= this is High, else Medium
POTD_HIGH_CONFIDENCE_EDGE_PCT = 8.0
# Minimum edge % to qualify as Play of the Day
POTD_MIN_EDGE_PCT = 4.0


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

    for league in ("NBA", "NCAAB"):
        league_plays = eligible[eligible["League"] == league]
        if league_plays.empty:
            continue
        # Sort: edge desc, bookmaker_count desc, model_prob_gap desc, has_injury_flag asc (False first)
        sorted_df = league_plays.sort_values(
            by=["Value (%)", "bookmaker_count", "model_prob_gap", "has_injury_flag"],
            ascending=[False, False, False, True],
        )
        top = sorted_df.iloc[0]
        # After merge, point may appear as point_x (left) or point
        point_val = top.get("point_x") if "point_x" in top.index else top.get("point")
        pick: dict = {
            "League": league,
            "Event": top.get("Event", ""),
            "Selection": top.get("Selection", ""),
            "Market": top.get("Market", ""),
            "Odds": int(top.get("Odds", 0)),
            "Value (%)": float(top.get("Value (%)", 0)),
            "point": point_val,
            "Recommended Stake": top.get("Recommended Stake", 0),
            "Start Time": top.get("Start Time", "—"),
            "Injury Alert": top.get("Injury Alert", "—"),
        }
        result[league] = pick

    return result


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


def _one_sentence_reasoning(row: pd.Series) -> str:
    """One-sentence summary of why this play was flagged (e.g. rest, pace, key number)."""
    market = str(row.get("Market", ""))
    if market == "spreads":
        return "Power ratings and schedule fatigue (back-to-backs) suggest the market is mispricing this spread."
    if market == "totals":
        return "Pace-adjusted projections and key-number analysis (3, 7, 14) indicate the total is off."
    return "Implied probability vs our model probability shows meaningful edge on the moneyline."


def _potd_badge_text(row: pd.Series) -> str:
    """Recommended bet label for badge: e.g. 'Spread · Lakers -3.5', 'Over 220.5', 'Moneyline · Lakers'."""
    market = str(row.get("Market", ""))
    selection = str(row.get("Selection", ""))
    market_label = MARKET_LABELS.get(market, market)
    point = row.get("point")
    if point is not None:
        try:
            pt = float(point)
            if market == "totals":
                return f"{market_label} · {selection} {pt:.1f}"
            return f"{market_label} · {selection} {pt:+.1f}"
        except (TypeError, ValueError):
            pass
    return f"{market_label} · {selection}"


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
.potd-badge {
    display: inline-block;
    padding: 0.35rem 0.75rem;
    border-radius: 8px;
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
}
.potd-card--blue .potd-badge  { background: rgba(30,136,229,0.4); color: #90caf9; }
.potd-card--orange .potd-badge { background: rgba(245,124,0,0.4); color: #ffe0b2; }
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
.potd-reason {
    font-size: 0.9rem;
    line-height: 1.4;
    color: rgba(255,255,255,0.85);
    margin-bottom: 0.6rem;
}
.potd-odds {
    font-size: 1.1rem;
    font-weight: 700;
    color: rgba(255,255,255,0.95);
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
.vp-confidence-wrap { margin-top: 0.5rem; }
.vp-confidence-bar { height: 6px; border-radius: 3px; background: rgba(255,255,255,0.15); overflow: hidden; }
.vp-confidence-fill { height: 100%; border-radius: 3px; background: linear-gradient(90deg, #43a047, #66bb6a); transition: width 0.2s ease; }
.vp-reasoning { font-size: 0.8rem; color: rgba(255,255,255,0.65); font-style: italic; margin-top: 0.5rem; line-height: 1.35; }
</style>
"""


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


def _render_value_play_card_html(row: pd.Series, edge_max_pct: float = 15.0) -> str:
    """One All Value Plays card: matchup, bet type, edge %, odds, confidence bar, reasoning."""
    league = str(row.get("League", ""))
    league_class = "nba" if league == "NBA" else "ncaab"
    matchup = _html_escape(str(row.get("Event", "")))
    bet_type = _html_escape(_vp_bet_type_label(row))
    edge_pct = float(row.get("Value (%)", 0))
    odds_str = format_american(row.get("Odds", 0))
    bar_pct = min(100.0, max(0.0, (edge_pct / edge_max_pct) * 100.0))
    reasoning = _html_escape(_value_play_reasoning(row))
    return f"""
    <div class="vp-card">
        <div class="vp-card-league {league_class}">{_html_escape(league)}</div>
        <div class="vp-matchup">{matchup}</div>
        <div class="vp-meta">
            <span class="vp-bet-type">{bet_type}</span>
            <span class="vp-edge">{edge_pct:.1f}% edge</span>
            <span class="vp-odds">{odds_str}</span>
        </div>
        <div class="vp-confidence-wrap">
            <div class="vp-confidence-bar"><div class="vp-confidence-fill" style="width:{bar_pct:.1f}%"></div></div>
        </div>
        <div class="vp-reasoning">{reasoning}</div>
    </div>
    """


def _render_potd_card_html(league: str, row: Optional[pd.Series], accent: str) -> str:
    """Return HTML for one Play of the Day card. accent: 'blue' | 'orange' | 'grey'. Grey when no play."""
    if row is None or (isinstance(row, pd.Series) and (row.empty or len(row) == 0)):
        return f"""
        <div class="potd-card potd-card--grey">
            <div class="potd-league">{_html_escape(league)}</div>
            <div class="potd-empty-msg">No Play Today</div>
        </div>
        """
    r = row
    edge_pct = float(r.get("Value (%)", 0))
    confidence = "High" if edge_pct >= POTD_HIGH_CONFIDENCE_EDGE_PCT else "Medium"
    away, home = _parse_event_teams(str(r.get("Event", "")))
    badge_text = _potd_badge_text(r)
    reason = _one_sentence_reasoning(r)
    odds_str = format_american(r.get("Odds", 0))
    return f"""
    <div class="potd-card potd-card--{accent}">
        <div class="potd-league">{_html_escape(league)}</div>
        <div class="potd-matchup">{_html_escape(away)} <span class="potd-vs">@</span> {_html_escape(home)}</div>
        <div class="potd-badge">{_html_escape(badge_text)}</div>
        <div class="potd-edge">{edge_pct:.1f}% Edge</div>
        <div class="potd-confidence">Confidence: {_html_escape(confidence)}</div>
        <div class="potd-reason">{_html_escape(reason)}</div>
        <div class="potd-odds">{odds_str}</div>
    </div>
    """


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
            model_prob = predict_totals_prob(
                feature_row, market_total, prefer_over, fallback_prob=fallback
            ) if feature_row is not None else fallback
        elif market_type == "spreads" and point is not None:
            market_spread = -float(point)
            home_rating = power_ratings.get(home_team, default_rating) - get_schedule_fatigue_penalty(home_team, as_of_date)
            away_rating = power_ratings.get(away_team, default_rating) - get_schedule_fatigue_penalty(away_team, as_of_date)
            in_house_spread = in_house_spread_from_ratings(home_rating, away_rating)
            we_cover_favorite = "-" in selection
            fallback = model_prob_from_in_house_spread(in_house_spread, market_spread, we_cover_favorite)
            model_prob = predict_spread_prob(
                feature_row, market_spread, we_cover_favorite, fallback_prob=fallback
            ) if feature_row is not None else fallback
        elif market_type == "h2h":
            home_rating = power_ratings.get(home_team, default_rating) - get_schedule_fatigue_penalty(home_team, as_of_date)
            away_rating = power_ratings.get(away_team, default_rating) - get_schedule_fatigue_penalty(away_team, as_of_date)
            selection_is_home = (
                str(selection).strip().lower() == str(home_team).strip().lower()
                or str(home_team).strip().lower() in str(selection).strip().lower()
                or str(selection).strip().lower() in str(home_team).strip().lower()
            )
            fallback = model_prob_from_ratings_moneyline(home_rating, away_rating, selection_is_home)
            model_prob = predict_moneyline_prob(
                feature_row, selection_is_home, fallback_prob=fallback
            ) if feature_row is not None else fallback
        else:
            implied = implied_probability_no_vig(odds_val)
            edge = np.random.uniform(0, 0.08)
            model_prob = damp_probability(min(0.92, implied + edge))

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
        implied_prob = implied_probability_no_vig(odds_val)
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
        })
    return pd.DataFrame(rows), n_flagged


# Empty historical dataset columns for BettingEngine (no mock data; use real data or upload)
BASKETBALL_HISTORICAL_COLUMNS = ["event_id", "event_name", "odds", "model_prob", "result"]


# -----------------------------------------------------------------------------
# Sidebar — single combined model (no strategy dropdown)
# -----------------------------------------------------------------------------

st.sidebar.header("Settings")
starting_bankroll = st.sidebar.number_input(
    "Starting bankroll",
    min_value=100.0,
    max_value=1_000_000.0,
    value=1000.0,
    step=100.0,
    format="%.0f",
    help="Used for recommended stakes and simulation.",
)

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

# -----------------------------------------------------------------------------
# Engine run — basketball (real historical data only; empty until user provides data)
# -----------------------------------------------------------------------------

basketball_historical_df = pd.DataFrame(columns=BASKETBALL_HISTORICAL_COLUMNS)
engine_basketball = BettingEngine(basketball_historical_df, strategy_fn, starting_bankroll)
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
        value_plays_df, value_plays_flagged_count = _live_odds_to_value_plays(
            live_odds_df,
            bankroll=starting_bankroll,
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
    """From value_plays_df (today's data), compute: date_str, total_plays, n_nba, n_ncaab, top_edge_label."""
    date_str = _today_str()
    # Use same definition as All Value Plays: one best play per game
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
    total_plays = len(best_per_game)
    n_nba = int((best_per_game["League"] == "NBA").sum())
    n_ncaab = int((best_per_game["League"] == "NCAAB").sum())
    if best_per_game.empty:
        top_edge_label = "—"
    else:
        top_row = best_per_game.iloc[0]
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
        "bankroll_after": "Bankroll after",
    })
    bet_display = bet_df.copy()
    if "Odds" in bet_display.columns:
        bet_display["Odds"] = bet_display["Odds"].apply(format_american)
    for col in ("Stake", "P/L", "Bankroll after"):
        if col in bet_display.columns:
            bet_display[col] = bet_display[col].apply(lambda x: format_currency(x) if pd.notna(x) else "—")
    cols = ["Step", "Event", "Odds", "Model prob", "Stake", "Result", "P/L", "Bankroll after"]
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
            "Bankroll after": st.column_config.TextColumn("Bankroll after", width="small"),
        },
    )

tab_overview, tab_ncaab, tab_nba, tab_clv = st.tabs(["Overview", "NCAAB", "NBA", "CLV Summary"])

with tab_overview:
    # ——— Play of the Day (backend selection: edge > 4%, tie-break bookmakers / model gap / no injury) ———
    st.subheader("Play of the Day")
    st.markdown(POTD_CARD_CSS, unsafe_allow_html=True)
    potd_picks = select_play_of_the_day(value_plays_df, live_odds_df, min_edge_pct=POTD_MIN_EDGE_PCT)
    pod_nba = potd_picks["NBA"]
    pod_ncaab = potd_picks["NCAAB"]
    col_nba, col_ncaab = st.columns(2)
    with col_nba:
        st.markdown(_render_potd_card_html("NBA", pod_nba, "blue"), unsafe_allow_html=True)
    with col_ncaab:
        st.markdown(_render_potd_card_html("NCAAB", pod_ncaab, "orange"), unsafe_allow_html=True)
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
        # Filter by min edge and league; one best play per game; sort by edge desc
        eligible = value_plays_df[value_plays_df["Value (%)"] > min_edge].copy()
        if league_filter != "Both":
            eligible = eligible[eligible["League"] == league_filter]
        value_plays_list = (
            eligible.sort_values("Value (%)", ascending=False)
            .groupby("Event")
            .head(1)
            .reset_index(drop=True)
        )
        if len(value_plays_list) < 3:
            st.warning("**Low volume day** — fewer than 3 qualifying plays. Consider lowering the min edge or check back later.")
        for _, row in value_plays_list.iterrows():
            st.markdown(_render_value_play_card_html(row), unsafe_allow_html=True)
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
        st.caption("**Best value plays (NCAAB)** — same model as main table")
        _vp = ncaab_best.copy()
        _vp["Odds"] = _vp["Odds"].apply(format_american)
        if "Recommended Stake" in _vp.columns:
            _vp["Recommended Stake"] = _vp["Recommended Stake"].apply(lambda x: format_currency(x) if pd.notna(x) else "—")
        if "Market" in _vp.columns:
            _vp["Market"] = _vp["Market"].map(lambda x: MARKET_LABELS.get(x, x) if pd.notna(x) else x)
        st.dataframe(_vp.rename(columns={"Event": "Game"}), use_container_width=True, hide_index=True)
    else:
        if (odds_api_key or "").strip():
            st.info("No NCAAB games for today from the Odds API, or no value plays with 3% ≤ Value % < 15%. Try again later or use Refresh.")
        else:
            st.info("Set Odds API key in the sidebar to load today's NCAAB games and value plays.")

with tab_clv:
    st.subheader("CLV Summary")
    st.caption("Average Closing Line Value by sport and bet type over the last 30 days. Positive CLV = you got a better number than the closing line.")
    _clv_summary = get_clv_summary_last_30_days()
    if not _clv_summary.empty and _clv_summary["n_bets"].sum() > 0:
        _clv_summary = _clv_summary.copy()
        _clv_summary["bet_type"] = _clv_summary["bet_type"].map(lambda x: MARKET_LABELS.get(str(x).lower(), str(x)))
        _disp = _clv_summary.rename(columns={"league": "Sport", "avg_clv_pct": "Avg CLV %", "n_bets": "Bets"})
        _disp["Avg CLV %"] = _disp["Avg CLV %"].apply(lambda x: f"{x:+.2f}%")
        st.dataframe(
            _disp,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Sport": st.column_config.TextColumn("Sport", width="small"),
                "bet_type": st.column_config.TextColumn("Bet type", width="small"),
                "Avg CLV %": st.column_config.TextColumn("Avg CLV %", width="small"),
                "Bets": st.column_config.NumberColumn("Bets", width="small"),
            },
        )
    else:
        st.info("No CLV data yet. Record recommendations (use the app to load value plays) and run odds fetches so closing odds can be backfilled. Data appears here once you have bets with closing odds in the last 30 days.")

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
        if "Recommended Stake" in _vp.columns:
            _vp["Recommended Stake"] = _vp["Recommended Stake"].apply(lambda x: format_currency(x) if pd.notna(x) else "—")
        if "Market" in _vp.columns:
            _vp["Market"] = _vp["Market"].map(lambda x: MARKET_LABELS.get(x, x) if pd.notna(x) else x)
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

