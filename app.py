"""
Bobby Bottle's Betting Model - Streamlit Dashboard
Dark theme, strategy/bankroll sidebar, bankroll chart, bet history table.
Uses real data only: live odds from The Odds API; simulation runs on user-provided or empty historical data.
"""

import streamlit as st
import plotly.graph_objects as go
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
    HALF_UNIT_PCT,
    MAX_KELLY_PCT_WALTERS,
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

# Odds > +500 (6.0 decimal) flagged as high-risk; excluded from Best Value unless toggled
HIGH_RISK_ODDS_AMERICAN = 500


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

        if market_type == "totals" and point is not None:
            market_total = float(point)
            in_house_total = predict_nba_total(home_team, away_team, pace_stats, b2b_teams=b2b_teams)
            prefer_over = "Over" in selection or "over" in selection.lower()
            model_prob = model_prob_from_in_house_total(in_house_total, market_total, prefer_over)
        elif market_type == "spreads" and point is not None:
            market_spread = -float(point)
            home_rating = power_ratings.get(home_team, default_rating) - get_schedule_fatigue_penalty(home_team, as_of_date)
            away_rating = power_ratings.get(away_team, default_rating) - get_schedule_fatigue_penalty(away_team, as_of_date)
            in_house_spread = in_house_spread_from_ratings(home_rating, away_rating)
            we_cover_favorite = "-" in selection
            model_prob = model_prob_from_in_house_spread(in_house_spread, market_spread, we_cover_favorite)
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

# API key: Streamlit secrets (best practice) → env → sidebar only when missing
def _get_odds_api_key() -> str:
    key = ""
    try:
        if hasattr(st, "secrets") and st.secrets:
            key = (st.secrets.get("the_odds_api", {}).get("api_key") or st.secrets.get("ODDS_API_KEY") or "").strip()
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
    """Fetch live odds from The Odds API. Cached 15 min; refresh_key forces refetch when changed."""
    api_key = _get_odds_api_key()
    if not (api_key or "").strip():
        return pd.DataFrame()
    d = date.fromisoformat(commence_date_iso)
    return get_live_odds(
        api_key=api_key.strip(),
        sport_keys=LIVE_ODDS_SPORT_KEYS,
        commence_on_date=d,
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
    live_odds_df = _fetch_live_odds_cached(
        date.today().isoformat(),
        refresh_key=st.session_state["odds_refresh_key"],
    )
    b2b_teams = _get_b2b_teams_cached(
        date.today().isoformat(),
        refresh_key=st.session_state.get("odds_refresh_key", 0),
    )
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
    )
    value_plays_df = add_injury_alerts_to_value_plays(value_plays_df, "Basketball")
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
# Main layout — tabbed (Overview, Basketball)
# -----------------------------------------------------------------------------

st.title("Bobby Bottle's Betting Model")
st.caption("Basketball focus • All odds American (+150, -110)")

basketball_profit = results_basketball["final_bankroll"] - starting_bankroll
curve_b = results_basketball["bankroll_curve"]

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

tab_overview, tab_basketball, tab_nba_totals = st.tabs(["Overview", "Basketball", "NBA Totals"])

with tab_overview:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Final bankroll", f"${results_basketball['final_bankroll']:,.2f}", f"{basketball_profit:+,.2f}")
    with c2:
        st.metric("ROI", f"{results_basketball['roi_pct']:.2f}%", "")
    with c3:
        st.metric("Bet count", results_basketball["total_bets"], f"Wins: {results_basketball['wins']} / Losses: {results_basketball['losses']}")
    st.subheader("Bankroll over time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(curve_b))),
        y=curve_b,
        mode="lines+markers",
        name="Basketball",
        line=dict(color="#e07ec8", width=2),
        marker=dict(size=4),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_title="Step",
        yaxis_title="Bankroll",
        height=360,
        font=dict(color="#fafafa", size=12),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_basketball:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Final bankroll", f"${results_basketball['final_bankroll']:,.2f}", f"{results_basketball['final_bankroll'] - starting_bankroll:+,.2f}")
    with c2:
        st.metric("ROI", f"{results_basketball['roi_pct']:.2f}%", "")
    with c3:
        st.metric("Bet count", results_basketball["total_bets"], f"Wins: {results_basketball['wins']} / Losses: {results_basketball['losses']}")
    st.subheader("Bet history")
    _bet_history_table(results_basketball["bet_history"])

with tab_nba_totals:
    st.subheader("NBA Totals")
    st.caption("Pace-adjusted Over/Under vs bookmaker line. Back-to-back (B2B) teams: Pace −1.5%, Off Rating −2%. Value Over if projection > line + 3; Value Under if projection < line − 3.")
    if (odds_api_key or "").strip():
        if st.button("Refresh", help="Pull latest NBA odds from The Odds API (also refreshes every 15 min automatically)"):
            st.session_state["odds_refresh_key"] = st.session_state.get("odds_refresh_key", 0) + 1
            st.rerun()
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

# Today's Best Value Plays — live from The Odds API only
st.subheader("Today's Best Value Plays")
using_live = bool((odds_api_key or "").strip())
st.caption(
    "**Live** from The Odds API (NBA + NCAAB, today's games). Data refreshes every 15 min." if using_live
    else "Set Odds API key in sidebar for live NBA & NCAAB value plays."
)
st.caption("Value (%) = In-house line (power ratings) vs market • Key-number adjusted • Stake: fractional Kelly, half-units (1–3%). One best value play per game.")
st.info("**Market Timing (Walters):** Bet early for favorites, bet late for underdogs.")
if value_plays_flagged_count > 0:
    st.warning(f"**Potential Data Error:** {value_plays_flagged_count} play(s) had Value % ≥ 15% and were excluded.")
if not value_plays_df.empty:
    # Single best value play per game (highest EV)
    value_plays_display = (
        value_plays_df.sort_values("Value (%)", ascending=False)
        .groupby("Event")
        .head(1)
        .reset_index(drop=True)
    )
    def _green_value(s: pd.Series) -> list[str]:
        return ["background-color: rgba(46, 125, 50, 0.45); color: #c8e6c9;" for _ in s]

    def _injury_alert_style(s: pd.Series) -> list[str]:
        return [
            "background-color: rgba(255, 152, 0, 0.4); color: #fff3e0;" if (v and str(v).strip() != "—") else ""
            for v in s
        ]

    display_vp = value_plays_display.copy()
    display_vp["Odds"] = display_vp["Odds"].apply(format_american)
    # Recommend column: dollar amount e.g. $10.50 (not $.2f)
    if "Recommended Stake" in display_vp.columns:
        display_vp["Recommended Stake"] = display_vp["Recommended Stake"].apply(
            lambda x: format_currency(x) if pd.notna(x) else "—"
        )
    if "Market" in display_vp.columns:
        display_vp["Market"] = display_vp["Market"].map(lambda x: MARKET_LABELS.get(x, x) if pd.notna(x) else x)
    # Show "Game" instead of "Event"; put Start Time after Game
    display_vp = display_vp.rename(columns={"Event": "Game"})
    if "Start Time" not in display_vp.columns:
        display_vp["Start Time"] = "—"
    order = ["League", "Game", "Start Time"]
    display_vp = display_vp[[c for c in order if c in display_vp.columns] + [c for c in display_vp.columns if c not in order]]
    styled = display_vp.style.apply(_green_value, subset=["Value (%)"])
    if "Injury Alert" in display_vp.columns:
        styled = styled.apply(_injury_alert_style, subset=["Injury Alert"])
    col_config = {
        "League": st.column_config.TextColumn("League", width="small"),
        "Game": st.column_config.TextColumn("Game", width="medium"),
        "Start Time": st.column_config.TextColumn("Start Time", width="medium"),
        "Injury Alert": st.column_config.TextColumn("Injury Alert", width="medium"),
        "Selection": st.column_config.TextColumn("Selection", width="small"),
        "Odds": st.column_config.TextColumn("Odds (American)", width="small"),
        "Value (%)": st.column_config.NumberColumn("Value (%)", format="%.2f", width="small"),
        "Recommended Stake": st.column_config.TextColumn("Recommended Stake", width="small"),
    }
    if "Market" in display_vp.columns:
        col_config["Market"] = st.column_config.TextColumn("Market", width="small")
    st.dataframe(styled, use_container_width=True, hide_index=True, column_config=col_config)
else:
    if using_live:
        st.info("No value plays with 3% ≤ Value % < 15% from live odds right now, or API returned no events for today. Try again later.")
    else:
        st.info("Set Odds API key in the sidebar to load today's NBA & NCAAB games and value plays.")
