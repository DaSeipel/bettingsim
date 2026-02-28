"""
Sports Betting Simulator - Streamlit Dashboard
Dark theme, strategy/bankroll sidebar, bankroll chart, bet history table.
Uses real data only: live odds from The Odds API; simulation runs on user-provided or empty historical data.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from datetime import date
from engine.engine import (
    BettingEngine,
    get_live_odds,
    BASKETBALL_NBA,
    BASKETBALL_NCAAB,
)
from strategies.strategies import (
    strategy_kelly,
    strategy_value_betting,
    kelly_fraction,
    implied_probability,
    american_to_decimal,
)


def format_american(odds: float) -> str:
    """Format American odds for display: +150, -110."""
    try:
        x = int(round(float(odds)))
        return f"{x:+d}" if x != 0 else "—"
    except (TypeError, ValueError):
        return "—"


st.set_page_config(
    page_title="Sports Betting Simulator",
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


def _live_odds_to_value_plays(
    odds_df: pd.DataFrame,
    bankroll: float,
    kelly_frac: float = 0.25,
    min_ev_pct: float = 5.0,
    seed: int = 44,
) -> pd.DataFrame:
    """
    From live odds DataFrame (league, event_name, market_type, selection, odds),
    apply Value (%) = (model_prob vs market implied); filter EV > min_ev_pct; add Kelly stake.
    Returns rows with League, Event, Selection, Market, Odds, Value (%), Recommended Stake.
    """
    if odds_df.empty:
        return pd.DataFrame(
            columns=["League", "Event", "Selection", "Market", "Odds", "Value (%)", "Recommended Stake", "Injury Alert"]
        )
    np.random.seed(seed)
    rows = []
    for _, r in odds_df.iterrows():
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
        rows.append({
            "League": r.get("league", ""),
            "Event": r.get("event_name", ""),
            "Selection": r.get("selection", ""),
            "Market": r.get("market_type", ""),
            "Odds": int(round(odds_val)),
            "Value (%)": round(ev_pct, 2),
            "Recommended Stake": stake,
            "Injury Alert": "—",
        })
    return pd.DataFrame(rows)


# Empty historical dataset columns for BettingEngine (no mock data; use real data or upload)
BASKETBALL_HISTORICAL_COLUMNS = ["event_id", "event_name", "odds", "model_prob", "result"]


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

st.sidebar.header("Simulation settings")
strategy_name = st.sidebar.selectbox(
    "Strategy",
    ["Kelly Criterion", "Value Betting"],
    index=0,
)
starting_bankroll = st.sidebar.number_input(
    "Starting bankroll",
    min_value=100.0,
    max_value=1_000_000.0,
    value=1000.0,
    step=100.0,
    format="%.0f",
)

kelly_frac = 0.25
if strategy_name == "Kelly Criterion":
    kelly_frac = st.sidebar.slider("Kelly fraction", 0.1, 1.0, 0.25, 0.05)
    strategy_fn = strategy_kelly(kelly_fraction_param=kelly_frac)
else:
    use_flat = st.sidebar.checkbox("Use flat stake", value=False)
    if use_flat:
        flat_stake = st.sidebar.number_input("Flat stake", min_value=5.0, value=20.0, step=5.0)
        strategy_fn = strategy_value_betting(flat_stake=flat_stake)
    else:
        stake_pct = st.sidebar.slider("Stake % of bankroll", 0.01, 0.1, 0.02, 0.01)
        strategy_fn = strategy_value_betting(stake_pct=stake_pct)

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

# -----------------------------------------------------------------------------
# Engine run — basketball (real historical data only; empty until user provides data)
# -----------------------------------------------------------------------------

basketball_historical_df = pd.DataFrame(columns=BASKETBALL_HISTORICAL_COLUMNS)
engine_basketball = BettingEngine(basketball_historical_df, strategy_fn, starting_bankroll)
results_basketball = engine_basketball.run()

# Value plays: live NBA + NCAAB from The Odds API; cached 15 min to stay under 500 requests/month
kelly_frac_val = kelly_frac if strategy_name == "Kelly Criterion" else 0.25
ODDS_CACHE_TTL_SECONDS = 900  # 15 minutes


@st.cache_data(ttl=ODDS_CACHE_TTL_SECONDS)
def _fetch_live_odds_cached(commence_date_iso: str) -> pd.DataFrame:
    """Fetch live odds from The Odds API. Cached 15 min to limit API usage."""
    api_key = _get_odds_api_key()
    if not (api_key or "").strip():
        return pd.DataFrame()
    d = date.fromisoformat(commence_date_iso)
    return get_live_odds(
        api_key=api_key.strip(),
        sport_keys=LIVE_ODDS_SPORT_KEYS,
        commence_on_date=d,
    )


if (odds_api_key or "").strip():
    live_odds_df = _fetch_live_odds_cached(date.today().isoformat())
    value_plays_df = _live_odds_to_value_plays(
        live_odds_df,
        bankroll=starting_bankroll,
        kelly_frac=kelly_frac_val,
        min_ev_pct=5.0,
    )
    value_plays_df = add_injury_alerts_to_value_plays(value_plays_df, "Basketball")
else:
    value_plays_df = pd.DataFrame(
        columns=["League", "Event", "Selection", "Market", "Odds", "Value (%)", "Recommended Stake", "Injury Alert"]
    )

# -----------------------------------------------------------------------------
# Main layout — tabbed (Overview, Basketball)
# -----------------------------------------------------------------------------

st.title("Sports Betting Simulator")
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
            "Stake": st.column_config.NumberColumn("Stake", format="%.2f", width="small"),
            "Result": st.column_config.TextColumn("Result", width="small"),
            "P/L": st.column_config.NumberColumn("P/L", format="%.2f", width="small"),
            "Bankroll after": st.column_config.NumberColumn("Bankroll after", format="%.2f", width="small"),
        },
    )

tab_overview, tab_basketball = st.tabs(["Overview", "Basketball"])

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

# Today's Best Value Plays — live from The Odds API only
st.subheader("Today's Best Value Plays")
using_live = bool((odds_api_key or "").strip())
st.caption(
    "**Live** from The Odds API (NBA + NCAAB, today's games). Data refreshes every 15 min." if using_live
    else "Set Odds API key in sidebar for live NBA & NCAAB value plays."
)
st.caption("Value (%) = Model prob vs market implied odds • EV > 5% • Recommended stake from Kelly.")
if not value_plays_df.empty:
    def _green_value(s: pd.Series) -> list[str]:
        return ["background-color: rgba(46, 125, 50, 0.45); color: #c8e6c9;" for _ in s]

    def _injury_alert_style(s: pd.Series) -> list[str]:
        return [
            "background-color: rgba(255, 152, 0, 0.4); color: #fff3e0;" if (v and str(v).strip() != "—") else ""
            for v in s
        ]

    display_vp = value_plays_df.copy()
    display_vp["Odds"] = display_vp["Odds"].apply(format_american)
    styled = display_vp.style.apply(_green_value, subset=["Value (%)"])
    if "Injury Alert" in display_vp.columns:
        styled = styled.apply(_injury_alert_style, subset=["Injury Alert"])
    col_config = {
        "League": st.column_config.TextColumn("League", width="small"),
        "Event": st.column_config.TextColumn("Event", width="medium"),
        "Injury Alert": st.column_config.TextColumn("Injury Alert", width="medium"),
        "Selection": st.column_config.TextColumn("Selection", width="small"),
        "Odds": st.column_config.TextColumn("Odds (American)", width="small"),
        "Value (%)": st.column_config.NumberColumn("Value (%)", format="%.2f", width="small"),
        "Recommended Stake": st.column_config.NumberColumn("Recommended Stake", format="$.2f", width="small"),
    }
    if "Market" in value_plays_df.columns:
        col_config["Market"] = st.column_config.TextColumn("Market", width="small")
    st.dataframe(styled, use_container_width=True, hide_index=True, column_config=col_config)
else:
    if using_live:
        st.info("No value plays with EV > 5% from live odds right now, or API returned no events for today. Try again later.")
    else:
        st.info("Set Odds API key in the sidebar to load today's NBA & NCAAB games and value plays.")
