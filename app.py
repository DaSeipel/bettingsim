"""
Sports Betting Simulator - Streamlit Dashboard
Dark theme, strategy/bankroll sidebar, bankroll chart, bet history table.
Uses engine + strategies with mock data.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from engine.engine import BettingEngine, get_daily_fixtures, get_basketball_odds
from strategies.strategies import (
    strategy_kelly,
    strategy_value_betting,
    kelly_fraction,
    implied_probability,
    american_to_decimal,
    decimal_to_american,
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


def get_injury_alerts(sport: str, seed: int = 99) -> dict[str, str]:
    """
    Return a map of event_name -> injury alert text for games with star players questionable.
    Mock implementation; replace with real injury API (e.g. ESPN, official league) in production.
    Flags events where a star is questionable — drastically changes true probability.
    """
    np.random.seed(seed)
    if sport == "Football":
        events_pool = [
            "Team A vs Team B",
            "Team C vs Team D",
            "Team E vs Team F",
            "Team G vs Team H",
        ]
    else:
        events_pool = [
            "Lakers @ Celtics",
            "Warriors @ Nets",
            "Duke @ UNC",
            "Kentucky @ Kansas",
        ]
    # Always flag at least one event so the column is visible; otherwise ~40% chance per event
    alerts = {}
    for i, e in enumerate(events_pool):
        if i == 0 or np.random.random() < 0.4:
            alerts[e] = "Star player questionable"
        else:
            alerts[e] = ""
    return alerts


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


def generate_mock_dataset(n_rows: int = 80, seed: int = 42) -> pd.DataFrame:
    """Mock historical dataset: odds (American), model_prob, result (1=win, 0=loss)."""
    np.random.seed(seed)
    odds_decimal = np.round(np.random.uniform(1.6, 3.2, n_rows), 2)
    odds_american = np.array([int(round(decimal_to_american(d))) for d in odds_decimal])
    implied = 1.0 / odds_decimal
    model_prob = np.clip(implied + np.random.randn(n_rows) * 0.08, 0.15, 0.85)
    model_prob = np.round(model_prob, 3)
    edge = model_prob - implied
    win_prob = np.clip(0.5 + edge * 2, 0.2, 0.8)
    result = (np.random.random(n_rows) < win_prob).astype(int)
    event_names = [f"Match {i+1}" for i in range(n_rows)]
    return pd.DataFrame({
        "event_id": np.arange(n_rows),
        "event_name": event_names,
        "odds": odds_american,
        "model_prob": model_prob,
        "result": result,
    })


def generate_mock_basketball_dataset(n_rows: int = 60, seed: int = 43) -> pd.DataFrame:
    """Mock basketball historical data (spreads/totals): odds (American), model_prob, result."""
    np.random.seed(seed)
    odds_decimal = np.round(np.random.uniform(1.75, 2.25, n_rows), 2)
    odds_american = np.array([int(round(decimal_to_american(d))) for d in odds_decimal])
    implied = 1.0 / odds_decimal
    model_prob = np.clip(implied + np.random.randn(n_rows) * 0.06, 0.2, 0.85)
    model_prob = np.round(model_prob, 3)
    edge = model_prob - implied
    win_prob = np.clip(0.5 + edge * 2, 0.25, 0.75)
    result = (np.random.random(n_rows) < win_prob).astype(int)
    teams = ["Lakers", "Celtics", "Warriors", "Nets", "Duke", "UNC", "Kentucky", "Kansas", "Bucks", "Suns"]
    event_names = [f"{teams[i % 10]} @ {teams[(i + 3) % 10]}" for i in range(n_rows)]
    return pd.DataFrame({
        "event_id": np.arange(n_rows),
        "event_name": event_names,
        "odds": odds_american,
        "model_prob": model_prob,
        "result": result,
    })


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

st.sidebar.header("Simulation settings")
sport_selector = st.sidebar.selectbox(
    "Sport (for Best Value Plays table)",
    ["Football", "Basketball"],
    index=0,
)
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

odds_api_key = ""
if sport_selector == "Basketball":
    odds_api_key = st.sidebar.text_input("Odds API key (optional, for live NBA/NCAAB)", type="password", value="", help="Get a key at the-odds-api.com")

# -----------------------------------------------------------------------------
# Mock data and engine run — separate bankroll per sport
# -----------------------------------------------------------------------------

mock_football_df = generate_mock_dataset()
mock_basketball_df = generate_mock_basketball_dataset()
engine_football = BettingEngine(mock_football_df, strategy_fn, starting_bankroll)
engine_basketball = BettingEngine(mock_basketball_df, strategy_fn, starting_bankroll)
results_football = engine_football.run()
results_basketball = engine_basketball.run()

# Value plays: Football (fixtures) vs Basketball (NBA/NCAAB)
fixtures_df = get_daily_fixtures()
if fixtures_df.empty or len(fixtures_df) < 2:
    fixtures_df = pd.DataFrame({
        "event_name": ["Team A vs Team B", "Team C vs Team D", "Team E vs Team F", "Team G vs Team H"],
        "odds_home": [110, -120, 140, -105],   # American
        "odds_draw": [220, 260, 210, 250],
        "odds_away": [250, 280, 190, 280],
    })
kelly_frac_val = kelly_frac if strategy_name == "Kelly Criterion" else 0.25
value_plays_football = _fixtures_to_value_plays(
    fixtures_df, bankroll=starting_bankroll, kelly_frac=kelly_frac_val, min_ev_pct=5.0,
)
value_plays_football = add_injury_alerts_to_value_plays(value_plays_football, "Football")
# Basketball: live from API or mock
if (odds_api_key or "").strip():
    basketball_odds_df = get_basketball_odds(api_key=odds_api_key.strip())
else:
    basketball_odds_df = pd.DataFrame({
        "event_name": ["Lakers @ Celtics", "Warriors @ Nets", "Duke @ UNC", "Kentucky @ Kansas"],
        "market_type": ["spreads", "spreads", "totals", "totals"],
        "selection": ["Lakers -3.5", "Nets +2", "Over 145.5", "Under 138"],
        "odds": [-110, -108, -112, -105],  # American
    })
value_plays_basketball = _basketball_to_value_plays(
    basketball_odds_df, bankroll=starting_bankroll, kelly_frac=kelly_frac_val, min_ev_pct=5.0,
)
value_plays_basketball = add_injury_alerts_to_value_plays(value_plays_basketball, "Basketball")
value_plays_df = value_plays_basketball if sport_selector == "Basketball" else value_plays_football

# -----------------------------------------------------------------------------
# Main layout — tabbed (Overview, Football, Basketball)
# -----------------------------------------------------------------------------

st.title("Sports Betting Simulator")
st.caption("Strategy backtest • All odds American (+150, -110)")

combined_profit = (results_football["final_bankroll"] - starting_bankroll) + (results_basketball["final_bankroll"] - starting_bankroll)
curve_f = results_football["bankroll_curve"]
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

tab_overview, tab_football, tab_basketball = st.tabs(["Overview", "Football", "Basketball"])

with tab_overview:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Combined profit", f"${combined_profit:+,.2f}", f"From ${starting_bankroll:,.0f} total bankroll")
    with c2:
        st.metric("Football profit", f"${results_football['final_bankroll'] - starting_bankroll:+,.2f}", f"ROI {results_football['roi_pct']:.2f}%")
    with c3:
        st.metric("Basketball profit", f"${results_basketball['final_bankroll'] - starting_bankroll:+,.2f}", f"ROI {results_basketball['roi_pct']:.2f}%")
    st.subheader("Bankroll over time (by sport)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(curve_f))),
        y=curve_f,
        mode="lines+markers",
        name="Football",
        line=dict(color="#7ec8e3", width=2),
        marker=dict(size=4),
    ))
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_football:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Final bankroll", f"${results_football['final_bankroll']:,.2f}", f"{results_football['final_bankroll'] - starting_bankroll:+,.2f}")
    with c2:
        st.metric("ROI", f"{results_football['roi_pct']:.2f}%", "")
    with c3:
        st.metric("Bet count", results_football["total_bets"], f"Wins: {results_football['wins']} / Losses: {results_football['losses']}")
    st.subheader("Bet history")
    _bet_history_table(results_football["bet_history"])

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

# Today's Best Value Plays (EV > 5%, Kelly stake) — sport from sidebar + Injury Alert
st.subheader("Today's Best Value Plays")
st.caption(f"Showing **{sport_selector}** • EV > 5% • Injury Alert flags star players questionable (affects true probability).")
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
    st.info("No value plays today with EV > 5%. Try again later or adjust the threshold.")
