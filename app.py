"""
Sports Betting Simulator - Streamlit Dashboard
Dark theme, strategy/bankroll sidebar, bankroll chart, bet history table.
Uses engine + strategies with mock data.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from engine.engine import BettingEngine
from strategies.strategies import strategy_kelly, strategy_value_betting


st.set_page_config(
    page_title="Sports Betting Simulator",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
)


def generate_mock_dataset(n_rows: int = 80, seed: int = 42) -> pd.DataFrame:
    """Mock historical dataset: odds, model_prob, result (1=win, 0=loss)."""
    np.random.seed(seed)
    odds = np.round(np.random.uniform(1.6, 3.2, n_rows), 2)
    # Sometimes model disagrees with book (value opportunities)
    implied = 1.0 / odds
    model_prob = np.clip(
        implied + np.random.randn(n_rows) * 0.08,
        0.15,
        0.85,
    )
    model_prob = np.round(model_prob, 3)
    # Simulate outcomes with slight favor to model when it has edge
    edge = model_prob - implied
    win_prob = np.clip(0.5 + edge * 2, 0.2, 0.8)
    result = (np.random.random(n_rows) < win_prob).astype(int)
    event_names = [f"Match {i+1}" for i in range(n_rows)]
    return pd.DataFrame({
        "event_id": np.arange(n_rows),
        "event_name": event_names,
        "odds": odds,
        "model_prob": model_prob,
        "result": result,
    })


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

# -----------------------------------------------------------------------------
# Mock data and engine run (reruns when sidebar inputs change)
# -----------------------------------------------------------------------------

mock_df = generate_mock_dataset()
engine = BettingEngine(mock_df, strategy_fn, starting_bankroll)
results = engine.run()

# -----------------------------------------------------------------------------
# Main layout
# -----------------------------------------------------------------------------

st.title("Sports Betting Simulator")
st.caption("Strategy backtest over historical (mock) data • Dark theme")

# Metrics row
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Final bankroll", f"${results['final_bankroll']:,.2f}", f"{results['final_bankroll'] - starting_bankroll:+,.2f}")
with col2:
    st.metric("Wins / Losses", f"{results['wins']} / {results['losses']}", f"Bets: {results['total_bets']}")
with col3:
    st.metric("Total staked", f"${results['total_staked']:,.2f}", "")
with col4:
    st.metric("Total profit", f"${results['total_profit']:,.2f}", "")
with col5:
    st.metric("ROI", f"{results['roi_pct']:.2f}%", "")

# Bankroll over time — Plotly line chart
st.subheader("Bankroll over time")
steps = list(range(len(results["bankroll_curve"])))
fig = go.Figure(
    data=go.Scatter(
        x=steps,
        y=results["bankroll_curve"],
        mode="lines+markers",
        line=dict(color="#7ec8e3", width=2),
        marker=dict(size=4),
        fill="tozeroy",
        fillcolor="rgba(126, 200, 227, 0.15)",
    )
)
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

# Bet history table
st.subheader("Bet history")
if results["bet_history"]:
    bet_df = pd.DataFrame(results["bet_history"])
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
    st.dataframe(
        bet_df[["Step", "Event", "Odds", "Model prob", "Stake", "Result", "P/L", "Bankroll after"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Step": st.column_config.NumberColumn("Step", width="small"),
            "Event": st.column_config.TextColumn("Event", width="medium"),
            "Odds": st.column_config.NumberColumn("Odds", format="%.2f", width="small"),
            "Model prob": st.column_config.NumberColumn("Model prob", format="%.2f", width="small"),
            "Stake": st.column_config.NumberColumn("Stake", format="%.2f", width="small"),
            "Result": st.column_config.TextColumn("Result", width="small"),
            "P/L": st.column_config.NumberColumn("P/L", format="%.2f", width="small"),
            "Bankroll after": st.column_config.NumberColumn("Bankroll after", format="%.2f", width="small"),
        },
    )
else:
    st.info("No bets placed with current strategy and data. Adjust strategy or bankroll.")
