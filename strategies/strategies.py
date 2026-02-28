"""
Betting strategies: Kelly Criterion (optimal bet sizing) and Value Betting.
Works for moneyline, spreads, and totals (Over/Under); basketball commonly uses
spreads and totals, so value is evaluated the same way (decimal odds + model prob).
"""

from __future__ import annotations

import pandas as pd
from typing import Callable

# -----------------------------------------------------------------------------
# Core value logic (market-agnostic: moneyline, spread, or total)
# -----------------------------------------------------------------------------


def implied_probability(odds: float) -> float:
    """Convert decimal odds to implied probability. Works for any market (ML, spread, total)."""
    return 1.0 / odds if odds > 0 else 0.0


def is_value_bet(odds: float, model_prob: float) -> bool:
    """
    True when our model's probability exceeds the bookie's implied probability.
    We have edge when model_prob > 1/odds. Applies to moneyline, spreads, and totals.
    """
    return model_prob > implied_probability(odds)


def expected_value_pct(model_prob: float, odds: float) -> float:
    """Expected value as decimal: (model_prob * odds) - 1. Multiply by 100 for %. Same for ML/spread/total."""
    return (model_prob * odds) - 1.0


def find_value_bets(market_odds: float, model_probability: float) -> dict[str, float | bool]:
    """
    Evaluate a single bet for value and expected value.

    - Implied probability from decimal odds: 1 / odds.
    - Value when Model Probability > Implied Probability.
    - EV% formula: EV = (Model Prob × Odds) - 1 (returned as decimal; ×100 for %).

    Returns dict with: implied_probability, is_value, ev_pct (as decimal).
    Works for moneyline, basketball spreads, and totals (Over/Under).
    """
    impl = implied_probability(market_odds)
    is_value = model_probability > impl
    ev = expected_value_pct(model_probability, market_odds)
    return {
        "implied_probability": impl,
        "is_value": is_value,
        "ev_pct": ev,
    }


def value_bet_spread(odds: float, model_prob: float) -> dict[str, float | bool]:
    """Value and EV for a spread (point spread) line. Same logic as find_value_bets; use for basketball spreads."""
    return find_value_bets(odds, model_prob)


def value_bet_total(odds: float, model_prob: float) -> dict[str, float | bool]:
    """Value and EV for a total (Over/Under) line. Same logic as find_value_bets; use for basketball totals."""
    return find_value_bets(odds, model_prob)


def kelly_fraction(
    odds: float,
    model_prob: float,
    fraction: float = 0.25,
) -> float:
    """
    Optimal bet size as fraction of bankroll (Kelly Criterion).
    f* = (bp - q) / b  with b = odds - 1, p = model_prob, q = 1 - p.
    Returns 0 if no edge (f* <= 0). fraction caps full Kelly (e.g. 0.25 = quarter Kelly).
    """
    if odds <= 1.0:
        return 0.0
    b = odds - 1.0
    p = model_prob
    q = 1.0 - p
    edge = b * p - q
    if edge <= 0:
        return 0.0
    full_kelly = edge / b
    return min(full_kelly * fraction, 1.0)


def strategy_kelly(kelly_fraction_param: float = 0.25) -> Callable[[pd.Series, float], float]:
    """
    Returns a strategy function: bet only on value, stake = Kelly fraction of bankroll.
    Row must have 'odds' and 'model_prob'.
    """
    def _strategy(row: pd.Series, bankroll: float) -> float:
        odds = float(row["odds"])
        model_prob = float(row["model_prob"])
        if not is_value_bet(odds, model_prob) or bankroll <= 0:
            return 0.0
        frac = kelly_fraction(odds, model_prob, fraction=kelly_fraction_param)
        stake = bankroll * frac
        return max(0.0, round(stake, 2))
    return _strategy


def strategy_value_betting(flat_stake: float | None = None, stake_pct: float = 0.02) -> Callable[[pd.Series, float], float]:
    """
    Returns a strategy function: bet when model_prob > implied prob.
    If flat_stake is set, use that fixed stake; else use stake_pct * bankroll.
    Works for moneyline, spreads, and totals (same odds + model_prob).
    """
    def _strategy(row: pd.Series, bankroll: float) -> float:
        odds = float(row["odds"])
        model_prob = float(row["model_prob"])
        if not is_value_bet(odds, model_prob) or bankroll <= 0:
            return 0.0
        if flat_stake is not None and flat_stake > 0:
            stake = min(flat_stake, bankroll)
        else:
            stake = bankroll * stake_pct
        return max(0.0, round(stake, 2))
    return _strategy


def strategy_value_betting_basketball(
    flat_stake: float | None = None,
    stake_pct: float = 0.02,
    market_types: tuple[str, ...] = ("spreads", "totals"),
) -> Callable[[pd.Series, float], float]:
    """
    Value-betting strategy for basketball: prefers spreads and totals (Over/Under).
    Row must have 'odds' and 'model_prob'; if 'market_type' is present, only bet
    when market_type is in market_types (e.g. 'spreads', 'totals'). Same EV/value
    logic as strategy_value_betting.
    """
    def _strategy(row: pd.Series, bankroll: float) -> float:
        if "market_type" in row.index and row.get("market_type") not in market_types:
            return 0.0
        odds = float(row["odds"])
        model_prob = float(row["model_prob"])
        if not is_value_bet(odds, model_prob) or bankroll <= 0:
            return 0.0
        if flat_stake is not None and flat_stake > 0:
            stake = min(flat_stake, bankroll)
        else:
            stake = bankroll * stake_pct
        return max(0.0, round(stake, 2))
    return _strategy
