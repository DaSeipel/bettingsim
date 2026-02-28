"""
Betting strategies: Kelly Criterion (optimal bet sizing) and Value Betting.
"""

from __future__ import annotations

import pandas as pd
from typing import Callable


def implied_probability(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    return 1.0 / odds if odds > 0 else 0.0


def is_value_bet(odds: float, model_prob: float) -> bool:
    """
    True when our model's probability exceeds the bookie's implied probability.
    We have edge when model_prob > 1/odds.
    """
    return model_prob > implied_probability(odds)


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
