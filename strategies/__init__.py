"""Betting strategies. All odds are American (+150, -110)."""
from .strategies import (
    american_to_decimal,
    decimal_to_american,
    implied_probability_fair_two_sided,
    kelly_fraction,
    is_value_bet,
    find_value_bets,
    expected_value_pct,
    value_bet_spread,
    value_bet_total,
    strategy_kelly,
    strategy_value_betting,
    strategy_value_betting_basketball,
)

__all__ = [
    "american_to_decimal",
    "decimal_to_american",
    "implied_probability_fair_two_sided",
    "kelly_fraction",
    "is_value_bet",
    "find_value_bets",
    "expected_value_pct",
    "value_bet_spread",
    "value_bet_total",
    "strategy_kelly",
    "strategy_value_betting",
    "strategy_value_betting_basketball",
]
