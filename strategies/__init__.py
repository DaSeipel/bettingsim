"""Betting strategies."""
from .strategies import (
    kelly_fraction,
    is_value_bet,
    strategy_kelly,
    strategy_value_betting,
)

__all__ = ["kelly_fraction", "is_value_bet", "strategy_kelly", "strategy_value_betting"]
