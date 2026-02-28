"""
Betting strategies: Kelly Criterion (optimal bet sizing) and Value Betting.
Uses American odds throughout. Works for moneyline, spreads, and totals (Over/Under).
"""

from __future__ import annotations

import pandas as pd
from typing import Callable

# -----------------------------------------------------------------------------
# American odds conversion (canonical format: +150, -110)
# -----------------------------------------------------------------------------


def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal. +150 -> 2.5, -110 -> ~1.909."""
    a = float(american)
    if a >= 100:
        return 1.0 + a / 100.0
    if a <= -100:
        return 1.0 + 100.0 / abs(a)
    return 1.0


def decimal_to_american(decimal: float) -> float:
    """Convert decimal odds to American. 2.5 -> +150, 1.909 -> -110."""
    d = float(decimal)
    if d <= 1.0:
        return 100.0
    if d >= 2.0:
        return (d - 1.0) * 100.0
    return -100.0 / (d - 1.0)


def implied_probability(odds_american: float) -> float:
    """Convert American odds to implied probability. Works for any market (ML, spread, total)."""
    dec = american_to_decimal(odds_american)
    return 1.0 / dec if dec > 0 else 0.0


# Vig adjustment: assume 4.5% sportsbook hold; no-vig implied prob for EV comparison
VIG_HOLD = 0.045


def implied_probability_no_vig(odds_american: float) -> float:
    """No-vig implied probability (fair market). Raw implied / (1 + vig hold)."""
    raw = implied_probability(odds_american)
    return raw / (1.0 + VIG_HOLD) if raw > 0 else 0.0


# Probability damping: pull extreme predictions toward league average (more realistic for NBA)
DAMPING_FACTOR = 0.8
LEAGUE_AVG_PROB = 0.5


def damp_probability(
    prob: float,
    league_avg: float = LEAGUE_AVG_PROB,
    damping: float = DAMPING_FACTOR,
) -> float:
    """Pull probability toward league average: damped = league_avg + damping * (prob - league_avg)."""
    return league_avg + damping * (float(prob) - league_avg)


def is_value_bet(odds_american: float, model_prob: float) -> bool:
    """
    True when our model's probability exceeds the bookie's implied probability.
    Odds are American. Applies to moneyline, spreads, and totals.
    """
    return model_prob > implied_probability(odds_american)


def expected_value_pct(model_prob: float, odds_american: float) -> float:
    """Expected value as decimal: (model_prob * decimal_odds) - 1. Odds are American."""
    dec = american_to_decimal(odds_american)
    return (model_prob * dec) - 1.0


def find_value_bets(market_odds_american: float, model_probability: float) -> dict[str, float | bool]:
    """
    Evaluate a single bet for value and expected value. Odds are American.

    - Implied probability from American odds.
    - Value when Model Probability > Implied Probability.
    - EV% formula: EV = (Model Prob × Decimal Odds) - 1 (returned as decimal; ×100 for %).

    Returns dict with: implied_probability, is_value, ev_pct (as decimal).
    """
    impl = implied_probability(market_odds_american)
    is_value = model_probability > impl
    ev = expected_value_pct(model_probability, market_odds_american)
    return {
        "implied_probability": impl,
        "is_value": is_value,
        "ev_pct": ev,
    }


def value_bet_spread(odds_american: float, model_prob: float) -> dict[str, float | bool]:
    """Value and EV for a spread (point spread) line. Odds are American."""
    return find_value_bets(odds_american, model_prob)


def value_bet_total(odds_american: float, model_prob: float) -> dict[str, float | bool]:
    """Value and EV for a total (Over/Under) line. Odds are American."""
    return find_value_bets(odds_american, model_prob)


# -----------------------------------------------------------------------------
# NBA Pace-Adjusted Totals
# -----------------------------------------------------------------------------

# League averages when team not in pace_stats (must match engine.NBA_LEAGUE_AVG_* if no data)
_NBA_DEFAULT_PACE = 100.0
_NBA_DEFAULT_OFF_RATING = 115.0


def predict_nba_total(
    home_team: str,
    away_team: str,
    pace_stats: dict[str, dict[str, float]],
    default_pace: float = _NBA_DEFAULT_PACE,
    default_off_rating: float = _NBA_DEFAULT_OFF_RATING,
    b2b_teams: set[str] | None = None,
    b2b_pace_mult: float = 0.985,
    b2b_off_rating_mult: float = 0.98,
) -> float:
    """
    Pace-adjusted projected game total.
    Projected Pace = avg of both teams' Pace (possessions/48m).
    Projected Score = (Projected Pace * Team OffRating) / 100 per team.
    If a team is in b2b_teams (played last night), their Pace is reduced by (1 - b2b_pace_mult)
    and Off Rating by (1 - b2b_off_rating_mult) for fatigue. Default: -1.5% pace, -2% off_rating.
    Returns sum of home and away projected scores.
    """
    h = pace_stats.get(home_team) or {"pace": default_pace, "off_rating": default_off_rating}
    a = pace_stats.get(away_team) or {"pace": default_pace, "off_rating": default_off_rating}
    b2b = b2b_teams or set()
    if home_team in b2b:
        h = {"pace": h["pace"] * b2b_pace_mult, "off_rating": h["off_rating"] * b2b_off_rating_mult}
    if away_team in b2b:
        a = {"pace": a["pace"] * b2b_pace_mult, "off_rating": a["off_rating"] * b2b_off_rating_mult}
    projected_pace = (h["pace"] + a["pace"]) / 2.0
    home_pts = (projected_pace * h["off_rating"]) / 100.0
    away_pts = (projected_pace * a["off_rating"]) / 100.0
    return home_pts + away_pts


def get_totals_value(
    predicted_total: float,
    market_total: float,
    threshold: float = 3.0,
) -> str | None:
    """
    Compare predicted total to bookmaker O/U line.
    Returns "over" if prediction > market + threshold (Value Over),
    "under" if prediction < market - threshold (Value Under), else None.
    """
    if predicted_total > market_total + threshold:
        return "over"
    if predicted_total < market_total - threshold:
        return "under"
    return None


def kelly_fraction(
    odds_american: float,
    model_prob: float,
    fraction: float = 0.25,
) -> float:
    """
    Optimal bet size as fraction of bankroll (Kelly Criterion). Odds are American.
    f* = (bp - q) / b  with b = decimal_odds - 1, p = model_prob, q = 1 - p.
    Returns 0 if no edge (f* <= 0). fraction caps full Kelly (e.g. 0.25 = quarter Kelly).
    """
    odds_dec = american_to_decimal(odds_american)
    if odds_dec <= 1.0:
        return 0.0
    b = odds_dec - 1.0
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
    Row must have 'odds' (American) and 'model_prob'.
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
    Row 'odds' are American. If flat_stake set use it; else stake_pct * bankroll.
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
    Value-betting strategy for basketball: prefers spreads and totals.
    Row 'odds' are American. Same EV/value logic as strategy_value_betting.
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
