"""
Betting strategies: Kelly Criterion (optimal bet sizing) and Value Betting.
Uses American odds throughout. Works for moneyline, spreads, and totals (Over/Under).
"""

from __future__ import annotations

import pandas as pd
from typing import Callable

# -----------------------------------------------------------------------------
# American odds conversion (canonical format: +150, -110)
# Standard formulas: +X -> implied = 100/(X+100); -X -> implied = |X|/(|X|+100).
# Equivalently: decimal = 1 + 100/|american| for negative, 1 + american/100 for positive;
# implied = 1 / decimal.
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
    """
    Convert American odds to implied probability. Works for any market (ML, spread, total).
    Formula: implied = 1 / decimal_odds (equivalent to 100/(X+100) for +X, |X|/(|X|+100) for -X).
    """
    dec = american_to_decimal(odds_american)
    return 1.0 / dec if dec > 0 else 0.0


# Vig removal: approximate fair probability by scaling down book implied.
# Typical overround is ~4.5%; we use fair ≈ raw_implied / (1 + VIG_HOLD).
# For exact no-vig across two outcomes use implied_probability_fair_two_sided.
VIG_HOLD = 0.045


def implied_probability_no_vig(odds_american: float) -> float:
    """No-vig implied probability (fair market). Raw implied / (1 + vig hold). Use only when other side unavailable."""
    raw = implied_probability(odds_american)
    return raw / (1.0 + VIG_HOLD) if raw > 0 else 0.0


def implied_probability_fair_two_sided(odds_american_a: float, odds_american_b: float) -> tuple[float, float]:
    """
    Standard two-outcome vig removal: implied = 1 / decimal_odds for each side,
    then normalize so both sum to 100%. Returns (fair_a, fair_b).
    """
    impl_a = implied_probability(odds_american_a)
    impl_b = implied_probability(odds_american_b)
    total = impl_a + impl_b
    if total <= 0:
        return (0.0, 0.0)
    return (impl_a / total, impl_b / total)


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


# -----------------------------------------------------------------------------
# Billy Walters-style: Key Numbers, Fractional Kelly (1–3%), In-House Line
# -----------------------------------------------------------------------------

# Key numbers: crossing these increases push/win probability (NFL 3/7, NBA common spreads/totals)
KEY_NUMBERS_NFL_SPREAD = {3, 7}
KEY_NUMBERS_NBA_SPREAD = {2.5, 3, 5, 5.5, 7, 7.5}
KEY_NUMBERS_NBA_TOTAL = {210, 215, 220, 225, 230, 235, 240}
KEY_NUMBER_ADJUSTMENT_PCT = -0.3  # Slight reduction in Value % when line sits on key (push risk)


def key_number_value_adjustment(
    market_line: float,
    sport: str,
    market_type: str,
) -> float:
    """
    If the market line is on or just across a key number, adjust Value % (push/win probability).
    Returns adjustment in percentage points to add to Value % (usually small negative).
    """
    if market_type == "totals":
        keys = KEY_NUMBERS_NBA_TOTAL if "nba" in sport.lower() or "basketball" in sport.lower() else set()
    else:
        keys = KEY_NUMBERS_NBA_SPREAD if "nba" in sport.lower() or "basketball" in sport.lower() else KEY_NUMBERS_NFL_SPREAD
    for k in keys:
        if abs(market_line - k) < 0.6:
            return KEY_NUMBER_ADJUSTMENT_PCT
    return 0.0


# Walters: 1%–3% of bankroll; round down to half-units (0.5%) to protect vs variance
HALF_UNIT_PCT = 0.005
MAX_KELLY_PCT_WALTERS = 0.03


def fractional_kelly_half_units(
    bankroll: float,
    kelly_fraction_decimal: float,
    unit_pct: float = HALF_UNIT_PCT,
    max_pct: float = MAX_KELLY_PCT_WALTERS,
) -> float:
    """
    Fractional Kelly rounded down to half-units (0.5% of bankroll). Cap at max_pct (e.g. 3%).
    Returns stake in dollars.
    """
    if bankroll <= 0 or kelly_fraction_decimal <= 0:
        return 0.0
    raw_pct = min(kelly_fraction_decimal, max_pct)
    half_units = int(raw_pct / unit_pct)
    if half_units < 1:
        return 0.0
    stake_pct = half_units * unit_pct
    return round(bankroll * stake_pct, 2)


def in_house_spread_from_ratings(home_rating: float, away_rating: float) -> float:
    """In-house spread: positive = home favored. home_rating - away_rating."""
    return home_rating - away_rating


def model_prob_from_in_house_total(in_house_total: float, market_total: float, prefer_over: bool) -> float:
    """
    Approximate probability we beat the market on a total (Over/Under).
    prefer_over True = we're betting Over. Based on how much in-house differs from market.
    """
    diff = in_house_total - market_total
    if prefer_over and diff <= 0:
        return 0.45
    if not prefer_over and diff >= 0:
        return 0.45
    edge_pts = abs(diff)
    if edge_pts >= 10:
        return 0.62
    if edge_pts >= 5:
        return 0.56
    if edge_pts >= 3:
        return 0.53
    return 0.51


def model_prob_from_in_house_spread(in_house_spread: float, market_spread: float, we_cover_favorite: bool) -> float:
    """
    Approximate probability we cover the spread. market_spread is e.g. -3.5 (home -3.5).
    we_cover_favorite True = we're on the favorite (home/away side that's favored).
    """
    # Our line says home by in_house_spread; market says home by market_spread (negative = away favored).
    if we_cover_favorite and in_house_spread <= market_spread:
        return 0.45
    if not we_cover_favorite and in_house_spread >= market_spread:
        return 0.45
    edge = abs(in_house_spread - market_spread)
    if edge >= 5:
        return 0.58
    if edge >= 3:
        return 0.54
    if edge >= 1.5:
        return 0.52
    return 0.51


def model_prob_from_ratings_moneyline(
    home_rating: float, away_rating: float, selection_is_home: bool
) -> float:
    """
    Win probability for moneyline (h2h) from the same power ratings used for spreads.
    in_house_spread = home_rating - away_rating (positive = home favored).
    Converts spread to win prob: ~3% per point (e.g. 3-pt favorite ≈ 59%).
    """
    in_house_spread = home_rating - away_rating
    # ~3% win probability per point of spread; cap to [0.02, 0.98]
    home_win_prob = max(0.02, min(0.98, 0.5 + in_house_spread * 0.03))
    return home_win_prob if selection_is_home else (1.0 - home_win_prob)


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
