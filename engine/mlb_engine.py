"""
MLB betting helpers: moneyline and totals, edge vs implied probability.

Live odds are read from data/odds/live_mlb_odds.json (populated by
scripts/fetch_mlb_odds.py: statsapi schedule + web-scraped consensus lines). The engine does not call external APIs.

Edge (expected value as decimal) matches strategies.expected_value_pct:
  Edge = (model_prob * decimal_odds) - 1

Kelly staking uses strategies.kelly_fraction / strategy_kelly (American odds).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from strategies.strategies import (
    american_to_decimal,
    expected_value_pct,
    implied_probability,
    implied_probability_fair_two_sided,
    is_value_bet,
    kelly_fraction,
    strategy_kelly,
)

MarketMLB = Literal["moneyline", "total"]

DEFAULT_LIVE_MLB_ODDS_PATH = Path(__file__).resolve().parent.parent / "data" / "odds" / "live_mlb_odds.json"
# Same path as scripts/fetch_mlb_odds.py OUTPUT_PATH (project root data/odds/live_mlb_odds.json).
DEFAULT_MLB_TEAM_STATS_CSV = Path(__file__).resolve().parent.parent / "data" / "mlb" / "team_stats.csv"

# Map alternate API spellings → name used in data/mlb/team_stats.csv (from MLB Stats API teams.name).
# Both API-Sports and MLB Stats API usually use full names like "New York Mets"; add rows here if joins fail.
MLB_TEAM_NAME_ALIASES: dict[str, str] = {
    "Arizona D-backs": "Arizona Diamondbacks",
}


def normalize_mlb_team_name_for_join(name: str) -> str:
    """
    Normalize a team string for merging odds JSON (`home_team` / `away_team`) with `team_stats.csv`
    (`team_name`). Strips whitespace, applies MLB_TEAM_NAME_ALIASES, keeps case as in stats file.
    """
    s = " ".join(str(name).split()).strip()
    if not s:
        return s
    return MLB_TEAM_NAME_ALIASES.get(s, s)


def load_live_mlb_odds(path: Path | str | None = None) -> dict[str, Any]:
    """
    Primary source for market odds: cached JSON written by scripts/fetch_mlb_odds.py.
    Returns a dict with keys like: fetched_at_utc, sport_key, games (list of game dicts with
    moneyline and total). Missing file returns empty games and error hint.
    """
    p = Path(path) if path else DEFAULT_LIVE_MLB_ODDS_PATH
    if not p.exists():
        return {
            "fetched_at_utc": None,
            "sport_key": "baseball_mlb",
            "games": [],
            "error": "missing_file",
            "path": str(p),
        }
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return {
            "fetched_at_utc": None,
            "sport_key": "baseball_mlb",
            "games": [],
            "error": str(e),
            "path": str(p),
        }
    if not isinstance(data, dict):
        return {"fetched_at_utc": None, "games": [], "error": "invalid_json_shape"}
    data.setdefault("games", [])
    return data


def live_mlb_odds_dataframe(path: Path | str | None = None) -> pd.DataFrame:
    """
    Flatten live_mlb_odds.json into a DataFrame similar to basketball odds rows:
    event_id, commence_time, home_team, away_team, market_type (h2h|totals),
    selection, point, odds (American).
    """
    blob = load_live_mlb_odds(path)
    rows: list[dict[str, Any]] = []
    for g in blob.get("games") or []:
        if not isinstance(g, dict):
            continue
        eid = g.get("event_id", "")
        commence = g.get("commence_time", "")
        home = g.get("home_team", "")
        away = g.get("away_team", "")
        ml = g.get("moneyline") or {}
        ho = ml.get("home_odds")
        ao = ml.get("away_odds")
        if ho is not None:
            rows.append(
                {
                    "sport_key": blob.get("sport_key", "baseball_mlb"),
                    "league": "MLB",
                    "event_id": eid,
                    "commence_time": commence,
                    "home_team": home,
                    "away_team": away,
                    "event_name": f"{away} @ {home}",
                    "market_type": "h2h",
                    "selection": home,
                    "point": None,
                    "odds": ho,
                }
            )
        if ao is not None:
            rows.append(
                {
                    "sport_key": blob.get("sport_key", "baseball_mlb"),
                    "league": "MLB",
                    "event_id": eid,
                    "commence_time": commence,
                    "home_team": home,
                    "away_team": away,
                    "event_name": f"{away} @ {home}",
                    "market_type": "h2h",
                    "selection": away,
                    "point": None,
                    "odds": ao,
                }
            )
        tot = g.get("total") or {}
        line = tot.get("line")
        over_o = tot.get("over_odds")
        under_o = tot.get("under_odds")
        if over_o is not None and line is not None:
            rows.append(
                {
                    "sport_key": blob.get("sport_key", "baseball_mlb"),
                    "league": "MLB",
                    "event_id": eid,
                    "commence_time": commence,
                    "home_team": home,
                    "away_team": away,
                    "event_name": f"{away} @ {home}",
                    "market_type": "totals",
                    "selection": f"Over {line}",
                    "point": float(line),
                    "odds": over_o,
                }
            )
        if under_o is not None and line is not None:
            rows.append(
                {
                    "sport_key": blob.get("sport_key", "baseball_mlb"),
                    "league": "MLB",
                    "event_id": eid,
                    "commence_time": commence,
                    "home_team": home,
                    "away_team": away,
                    "event_name": f"{away} @ {home}",
                    "market_type": "totals",
                    "selection": f"Under {line}",
                    "point": float(line),
                    "odds": under_o,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "sport_key",
                "league",
                "event_id",
                "commence_time",
                "home_team",
                "away_team",
                "event_name",
                "market_type",
                "selection",
                "point",
                "odds",
            ]
        )
    return pd.DataFrame(rows)


def edge_vs_implied(model_prob: float, odds_american: float) -> float:
    """
    Expected value as a decimal (not percent): (model_prob × decimal_odds) − 1,
    where decimal_odds comes from american_to_decimal(odds_american). Same as
    strategies.expected_value_pct. Example: EV = 0.1122 ⇒ edge_pct display = 11.22 (% points).
    """
    return expected_value_pct(model_prob, odds_american)


def value_summary_moneyline(
    model_prob_side: float,
    odds_american_side: float,
    odds_american_other: float | None = None,
) -> dict[str, float | bool]:
    """
    Moneyline value metrics. If both sides' American odds are given, implied uses two-way no-vig normalization.
    Otherwise uses single-side implied from strategies.implied_probability.
    """
    if odds_american_other is not None:
        fair_p, _ = implied_probability_fair_two_sided(odds_american_side, odds_american_other)
        impl = fair_p
        dec = american_to_decimal(odds_american_side)
        ev = model_prob_side * dec - 1.0
        is_val = model_prob_side > impl
    else:
        impl = implied_probability(odds_american_side)
        ev = edge_vs_implied(model_prob_side, odds_american_side)
        is_val = is_value_bet(odds_american_side, model_prob_side)
    return {
        "implied_probability": impl,
        "model_probability": model_prob_side,
        "edge": ev,  # decimal EV; edge_pct below is percentage points (×100)
        "edge_pct": ev * 100.0,
        "is_value": bool(is_val),
        "decimal_odds": american_to_decimal(odds_american_side),
    }


def value_summary_total(
    model_prob_over: float,
    odds_american_over: float,
    odds_american_under: float,
) -> dict[str, Any]:
    """Totals: no-vig fair prob for Over, edge on Over; Under is complementary."""
    fair_over, fair_under = implied_probability_fair_two_sided(odds_american_over, odds_american_under)
    dec_o = american_to_decimal(odds_american_over)
    dec_u = american_to_decimal(odds_american_under)
    ev_over = model_prob_over * dec_o - 1.0
    ev_under = (1.0 - model_prob_over) * dec_u - 1.0
    return {
        "fair_implied_over": fair_over,
        "fair_implied_under": fair_under,
        "model_prob_over": model_prob_over,
        "model_prob_under": 1.0 - model_prob_over,
        "edge_over": ev_over,
        "edge_under": ev_under,
        "edge_over_pct": ev_over * 100.0,
        "edge_under_pct": ev_under * 100.0,
        "is_value_over": model_prob_over > fair_over,
        "is_value_under": (1.0 - model_prob_over) > fair_under,
    }


def kelly_stake_dollars(
    odds_american: float,
    model_prob: float,
    bankroll: float,
    kelly_multiplier: float = 0.25,
) -> float:
    """Recommended stake in dollars using fractional Kelly from strategies."""
    row = pd.Series({"odds": float(odds_american), "model_prob": float(model_prob)})
    strat = strategy_kelly(kelly_fraction_param=kelly_multiplier)
    return strat(row, float(bankroll))


def kelly_fraction_for_side(odds_american: float, model_prob: float, fraction: float = 0.25) -> float:
    """Expose Kelly fraction of bankroll (0–1) for MLB plays."""
    return kelly_fraction(odds_american, model_prob, fraction=fraction)
