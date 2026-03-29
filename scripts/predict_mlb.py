#!/usr/bin/env python3
"""
Generate data/cache/mlb_value_plays.json from data/odds/live_mlb_odds.json
and data/mlb/team_stats.csv (from scripts/fetch_mlb_stats.py).

Requires fetch_mlb_odds.py to have run so live_mlb_odds.json exists.

card_date: prefers games_date_et from live odds (MLB calendar day in ET); else America/New_York today.
"""

from __future__ import annotations

import json
import math
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*urllib3 v2 only supports OpenSSL.*")
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ODDS_PATH = ROOT / "data" / "odds" / "live_mlb_odds.json"
OUT_PATH = ROOT / "data" / "cache" / "mlb_value_plays.json"

# Temporary: 0% minimum edge (decimal EV >= 0) to verify odds + stats → plays.
MIN_EDGE_DECIMAL = 0.0

# Schedule/API sometimes uses a short nickname; CSV uses full MLB Stats API team_name.
SCHEDULE_NAME_TO_STATS_NAME: dict[str, str] = {
    "Athletics": "Oakland Athletics",
    "Angels": "Los Angeles Angels",
    "Dodgers": "Los Angeles Dodgers",
    "Cubs": "Chicago Cubs",
    "White Sox": "Chicago White Sox",
    "Mets": "New York Mets",
    "Yankees": "New York Yankees",
    "Red Sox": "Boston Red Sox",
    "Blue Jays": "Toronto Blue Jays",
    "Guardians": "Cleveland Guardians",
    "Rays": "Tampa Bay Rays",
}


def _card_date_iso(blob: dict) -> str:
    gde = str(blob.get("games_date_et") or "").strip()
    if gde:
        return gde
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def _american_to_float(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v > 0 and v < 100:
        return None
    if v < 0 and v > -100:
        return None
    return v


def _find_team_row(stats_df, raw_name: str):
    """Return one stats row for this team name, or None."""
    from engine.mlb_engine import normalize_mlb_team_name_for_join

    if stats_df is None or stats_df.empty or "team_name" not in stats_df.columns:
        return None
    norm = normalize_mlb_team_name_for_join(raw_name)
    if not norm:
        return None
    full = SCHEDULE_NAME_TO_STATS_NAME.get(norm, norm)
    m = stats_df[stats_df["team_name"].astype(str).str.strip() == full]
    if len(m) == 1:
        return m.iloc[0]
    m = stats_df[stats_df["team_name"].astype(str).str.strip() == norm]
    if len(m) == 1:
        return m.iloc[0]
    suf = stats_df[stats_df["team_name"].astype(str).str.endswith(" " + norm, na=False)]
    if len(suf) == 1:
        return suf.iloc[0]
    return None


def _team_strength(row) -> float:
    """Higher = better team (rough heuristic from era / woba)."""
    try:
        era = float(row.get("era"))
    except (TypeError, ValueError):
        era = float("nan")
    try:
        woba = float(row.get("woba"))
    except (TypeError, ValueError):
        woba = float("nan")
    if era != era:  # NaN
        era = 4.2
    if woba != woba:
        woba = 0.305
    era = min(max(era, 2.8), 6.5)
    woba = min(max(woba, 0.26), 0.38)
    return (4.2 - era) * 0.28 + (woba - 0.305) * 42.0


def _home_win_prob(home_row, away_row) -> float:
    rh = _team_strength(home_row)
    ra = _team_strength(away_row)
    diff = (rh - ra) + 0.05
    return 1.0 / (1.0 + math.exp(-diff * 3.2))


def main() -> int:
    import pandas as pd

    from engine.mlb_engine import (
        DEFAULT_MLB_TEAM_STATS_CSV,
        load_live_mlb_odds,
        value_summary_moneyline,
    )

    if not ODDS_PATH.exists():
        print("Missing data/odds/live_mlb_odds.json — run scripts/fetch_mlb_odds.py first.", file=sys.stderr)
        return 1

    blob = load_live_mlb_odds(ODDS_PATH)
    card_date = _card_date_iso(blob)

    stats_path = DEFAULT_MLB_TEAM_STATS_CSV
    if not stats_path.exists():
        print(
            f"MLB predict: no team stats at {stats_path} — matched 0 games (run scripts/fetch_mlb_stats.py).",
            file=sys.stderr,
        )
        stats_df = pd.DataFrame()
    else:
        try:
            stats_df = pd.read_csv(stats_path)
        except Exception as e:
            print(f"MLB predict: failed to read team stats: {e} — matched 0 games.", file=sys.stderr)
            stats_df = pd.DataFrame()

    games = blob.get("games") or []
    matched = 0
    plays: list[dict] = []

    for g in games:
        if not isinstance(g, dict):
            continue
        home = str(g.get("home_team") or "").strip()
        away = str(g.get("away_team") or "").strip()
        ml = g.get("moneyline") or {}
        ho = _american_to_float(ml.get("home_odds"))
        ao = _american_to_float(ml.get("away_odds"))

        hr = _find_team_row(stats_df, home)
        ar = _find_team_row(stats_df, away)
        if hr is not None and ar is not None:
            matched += 1

        if hr is None or ar is None or ho is None or ao is None:
            continue

        p_home = _home_win_prob(hr, ar)
        vh = value_summary_moneyline(p_home, ho, ao)
        va = value_summary_moneyline(1.0 - p_home, ao, ho)
        edge_h = float(vh["edge"])
        edge_a = float(va["edge"])

        if edge_h >= edge_a:
            if edge_h < MIN_EDGE_DECIMAL:
                continue
            pick, odds_am, model_p, edge = home, ho, p_home, edge_h
        else:
            if edge_a < MIN_EDGE_DECIMAL:
                continue
            pick, odds_am, model_p, edge = away, ao, 1.0 - p_home, edge_a

        plays.append(
            {
                "card_date": card_date,
                "event_id": str(g.get("event_id") or ""),
                "commence_time": str(g.get("commence_time") or ""),
                "home_team": home,
                "away_team": away,
                "home_pitcher": g.get("home_pitcher"),
                "away_pitcher": g.get("away_pitcher"),
                "market": "moneyline",
                "selection": pick,
                "odds_american": float(odds_am),
                "model_prob": float(model_p),
                "edge": float(edge),
            }
        )

    print(
        f"MLB predict: matched {matched} game(s) between odds slate and team_stats (both teams found).",
        flush=True,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"card_date": card_date, "plays": plays}
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {OUT_PATH} ({len(plays)} play(s)), card_date={card_date}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
