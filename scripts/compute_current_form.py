#!/usr/bin/env python3
"""
Compute current-season form (last 5 / last 10 margin and win%) for each team.

Reads data/ncaab/historical_games.csv, filters to the current 2026 season,
and for each team computes last 5 and last 10 game averages as of today.
Saves to data/ncaab/current_form_2026.csv with columns:
  team, last5_margin, last10_margin, last5_winpct

Used by predict_games.py to populate XGBoost form features for live predictions.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import pandas as pd

from engine.utils import game_season_from_date

NCAAB_DIR = APP_ROOT / "data" / "ncaab"
HISTORICAL_GAMES_PATH = NCAAB_DIR / "historical_games.csv"
OUTPUT_PATH = NCAAB_DIR / "current_form_2026.csv"
CURRENT_SEASON = 2026


def main() -> None:
    if not HISTORICAL_GAMES_PATH.exists():
        print(f"Missing {HISTORICAL_GAMES_PATH}", file=sys.stderr)
        sys.exit(1)

    games = pd.read_csv(HISTORICAL_GAMES_PATH)
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    games = games.dropna(subset=["date"])
    games["season"] = games["date"].apply(lambda d: game_season_from_date(d))
    games = games[games["season"] == CURRENT_SEASON].copy()
    if games.empty:
        pd.DataFrame(columns=["team", "last5_margin", "last10_margin", "last5_winpct"]).to_csv(
            OUTPUT_PATH, index=False
        )
        print(f"No {CURRENT_SEASON} season games. Wrote empty {OUTPUT_PATH}")
        return

    # As of today: only games before today
    today = pd.Timestamp.now().normalize()
    games = games[games["date"] < today].copy()
    games = games.sort_values("date")

    # Per-team history: list of (margin from team POV, win 0/1)
    team_hist: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for _, row in games.iterrows():
        home = str(row.get("home_team", "")).strip()
        away = str(row.get("away_team", "")).strip()
        margin = float(row.get("margin", 0))
        if not home or not away:
            continue
        team_hist[home].append((margin, 1.0 if margin > 0 else 0.0))
        team_hist[away].append((-margin, 1.0 if margin < 0 else 0.0))

    def avg_margin_winpct(hist: list, n: int) -> tuple[float, float]:
        if not hist:
            return 0.0, 0.0
        take = hist[-n:] if len(hist) >= n else hist
        margins = [x[0] for x in take]
        wins = [x[1] for x in take]
        avg_m = sum(margins) / len(margins)
        winpct = sum(wins) / len(wins)
        return avg_m, winpct

    rows = []
    for team in sorted(team_hist.keys()):
        hist = team_hist[team]
        last5_m, last5_w = avg_margin_winpct(hist, 5)
        last10_m, last10_w = avg_margin_winpct(hist, 10)
        rows.append({
            "team": team,
            "last5_margin": round(last5_m, 4),
            "last10_margin": round(last10_m, 4),
            "last5_winpct": round(last5_w, 4),
        })

    out_df = pd.DataFrame(rows)
    NCAAB_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(out_df)} teams to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
