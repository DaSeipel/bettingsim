#!/usr/bin/env python3
"""Test ESPN scoreboard for NCAAB: print URL and how many games for March 7."""
from datetime import date
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.espn_odds import (
    _fetch_scoreboard,
    _scoreboard_url,
    get_espn_live_odds_with_stats,
)

def main() -> None:
    session = __import__("requests").Session()
    march_7 = date(2026, 3, 7)
    url = _scoreboard_url("mens-college-basketball", march_7)
    data = _fetch_scoreboard("mens-college-basketball", session, march_7)
    events = (data or {}).get("events") or []
    print("Scoreboard URL (March 7, 2026):", url)
    print("NCAAB games in scoreboard response for March 7:", len(events))
    # Also run full fetch with today = March 7 (no future filter bypass)
    df, n_games, n_rows = get_espn_live_odds_with_stats(
        sport_keys=["basketball_ncaab"],
        commence_on_date=march_7,
    )
    print("NCAAB games after filters (today ET + future):", n_games, "(0 expected if run after March 7)")


if __name__ == "__main__":
    main()
