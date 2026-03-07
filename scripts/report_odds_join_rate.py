#!/usr/bin/env python3
"""
Report closing-line join rate by league (NBA, NCAAB) from games_with_team_stats.
Optionally run merge first so the table is up to date. Writes unmatched NCAAB rows to data/raw_odds/ncaab_odds_unmatched.csv.
Usage: python scripts/report_odds_join_rate.py [--merge]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.historical_odds import merge_historical_closing_into_games, report_closing_join_rate_by_league

DATA_DIR = ROOT / "data"


def main() -> int:
    argv = [a for a in sys.argv[1:] if a.startswith("--")]
    do_merge = "--merge" in argv
    espn_db = DATA_DIR / "espn.db"
    odds_db = DATA_DIR / "odds.db"
    if do_merge:
        if not odds_db.exists():
            print("odds.db not found; run pipeline with historical odds first.")
            return 1
        print("Merging historical closing lines into games_with_team_stats...")
        merge_historical_closing_into_games(espn_db_path=espn_db, odds_db_path=odds_db)
        print("Done. Join rate by league:")
    else:
        print("Join rate by league (use --merge to refresh merge first):")
    report_closing_join_rate_by_league(espn_db_path=espn_db, write_unmatched_ncaab_csv=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
