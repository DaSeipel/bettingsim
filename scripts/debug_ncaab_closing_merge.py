#!/usr/bin/env python3
"""Debug why merge_historical_closing_into_games returns 0 NCAAB matches. Prints (1)-(3) then runs fix and new count."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

def main() -> int:
    odds_db = ROOT / "data" / "odds.db"
    espn_db = ROOT / "data" / "espn.db"

    # (1) First 5 rows of historical_covers_odds where sport='ncaab'
    print("(1) historical_covers_odds (sport=ncaab) — first 5 rows:")
    print("    Columns: date, home_team, away_team. Exact date format (repr):")
    conn = sqlite3.connect(odds_db)
    cur = conn.execute(
        "SELECT date, home_team, away_team FROM historical_covers_odds WHERE sport='ncaab' LIMIT 5"
    )
    for row in cur.fetchall():
        print(f"      date={repr(row[0])} home_team={repr(row[1])} away_team={repr(row[2])}")
    conn.close()

    # (2) First 5 rows of games_with_team_stats NCAAB 2025-26
    print("\n(2) games_with_team_stats (NCAAB 2025-26) — first 5 rows:")
    print("    Columns: game_date, home_team_name, away_team_name. Exact format (repr):")
    conn = sqlite3.connect(espn_db)
    cur = conn.execute(
        "SELECT game_date, home_team_name, away_team_name FROM games_with_team_stats WHERE league='ncaab' AND game_date >= '2025-11-01' ORDER BY game_date LIMIT 5"
    )
    for row in cur.fetchall():
        print(f"      game_date={repr(row[0])} home={repr(row[1])} away={repr(row[2])}")
    conn.close()

    # (3) Manual match for one game that appears in both (e.g. 2025-11-03 St. Bonaventure vs Bradley)
    print("\n(3) Manual match: 2025-11-03 St. Bonaventure vs Bradley")
    from engine.historical_odds import _normalize_team, _team_match_join
    covers_date = "2025-11-03"
    covers_home, covers_away = "St. Bonaventure", "Bradley"
    game_date = "2025-11-03T16:00Z"
    game_home, game_away = "St. Bonaventure Bonnies", "Bradley Braves"
    d_covers = pd.to_datetime(covers_date, errors="coerce").normalize()
    d_game = pd.to_datetime(game_date, errors="coerce").normalize()
    print(f"    Covers date (normalized): {repr(d_covers)} tz={getattr(d_covers, 'tz', None)}")
    print(f"    Game date (normalized):   {repr(d_game)} tz={getattr(d_game, 'tz', None)}")
    print(f"    Date equal? {d_covers == d_game}")
    h_norm, a_norm = _normalize_team(game_home), _normalize_team(game_away)
    print(f"    Team match home: covers {repr(covers_home)} vs game {repr(h_norm)} -> {_team_match_join(covers_home, h_norm)}")
    print(f"    Team match away: covers {repr(covers_away)} vs game {repr(a_norm)} -> {_team_match_join(covers_away, a_norm)}")
    print("    => Join fails because date comparison is False (naive vs timezone-aware).")

    # Fix: run merge then print new count
    print("\n--- Running merge_historical_closing_into_games (with fix) ---")
    from engine.historical_odds import merge_historical_closing_into_games
    merge_historical_closing_into_games(espn_db_path=espn_db, odds_db_path=odds_db)
    conn = sqlite3.connect(espn_db)
    cur = conn.execute(
        "SELECT COUNT(*) FROM games_with_team_stats WHERE league='ncaab' AND closing_home_spread IS NOT NULL"
    )
    n = cur.fetchone()[0]
    conn.close()
    print(f"NCAAB games with non-null closing_home_spread (after fix): {n}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
