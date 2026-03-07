#!/usr/bin/env python3
"""
Run only Step 1 (ESPN 2025-26 NCAAB fetch) and Step 2 (situational features) from the pipeline.
No Sports-Reference, momentum, or advanced analytics. Then print games_with_team_stats row count and latest game_date.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NCAAB_2026_START = "2025-11-01"


def main() -> int:
    db_path = ROOT / "data" / "espn.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print("(1) Pulling 2025-26 NCAAB games from ESPN (season 2026, Nov 2025 through today)...", flush=True)
    from engine.espn_collector import (
        collect_games_and_schedules,
        load_games_from_sqlite,
        save_games_to_sqlite,
    )

    games_2026, _ = collect_games_and_schedules("ncaab", seasons=[2026])
    if games_2026.empty:
        print("    No 2026 NCAAB games returned from ESPN.", flush=True)
    else:
        game_dates = games_2026["game_date"].astype(str)
        mask = game_dates >= NCAAB_2026_START
        games_2026 = games_2026.loc[mask].copy()
        print(f"    2025-26 games (>= {NCAAB_2026_START}): {len(games_2026)}", flush=True)

    existing = load_games_from_sqlite(league_key="ncaab", db_path=db_path)
    if not games_2026.empty:
        combined = (
            pd.concat([existing, games_2026], ignore_index=True)
            if not existing.empty
            else games_2026
        )
        combined = combined.drop_duplicates(subset=["league", "game_id"], keep="last")
        save_games_to_sqlite(combined, "ncaab", db_path)
        n_in_db = len(combined)
    else:
        n_in_db = len(existing)
    print(f"Step 1 complete - NCAAB games in DB: {n_in_db}", flush=True)

    print("\n(2) Rebuilding situational features for NCAAB...", flush=True)
    ncaab_games = load_games_from_sqlite(league_key="ncaab", db_path=db_path)
    if not ncaab_games.empty:
        from engine.situational_features import build_and_save_situational_features
        build_and_save_situational_features(
            ncaab_games, league="ncaab", db_path=db_path, use_geopy=True
        )
    print("Step 2 complete - situational features built for NCAAB", flush=True)

    print("\nGames_with_team_stats (unchanged by this run):", flush=True)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute("SELECT COUNT(*) FROM games_with_team_stats")
        rows = cur.fetchone()[0]
        cur = conn.execute("SELECT MAX(game_date) FROM games_with_team_stats")
        latest = cur.fetchone()[0]
    finally:
        conn.close()
    print(f"  Rows: {rows}", flush=True)
    print(f"  Latest game_date: {latest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
