#!/usr/bin/env python3
"""
Pull 2025-26 NCAAB games from ESPN (Nov 2025 through today), add to games table,
run KenPom merge, merge_historical_closing_into_games, then count NCAAB with closing_home_spread.
If >= 50, retrain NCAAB model without --allow-proxy.

With --skip-espn: skip ESPN fetch and situational (steps 1-2). Use when 2025-26 games
are already in the games table. Runs fetch_merge_and_save with skip_advanced_analytics
and skip_fetch_team_stats (no Sports-Reference), then steps 4, 5, 6.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# November 2025 start to match 2026ncaablines.csv
NCAAB_2026_START = "2025-11-01"


def main() -> int:
    argv = [a for a in sys.argv[1:] if a.startswith("--")]
    skip_espn = "--skip-espn" in argv

    db_path = ROOT / "data" / "espn.db"
    odds_db_path = ROOT / "data" / "odds.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not skip_espn:
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
        print(f"Step 1 complete - {len(games_2026) if not games_2026.empty else 0} games pulled from ESPN (NCAAB in DB: {n_in_db})", flush=True)

        print("\n(2) Rebuilding situational features for NCAAB...", flush=True)
        ncaab_games = load_games_from_sqlite(league_key="ncaab", db_path=db_path)
        if not ncaab_games.empty:
            from engine.situational_features import build_and_save_situational_features
            build_and_save_situational_features(
                ncaab_games, league="ncaab", db_path=db_path, use_geopy=True
            )
        print("Step 2 complete - situational features built for NCAAB", flush=True)
    else:
        print("Step 1 skipped (--skip-espn)", flush=True)
        print("Step 2 skipped (--skip-espn)", flush=True)

    print("\n(3) fetch_merge_and_save (no Sports-Reference, no advanced analytics)...", flush=True)
    from engine.sportsref_stats import fetch_merge_and_save
    merged = fetch_merge_and_save(
        seasons=[2026, 2025, 2024],
        db_path=db_path,
        skip_advanced_analytics=True,
        skip_fetch_team_stats=True,
    )
    if merged.empty:
        print("    fetch_merge_and_save returned empty.", flush=True)
        return 1
    print(f"Step 3 complete - games_with_team_stats has {len(merged)} rows", flush=True)

    print("\n(4) KenPom merge (merge_ncaab_kenpom_into_games)...", flush=True)
    rc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "merge_ncaab_kenpom_into_games.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    print(rc.stdout or "", flush=True)
    if rc.stderr:
        print(rc.stderr, file=sys.stderr, flush=True)
    if rc.returncode != 0:
        print("    KenPom merge script failed (non-zero exit). Continuing.", flush=True)
    print("Step 4 complete - KenPom merge done", flush=True)

    print("\n(5) merge_historical_closing_into_games()...", flush=True)
    from engine.historical_odds import merge_historical_closing_into_games
    merge_historical_closing_into_games(espn_db_path=db_path, odds_db_path=odds_db_path)

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT COUNT(*) FROM games_with_team_stats WHERE league = 'ncaab' AND closing_home_spread IS NOT NULL"
        )
        n_with_closing = cur.fetchone()[0]
    finally:
        conn.close()

    print(f"Step 5 complete - historical closing lines merged. NCAAB with closing_home_spread: {n_with_closing}", flush=True)

    if n_with_closing >= 50:
        print("\n(6) Retraining NCAAB model (no --allow-proxy)...", flush=True)
        rc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "train_ncaab_kenpom.py")],
            cwd=str(ROOT),
            capture_output=False,
            text=True,
        )
        if rc.returncode != 0:
            print("    Train script exited with code:", rc.returncode, flush=True)
            return rc.returncode
        print("Step 6 complete - NCAAB model retrained", flush=True)
    else:
        print(f"\n(6) Skipping train (need >= 50 games with closing lines; have {n_with_closing}).", flush=True)
        print("Step 6 skipped - need >= 50 games with closing lines to retrain", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
