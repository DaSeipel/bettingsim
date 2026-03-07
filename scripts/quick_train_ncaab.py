#!/usr/bin/env python3
"""
Quick NCAAB merge + train: no ESPN, no Sports-Reference, no momentum/situational.
(1) KenPom merge on existing games_with_team_stats
(2) merge_historical_closing_into_games() to attach 2026 closing lines
(3) Print how many NCAAB games have non-null closing_home_spread
(4) If >= 50, retrain NCAAB model without --allow-proxy and print results
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    db_path = ROOT / "data" / "espn.db"
    odds_db_path = ROOT / "data" / "odds.db"

    print("(1) KenPom merge...", flush=True)
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
        print("KenPom merge failed (non-zero exit). Continuing.", flush=True)
    print("Step 1 complete - KenPom merge done", flush=True)

    print("\n(2) merge_historical_closing_into_games()...", flush=True)
    from engine.historical_odds import merge_historical_closing_into_games
    merge_historical_closing_into_games(espn_db_path=db_path, odds_db_path=odds_db_path)
    print("Step 2 complete - closing lines merged", flush=True)

    print("\n(3) Counting NCAAB games with closing_home_spread...", flush=True)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT COUNT(*) FROM games_with_team_stats WHERE league = 'ncaab' AND closing_home_spread IS NOT NULL"
        )
        n_with_closing = cur.fetchone()[0]
    finally:
        conn.close()
    print(f"NCAAB games with non-null closing_home_spread: {n_with_closing}", flush=True)

    if n_with_closing >= 50:
        print("\n(4) Retraining NCAAB model (no --allow-proxy)...", flush=True)
        rc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "train_ncaab_kenpom.py")],
            cwd=str(ROOT),
            capture_output=False,
            text=True,
        )
        if rc.returncode != 0:
            print("Train script exited with code:", rc.returncode, flush=True)
            return rc.returncode
        print("Step 4 complete - NCAAB model retrained", flush=True)
    else:
        print(f"\n(4) Skipping train (need >= 50; have {n_with_closing}).", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
