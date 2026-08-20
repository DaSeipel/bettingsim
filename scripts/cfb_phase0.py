#!/usr/bin/env python3
"""
CFB Phase 0 — data layer only (no model, no predictions, no UI changes).

Steps:
  1. Probe CFBD auth via GET /teams?year=2025 and print raw response shape
  2. Create cfb_* tables in data/espn.db
  3. Backfill seasons 2015-2025 (regular + postseason)
  4. Build cfb_team_alias from CFBD 2026 FBS + live Odds API names
  5. Print validation report

Usage:
  python3 scripts/cfb_phase0.py
  python3 scripts/cfb_phase0.py --probe-only
  python3 scripts/cfb_phase0.py --start-season 2024 --end-season 2025
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.cfb_client import CFBDClient
from engine.cfb_config import BACKFILL_END_SEASON, BACKFILL_START_SEASON, ESPN_DB_PATH
from engine.cfb_ingest import backfill_all, probe_teams
from engine.cfb_schema import ensure_cfb_schema
from engine.cfb_validate import print_validation_report, validate_cfb_data


def main() -> int:
    parser = argparse.ArgumentParser(description="CFB Phase 0 data ingestion")
    parser.add_argument("--probe-only", action="store_true", help="Only run /teams?year=2025 auth probe")
    parser.add_argument("--start-season", type=int, default=BACKFILL_START_SEASON)
    parser.add_argument("--end-season", type=int, default=BACKFILL_END_SEASON)
    parser.add_argument("--db-path", type=Path, default=ESPN_DB_PATH)
    args = parser.parse_args()

    ESPN_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    client = CFBDClient()

    print("Step 1: CFBD auth probe (/teams?year=2025)", flush=True)
    probe_teams(client, year=2025)
    if args.probe_only:
        print(f"Probe complete. CFBD calls={client.call_count}")
        return 0

    conn = sqlite3.connect(args.db_path)
    try:
        print("\nStep 2: Ensure cfb_* schema", flush=True)
        ensure_cfb_schema(conn)

        print("\nStep 3-4: Backfill + aliases", flush=True)
        summary = backfill_all(
            conn,
            client,
            start_season=args.start_season,
            end_season=args.end_season,
        )

        print("\nAlias summary:")
        aliases = summary.get("aliases", {})
        print(f"  exact matches: {len(aliases.get('exact_matches', []))}")
        print(f"  unmatched CFBD: {len(aliases.get('unmatched_cfbd', []))}")
        print(f"  unmatched Odds API: {len(aliases.get('unmatched_odds', []))}")
        if aliases.get("unmatched_cfbd"):
            print("  UNMATCHED CFBD names:")
            for name in aliases["unmatched_cfbd"]:
                print(f"    - {name}")
        if aliases.get("unmatched_odds"):
            print("  UNMATCHED Odds API names:")
            for name in aliases["unmatched_odds"]:
                print(f"    - {name}")

        print("\nStep 5: Validation", flush=True)
        report = validate_cfb_data(conn)
        print_validation_report(report, cfbd_calls=client.call_count)
    finally:
        conn.close()

    print("\nCFB Phase 0 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
