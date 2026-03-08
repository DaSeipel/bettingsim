#!/usr/bin/env python3
"""
Archive a cleaned matchups snapshot into data/team_stats_history.csv.
Adds a date column (e.g. 2026-03-07), appends to the master history file, and dedupes by (Team, date).

Usage:
  python scripts/ingest_cleaned_matchups.py [--csv PATH] [--date YYYY-MM-DD]

Default: cleaned_matchups_march_7.csv in project root, date 2026-03-07.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CSV = ROOT / "cleaned_matchups_march_7.csv"
DEFAULT_DATE = "2026-03-07"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest cleaned matchups CSV into team_stats_history")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to cleaned matchups CSV")
    parser.add_argument("--date", type=str, default=DEFAULT_DATE, help="Snapshot date YYYY-MM-DD")
    args = parser.parse_args()
    if not args.csv.exists():
        print(f"Error: CSV not found: {args.csv}")
        return
    from engine.team_stats_history import append_snapshot_to_history, load_team_stats_history, TEAM_STATS_HISTORY_PATH
    import pandas as pd
    df = pd.read_csv(args.csv)
    if df.empty or "Team" not in df.columns:
        print("Error: CSV must have a 'Team' column and at least one row.")
        return
    n_before = 0
    if TEAM_STATS_HISTORY_PATH.exists():
        existing = load_team_stats_history()
        n_before = len(existing)
    append_snapshot_to_history(df, args.date)
    after = load_team_stats_history()
    n_after = len(after)
    print(f"Snapshot date: {args.date}")
    print(f"Rows in CSV: {len(df)}")
    print(f"History before: {n_before} rows → after: {n_after} rows")
    print(f"Saved to {TEAM_STATS_HISTORY_PATH}")


if __name__ == "__main__":
    main()
