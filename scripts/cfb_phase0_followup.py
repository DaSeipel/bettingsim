#!/usr/bin/env python3
"""
CFB Phase 0 follow-up: coverage recompute, alias map, season flags, lines audit.
No CFBD API calls.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.cfb_alias import match_aliases, persist_aliases
from engine.cfb_config import ESPN_DB_PATH
from engine.cfb_phase0_followup import (
    game_stats_row_shape,
    lines_provider_report,
    print_lines_report,
    print_stats_coverage,
    stats_coverage_by_population,
    upsert_season_flags,
)
from engine.cfb_schema import ensure_cfb_schema

ODDS_NAMES_PATH = ROOT / "data" / "cfb" / "odds_api_names_phase0.json"


def main() -> int:
    conn = sqlite3.connect(ESPN_DB_PATH)
    try:
        ensure_cfb_schema(conn)

        shape = game_stats_row_shape(conn)
        coverage = stats_coverage_by_population(conn)
        print_stats_coverage(coverage, shape)

        print("\n=== 2. Alias map (exact / prefix / normalized_prefix) ===")
        cfbd_names = [
            row[0]
            for row in conn.execute(
                "SELECT cfbd_name FROM cfb_team_alias WHERE cfbd_name IS NOT NULL ORDER BY cfbd_name"
            ).fetchall()
        ]
        odds_names = json.loads(ODDS_NAMES_PATH.read_text(encoding="utf-8"))
        result = match_aliases(cfbd_names, odds_names)
        persist_aliases(conn, match_result=result)
        print("Matched per rule:")
        for rule, n in result["counts"].items():
            print(f"  {rule}: {n}")
        print(f"Ambiguous CFBD names ({len(result['ambiguous'])}):")
        for item in result["ambiguous"]:
            print(f"  {item['cfbd_name']} [{item['rule']}]:")
            for cand in item["candidates"]:
                print(f"    - {cand}")
        print(f"Remaining unmatched CFBD FBS ({len(result['unmatched_cfbd'])}):")
        for name in result["unmatched_cfbd"]:
            print(f"  - {name}")
        print(f"Odds API names with no CFBD counterpart ({len(result['unmatched_odds'])}) — expect FCS:")
        for name in result["unmatched_odds"]:
            print(f"  - {name}")

        print("\n=== 3. Season quality flags ===")
        upsert_season_flags(conn)
        for row in conn.execute(
            "SELECT season, is_anomalous, exclude_from_hfa, exclude_from_training, note "
            "FROM cfb_season_flags ORDER BY season"
        ):
            print(
                f"  {row[0]}: anomalous={bool(row[1])} exclude_hfa={bool(row[2])} "
                f"exclude_train={bool(row[3])} | {row[4]}"
            )

        lines = lines_provider_report(conn)
        print_lines_report(lines)
    finally:
        conn.close()
    print("\nCFB Phase 0 follow-up complete. CFBD calls this run: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
