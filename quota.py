#!/usr/bin/env python3
"""
CLI for The Odds API quota status.
Usage: python quota.py status
"""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.odds_quota import get_quota_status, MIN_REMAINING_BEFORE_SKIP


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1].lower() != "status":
        print("Usage: python quota.py status")
        print("  Prints current Odds API usage (x-requests-remaining, x-requests-used).")
        return 1
    status = get_quota_status()
    remaining = status.get("requests_remaining")
    used = status.get("requests_used")
    last = status.get("requests_last")
    updated = status.get("last_updated", "")
    if remaining is None and used is None:
        print("No quota data yet. Make an Odds API request to populate from response headers.")
        return 0
    print("The Odds API — quota status")
    print("-" * 40)
    if remaining is not None:
        print(f"  Requests remaining:  {remaining}")
        if remaining < MIN_REMAINING_BEFORE_SKIP:
            print(f"  (Calls will be skipped when remaining < {MIN_REMAINING_BEFORE_SKIP})")
    if used is not None:
        print(f"  Requests used:      {used}")
    if last is not None:
        print(f"  Last call cost:     {last}")
    if updated:
        print(f"  Last updated:       {updated}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
