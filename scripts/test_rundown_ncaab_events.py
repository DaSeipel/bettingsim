#!/usr/bin/env python3
"""
Standalone test of The Rundown API for today's date (NCAAB only).
Prints: (1) number of NCAAB events, (2) whether Vanderbilt@Tennessee or Alabama@Auburn are present,
(3) full list of NCAAB event names.
"""
from datetime import date
import sys
from pathlib import Path

# Allow importing engine from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from engine.rundown_odds import (
    _load_rundown_secrets,
    _fetch_rundown_events,
    _teams_from_event,
)

NCAAB_SPORT_ID = 5


def _event_name(ev: dict) -> str:
    away, home = _teams_from_event(ev)
    if not away and not home:
        return ev.get("schedule", {}).get("event_name") or ev.get("event_name") or ev.get("name") or "(no name)"
    return f"{away} @ {home}" if (away and home) else (away or home or "(no name)")


def main() -> None:
    today = date.today()
    api_key, host = _load_rundown_secrets()
    if not api_key.strip():
        print("RUNDOWN_API_KEY not set (env or .streamlit/secrets.toml [therundown]). Skipping request.")
        sys.exit(1)
    session = requests.Session()
    events = _fetch_rundown_events(NCAAB_SPORT_ID, today, api_key, host, session)
    if events is None:
        print("The Rundown API returned None (error or no data).")
        sys.exit(1)

    names = [_event_name(ev) for ev in events]

    print("=" * 60)
    print("THE RUNDOWN API — NCAAB EVENTS (today's date)")
    print(f"Date requested: {today}")
    print("=" * 60)
    print(f"(1) Number of NCAAB events returned: {len(events)}")
    print()

    vandy_tenn = any(
        "vanderbilt" in n.lower() and "tennessee" in n.lower()
        for n in names
    )
    bama_auburn = any(
        "alabama" in n.lower() and "auburn" in n.lower()
        for n in names
    )
    print("(2) Response includes:")
    print(f"    Vanderbilt at Tennessee: {'Yes' if vandy_tenn else 'No'}")
    print(f"    Alabama at Auburn:       {'Yes' if bama_auburn else 'No'}")
    print()

    print("(3) Full list of NCAAB event names:")
    for i, name in enumerate(names, 1):
        print(f"    {i:3}. {name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
