#!/usr/bin/env python3
"""Fetch today's full Rundown API response for NCAAB and save to data/cache/rapid_odds.json."""
from datetime import date
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import requests
from engine.rundown_odds import RUNDOWN_BASE, REQUEST_TIMEOUT, _load_rundown_secrets

NCAAB_SPORT_ID = 5


def main() -> int:
    api_key, host = _load_rundown_secrets()
    if not api_key.strip():
        print("RUNDOWN_API_KEY not set. Configure .streamlit/secrets.toml [therundown] or env.", file=sys.stderr)
        return 1
    today = date.today()
    date_str = today.strftime("%Y-%m-%d")
    url = f"{RUNDOWN_BASE}/sports/{NCAAB_SPORT_ID}/events/{date_str}"
    params = {"include": "scores", "affiliate_ids": "1,2,3", "offset": 0}
    headers = {"x-rapidapi-host": host.strip(), "x-rapidapi-key": api_key.strip()}
    r = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    out_path = Path(__file__).resolve().parent.parent / "data" / "cache" / "rapid_odds.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    n = len(data.get("events") or data.get("event") or [])
    print(f"Saved {n} NCAAB events to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
