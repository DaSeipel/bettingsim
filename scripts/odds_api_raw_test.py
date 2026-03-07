#!/usr/bin/env python3
"""Raw Odds API call: no cache, no engine. Print status, remaining credits, event count."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
key = ""
if (ROOT / ".streamlit" / "secrets.toml").exists():
    m = re.search(r'api_key\s*=\s*["\']([^"\']+)["\']', (ROOT / ".streamlit" / "secrets.toml").read_text())
    if m:
        key = m.group(1).strip()
if not key:
    key = __import__("os").environ.get("ODDS_API_KEY", "").strip()

import requests
resp = requests.get("https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds", params={"regions": "us", "markets": "spreads", "oddsFormat": "american", "apiKey": key}, timeout=15)
data = resp.json() if resp.ok and resp.text else []
events = data if isinstance(data, list) else []
remaining = resp.headers.get("x-requests-remaining", "N/A")
print("status_code:", resp.status_code)
print("x-requests-remaining:", remaining)
print("events_count:", len(events))
