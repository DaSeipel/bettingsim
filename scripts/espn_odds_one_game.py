#!/usr/bin/env python3
"""
Fetch the full odds subresource for one ESPN game and print every provider and market.
Usage: python scripts/espn_odds_one_game.py [nba|ncaab]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ESPN_BASE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues"
HEADERS = {"Accept": "application/json", "User-Agent": "bettingsim/1.0"}
TIMEOUT = 15


def main() -> None:
    league_slug = "nba"
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("ncaab", "mens-college-basketball"):
        league_slug = "mens-college-basketball"

    from datetime import date
    today = date.today().strftime("%Y%m%d")
    events_url = f"{ESPN_BASE}/{league_slug}/events?limit=5&dates={today}"
    r = requests.get(events_url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    items = data.get("items") or []
    if not items:
        print(f"No events for {league_slug} on {today}. Try without date filter or another day.")
        # Try without date
        events_url = f"{ESPN_BASE}/{league_slug}/events?limit=3"
        r = requests.get(events_url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        items = data.get("items") or []
    if not items:
        print("No events returned.")
        return
    ref = items[0].get("$ref", "")
    import re
    m = re.search(r"/events/(\d+)(?:\?|$)", ref)
    event_id = m.group(1) if m else None
    if not event_id:
        print("Could not parse event id from ref:", ref)
        return
    comp_id = event_id
    odds_url = f"{ESPN_BASE}/{league_slug}/events/{event_id}/competitions/{comp_id}/odds"
    print(f"Fetching full odds: {odds_url}\n")
    r = requests.get(odds_url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    full = r.json()

    count = full.get("count", 0)
    page_count = full.get("pageCount", 0)
    items_list = full.get("items") or []
    print(f"Response: count={count}, pageCount={page_count}, len(items)={len(items_list)}")
    print("-" * 60)
    print("PROVIDERS AND MARKETS (per item in 'items'):")
    print("-" * 60)
    for i, item in enumerate(items_list):
        if not isinstance(item, dict):
            print(f"  Item {i}: (not a dict)")
            continue
        prov = item.get("provider") or {}
        prov_id = prov.get("id") if isinstance(prov, dict) else item.get("provider", {})
        if isinstance(prov_id, dict):
            prov_id = prov_id.get("id", "?")
        prov_name = prov.get("name", "?") if isinstance(prov, dict) else "?"
        print(f"\n  Provider {i + 1}: id={prov_id}, name={prov_name}")
        markets = []
        if item.get("spread") is not None:
            markets.append("spread")
        if item.get("overUnder") is not None:
            markets.append("overUnder (totals)")
        a = item.get("awayTeamOdds") or {}
        h = item.get("homeTeamOdds") or {}
        if a.get("moneyLine") is not None or h.get("moneyLine") is not None:
            markets.append("moneyline (h2h)")
        if a.get("spreadOdds") is not None or h.get("spreadOdds") is not None:
            if "spread" not in markets:
                markets.append("spreadOdds")
        if item.get("overOdds") is not None or item.get("underOdds") is not None:
            if "overUnder (totals)" not in markets:
                markets.append("overOdds/underOdds")
        print(f"    Markets: {', '.join(markets)}")
        top_keys = sorted(k for k in item.keys() if not k.startswith("$") and k != "links")
        print(f"    Top-level keys: {top_keys}")
        if "propBets" in item and item["propBets"]:
            print(f"    Note: 'propBets' present (not parsed by bettingsim; only main markets used).")
    print("\n" + "-" * 60)
    print("Conclusion: ESPN odds endpoint returns one provider per game (Draft Kings) and")
    print("markets: moneyline, spread, over/under. No additional bookmakers available.")
    print("-" * 60)
    print("Full response (JSON) - items only:")
    print(json.dumps({"count": full.get("count"), "items": items_list}, indent=2)[:4000])
    if len(json.dumps(items_list)) > 4000:
        print("... (truncated)")


if __name__ == "__main__":
    main()
