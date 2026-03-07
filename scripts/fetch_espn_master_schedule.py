#!/usr/bin/env python3
"""
Fetch ESPN NCAAB scoreboard for today and save a clean list of matchups.
Uses: site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard
"""
from datetime import date
import json
import sys
from pathlib import Path

import requests

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
REQUEST_TIMEOUT = 30


def _parse_home_away(competitors: list) -> tuple[str, str]:
    """From competition.competitors (homeAway + team.displayName), return (home_team, away_team)."""
    home_team = ""
    away_team = ""
    for c in competitors or []:
        if not isinstance(c, dict):
            continue
        ha = (c.get("homeAway") or "").strip().lower()
        team = c.get("team") if isinstance(c.get("team"), dict) else {}
        name = (team.get("displayName") or team.get("name") or "").strip()
        if not name:
            continue
        if ha == "home":
            home_team = name
        elif ha == "away":
            away_team = name
    return (home_team, away_team)


def main() -> int:
    today = date.today()
    dates_param = today.strftime("%Y%m%d")
    url = f"{BASE_URL}?limit=500&groups=50&dates={dates_param}"

    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        print(f"Request failed: {e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}", file=sys.stderr)
        return 1

    events = data.get("events") or []
    matchups = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        game_id = ev.get("id") or ev.get("$ref", "").split("/")[-1].split("?")[0]
        if not game_id:
            continue
        comps = ev.get("competitions") or []
        if not comps:
            continue
        competitors = comps[0].get("competitors") or []
        home_team, away_team = _parse_home_away(competitors)
        if not home_team or not away_team:
            continue
        matchups.append({
            "game_id": str(game_id),
            "home_team": home_team,
            "away_team": away_team,
        })

    out_path = Path(__file__).resolve().parent.parent / "data" / "cache" / "espn_master_schedule.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {"date": today.isoformat(), "matchups": matchups},
            f,
            indent=2,
        )

    n = len(matchups)
    print(f"ESPN NCAAB master schedule ({today.isoformat()})")
    print(f"Total Division I games: {n}")
    print(f"Saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
