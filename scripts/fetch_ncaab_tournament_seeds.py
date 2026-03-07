#!/usr/bin/env python3
"""
Fetch NCAAB tournament bracket from ESPN and save team + seed to data/ncaab_seeds.csv.
Runs after Selection Sunday; uses ESPN's tournament API(s). Output: CSV with columns team, seed.
"""

from __future__ import annotations

import csv
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SEEDS_CSV = DATA_DIR / "ncaab_seeds.csv"

# Try both URL patterns (user specified apis/v2; site/v2 also exists and may return bracket)
TOURNAMENT_URLS = [
    "https://site.api.espn.com/apis/v2/sports/basketball/mens-college-basketball/tournaments",
    "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/tournaments",
]


def _get_tournament_data() -> dict | list:
    """Fetch tournament JSON from first successful URL. Optional year param for current March."""
    from datetime import date
    year = date.today().year
    headers = {"User-Agent": "Mozilla/5.0 (compatible; bettingsim/1.0)"}
    for base in TOURNAMENT_URLS:
        for url in [f"{base}?year={year}", base]:
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as r:
                    data = json.load(r)
                if data:
                    return data
            except Exception:
                continue
    return {}


def _extract_teams_and_seeds(data: dict | list) -> list[tuple[str, int]]:
    """
    Parse ESPN tournament response into (team_display_name, seed) list.
    Handles: tournaments[].events[].competitions[].competitors[].team.displayName + .seed,
    or tournaments[].bracket.*.competitors, or events[].competitions[].competitors with seed.
    """
    out: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()

    def add(name: str, seed_val: int) -> None:
        if not name or not str(name).strip():
            return
        try:
            s = int(seed_val)
            if 1 <= s <= 16 and (name.strip(), s) not in seen:
                seen.add((name.strip(), s))
                out.append((name.strip(), s))
        except (TypeError, ValueError):
            pass

    def walk_competitors(obj: dict, team_key: str = "team", seed_key: str = "seed") -> None:
        comps = obj.get("competitors") if isinstance(obj, dict) else None
        if not isinstance(comps, list):
            return
        for c in comps:
            if not isinstance(c, dict):
                continue
            team = c.get(team_key)
            seed = c.get(seed_key)
            if isinstance(team, dict):
                name = team.get("displayName") or team.get("name") or team.get("shortDisplayName")
                if name:
                    add(str(name), seed if seed is not None else 99)
            elif seed is not None:
                add(str(c.get("name", c.get("displayName", "")) or ""), seed)

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                walk_competitors(item)
                for key in ("events", "bracket", "groups"):
                    sub = item.get(key)
                    if isinstance(sub, list):
                        for s in sub:
                            if isinstance(s, dict):
                                walk_competitors(s)
                                for ev in s.get("events", []):
                                    if isinstance(ev, dict):
                                        for comp in ev.get("competitions", []):
                                            walk_competitors(comp)
    elif isinstance(data, dict):
        tournaments = data.get("tournaments") or data.get("events") or []
        if not tournaments and "events" in data:
            tournaments = data["events"]
        if not isinstance(tournaments, list):
            tournaments = [data]
        for t in tournaments:
            if not isinstance(t, dict):
                continue
            walk_competitors(t)
            for ev in t.get("events", []):
                if not isinstance(ev, dict):
                    continue
                for comp in ev.get("competitions", []):
                    walk_competitors(comp)
            # Nested bracket (e.g. regions)
            for region in t.get("groups", t.get("bracket", [])) or []:
                if not isinstance(region, dict):
                    continue
                for ev in region.get("events", []):
                    if isinstance(ev, dict):
                        for comp in ev.get("competitions", []):
                            walk_competitors(comp)

    return sorted(out, key=lambda x: (x[1], x[0]))


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = _get_tournament_data()
    pairs = _extract_teams_and_seeds(data)
    # Dedupe by team name (keep lowest seed if duplicate)
    by_team: dict[str, int] = {}
    for name, seed in pairs:
        if name not in by_team or seed < by_team[name]:
            by_team[name] = seed
    rows = [(team, by_team[team]) for team in sorted(by_team.keys())]
    with open(SEEDS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["team", "seed"])
        w.writerows(rows)
    print(f"NCAAB tournament seeds: {len(rows)} teams saved to {SEEDS_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
