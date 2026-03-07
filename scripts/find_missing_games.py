#!/usr/bin/env python3
"""
Diagnostic: compare ESPN master schedule (source of truth) to Rundown (rapid_odds.json).
Prints every game on ESPN that is missing from The Rundown, with optional fuzzy match suggestions.

Usage:
  python scripts/find_missing_games.py [rapid_odds.json [espn_master_schedule.json]]

Defaults: data/cache/rapid_odds.json, data/cache/espn_master_schedule.json.
rapid_odds.json = your latest Rundown API response (JSON with "events" array and teams),
  or a file with "matchups": [{"away_team", "home_team"}].
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from difflib import SequenceMatcher

# Project root
ROOT = Path(__file__).resolve().parent.parent
ESPN_PATH = ROOT / "data" / "cache" / "espn_master_schedule.json"
RAPID_ODDS_PATH = ROOT / "data" / "cache" / "rapid_odds.json"

# Common nicknames / abbreviations for fuzzy suggestion (ESPN or Rundown may use either)
TEAM_ALIASES: dict[str, str] = {
    "unc": "north carolina",
    "north carolina": "north carolina",
    "duke": "duke",
    "uk": "kentucky",
    "kentucky": "kentucky",
    "ku": "kansas",
    "kansas": "kansas",
    "lsu": "louisiana state",
    "uconn": "connecticut",
    "connecticut": "connecticut",
    "usc": "southern california",
    "ucla": "ucla",
    "byu": "brigham young",
    "texas tech": "texas tech",
    "vt": "virginia tech",
    "virginia tech": "virginia tech",
    "fsu": "florida state",
    "florida state": "florida state",
    "miami": "miami",
    "nc state": "north carolina state",
    "n.c. state": "north carolina state",
    "north carolina state": "north carolina state",
    "wake forest": "wake forest",
    "syracuse": "syracuse",
    "cuse": "syracuse",
    "pitt": "pittsburgh",
    "pittsburgh": "pittsburgh",
    "lsu": "louisiana state",
    "ole miss": "mississippi",
    "mississippi": "mississippi",
    "mississippi state": "mississippi state",
    "arkansas": "arkansas",
    "alabama": "alabama",
    "auburn": "auburn",
    "tennessee": "tennessee",
    "vanderbilt": "vanderbilt",
    "georgia": "georgia",
    "florida": "florida",
    "south carolina": "south carolina",
    "texas a&m": "texas a&m",
    "texas am": "texas a&m",
}


def _normalize_team(s: str) -> str:
    """Lowercase, strip, collapse spaces."""
    if not s or not isinstance(s, str):
        return ""
    return " ".join(s.lower().strip().split())


def _team_core_name(s: str) -> str:
    """Core name for matching: drop common suffixes like 'Cyclones', 'Wildcats', 'Volunteers'."""
    n = _normalize_team(s)
    # Remove trailing " Mascot" (e.g. "Tennessee Volunteers" -> "tennessee")
    for suffix in (
        " cyclones", " wildcats", " volunteers", " commodores", " bulldogs", " tigers", " crimson tide",
        " tar heels", " blue devils", " jayhawks", " cardinals", " hurricanes", " seminoles",
        " gators", " gamecocks", " razorbacks", " rebels", " bulldogs", " aggies", " horned frogs",
        " sooners", " longhorns", " red raiders", " cougars", " bearcats", " mountaineers",
        " buckeyes", " wolverines", " spartans", " hoosiers", " boilermakers", " fighting illini",
        " hawkeyes", " sun devils", " bruins", " trojans", " huskies", " shockers", " panthers",
        " orange", " demon deacons", " eagles", " ramblers", " revolutionaries", " sharks",
        " seahawks", " catamounts", " big green", " big red", " crimson",
    ):
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
            break
    return n


def _resolve_alias(name: str) -> str:
    """Return canonical form for fuzzy match (optional)."""
    c = _team_core_name(name)
    return TEAM_ALIASES.get(c, c)


def _teams_from_rundown_event(ev: dict) -> tuple[str, str]:
    """(away_team, home_team) from Rundown API event."""
    teams = ev.get("teams")
    if isinstance(teams, list) and len(teams) >= 2:
        away_team = ""
        home_team = ""
        for t in teams:
            if not isinstance(t, dict):
                continue
            n = (t.get("name") or "").strip()
            if not n:
                continue
            is_home = t.get("is_home") in (True, "true", "1", 1) or (str(t.get("homeAway") or "").lower() == "home")
            is_away = t.get("is_away") in (True, "true", "1", 1) or (str(t.get("homeAway") or "").lower() == "away")
            if is_home:
                home_team = n
            if is_away:
                away_team = n
        if away_team and home_team:
            return (away_team, home_team)
        # Fallback: assume first = away, second = home (Rundown order)
        t0 = teams[0].get("name") if isinstance(teams[0], dict) else ""
        t1 = teams[1].get("name") if isinstance(teams[1], dict) else ""
        if t0 and t1:
            return (str(t0).strip(), str(t1).strip())
    if isinstance(teams, list) and len(teams) == 1:
        n = teams[0].get("name") if isinstance(teams[0], dict) else ""
        if n:
            return (str(n).strip(), "")
    name = ev.get("event_name") or ev.get("name") or ""
    if " @ " in str(name):
        parts = str(name).split(" @ ", 1)
        return (parts[0].strip(), parts[1].strip() if len(parts) > 1 else "")
    if " at " in str(name):
        parts = str(name).split(" at ", 1)
        return (parts[0].strip(), parts[1].strip() if len(parts) > 1 else "")
    return ("", "")


def _load_espn_matchups(path: Path) -> list[tuple[str, str, str]]:
    """(game_id, away_team, home_team) from espn_master_schedule.json."""
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    matchups = data.get("matchups") or []
    out = []
    for m in matchups:
        if not isinstance(m, dict):
            continue
        gid = m.get("game_id") or ""
        away = (m.get("away_team") or "").strip()
        home = (m.get("home_team") or "").strip()
        if away and home:
            out.append((gid, away, home))
    return out


def _load_rundown_games(path: Path) -> list[tuple[str, str]]:
    """List of (away_team, home_team) from rapid_odds.json (Rundown API response or matchups)."""
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    out: list[tuple[str, str]] = []
    # Format 1: { "events": [ ... ] } (Rundown API)
    events = data.get("events") or data.get("event")
    if isinstance(events, list):
        for ev in events:
            away, home = _teams_from_rundown_event(ev)
            if away and home:
                out.append((away, home))
        return out
    # Format 2: { "matchups": [ { "away_team", "home_team" } ] }
    matchups = data.get("matchups")
    if isinstance(matchups, list):
        for m in matchups:
            if not isinstance(m, dict):
                continue
            away = (m.get("away_team") or "").strip()
            home = (m.get("home_team") or "").strip()
            if away and home:
                out.append((away, home))
    return out


def _game_key(away: str, home: str) -> tuple[str, str]:
    """Normalized key for comparison: (core_away, core_home)."""
    return (_team_core_name(away), _team_core_name(home))


def _similarity(a: str, b: str) -> float:
    """0..1 similarity."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _best_fuzzy_match(
    espn_away: str,
    espn_home: str,
    rundown_games: list[tuple[str, str]],
) -> tuple[str, str, float] | None:
    """Return (rundown_away, rundown_home, score) for best matching Rundown game, or None."""
    if not rundown_games:
        return None
    ek_away = _team_core_name(espn_away)
    ek_home = _team_core_name(espn_home)
    best: tuple[str, str, float] | None = None
    for r_away, r_home in rundown_games:
        rk_away = _team_core_name(r_away)
        rk_home = _team_core_name(r_home)
        # Same matchup (order-independent): both away and home match
        s_away = max(
            _similarity(ek_away, rk_away),
            _similarity(ek_away, rk_home),
            _similarity(_resolve_alias(espn_away), rk_away),
            _similarity(_resolve_alias(espn_away), rk_home),
        )
        s_home = max(
            _similarity(ek_home, rk_home),
            _similarity(ek_home, rk_away),
            _similarity(_resolve_alias(espn_home), rk_home),
            _similarity(_resolve_alias(espn_home), rk_away),
        )
        score = (s_away + s_home) / 2.0
        # Require both sides to have some match (avoid suggesting unrelated games)
        if score >= 0.55 and min(s_away, s_home) >= 0.35 and (best is None or score > best[2]):
            best = (r_away, r_home, score)
    return best


def main() -> int:
    espn_path = ESPN_PATH
    rapid_path = RAPID_ODDS_PATH
    if len(sys.argv) >= 2:
        rapid_path = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        espn_path = Path(sys.argv[2])

    if not espn_path.exists():
        print(f"ESPN source of truth not found: {espn_path}", file=sys.stderr)
        print("Run scripts/fetch_espn_master_schedule.py first.", file=sys.stderr)
        return 1
    if not rapid_path.exists():
        print(f"Rundown data not found: {rapid_path}", file=sys.stderr)
        print("Save your latest Rundown API response to data/cache/rapid_odds.json", file=sys.stderr)
        return 1

    espn_games = _load_espn_matchups(espn_path)
    rundown_games = _load_rundown_games(rapid_path)
    rundown_keys = {_game_key(a, h) for a, h in rundown_games}

    # Also allow match if core names match in either order (away/home swap)
    def _in_rundown(away: str, home: str) -> bool:
        k = _game_key(away, home)
        if k in rundown_keys:
            return True
        # swapped
        k2 = (k[1], k[0])
        return k2 in rundown_keys

    missing: list[tuple[str, str, str]] = []
    for gid, away, home in espn_games:
        if not _in_rundown(away, home):
            missing.append((gid, away, home))

    print("=" * 70)
    print("Missing games: ESPN has them, The Rundown does not")
    print("=" * 70)
    print(f"ESPN total games:     {len(espn_games)}")
    print(f"Rundown total games:   {len(rundown_games)}")
    print(f"Missing from Rundown: {len(missing)}")
    print()

    if not missing:
        print("No missing games.")
        return 0

    for gid, away, home in missing:
        print(f"  [{gid}]  {away}  @  {home}")
        fuzzy = _best_fuzzy_match(away, home, rundown_games)
        if fuzzy:
            r_away, r_home, score = fuzzy
            print(f"         Fuzzy match (score {score:.2f}):  {r_away} @ {r_home}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
