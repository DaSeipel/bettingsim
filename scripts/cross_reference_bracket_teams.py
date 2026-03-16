#!/usr/bin/env python3
"""
Cross-reference every team name in data/ncaab/bracket_2026.csv against
ncaab_team_season_stats in the database. Print mismatches as:
  [CSV Name] → [Closest DB Match]
for confirmation before updating the alias map.
"""
from pathlib import Path
import sqlite3
import sys
from difflib import get_close_matches
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.bracket_analysis import resolve_team_name, parse_bracket_csv


def normalize(s: str) -> str:
    return (s or "").strip().lower()


def find_in_db(name: str, db_teams: list[str]) -> Optional[str]:
    """Return DB team if name matches (exact, normalized, or substring); else None."""
    if not name or not db_teams:
        return None
    n = normalize(name)
    if name in db_teams:
        return name
    db_lower = [t.strip().lower() for t in db_teams]
    if n in db_lower:
        return db_teams[db_lower.index(n)]
    for st in db_teams:
        if n == normalize(st):
            return st
        if n in st.lower() or st.lower() in n:
            return st
    return None


# Known bracket display names -> DB name (for better suggestion when similarity is ambiguous)
KNOWN_BRACKET_TO_DB = {
    "Long Island": "LIU",
}

def closest_db_match(name: str, db_teams: list[str], n: int = 3) -> Optional[str]:
    """Return closest DB team by string similarity, or known mapping."""
    if not name or not db_teams:
        return None
    if name in KNOWN_BRACKET_TO_DB and KNOWN_BRACKET_TO_DB[name] in db_teams:
        return KNOWN_BRACKET_TO_DB[name]
    matches = get_close_matches(name, db_teams, n=n, cutoff=0.4)
    return matches[0] if matches else None


def main() -> int:
    bracket_path = ROOT / "data" / "ncaab" / "bracket_2026.csv"
    if not bracket_path.exists():
        print(f"Not found: {bracket_path}")
        return 1

    bracket_content = bracket_path.read_text()
    bracket = parse_bracket_csv(bracket_content)
    bracket_teams = set()
    for m in bracket:
        bracket_teams.add(m["team_a"].strip())
        bracket_teams.add(m["team_b"].strip())
    bracket_teams = sorted(bracket_teams)

    db_path = ROOT / "data" / "espn.db"
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return 1
    conn = sqlite3.connect(db_path)
    try:
        db_teams = pd.read_sql_query(
            "SELECT DISTINCT TEAM FROM ncaab_team_season_stats WHERE season IN (2026, 2025) ORDER BY TEAM",
            conn,
        )
    except Exception as e:
        print(f"DB error: {e}")
        return 1
    finally:
        conn.close()

    db_list = db_teams["TEAM"].astype(str).str.strip().dropna().tolist()

    # For each bracket team: resolve with current alias map, then check DB
    mismatches = []
    for team in bracket_teams:
        resolved = resolve_team_name(team)
        found = find_in_db(resolved, db_list)
        if found is None:
            found = find_in_db(team, db_list)
        if found is not None:
            continue
        closest = closest_db_match(team, db_list) or closest_db_match(resolved, db_list)
        mismatches.append((team, closest or "(no close match)"))

    if not mismatches:
        print("All bracket teams matched the database (exact or via current alias map).")
        return 0

    print("Bracket team names that do not match the database (exact or via current alias map):")
    print("(Add CSV Name → Closest DB Match to BRACKET_TEAM_ALIAS_MAP if correct.)")
    print()
    max_csv = max(len(m[0]) for m in mismatches)
    fmt = f"  {{:{max_csv}}}  →  {{}}"
    print(f"  {'CSV Name':<{max_csv}}  →  Closest DB Match")
    print("  " + "-" * (max_csv + 6 + max(len(str(m[1])) for m in mismatches)))
    for csv_name, closest in mismatches:
        print(fmt.format(csv_name, closest))
    print()
    print("If the 'Closest DB Match' is correct for a row, add to BRACKET_TEAM_ALIAS_MAP:")
    print("  \"CSV Name\" -> \"Closest DB Match\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
