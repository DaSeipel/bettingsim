#!/usr/bin/env python3
"""
For the 3 Walters Play matchups (UCLA vs UCF, Ohio State vs TCU, Miami vs Missouri),
pull raw experience/roster data from ncaab_team_season_stats for all 6 teams.
Explain why experience is 0 or null (missing, not imported, or below veteran threshold 2.2).
Print actual roster experience averages per team.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Bracket name -> possible DB names (ncaab_team_season_stats.TEAM)
TEAM_DB_NAMES = {
    "UCLA": ["UCLA", "California Los Angeles"],
    "UCF": ["UCF"],
    "Ohio State": ["Ohio State", "Ohio St."],
    "TCU": ["TCU"],
    "Miami": ["Miami", "Miami FL"],
    "Missouri": ["Missouri"],
}

VETERAN_THRESHOLD = 2.2  # bracket_analysis.VETERAN_EXP_MIN


def main() -> int:
    db_path = ROOT / "data" / "espn.db"
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        return 1

    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Prefer 2026 season; fallback to latest available
    cursor = conn.execute(
        "SELECT DISTINCT season FROM ncaab_team_season_stats ORDER BY season DESC LIMIT 1"
    )
    row = cursor.fetchone()
    latest_season = row["season"] if row else None

    print("=" * 70)
    print("  Walters Play matchups: roster experience from ncaab_team_season_stats")
    print("=" * 70)
    print(f"  Season used: {latest_season} (latest in DB)")
    print(f"  Veteran threshold (VETERAN_EXP_MIN): {VETERAN_THRESHOLD} yrs")
    print()

    teams_to_resolve = list(TEAM_DB_NAMES.keys())
    found_teams = {}

    for display_name in teams_to_resolve:
        candidates = TEAM_DB_NAMES[display_name]
        for db_name in candidates:
            if latest_season is not None:
                cursor = conn.execute(
                    "SELECT TEAM, season, roster_experience_years, free_throw_pct, three_point_pct "
                    "FROM ncaab_team_season_stats WHERE TEAM = ? AND season = ?",
                    (db_name, latest_season),
                )
            else:
                cursor = conn.execute(
                    "SELECT TEAM, season, roster_experience_years, free_throw_pct, three_point_pct "
                    "FROM ncaab_team_season_stats WHERE TEAM = ? ORDER BY season DESC LIMIT 1",
                    (db_name,),
                )
            row = cursor.fetchone()
            if row is not None:
                found_teams[display_name] = dict(row)
                break
        if display_name not in found_teams:
            # Try any season
            for db_name in candidates:
                cursor = conn.execute(
                    "SELECT TEAM, season, roster_experience_years, free_throw_pct, three_point_pct "
                    "FROM ncaab_team_season_stats WHERE TEAM = ? ORDER BY season DESC LIMIT 1",
                    (db_name,),
                )
                row = cursor.fetchone()
                if row is not None:
                    found_teams[display_name] = dict(row)
                    break

    # Print table: Team (display), DB name, season, roster_experience_years, diagnosis
    print(f"  {'Team (display)':<16} {'TEAM in DB':<22} {'Season':<8} {'Exp (yrs)':<10}  Diagnosis")
    print("-" * 70)

    for display_name in teams_to_resolve:
        if display_name not in found_teams:
            print(f"  {display_name:<16} {'(not in DB)':<22} {'—':<8} {'—':<10}  Data missing: no row in ncaab_team_season_stats")
            continue
        r = found_teams[display_name]
        db_team = r["TEAM"]
        season = r["season"]
        exp = r["roster_experience_years"]
        exp_str = f"{exp:.2f}" if exp is not None and str(exp).strip() != "" else "NULL"
        exp_val = float(exp) if exp is not None and str(exp).strip() != "" else None

        if exp_val is None:
            diagnosis = "Data missing / not imported: column empty for this team+season"
        elif exp_val == 0:
            diagnosis = "Stored as 0: either not imported or true 0 (check source)"
        elif exp_val < VETERAN_THRESHOLD:
            diagnosis = f"Imported; below veteran threshold ({VETERAN_THRESHOLD} yrs) — no veteran edge"
        else:
            diagnosis = f"Imported; meets veteran threshold (≥{VETERAN_THRESHOLD} yrs)"

        print(f"  {display_name:<16} {db_team:<22} {season!s:<8} {exp_str:<10}  {diagnosis}")

    conn.close()

    print()
    print("  Note: In bracket_analysis, NULL/empty experience is stored as 0.0 in team_stats,")
    print("  so the app shows 0 yrs for UCF even though the DB has no value imported.")
    print()
    print("  Summary: Actual roster experience averages (from DB)")
    print("  " + "-" * 50)
    for display_name in teams_to_resolve:
        if display_name not in found_teams:
            print(f"  {display_name}: (no data)")
            continue
        r = found_teams[display_name]
        exp = r["roster_experience_years"]
        if exp is not None and str(exp).strip() != "":
            print(f"  {display_name}: {float(exp):.2f} yrs")
        else:
            print(f"  {display_name}: NULL / not imported")

    return 0


if __name__ == "__main__":
    sys.exit(main())
