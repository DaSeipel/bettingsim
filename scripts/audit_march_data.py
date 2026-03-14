#!/usr/bin/env python3
"""
Audit 2026 NCAAB team statistics for Tournament Multipliers (Veteran Edge, Closer Factor).
Verifies ncaab_team_season_stats has the data needed by engine/bracket_analysis.py and flags
missing/zero FT% or 3P%, and naming consistency with games_with_team_stats.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "espn.db"

QUERY = """
SELECT
    TEAM AS team_name,
    free_throw_pct,
    three_point_pct,
    roster_experience_years
FROM ncaab_team_season_stats
WHERE season = 2026
ORDER BY roster_experience_years DESC, free_throw_pct DESC
LIMIT 25;
"""


def get_table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def main() -> int:
    if not DB_PATH.exists():
        print(f"ERROR: Database not found: {DB_PATH}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Check that ncaab_team_season_stats exists and has required columns
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ncaab_team_season_stats'"
        )
        if cur.fetchone() is None:
            print("ERROR: Table ncaab_team_season_stats does not exist.", file=sys.stderr)
            conn.close()
            return 1
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        conn.close()
        return 1

    cols = get_table_columns(conn, "ncaab_team_season_stats")
    if "TEAM" not in cols:
        print("ERROR: ncaab_team_season_stats has no TEAM column.", file=sys.stderr)
        conn.close()
        return 1

    has_ft = "free_throw_pct" in cols
    has_3p = "three_point_pct" in cols
    has_exp = "roster_experience_years" in cols
    has_ftr = "FTR" in cols
    has_3po = "3P_O" in cols

    # Prefer exact columns for Tournament Multipliers; fallback to FTR / 3P_O if missing
    if not has_ft:
        print("WARNING: free_throw_pct not in table; using FTR*100 if present.", file=sys.stderr)
    if not has_3p:
        print("WARNING: three_point_pct not in table; using 3P_O if present.", file=sys.stderr)
    if not has_exp:
        print("WARNING: roster_experience_years not in table.", file=sys.stderr)

    select_ft = "free_throw_pct" if has_ft else ("FTR * 100.0 AS free_throw_pct" if has_ftr else "NULL AS free_throw_pct")
    select_3p = "three_point_pct" if has_3p else (
        "CASE WHEN \"3P_O\" <= 1 THEN \"3P_O\" * 100.0 ELSE \"3P_O\" END AS three_point_pct" if has_3po else "NULL AS three_point_pct"
    )
    select_exp = "roster_experience_years" if has_exp else "NULL AS roster_experience_years"
    order_exp = "COALESCE(roster_experience_years, -1) DESC" if has_exp else "1"
    order_ft = "COALESCE(free_throw_pct, -1) DESC" if has_ft else ("COALESCE(FTR, -1) DESC" if has_ftr else "1")

    query = f"""
SELECT
    TEAM AS team_name,
    {select_ft},
    {select_3p},
    {select_exp}
FROM ncaab_team_season_stats
WHERE season = 2026
ORDER BY {order_exp}, {order_ft}
LIMIT 40;
"""

    try:
        rows = conn.execute(query).fetchall()
    except sqlite3.OperationalError as e:
        print(f"ERROR running query: {e}", file=sys.stderr)
        conn.close()
        return 1

    if not rows:
        print("No rows returned (no season = 2026 in ncaab_team_season_stats?).")
        conn.close()
        return 0

    # Unique team names from games table (for join consistency)
    games_teams: set[str] = set()
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('games_with_team_stats', 'games')"
    )
    tables = [r[0] for r in cur.fetchall()]
    for tbl in ["games_with_team_stats", "games"]:
        if tbl not in tables:
            continue
        try:
            for col in ("home_team_name", "away_team_name"):
                cur = conn.execute(f"SELECT DISTINCT {col} FROM {tbl} WHERE league = 'ncaab'")
                games_teams.update(str(r[0]).strip() for r in cur.fetchall() if r[0])
        except sqlite3.OperationalError:
            continue
        break

    conn.close()

    # Validation: flag 0 or NULL for free_throw_pct or three_point_pct
    flagged = []
    for r in rows:
        name = (r["team_name"] or "").strip()
        ft = r["free_throw_pct"]
        tp = r["three_point_pct"]
        ft_bad = ft is None or (isinstance(ft, (int, float)) and float(ft) == 0)
        tp_bad = tp is None or (isinstance(tp, (int, float)) and float(tp) == 0)
        if ft_bad or tp_bad:
            reasons = []
            if ft_bad:
                reasons.append("free_throw_pct 0/NULL")
            if tp_bad:
                reasons.append("three_point_pct 0/NULL")
            flagged.append((name, reasons))

    # Naming: stats team not in games
    stats_teams = [(r["team_name"] or "").strip() for r in rows]
    missing_in_games = [t for t in stats_teams if t and t not in games_teams]

    # Print table
    print("=" * 90)
    print("Top 40 Veteran & High-Efficiency (2026) — ncaab_team_season_stats")
    print("=" * 90)
    fmt = "{:<28} {:>12} {:>14} {:>10}"
    print(fmt.format("team_name", "ft_pct", "three_p_pct", "exp_yrs"))
    print("-" * 90)
    def _fmt_val(v):
        if v is None:
            return "NULL"
        try:
            f = float(v)
            return f"{f:.1f}" if f != 0 else "0"
        except (TypeError, ValueError):
            return str(v)

    for r in rows:
        name = (r["team_name"] or "—")[:28]
        ft_s = _fmt_val(r["free_throw_pct"])
        tp_s = _fmt_val(r["three_point_pct"])
        exp_s = _fmt_val(r["roster_experience_years"])
        print(fmt.format(name, ft_s, tp_s, exp_s))
    print("=" * 90)

    top40_names = {(r["team_name"] or "").strip() for r in rows}
    n_flagged_in_top40 = len([name for name, _ in flagged if name in top40_names]) if flagged else 0

    if flagged:
        print("\n*** VALIDATION FLAGS (0 or NULL — Tournament Multipliers may skip these teams):")
        for name, reasons in flagged:
            print(f"  • {name}: {', '.join(reasons)}")
        if n_flagged_in_top40 > 0:
            print(f"\n*** FAIL: {n_flagged_in_top40} team(s) in Top 40 have 0/NULL FT% or 3P%. Run update_ncaab_march_stats.py and ensure VALIDATION FLAGS are clear for at least the Top 40 before Selection Show.")
        else:
            print("\n(Flagged teams are outside Top 40; Top 40 is clear.)")
    else:
        print("\nValidation: No teams in Top 40 with 0/NULL free_throw_pct or three_point_pct. OK for Selection Show.")

    if missing_in_games and games_teams:
        print("\n*** NAMING: Stats team_name not found in games (possible join errors in simulation):")
        for t in missing_in_games:
            print(f"  • {t}")
    elif games_teams:
        print("\nNaming: All top 25 stats team_names appear in games table.")
    else:
        print("\nNaming: No games table / NCAAB games to compare (skipping consistency check).")

    print()
    # Exit 1 if any Top 40 team has validation flags (so CI/cron can require clear before Selection Show)
    return 1 if n_flagged_in_top40 > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
