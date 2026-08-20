#!/usr/bin/env python3
"""CFB Phase 0 completion: manual aliases, backtest line flags, reports. No CFBD calls."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.cfb_config import ESPN_DB_PATH
from engine.cfb_schema import ensure_cfb_schema

MANUAL_ALIASES: dict[str, str] = {
    "Arizona": "Arizona Wildcats",
    "Arkansas": "Arkansas Razorbacks",
    "Colorado": "Colorado Buffaloes",
    "Florida": "Florida Gators",
    "Georgia": "Georgia Bulldogs",
    "Houston": "Houston Cougars",
    "Indiana": "Indiana Hoosiers",
    "Iowa": "Iowa Hawkeyes",
    "Kansas": "Kansas Jayhawks",
    "Michigan": "Michigan Wolverines",
    "Missouri": "Missouri Tigers",
    "New Mexico": "New Mexico Lobos",
    "North Carolina": "North Carolina Tar Heels",
    "Northwestern": "Northwestern Wildcats",
    "Oklahoma": "Oklahoma Sooners",
    "Oregon": "Oregon Ducks",
    "Tennessee": "Tennessee Volunteers",
    "Texas": "Texas Longhorns",
    "Utah": "Utah Utes",
    "Virginia": "Virginia Cavaliers",
    "Miami": "Miami Hurricanes",
    "Ohio": "Ohio Bobcats",
    "Washington": "Washington Huskies",
    "Louisiana": "Louisiana Ragin Cajuns",
    "App State": "Appalachian State Mountaineers",
    "Massachusetts": "UMass Minutemen",
    "Southern Miss": "Southern Mississippi Golden Eagles",
    "Maryland": "Maryland Terrapins",
}

VERIFY_TEAMS = (
    "Miami",
    "Miami (OH)",
    "Ohio",
    "Ohio State",
    "Washington",
    "Washington State",
    "Louisiana",
    "UL Monroe",
)


def apply_manual_aliases(conn: sqlite3.Connection) -> None:
    for cfbd, odds in MANUAL_ALIASES.items():
        conn.execute(
            """
            UPDATE cfb_team_alias
            SET odds_api_name = ?, match_method = 'manual'
            WHERE cfbd_name = ? OR canonical_name = ?
            """,
            (odds, cfbd, cfbd),
        )
    conn.commit()


def ensure_unique_odds_api_name(conn: sqlite3.Connection) -> None:
    dups = conn.execute(
        """
        SELECT odds_api_name, COUNT(*)
        FROM cfb_team_alias
        WHERE odds_api_name IS NOT NULL
        GROUP BY odds_api_name
        HAVING COUNT(*) > 1
        """
    ).fetchall()
    if dups:
        raise RuntimeError(f"Cannot add UNIQUE on odds_api_name; duplicates: {dups}")
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_cfb_team_alias_odds_api_name
        ON cfb_team_alias(odds_api_name)
        WHERE odds_api_name IS NOT NULL
        """
    )
    conn.commit()


def ensure_backtest_reference_column(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(cfb_lines)").fetchall()}
    if "is_backtest_reference" not in cols:
        conn.execute(
            "ALTER TABLE cfb_lines ADD COLUMN is_backtest_reference INTEGER NOT NULL DEFAULT 0"
        )
    conn.execute("UPDATE cfb_lines SET is_backtest_reference = 0")
    conn.execute(
        """
        UPDATE cfb_lines
        SET is_backtest_reference = 1
        WHERE provider = 'Bovada'
          AND game_id IN (SELECT game_id FROM cfb_games WHERE season BETWEEN 2021 AND 2025)
        """
    )
    conn.execute(
        """
        UPDATE cfb_lines
        SET is_backtest_reference = 1
        WHERE provider = 'teamrankings'
          AND game_id IN (SELECT game_id FROM cfb_games WHERE season BETWEEN 2015 AND 2020)
        """
    )
    conn.commit()


def verify_aliases(conn: sqlite3.Connection) -> None:
    print("\n=== 1. Alias verification ===")
    total = conn.execute("SELECT COUNT(*) FROM cfb_team_alias").fetchone()[0]
    mapped = conn.execute(
        "SELECT COUNT(*) FROM cfb_team_alias WHERE odds_api_name IS NOT NULL"
    ).fetchone()[0]
    unmapped = conn.execute(
        "SELECT cfbd_name FROM cfb_team_alias WHERE odds_api_name IS NULL ORDER BY cfbd_name"
    ).fetchall()
    dups = conn.execute(
        """
        SELECT odds_api_name, GROUP_CONCAT(cfbd_name, ' | ')
        FROM cfb_team_alias
        WHERE odds_api_name IS NOT NULL
        GROUP BY odds_api_name
        HAVING COUNT(*) > 1
        """
    ).fetchall()
    print(f"Total FBS alias rows: {total}")
    print(f"Mapped (odds_api_name set): {mapped}")
    if unmapped:
        print(f"Unmapped CFBD FBS ({len(unmapped)}):")
        for (name,) in unmapped:
            print(f"  - {name}")
    else:
        print("Every CFBD FBS team has exactly one alias row with odds_api_name set.")
    if dups:
        print("DUPLICATE odds_api_name claims:")
        for odds, teams in dups:
            print(f"  {odds}: {teams}")
    else:
        print("No Odds API name is claimed by two CFBD teams.")
    print("\nSpot-check rows:")
    for team in VERIFY_TEAMS:
        row = conn.execute(
            """
            SELECT canonical_name, cfbd_name, odds_api_name, match_method
            FROM cfb_team_alias
            WHERE cfbd_name = ? OR canonical_name = ?
            """,
            (team, team),
        ).fetchone()
        print(f"  {team}: {row}")


def report_backtest_coverage(conn: sqlite3.Connection) -> None:
    print("\n=== 2. Backtest reference line coverage (FBS-vs-FBS) ===")
    rows = conn.execute(
        """
        WITH fbs AS (
            SELECT game_id, season
            FROM cfb_games
            WHERE lower(coalesce(home_division, '')) = 'fbs'
              AND lower(coalesce(away_division, '')) = 'fbs'
        )
        SELECT
            f.season,
            COUNT(*) AS fbs_games,
            SUM(CASE WHEN l.game_id IS NOT NULL THEN 1 ELSE 0 END) AS with_ref,
            ROUND(100.0 * SUM(CASE WHEN l.game_id IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct
        FROM fbs f
        LEFT JOIN cfb_lines l ON l.game_id = f.game_id AND l.is_backtest_reference = 1
        GROUP BY f.season
        ORDER BY f.season
        """
    ).fetchall()
    below = []
    for season, n, with_ref, pct in rows:
        flag = "  *** BELOW 97%" if pct < 97.0 else ""
        if pct < 97.0:
            below.append((season, pct))
        print(f"  {season}: {with_ref}/{n} ({pct}%){flag}")
    if below:
        print("Seasons below 97%:", below)
    else:
        print("All seasons >= 97% reference-line coverage.")


def report_provider_overlap(conn: sqlite3.Connection) -> None:
    print("\n=== 2b. Bovada vs teamrankings overlap (2021-2022, same game) ===")
    summary = conn.execute(
        """
        WITH fbs AS (
            SELECT game_id, season, week, home_team, away_team
            FROM cfb_games
            WHERE season IN (2021, 2022)
              AND lower(coalesce(home_division, '')) = 'fbs'
              AND lower(coalesce(away_division, '')) = 'fbs'
        ),
        paired AS (
            SELECT
                f.season,
                f.week,
                f.game_id,
                f.home_team,
                f.away_team,
                b.spread AS bovada_spread,
                t.spread AS tr_spread,
                ABS(b.spread - t.spread) AS abs_diff
            FROM fbs f
            JOIN cfb_lines b ON b.game_id = f.game_id AND b.provider = 'Bovada'
            JOIN cfb_lines t ON t.game_id = f.game_id AND t.provider = 'teamrankings'
            WHERE b.spread IS NOT NULL AND t.spread IS NOT NULL
        )
        SELECT
            COUNT(*) AS n_games,
            AVG(abs_diff) AS mean_abs_diff,
            SUM(CASE WHEN abs_diff > 1.0 THEN 1 ELSE 0 END) AS gt_1,
            SUM(CASE WHEN abs_diff > 2.0 THEN 1 ELSE 0 END) AS gt_2
        FROM paired
        """
    ).fetchone()
    print(
        f"  games with both: {summary[0]}",
        f"mean |spread diff|: {summary[1]:.3f}" if summary[1] is not None else "",
        f">1.0 pt: {summary[2]}",
        f">2.0 pt: {summary[3]}",
    )
    print("  Largest 10 disagreements:")
    top = conn.execute(
        """
        WITH fbs AS (
            SELECT game_id, season, week, home_team, away_team
            FROM cfb_games
            WHERE season IN (2021, 2022)
              AND lower(coalesce(home_division, '')) = 'fbs'
              AND lower(coalesce(away_division, '')) = 'fbs'
        )
        SELECT
            f.season,
            f.week,
            f.away_team,
            f.home_team,
            b.spread,
            t.spread,
            ABS(b.spread - t.spread) AS abs_diff
        FROM fbs f
        JOIN cfb_lines b ON b.game_id = f.game_id AND b.provider = 'Bovada'
        JOIN cfb_lines t ON t.game_id = f.game_id AND t.provider = 'teamrankings'
        WHERE b.spread IS NOT NULL AND t.spread IS NOT NULL
        ORDER BY abs_diff DESC
        LIMIT 10
        """
    ).fetchall()
    for row in top:
        print(f"    {row}")


def report_open_vs_close(conn: sqlite3.Connection) -> None:
    print("\n=== 3. Opening vs closing (FBS-vs-FBS, spread_open and spread both set) ===")
    rows = conn.execute(
        """
        WITH fbs AS (
            SELECT game_id, season
            FROM cfb_games
            WHERE lower(coalesce(home_division, '')) = 'fbs'
              AND lower(coalesce(away_division, '')) = 'fbs'
        )
        SELECT
            g.season,
            l.provider,
            COUNT(*) AS n,
            AVG(ABS(l.spread - l.spread_open)) AS mean_move,
            ROUND(100.0 * SUM(CASE WHEN ABS(l.spread - l.spread_open) >= 1.0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_moved_1plus
        FROM cfb_lines l
        JOIN fbs g ON g.game_id = l.game_id
        WHERE l.spread IS NOT NULL
          AND l.spread_open IS NOT NULL
        GROUP BY g.season, l.provider
        HAVING COUNT(*) > 0
        ORDER BY g.season, l.provider
        """
    ).fetchall()
    for season, provider, n, mean_move, pct in rows:
        print(
            f"  {season} {provider}: n={n}, mean |move|={mean_move:.3f}, "
            f"moved 1+ pt={pct}%"
        )


def main() -> int:
    conn = sqlite3.connect(ESPN_DB_PATH)
    try:
        ensure_cfb_schema(conn)
        apply_manual_aliases(conn)
        ensure_unique_odds_api_name(conn)
        ensure_backtest_reference_column(conn)
        verify_aliases(conn)
        report_backtest_coverage(conn)
        report_provider_overlap(conn)
        report_open_vs_close(conn)
    finally:
        conn.close()
    print("\nCFB Phase 0 completion script done. CFBD calls this run: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
