"""CFB Phase 0 validation report."""

from __future__ import annotations

import sqlite3
from typing import Any


def _table_season_counts(conn: sqlite3.Connection, table: str, season_col: str = "season") -> list[tuple[int, int]]:
    rows = conn.execute(
        f"SELECT {season_col}, COUNT(*) FROM {table} GROUP BY {season_col} ORDER BY {season_col}"
    ).fetchall()
    return [(int(r[0]), int(r[1])) for r in rows if r[0] is not None]


def validate_cfb_data(conn: sqlite3.Connection) -> dict[str, Any]:
    report: dict[str, Any] = {}

    tables_with_season = [
        "cfb_games",
        "cfb_team_stats_adv",
        "cfb_ppa",
        "cfb_ratings_sp",
        "cfb_returning",
    ]
    report["row_counts_by_season"] = {
        table: _table_season_counts(conn, table) for table in tables_with_season
    }

    report["game_stats_rows_by_season"] = conn.execute(
        """
        SELECT g.season, COUNT(*)
        FROM cfb_game_stats gs
        JOIN cfb_games g ON g.game_id = gs.game_id
        GROUP BY g.season
        ORDER BY g.season
        """
    ).fetchall()

    report["lines_rows_by_season"] = conn.execute(
        """
        SELECT g.season, COUNT(*)
        FROM cfb_lines l
        JOIN cfb_games g ON g.game_id = l.game_id
        GROUP BY g.season
        ORDER BY g.season
        """
    ).fetchall()

    report["games_per_season"] = conn.execute(
        """
        SELECT season, COUNT(*)
        FROM cfb_games
        GROUP BY season
        ORDER BY season
        """
    ).fetchall()

    fbs_line_coverage = conn.execute(
        """
        WITH fbs_games AS (
            SELECT game_id, season
            FROM cfb_games
            WHERE lower(coalesce(home_division, '')) = 'fbs'
              AND lower(coalesce(away_division, '')) = 'fbs'
        ),
        lined AS (
            SELECT DISTINCT game_id FROM cfb_lines
        )
        SELECT fg.season,
               COUNT(*) AS fbs_games,
               SUM(CASE WHEN l.game_id IS NOT NULL THEN 1 ELSE 0 END) AS with_line,
               ROUND(100.0 * SUM(CASE WHEN l.game_id IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct
        FROM fbs_games fg
        LEFT JOIN lined l ON l.game_id = fg.game_id
        GROUP BY fg.season
        ORDER BY fg.season
        """
    ).fetchall()
    report["fbs_vs_fbs_line_coverage"] = fbs_line_coverage
    report["seasons_below_95pct_line_coverage"] = [
        row for row in fbs_line_coverage if row[3] is not None and float(row[3]) < 95.0
    ]

    report["scored_games_without_line"] = conn.execute(
        """
        SELECT g.season, g.week, g.game_id, g.away_team, g.home_team
        FROM cfb_games g
        LEFT JOIN (SELECT DISTINCT game_id FROM cfb_lines) l ON l.game_id = g.game_id
        WHERE g.home_points IS NOT NULL
          AND g.away_points IS NOT NULL
          AND l.game_id IS NULL
        ORDER BY g.season, g.week, g.game_id
        LIMIT 50
        """
    ).fetchall()

    report["lines_without_score"] = conn.execute(
        """
        SELECT g.season, g.week, g.game_id, g.away_team, g.home_team
        FROM cfb_games g
        JOIN (SELECT DISTINCT game_id FROM cfb_lines) l ON l.game_id = g.game_id
        WHERE g.home_points IS NULL OR g.away_points IS NULL
        ORDER BY g.season, g.week, g.game_id
        LIMIT 50
        """
    ).fetchall()

    completed_games = conn.execute(
        """
        SELECT season, week, game_id
        FROM cfb_games
        WHERE home_points IS NOT NULL AND away_points IS NOT NULL
        """
    ).fetchall()
    games_with_stats = {
        row[0]
        for row in conn.execute("SELECT DISTINCT game_id FROM cfb_game_stats").fetchall()
    }
    missing_stats = [
        (season, week, game_id)
        for season, week, game_id in completed_games
        if game_id not in games_with_stats
    ]
    report["completed_games_missing_stats_count"] = len(missing_stats)
    report["completed_games_missing_stats_sample"] = missing_stats[:50]

    missing_by_season_week: dict[tuple[int, int | None], int] = {}
    for season, week, _game_id in missing_stats:
        key = (season, week)
        missing_by_season_week[key] = missing_by_season_week.get(key, 0) + 1
    report["missing_game_stats_by_season_week"] = sorted(
        [{"season": s, "week": w, "missing_count": c} for (s, w), c in missing_by_season_week.items()],
        key=lambda x: (x["season"], x["week"] if x["week"] is not None else -1),
    )

    alias_rows = conn.execute(
        """
        SELECT canonical_name, cfbd_name, odds_api_name
        FROM cfb_team_alias
        ORDER BY canonical_name
        """
    ).fetchall()
    unmatched_cfbd = [row[0] for row in alias_rows if row[2] is None]
    odds_names = conn.execute(
        "SELECT odds_api_name FROM cfb_team_alias WHERE odds_api_name IS NOT NULL"
    ).fetchall()
    report["alias_rows"] = len(alias_rows)
    report["unmatched_cfbd_names"] = unmatched_cfbd
    report["matched_alias_count"] = len(odds_names)

    return report


def print_validation_report(report: dict[str, Any], *, cfbd_calls: int) -> None:
    print("\n=== CFB Phase 0 Validation ===")

    print("\n(a) Row counts per table per season")
    for table, rows in report.get("row_counts_by_season", {}).items():
        print(f"  {table}:")
        for season, count in rows:
            print(f"    {season}: {count}")

    print("\n  cfb_game_stats (via join):")
    for season, count in report.get("game_stats_rows_by_season", []):
        print(f"    {season}: {count}")

    print("\n  cfb_lines (via join):")
    for season, count in report.get("lines_rows_by_season", []):
        print(f"    {season}: {count}")

    print("\n(b) Games per season")
    for season, count in report.get("games_per_season", []):
        print(f"  {season}: {count}")

    print("\n(c) FBS-vs-FBS games with at least one line record")
    for season, total, with_line, pct in report.get("fbs_vs_fbs_line_coverage", []):
        flag = "  *** BELOW 95%" if pct is not None and float(pct) < 95.0 else ""
        print(f"  {season}: {with_line}/{total} ({pct}%){flag}")

    print("\n(d) Score/line mismatches (sample up to 50 each)")
    print("  Scored games without line:", len(report.get("scored_games_without_line", [])))
    for row in report.get("scored_games_without_line", [])[:10]:
        print("   ", row)
    print("  Lines without final score:", len(report.get("lines_without_score", [])))
    for row in report.get("lines_without_score", [])[:10]:
        print("   ", row)

    print("\n(e) Completed games missing cfb_game_stats")
    print("  missing count:", report.get("completed_games_missing_stats_count", 0))
    for item in report.get("missing_game_stats_by_season_week", [])[:20]:
        print(f"  season={item['season']} week={item['week']} missing={item['missing_count']}")

    print("\n(f) Unmatched team names (exact-match only)")
    unmatched = report.get("unmatched_cfbd_names", [])
    print(f"  unmatched CFBD FBS names ({len(unmatched)}):")
    for name in unmatched:
        print(f"    - {name}")

    print("\n(g) Total CFBD API calls consumed:", cfbd_calls)
