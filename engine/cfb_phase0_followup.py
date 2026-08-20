"""Phase 0 follow-up reports: stats coverage, season flags, lines providers. No CFBD calls."""

from __future__ import annotations

import sqlite3
from typing import Any


def game_stats_row_shape(conn: sqlite3.Connection) -> dict[str, Any]:
    dist = conn.execute(
        """
        SELECT cnt, COUNT(*)
        FROM (SELECT game_id, COUNT(*) AS cnt FROM cfb_game_stats GROUP BY game_id)
        GROUP BY cnt
        ORDER BY cnt
        """
    ).fetchall()
    pk = conn.execute("PRAGMA table_info(cfb_game_stats)").fetchall()
    pk_cols = [row[1] for row in pk if row[5]]
    return {
        "rows_per_game_id_distribution": dist,
        "primary_key_columns": pk_cols,
        "one_row_per_team_per_game": pk_cols == ["game_id", "team"] or set(pk_cols) == {"game_id", "team"},
    }


def stats_coverage_by_population(conn: sqlite3.Connection) -> dict[str, Any]:
    """Completed games only. FBS-vs-FBS vs FBS-vs-FCS."""
    sql = """
    WITH completed AS (
        SELECT
            game_id,
            season,
            lower(coalesce(home_division, '')) AS home_div,
            lower(coalesce(away_division, '')) AS away_div
        FROM cfb_games
        WHERE home_points IS NOT NULL AND away_points IS NOT NULL
    ),
    stats_n AS (
        SELECT game_id, COUNT(*) AS n_rows
        FROM cfb_game_stats
        GROUP BY game_id
    )
    SELECT
        c.season,
        CASE
            WHEN c.home_div = 'fbs' AND c.away_div = 'fbs' THEN 'fbs_vs_fbs'
            WHEN (c.home_div = 'fbs' AND c.away_div = 'fcs')
              OR (c.home_div = 'fcs' AND c.away_div = 'fbs') THEN 'fbs_vs_fcs'
            ELSE 'other'
        END AS population,
        COUNT(*) AS completed,
        SUM(CASE WHEN coalesce(s.n_rows, 0) >= 2 THEN 1 ELSE 0 END) AS both_teams,
        SUM(CASE WHEN coalesce(s.n_rows, 0) = 1 THEN 1 ELSE 0 END) AS one_team,
        SUM(CASE WHEN coalesce(s.n_rows, 0) = 0 THEN 1 ELSE 0 END) AS none
    FROM completed c
    LEFT JOIN stats_n s ON s.game_id = c.game_id
    GROUP BY c.season, population
    ORDER BY c.season, population
    """
    rows = conn.execute(sql).fetchall()
    by_season: dict[int, dict[str, dict[str, int]]] = {}
    for season, population, completed, both, one, none in rows:
        by_season.setdefault(int(season), {})[str(population)] = {
            "completed": int(completed),
            "both_teams": int(both),
            "one_team": int(one),
            "none": int(none),
        }
    fbs_fbs_below_98: list[dict[str, Any]] = []
    for season, pops in sorted(by_season.items()):
        ff = pops.get("fbs_vs_fbs")
        if not ff or ff["completed"] == 0:
            continue
        pct = 100.0 * ff["both_teams"] / ff["completed"]
        if pct < 98.0:
            fbs_fbs_below_98.append(
                {
                    "season": season,
                    "completed": ff["completed"],
                    "both_teams": ff["both_teams"],
                    "coverage_pct": round(pct, 2),
                    "missing": ff["completed"] - ff["both_teams"],
                }
            )
    return {"by_season": by_season, "fbs_vs_fbs_below_98pct": fbs_fbs_below_98}


def estimated_games_teams_repull_calls(seasons: list[int]) -> int:
    """Calendar + per-week /games/teams for regular and postseason (same as ingest)."""
    # Conservative: ~15 regular weeks + 1 postseason week, plus 2 calendar calls.
    return len(seasons) * (2 + 16)


def upsert_season_flags(conn: sqlite3.Connection) -> None:
    from engine.cfb_schema import ensure_cfb_schema

    ensure_cfb_schema(conn)
    notes = {
        2015: "Source-coverage era 2015-2021: FBS-vs-FBS game_stats are complete; lower-division /games/teams coverage is sparse vs 2022+. Not an exclusion.",
        2016: "Source-coverage era 2015-2021: FBS-vs-FBS game_stats are complete; lower-division /games/teams coverage is sparse vs 2022+. Not an exclusion.",
        2017: "Source-coverage era 2015-2021: FBS-vs-FBS game_stats are complete; lower-division /games/teams coverage is sparse vs 2022+. Not an exclusion.",
        2018: "Source-coverage era 2015-2021: FBS-vs-FBS game_stats are complete; lower-division /games/teams coverage is sparse vs 2022+. Not an exclusion.",
        2019: "Source-coverage era 2015-2021: FBS-vs-FBS game_stats are complete; lower-division /games/teams coverage is sparse vs 2022+. Not an exclusion.",
        2020: (
            "COVID — empty stadiums, opt-outs, truncated schedule; HFA near zero. "
            "Also source-coverage era 2015-2021 (FBS-vs-FBS game_stats complete; lower-division box scores sparse vs 2022+)."
        ),
        2021: "Source-coverage era 2015-2021: FBS-vs-FBS game_stats are complete; lower-division /games/teams coverage is sparse vs 2022+. Not an exclusion.",
        2022: "Source-coverage era 2022-2025: FBS-vs-FBS game_stats are complete; denser lower-division /games/teams coverage. Not an exclusion.",
        2023: "Source-coverage era 2022-2025: FBS-vs-FBS game_stats are complete; denser lower-division /games/teams coverage. Not an exclusion.",
        2024: "Source-coverage era 2022-2025: FBS-vs-FBS game_stats are complete; denser lower-division /games/teams coverage. Not an exclusion.",
        2025: "Source-coverage era 2022-2025: FBS-vs-FBS game_stats are complete; denser lower-division /games/teams coverage. Not an exclusion.",
    }
    for season in range(2015, 2026):
        is_2020 = season == 2020
        conn.execute(
            """
            INSERT OR REPLACE INTO cfb_season_flags
            (season, is_anomalous, exclude_from_hfa, exclude_from_training, note)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                season,
                1 if is_2020 else 0,
                1 if is_2020 else 0,
                1 if is_2020 else 0,
                notes[season],
            ),
        )
    conn.commit()


def lines_provider_report(conn: sqlite3.Connection) -> dict[str, Any]:
    counts = conn.execute(
        """
        SELECT provider, COUNT(*) AS n
        FROM cfb_lines
        GROUP BY provider
        ORDER BY n DESC
        """
    ).fetchall()
    fbs_coverage = conn.execute(
        """
        WITH fbs AS (
            SELECT game_id, season
            FROM cfb_games
            WHERE lower(coalesce(home_division, '')) = 'fbs'
              AND lower(coalesce(away_division, '')) = 'fbs'
        ),
        providers AS (
            SELECT DISTINCT provider FROM cfb_lines
        )
        SELECT
            f.season,
            p.provider,
            COUNT(*) AS fbs_games,
            SUM(CASE WHEN l.game_id IS NOT NULL THEN 1 ELSE 0 END) AS with_line,
            ROUND(100.0 * SUM(CASE WHEN l.game_id IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct
        FROM fbs f
        CROSS JOIN providers p
        LEFT JOIN cfb_lines l ON l.game_id = f.game_id AND l.provider = p.provider
        GROUP BY f.season, p.provider
        ORDER BY f.season, with_line DESC
        """
    ).fetchall()
    opener = conn.execute(
        """
        SELECT
            COUNT(*) AS total_rows,
            SUM(CASE WHEN spread_open IS NOT NULL THEN 1 ELSE 0 END) AS non_null_open,
            SUM(CASE WHEN spread_open IS NOT NULL AND spread IS NOT NULL
                      AND spread_open != spread THEN 1 ELSE 0 END) AS open_distinct_from_spread
        FROM cfb_lines
        """
    ).fetchone()
    opener_by_provider = conn.execute(
        """
        SELECT
            provider,
            COUNT(*) AS total_rows,
            SUM(CASE WHEN spread_open IS NOT NULL THEN 1 ELSE 0 END) AS non_null_open,
            SUM(CASE WHEN spread_open IS NOT NULL AND spread IS NOT NULL
                      AND spread_open != spread THEN 1 ELSE 0 END) AS open_distinct_from_spread
        FROM cfb_lines
        GROUP BY provider
        ORDER BY open_distinct_from_spread DESC
        """
    ).fetchall()
    return {
        "provider_counts": counts,
        "fbs_vs_fbs_coverage": fbs_coverage,
        "opener": {
            "total_rows": opener[0],
            "non_null_open": opener[1],
            "open_distinct_from_spread": opener[2],
        },
        "opener_by_provider": opener_by_provider,
    }


def print_stats_coverage(report: dict[str, Any], shape: dict[str, Any]) -> None:
    print("\n=== 1. FBS/FCS game-stats coverage (completed games only) ===")
    if shape["one_row_per_team_per_game"]:
        print("cfb_game_stats grain: ONE ROW PER TEAM PER GAME (PK game_id, team).")
        print("A completed game has full box-score coverage when BOTH teams have a row.")
    else:
        print("cfb_game_stats grain: unexpected PK", shape["primary_key_columns"])
    print("rows-per-game_id distribution:", shape["rows_per_game_id_distribution"])
    print()
    print(f"{'season':>6} {'pop':<12} {'completed':>9} {'both':>6} {'one':>5} {'none':>6} {'both%':>7}")
    for season, pops in report["by_season"].items():
        for pop in ("fbs_vs_fbs", "fbs_vs_fcs"):
            if pop not in pops:
                continue
            r = pops[pop]
            pct = 100.0 * r["both_teams"] / r["completed"] if r["completed"] else 0.0
            print(
                f"{season:>6} {pop:<12} {r['completed']:>9} {r['both_teams']:>6} "
                f"{r['one_team']:>5} {r['none']:>6} {pct:>6.1f}%"
            )
    below = report["fbs_vs_fbs_below_98pct"]
    if not below:
        print("\nFBS-vs-FBS both-team coverage is >= 98% in every season. No /games/teams re-pull.")
    else:
        print("\nFBS-vs-FBS seasons below 98% both-team coverage:")
        for item in below:
            print(
                f"  {item['season']}: {item['both_teams']}/{item['completed']} "
                f"({item['coverage_pct']}%), missing={item['missing']}"
            )
        seasons = [item["season"] for item in below]
        calls = estimated_games_teams_repull_calls(seasons)
        print(
            f"Targeted re-pull estimate: {len(seasons)} season(s) × "
            f"(2 calendar + ~16 weekly /games/teams) ≈ {calls} CFBD calls."
        )
        print("Not running the re-pull (report-only).")


def print_lines_report(report: dict[str, Any]) -> None:
    print("\n=== 4. cfb_lines providers ===")
    print("Distinct providers (row counts):")
    for provider, n in report["provider_counts"]:
        print(f"  {provider}: {n}")
    print("\nFBS-vs-FBS fraction with a line from each provider, by season:")
    current_season = None
    for season, provider, fbs_games, with_line, pct in report["fbs_vs_fbs_coverage"]:
        if season != current_season:
            current_season = season
            print(f"  {season} (n={fbs_games} FBS-vs-FBS games):")
        print(f"    {provider}: {with_line}/{fbs_games} ({pct}%)")
    opener = report["opener"]
    print("\nOpening-line coverage (all cfb_lines rows):")
    print(f"  total rows: {opener['total_rows']}")
    print(f"  non-null spread_open: {opener['non_null_open']}")
    print(f"  spread_open non-null AND distinct from spread: {opener['open_distinct_from_spread']}")
    print("  by provider (open distinct from spread):")
    for provider, total, non_null, distinct in report["opener_by_provider"]:
        print(f"    {provider}: open={non_null}/{total}, open!=spread={distinct}")
