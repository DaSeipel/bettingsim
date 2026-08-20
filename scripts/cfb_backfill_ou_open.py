#!/usr/bin/env python3
"""Backfill cfb_lines.over_under_open from CFBD /lines for 2021-2025."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.cfb_client import CFBDClient
from engine.cfb_config import ESPN_DB_PATH
from engine.cfb_schema import ensure_cfb_schema

RESERVE_CALLS = 150
SEASONS = (2021, 2022, 2023, 2024, 2025)


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def estimate_calls(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT season, season_type, week
            FROM cfb_games
            WHERE season BETWEEN 2021 AND 2025 AND week IS NOT NULL
            GROUP BY season, season_type, week
        )
        """
    ).fetchone()
    return int(row[0])


def backfill_over_under_open(
    conn: sqlite3.Connection,
    client: CFBDClient,
    *,
    seasons: tuple[int, ...] = SEASONS,
) -> tuple[int, int]:
    groups = conn.execute(
        """
        SELECT season, season_type, week
        FROM cfb_games
        WHERE season BETWEEN ? AND ? AND week IS NOT NULL
        GROUP BY season, season_type, week
        ORDER BY season, season_type, week
        """,
        (min(seasons), max(seasons)),
    ).fetchall()
    updated = 0
    seen = 0
    for season, season_type, week in groups:
        if season not in seasons:
            continue
        rows = client.get_json(
            "/lines",
            {"year": season, "week": week, "seasonType": season_type},
        )
        if not isinstance(rows, list):
            continue
        for game in rows:
            game_id = game.get("id")
            if game_id is None:
                continue
            for line in game.get("lines") or []:
                provider = line.get("provider")
                if not provider:
                    continue
                ou_open = _safe_float(
                    line.get("overUnderOpen")
                    if "overUnderOpen" in line
                    else line.get("over_under_open")
                )
                if ou_open is None:
                    continue
                seen += 1
                cur = conn.execute(
                    """
                    UPDATE cfb_lines
                    SET over_under_open = ?
                    WHERE game_id = ? AND provider = ?
                    """,
                    (ou_open, game_id, provider),
                )
                updated += cur.rowcount
    conn.commit()
    return updated, seen


def main() -> None:
    conn = sqlite3.connect(ESPN_DB_PATH)
    ensure_cfb_schema(conn)
    est = estimate_calls(conn)
    print(f"Estimated CFBD calls for /lines backfill 2021-2025: {est}")
    print(f"Reserve required for in-season use: {RESERVE_CALLS}")
    print(f"Calls after backfill (est.): current_remaining - {est} >= {RESERVE_CALLS}")
    if est + RESERVE_CALLS > 525:
        print(
            f"ABORT: estimated {est} + reserve {RESERVE_CALLS} exceeds conservative budget."
        )
        conn.close()
        sys.exit(1)
    print("Proceeding — budget OK.")
    client = CFBDClient()
    updated, seen = backfill_over_under_open(conn, client)
    print(f"Non-null overUnderOpen values seen in API: {seen}")
    print(f"cfb_lines rows updated: {updated}")
    print(f"CFBD calls consumed this run: {client.call_count}")
    cov = conn.execute(
        """
        SELECT COUNT(*)
        FROM cfb_lines l
        JOIN cfb_games g ON g.game_id = l.game_id
        WHERE l.is_backtest_reference = 1
          AND g.season BETWEEN 2021 AND 2025
          AND l.over_under_open IS NOT NULL
        """
    ).fetchone()[0]
    ref = conn.execute(
        """
        SELECT COUNT(*)
        FROM cfb_lines l
        JOIN cfb_games g ON g.game_id = l.game_id
        WHERE l.is_backtest_reference = 1
          AND g.season BETWEEN 2021 AND 2025
        """
    ).fetchone()[0]
    print(f"Reference lines with over_under_open (2021-2025): {cov}/{ref}")
    conn.close()


if __name__ == "__main__":
    main()
