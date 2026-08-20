#!/usr/bin/env python3
"""Ingest CFBD /stats/game/advanced for 2021-2025 into cfb_game_stats_adv."""

from __future__ import annotations

import json
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


def _estimate_calls(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT season, season_type, week
            FROM cfb_games
            WHERE season BETWEEN 2021 AND 2025 AND week IS NOT NULL
            GROUP BY season, season_type, week
        )
        """
    ).fetchone()[0]
    return int(rows)


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ingest_game_advanced(
    conn: sqlite3.Connection,
    client: CFBDClient,
    *,
    seasons: tuple[int, ...] = SEASONS,
) -> int:
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
    home_map = {
        (gid, team): is_home
        for gid, team, is_home in conn.execute(
            "SELECT game_id, team, is_home FROM cfb_game_stats"
        ).fetchall()
    }
    total = 0
    for season, season_type, week in groups:
        if season not in seasons:
            continue
        rows = client.get_json(
            "/stats/game/advanced",
            {"year": season, "week": week, "seasonType": season_type},
        )
        if not isinstance(rows, list):
            continue
        for row in rows:
            game_id = row.get("gameId")
            team = row.get("team")
            if game_id is None or not team:
                continue
            is_home = home_map.get((game_id, team))
            if is_home is None:
                g = conn.execute(
                    "SELECT home_team, away_team FROM cfb_games WHERE game_id = ?",
                    (game_id,),
                ).fetchone()
                if not g:
                    continue
                is_home = 1 if team == g[0] else 0 if team == g[1] else 0
            offense = row.get("offense") or {}
            defense = row.get("defense") or {}
            conn.execute(
                """
                INSERT OR REPLACE INTO cfb_game_stats_adv (
                    game_id, team, is_home,
                    off_ppa, off_success_rate, off_explosiveness, off_line_yards,
                    def_ppa, def_success_rate, def_explosiveness, def_line_yards,
                    stats_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    game_id,
                    team,
                    is_home,
                    _safe_float(offense.get("ppa")),
                    _safe_float(offense.get("successRate")),
                    _safe_float(offense.get("explosiveness")),
                    _safe_float(offense.get("lineYards")),
                    _safe_float(defense.get("ppa")),
                    _safe_float(defense.get("successRate")),
                    _safe_float(defense.get("explosiveness")),
                    _safe_float(defense.get("lineYards")),
                    json.dumps(row),
                ),
            )
            total += 1
    conn.commit()
    return total


def main() -> None:
    conn = sqlite3.connect(ESPN_DB_PATH)
    ensure_cfb_schema(conn)
    est = _estimate_calls(conn)
    print(f"Estimated CFBD calls for /stats/game/advanced 2021-2025: {est}")
    print(f"Reserve held back for in-season pulls: {RESERVE_CALLS}")
    client = CFBDClient()
    # Probe budget (1 call consumed by client on first real pull)
    if est + RESERVE_CALLS > 525:
        print(
            f"ABORT: estimated {est} + reserve {RESERVE_CALLS} exceeds conservative budget."
        )
        conn.close()
        sys.exit(1)
    print(f"Proceeding — budget OK (remaining before run ~525, need {est}).")
    rows = ingest_game_advanced(conn, client, seasons=SEASONS)
    print(f"Ingested {rows} team-game advanced stat rows.")
    print(f"CFBD calls consumed this run: {client.call_count}")
    conn.close()


if __name__ == "__main__":
    main()
