"""CFB Phase 0 ingestion — backfill games, stats, lines, ratings into espn.db."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

import requests

from engine.cfb_client import CFBDClient
from engine.cfb_config import (
    BACKFILL_END_SEASON,
    BACKFILL_START_SEASON,
    CFB_SPORT_KEY,
    ODDS_API_BASE,
    RATING_SCOPE_END_OF_SEASON,
    SEASON_TYPES,
    get_odds_api_key,
)
from engine.cfb_schema import ensure_cfb_schema


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_bool_int(value: Any) -> int | None:
    if value is None:
        return None
    return 1 if bool(value) else 0


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ppa_side(side: dict[str, Any] | None) -> tuple[float | None, float | None, float | None]:
    if not isinstance(side, dict):
        return None, None, None
    return (
        _safe_float(side.get("overall")),
        _safe_float(side.get("passing")),
        _safe_float(side.get("rushing")),
    )


def _sp_component(block: dict[str, Any] | None) -> float | None:
    if not isinstance(block, dict):
        return None
    return _safe_float(block.get("rating"))


def probe_teams(client: CFBDClient, year: int = 2025) -> list[dict[str, Any]]:
    raw = client.get_json("/teams", {"year": year})
    CFBDClient.print_teams_probe(raw)
    if not isinstance(raw, list):
        raise RuntimeError(f"/teams?year={year} returned non-list payload")
    return raw


def ingest_venues(conn: sqlite3.Connection, client: CFBDClient) -> int:
    rows = client.get_json("/venues")
    if not isinstance(rows, list):
        return 0
    conn.execute("DELETE FROM cfb_venues")
    inserted = 0
    for row in rows:
        conn.execute(
            """
            INSERT OR REPLACE INTO cfb_venues
            (venue_id, name, city, state, elevation, latitude, longitude, capacity, dome, venue_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _safe_int(row.get("id")),
                row.get("name"),
                row.get("city"),
                row.get("state"),
                _safe_float(row.get("elevation")),
                _safe_float(row.get("latitude")),
                _safe_float(row.get("longitude")),
                _safe_int(row.get("capacity")),
                _as_bool_int(row.get("dome")),
                json.dumps(row),
            ),
        )
        inserted += 1
    conn.commit()
    return inserted


def _calendar_weeks(client: CFBDClient, season: int, season_type: str) -> list[int]:
    cal = client.get_json("/calendar", {"year": season})
    if not isinstance(cal, list):
        return []
    weeks = [
        _safe_int(item.get("week"))
        for item in cal
        if str(item.get("seasonType") or item.get("season_type") or "").lower() == season_type
    ]
    weeks = [w for w in weeks if w is not None]
    return sorted(set(weeks))


def ingest_games_for_season(conn: sqlite3.Connection, client: CFBDClient, season: int) -> int:
    total = 0
    for season_type in SEASON_TYPES:
        rows = client.get_json(
            "/games",
            {
                "year": season,
                "seasonType": season_type,
            },
        )
        if not isinstance(rows, list):
            continue
        for row in rows:
            game_id = _safe_int(row.get("id"))
            if game_id is None:
                continue
            conn.execute(
                """
                INSERT OR REPLACE INTO cfb_games
                (game_id, season, week, season_type, start_date, home_team, away_team,
                 home_points, away_points, venue_id, neutral_site, conference_game,
                 home_conference, away_conference, home_division, away_division)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    game_id,
                    season,
                    _safe_int(row.get("week")),
                    str(row.get("seasonType") or row.get("season_type") or season_type),
                    row.get("startDate") or row.get("start_date"),
                    row.get("homeTeam") or row.get("home_team"),
                    row.get("awayTeam") or row.get("away_team"),
                    _safe_int(row.get("homePoints") if "homePoints" in row else row.get("home_points")),
                    _safe_int(row.get("awayPoints") if "awayPoints" in row else row.get("away_points")),
                    _safe_int(row.get("venueId") if "venueId" in row else row.get("venue_id")),
                    _as_bool_int(row.get("neutralSite") if "neutralSite" in row else row.get("neutral_site")),
                    _as_bool_int(row.get("conferenceGame") if "conferenceGame" in row else row.get("conference_game")),
                    row.get("homeConference") or row.get("home_conference"),
                    row.get("awayConference") or row.get("away_conference"),
                    row.get("homeClassification") or row.get("home_classification"),
                    row.get("awayClassification") or row.get("away_classification"),
                ),
            )
            total += 1
    conn.commit()
    return total


def ingest_game_stats_for_season(conn: sqlite3.Connection, client: CFBDClient, season: int) -> int:
    total = 0
    for season_type in SEASON_TYPES:
        weeks = _calendar_weeks(client, season, season_type)
        if not weeks:
            weeks = [None]
        for week in weeks:
            params: dict[str, Any] = {"year": season, "seasonType": season_type}
            if week is not None:
                params["week"] = week
            rows = client.get_json("/games/teams", params)
            if not isinstance(rows, list):
                continue
            for block in rows:
                game_id = _safe_int(block.get("id"))
                if game_id is None:
                    continue
                teams = block.get("teams") or []
                for team_row in teams:
                    team_name = team_row.get("team")
                    if not team_name:
                        continue
                    home_away = str(team_row.get("homeAway") or team_row.get("home_away") or "").lower()
                    is_home = 1 if home_away == "home" else 0
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO cfb_game_stats
                        (game_id, team, is_home, points, stats_json)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            game_id,
                            team_name,
                            is_home,
                            _safe_int(team_row.get("points")),
                            json.dumps(team_row.get("stats") or []),
                        ),
                    )
                    total += 1
    conn.commit()
    return total


def ingest_lines_for_season(conn: sqlite3.Connection, client: CFBDClient, season: int) -> int:
    captured_at = _utc_now_iso()
    total = 0
    for season_type in SEASON_TYPES:
        weeks = _calendar_weeks(client, season, season_type)
        if not weeks:
            weeks = [None]
        for week in weeks:
            params: dict[str, Any] = {"year": season, "seasonType": season_type}
            if week is not None:
                params["week"] = week
            rows = client.get_json("/lines", params)
            if not isinstance(rows, list):
                continue
            for game in rows:
                game_id = _safe_int(game.get("id"))
                if game_id is None:
                    continue
                for line in game.get("lines") or []:
                    provider = line.get("provider")
                    if not provider:
                        continue
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO cfb_lines
                        (game_id, provider, spread, spread_open, over_under,
                         home_moneyline, away_moneyline, captured_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            game_id,
                            provider,
                            _safe_float(line.get("spread")),
                            _safe_float(line.get("spreadOpen") if "spreadOpen" in line else line.get("spread_open")),
                            _safe_float(line.get("overUnder") if "overUnder" in line else line.get("over_under")),
                            _safe_float(line.get("homeMoneyline") if "homeMoneyline" in line else line.get("home_moneyline")),
                            _safe_float(line.get("awayMoneyline") if "awayMoneyline" in line else line.get("away_moneyline")),
                            captured_at,
                        ),
                    )
                    total += 1
    conn.commit()
    return total


def ingest_team_stats_adv_for_season(conn: sqlite3.Connection, client: CFBDClient, season: int) -> int:
    rows = client.get_json(
        "/stats/season/advanced",
        {"year": season, "excludeGarbageTime": "true"},
    )
    if not isinstance(rows, list):
        return 0
    total = 0
    for row in rows:
        team = row.get("team")
        if not team:
            continue
        offense = row.get("offense")
        defense = row.get("defense")
        conn.execute(
            """
            INSERT OR REPLACE INTO cfb_team_stats_adv
            (season, team, conference, offense_json, defense_json, stats_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                season,
                team,
                row.get("conference"),
                json.dumps(offense) if offense is not None else None,
                json.dumps(defense) if defense is not None else None,
                json.dumps(row),
            ),
        )
        total += 1
    conn.commit()
    return total


def ingest_ppa_for_season(conn: sqlite3.Connection, client: CFBDClient, season: int) -> int:
    rows = client.get_json(
        "/ppa/teams",
        {"year": season, "excludeGarbageTime": "true"},
    )
    if not isinstance(rows, list):
        return 0
    total = 0
    for row in rows:
        team = row.get("team")
        if not team:
            continue
        off_overall, off_pass, off_rush = _ppa_side(row.get("offense"))
        def_overall, def_pass, def_rush = _ppa_side(row.get("defense"))
        conn.execute(
            """
            INSERT OR REPLACE INTO cfb_ppa
            (season, team, conference, off_ppa_overall, off_ppa_pass, off_ppa_rush,
             def_ppa_overall, def_ppa_pass, def_ppa_rush, ppa_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                season,
                team,
                row.get("conference"),
                off_overall,
                off_pass,
                off_rush,
                def_overall,
                def_pass,
                def_rush,
                json.dumps(row),
            ),
        )
        total += 1
    conn.commit()
    return total


def ingest_sp_ratings_for_season(
    conn: sqlite3.Connection,
    client: CFBDClient,
    season: int,
    *,
    rating_scope: str = RATING_SCOPE_END_OF_SEASON,
) -> int:
    rows = client.get_json("/ratings/sp", {"year": season})
    if not isinstance(rows, list):
        return 0
    total = 0
    for row in rows:
        team = row.get("team")
        if not team:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO cfb_ratings_sp
            (season, team, conference, rating, offense, defense, special, rating_scope, sp_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                season,
                team,
                row.get("conference"),
                _safe_float(row.get("rating")),
                _sp_component(row.get("offense")),
                _sp_component(row.get("defense")),
                _sp_component(row.get("specialTeams") if "specialTeams" in row else row.get("special_teams")),
                rating_scope,
                json.dumps(row),
            ),
        )
        total += 1
    conn.commit()
    return total


def ingest_returning_for_season(conn: sqlite3.Connection, client: CFBDClient, season: int) -> int:
    rows = client.get_json("/player/returning", {"year": season})
    if not isinstance(rows, list):
        return 0
    total = 0
    for row in rows:
        team = row.get("team")
        if not team:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO cfb_returning
            (season, team, conference, total_ppa, total_passing_ppa, total_receiving_ppa,
             total_rushing_ppa, percent_ppa, percent_passing_ppa, percent_receiving_ppa,
             percent_rushing_ppa, usage, passing_usage, receiving_usage, rushing_usage, returning_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                season,
                team,
                row.get("conference"),
                _safe_float(row.get("totalPPA") if "totalPPA" in row else row.get("total_ppa")),
                _safe_float(row.get("totalPassingPPA") if "totalPassingPPA" in row else row.get("total_passing_ppa")),
                _safe_float(row.get("totalReceivingPPA") if "totalReceivingPPA" in row else row.get("total_receiving_ppa")),
                _safe_float(row.get("totalRushingPPA") if "totalRushingPPA" in row else row.get("total_rushing_ppa")),
                _safe_float(row.get("percentPPA") if "percentPPA" in row else row.get("percent_ppa")),
                _safe_float(row.get("percentPassingPPA") if "percentPassingPPA" in row else row.get("percent_passing_ppa")),
                _safe_float(row.get("percentReceivingPPA") if "percentReceivingPPA" in row else row.get("percent_receiving_ppa")),
                _safe_float(row.get("percentRushingPPA") if "percentRushingPPA" in row else row.get("percent_rushing_ppa")),
                _safe_float(row.get("usage")),
                _safe_float(row.get("passingUsage") if "passingUsage" in row else row.get("passing_usage")),
                _safe_float(row.get("receivingUsage") if "receivingUsage" in row else row.get("receiving_usage")),
                _safe_float(row.get("rushingUsage") if "rushingUsage" in row else row.get("rushing_usage")),
                json.dumps(row),
            ),
        )
        total += 1
    conn.commit()
    return total


def fetch_odds_api_team_names() -> list[str]:
    api_key = get_odds_api_key()
    if not api_key:
        raise RuntimeError("Odds API key missing for alias build")
    url = f"{ODDS_API_BASE}/sports/{CFB_SPORT_KEY}/odds"
    response = requests.get(
        url,
        params={
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    names: set[str] = set()
    if isinstance(payload, list):
        for event in payload:
            if event.get("home_team"):
                names.add(str(event["home_team"]).strip())
            if event.get("away_team"):
                names.add(str(event["away_team"]).strip())
    return sorted(names)


def build_team_aliases(conn: sqlite3.Connection, client: CFBDClient) -> dict[str, list[str]]:
    teams_2026 = client.get_json("/teams", {"year": 2026})
    if not isinstance(teams_2026, list):
        raise RuntimeError("/teams?year=2026 returned non-list payload")

    fbs_rows = [
        row for row in teams_2026
        if str(row.get("classification") or "").lower() == "fbs"
    ]
    cfbd_names = sorted({str(row.get("school") or "").strip() for row in fbs_rows if row.get("school")})
    odds_names = fetch_odds_api_team_names()

    exact_matches = sorted(set(cfbd_names) & set(odds_names))
    unmatched_cfbd = sorted(set(cfbd_names) - set(odds_names))
    unmatched_odds = sorted(set(odds_names) - set(cfbd_names))

    conn.execute("DELETE FROM cfb_team_alias")
    conf_by_school = {
        str(row.get("school") or "").strip(): row.get("conference")
        for row in fbs_rows
        if row.get("school")
    }
    for name in cfbd_names:
        odds_name = name if name in exact_matches else None
        conn.execute(
            """
            INSERT OR REPLACE INTO cfb_team_alias
            (canonical_name, cfbd_name, odds_api_name, espn_name, conference_2026)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name, name, odds_name, None, conf_by_school.get(name)),
        )
    conn.commit()
    return {
        "exact_matches": exact_matches,
        "unmatched_cfbd": unmatched_cfbd,
        "unmatched_odds": unmatched_odds,
    }


def backfill_all(
    conn: sqlite3.Connection,
    client: CFBDClient,
    *,
    start_season: int = BACKFILL_START_SEASON,
    end_season: int = BACKFILL_END_SEASON,
) -> dict[str, Any]:
    ensure_cfb_schema(conn)
    summary: dict[str, Any] = {"seasons": {}}

    venue_count = ingest_venues(conn, client)
    summary["venues"] = venue_count

    for season in range(start_season, end_season + 1):
        print(f"\n=== CFB backfill season {season} ===", flush=True)
        season_summary = {
            "games": ingest_games_for_season(conn, client, season),
            "game_stats": ingest_game_stats_for_season(conn, client, season),
            "lines": ingest_lines_for_season(conn, client, season),
            "team_stats_adv": ingest_team_stats_adv_for_season(conn, client, season),
            "ppa": ingest_ppa_for_season(conn, client, season),
            "ratings_sp": ingest_sp_ratings_for_season(conn, client, season),
            "returning": ingest_returning_for_season(conn, client, season),
        }
        summary["seasons"][season] = season_summary
        print(json.dumps(season_summary), flush=True)

    alias_summary = build_team_aliases(conn, client)
    summary["aliases"] = alias_summary
    summary["cfbd_calls"] = client.call_count
    return summary
