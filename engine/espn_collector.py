"""
ESPN unofficial API data collector (no API key).
Pulls completed game scores, team season stats, and game-by-game schedules
for NFL, NBA, MLB, NHL going back 3 seasons. Stores in SQLite with 1s delay between requests.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
REQUEST_DELAY_SECONDS = 1.0

# (sport path, league path) for URL: {sport}/{league}/...
LEAGUES = {
    "nfl": ("football", "nfl"),
    "nba": ("basketball", "nba"),
    "mlb": ("baseball", "mlb"),
    "nhl": ("hockey", "nhl"),
}

# Seasons to fetch (e.g. 3 back from current year)
def _default_seasons() -> list[int]:
    y = datetime.now(timezone.utc).year
    return [y - 1, y - 2, y - 3]


def _data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def _db_path() -> Path:
    return _data_dir() / "espn.db"


def _get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | list[Any]:
    """GET with error handling and 1s delay. Returns parsed JSON or empty dict/list on failure."""
    time.sleep(REQUEST_DELAY_SECONDS)
    url = path if path.startswith("http") else f"{ESPN_BASE}/{path}"
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError):
        return {}


def _sport_league_path(league_key: str) -> str:
    sport, league = LEAGUES[league_key]
    return f"{sport}/{league}"


def fetch_teams(league_key: str) -> list[dict[str, Any]]:
    """Fetch all teams for a league. Returns list of team dicts."""
    path = f"{_sport_league_path(league_key)}/teams"
    data = _get(path)
    if not data or "sports" not in data:
        return []
    try:
        teams = data["sports"][0]["leagues"][0].get("teams", [])
        return [t.get("team", t) for t in teams]
    except (IndexError, KeyError, TypeError):
        return []


def fetch_team_schedule(league_key: str, team_id: str, season: int) -> list[dict[str, Any]]:
    """Fetch one team's schedule for a season. Returns list of event dicts."""
    path = f"{_sport_league_path(league_key)}/teams/{team_id}/schedule"
    data = _get(path, params={"season": season})
    if not data or "events" not in data:
        return []
    return data.get("events", [])


def fetch_team_detail(league_key: str, team_id: str) -> dict[str, Any]:
    """Fetch one team's detail (includes record, etc.)."""
    path = f"{_sport_league_path(league_key)}/teams/{team_id}"
    data = _get(path)
    if not data and "team" not in data:
        return {}
    return data.get("team", data)


def _parse_event_to_game_row(event: dict[str, Any], league_key: str) -> dict[str, Any] | None:
    """Parse one event into a flat game row (completed games with scores)."""
    try:
        event_id = event.get("id", "")
        date_str = event.get("date", "")
        name = event.get("name", "")
        status = event.get("status", {}) or {}
        status_type = status.get("type", {}) or {}
        completed = (
            status_type.get("completed", False)
            or (status_type.get("name", "") or "").lower() in ("final", "final/ot", "final/2ot")
        )
        competitions = event.get("competitions", [])
        if not competitions:
            return None
        comp = competitions[0]
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            return None
        home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
        home_team = home.get("team", {}).get("displayName", "")
        away_team = away.get("team", {}).get("displayName", "")
        home_id = home.get("team", {}).get("id", "")
        away_id = away.get("team", {}).get("id", "")
        score_home = home.get("score")
        score_away = away.get("score")
        if isinstance(score_home, dict):
            score_home = score_home.get("value") or score_home.get("displayValue")
        if isinstance(score_away, dict):
            score_away = score_away.get("value") or score_away.get("displayValue")
        try:
            score_home = float(score_home) if score_home not in (None, "") else None
        except (TypeError, ValueError):
            score_home = None
        try:
            score_away = float(score_away) if score_away not in (None, "") else None
        except (TypeError, ValueError):
            score_away = None
        # Consider completed if status says so or both scores are present
        if not completed and (score_home is None or score_away is None):
            return None
        return {
            "league": league_key,
            "game_id": event_id,
            "game_date": date_str,
            "game_name": name,
            "home_team_id": home_id,
            "home_team_name": home_team,
            "away_team_id": away_id,
            "away_team_name": away_team,
            "home_score": score_home,
            "away_score": score_away,
        }
    except (KeyError, TypeError, IndexError):
        return None


def _wins_losses_from_events(events: list[dict], team_id: str, league_key: str) -> tuple[int, int]:
    """Compute wins and losses for a team from a list of event dicts (completed games)."""
    wins = losses = 0
    for ev in events:
        row = _parse_event_to_game_row(ev, league_key)
        if not row:
            continue
        comps = ev.get("competitions", [{}])[0].get("competitors", [])
        for c in comps:
            if str(c.get("team", {}).get("id")) == str(team_id):
                s = c.get("score")
                if isinstance(s, dict):
                    s = s.get("value") or s.get("displayValue")
                try:
                    my_score = float(s)
                except (TypeError, ValueError):
                    my_score = 0
                other = [x for x in comps if str(x.get("team", {}).get("id")) != str(team_id)]
                other_score = 0
                if other:
                    o = other[0].get("score")
                    if isinstance(o, dict):
                        o = o.get("value") or o.get("displayValue")
                    try:
                        other_score = float(o)
                    except (TypeError, ValueError):
                        pass
                if my_score > other_score:
                    wins += 1
                elif my_score < other_score:
                    losses += 1
                break
    return wins, losses


def collect_games_and_schedules(
    league_key: str,
    seasons: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collect completed games and game-by-game schedule for a league over the given seasons.
    Uses team schedules and deduplicates by game_id.
    Returns (games_df, schedule_df). Schedule has same rows as games.
    """
    seasons = seasons or _default_seasons()
    teams = fetch_teams(league_key)
    empty = pd.DataFrame(columns=["league", "game_id", "game_date", "game_name", "home_team_id", "home_team_name", "away_team_id", "away_team_name", "home_score", "away_score"])
    if not teams:
        return empty.copy(), empty.copy()
    seen_events: dict[str, dict] = {}
    for team in teams:
        team_id = team.get("id", "")
        if not team_id:
            continue
        for season in seasons:
            events = fetch_team_schedule(league_key, team_id, season)
            for ev in events:
                eid = ev.get("id", "")
                if not eid or eid in seen_events:
                    continue
                row = _parse_event_to_game_row(ev, league_key)
                if row:
                    seen_events[eid] = row
    if not seen_events:
        return empty.copy(), empty.copy()
    games_df = pd.DataFrame(list(seen_events.values()))
    return games_df, games_df.copy()


def collect_team_season_stats(
    league_key: str,
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Collect team season stats (wins/losses from completed games) for a league over the given seasons."""
    seasons = seasons or _default_seasons()
    teams = fetch_teams(league_key)
    if not teams:
        return pd.DataFrame(columns=["league", "season", "team_id", "team_name", "wins", "losses", "record_summary"])
    rows = []
    for team in teams:
        team_id = team.get("id", "")
        if not team_id:
            continue
        team_name = team.get("displayName", team.get("name", ""))
        for season in seasons:
            events = fetch_team_schedule(league_key, team_id, season)
            wins, losses = _wins_losses_from_events(events, team_id, league_key)
            rows.append({
                "league": league_key,
                "season": season,
                "team_id": team_id,
                "team_name": team_name,
                "wins": wins,
                "losses": losses,
                "record_summary": f"{wins}-{losses}",
            })
    return pd.DataFrame(rows)


def save_games_to_sqlite(df: pd.DataFrame, league_key: str, db_path: Path | None = None) -> None:
    """Replace games for this league in SQLite."""
    if df.empty:
        return
    path = db_path or _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS games (
                league TEXT NOT NULL,
                game_id TEXT NOT NULL,
                game_date TEXT,
                game_name TEXT,
                home_team_id TEXT,
                home_team_name TEXT,
                away_team_id TEXT,
                away_team_name TEXT,
                home_score REAL,
                away_score REAL,
                PRIMARY KEY (league, game_id)
            )
        """)
        conn.execute("DELETE FROM games WHERE league = ?", (league_key,))
        df.to_sql("games", conn, if_exists="append", index=False)
        conn.commit()
    finally:
        conn.close()


def save_schedules_to_sqlite(df: pd.DataFrame, league_key: str, db_path: Path | None = None) -> None:
    """Replace schedule rows for this league (same schema as games)."""
    if df.empty:
        return
    path = db_path or _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schedules (
                league TEXT NOT NULL,
                game_id TEXT NOT NULL,
                game_date TEXT,
                game_name TEXT,
                home_team_id TEXT,
                home_team_name TEXT,
                away_team_id TEXT,
                away_team_name TEXT,
                home_score REAL,
                away_score REAL,
                PRIMARY KEY (league, game_id)
            )
        """)
        conn.execute("DELETE FROM schedules WHERE league = ?", (league_key,))
        df.to_sql("schedules", conn, if_exists="append", index=False)
        conn.commit()
    finally:
        conn.close()


def save_team_stats_to_sqlite(df: pd.DataFrame, league_key: str, db_path: Path | None = None) -> None:
    """Replace team season stats for this league."""
    if df.empty:
        return
    path = db_path or _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS team_season_stats (
                league TEXT NOT NULL,
                season INTEGER NOT NULL,
                team_id TEXT NOT NULL,
                team_name TEXT,
                wins INTEGER,
                losses INTEGER,
                record_summary TEXT,
                PRIMARY KEY (league, season, team_id)
            )
        """)
        conn.execute("DELETE FROM team_season_stats WHERE league = ?", (league_key,))
        df.to_sql("team_season_stats", conn, if_exists="append", index=False)
        conn.commit()
    finally:
        conn.close()


def collect_and_store_all(
    leagues: list[str] | None = None,
    seasons: list[int] | None = None,
    db_path: Path | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Collect games, schedules, and team season stats for each league (default: nfl, nba, mlb, nhl)
    for the last 3 seasons. Save to SQLite. Returns dict[league_key][table_name] -> DataFrame.
    """
    leagues = leagues or list(LEAGUES.keys())
    seasons = seasons or _default_seasons()
    db_path = db_path or _db_path()
    result: dict[str, dict[str, pd.DataFrame]] = {}
    for league_key in leagues:
        if league_key not in LEAGUES:
            continue
        result[league_key] = {}
        games_df, schedule_df = collect_games_and_schedules(league_key, seasons)
        save_games_to_sqlite(games_df, league_key, db_path)
        save_schedules_to_sqlite(schedule_df, league_key, db_path)
        result[league_key]["games"] = games_df
        result[league_key]["schedules"] = schedule_df
        stats_df = collect_team_season_stats(league_key, seasons)
        save_team_stats_to_sqlite(stats_df, league_key, db_path)
        result[league_key]["team_season_stats"] = stats_df
    return result


def load_games_from_sqlite(league_key: str | None = None, db_path: Path | None = None) -> pd.DataFrame:
    """Load games from SQLite."""
    path = db_path or _db_path()
    if not path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(path)
    try:
        if league_key:
            return pd.read_sql_query("SELECT * FROM games WHERE league = ?", conn, params=(league_key,))
        return pd.read_sql_query("SELECT * FROM games", conn)
    finally:
        conn.close()


def load_schedules_from_sqlite(league_key: str | None = None, db_path: Path | None = None) -> pd.DataFrame:
    """Load schedules from SQLite."""
    path = db_path or _db_path()
    if not path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(path)
    try:
        if league_key:
            return pd.read_sql_query("SELECT * FROM schedules WHERE league = ?", conn, params=(league_key,))
        return pd.read_sql_query("SELECT * FROM schedules", conn)
    finally:
        conn.close()


def load_team_stats_from_sqlite(league_key: str | None = None, db_path: Path | None = None) -> pd.DataFrame:
    """Load team season stats from SQLite."""
    path = db_path or _db_path()
    if not path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(path)
    try:
        if league_key:
            return pd.read_sql_query("SELECT * FROM team_season_stats WHERE league = ?", conn, params=(league_key,))
        return pd.read_sql_query("SELECT * FROM team_season_stats", conn)
    finally:
        conn.close()
