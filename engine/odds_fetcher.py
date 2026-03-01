"""
Fetch NBA and NCAAB odds from The Odds API (free tier).
Caches responses 15 minutes per sport, stores raw JSON, parses to DataFrame, saves to SQLite.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from engine.engine import (
    ODDS_API_BASE,
    BASKETBALL_NBA,
    BASKETBALL_NCAAB,
    REQUEST_HEADERS,
)
from engine.odds_quota import odds_api_get

# Cache TTL: 15 minutes per sport (free tier quota)
CACHE_TTL_SECONDS = 900
# Markets to fetch
MARKETS = ["h2h", "spreads", "totals"]
ODDS_FORMAT = "american"
REGIONS = "us"

# Default paths (project data dir)
def _data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def _cache_dir() -> Path:
    d = _data_dir() / "odds_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _db_path() -> Path:
    return _data_dir() / "odds.db"


def _cache_path(sport_key: str) -> Path:
    return _cache_dir() / f"{sport_key}.json"


def _fetch_raw(api_key: str, sport_key: str) -> list[dict[str, Any]]:
    """Call The Odds API for one sport. Returns list of events (raw). Skips if quota remaining < 10."""
    url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
    params = {
        "regions": REGIONS,
        "markets": ",".join(MARKETS),
        "oddsFormat": ODDS_FORMAT,
        "apiKey": api_key.strip(),
    }
    resp = odds_api_get(url, params=params, headers=REQUEST_HEADERS, timeout=15)
    if resp is None:
        return []
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else []


def _load_cached(sport_key: str) -> tuple[list[dict[str, Any]] | None, datetime | None]:
    """Load cached response for sport. Returns (data, fetched_at) or (None, None) if miss/expired."""
    path = _cache_path(sport_key)
    if not path.exists():
        return None, None
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        fetched_at = datetime.fromisoformat(obj["fetched_at"].replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        if (now - fetched_at).total_seconds() > CACHE_TTL_SECONDS:
            return None, None
        return obj["data"], fetched_at
    except (json.JSONDecodeError, KeyError, OSError):
        return None, None


def _save_cache(sport_key: str, data: list[dict[str, Any]]) -> None:
    """Write raw response to cache file."""
    path = _cache_path(sport_key)
    now = datetime.now(timezone.utc).isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"fetched_at": now, "data": data}, f, indent=None)


def fetch_odds(
    api_key: str,
    sport_key: str,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """
    Fetch odds for one sport (basketball_nba or basketball_ncaab).
    If use_cache is True and cache is younger than 15 minutes, return cached data.
    Otherwise hit the API, update cache, and return raw response.
    """
    if use_cache:
        cached, _ = _load_cached(sport_key)
        if cached is not None:
            return cached
    data = _fetch_raw(api_key, sport_key)
    if use_cache:
        _save_cache(sport_key, data)
    return data


def parse_raw_to_dataframe(raw: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Parse raw Odds API response into a DataFrame with columns:
    game_id, home_team, away_team, commence_time, bookmaker, market_type, outcome, price, point
    """
    rows: list[dict[str, Any]] = []
    for event in raw:
        if not isinstance(event, dict):
            continue
        game_id = event.get("id", "")
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        commence_time = event.get("commence_time", "")
        for bm in event.get("bookmakers") or []:
            bookmaker = bm.get("key") or bm.get("title") or ""
            for mkt in bm.get("markets") or []:
                mkt_key = mkt.get("key", "")
                if mkt_key not in ("h2h", "spreads", "totals"):
                    continue
                market_type = mkt_key
                for out in mkt.get("outcomes") or []:
                    outcome = out.get("name", "")
                    price = out.get("price")
                    if price is None:
                        continue
                    try:
                        price = float(price)
                    except (TypeError, ValueError):
                        continue
                    point = out.get("point")
                    if point is not None:
                        try:
                            point = float(point)
                        except (TypeError, ValueError):
                            point = None
                    rows.append({
                        "game_id": game_id,
                        "home_team": home_team,
                        "away_team": away_team,
                        "commence_time": commence_time,
                        "bookmaker": bookmaker,
                        "market_type": market_type,
                        "outcome": outcome,
                        "price": price,
                        "point": point,
                    })
    return pd.DataFrame(rows)


def save_raw_to_sqlite(sport_key: str, raw: list[dict[str, Any]], db_path: Path | None = None) -> None:
    """Store raw API response in SQLite table raw_responses."""
    path = db_path or _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS raw_responses (
                sport_key TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                response_json TEXT NOT NULL,
                PRIMARY KEY (sport_key)
            )
        """)
        conn.execute(
            "INSERT OR REPLACE INTO raw_responses (sport_key, fetched_at, response_json) VALUES (?, ?, ?)",
            (sport_key, datetime.now(timezone.utc).isoformat(), json.dumps(raw)),
        )
        conn.commit()
    finally:
        conn.close()


def save_dataframe_to_sqlite(df: pd.DataFrame, sport_key: str, db_path: Path | None = None) -> None:
    """Append parsed odds to SQLite table odds (replace data for this sport)."""
    if df.empty:
        return
    path = db_path or _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS odds (
                sport_key TEXT NOT NULL,
                game_id TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                commence_time TEXT NOT NULL,
                bookmaker TEXT NOT NULL,
                market_type TEXT NOT NULL,
                outcome TEXT NOT NULL,
                price REAL NOT NULL,
                point REAL,
                PRIMARY KEY (sport_key, game_id, bookmaker, market_type, outcome, point)
            )
        """)
        conn.execute("DELETE FROM odds WHERE sport_key = ?", (sport_key,))
        df["sport_key"] = sport_key
        df.to_sql("odds", conn, if_exists="append", index=False)
        conn.commit()
    finally:
        conn.close()


def fetch_and_store(
    api_key: str,
    sport_keys: list[str] | None = None,
    use_cache: bool = True,
    db_path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    For each sport (default: NBA and NCAAB), fetch odds (with 15-min cache),
    store raw response in SQLite and in cache, parse to DataFrame, save to SQLite.
    Returns dict mapping sport_key -> parsed DataFrame.
    """
    sport_keys = sport_keys or [BASKETBALL_NBA, BASKETBALL_NCAAB]
    db_path = db_path or _db_path()
    result: dict[str, pd.DataFrame] = {}
    for sport_key in sport_keys:
        raw = fetch_odds(api_key, sport_key, use_cache=use_cache)
        save_raw_to_sqlite(sport_key, raw, db_path=db_path)
        df = parse_raw_to_dataframe(raw)
        save_dataframe_to_sqlite(df, sport_key, db_path=db_path)
        result[sport_key] = df
    return result


def load_odds_from_sqlite(
    sport_key: str | None = None,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Load parsed odds from SQLite. If sport_key is None, load all sports."""
    path = db_path or _db_path()
    if not path.exists():
        return pd.DataFrame(columns=["game_id", "home_team", "away_team", "commence_time", "bookmaker", "market_type", "outcome", "price", "point"])
    conn = sqlite3.connect(path)
    try:
        if sport_key:
            df = pd.read_sql_query("SELECT game_id, home_team, away_team, commence_time, bookmaker, market_type, outcome, price, point FROM odds WHERE sport_key = ?", conn, params=(sport_key,))
        else:
            df = pd.read_sql_query("SELECT game_id, home_team, away_team, commence_time, bookmaker, market_type, outcome, price, point FROM odds", conn)
        return df
    finally:
        conn.close()
