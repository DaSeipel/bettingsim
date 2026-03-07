"""
Historical odds from The Odds API (v4 historical endpoint).
Fetches one week at a time, caches each response to disk, stores closing lines in data/odds.db (historical_odds table).
Used to join closing spread/total/moneyline to games_with_team_stats for edge calculation.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from engine.engine import ODDS_API_BASE, BASKETBALL_NBA, BASKETBALL_NCAAB, REQUEST_HEADERS
from engine.odds_quota import odds_api_get

HISTORICAL_MARKETS = ["h2h", "spreads", "totals"]
HISTORICAL_REGIONS = "us"
HISTORICAL_ODDS_FORMAT = "american"


def _data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def _odds_db_path() -> Path:
    return _data_dir() / "odds.db"


def _historical_cache_dir() -> Path:
    d = _data_dir() / "historical_odds_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path(sport_key: str, date_iso: str) -> Path:
    """One file per (sport, date) e.g. historical_odds_cache/basketball_nba/2023-01-09T12_00_00Z.json"""
    safe = date_iso.replace(":", "_").replace(" ", "_")[:19]
    sub = _historical_cache_dir() / sport_key.replace("/", "_")
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"{safe}.json"


def fetch_historical_odds(
    api_key: str,
    sport_key: str,
    date_iso: str,
    regions: str = HISTORICAL_REGIONS,
    markets: list[str] | None = None,
) -> dict[str, Any] | None:
    """
    GET /v4/historical/sports/{sport}/odds?date=...&regions=us&markets=h2h,spreads,totals&oddsFormat=american&apiKey=...
    Returns wrapped response { timestamp, previous_timestamp, next_timestamp, data } or None on failure.
    """
    if not (api_key or "").strip():
        return None
    url = f"{ODDS_API_BASE}/historical/sports/{sport_key}/odds"
    params = {
        "regions": regions,
        "markets": ",".join(markets or HISTORICAL_MARKETS),
        "oddsFormat": HISTORICAL_ODDS_FORMAT,
        "apiKey": api_key.strip(),
        "date": date_iso,
    }
    resp = odds_api_get(url, params=params, headers=REQUEST_HEADERS, timeout=20)
    if resp is None:
        return None
    try:
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def cache_response(sport_key: str, date_iso: str, response: dict[str, Any]) -> None:
    """Write API response to disk immediately to preserve credits."""
    path = _cache_path(sport_key, date_iso)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(response, f, indent=None)


def load_cached_response(sport_key: str, date_iso: str) -> dict[str, Any] | None:
    """Load cached response if present."""
    path = _cache_path(sport_key, date_iso)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _parse_historical_response(response: dict[str, Any], sport_key: str) -> list[dict[str, Any]]:
    """Flatten response into rows: sport_key, event_id, commence_time, home_team, away_team, snapshot_timestamp, bookmaker, market_type, outcome, point, price."""
    snapshot_ts = response.get("timestamp") or ""
    data = response.get("data")
    if not isinstance(data, list):
        return []
    rows = []
    for event in data:
        if not isinstance(event, dict):
            continue
        event_id = event.get("id", "")
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        commence_time = event.get("commence_time", "")
        for bm in event.get("bookmakers") or []:
            bookmaker = bm.get("key") or bm.get("title") or ""
            for mkt in bm.get("markets") or []:
                mkt_key = mkt.get("key", "")
                if mkt_key not in ("h2h", "spreads", "totals"):
                    continue
                for out in mkt.get("outcomes") or []:
                    name = out.get("name", "")
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
                        "sport_key": sport_key,
                        "event_id": event_id,
                        "commence_time": commence_time,
                        "home_team": home_team,
                        "away_team": away_team,
                        "snapshot_timestamp": snapshot_ts,
                        "bookmaker": bookmaker,
                        "market_type": mkt_key,
                        "outcome": name,
                        "point": point,
                        "price": price,
                    })
    return rows


def _create_historical_odds_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS historical_odds (
            sport_key TEXT NOT NULL,
            event_id TEXT NOT NULL,
            commence_time TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            snapshot_timestamp TEXT NOT NULL,
            bookmaker TEXT NOT NULL,
            market_type TEXT NOT NULL,
            outcome TEXT NOT NULL,
            point REAL,
            price REAL NOT NULL,
            point_key REAL NOT NULL,
            PRIMARY KEY (sport_key, event_id, snapshot_timestamp, bookmaker, market_type, outcome, point_key)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_historical_odds_lookup ON historical_odds(sport_key, commence_time, home_team, away_team)")


def append_live_snapshot_to_historical(
    sport_key: str,
    df_parsed: pd.DataFrame,
    snapshot_timestamp: str,
    db_path: Path | None = None,
) -> int:
    """
    Append one live snapshot (from odds_fetcher current/upcoming fetch) to historical_odds
    so it can be used later as closing lines when games are in the past.
    df_parsed must have columns: game_id, home_team, away_team, commence_time, bookmaker, market_type, outcome, price, point.
    snapshot_timestamp should be ISO (e.g. datetime.now(timezone.utc).isoformat()).
    Returns count of rows inserted.
    """
    if df_parsed.empty or not snapshot_timestamp:
        return 0
    required = {"game_id", "home_team", "away_team", "commence_time", "bookmaker", "market_type", "outcome", "price"}
    if not required.issubset(df_parsed.columns):
        return 0
    rows = []
    for _, r in df_parsed.iterrows():
        point = r.get("point")
        point_key = float(point) if point is not None and pd.notna(point) else -9999.0
        try:
            price = float(r["price"])
        except (TypeError, ValueError):
            continue
        rows.append({
            "sport_key": sport_key,
            "event_id": str(r["game_id"]),
            "commence_time": str(r["commence_time"]),
            "home_team": str(r["home_team"]),
            "away_team": str(r["away_team"]),
            "snapshot_timestamp": snapshot_timestamp,
            "bookmaker": str(r["bookmaker"]),
            "market_type": str(r["market_type"]),
            "outcome": str(r["outcome"]),
            "point": point if point is not None and pd.notna(point) else None,
            "price": price,
            "point_key": point_key,
        })
    return insert_historical_odds(rows, db_path=db_path)


def insert_historical_odds(rows: list[dict[str, Any]], db_path: Path | None = None) -> int:
    """Insert parsed rows into historical_odds. Returns count inserted."""
    if not rows:
        return 0
    path = db_path or _odds_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        _create_historical_odds_table(conn)
        cur = conn.cursor()
        n = 0
        for r in rows:
            point = r.get("point")
            point_key = float(point) if point is not None else -9999.0
            try:
                cur.execute(
                    """INSERT OR IGNORE INTO historical_odds
                       (sport_key, event_id, commence_time, home_team, away_team, snapshot_timestamp, bookmaker, market_type, outcome, point, price, point_key)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        r["sport_key"], r["event_id"], r["commence_time"], r["home_team"], r["away_team"],
                        r["snapshot_timestamp"], r["bookmaker"], r["market_type"], r["outcome"],
                        point, r["price"], point_key,
                    ),
                )
                n += cur.rowcount
            except sqlite3.IntegrityError:
                pass
        conn.commit()
        return n
    finally:
        conn.close()


def fetch_and_store_historical_week(
    api_key: str,
    sport_key: str,
    date_iso: str,
    db_path: Path | None = None,
    use_cache: bool = True,
) -> int:
    """Fetch one snapshot for the given date, cache to disk, parse and insert into historical_odds. Returns rows inserted."""
    if use_cache:
        cached = load_cached_response(sport_key, date_iso)
        if cached is not None:
            rows = _parse_historical_response(cached, sport_key)
            return insert_historical_odds(rows, db_path=db_path)
    resp = fetch_historical_odds(api_key, sport_key, date_iso)
    if resp is None:
        return 0
    cache_response(sport_key, date_iso, resp)
    rows = _parse_historical_response(resp, sport_key)
    return insert_historical_odds(rows, db_path=db_path)


def fetch_last_n_seasons(
    api_key: str,
    seasons: int = 2,
    sport_keys: list[str] | None = None,
    db_path: Path | None = None,
    use_cache: bool = True,
) -> dict[str, int]:
    """
    Iterate week by week over the last N seasons, fetch historical odds for each (sport, week), cache and store.
    Returns dict sport_key -> total rows inserted in this run.
    """
    sport_keys = sport_keys or [BASKETBALL_NBA, BASKETBALL_NCAAB]
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=365 * seasons)
    totals: dict[str, int] = {s: 0 for s in sport_keys}
    # One request per week: use Monday 12:00 UTC as snapshot date for that week
    current = start
    while current < now:
        date_iso = current.strftime("%Y-%m-%dT12:00:00Z")
        for sport_key in sport_keys:
            n = fetch_and_store_historical_week(
                api_key, sport_key, date_iso, db_path=db_path, use_cache=use_cache
            )
            totals[sport_key] += n
        current += timedelta(days=7)
    return totals


def load_historical_odds(
    sport_key: str | None = None,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Load historical_odds from SQLite."""
    path = db_path or _odds_db_path()
    if not path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='historical_odds'")
        if cur.fetchone() is None:
            return pd.DataFrame()
        if sport_key:
            return pd.read_sql_query(
                "SELECT * FROM historical_odds WHERE sport_key = ?", conn, params=(sport_key,)
            )
        return pd.read_sql_query("SELECT * FROM historical_odds", conn)
    finally:
        conn.close()


def get_closing_lines_per_game(historical_df: pd.DataFrame) -> pd.DataFrame:
    """
    From historical_odds rows, compute one closing line per (sport_key, event_id, market_type, outcome, point).
    Closing = latest snapshot where snapshot_timestamp <= commence_time.
    Aggregate across bookmakers: take median closing price per outcome.
    Returns DataFrame with columns: sport_key, event_id, commence_time, home_team, away_team, market_type, outcome, point, closing_price.
    """
    if historical_df.empty or "snapshot_timestamp" not in historical_df.columns:
        return pd.DataFrame()
    df = historical_df.copy()
    df["commence_time_dt"] = pd.to_datetime(df["commence_time"], errors="coerce")
    df["snapshot_dt"] = pd.to_datetime(df["snapshot_timestamp"], errors="coerce")
    df = df[df["snapshot_dt"] <= df["commence_time_dt"]]
    if df.empty:
        return pd.DataFrame()
    latest = df.loc[df.groupby(
        ["sport_key", "event_id", "commence_time", "home_team", "away_team", "market_type", "outcome", "point"]
    )["snapshot_dt"].idxmax()]
    closing = latest.groupby(
        ["sport_key", "event_id", "commence_time", "home_team", "away_team", "market_type", "outcome", "point"],
        dropna=False,
    )["price"].median().reset_index()
    closing = closing.rename(columns={"price": "closing_price"})
    return closing


def _league_to_sport_key(league: str) -> str:
    s = str(league).strip().lower()
    if s == "nba":
        return BASKETBALL_NBA
    if s == "ncaab":
        return BASKETBALL_NCAAB
    return s


def _normalize_team(s: object) -> str:
    return (str(s).strip() if s is not None and not pd.isna(s) else "") or ""


# NCAAB: map ESPN (games) team name -> Odds API name for join. Populate from ncaab_odds_unmatched.csv.
# Keys normalized lower; comparison uses Odds API side as-is, game side mapped to this.
NCAAB_ESPN_TO_ODDS_API: dict[str, str] = {
    # Example: "duke" -> "Duke Blue Devils"; add entries when diagnostic shows unmatched games.
}


def _ncaab_game_team_for_join(espn_name: str, sport_key: str) -> str:
    """For NCAAB merge, map ESPN name to Odds API style so join matches. Else return normalized name."""
    if sport_key != BASKETBALL_NCAAB:
        return _normalize_team(espn_name)
    key = _normalize_team(espn_name).lower()
    return NCAAB_ESPN_TO_ODDS_API.get(key, _normalize_team(espn_name))


def _team_match_join(a: str, b: str) -> bool:
    """Match for join: exact or one contains the other."""
    a, b = _normalize_team(a), _normalize_team(b)
    if not a or not b:
        return False
    if a == b:
        return True
    if a in b or b in a:
        return True
    return False


def merge_historical_closing_into_games(
    espn_db_path: Path | None = None,
    odds_db_path: Path | None = None,
) -> pd.DataFrame:
    """
    Load games_with_team_stats from espn.db and historical_odds from odds.db.
    Compute closing lines per game (latest snapshot <= commence_time, median across bookmakers),
    join to games on (sport_key, game_date, home_team, away_team), add closing columns, write back to espn.db.
    New columns: closing_home_spread, closing_away_spread, closing_total, closing_odds_* , closing_ml_home, closing_ml_away.
    """
    _espn = espn_db_path or _data_dir() / "espn.db"
    _odds = odds_db_path or _odds_db_path()
    if not _espn.exists():
        return pd.DataFrame()
    conn_espn = sqlite3.connect(_espn)
    try:
        games = pd.read_sql_query("SELECT * FROM games_with_team_stats", conn_espn)
    finally:
        conn_espn.close()
    if games.empty:
        return games

    def add_closing_columns(g: pd.DataFrame) -> pd.DataFrame:
        out = g.copy()
        for col in [
            "closing_home_spread", "closing_away_spread", "closing_total",
            "closing_odds_home_spread", "closing_odds_away_spread",
            "closing_odds_over", "closing_odds_under",
            "closing_ml_home", "closing_ml_away",
        ]:
            if col not in out.columns:
                out[col] = None
        return out

    games = add_closing_columns(games)
    ho = load_historical_odds(db_path=_odds)
    closing = pd.DataFrame()
    if not ho.empty:
        closing = get_closing_lines_per_game(ho)
    if not closing.empty:
        closing["commence_date"] = pd.to_datetime(closing["commence_time"], errors="coerce").dt.normalize()
    games["_game_date"] = pd.to_datetime(games["game_date"], errors="coerce").dt.normalize()
    games["_sport_key"] = games["league"].apply(_league_to_sport_key)
    games["_home"] = games["home_team_name"].apply(_normalize_team)
    games["_away"] = games["away_team_name"].apply(_normalize_team)
    if not closing.empty:
        spread_rows = closing[closing["market_type"] == "spreads"].copy()
        total_rows = closing[closing["market_type"] == "totals"].copy()
        ml_rows = closing[closing["market_type"] == "h2h"].copy()
        for idx, row in games.iterrows():
            sk = row["_sport_key"]
            gd = row["_game_date"]
            h, a = row["_home"], row["_away"]
            h_join = _ncaab_game_team_for_join(h, sk)
            a_join = _ncaab_game_team_for_join(a, sk)
            if pd.isna(gd):
                continue
            c_spread = spread_rows[
                (spread_rows["sport_key"] == sk) &
                (spread_rows["commence_date"] == gd) &
                (spread_rows["home_team"].apply(lambda x: _team_match_join(x, h_join))) &
                (spread_rows["away_team"].apply(lambda x: _team_match_join(x, a_join)))
            ]
            if not c_spread.empty:
                home_outcomes = c_spread[c_spread["outcome"].apply(lambda x: _team_match_join(x, h_join))]
                away_outcomes = c_spread[c_spread["outcome"].apply(lambda x: _team_match_join(x, a_join))]
                if not home_outcomes.empty:
                    games.at[idx, "closing_home_spread"] = home_outcomes["point"].median()
                    games.at[idx, "closing_odds_home_spread"] = home_outcomes["closing_price"].median()
                if not away_outcomes.empty:
                    games.at[idx, "closing_away_spread"] = away_outcomes["point"].median()
                    games.at[idx, "closing_odds_away_spread"] = away_outcomes["closing_price"].median()
            c_totals = total_rows[
                (total_rows["sport_key"] == sk) &
                (total_rows["commence_date"] == gd) &
                (total_rows["home_team"].apply(lambda x: _team_match_join(x, h_join))) &
                (total_rows["away_team"].apply(lambda x: _team_match_join(x, a_join)))
            ]
            if not c_totals.empty:
                over = c_totals[c_totals["outcome"].str.strip().str.lower() == "over"]
                under = c_totals[c_totals["outcome"].str.strip().str.lower() == "under"]
                if not over.empty:
                    games.at[idx, "closing_total"] = over["point"].median()
                    games.at[idx, "closing_odds_over"] = over["closing_price"].median()
                if not under.empty:
                    games.at[idx, "closing_odds_under"] = under["closing_price"].median()
            c_ml = ml_rows[
                (ml_rows["sport_key"] == sk) &
                (ml_rows["commence_date"] == gd) &
                (ml_rows["home_team"].apply(lambda x: _team_match_join(x, h_join))) &
                (ml_rows["away_team"].apply(lambda x: _team_match_join(x, a_join)))
            ]
            if not c_ml.empty:
                home_ml = c_ml[c_ml["outcome"].apply(lambda x: _team_match_join(x, h_join))]
                away_ml = c_ml[c_ml["outcome"].apply(lambda x: _team_match_join(x, a_join))]
                if not home_ml.empty:
                    games.at[idx, "closing_ml_home"] = home_ml["closing_price"].median()
                if not away_ml.empty:
                    games.at[idx, "closing_ml_away"] = away_ml["closing_price"].median()
    # Fill NCAAB rows still missing closing from ncaab_historical_odds (e.g. from Kaggle ingest)
    ncaab_odds = load_ncaab_historical_odds(db_path=_odds)
    if not ncaab_odds.empty and "_game_date" in games.columns:
        ncaab_odds["_date"] = pd.to_datetime(ncaab_odds["date"], errors="coerce").dt.normalize()
        ncaab_mask = games["_sport_key"] == BASKETBALL_NCAAB
        for idx in games.index:
            if not ncaab_mask.loc[idx]:
                continue
            if pd.notna(games.at[idx, "closing_home_spread"]):
                continue
            gd = games.at[idx, "_game_date"]
            h, a = games.at[idx, "_home"], games.at[idx, "_away"]
            if pd.isna(gd):
                continue
            matches = ncaab_odds[
                (ncaab_odds["_date"] == gd) &
                (ncaab_odds["home_team"].apply(lambda x: _team_match_join(x, h))) &
                (ncaab_odds["away_team"].apply(lambda x: _team_match_join(x, a)))
            ]
            if not matches.empty:
                row = matches.iloc[0]
                if pd.notna(row.get("closing_home_spread")):
                    games.at[idx, "closing_home_spread"] = row["closing_home_spread"]
                    games.at[idx, "closing_away_spread"] = -float(row["closing_home_spread"]) if pd.notna(row["closing_home_spread"]) else None
                if pd.notna(row.get("closing_total")):
                    games.at[idx, "closing_total"] = row["closing_total"]
    # Fill NCAAB rows still missing from historical_covers_odds (e.g. 2026ncaablines.csv ingest)
    covers_ncaab = load_historical_covers_odds_ncaab(db_path=_odds)
    if not covers_ncaab.empty and "_game_date" in games.columns:
        covers_ncaab["_date"] = pd.to_datetime(covers_ncaab["date"], errors="coerce").dt.normalize()
        ncaab_mask = games["_sport_key"] == BASKETBALL_NCAAB
        for idx in games.index:
            if not ncaab_mask.loc[idx]:
                continue
            if pd.notna(games.at[idx, "closing_home_spread"]):
                continue
            gd = games.at[idx, "_game_date"]
            h, a = games.at[idx, "_home"], games.at[idx, "_away"]
            if pd.isna(gd):
                continue
            matches = covers_ncaab[
                (covers_ncaab["_date"] == gd) &
                (covers_ncaab["home_team"].apply(lambda x: _team_match_join(x, h))) &
                (covers_ncaab["away_team"].apply(lambda x: _team_match_join(x, a)))
            ]
            if not matches.empty:
                row = matches.iloc[0]
                if pd.notna(row.get("closing_home_spread")):
                    games.at[idx, "closing_home_spread"] = row["closing_home_spread"]
                    games.at[idx, "closing_away_spread"] = -float(row["closing_home_spread"]) if pd.notna(row["closing_home_spread"]) else None
                if pd.notna(row.get("closing_total")):
                    games.at[idx, "closing_total"] = row["closing_total"]
                if pd.notna(row.get("closing_ml_home")):
                    games.at[idx, "closing_ml_home"] = row["closing_ml_home"]
                if pd.notna(row.get("closing_ml_away")):
                    games.at[idx, "closing_ml_away"] = row["closing_ml_away"]
    games = games.drop(columns=["_game_date", "_sport_key", "_home", "_away"], errors="ignore")
    if not games.empty and _espn.exists():
        conn_espn = sqlite3.connect(_espn)
        try:
            conn_espn.execute("DROP TABLE IF EXISTS games_with_team_stats")
            games.to_sql("games_with_team_stats", conn_espn, index=False)
            conn_espn.commit()
        finally:
            conn_espn.close()
    return games


def _create_ncaab_historical_odds_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ncaab_historical_odds (
            date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            closing_home_spread REAL,
            closing_total REAL,
            home_score REAL,
            away_score REAL,
            PRIMARY KEY (date, home_team, away_team)
        )
    """)


def load_ncaab_historical_odds(db_path: Path | None = None) -> pd.DataFrame:
    """Load ncaab_historical_odds table (from ingest script). Columns: date, home_team, away_team, closing_home_spread, closing_total."""
    path = db_path or _odds_db_path()
    if not path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ncaab_historical_odds'")
        if cur.fetchone() is None:
            return pd.DataFrame()
        return pd.read_sql_query("SELECT date, home_team, away_team, closing_home_spread, closing_total FROM ncaab_historical_odds", conn)
    finally:
        conn.close()


def load_historical_covers_odds_ncaab(db_path: Path | None = None) -> pd.DataFrame:
    """Load NCAAB rows from historical_covers_odds (e.g. from 2026ncaablines ingest). Returns closing_home_spread, closing_total, closing_ml_home, closing_ml_away."""
    path = db_path or _odds_db_path()
    if not path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='historical_covers_odds'")
        if cur.fetchone() is None:
            return pd.DataFrame()
        cur = conn.execute("PRAGMA table_info(historical_covers_odds)")
        cols = [row[1] for row in cur.fetchall()]
        sport_ok = "sport" in cols
        if not sport_ok:
            return pd.DataFrame()
        sql = "SELECT date, home_team, away_team, closing_spread AS closing_home_spread, closing_total"
        if "closing_ml_home" in cols:
            sql += ", closing_ml_home"
        if "closing_ml_away" in cols:
            sql += ", closing_ml_away"
        sql += " FROM historical_covers_odds WHERE sport = 'ncaab'"
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


def report_closing_join_rate_by_league(
    espn_db_path: Path | None = None,
    write_unmatched_ncaab_csv: bool = True,
) -> None:
    """
    Load games_with_team_stats and print closing-line join rate by league.
    Optionally write NCAAB rows missing closing_home_spread to data/raw_odds/ncaab_odds_unmatched.csv.
    """
    _espn = espn_db_path or _data_dir() / "espn.db"
    if not _espn.exists():
        print("espn.db not found")
        return
    conn = sqlite3.connect(_espn)
    try:
        cur = conn.execute("PRAGMA table_info(games_with_team_stats)")
        cols = [r[1] for r in cur.fetchall()]
        if "closing_home_spread" not in cols:
            print("games_with_team_stats has no closing_home_spread column (run pipeline with historical odds merge).")
            return
        df = pd.read_sql_query("SELECT league, closing_home_spread FROM games_with_team_stats", conn)
    finally:
        conn.close()
    if df.empty:
        print("games_with_team_stats is empty")
        return
    for league in df["league"].dropna().unique():
        sub = df[df["league"] == league]
        total = len(sub)
        with_spread = sub["closing_home_spread"].notna().sum()
        pct = (100.0 * with_spread / total) if total else 0
        print(f"  {league}: {with_spread}/{total} ({pct:.1f}%)")
    if write_unmatched_ncaab_csv:
        ncaab = df[df["league"].astype(str).str.strip().str.lower() == "ncaab"]
        unmatched = ncaab[ncaab["closing_home_spread"].isna()]
        if not unmatched.empty:
            conn = sqlite3.connect(_espn)
            try:
                full = pd.read_sql_query(
                    "SELECT game_date, home_team_name, away_team_name FROM games_with_team_stats WHERE league = 'ncaab' AND closing_home_spread IS NULL",
                    conn,
                )
            finally:
                conn.close()
            out_dir = _data_dir() / "raw_odds"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / "ncaab_odds_unmatched.csv"
            full.to_csv(path, index=False)
            print(f"  Unmatched NCAAB rows written to {path}")
