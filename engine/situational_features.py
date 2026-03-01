"""
Situational feature engineering for NBA and NCAAB.
Days of rest (0,1,2,3+), back-to-back, travel distance (geopy), 3-in-4 nights,
home/away win rate last 30 days, after win/loss (bounce-back/letdown).
All features use only data before game_date (no lookahead). Stored in game_situational_features table.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# NBA team display name (ESPN) -> city string for geopy
NBA_TEAM_CITY: dict[str, str] = {
    "Atlanta Hawks": "Atlanta, GA",
    "Boston Celtics": "Boston, MA",
    "Brooklyn Nets": "Brooklyn, NY",
    "Charlotte Hornets": "Charlotte, NC",
    "Chicago Bulls": "Chicago, IL",
    "Cleveland Cavaliers": "Cleveland, OH",
    "Dallas Mavericks": "Dallas, TX",
    "Denver Nuggets": "Denver, CO",
    "Detroit Pistons": "Detroit, MI",
    "Golden State Warriors": "San Francisco, CA",
    "Houston Rockets": "Houston, TX",
    "Indiana Pacers": "Indianapolis, IN",
    "Los Angeles Clippers": "Los Angeles, CA",
    "Los Angeles Lakers": "Los Angeles, CA",
    "Memphis Grizzlies": "Memphis, TN",
    "Miami Heat": "Miami, FL",
    "Milwaukee Bucks": "Milwaukee, WI",
    "Minnesota Timberwolves": "Minneapolis, MN",
    "New Orleans Pelicans": "New Orleans, LA",
    "New York Knicks": "New York, NY",
    "Oklahoma City Thunder": "Oklahoma City, OK",
    "Orlando Magic": "Orlando, FL",
    "Philadelphia 76ers": "Philadelphia, PA",
    "Phoenix Suns": "Phoenix, AZ",
    "Portland Trail Blazers": "Portland, OR",
    "Sacramento Kings": "Sacramento, CA",
    "San Antonio Spurs": "San Antonio, TX",
    "Toronto Raptors": "Toronto, ON, Canada",
    "Utah Jazz": "Salt Lake City, UT",
    "Washington Wizards": "Washington, DC",
}


def _data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def _db_path() -> Path:
    return _data_dir() / "espn.db"


def _parse_date(s: Any) -> Optional[datetime]:
    """Parse game_date string to date. Returns None if invalid."""
    if s is None or pd.isna(s):
        return None
    s = str(s).strip()
    if not s or len(s) < 10:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d")
    except ValueError:
        try:
            return datetime.strptime(s[:10], "%Y/%m/%d")
        except ValueError:
            return None


def _days_rest(team: str, game_date: datetime, games_before: pd.DataFrame) -> tuple[int, bool]:
    """
    Days of rest (0,1,2,3+) and is_b2b for team before game_date.
    games_before: rows with game_date < game_date where team played. Uses only past data (no lookahead).
    Returns (days_rest_capped: 0|1|2|3 for 3+, is_b2b).
    """
    team_cols = ["home_team_name", "away_team_name"]
    if games_before.empty or not all(c in games_before.columns for c in team_cols):
        return (3, False)  # no prior game -> treat as 3+ rest
    team_games = games_before[
        (games_before["home_team_name"].astype(str).str.strip() == team)
        | (games_before["away_team_name"].astype(str).str.strip() == team)
    ]
    if team_games.empty:
        return (3, False)
    dates = team_games["game_date"].apply(_parse_date).dropna()
    if dates.empty:
        return (3, False)
    last_dt = dates.max()
    delta = (game_date - last_dt).days
    if delta <= 0:
        return (0, False)
    is_b2b = delta == 1
    days_capped = min(3, delta)
    return (days_capped, is_b2b)


def _games_in_last_n_days(team: str, game_date: datetime, games_before: pd.DataFrame, n_days: int) -> int:
    """Count team's games in (game_date - n_days, game_date). No lookahead."""
    if games_before.empty:
        return 0
    start = game_date - timedelta(days=n_days)
    team_games = games_before[
        (
            (games_before["home_team_name"].astype(str).str.strip() == team)
            | (games_before["away_team_name"].astype(str).str.strip() == team)
        )
    ]
    if team_games.empty:
        return 0
    team_games = team_games.copy()
    team_games["_dt"] = team_games["game_date"].apply(_parse_date)
    team_games = team_games.dropna(subset=["_dt"])
    return int(((team_games["_dt"] > start) & (team_games["_dt"] < game_date)).sum())


def _win_pct_last30_home_away(
    team: str, game_date: datetime, games_before: pd.DataFrame
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    (home_win_pct, away_win_pct, overall_win_pct) for team in last 30 days before game_date.
    Uses only games with game_date < game_date. Returns (None, None, None) if no games.
    """
    if games_before.empty:
        return (None, None, None)
    start = game_date - timedelta(days=30)
    team_games = games_before[
        (
            (games_before["home_team_name"].astype(str).str.strip() == team)
            | (games_before["away_team_name"].astype(str).str.strip() == team)
        )
    ].copy()
    team_games["_dt"] = team_games["game_date"].apply(_parse_date)
    team_games = team_games.dropna(subset=["_dt"])
    team_games = team_games[(team_games["_dt"] >= start) & (team_games["_dt"] < game_date)]
    if team_games.empty:
        return (None, None, None)
    home_w, home_t, away_w, away_t = 0, 0, 0, 0
    for _, r in team_games.iterrows():
        h = str(r.get("home_team_name", "")).strip()
        a = str(r.get("away_team_name", "")).strip()
        sh = float(r.get("home_score") or 0)
        sa = float(r.get("away_score") or 0)
        if h == team:
            home_t += 1
            if sh > sa:
                home_w += 1
        else:
            away_t += 1
            if sa > sh:
                away_w += 1
    home_pct = (home_w / home_t) if home_t else None
    away_pct = (away_w / away_t) if away_t else None
    tot_w, tot_t = home_w + away_w, home_t + away_t
    overall_pct = (tot_w / tot_t) if tot_t else None
    return (home_pct, away_pct, overall_pct)


def _last_game_result(team: str, game_date: datetime, games_before: pd.DataFrame) -> Optional[bool]:
    """True if team won their most recent game before game_date, False if lost, None if no prior game."""
    if games_before.empty:
        return None
    team_games = games_before[
        (games_before["home_team_name"].astype(str).str.strip() == team)
        | (games_before["away_team_name"].astype(str).str.strip() == team)
    ].copy()
    team_games["_dt"] = team_games["game_date"].apply(_parse_date)
    team_games = team_games.dropna(subset=["_dt"])
    if team_games.empty:
        return None
    last = team_games.loc[team_games["_dt"].idxmax()]
    h = str(last["home_team_name"]).strip()
    sh = float(last.get("home_score") or 0)
    sa = float(last.get("away_score") or 0)
    if h == team:
        return sh > sa
    return sa > sh


def _travel_miles(
    team: str,
    current_game_city: str,
    game_date: datetime,
    games_before: pd.DataFrame,
    league: str,
) -> Optional[float]:
    """
    Travel distance in miles from team's last game city to current game city.
    current_game_city: where this game is played (home team's city for this game).
    Last game city: from last game, if team was home then home_team city, else away_team city.
    """
    if league != "nba":
        return None
    city_map = NBA_TEAM_CITY
    if not current_game_city:
        return None
    if games_before.empty:
        return None
    team_games = games_before[
        (games_before["home_team_name"].astype(str).str.strip() == team)
        | (games_before["away_team_name"].astype(str).str.strip() == team)
    ].copy()
    team_games["_dt"] = team_games["game_date"].apply(_parse_date)
    team_games = team_games.dropna(subset=["_dt"])
    if team_games.empty:
        return None
    last = team_games.loc[team_games["_dt"].idxmax()]
    h = str(last["home_team_name"]).strip()
    last_city = city_map.get(h) if h == team else city_map.get(str(last["away_team_name"]).strip())
    if not last_city or last_city == current_game_city:
        return 0.0
    try:
        from geopy.distance import geodesic
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="bettingsim_situational")
        loc_cur = geolocator.geocode(current_game_city, timeout=5)
        loc_last = geolocator.geocode(last_city, timeout=5)
        if loc_cur is None or loc_last is None:
            return None
        return float(geodesic((loc_cur.latitude, loc_cur.longitude), (loc_last.latitude, loc_last.longitude)).miles)
    except Exception:
        return None


def build_situational_features(
    games_df: pd.DataFrame,
    league: str,
    use_geopy: bool = True,
) -> pd.DataFrame:
    """
    Build situational features for each game.

    No lookahead: for each game only uses rows with game_date strictly before
    that game's game_date. Rest, B2B, 3-in-4, L30 win%, last result, and travel
    are all computed from past games only.
    """
    if games_df.empty or "game_date" not in games_df.columns:
        return pd.DataFrame()
    required = ["game_id", "game_date", "home_team_name", "away_team_name"]
    if not all(c in games_df.columns for c in required):
        return pd.DataFrame()
    g = games_df.sort_values("game_date").reset_index(drop=True)
    rows = []
    for i, row in g.iterrows():
        game_id = row["game_id"]
        game_date = _parse_date(row["game_date"])
        if game_date is None:
            continue
        home = str(row["home_team_name"]).strip()
        away = str(row["away_team_name"]).strip()
        # Only games strictly before this game (no lookahead)
        before = g[g["game_date"] < row["game_date"]]
        home_days, home_b2b = _days_rest(home, game_date, before)
        away_days, away_b2b = _days_rest(away, game_date, before)
        home_3in4 = 1 if _games_in_last_n_days(home, game_date, before, 4) >= 3 else 0
        away_3in4 = 1 if _games_in_last_n_days(away, game_date, before, 4) >= 3 else 0
        home_home_wp, home_away_wp, home_overall_wp = _win_pct_last30_home_away(home, game_date, before)
        away_home_wp, away_away_wp, away_overall_wp = _win_pct_last30_home_away(away, game_date, before)
        home_last_w = _last_game_result(home, game_date, before)
        away_last_w = _last_game_result(away, game_date, before)
        home_after_win = 1 if home_last_w is True else 0
        home_after_loss = 1 if home_last_w is False else 0
        away_after_win = 1 if away_last_w is True else 0
        away_after_loss = 1 if away_last_w is False else 0
        home_city = NBA_TEAM_CITY.get(home) if league == "nba" else None
        away_current_city = home_city  # away team is playing at home team's venue
        home_travel = (
            _travel_miles(home, home_city or "", game_date, before, league)
            if (use_geopy and league == "nba" and home_city) else None
        )
        away_travel = (
            _travel_miles(away, away_current_city or "", game_date, before, league)
            if (use_geopy and league == "nba" and away_current_city) else None
        )
        rows.append({
            "league": league,
            "game_id": game_id,
            "home_days_rest": home_days,
            "away_days_rest": away_days,
            "home_is_b2b": 1 if home_b2b else 0,
            "away_is_b2b": 1 if away_b2b else 0,
            "home_travel_miles": home_travel,
            "away_travel_miles": away_travel,
            "home_3_in_4": home_3in4,
            "away_3_in_4": away_3in4,
            "home_win_pct_last30": home_overall_wp,
            "away_win_pct_last30": away_overall_wp,
            "home_home_win_pct_last30": home_home_wp,
            "home_away_win_pct_last30": home_away_wp,
            "away_home_win_pct_last30": away_home_wp,
            "away_away_win_pct_last30": away_away_wp,
            "home_after_win": home_after_win,
            "home_after_loss": home_after_loss,
            "away_after_win": away_after_win,
            "away_after_loss": away_after_loss,
        })
    return pd.DataFrame(rows)


def _create_feature_table_if_not_exists(conn: sqlite3.Connection) -> None:
    """Create game_situational_features table with stable schema (no lookahead)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS game_situational_features (
            league TEXT NOT NULL,
            game_id TEXT NOT NULL,
            home_days_rest INTEGER NOT NULL,
            away_days_rest INTEGER NOT NULL,
            home_is_b2b INTEGER NOT NULL,
            away_is_b2b INTEGER NOT NULL,
            home_travel_miles REAL,
            away_travel_miles REAL,
            home_3_in_4 INTEGER NOT NULL,
            away_3_in_4 INTEGER NOT NULL,
            home_win_pct_last30 REAL,
            away_win_pct_last30 REAL,
            home_home_win_pct_last30 REAL,
            home_away_win_pct_last30 REAL,
            away_home_win_pct_last30 REAL,
            away_away_win_pct_last30 REAL,
            home_after_win INTEGER NOT NULL,
            home_after_loss INTEGER NOT NULL,
            away_after_win INTEGER NOT NULL,
            away_after_loss INTEGER NOT NULL,
            PRIMARY KEY (league, game_id)
        )
    """)


SITUATIONAL_FEATURE_COLUMNS = [
    "league", "game_id", "home_days_rest", "away_days_rest", "home_is_b2b", "away_is_b2b",
    "home_travel_miles", "away_travel_miles", "home_3_in_4", "away_3_in_4",
    "home_win_pct_last30", "away_win_pct_last30",
    "home_home_win_pct_last30", "home_away_win_pct_last30", "away_home_win_pct_last30", "away_away_win_pct_last30",
    "home_after_win", "home_after_loss", "away_after_win", "away_after_loss",
]


def save_situational_features_to_sqlite(df: pd.DataFrame, db_path: Optional[Path] = None) -> None:
    """Upsert game_situational_features: replace rows for leagues present in df, keep others."""
    if df.empty:
        return
    path = db_path or _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[[c for c in SITUATIONAL_FEATURE_COLUMNS if c in df.columns]].copy()
    if out.empty:
        return
    conn = sqlite3.connect(path)
    try:
        _create_feature_table_if_not_exists(conn)
        leagues_in_df = out["league"].unique().tolist()
        placeholders = ",".join("?" * len(leagues_in_df))
        conn.execute(f"DELETE FROM game_situational_features WHERE league IN ({placeholders})", leagues_in_df)
        out.to_sql("game_situational_features", conn, if_exists="append", index=False)
        conn.commit()
    finally:
        conn.close()


def load_situational_features_from_sqlite(
    league: Optional[str] = None, db_path: Optional[Path] = None
) -> pd.DataFrame:
    """Load game_situational_features from SQLite."""
    path = db_path or _db_path()
    if not path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(path)
    try:
        if league:
            return pd.read_sql_query(
                "SELECT * FROM game_situational_features WHERE league = ?", conn, params=(league,)
            )
        return pd.read_sql_query("SELECT * FROM game_situational_features", conn)
    except sqlite3.OperationalError:
        return pd.DataFrame()
    finally:
        conn.close()


def build_and_save_situational_features(
    games_df: pd.DataFrame,
    league: str,
    db_path: Optional[Path] = None,
    use_geopy: bool = True,
) -> pd.DataFrame:
    """Build features from games DataFrame and save to SQLite. Returns the features DataFrame."""
    existing = load_situational_features_from_sqlite(league=None, db_path=db_path)
    new_df = build_situational_features(games_df, league=league, use_geopy=use_geopy)
    if new_df.empty:
        return existing
    if not existing.empty:
        other = existing[existing["league"] != league]
        combined = pd.concat([other, new_df], ignore_index=True)
    else:
        combined = new_df
    save_situational_features_to_sqlite(combined, db_path=db_path)
    return new_df
