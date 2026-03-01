"""
Pull historical team-level stats from sportsreference (Sports-Reference / Basketball-Reference).
NBA and NCAAB Men's: offensive/defensive ratings, pace, turnover rate, rebound rate, strength of schedule.
Merge with ESPN games in SQLite using team name + season, with a name-mapping dictionary.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

# Last 5 seasons (year = season end, e.g. 2024 = 2023-24)
def _default_seasons() -> list[int]:
    from datetime import datetime
    y = datetime.now().year
    return [y - 1, y - 2, y - 3, y - 4, y - 5]


# Name mapping: ESPN (games table) name -> Sports-Reference name
# Keys are normalized (lower, stripped); values are as returned by sportsreference.
# Add entries when sources use different names (e.g. "St. John's" vs "St. John's (NY)").
TEAM_NAME_MAPPING: dict[str, str] = {
    # NBA examples (ESPN often matches; add only when different)
    "trail blazers": "Portland Trail Blazers",
    "blazers": "Portland Trail Blazers",
    # NCAAB: common discrepancies
    "st. john's": "St. John's (NY)",
    "st john's": "St. John's (NY)",
    "st. johns": "St. John's (NY)",
    "ucf": "Central Florida",
    "unlv": "Nevada-Las Vegas",
    "lsu": "Louisiana State",
    "usc": "Southern California",
    "byu": "Brigham Young",
    "smu": "Southern Methodist",
    "tcu": "Texas Christian",
    "ole miss": "Mississippi",
    "uconn": "Connecticut",
    "umass": "Massachusetts",
    "unc": "North Carolina",
    "nc state": "North Carolina State",
    "vcu": "Virginia Commonwealth",
    "fau": "Florida Atlantic",
}


def _normalize_name(name: str) -> str:
    """Normalize for lookup: strip, lower."""
    if not name or not isinstance(name, str):
        return ""
    return name.strip().lower()


def _apply_name_mapping(name: str, mapping: dict[str, str] | None) -> str:
    """Map ESPN/games name to sportsreference name. Returns original if no mapping."""
    if not name:
        return name
    m = mapping or TEAM_NAME_MAPPING
    key = _normalize_name(name)
    return m.get(key, name)


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _possessions(fga: float, orb: float, to: float, fta: float) -> float:
    """Approximate team possessions (Basketball-Reference style)."""
    return fga - orb + to + 0.44 * fta


def _offensive_rating(pts: float, poss: float) -> float | None:
    if poss is None or poss <= 0:
        return None
    return 100.0 * pts / poss


def _defensive_rating(opp_pts: float, opp_poss: float) -> float | None:
    if opp_poss is None or opp_poss <= 0:
        return None
    return 100.0 * opp_pts / opp_poss


def _pace(poss: float, opp_poss: float, games: float) -> float | None:
    """Possessions per game (team pace)."""
    if games is None or games <= 0:
        return None
    return (poss + opp_poss) / (2.0 * games)


def _turnover_rate(to: float, fga: float, fta: float) -> float | None:
    """TO / (FGA + 0.44*FTA + TO)."""
    denom = fga + 0.44 * fta + to
    if denom is None or denom <= 0:
        return None
    return 100.0 * to / denom


def _rebound_rate(orb: float, opp_drb: float) -> float | None:
    """Offensive rebound % = ORB / (ORB + opp_DRB)."""
    denom = orb + opp_drb
    if denom is None or denom <= 0:
        return None
    return 100.0 * orb / denom


def _team_row(team: Any, league: str, season: int) -> dict[str, Any] | None:
    """Build one row of team stats from a sportsreference Team object (NBA or NCAAB)."""
    try:
        name = getattr(team, "name", None)
        if not name:
            return None
        games = _safe_float(getattr(team, "games_played", None))
        pts = _safe_float(getattr(team, "points", None))
        fga = _safe_float(getattr(team, "field_goal_attempts", None))
        fta = _safe_float(getattr(team, "free_throw_attempts", None))
        orb = _safe_float(getattr(team, "offensive_rebounds", None))
        drb = _safe_float(getattr(team, "defensive_rebounds", None))
        to = _safe_float(getattr(team, "turnovers", None))
        opp_pts = _safe_float(getattr(team, "opp_points", None))
        opp_fga = _safe_float(getattr(team, "opp_field_goal_attempts", None))
        opp_fta = _safe_float(getattr(team, "opp_free_throw_attempts", None))
        opp_orb = _safe_float(getattr(team, "opp_offensive_rebounds", None))
        opp_drb = _safe_float(getattr(team, "opp_defensive_rebounds", None))
        opp_to = _safe_float(getattr(team, "opp_turnovers", None))
    except Exception:
        return None
    poss = _possessions(fga or 0, orb or 0, to or 0, fta or 0)
    opp_poss = _possessions(opp_fga or 0, opp_orb or 0, opp_to or 0, opp_fta or 0)
    return {
        "league": league,
        "season": season,
        "team_name": name,
        "offensive_rating": _offensive_rating(pts or 0, poss),
        "defensive_rating": _defensive_rating(opp_pts or 0, opp_poss),
        "pace": _pace(poss, opp_poss, games or 1),
        "turnover_rate": _turnover_rate(to or 0, fga or 0, fta or 0),
        "offensive_rebound_rate": _rebound_rate(orb or 0, opp_drb or 0),
        "defensive_rebound_rate": _rebound_rate(drb or 0, opp_orb or 0),
        "strength_of_schedule": None,  # sportsreference Team does not expose SOS; can be filled from elsewhere
    }


def fetch_nba_team_stats(seasons: list[int] | None = None) -> pd.DataFrame:
    """Fetch NBA team-level stats for the given seasons (season end year, e.g. 2023 = 2022-23)."""
    seasons = seasons or _default_seasons()
    rows: list[dict[str, Any]] = []
    try:
        from sportsreference.nba.teams import Teams
    except ImportError:
        return pd.DataFrame(columns=["league", "season", "team_name", "offensive_rating", "defensive_rating", "pace", "turnover_rate", "offensive_rebound_rate", "defensive_rebound_rate", "strength_of_schedule"])
    for year in seasons:
        try:
            t = Teams(year)
            for team in t:
                row = _team_row(team, "nba", year)
                if row:
                    rows.append(row)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["league", "season", "team_name", "offensive_rating", "defensive_rating", "pace", "turnover_rate", "offensive_rebound_rate", "defensive_rebound_rate", "strength_of_schedule"])
    return pd.DataFrame(rows)


def fetch_ncaab_team_stats(seasons: list[int] | None = None) -> pd.DataFrame:
    """Fetch NCAAB Men's team-level stats for the given seasons."""
    seasons = seasons or _default_seasons()
    rows = []
    try:
        from sportsreference.ncaab.teams import Teams
    except ImportError:
        return pd.DataFrame(columns=["league", "season", "team_name", "offensive_rating", "defensive_rating", "pace", "turnover_rate", "offensive_rebound_rate", "defensive_rebound_rate", "strength_of_schedule"])
    for year in seasons:
        try:
            t = Teams(year)
            for team in t:
                row = _team_row(team, "ncaab", year)
                if row:
                    rows.append(row)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["league", "season", "team_name", "offensive_rating", "defensive_rating", "pace", "turnover_rate", "offensive_rebound_rate", "defensive_rebound_rate", "strength_of_schedule"])
    return pd.DataFrame(rows)


def fetch_all_team_stats(seasons: list[int] | None = None) -> pd.DataFrame:
    """Fetch NBA and NCAAB team stats for the last 5 seasons (default)."""
    seasons = seasons or _default_seasons()
    nba = fetch_nba_team_stats(seasons)
    ncaab = fetch_ncaab_team_stats(seasons)
    if nba.empty and ncaab.empty:
        return pd.DataFrame(columns=["league", "season", "team_name", "offensive_rating", "defensive_rating", "pace", "turnover_rate", "offensive_rebound_rate", "defensive_rebound_rate", "strength_of_schedule"])
    if nba.empty:
        return ncaab
    if ncaab.empty:
        return nba
    return pd.concat([nba, ncaab], ignore_index=True)


def _data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def _espn_db_path() -> Path:
    return _data_dir() / "espn.db"


def _game_season_from_date(game_date: Any) -> int | None:
    """Derive season (end year) from game_date. Oct+ -> next year; else current year."""
    if game_date is None or pd.isna(game_date):
        return None
    s = str(game_date).strip()
    if not s or len(s) < 4:
        return None
    try:
        year = int(s[:4])
        if len(s) >= 7:
            month = int(s[5:7])
            if month >= 10:
                return year + 1
        return year
    except (ValueError, TypeError):
        return None


def load_games_with_season(db_path: Path | None = None, league: str | None = None) -> pd.DataFrame:
    """Load games from ESPN SQLite and add a 'season' column derived from game_date."""
    path = db_path or _espn_db_path()
    if not path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(path)
    try:
        if league:
            df = pd.read_sql_query("SELECT * FROM games WHERE league = ?", conn, params=(league,))
        else:
            df = pd.read_sql_query("SELECT * FROM games", conn)
        if df.empty:
            return df
        if "game_date" in df.columns:
            df = df.copy()
            df["season"] = df["game_date"].apply(_game_season_from_date)
        return df
    finally:
        conn.close()


def save_team_stats_to_sqlite(df: pd.DataFrame, db_path: Path | None = None) -> None:
    """Save sportsreference team stats to SQLite (table team_advanced_stats)."""
    if df.empty:
        return
    path = db_path or _espn_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS team_advanced_stats (
                league TEXT NOT NULL,
                season INTEGER NOT NULL,
                team_name TEXT NOT NULL,
                offensive_rating REAL,
                defensive_rating REAL,
                pace REAL,
                turnover_rate REAL,
                offensive_rebound_rate REAL,
                defensive_rebound_rate REAL,
                strength_of_schedule REAL,
                PRIMARY KEY (league, season, team_name)
            )
        """)
        conn.execute("DELETE FROM team_advanced_stats")
        df.to_sql("team_advanced_stats", conn, if_exists="append", index=False)
        conn.commit()
    finally:
        conn.close()


def merge_games_with_team_stats(
    games_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    name_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Merge games with team stats on team name and season.
    Uses name_mapping to align ESPN names to sportsreference names (for stats lookup).
    Adds home_* and away_* stat columns for each game.
    """
    if games_df.empty or stats_df.empty:
        return games_df
    name_mapping = name_mapping or TEAM_NAME_MAPPING
    g = games_df.copy()
    if "season" not in g.columns and "game_date" in g.columns:
        g["season"] = g["game_date"].apply(_game_season_from_date)
    if "season" not in g.columns:
        return g
    g["_home_name"] = g["home_team_name"].apply(lambda x: _apply_name_mapping(x, name_mapping))
    g["_away_name"] = g["away_team_name"].apply(lambda x: _apply_name_mapping(x, name_mapping))
    stat_cols = ["offensive_rating", "defensive_rating", "pace", "turnover_rate", "offensive_rebound_rate", "defensive_rebound_rate", "strength_of_schedule"]
    s = stats_df.copy()
    # Home: merge on league, season, _home_name = team_name
    home_stats = s.rename(columns={k: f"home_{k}" for k in stat_cols})
    g = g.merge(
        home_stats[["league", "season", "team_name"] + [f"home_{k}" for k in stat_cols]],
        left_on=["league", "season", "_home_name"],
        right_on=["league", "season", "team_name"],
        how="left",
        suffixes=("", "_h"),
    )
    g = g.drop(columns=[c for c in g.columns if c.endswith("_h")], errors="ignore")
    if "team_name" in g.columns:
        g = g.drop(columns=["team_name"], errors="ignore")
    # Away: merge on league, season, _away_name = team_name
    away_stats = s.rename(columns={k: f"away_{k}" for k in stat_cols})
    g = g.merge(
        away_stats[["league", "season", "team_name"] + [f"away_{k}" for k in stat_cols]],
        left_on=["league", "season", "_away_name"],
        right_on=["league", "season", "team_name"],
        how="left",
        suffixes=("", "_a"),
    )
    g = g.drop(columns=[c for c in g.columns if c.endswith("_a")], errors="ignore")
    if "team_name" in g.columns:
        g = g.drop(columns=["team_name"], errors="ignore")
    g = g.drop(columns=["_home_name", "_away_name"], errors="ignore")
    return g


def fetch_merge_and_save(
    seasons: list[int] | None = None,
    db_path: Path | None = None,
    name_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Fetch NBA + NCAAB team stats, load games from SQLite, merge with name mapping, return merged games.
    Optionally save team_advanced_stats to SQLite (always) and merged games to a new table (optional).
    """
    seasons = seasons or _default_seasons()
    db_path = db_path or _espn_db_path()
    name_mapping = name_mapping or TEAM_NAME_MAPPING
    stats_df = fetch_all_team_stats(seasons)
    if not stats_df.empty:
        save_team_stats_to_sqlite(stats_df, db_path)
    games_df = load_games_with_season(db_path)
    if games_df.empty:
        return games_df
    merged = merge_games_with_team_stats(games_df, stats_df, name_mapping)
    # Save merged to SQLite for reuse
    if not merged.empty and db_path.exists():
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("DROP TABLE IF EXISTS games_with_team_stats")
            merged.to_sql("games_with_team_stats", conn, index=False)
            conn.commit()
        finally:
            conn.close()
    return merged


def load_team_advanced_stats_from_sqlite(league: str | None = None, db_path: Path | None = None) -> pd.DataFrame:
    """Load team_advanced_stats from SQLite (saved by fetch_merge_and_save or save_team_stats_to_sqlite)."""
    path = db_path or _espn_db_path()
    if not path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(path)
    try:
        if league:
            return pd.read_sql_query("SELECT * FROM team_advanced_stats WHERE league = ?", conn, params=(league,))
        return pd.read_sql_query("SELECT * FROM team_advanced_stats", conn)
    finally:
        conn.close()


def load_merged_games_from_sqlite(league: str | None = None, db_path: Path | None = None) -> pd.DataFrame:
    """Load games_with_team_stats from SQLite (merged games + home/away advanced stats)."""
    path = db_path or _espn_db_path()
    if not path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(path)
    try:
        if league:
            return pd.read_sql_query("SELECT * FROM games_with_team_stats WHERE league = ?", conn, params=(league,))
        return pd.read_sql_query("SELECT * FROM games_with_team_stats", conn)
    finally:
        conn.close()
