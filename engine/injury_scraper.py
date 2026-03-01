"""
Injury report scraper: ESPN unofficial API (all 4 sports) + nba_api (NBA availability, usage).
Flags games where a top-5 player by usage is out/doubtful and adds injury_impact_score to the feature matrix.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
REQUEST_DELAY = 0.5

# (sport, league) for URL
INJURY_LEAGUES = {
    "nfl": ("football", "nfl"),
    "nba": ("basketball", "nba"),
    "mlb": ("baseball", "mlb"),
    "nhl": ("hockey", "nhl"),
}

# Statuses considered "out or doubtful" for flagging
OUT_OR_DOUBTFUL = frozenset({"out", "doubtful", "out for season", "out indefinitely", "doubtful (illness)"})


def _normalize_status(s: str) -> str:
    return (s or "").strip().lower()


def _is_out_or_doubtful(status: str) -> bool:
    n = _normalize_status(status)
    if not n:
        return False
    return any(x in n for x in OUT_OR_DOUBTFUL) or n in OUT_OR_DOUBTFUL


def fetch_espn_injuries(league_key: str) -> list[dict[str, Any]]:
    """
    Fetch current injury data from ESPN for one league (nfl, nba, mlb, nhl).
    Returns list of injury records: team_id, team_name, player_id, player_name, status, short_comment, date.
    """
    if league_key not in INJURY_LEAGUES:
        return []
    time.sleep(REQUEST_DELAY)
    sport, league = INJURY_LEAGUES[league_key]
    url = f"{ESPN_BASE}/{sport}/{league}/injuries"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []
    injuries = data.get("injuries") or []
    rows = []
    for team_block in injuries:
        team_id = team_block.get("id", "")
        team_name = team_block.get("displayName", "")
        for inj in team_block.get("injuries") or []:
            athlete = inj.get("athlete") or {}
            rows.append({
                "league": league_key,
                "team_id": team_id,
                "team_name": team_name,
                "player_id": inj.get("id", ""),
                "player_name": athlete.get("displayName", ""),
                "status": inj.get("status", ""),
                "short_comment": inj.get("shortComment", ""),
                "date": inj.get("date", ""),
            })
    return rows


def fetch_espn_injuries_all() -> pd.DataFrame:
    """Fetch ESPN injuries for all four leagues. Returns DataFrame with league, team_id, team_name, player_id, player_name, status, short_comment, date."""
    rows = []
    for league_key in INJURY_LEAGUES:
        rows.extend(fetch_espn_injuries(league_key))
    if not rows:
        return pd.DataFrame(columns=["league", "team_id", "team_name", "player_id", "player_name", "status", "short_comment", "date"])
    return pd.DataFrame(rows)


# NBA TEAM_ID -> full name (ESPN-style) for matching
NBA_TEAM_ID_TO_NAME: dict[int, str] = {
    1610612737: "Atlanta Hawks", 1610612738: "Boston Celtics", 1610612751: "Brooklyn Nets",
    1610612766: "Charlotte Hornets", 1610612741: "Chicago Bulls", 1610612739: "Cleveland Cavaliers",
    1610612742: "Dallas Mavericks", 1610612743: "Denver Nuggets", 1610612765: "Detroit Pistons",
    1610612744: "Golden State Warriors", 1610612745: "Houston Rockets", 1610612754: "Indiana Pacers",
    1610612746: "LA Clippers", 1610612747: "Los Angeles Lakers", 1610612763: "Memphis Grizzlies",
    1610612748: "Miami Heat", 1610612749: "Milwaukee Bucks", 1610612750: "Minnesota Timberwolves",
    1610612740: "New Orleans Pelicans", 1610612752: "New York Knicks", 1610612760: "Oklahoma City Thunder",
    1610612753: "Orlando Magic", 1610612755: "Philadelphia 76ers", 1610612756: "Phoenix Suns",
    1610612757: "Portland Trail Blazers", 1610612758: "Sacramento Kings", 1610612759: "San Antonio Spurs",
    1610612761: "Toronto Raptors", 1610612762: "Utah Jazz", 1610612764: "Washington Wizards",
}


def _nba_team_id_to_name() -> dict[int | str, str]:
    """TEAM_ID -> full team name for NBA (used for top5 usage)."""
    return {k: v for k, v in NBA_TEAM_ID_TO_NAME.items()}


def fetch_nba_top5_usage(season: str = "2024-25") -> pd.DataFrame:
    """
    Fetch top 5 players by usage rate per team for NBA (nba_api PlayerEstimatedMetrics).
    Returns DataFrame: TEAM_ID, team_name, PLAYER_ID, PLAYER_NAME, E_USG_PCT, usage_rank.
    """
    try:
        from nba_api.stats.endpoints import playerestimatedmetrics
    except ImportError:
        return pd.DataFrame()
    time.sleep(REQUEST_DELAY)
    try:
        pm = playerestimatedmetrics.PlayerEstimatedMetrics(season=season, season_type="Regular Season")
        df = pm.get_data_frames()[0]
    except Exception:
        return pd.DataFrame()
    if df.empty or "E_USG_PCT" not in df.columns:
        return pd.DataFrame()
    # PlayerEstimatedMetrics may not have TEAM_ID; leaguedashplayerstats has both
    if "TEAM_ID" not in df.columns:
        try:
            from nba_api.stats.endpoints import leaguedashplayerstats
            time.sleep(REQUEST_DELAY)
            ld = leaguedashplayerstats.LeagueDashPlayerStats(season=season)
            team_df = ld.get_data_frames()[0]
            if not team_df.empty and "TEAM_ID" in team_df.columns and "PLAYER_ID" in team_df.columns:
                df = df.merge(team_df[["PLAYER_ID", "TEAM_ID"]].drop_duplicates(), on="PLAYER_ID", how="left")
        except Exception:
            pass
    if "TEAM_ID" not in df.columns:
        return pd.DataFrame()
    df = df.sort_values("E_USG_PCT", ascending=False)
    top5 = df.groupby("TEAM_ID", dropna=False).head(5).copy()
    top5["usage_rank"] = top5.groupby("TEAM_ID", dropna=False).cumcount() + 1
    tid2name = _nba_team_id_to_name()
    def _tid_to_name(x):
        try:
            return tid2name.get(int(float(x)), "") or ""
        except (TypeError, ValueError):
            return ""
    top5["team_name"] = top5["TEAM_ID"].apply(_tid_to_name)
    return top5


def fetch_nba_availability_nba_api(game_date: str) -> pd.DataFrame:
    """
    Fetch player availability for NBA games on a date (nba_api ScoreboardV2).
    game_date format: MM/DD/YYYY. Returns DataFrame with GAME_ID, and availability info if present.
    """
    try:
        from nba_api.stats.endpoints import scoreboardv2
    except ImportError:
        return pd.DataFrame()
    time.sleep(REQUEST_DELAY)
    try:
        sb = scoreboardv2.ScoreboardV2(game_date=game_date, day_offset=0, league_id="00")
        d = sb.get_dict()
    except Exception:
        return pd.DataFrame()
    result_sets = d.get("resultSets") or []
    available = None
    for rs in result_sets:
        if rs.get("name") == "Available":
            available = rs
            break
    if not available or not available.get("rowSet"):
        return pd.DataFrame()
    headers = available.get("headers", [])
    return pd.DataFrame(available["rowSet"], columns=headers)


def nba_injuries_with_usage(
    espn_injuries_df: pd.DataFrame,
    top5_usage_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For NBA only: join ESPN injuries with top-5 usage. Returns injuries where player is in top-5 usage for that team.
    Adds columns: is_out_or_doubtful, usage_pct, usage_rank.
    """
    if espn_injuries_df.empty or top5_usage_df.empty:
        return pd.DataFrame()
    nba_inj = espn_injuries_df[espn_injuries_df["league"] == "nba"].copy()
    if nba_inj.empty:
        return pd.DataFrame()
    nba_inj["_team_key"] = nba_inj["team_name"].str.strip().str.upper()
    nba_inj["_player_key"] = nba_inj["player_name"].str.strip().str.upper()
    top5 = top5_usage_df.copy()
    if "team_name" not in top5.columns:
        return pd.DataFrame()
    top5["_team_key"] = top5["team_name"].str.strip().str.upper()
    top5["_player_key"] = top5["PLAYER_NAME"].str.strip().str.upper()
    merged = nba_inj.merge(
        top5[["_team_key", "_player_key", "TEAM_ID", "PLAYER_ID", "PLAYER_NAME", "E_USG_PCT", "usage_rank"]],
        on=["_team_key", "_player_key"],
        how="inner",
    )
    merged = merged.drop(columns=["_team_key", "_player_key"], errors="ignore")
    merged["is_out_or_doubtful"] = merged["status"].apply(_is_out_or_doubtful)
    merged["usage_pct"] = merged["E_USG_PCT"]
    return merged


def flag_games_top5_out(
    games_df: pd.DataFrame,
    espn_injuries_df: pd.DataFrame,
    top5_usage_df: pd.DataFrame,
    league: str = "nba",
) -> pd.DataFrame:
    """
    Flag games where a top-5 usage player is out or doubtful.
    Expects games_df with home_team_name, away_team_name (and optionally game_id, game_date).
    Adds columns: top5_out_or_doubtful_home, top5_out_or_doubtful_away (bool).
    """
    if games_df.empty or league != "nba":
        g = games_df.copy()
        g["top5_out_or_doubtful_home"] = False
        g["top5_out_or_doubtful_away"] = False
        return g
    nba_usage_inj = nba_injuries_with_usage(espn_injuries_df, top5_usage_df)
    if nba_usage_inj.empty:
        g = games_df.copy()
        g["top5_out_or_doubtful_home"] = False
        g["top5_out_or_doubtful_away"] = False
        return g
    out_teams = set(
        nba_usage_inj[nba_usage_inj["is_out_or_doubtful"]]["team_name"].str.strip().str.upper()
    )
    g = games_df.copy()
    g["top5_out_or_doubtful_home"] = (g["home_team_name"].str.strip().str.upper().fillna("")).isin(out_teams)
    g["top5_out_or_doubtful_away"] = (g["away_team_name"].str.strip().str.upper().fillna("")).isin(out_teams)
    return g


def injury_impact_score(
    games_df: pd.DataFrame,
    espn_injuries_df: pd.DataFrame,
    top5_usage_df: pd.DataFrame,
    league: str = "nba",
) -> pd.Series:
    """
    Compute injury_impact_score per game (NBA): sum of (usage_pct/100) for top-5 players who are out or doubtful, for both teams; then take max(home_impact, away_impact) or average. Score in [0, 1+] — cap at 1.0 for interpretability.
    """
    if games_df.empty or league != "nba":
        return pd.Series(0.0, index=games_df.index)
    nba_usage_inj = nba_injuries_with_usage(espn_injuries_df, top5_usage_df)
    if nba_usage_inj.empty:
        return pd.Series(0.0, index=games_df.index)
    out = nba_usage_inj[nba_usage_inj["is_out_or_doubtful"]].copy()
    if out.empty:
        return pd.Series(0.0, index=games_df.index)
    out["impact"] = out["usage_pct"].fillna(0) / 100.0
    team_impact = out.groupby(out["team_name"].str.strip().str.upper())["impact"].sum()
    g = games_df.copy()
    home_impact = g["home_team_name"].str.strip().str.upper().map(team_impact).fillna(0)
    away_impact = g["away_team_name"].str.strip().str.upper().map(team_impact).fillna(0)
    score = (home_impact + away_impact) / 2.0  # or .max(axis=1) for max
    return score.clip(upper=1.0)


def add_injury_features(
    games_df: pd.DataFrame,
    league: str = "nba",
    nba_season: str = "2024-25",
) -> pd.DataFrame:
    """
    Add injury_impact_score and top-5 out/doubtful flags to a games DataFrame (feature matrix).
    Fetches current ESPN injuries and (for NBA) nba_api top-5 usage, then merges.
    """
    if games_df.empty:
        return games_df
    espn = fetch_espn_injuries_all()
    top5 = pd.DataFrame()
    if league == "nba":
        top5 = fetch_nba_top5_usage(season=nba_season)
    g = flag_games_top5_out(games_df, espn, top5, league=league)
    g["injury_impact_score"] = injury_impact_score(games_df, espn, top5, league=league)
    return g


def get_injury_impact_for_feature_matrix(
    feature_matrix: pd.DataFrame,
    league: str = "nba",
    nba_season: str = "2024-25",
) -> pd.DataFrame:
    """
    Add injury_impact_score and optional top5 flags to your feature matrix.
    feature_matrix must have home_team_name and away_team_name (and optionally game_id).
    Returns the same DataFrame with injury_impact_score, top5_out_or_doubtful_home, top5_out_or_doubtful_away.
    """
    return add_injury_features(feature_matrix.copy(), league=league, nba_season=nba_season)
