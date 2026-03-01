"""
Line movement tracking using The Odds API snapshots (stored on each odds fetch).
Computes: spread move since open, direction, velocity in last 6h, and sharp money indicator
(line moves opposite to public %). All features are merged into the feature matrix (games_with_team_stats)
and persisted to SQLite. Additive only: no existing feature columns are replaced.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from engine.engine import BASKETBALL_NBA, BASKETBALL_NCAAB

# Columns added to the feature matrix (games_with_team_stats). Use when building training features.
LINE_MOVEMENT_FEATURE_COLUMNS = [
    "line_move_direction",
    "line_move_magnitude",
    "line_move_velocity_6h",
    "sharp_money_indicator",
]

# Odds DB (snapshots) default path
def _odds_db_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "odds.db"


def _league_to_sport_key(league: str) -> str:
    s = str(league).strip().lower()
    if s == "nba":
        return BASKETBALL_NBA
    if s == "ncaab":
        return BASKETBALL_NCAAB
    return s


def _parse_utc(s: str) -> Optional[datetime]:
    if not s or pd.isna(s):
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _team_match(a: str, b: str) -> bool:
    """True if team names refer to the same team (exact or substring)."""
    if pd.isna(a) or pd.isna(b):
        return False
    x, y = str(a).strip(), str(b).strip()
    if not x or not y:
        return False
    if x == y:
        return True
    if x in y or y in x:
        return True
    return False


def _load_snapshots(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='odds_snapshots'"
        )
        if cur.fetchone() is None:
            return pd.DataFrame()
        df = pd.read_sql_query(
            "SELECT snapshot_at, sport_key, game_id, home_team, away_team, commence_time, "
            "bookmaker, market_type, outcome, price, point FROM odds_snapshots WHERE market_type = 'spreads'",
            conn,
        )
        return df
    except sqlite3.OperationalError:
        return pd.DataFrame()
    finally:
        conn.close()


def _home_spread_per_snapshot(snap: pd.DataFrame) -> pd.DataFrame:
    """
    From spreads snapshots (one row per bookmaker/outcome), get one home spread per (game_id, commence_time, snapshot_at).
    Home spread = point where outcome == home_team; aggregate across bookmakers (median).
    """
    if snap.empty or "point" not in snap.columns:
        return pd.DataFrame()
    rows = []
    for key, grp in snap.groupby(["sport_key", "game_id", "commence_time", "snapshot_at"]):
        sport_key, game_id, commence_time, snapshot_at = key
        home_team = grp["home_team"].iloc[0]
        away_team = grp["away_team"].iloc[0]
        # Home outcome row: outcome equals home_team
        home_rows = grp[grp["outcome"].apply(lambda o: _team_match(o, home_team))]
        if home_rows.empty:
            continue
        point = home_rows["point"].median()
        rows.append({
            "sport_key": sport_key,
            "game_id": game_id,
            "commence_time": commence_time,
            "snapshot_at": snapshot_at,
            "home_team": home_team,
            "away_team": away_team,
            "home_spread": point,
        })
    return pd.DataFrame(rows)


def _open_current_6h(spread_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (sport_key, game_id, commence_time), compute open_spread, current_spread, spread_6h_ago.
    Open = home_spread at earliest snapshot_at; current = latest; 6h ago = latest snapshot_at <= commence_time - 6h.
    """
    if spread_df.empty:
        return pd.DataFrame()
    spread_df = spread_df.copy()
    spread_df["_commence_utc"] = spread_df["commence_time"].apply(_parse_utc)
    spread_df["_snap_utc"] = spread_df["snapshot_at"].apply(_parse_utc)
    spread_df = spread_df.dropna(subset=["_commence_utc", "_snap_utc"])
    if spread_df.empty:
        return pd.DataFrame()
    spread_df["_cut_6h"] = spread_df["_commence_utc"] - timedelta(hours=6)
    out = []
    for key, grp in spread_df.groupby(["sport_key", "game_id", "commence_time", "home_team", "away_team"]):
        sport_key, game_id, commence_time, home_team, away_team = key
        grp = grp.sort_values("_snap_utc")
        snap_times = grp["_snap_utc"]
        open_spread = grp.iloc[0]["home_spread"]
        current_spread = grp.iloc[-1]["home_spread"]
        commence_utc = grp["_commence_utc"].iloc[0]
        cut_6h = commence_utc - timedelta(hours=6)
        # Latest snapshot at or before 6h before commence
        past = grp[grp["_snap_utc"] <= cut_6h]
        if past.empty:
            spread_6h_ago = None
        else:
            spread_6h_ago = past.iloc[-1]["home_spread"]
        out.append({
            "sport_key": sport_key,
            "game_id": game_id,
            "commence_time": commence_time,
            "commence_date": commence_utc.date() if commence_utc else None,
            "home_team": home_team,
            "away_team": away_team,
            "open_spread": open_spread,
            "current_spread": current_spread,
            "spread_6h_ago": spread_6h_ago,
        })
    return pd.DataFrame(out)


def compute_line_movement_features(
    games_df: pd.DataFrame,
    odds_db_path: Path | None = None,
    public_pct_home: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    For each game in games_df (league, game_id, game_date, home_team_name, away_team_name),
    compute line_move_direction (+1 = toward home, -1 = toward away, 0 = no move),
    line_move_magnitude (abs move in points), line_move_velocity_6h (move in last 6h in points),
    and sharp_money_indicator (True if line moved opposite to public %).
    public_pct_home: optional Series indexed like games_df (0-1 fraction on home). If not provided, sharp_money_indicator is False.
    Returns DataFrame with columns: league, game_id, line_move_direction, line_move_magnitude, line_move_velocity_6h, sharp_money_indicator.
    """
    if games_df.empty or "league" not in games_df.columns or "game_id" not in games_df.columns:
        return pd.DataFrame(columns=["league", "game_id", "line_move_direction", "line_move_magnitude", "line_move_velocity_6h", "sharp_money_indicator"])
    path = odds_db_path or _odds_db_path()
    snap = _load_snapshots(path)
    if snap.empty:
        return pd.DataFrame(columns=["league", "game_id", "line_move_direction", "line_move_magnitude", "line_move_velocity_6h", "sharp_money_indicator"])
    spread_per_snap = _home_spread_per_snapshot(snap)
    if spread_per_snap.empty:
        return pd.DataFrame(columns=["league", "game_id", "line_move_direction", "line_move_magnitude", "line_move_velocity_6h", "sharp_money_indicator"])
    moves = _open_current_6h(spread_per_snap)
    if moves.empty:
        return pd.DataFrame(columns=["league", "game_id", "line_move_direction", "line_move_magnitude", "line_move_velocity_6h", "sharp_money_indicator"])
    # Line move from open to current
    moves["line_move_magnitude"] = (moves["current_spread"] - moves["open_spread"]).abs()
    delta = moves["current_spread"] - moves["open_spread"]
    moves["line_move_direction"] = 0
    moves.loc[delta > 0, "line_move_direction"] = 1   # toward home
    moves.loc[delta < 0, "line_move_direction"] = -1  # toward away
    moves["line_move_velocity_6h"] = None
    mask_6h = moves["spread_6h_ago"].notna()
    moves.loc[mask_6h, "line_move_velocity_6h"] = moves.loc[mask_6h, "current_spread"] - moves.loc[mask_6h, "spread_6h_ago"]
    # Sharp: line moved opposite to public. Public on home (public_pct_home > 0.5) and line moved toward away (direction -1) => sharp. Public on away and line moved toward home (direction 1) => sharp.
    moves["sharp_money_indicator"] = False
    # We'll set sharp when we merge to games and have public_pct_home
    # For now keep False; set below per game if public_pct_home provided
    # Build key for merge: (sport_key, home_team, away_team, commence_date)
    moves = moves.dropna(subset=["commence_date"])
    moves["commence_date"] = pd.to_datetime(moves["commence_date"]).dt.normalize()
    # Merge to games: games have league, game_id, game_date, home_team_name, away_team_name
    g = games_df[["league", "game_id", "game_date", "home_team_name", "away_team_name"]].copy()
    g["_sport_key"] = g["league"].apply(_league_to_sport_key)
    g["_game_date"] = pd.to_datetime(g["game_date"], errors="coerce").dt.normalize()
    # Match games to moves by (sport_key, home, away, date). Multiple games same day same teams: take first move row.
    move_cols = ["sport_key", "home_team", "away_team", "commence_date", "line_move_direction", "line_move_magnitude", "line_move_velocity_6h", "sharp_money_indicator"]
    merged = []
    for idx, row in g.iterrows():
        sk = row["_sport_key"]
        ht = str(row["home_team_name"]).strip()
        at = str(row["away_team_name"]).strip()
        gd = row["_game_date"]
        if pd.isna(gd):
            merged.append({"league": row["league"], "game_id": row["game_id"], "line_move_direction": None, "line_move_magnitude": None, "line_move_velocity_6h": None, "sharp_money_indicator": False})
            continue
        cand = moves[
            (moves["sport_key"] == sk) &
            (moves["commence_date"] == gd) &
            (moves["home_team"].apply(lambda x: _team_match(x, ht))) &
            (moves["away_team"].apply(lambda x: _team_match(x, at)))
        ]
        if cand.empty:
            merged.append({"league": row["league"], "game_id": row["game_id"], "line_move_direction": None, "line_move_magnitude": None, "line_move_velocity_6h": None, "sharp_money_indicator": False})
            continue
        m = cand.iloc[0]
        sharp = m["sharp_money_indicator"]
        if public_pct_home is not None and idx in public_pct_home.index:
            pct = public_pct_home.loc[idx]
            if pd.notna(pct) and 0 <= pct <= 1:
                # Line moved toward away (dir -1) and public on home (pct > 0.5) => sharp. Line moved toward home (dir 1) and public on away (pct < 0.5) => sharp.
                sharp = (m["line_move_direction"] == -1 and pct > 0.5) or (m["line_move_direction"] == 1 and pct < 0.5)
        merged.append({
            "league": row["league"],
            "game_id": row["game_id"],
            "line_move_direction": int(m["line_move_direction"]) if pd.notna(m["line_move_direction"]) else None,
            "line_move_magnitude": float(m["line_move_magnitude"]) if pd.notna(m["line_move_magnitude"]) else None,
            "line_move_velocity_6h": float(m["line_move_velocity_6h"]) if pd.notna(m["line_move_velocity_6h"]) else None,
            "sharp_money_indicator": bool(sharp),
        })
    out = pd.DataFrame(merged)
    return out[["league", "game_id", "line_move_direction", "line_move_magnitude", "line_move_velocity_6h", "sharp_money_indicator"]]


def merge_line_movement_into_feature_matrix(
    feature_matrix: pd.DataFrame,
    odds_db_path: Path | None = None,
    public_pct_home: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Add line movement features to the feature matrix. Merges on (league, game_id).
    Drops existing line_move_* and sharp_money_indicator columns if present, then merges.
    """
    if feature_matrix.empty:
        return feature_matrix
    needed = ["league", "game_id", "game_date", "home_team_name", "away_team_name"]
    if not all(c in feature_matrix.columns for c in needed):
        return feature_matrix
    fm = feature_matrix.drop(columns=[c for c in LINE_MOVEMENT_FEATURE_COLUMNS if c in feature_matrix.columns], errors="ignore")
    lm = compute_line_movement_features(fm, odds_db_path=odds_db_path, public_pct_home=public_pct_home)
    if lm.empty:
        fm["line_move_direction"] = None
        fm["line_move_magnitude"] = None
        fm["line_move_velocity_6h"] = None
        fm["sharp_money_indicator"] = False
        return fm
    merge_cols = ["league", "game_id"] + [c for c in LINE_MOVEMENT_FEATURE_COLUMNS if c in lm.columns]
    fm = fm.merge(lm[merge_cols], on=["league", "game_id"], how="left")
    return fm
