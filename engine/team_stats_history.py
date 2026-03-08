"""
Team stats history: daily snapshots (e.g. from cleaned matchups CSV) stored in data/team_stats_history.csv.
Used for: (1) most-recent lookup at prediction time, (2) training labels (e.g. did High ROff predict winner?).
Schema: date (YYYY-MM-DD), Team, plus stat columns (SU, ATS, avg_points_for, ROff, RDef, last_5_wins, etc.).
Duplicate rule: one row per (Team, date); new snapshots overwrite same team+date.

Training-ready: For each game (game_date, home_team_name, away_team_name), join to history by taking
the most recent row per team where date <= game_date. That gives home_ROff, away_ROff, home_last_5_wins, etc.
You can then label outcomes (e.g. home_covered) and evaluate whether High ROff / last_5_wins predicted the winner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TEAM_STATS_HISTORY_PATH = ROOT / "data" / "team_stats_history.csv"

# Columns from cleaned matchups / snapshot CSVs that become training-ready features (home_* / away_* at lookup).
SNAPSHOT_STAT_COLUMNS = [
    "SU", "ATS", "avg_points_for", "avg_points_against",
    "avg_points_for_ha", "avg_points_against_ha",
    "ROff", "RDef", "last_5_wins",
]


def load_team_stats_history(path: Optional[Path] = None) -> pd.DataFrame:
    """Load team_stats_history.csv. Returns empty DataFrame if missing or invalid."""
    p = path or TEAM_STATS_HISTORY_PATH
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        if "date" not in df.columns or "Team" not in df.columns:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df = df.dropna(subset=["date"])
        return df
    except Exception:
        return pd.DataFrame()


def append_snapshot_to_history(
    snapshot_df: pd.DataFrame,
    snapshot_date: str,
    path: Optional[Path] = None,
) -> None:
    """
    Add a date column to snapshot_df, append to team_stats_history, and dedupe by (Team, date).
    snapshot_date: YYYY-MM-DD. Existing rows with same (Team, date) are replaced (last write wins).
    """
    if snapshot_df.empty or "Team" not in snapshot_df.columns:
        return
    p = path or TEAM_STATS_HISTORY_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    # Normalize date
    try:
        d = pd.to_datetime(snapshot_date).strftime("%Y-%m-%d")
    except Exception:
        d = str(snapshot_date).strip()[:10]
    added = snapshot_df.copy()
    added["date"] = d
    existing = load_team_stats_history(path=p)
    if existing.empty:
        combined = added
    else:
        # Drop existing rows that have (Team, date) in the new snapshot so we don't double-count
        key = existing["Team"].astype(str).str.strip() + "|" + existing["date"].astype(str)
        new_keys = set(added["Team"].astype(str).str.strip() + "|" + added["date"].astype(str))
        existing = existing.loc[~key.isin(new_keys)]
        combined = pd.concat([existing, added], ignore_index=True)
    # Dedupe: keep last per (Team, date)
    combined = combined.drop_duplicates(subset=["Team", "date"], keep="last")
    combined = combined.sort_values(["date", "Team"]).reset_index(drop=True)
    combined.to_csv(p, index=False)


def get_most_recent_team_stats(
    history_df: pd.DataFrame,
    team: str,
    as_of_date: str,
) -> Optional[pd.Series]:
    """
    Return the most recent stats row for team where row['date'] <= as_of_date.
    Team matching is case-insensitive strip; exact match preferred, then contains.
    """
    if history_df.empty or team is None or not str(team).strip():
        return None
    team_clean = str(team).strip()
    as_of = str(as_of_date).strip()[:10]
    try:
        history_df = history_df.copy()
        history_df["_date"] = pd.to_datetime(history_df["date"], errors="coerce")
        as_of_dt = pd.to_datetime(as_of, errors="coerce")
        if pd.isna(as_of_dt):
            return None
    except Exception:
        return None
    # Filter to this team (flexible match)
    team_norm = team_clean.lower()
    history_df["_team_norm"] = history_df["Team"].astype(str).str.strip().str.lower()
    mask = (history_df["_date"] <= as_of_dt) & (
        (history_df["_team_norm"] == team_norm)
        | (history_df["_team_norm"].str.contains(team_norm, regex=False, na=False))
    )
    # Also match if query is contained in a history team (e.g. "FGCU" in "Florida Gulf Coast" -> no; "Alabama" in "Alabama" -> yes)
    def _team_match(r: pd.Series) -> bool:
        hn = r.get("_team_norm", "") or ""
        return hn == team_norm or (team_norm in hn) or (hn in team_norm)
    mask = mask | (history_df["_date"] <= as_of_dt) & history_df.apply(_team_match, axis=1)
    subset = history_df.loc[mask]
    if subset.empty:
        return None
    exact = subset[subset["_team_norm"] == team_norm]
    if not exact.empty:
        subset = exact
    best_idx = subset["_date"].idxmax()
    row = subset.loc[best_idx].drop(labels=["_date", "_team_norm"], errors="ignore")
    return row


def get_most_recent_team_stats_resolved(
    history_df: pd.DataFrame,
    team: str,
    as_of_date: str,
    candidate_teams: Optional[list[str]] = None,
) -> Optional[pd.Series]:
    """
    Like get_most_recent_team_stats but resolve team name against candidate_teams (e.g. from history).
    If candidate_teams given, try to map 'team' to one of them (exact then normalized) and look up by that.
    """
    if candidate_teams:
        team_norm = str(team).strip().lower()
        for c in candidate_teams:
            if c.strip().lower() == team_norm:
                return get_most_recent_team_stats(history_df, c, as_of_date)
            if team_norm in c.strip().lower() or c.strip().lower() in team_norm:
                out = get_most_recent_team_stats(history_df, c, as_of_date)
                if out is not None:
                    return out
    return get_most_recent_team_stats(history_df, team, as_of_date)
