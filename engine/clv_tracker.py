"""
Closing Line Value (CLV) tracker. For every play the model recommends, record odds at
recommendation and closing odds just before tip-off. Positive CLV = got a better number
than the closing line. Stored in table clv_tracker in the existing SQLite database (espn.db).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


def _data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def _default_db_path() -> Path:
    """Existing app database (espn.db) where clv_tracker table lives."""
    return _data_dir() / "espn.db"


def _odds_snapshots_db_path() -> Path:
    """Database that contains odds_snapshots (odds.db). Used only when updating closing odds."""
    return _data_dir() / "odds.db"


def _implied_pct(odds_american: float) -> float:
    """Implied probability as percentage (0-100)."""
    if odds_american is None or (isinstance(odds_american, float) and pd.isna(odds_american)):
        return 0.0
    a = float(odds_american)
    if a >= 100:
        dec = 1.0 + a / 100.0
    elif a <= -100:
        dec = 1.0 + 100.0 / abs(a)
    else:
        return 0.0
    return (1.0 / dec * 100.0) if dec > 0 else 0.0


def _create_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS clv_tracker (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recorded_at TEXT NOT NULL,
            league TEXT NOT NULL,
            event_name TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            commence_time TEXT NOT NULL,
            market_type TEXT NOT NULL,
            selection TEXT NOT NULL,
            point REAL,
            odds_at_recommendation INTEGER,
            closing_odds INTEGER,
            clv_implied_pct REAL,
            UNIQUE(home_team, away_team, commence_time, market_type, selection, point)
        )
    """)


def record_recommendations(
    value_plays_df: pd.DataFrame,
    db_path: Path | None = None,
) -> int:
    """
    Record each recommended play with odds at time of recommendation.
    value_plays_df must have: League, Event, Selection, Market, Odds, home_team, away_team,
    commence_time, point (optional). Uses INSERT OR IGNORE so the first record per play is kept.
    Returns number of new rows inserted.
    """
    required = ["League", "Event", "Selection", "Market", "Odds", "home_team", "away_team", "commence_time"]
    if value_plays_df.empty or not all(c in value_plays_df.columns for c in required):
        return 0
    path = db_path or _default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    inserted = 0
    try:
        _create_table(conn)
        now = datetime.now(timezone.utc).isoformat()
        for _, row in value_plays_df.iterrows():
            league = str(row.get("League", "")).strip()
            event_name = str(row.get("Event", "")).strip()
            home_team = str(row.get("home_team", "")).strip()
            away_team = str(row.get("away_team", "")).strip()
            commence_time = str(row.get("commence_time", "")).strip()
            market_type = str(row.get("Market", "")).strip().lower() or "h2h"
            selection = str(row.get("Selection", "")).strip()
            point = row.get("point")
            if point is not None and not pd.isna(point):
                try:
                    point = float(point)
                except (TypeError, ValueError):
                    point = None
            odds = row.get("Odds")
            if odds is None or pd.isna(odds):
                continue
            try:
                odds = int(round(float(odds)))
            except (TypeError, ValueError):
                continue
            if abs(odds) < 100:
                continue
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO clv_tracker
                    (recorded_at, league, event_name, home_team, away_team, commence_time,
                     market_type, selection, point, odds_at_recommendation)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (now, league, event_name, home_team, away_team, commence_time, market_type, selection, point, odds),
                )
                if conn.total_changes > 0:
                    inserted += 1
            except sqlite3.IntegrityError:
                pass
        conn.commit()
    finally:
        conn.close()
    return inserted


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s or pd.isna(s):
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def update_closing_odds(
    db_path: Path | None = None,
    odds_snapshots_db_path: Path | None = None,
) -> int:
    """
    For each clv_tracker row with null closing_odds and commence_time in the past,
    set closing_odds from the latest odds_snapshot before tip-off. Set clv_implied_pct
    (closing_implied - recommendation_implied, in percentage points; positive = better than close).
    clv_tracker is in db_path (espn.db); odds_snapshots are read from odds_snapshots_db_path (odds.db).
    Returns number of rows updated.
    """
    path = db_path or _default_db_path()
    snap_path = odds_snapshots_db_path or _odds_snapshots_db_path()
    if not path.exists():
        return 0
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='clv_tracker'")
        if cur.fetchone() is None:
            return 0
        open_rows = pd.read_sql_query(
            """
            SELECT id, home_team, away_team, commence_time, market_type, selection, point, odds_at_recommendation
            FROM clv_tracker WHERE closing_odds IS NULL AND commence_time != ''
            """,
            conn,
        )
        if open_rows.empty:
            return 0
        if not snap_path.exists():
            return 0
        snap_conn = sqlite3.connect(snap_path)
        try:
            cur2 = snap_conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='odds_snapshots'")
            if cur2.fetchone() is None:
                return 0
            snap = pd.read_sql_query(
                "SELECT snapshot_at, home_team, away_team, commence_time, market_type, outcome, point, price FROM odds_snapshots",
                snap_conn,
            )
        finally:
            snap_conn.close()
        if snap.empty:
            return 0
        snap["_snap_dt"] = snap["snapshot_at"].apply(_parse_iso)
        snap["_commence_dt"] = snap["commence_time"].apply(_parse_iso)
        snap = snap.dropna(subset=["_snap_dt", "_commence_dt"])
        now_utc = datetime.now(timezone.utc)
        updated = 0
        for _, r in open_rows.iterrows():
            commence_time = str(r["commence_time"])
            commence_dt = _parse_iso(commence_time)
            if commence_dt is None or commence_dt > now_utc:
                continue
            home_team = str(r["home_team"]).strip()
            away_team = str(r["away_team"]).strip()
            market_type = str(r["market_type"]).strip().lower()
            selection = str(r["selection"]).strip()
            point = r["point"]
            mask = (
                (snap["home_team"].astype(str).str.strip() == home_team)
                & (snap["away_team"].astype(str).str.strip() == away_team)
                & (snap["commence_time"].astype(str) == commence_time)
                & (snap["market_type"].astype(str).str.strip().str.lower() == market_type)
                & (snap["_snap_dt"] <= commence_dt)
            )
            if point is not None and not pd.isna(point):
                mask = mask & (snap["point"].astype(float) == float(point))
            sel_match = snap["outcome"].astype(str).str.strip() == selection
            if not sel_match.any():
                sel_match = snap["outcome"].astype(str).str.strip().str.lower().str.contains(selection.lower(), na=False)
            mask = mask & sel_match
            closing = snap.loc[mask].sort_values("_snap_dt", ascending=False)
            if closing.empty:
                continue
            closing_odds = int(round(float(closing.iloc[0]["price"])))
            odds_rec = r["odds_at_recommendation"]
            impl_close = _implied_pct(closing_odds)
            impl_rec = _implied_pct(odds_rec)
            clv_implied_pct = round(impl_close - impl_rec, 4)
            conn.execute(
                "UPDATE clv_tracker SET closing_odds = ?, clv_implied_pct = ? WHERE id = ?",
                (closing_odds, clv_implied_pct, int(r["id"])),
            )
            updated += 1
        conn.commit()
    finally:
        conn.close()
    return updated


def get_clv_summary_last_30_days(db_path: Path | None = None) -> pd.DataFrame:
    """
    Returns DataFrame with columns: league, market_type (bet_type), avg_clv_pct, n_bets.
    Only rows with non-null closing_odds and recorded_at in the last 30 days.
    """
    path = db_path or _default_db_path()
    if not path.exists():
        return pd.DataFrame(columns=["league", "bet_type", "avg_clv_pct", "n_bets"])
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='clv_tracker'")
        if cur.fetchone() is None:
            return pd.DataFrame(columns=["league", "bet_type", "avg_clv_pct", "n_bets"])
        df = pd.read_sql_query(
            """
            SELECT league, market_type, clv_implied_pct, recorded_at
            FROM clv_tracker
            WHERE closing_odds IS NOT NULL AND clv_implied_pct IS NOT NULL
              AND recorded_at >= datetime('now', '-30 days')
            """,
            conn,
        )
    finally:
        conn.close()
    if df.empty:
        return pd.DataFrame(columns=["league", "bet_type", "avg_clv_pct", "n_bets"])
    summary = (
        df.groupby(["league", "market_type"], dropna=False)
        .agg(avg_clv_pct=("clv_implied_pct", "mean"), n_bets=("clv_implied_pct", "count"))
        .reset_index()
    )
    summary = summary.rename(columns={"market_type": "bet_type"})
    summary["avg_clv_pct"] = summary["avg_clv_pct"].round(2)
    return summary


def load_clv_tracker(league: Optional[str] = None, db_path: Path | None = None) -> pd.DataFrame:
    """Load full clv_tracker table (for debugging or export)."""
    path = db_path or _default_db_path()
    if not path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='clv_tracker'")
        if cur.fetchone() is None:
            return pd.DataFrame()
        if league:
            df = pd.read_sql_query("SELECT * FROM clv_tracker WHERE league = ?", conn, params=(league,))
        else:
            df = pd.read_sql_query("SELECT * FROM clv_tracker", conn)
        return df
    finally:
        conn.close()
