"""
Play history archive. Saves every value play generated each day to play_history table
before games start, so no play is lost after the dashboard refreshes.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


def _data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def _default_db_path() -> Path:
    return _data_dir() / "espn.db"


def _create_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS play_history (
            play_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date_generated TEXT NOT NULL,
            sport TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            bet_type TEXT NOT NULL,
            recommended_side TEXT NOT NULL,
            spread_or_total REAL,
            my_edge_pct REAL NOT NULL,
            my_probability REAL NOT NULL,
            market_odds_at_time INTEGER NOT NULL,
            recommended_stake REAL,
            confidence_tier TEXT NOT NULL,
            reasoning_summary TEXT,
            result TEXT,
            actual_payout REAL,
            UNIQUE(date_generated, sport, home_team, away_team, bet_type, recommended_side, spread_or_total)
        )
    """)
    _ensure_result_columns(conn)


def _ensure_result_columns(conn: sqlite3.Connection) -> None:
    """Add recommended_stake, result, actual_payout, manual_review_flag if missing (migration)."""
    cur = conn.execute("PRAGMA table_info(play_history)")
    cols = [row[1] for row in cur.fetchall()]
    if "recommended_stake" not in cols:
        conn.execute("ALTER TABLE play_history ADD COLUMN recommended_stake REAL")
    if "result" not in cols:
        conn.execute("ALTER TABLE play_history ADD COLUMN result TEXT")
    if "actual_payout" not in cols:
        conn.execute("ALTER TABLE play_history ADD COLUMN actual_payout REAL")
    if "manual_review_flag" not in cols:
        conn.execute("ALTER TABLE play_history ADD COLUMN manual_review_flag INTEGER DEFAULT 0")


def archive_value_plays(
    value_plays_df: pd.DataFrame,
    db_path: Path | None = None,
    as_of_date: date | None = None,
) -> int:
    """
    Insert today's value plays into play_history. Uses INSERT OR REPLACE so re-archiving
    updates existing rows. value_plays_df must have columns matching the table (or we map
    from standard names: League, Event, Selection, Market, point, Value (%), model_prob,
    Odds, confidence_tier, reasoning_summary).
    Returns number of rows inserted/replaced.
    """
    if value_plays_df.empty:
        return 0
    path = db_path or _default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    as_of_date = as_of_date or date.today()
    date_str = as_of_date.isoformat()

    required = ["League", "Event", "Selection", "Market", "Odds", "Value (%)", "model_prob", "home_team", "away_team"]
    if not all(c in value_plays_df.columns for c in required):
        return 0

    conn = sqlite3.connect(path)
    n = 0
    try:
        _create_table(conn)
        for _, row in value_plays_df.iterrows():
            sport = str(row.get("League", "")).strip() or "—"
            home_team = str(row.get("home_team", "")).strip() or "—"
            away_team = str(row.get("away_team", "")).strip() or "—"
            market = str(row.get("Market", "")).strip().lower() or "h2h"
            bet_type = {"h2h": "Moneyline", "spreads": "Spread", "totals": "Over/Under"}.get(market, market)
            recommended_side = str(row.get("Selection", "")).strip() or "—"
            # Always store team name, not "Home"/"Away", so cards show correct pick (e.g. Pennsylvania +9.5 not YALE +9.5)
            if recommended_side.strip().lower() == "home" and home_team:
                recommended_side = home_team
            elif recommended_side.strip().lower() == "away" and away_team:
                recommended_side = away_team
            point = row.get("point")
            if point is not None and not pd.isna(point):
                try:
                    spread_or_total = float(point)
                    # NCAAB Spread: store line from picked side's perspective (away pick with negative = home spread stored by mistake → flip)
                    if sport == "NCAAB" and bet_type == "Spread" and spread_or_total < 0:
                        sel, ht, at = recommended_side.strip().lower(), home_team.strip().lower(), away_team.strip().lower()
                        if sel and at and (sel == at or sel in at or at in sel):
                            spread_or_total = -spread_or_total
                except (TypeError, ValueError):
                    spread_or_total = None
            else:
                spread_or_total = None
            # Use -999 for UNIQUE key when no line (SQLite treats NULLs as distinct)
            spread_or_total_key = spread_or_total if spread_or_total is not None else -999.0
            my_edge_pct = float(row.get("Value (%)", 0))
            my_probability = float(row.get("model_prob", 0))
            market_odds_at_time = int(round(float(row.get("Odds", 0))))
            confidence_tier = str(row.get("confidence_tier", "Medium")).strip() or "Medium"
            reasoning_summary = row.get("reasoning_summary")
            if reasoning_summary is not None and not pd.isna(reasoning_summary):
                reasoning_summary = str(reasoning_summary).strip() or None
            else:
                reasoning_summary = None
            recommended_stake = row.get("Recommended Stake")
            if recommended_stake is not None and not pd.isna(recommended_stake):
                try:
                    recommended_stake = float(recommended_stake)
                except (TypeError, ValueError):
                    recommended_stake = None
            else:
                recommended_stake = None
            conn.execute(
                """
                INSERT INTO play_history
                (date_generated, sport, home_team, away_team, bet_type, recommended_side,
                 spread_or_total, my_edge_pct, my_probability, market_odds_at_time,
                 recommended_stake, confidence_tier, reasoning_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date_generated, sport, home_team, away_team, bet_type, recommended_side, spread_or_total)
                DO UPDATE SET
                    my_edge_pct = excluded.my_edge_pct,
                    my_probability = excluded.my_probability,
                    market_odds_at_time = excluded.market_odds_at_time,
                    recommended_stake = excluded.recommended_stake,
                    confidence_tier = excluded.confidence_tier,
                    reasoning_summary = excluded.reasoning_summary
                """,
                (
                    date_str,
                    sport,
                    home_team,
                    away_team,
                    bet_type,
                    recommended_side,
                    spread_or_total_key,
                    my_edge_pct,
                    my_probability,
                    market_odds_at_time,
                    recommended_stake,
                    confidence_tier,
                    reasoning_summary,
                ),
            )
            n += 1
        conn.commit()
    finally:
        conn.close()
    return n


def _american_to_decimal(american: float) -> float:
    a = float(american)
    if a >= 100:
        return 1.0 + a / 100.0
    if a <= -100:
        return 1.0 + 100.0 / abs(a)
    return 1.0


def update_play_result(
    play_id: int,
    result: str,
    db_path: Path | None = None,
) -> bool:
    """
    Set result for a play to W, L, or P. Computes actual_payout from recommended_stake and odds.
    Returns True if a row was updated.
    """
    if result not in ("W", "L", "P"):
        return False
    path = db_path or _default_db_path()
    if not path.exists():
        return False
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute(
            "SELECT recommended_stake, market_odds_at_time FROM play_history WHERE play_id = ?",
            (play_id,),
        )
        row = cur.fetchone()
        if row is None:
            return False
        stake, odds_american = row[0], row[1]
        try:
            stake = float(stake) if stake is not None and str(stake).strip() != "" and not (isinstance(stake, float) and pd.isna(stake)) else 0.0
        except (TypeError, ValueError):
            stake = 0.0
        if result == "P":
            actual_payout = 0.0
        elif result == "W":
            try:
                dec = _american_to_decimal(float(odds_american))
                actual_payout = stake * (dec - 1.0) if stake else 0.0
            except (TypeError, ValueError):
                actual_payout = 0.0
        else:
            actual_payout = -stake if stake else 0.0
        cur = conn.execute(
            "UPDATE play_history SET result = ?, actual_payout = ? WHERE play_id = ?",
            (result, round(actual_payout, 2), play_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_play(play_id: int, db_path: Path | None = None) -> Optional[dict]:
    """
    Permanently delete a play from play_history. Returns the deleted row as dict
    (date_generated, home_team, away_team, recommended_side, spread_or_total) for use in
    removing from historical_betting_performance.csv, or None if not found.
    """
    path = db_path or _default_db_path()
    if not path.exists():
        return None
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute(
            "SELECT date_generated, home_team, away_team, recommended_side, spread_or_total FROM play_history WHERE play_id = ?",
            (play_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        out = {
            "date_generated": row[0],
            "home_team": row[1],
            "away_team": row[2],
            "recommended_side": row[3],
            "spread_or_total": row[4],
        }
        cur = conn.execute("DELETE FROM play_history WHERE play_id = ?", (play_id,))
        conn.commit()
        return out if cur.rowcount > 0 else None
    finally:
        conn.close()


def set_manual_review_flag(play_id: int, flag: bool = True, db_path: Path | None = None) -> bool:
    """Set or clear manual_review_flag for a play (e.g. when ESPN could not resolve result)."""
    path = db_path or _default_db_path()
    if not path.exists():
        return False
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute(
            "UPDATE play_history SET manual_review_flag = ? WHERE play_id = ?",
            (1 if flag else 0, play_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def load_play_history(
    league: Optional[str] = None,
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Load play_history with optional filters. Returns DataFrame with all columns."""
    path = db_path or _default_db_path()
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "play_id", "date_generated", "sport", "home_team", "away_team", "bet_type",
                "recommended_side", "spread_or_total", "my_edge_pct", "my_probability",
                "market_odds_at_time", "recommended_stake", "confidence_tier", "reasoning_summary",
                "result", "actual_payout", "manual_review_flag",
            ]
        )
    conn = sqlite3.connect(path)
    try:
        _create_table(conn)
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='play_history'")
        if cur.fetchone() is None:
            return pd.DataFrame(
                columns=[
                    "play_id", "date_generated", "sport", "home_team", "away_team", "bet_type",
                    "recommended_side", "spread_or_total", "my_edge_pct", "my_probability",
                    "market_odds_at_time", "recommended_stake", "confidence_tier", "reasoning_summary",
                    "result", "actual_payout", "manual_review_flag",
                ]
            )
        q = "SELECT play_id, date_generated, sport, home_team, away_team, bet_type, recommended_side, spread_or_total, my_edge_pct, my_probability, market_odds_at_time, recommended_stake, confidence_tier, reasoning_summary, result, actual_payout, COALESCE(manual_review_flag, 0) AS manual_review_flag FROM play_history WHERE 1=1"
        params: list = []
        if league:
            q += " AND sport = ?"
            params.append(league.strip())
        if from_date:
            q += " AND date_generated >= ?"
            params.append(from_date.isoformat())
        if to_date:
            q += " AND date_generated <= ?"
            params.append(to_date.isoformat())
        q += " ORDER BY date_generated DESC, play_id DESC"
        df = pd.read_sql_query(q, conn, params=params if params else None)
        return df
    finally:
        conn.close()


def import_csv_tournament_picks(
    csv_path: Path | None = None,
    db_path: Path | None = None,
    min_date: str = "2026-03-19",
) -> int:
    """
    Import NCAA Tournament / NIT picks from historical_betting_performance.csv
    into play_history SQLite. Maps CSV schema to SQLite schema.
    Uses sport='NCAAB Tournament' so they can be distinguished from pipeline picks.
    Skips rows already present (INSERT OR IGNORE via UNIQUE constraint).
    Returns number of rows inserted.
    """
    data_dir = _data_dir()
    csv_path = csv_path or data_dir / "historical_betting_performance.csv"
    db_path = db_path or _default_db_path()
    if not csv_path.exists():
        return 0

    df = pd.read_csv(csv_path)
    df = df[df["Date"] >= min_date].copy()
    if df.empty:
        return 0

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    n = 0
    try:
        _create_table(conn)
        for _, row in df.iterrows():
            date_str = str(row["Date"]).strip()
            home = str(row.get("Home", "")).strip()
            away = str(row.get("Away", "")).strip()
            spread = float(row.get("Market_Spread", 0))
            pick_dir = str(row.get("Pick_Spread", "")).strip()
            conf = str(row.get("Confidence_Level", "Medium")).strip()
            edge = float(row.get("Edge_Points", 0) or 0)
            prob = float(row.get("Spread_Prob", 0.5) or 0.5)
            stake_pct = row.get("Suggested_Stake_Pct")
            try:
                stake = float(stake_pct) if stake_pct is not None and not pd.isna(stake_pct) else 1.0
            except (TypeError, ValueError):
                stake = 1.0

            if pick_dir == "Home":
                recommended_side = home
                line = spread
            elif pick_dir == "Away":
                recommended_side = away
                line = abs(spread)
            else:
                continue

            ats = row.get("ATS_Result")
            result_map = {"Win": "W", "Loss": "L", "Push": "P"}
            result = result_map.get(str(ats).strip()) if ats is not None and not pd.isna(ats) else None

            payout = None
            if result == "W":
                payout = round(stake * 0.909, 2)
            elif result == "L":
                payout = round(-stake, 2)
            elif result == "P":
                payout = 0.0

            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO play_history
                    (date_generated, sport, home_team, away_team, bet_type, recommended_side,
                     spread_or_total, my_edge_pct, my_probability, market_odds_at_time,
                     recommended_stake, confidence_tier, reasoning_summary, result, actual_payout)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        date_str,
                        "NCAAB Tournament",
                        home,
                        away,
                        "Spread",
                        recommended_side,
                        line,
                        abs(edge),
                        prob,
                        -110,
                        stake,
                        conf,
                        "NCAA Tournament" if date_str >= "2026-03-19" else "NIT",
                        result,
                        payout,
                    ),
                )
                if conn.total_changes:
                    n += 1
            except sqlite3.IntegrityError:
                pass
        conn.commit()
    finally:
        conn.close()
    return n
