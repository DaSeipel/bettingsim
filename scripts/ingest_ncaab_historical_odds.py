#!/usr/bin/env python3
"""
Ingest NCAAB historical odds from a CSV (e.g. from Kaggle Robby Peery or similar) into data/odds.db.
Table: ncaab_historical_odds (date, home_team, away_team, closing_home_spread, closing_total, home_score, away_score).
CSV should have columns for date, home team, away team, and optionally spread/total/scores.
Usage: python scripts/ingest_ncaab_historical_odds.py path/to/file.csv
       Or set NCAAB_ODDS_CSV env var. Column names are normalized (date, game_date, home, home_team, away, away_team,
       spread, closing_spread, total, closing_total, home_score, away_score).
"""

from __future__ import annotations

import csv
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from engine.historical_odds import NCAAB_ESPN_TO_ODDS_API, _data_dir, _normalize_team

ODDS_DB = ROOT / "data" / "odds.db"


def _parse_float(x: str | None) -> float | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().replace(",", ".")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_date(s: str | None) -> str | None:
    """Normalize to YYYY-MM-DD."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = str(s).strip()
    if not s:
        return None
    # Try common formats
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d"):
        try:
            dt = pd.to_datetime(s, format=fmt)
            return dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    try:
        dt = pd.to_datetime(s)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def _normalize_csv_team(name: str) -> str:
    """Normalize team name for storage; optionally map from Odds API style to ESPN style for join."""
    n = _normalize_team(name)
    if not n:
        return n
    key = n.lower()
    for espn_key, odds_val in NCAAB_ESPN_TO_ODDS_API.items():
        if key == odds_val.lower() or key in odds_val.lower():
            return espn_key
    return n


def _infer_csv_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map standard names to actual column names (case-insensitive)."""
    cols_lower = {c.strip().lower(): c for c in df.columns}
    out = {}
    for std, candidates in [
        ("date", ["date", "game_date", "game date", "gm_date"]),
        ("home", ["home_team", "home", "home team", "home_team_name", "team_home"]),
        ("away", ["away_team", "away", "away team", "away_team_name", "team_away"]),
        ("spread", ["closing_home_spread", "closing_spread", "spread", "home_spread", "line"]),
        ("total", ["closing_total", "total", "over_under", "ou", "total_pts"]),
        ("home_score", ["home_score", "home_pts", "home points", "pts_home"]),
        ("away_score", ["away_score", "away_pts", "away points", "pts_away"]),
    ]:
        for c in candidates:
            if c in cols_lower:
                out[std] = cols_lower[c]
                break
    return out


def load_and_normalize_csv(path: Path) -> pd.DataFrame:
    """Load CSV and return DataFrame with columns: date, home_team, away_team, closing_home_spread, closing_total, home_score, away_score."""
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", low_memory=False)
    if df.empty:
        return df
    col_map = _infer_csv_columns(df)
    rows = []
    date_col = col_map.get("date")
    home_col = col_map.get("home")
    away_col = col_map.get("away")
    spread_col = col_map.get("spread")
    total_col = col_map.get("total")
    home_score_col = col_map.get("home_score")
    away_score_col = col_map.get("away_score")
    if not date_col or not home_col or not away_col:
        return pd.DataFrame()
    for _, r in df.iterrows():
        dt = _parse_date(r.get(date_col))
        if not dt:
            continue
        home = _normalize_csv_team(str(r.get(home_col, "")))
        away = _normalize_csv_team(str(r.get(away_col, "")))
        if not home or not away:
            continue
        spread = _parse_float(r.get(spread_col)) if spread_col else None
        total = _parse_float(r.get(total_col)) if total_col else None
        home_score = _parse_float(r.get(home_score_col)) if home_score_col else None
        away_score = _parse_float(r.get(away_score_col)) if away_score_col else None
        rows.append({
            "date": dt,
            "home_team": home,
            "away_team": away,
            "closing_home_spread": spread,
            "closing_total": total,
            "home_score": home_score,
            "away_score": away_score,
        })
    return pd.DataFrame(rows)


def create_ncaab_historical_odds_table(conn: sqlite3.Connection) -> None:
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


def ingest_ncaab_historical_odds(csv_path: Path, db_path: Path | None = None) -> int:
    """Load CSV and insert into ncaab_historical_odds. Returns count inserted."""
    path = db_path or ODDS_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    df = load_and_normalize_csv(csv_path)
    if df.empty:
        return 0
    conn = sqlite3.connect(path)
    try:
        create_ncaab_historical_odds_table(conn)
        cur = conn.cursor()
        n = 0
        for _, row in df.iterrows():
            try:
                cur.execute(
                    """INSERT OR REPLACE INTO ncaab_historical_odds
                       (date, home_team, away_team, closing_home_spread, closing_total, home_score, away_score)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        row["date"], row["home_team"], row["away_team"],
                        row.get("closing_home_spread"), row.get("closing_total"),
                        row.get("home_score"), row.get("away_score"),
                    ),
                )
                n += cur.rowcount
            except sqlite3.IntegrityError:
                pass
        conn.commit()
        return n
    finally:
        conn.close()


def main() -> int:
    import os
    csv_path = None
    for a in sys.argv[1:]:
        if not a.startswith("--") and Path(a).exists():
            csv_path = Path(a)
            break
    if not csv_path:
        csv_path = os.environ.get("NCAAB_ODDS_CSV")
        if csv_path:
            csv_path = Path(csv_path)
    if not csv_path or not csv_path.exists():
        print("Usage: python scripts/ingest_ncaab_historical_odds.py path/to/ncaab_odds.csv")
        print("Or set NCAAB_ODDS_CSV to the CSV path.")
        return 1
    n = ingest_ncaab_historical_odds(csv_path)
    print(f"Inserted/updated {n} rows into ncaab_historical_odds ({ODDS_DB})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
