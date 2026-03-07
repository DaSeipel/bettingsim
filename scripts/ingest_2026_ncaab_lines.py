#!/usr/bin/env python3
"""
Ingest data/ncaab/2026ncaablines.csv into historical_covers_odds in data/odds.db.
Maps: StartDate→date, HomeTeam→home_team, AwayTeam→away_team, Spread→closing_spread,
OverUnder→closing_total, HomeScore→home_score, AwayScore→away_score,
HomeMoneyline→closing_ml_home, AwayMoneyline→closing_ml_away, OpeningSpread→opening_spread.
Adds sport='ncaab'.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

ODDS_DB = ROOT / "data" / "odds.db"
CSV_PATH = ROOT / "data" / "ncaab" / "2026ncaablines.csv"


def _ensure_columns(conn: sqlite3.Connection) -> None:
    cur = conn.execute("PRAGMA table_info(historical_covers_odds)")
    existing = {row[1] for row in cur.fetchall()}
    for col, typ in [("sport", "TEXT"), ("closing_ml_home", "REAL"), ("closing_ml_away", "REAL")]:
        if col not in existing:
            conn.execute(f"ALTER TABLE historical_covers_odds ADD COLUMN {col} {typ}")
    conn.commit()


def main() -> int:
    if not CSV_PATH.exists():
        print(f"Not found: {CSV_PATH}")
        return 1
    df = pd.read_csv(CSV_PATH)
    # Map columns
    df = df.rename(columns={
        "StartDate": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "Spread": "closing_spread",
        "OverUnder": "closing_total",
        "HomeScore": "home_score",
        "AwayScore": "away_score",
        "HomeMoneyline": "closing_ml_home",
        "AwayMoneyline": "closing_ml_away",
        "OpeningSpread": "opening_spread",
    })
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date", "home_team", "away_team"])
    df["sport"] = "ncaab"
    for col in ["opening_spread", "closing_spread", "closing_total", "home_score", "away_score", "closing_ml_home", "closing_ml_away"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "OpeningOverUnder" in df.columns:
        df["opening_total"] = pd.to_numeric(df["OpeningOverUnder"], errors="coerce")
    else:
        df["opening_total"] = None
    if "Season" in df.columns:
        df["season"] = df["Season"].astype(str)
    else:
        df["season"] = None

    cols = ["date", "home_team", "away_team", "opening_spread", "closing_spread", "opening_total", "closing_total", "away_score", "home_score", "season", "sport", "closing_ml_home", "closing_ml_away"]
    # Use only columns that exist in our df
    df_out = df[[c for c in cols if c in df.columns]].copy()

    ODDS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(ODDS_DB)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS historical_covers_odds (
                date TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                opening_spread REAL,
                closing_spread REAL,
                opening_total REAL,
                closing_total REAL,
                away_score REAL,
                home_score REAL,
                season TEXT,
                sport TEXT,
                closing_ml_home REAL,
                closing_ml_away REAL,
                PRIMARY KEY (date, home_team, away_team)
            )
        """)
        _ensure_columns(conn)
        cur = conn.cursor()
        n = 0
        for _, row in df_out.iterrows():
            try:
                cur.execute(
                    """INSERT OR REPLACE INTO historical_covers_odds
                       (date, home_team, away_team, opening_spread, closing_spread, opening_total, closing_total, away_score, home_score, season, sport, closing_ml_home, closing_ml_away)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(row["date"]), str(row["home_team"]), str(row["away_team"]),
                        row.get("opening_spread") if pd.notna(row.get("opening_spread")) else None,
                        row.get("closing_spread") if pd.notna(row.get("closing_spread")) else None,
                        row.get("opening_total") if pd.notna(row.get("opening_total")) else None,
                        row.get("closing_total") if pd.notna(row.get("closing_total")) else None,
                        row.get("away_score") if pd.notna(row.get("away_score")) else None,
                        row.get("home_score") if pd.notna(row.get("home_score")) else None,
                        str(row["season"]) if pd.notna(row.get("season")) else None,
                        "ncaab",
                        row.get("closing_ml_home") if pd.notna(row.get("closing_ml_home")) else None,
                        row.get("closing_ml_away") if pd.notna(row.get("closing_ml_away")) else None,
                    ),
                )
                n += cur.rowcount
            except sqlite3.IntegrityError:
                pass
        conn.commit()
        print(f"Ingested {n} rows into historical_covers_odds (sport=ncaab) from {CSV_PATH.name}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
