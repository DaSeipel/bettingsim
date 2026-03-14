#!/usr/bin/env python3
"""
Update ncaab_team_season_stats with March multiplier columns and backfill 2026 data.

1. Schema: Ensures free_throw_pct, three_point_pct, roster_experience_years exist.
2. Data: Maps FTR -> free_throw_pct, 3P_O -> three_point_pct; fills roster_experience_years
   from a Top-50 lookup (KenPom/BartTorvik-style experience) so Veteran Edge has data.
3. Can merge 2026 from data/ncaab/team_stats_2026.csv if DB has no 2026 rows.

Run after fetch_barttorvik_2026.py. Then run scripts/audit_march_data.py to validate.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "espn.db"
TEAM_STATS_2026_CSV = ROOT / "data" / "ncaab" / "team_stats_2026.csv"
TEAM_STATS_COMBINED_CSV = ROOT / "data" / "ncaab" / "team_stats_combined.csv"

MARCH_COLUMNS = ["free_throw_pct", "three_point_pct", "roster_experience_years"]

# Top ~50 teams (likely tournament field): roster_experience_years from KenPom/BartTorvik
# experience rankings. Placeholder values (1.5–2.8) so Veteran Edge has data; replace with
# real experience data when available.
ROSTER_EXPERIENCE_LOOKUP_2026: dict[str, float] = {
    "Purdue": 2.4, "UConn": 2.2, "Houston": 2.6, "North Carolina": 2.3, "Tennessee": 2.1,
    "Arizona": 2.5, "Kansas": 2.4, "Duke": 1.8, "Marquette": 2.3, "Iowa State": 2.2,
    "Baylor": 2.0, "Illinois": 2.4, "Kentucky": 1.6, "Alabama": 2.1, "Creighton": 2.5,
    "Gonzaga": 2.6, "San Diego State": 2.7, "Wisconsin": 2.3, "Michigan State": 2.4,
    "Texas": 2.0, "Villanova": 2.2, "Florida": 2.1, "Texas Tech": 2.0, "BYU": 2.3,
    "Auburn": 2.2, "Maryland": 2.1, "St. Mary's": 2.5, "Colorado": 2.2, "Virginia": 2.6,
    "Oklahoma": 1.9, "Nebraska": 2.0, "Mississippi State": 2.1, "South Carolina": 2.0,
    "New Mexico": 2.2, "Clemson": 2.3, "Utah State": 2.4, "Florida Atlantic": 2.2,
    "Northwestern": 2.3, "Michigan": 1.9, "Indiana": 2.0, "Ohio State": 2.1,
    "Oregon": 2.2, "Pittsburgh": 2.0, "Seton Hall": 2.3, "Providence": 2.2,
    "Xavier": 2.1, "Butler": 2.2, "St. John's": 1.9, "Boise State": 2.3,
    "Memphis": 2.0, "Washington State": 2.1, "Nevada": 2.2, "Dayton": 2.4,
    "Grand Canyon": 2.3, "Drake": 2.5, "James Madison": 2.0, "Yale": 2.4,
}


def _normalize_team_name(name: str) -> str:
    return str(name or "").strip()


def _ensure_march_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add March columns if missing; return DataFrame with all three."""
    out = df.copy()
    for col in MARCH_COLUMNS:
        if col not in out.columns:
            out[col] = float("nan")
    return out


def _backfill_march_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fill free_throw_pct from FTR, three_point_pct from 3P_O, roster_experience_years from lookup (2026 top 50)."""
    out = df.copy()
    if "FTR" in out.columns:
        out["free_throw_pct"] = out["FTR"].apply(
            lambda x: float(x) * 100.0 if pd.notna(x) and float(x) <= 1 else (float(x) if pd.notna(x) else float("nan"))
        )
    if "3P_O" in out.columns:
        out["three_point_pct"] = out["3P_O"].apply(
            lambda x: float(x) * 100.0 if pd.notna(x) and float(x) <= 1 else (float(x) if pd.notna(x) else float("nan"))
        )
    if "season" in out.columns and "TEAM" in out.columns:
        df_2026 = out[out["season"].astype(int) == 2026]
        if "BARTHAG" in out.columns and not df_2026.empty:
            top50_teams = set(df_2026.nlargest(50, "BARTHAG")["TEAM"].apply(_normalize_team_name).tolist())
        else:
            top50_teams = set(df_2026["TEAM"].apply(_normalize_team_name).head(50).tolist())

        def exp_val(row):
            if int(row.get("season", 0)) != 2026:
                return row.get("roster_experience_years", float("nan"))
            team = _normalize_team_name(row.get("TEAM", ""))
            for key, val in ROSTER_EXPERIENCE_LOOKUP_2026.items():
                if key in team or team in key:
                    return val
            return 2.0 if team in top50_teams else float("nan")

        out["roster_experience_years"] = out.apply(exp_val, axis=1)
    return out


def load_db_table(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load ncaab_team_season_stats; empty DataFrame if table missing."""
    try:
        df = pd.read_sql_query("SELECT * FROM ncaab_team_season_stats", conn)
        return df
    except sqlite3.OperationalError:
        return pd.DataFrame()


def main() -> int:
    if not DB_PATH.exists():
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    # Load existing table (may be empty or have no 2026)
    db_df = load_db_table(conn)
    has_2026 = not db_df.empty and (db_df["season"].astype(int) == 2026).any()

    # Load 2026 from CSV if we have it (fetch_barttorvik_2026 output)
    csv_2026 = pd.DataFrame()
    if TEAM_STATS_2026_CSV.exists():
        csv_2026 = pd.read_csv(TEAM_STATS_2026_CSV)
        if "season" not in csv_2026.columns:
            csv_2026["season"] = 2026
        csv_2026["season"] = csv_2026["season"].astype(int)

    # Build combined: existing DB rows (optionally without 2026) + 2026 from CSV if needed
    if db_df.empty and not csv_2026.empty:
        combined = csv_2026.copy()
        print(f"Using 2026 from {TEAM_STATS_2026_CSV} ({len(combined)} teams).")
    elif not db_df.empty and not has_2026 and not csv_2026.empty:
        combined = pd.concat([db_df, csv_2026], ignore_index=True)
        combined = combined.drop_duplicates(subset=["season", "TEAM"], keep="last")
        print(f"Appended 2026 from CSV to existing table ({len(csv_2026)} teams).")
    elif not db_df.empty:
        combined = db_df.copy()
        if has_2026 and not csv_2026.empty:
            # Refresh 2026 rows from CSV so we get latest FTR/3P_O and March columns
            other = db_df[db_df["season"].astype(int) != 2026]
            # Align columns: add missing in csv_2026 from other, add missing in other from csv_2026
            for c in other.columns:
                if c not in csv_2026.columns:
                    csv_2026 = csv_2026.assign(**{c: float("nan")})
            for c in csv_2026.columns:
                if c not in other.columns:
                    other[c] = float("nan")
            combined = pd.concat([other, csv_2026], ignore_index=True)
            combined = combined.drop_duplicates(subset=["season", "TEAM"], keep="last")
            print("Refreshed 2026 rows from CSV.")
    else:
        if TEAM_STATS_COMBINED_CSV.exists():
            combined = pd.read_csv(TEAM_STATS_COMBINED_CSV)
            combined["season"] = combined["season"].astype(int)
            print(f"Loaded from {TEAM_STATS_COMBINED_CSV}.")
        else:
            print("No ncaab_team_season_stats and no team_stats_2026.csv or team_stats_combined.csv.", file=sys.stderr)
            conn.close()
            return 1

    combined = _ensure_march_columns(combined)
    combined = _backfill_march_columns(combined)
    combined = _ensure_march_columns(combined)

    # Replace table
    conn.execute("DROP TABLE IF EXISTS ncaab_team_season_stats")
    combined.to_sql("ncaab_team_season_stats", conn, index=False)
    conn.commit()
    conn.close()

    n_2026 = (combined["season"].astype(int) == 2026).sum()
    print(f"Updated ncaab_team_season_stats: {len(combined)} rows ({n_2026} for 2026).")
    print("Columns include: free_throw_pct, three_point_pct, roster_experience_years.")
    print("Run: python3 scripts/audit_march_data.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
