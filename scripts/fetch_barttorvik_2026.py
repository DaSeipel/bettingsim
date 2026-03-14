#!/usr/bin/env python3
"""
Fetch 2025-26 NCAAB team stats from Bart Torvik's team-level CSV and save in our format.

URL: https://barttorvik.com/2026_team_results.csv
Returns team-level stats with headers: rank, team, conf, record, adjoe, adjde, barthag, adjt, etc.

Saves: data/ncaab/team_stats_2026.csv with season=2026 and column names matching
team_stats_2025 (TEAM, ADJOE, ADJDE, BARTHAG, ADJ_T, CONF; EFG_O, EFG_D, TOR, ORB set to NaN).
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

BARTTORVIK_URL = "https://barttorvik.com/2026_team_results.csv"
REQUEST_TIMEOUT = 30
SEASON = 2026
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "ncaab" / "team_stats_2026.csv"

# Map API column names to our canonical names
COLUMN_RENAME = {
    "team": "TEAM",
    "conf": "CONF",
    "adjoe": "ADJOE",
    "adjde": "ADJDE",
    "barthag": "BARTHAG",
    "adjt": "ADJ_T",
}

# March multiplier columns (Tournament Success: Veteran Edge, Closer FT, 3P variance)
MARCH_COLUMNS = ["free_throw_pct", "three_point_pct", "roster_experience_years"]

# Canonical columns (same order as team_stats_2025 for compatibility)
CANONICAL_COLUMNS = [
    "season",
    "TEAM",
    "CONF",
    "G",
    "W",
    "ADJOE",
    "ADJDE",
    "BARTHAG",
    "EFG_O",
    "EFG_D",
    "TOR",
    "TORD",
    "ORB",
    "DRB",
    "FTR",
    "FTRD",
    "2P_O",
    "2P_D",
    "3P_O",
    "3P_D",
    "3PR",
    "3PRD",
    "ADJ_T",
    "WAB",
    "POSTSEASON",
    "SEED",
] + MARCH_COLUMNS


def fetch_barttorvik_2026() -> pd.DataFrame:
    """Fetch team-level CSV from Bart Torvik, map columns, return DataFrame in our format."""
    r = requests.get(BARTTORVIK_URL, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if df.empty:
        return df

    # Rename to our format
    rename = {c: COLUMN_RENAME[c] for c in df.columns if c in COLUMN_RENAME}
    df = df.rename(columns=rename)

    # Parse record (e.g. "29-2") for G and W if present
    if "record" in df.columns:
        g_list, w_list = [], []
        for rec in df["record"].astype(str):
            try:
                w, l = rec.split("-")
                w_list.append(int(w.strip()))
                g_list.append(int(w.strip()) + int(l.strip()))
            except (ValueError, AttributeError):
                w_list.append(0)
                g_list.append(0)
        df["W"] = w_list
        df["G"] = g_list
    else:
        df["G"] = 0
        df["W"] = 0

    df["season"] = SEASON
    # Set EFG_O, EFG_D, TOR, ORB to NaN as requested
    for col in ["EFG_O", "EFG_D", "TOR", "ORB"]:
        df[col] = float("nan")
    for col in ["TORD", "DRB", "FTR", "FTRD", "2P_O", "2P_D", "3P_O", "3P_D", "3PR", "3PRD", "WAB", "POSTSEASON", "SEED"]:
        if col not in df.columns:
            df[col] = float("nan") if col != "POSTSEASON" else ""
    if "POSTSEASON" in df.columns and df["POSTSEASON"].dtype != object:
        df["POSTSEASON"] = df["POSTSEASON"].fillna("").astype(str)

    # Map to March multiplier columns (for bracket_analysis.py Veteran Edge, Closer Factor)
    if "FTR" in df.columns:
        df["free_throw_pct"] = df["FTR"].apply(
            lambda x: float(x) * 100.0 if pd.notna(x) and float(x) <= 1 else (float(x) if pd.notna(x) else float("nan"))
        )
    else:
        df["free_throw_pct"] = float("nan")
    if "3P_O" in df.columns:
        df["three_point_pct"] = df["3P_O"].apply(
            lambda x: float(x) * 100.0 if pd.notna(x) and float(x) <= 1 else (float(x) if pd.notna(x) else float("nan"))
        )
    else:
        df["three_point_pct"] = float("nan")
    df["roster_experience_years"] = float("nan")  # filled by update_ncaab_march_stats.py from lookup

    cols = [c for c in CANONICAL_COLUMNS if c in df.columns]
    return df[cols].copy()


def main() -> None:
    print("Fetching 2025-26 NCAAB team stats from Bart Torvik (team-level CSV)...")
    df = fetch_barttorvik_2026()
    if df.empty:
        print("No data returned from API.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} teams to {OUTPUT_PATH}")

    # Show Maryland and Illinois ADJOE, ADJDE, BARTHAG to confirm range (ADJOE 100-130, ADJDE 85-110)
    for name in ("Maryland", "Illinois"):
        row = df.loc[df["TEAM"].str.strip().str.lower() == name.lower()]
        if row.empty:
            print(f"\n{name}: not found in data")
            continue
        row = row.iloc[0]
        print(f"\n{name}: ADJOE={row.get('ADJOE', 'N/A')}, ADJDE={row.get('ADJDE', 'N/A')}, BARTHAG={row.get('BARTHAG', 'N/A')}")


if __name__ == "__main__":
    main()
