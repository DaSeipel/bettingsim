#!/usr/bin/env python3
"""
Normalize and combine CBB (college basketball) team-stats CSVs (KenPom-style)
from multiple seasons into a single schema and output directory.

Reads cbb22.csv, cbb23.csv, cbb24.csv, cbb25.csv (from repo root or --input-dir),
normalizes column names (TEAM, EFG_O, EFG_D, etc.), adds SEASON,
and writes:
  - data/ncaab/team_stats_combined.csv  (all seasons)
  - data/ncaab/team_stats_2022.csv ... team_stats_2025.csv (optional, per season)

Usage:
  python scripts/normalize_cbb_team_stats.py [--input-dir DIR] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# Canonical columns (order for output). 3PR/3PRD only in 2025; filled NaN for earlier seasons.
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
]

# Season label -> filename (without path)
SEASON_FILES = {
    2022: "cbb22.csv",
    2023: "cbb23.csv",
    2024: "cbb24.csv",
    2025: "cbb25.csv",
}


def _normalize_efg_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename EFG%/EFGD% or EFGD_D to EFG_O/EFG_D."""
    rename = {}
    for c in df.columns:
        if c == "EFG%":
            rename[c] = "EFG_O"
        elif c in ("EFGD%", "EFGD_D"):
            rename[c] = "EFG_D"
    if rename:
        df = df.rename(columns=rename)
    return df


def load_season(path: Path, season: int) -> pd.DataFrame:
    """Load one season CSV and normalize to canonical columns."""
    df = pd.read_csv(path)
    # Normalize TEAM column (cbb25 uses 'Team')
    if "Team" in df.columns and "TEAM" not in df.columns:
        df = df.rename(columns={"Team": "TEAM"})
    # Drop RK if present (cbb25)
    if "RK" in df.columns:
        df = df.drop(columns=["RK"])
    # EFG naming
    df = _normalize_efg_columns(df)
    # POSTSEASON missing in cbb25
    if "POSTSEASON" not in df.columns:
        df["POSTSEASON"] = ""
    # 3PR / 3PRD only in cbb25
    if "3PR" not in df.columns:
        df["3PR"] = float("nan")
    if "3PRD" not in df.columns:
        df["3PRD"] = float("nan")
    df["season"] = season
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize and combine CBB team stats CSVs.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Directory containing cbb22.csv, cbb23.csv, etc.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "ncaab",
        help="Directory for output CSVs.",
    )
    parser.add_argument(
        "--per-season",
        action="store_true",
        default=True,
        help="Also write per-season files (default: True).",
    )
    args = parser.parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for season, filename in SEASON_FILES.items():
        path = input_dir / filename
        if not path.exists():
            print(f"Skip (not found): {path}")
            continue
        df = load_season(path, season)
        # Ensure column order matches canonical; only include columns we have
        cols = [c for c in CANONICAL_COLUMNS if c in df.columns]
        df = df[cols]
        frames.append(df)
        if args.per_season:
            out_path = output_dir / f"team_stats_{season}.csv"
            df.to_csv(out_path, index=False)
            print(f"Wrote {out_path} ({len(df)} rows)")

    if not frames:
        print("No input files found. Exiting.")
        return

    combined = pd.concat(frames, ignore_index=True)
    # Reorder to canonical
    combined = combined[[c for c in CANONICAL_COLUMNS if c in combined.columns]]
    combined_path = output_dir / "team_stats_combined.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Wrote {combined_path} ({len(combined)} rows, seasons {sorted(combined['season'].unique().tolist())})")


if __name__ == "__main__":
    main()
