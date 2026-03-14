#!/usr/bin/env python3
"""
Load data/ncaab/team_stats_combined.csv into SQLite table ncaab_team_season_stats in data/espn.db.
Merge these season-level stats into games_with_team_stats by joining on team name and season,
adding home_* and away_* for ADJOE, ADJDE, BARTHAG, EFG_O, EFG_D, TOR, TORD, ORB, DRB, FTR, FTRD, ADJ_T, SEED.
Uses thefuzz for fuzzy team name matching between CSV and ESPN data.
Prints how many NCAAB games got stats merged vs no match.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from engine.utils import game_season_from_date, effective_kenpom_season

try:
    from thefuzz import fuzz
    from thefuzz import process as fuzz_process
except ImportError:
    fuzz = None
    fuzz_process = None

DATA_DIR = ROOT / "data"
ESPN_DB = DATA_DIR / "espn.db"
NCAAB_CSV = ROOT / "data" / "ncaab" / "team_stats_combined.csv"

# Columns to merge as home_* / away_*
KENPOM_STAT_COLUMNS = [
    "ADJOE", "ADJDE", "BARTHAG", "EFG_O", "EFG_D",
    "TOR", "TORD", "ORB", "DRB", "FTR", "FTRD",
    "ADJ_T", "SEED",
]
# March multiplier columns (Veteran Edge, Closer FT, 3P variance) — kept in table for bracket_analysis
MARCH_COLUMNS = ["free_throw_pct", "three_point_pct", "roster_experience_years"]


def _kenpom_column_name(prefix: str, stat: str) -> str:
    """Standardized column name: SEED -> seed, others unchanged (e.g. home_seed, home_ADJOE)."""
    if stat == "SEED":
        return f"{prefix}seed"
    return f"{prefix}{stat}"


def load_csv_into_ncaab_team_season_stats(csv_path: Path, db_path: Path) -> pd.DataFrame:
    """Load CSV and write to ncaab_team_season_stats; return DataFrame for merging."""
    df = pd.read_csv(csv_path)
    # Ensure season is int
    df["season"] = df["season"].astype(int)
    # Keep columns for table: KenPom stats + March multiplier columns (schema for bracket_analysis)
    keep = ["season", "TEAM"] + [c for c in KENPOM_STAT_COLUMNS if c in df.columns]
    keep += [c for c in MARCH_COLUMNS if c in df.columns]
    df = df[[c for c in keep if c in df.columns]].copy()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS ncaab_team_season_stats")
        df.to_sql("ncaab_team_season_stats", conn, index=False)
        conn.commit()
    finally:
        conn.close()
    return df


def build_team_name_mapping(
    espn_names: list[str],
    csv_teams: list[str],
    min_score: int = 80,
) -> dict[str, str]:
    """Build ESPN name -> CSV TEAM mapping using thefuzz. Returns dict: espn_name -> csv_team."""
    if not fuzz_process or not csv_teams:
        return {}
    mapping = {}
    espn_unique = sorted(set(n for n in espn_names if n and str(n).strip()))
    for espn_name in espn_unique:
        name = str(espn_name).strip()
        if not name:
            continue
        match = fuzz_process.extractOne(name, csv_teams, scorer=fuzz.ratio)
        if match and match[1] >= min_score:
            mapping[name] = match[0]
        # Also try token_set_ratio for "X Y Z" vs "Y Z" (e.g. "Kansas Jayhawks" -> "Kansas")
        if name not in mapping:
            match = fuzz_process.extractOne(name, csv_teams, scorer=fuzz.token_set_ratio)
            if match and match[1] >= min_score:
                mapping[name] = match[0]
    return mapping


# Games before this date (in season year) use prior season's KenPom (preseason proxy); avoids end-of-season leakage.
KENPOM_AS_OF_CUTOFF_MONTH = 1
KENPOM_AS_OF_CUTOFF_DAY = 1


def merge_ncaab_kenpom_into_games(
    games_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    name_mapping: dict[str, str],
    cutoff_month: int = KENPOM_AS_OF_CUTOFF_MONTH,
    cutoff_day: int = KENPOM_AS_OF_CUTOFF_DAY,
) -> pd.DataFrame:
    """Add home_* and away_* KenPom columns to games_df for NCAAB rows. Non-NCAAB get NaN.
    Early-season games (before cutoff date in season year) use prior season's KenPom to avoid leakage."""
    g = games_df.copy()
    # Drop existing KenPom columns so merge does not create duplicates (e.g. when rebuild already attached 2025 stats)
    for c in KENPOM_STAT_COLUMNS:
        g = g.drop(columns=[f"home_{c}", f"away_{c}"], errors="ignore")
    g = g.drop(columns=["home_seed", "away_seed"], errors="ignore")
    if "season" not in g.columns and "game_date" in g.columns:
        g["season"] = g["game_date"].apply(game_season_from_date)
    if "season" not in g.columns or "league" not in g.columns:
        return g

    # Lag: games before cutoff (e.g. Jan 1) use prior season's ratings (preseason / earliest available proxy)
    g["_effective_season"] = g.apply(
        lambda row: effective_kenpom_season(row["game_date"], row["season"], cutoff_month, cutoff_day),
        axis=1,
    )

    stat_cols = [c for c in KENPOM_STAT_COLUMNS if c in stats_df.columns]
    g["_home_team"] = g["home_team_name"].apply(
        lambda x: name_mapping.get(str(x or "").strip(), str(x or "").strip())
    )
    g["_away_team"] = g["away_team_name"].apply(
        lambda x: name_mapping.get(str(x or "").strip(), str(x or "").strip())
    )

    # Home merge: (_effective_season, _home_team) -> stats. Standardized naming: SEED -> home_seed/away_seed.
    home_stats = stats_df[["season", "TEAM"] + stat_cols].copy()
    home_rename = {"TEAM": "_home_team", **{c: _kenpom_column_name("home_", c) for c in stat_cols}}
    home_stats = home_stats.rename(columns=home_rename)
    home_out_cols = [home_rename[c] for c in stat_cols]
    home_stats = home_stats[["season", "_home_team"] + home_out_cols]
    g = g.merge(home_stats, left_on=["_effective_season", "_home_team"], right_on=["season", "_home_team"], how="left", suffixes=("", "_kenpom"))
    g = g.drop(columns=[c for c in g.columns if c.endswith("_kenpom")], errors="ignore")

    # Away merge
    away_stats = stats_df[["season", "TEAM"] + stat_cols].copy()
    away_rename = {"TEAM": "_away_team", **{c: _kenpom_column_name("away_", c) for c in stat_cols}}
    away_stats = away_stats.rename(columns=away_rename)
    away_out_cols = [away_rename[c] for c in stat_cols]
    away_stats = away_stats[["season", "_away_team"] + away_out_cols]
    g = g.merge(away_stats, left_on=["_effective_season", "_away_team"], right_on=["season", "_away_team"], how="left", suffixes=("", "_kenpom"))
    g = g.drop(columns=[c for c in g.columns if c.endswith("_kenpom")], errors="ignore")

    g = g.drop(columns=["_effective_season", "_home_team", "_away_team"], errors="ignore")
    return g


def main() -> None:
    if not NCAAB_CSV.exists():
        print(f"CSV not found: {NCAAB_CSV}")
        return
    if not ESPN_DB.exists():
        print(f"Database not found: {ESPN_DB}. Run the data pipeline first.")
        return
    if fuzz is None or fuzz_process is None:
        print("Install thefuzz: pip install thefuzz[speedup]")
        return

    # 1) Load CSV into ncaab_team_season_stats
    stats_df = load_csv_into_ncaab_team_season_stats(NCAAB_CSV, ESPN_DB)
    print(f"Loaded {len(stats_df)} rows into ncaab_team_season_stats")

    # Use 2025 season as fallback for 2026 games (no 2026 KenPom yet)
    if "season" in stats_df.columns and 2026 not in stats_df["season"].astype(int).values and (stats_df["season"].astype(int) == 2025).any():
        fallback = stats_df[stats_df["season"].astype(int) == 2025].copy()
        fallback["season"] = 2026
        stats_df = pd.concat([stats_df, fallback], ignore_index=True)
        print("Using 2025 KenPom as fallback for 2026 season games")

    csv_teams = stats_df["TEAM"].astype(str).str.strip().dropna().unique().tolist()

    # 2) Load games_with_team_stats (or games if table missing)
    conn = sqlite3.connect(ESPN_DB)
    try:
        try:
            games_df = pd.read_sql_query("SELECT * FROM games_with_team_stats", conn)
        except Exception:
            games_df = pd.read_sql_query("SELECT * FROM games", conn)
            if "season" not in games_df.columns and "game_date" in games_df.columns:
                games_df = games_df.copy()
                games_df["season"] = games_df["game_date"].apply(game_season_from_date)
    finally:
        conn.close()

    if games_df.empty:
        print("No games in database.")
        return

    # 3) Build fuzzy mapping from ESPN names to CSV TEAM (only for NCAAB team names that appear in games)
    ncaab_games = games_df[games_df["league"].astype(str).str.strip().str.lower() == "ncaab"]
    espn_home = ncaab_games["home_team_name"].astype(str).str.strip().dropna().tolist()
    espn_away = ncaab_games["away_team_name"].astype(str).str.strip().dropna().tolist()
    espn_names = list(set(espn_home + espn_away))
    name_mapping = build_team_name_mapping(espn_names, csv_teams)
    print(f"Fuzzy mapping: {len(name_mapping)} ESPN names mapped to CSV TEAM")

    # 4) Merge KenPom stats into games
    merged = merge_ncaab_kenpom_into_games(games_df, stats_df, name_mapping)

    # 5) Count NCAAB games: merged (both home and away have at least one stat) vs no match
    ncaab_mask = merged["league"].astype(str).str.strip().str.lower() == "ncaab"
    ncaab_total = ncaab_mask.sum()
    # "Success" = both home and away have non-null ADJOE (or any key stat)
    key_col = "home_ADJOE"
    if key_col not in merged.columns:
        key_col = "home_BARTHAG" if "home_BARTHAG" in merged.columns else None
    if key_col:
        has_home = merged[key_col].notna()
        away_col = key_col.replace("home_", "away_")
        has_away = merged[away_col].notna() if away_col in merged.columns else False
        merged_count = (ncaab_mask & has_home & has_away).sum()
    else:
        merged_count = 0
    no_match_count = ncaab_total - merged_count

    print(f"NCAAB games with KenPom stats merged: {merged_count}")
    print(f"NCAAB games with no match: {no_match_count}")
    print(f"NCAAB games total: {ncaab_total}")

    # 6) Write back to espn.db
    conn = sqlite3.connect(ESPN_DB)
    try:
        conn.execute("DROP TABLE IF EXISTS games_with_team_stats")
        merged.to_sql("games_with_team_stats", conn, index=False)
        conn.commit()
    finally:
        conn.close()
    print("Updated games_with_team_stats in espn.db.")


if __name__ == "__main__":
    main()
