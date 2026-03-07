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
from pathlib import Path

import pandas as pd

try:
    from thefuzz import fuzz
    from thefuzz import process as fuzz_process
except ImportError:
    fuzz = None
    fuzz_process = None

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ESPN_DB = DATA_DIR / "espn.db"
NCAAB_CSV = ROOT / "data" / "ncaab" / "team_stats_combined.csv"

# Columns to merge as home_* / away_*
KENPOM_STAT_COLUMNS = [
    "ADJOE", "ADJDE", "BARTHAG", "EFG_O", "EFG_D",
    "TOR", "TORD", "ORB", "DRB", "FTR", "FTRD",
    "ADJ_T", "SEED",
]


def _game_season_from_date(game_date) -> int | None:
    """Derive season (end year) from game_date. Oct+ -> next year; else current year."""
    if game_date is None or pd.isna(game_date):
        return None
    s = str(game_date).strip()
    if not s or len(s) < 4:
        return None
    try:
        year = int(s[:4])
        if len(s) >= 7:
            month = int(s[5:7])
            if month >= 10:
                return year + 1
        return year
    except (ValueError, TypeError):
        return None


def load_csv_into_ncaab_team_season_stats(csv_path: Path, db_path: Path) -> pd.DataFrame:
    """Load CSV and write to ncaab_team_season_stats; return DataFrame for merging."""
    df = pd.read_csv(csv_path)
    # Ensure season is int
    df["season"] = df["season"].astype(int)
    # Keep only columns we need for the table and merge
    keep = ["season", "TEAM"] + [c for c in KENPOM_STAT_COLUMNS if c in df.columns]
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


def merge_ncaab_kenpom_into_games(
    games_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    name_mapping: dict[str, str],
) -> pd.DataFrame:
    """Add home_* and away_* KenPom columns to games_df for NCAAB rows. Non-NCAAB get NaN."""
    g = games_df.copy()
    if "season" not in g.columns and "game_date" in g.columns:
        g["season"] = g["game_date"].apply(_game_season_from_date)
    if "season" not in g.columns or "league" not in g.columns:
        return g

    stat_cols = [c for c in KENPOM_STAT_COLUMNS if c in stats_df.columns]
    g["_home_team"] = g["home_team_name"].apply(
        lambda x: name_mapping.get(str(x or "").strip(), str(x or "").strip())
    )
    g["_away_team"] = g["away_team_name"].apply(
        lambda x: name_mapping.get(str(x or "").strip(), str(x or "").strip())
    )

    # Home merge: (season, _home_team) -> stats
    home_stats = stats_df[["season", "TEAM"] + stat_cols].copy()
    home_stats = home_stats.rename(columns={"TEAM": "_home_team", **{c: f"home_{c}" for c in stat_cols}})
    home_stats = home_stats[["season", "_home_team"] + [f"home_{c}" for c in stat_cols]]
    g = g.merge(home_stats, on=["season", "_home_team"], how="left")

    # Away merge
    away_stats = stats_df[["season", "TEAM"] + stat_cols].copy()
    away_stats = away_stats.rename(columns={"TEAM": "_away_team", **{c: f"away_{c}" for c in stat_cols}})
    away_stats = away_stats[["season", "_away_team"] + [f"away_{c}" for c in stat_cols]]
    g = g.merge(away_stats, on=["season", "_away_team"], how="left")

    g = g.drop(columns=["_home_team", "_away_team"], errors="ignore")
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
                games_df["season"] = games_df["game_date"].apply(_game_season_from_date)
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
