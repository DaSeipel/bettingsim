#!/usr/bin/env python3
"""
Rebuild games_with_team_stats for NCAAB 2025-26 by copying from the games table
and merging KenPom from ncaab_team_season_stats only. No Sports-Reference.
Keeps existing NBA and older NCAAB rows; replaces/adds NCAAB 2025-26 from games + KenPom.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
ESPN_DB = DATA_DIR / "espn.db"
NCAAB_2026_START = "2025-11-01"

KENPOM_STAT_COLUMNS = [
    "ADJOE", "ADJDE", "BARTHAG", "EFG_O", "EFG_D",
    "TOR", "TORD", "ORB", "DRB", "FTR", "FTRD",
    "ADJ_T", "SEED",
]

try:
    from thefuzz import fuzz
    from thefuzz import process as fuzz_process
except ImportError:
    fuzz_process = None


def _game_season_from_date(game_date) -> int | None:
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


def build_team_name_mapping(espn_names: list[str], csv_teams: list[str], min_score: int = 80) -> dict[str, str]:
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
        if name not in mapping:
            match = fuzz_process.extractOne(name, csv_teams, scorer=fuzz.token_set_ratio)
            if match and match[1] >= min_score:
                mapping[name] = match[0]
    return mapping


def merge_kenpom_into_games(
    games_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    name_mapping: dict[str, str],
) -> pd.DataFrame:
    g = games_df.copy()
    if "season" not in g.columns and "game_date" in g.columns:
        g["season"] = g["game_date"].apply(_game_season_from_date)
    if "season" not in g.columns or "league" not in g.columns:
        return g
    stat_cols = [c for c in KENPOM_STAT_COLUMNS if c in stats_df.columns]
    if not stat_cols:
        return g
    g["_home_team"] = g["home_team_name"].apply(
        lambda x: name_mapping.get(str(x or "").strip(), str(x or "").strip())
    )
    g["_away_team"] = g["away_team_name"].apply(
        lambda x: name_mapping.get(str(x or "").strip(), str(x or "").strip())
    )
    home_stats = stats_df[["season", "TEAM"] + stat_cols].copy()
    home_stats = home_stats.rename(columns={"TEAM": "_home_team", **{c: f"home_{c}" for c in stat_cols}})
    home_stats = home_stats[["season", "_home_team"] + [f"home_{c}" for c in stat_cols]]
    g = g.merge(home_stats, on=["season", "_home_team"], how="left")
    away_stats = stats_df[["season", "TEAM"] + stat_cols].copy()
    away_stats = away_stats.rename(columns={"TEAM": "_away_team", **{c: f"away_{c}" for c in stat_cols}})
    away_stats = away_stats[["season", "_away_team"] + [f"away_{c}" for c in stat_cols]]
    g = g.merge(away_stats, on=["season", "_away_team"], how="left")
    g = g.drop(columns=["_home_team", "_away_team"], errors="ignore")
    return g


def main() -> int:
    if not ESPN_DB.exists():
        print(f"Database not found: {ESPN_DB}")
        return 1
    if fuzz_process is None:
        print("Install thefuzz: pip install thefuzz[speedup]")
        return 1

    conn = sqlite3.connect(ESPN_DB)
    try:
        # Load existing games_with_team_stats; keep rows that are NOT NCAAB 2025-26
        existing = pd.read_sql_query("SELECT * FROM games_with_team_stats", conn)
        if existing.empty:
            print("games_with_team_stats is empty; will build from games table only.")
            keep_mask = pd.Series(dtype=bool)
        else:
            is_ncaab = existing["league"].astype(str).str.strip().str.lower() == "ncaab"
            game_dates = existing["game_date"].astype(str)
            is_2026 = game_dates >= NCAAB_2026_START
            drop_mask = is_ncaab & is_2026
            keep_mask = ~drop_mask
        old_rows = existing[keep_mask] if not existing.empty and keep_mask.any() else pd.DataFrame()

        # Load NCAAB 2025-26 from games table
        games_ncaab = pd.read_sql_query(
            "SELECT * FROM games WHERE league = 'ncaab' AND game_date >= ?",
            conn,
            params=(NCAAB_2026_START,),
        )
        if games_ncaab.empty:
            print("No NCAAB 2025-26 games in games table.")
            if old_rows.empty:
                return 1
            combined = old_rows
        else:
            games_ncaab = games_ncaab.copy()
            # Use season 2025 KenPom for ALL 2025-26 games (we don't have 2026 KenPom yet)
            games_ncaab["season"] = 2025
            stats_df = pd.read_sql_query("SELECT * FROM ncaab_team_season_stats", conn)
            if stats_df.empty:
                print("ncaab_team_season_stats is empty; NCAAB 2025-26 will have no KenPom.")
            else:
                stats_2025 = stats_df[stats_df["season"].astype(int) == 2025].copy()
                if stats_2025.empty:
                    print("No 2025 season in ncaab_team_season_stats; NCAAB 2025-26 will have no KenPom.")
                else:
                    csv_teams = stats_2025["TEAM"].astype(str).str.strip().dropna().unique().tolist()
                    espn_names = list(set(
                        games_ncaab["home_team_name"].astype(str).str.strip().dropna().tolist() +
                        games_ncaab["away_team_name"].astype(str).str.strip().dropna().tolist()
                    ))
                    name_mapping = build_team_name_mapping(espn_names, csv_teams)
                    games_ncaab = merge_kenpom_into_games(games_ncaab, stats_2025, name_mapping)
            games_ncaab["season"] = 2026
            # Align columns with existing table
            if not old_rows.empty:
                missing = [c for c in old_rows.columns if c not in games_ncaab.columns]
                if missing:
                    games_ncaab = pd.concat([games_ncaab, pd.DataFrame({c: [None] * len(games_ncaab) for c in missing})], axis=1)
                games_ncaab = games_ncaab.reindex(columns=old_rows.columns, fill_value=None)
                combined = pd.concat([old_rows, games_ncaab], ignore_index=True)
            else:
                combined = games_ncaab
        conn.execute("DROP TABLE IF EXISTS games_with_team_stats")
        combined.to_sql("games_with_team_stats", conn, index=False)
        conn.commit()
        print(f"Rebuilt games_with_team_stats: {len(combined)} rows ({len(games_ncaab) if not games_ncaab.empty else 0} NCAAB 2025-26 from games + KenPom)")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
