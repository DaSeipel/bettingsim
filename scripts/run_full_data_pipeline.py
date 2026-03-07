#!/usr/bin/env python3
"""
Run the full data pipeline from scratch: ESPN games (2 seasons NBA + NCAAB),
situational features (rest, travel, ATS, B2B), then fetch_merge_and_save to produce
games_with_team_stats. Verify >= 500 rows and print count + 5 sample rows.

Options:
  --skip-espn                 Use existing games in DB (skip ESPN API; faster re-runs).
  --skip-advanced-analytics   Skip nba_api rolling (faster when Sports Reference is down).
  --skip-historical-odds      Skip fetching historical odds (use existing cache/DB).
  --max-games=N               Process only N most recent games (faster; e.g. --max-games=600).

Historical odds: set ODDS_API_KEY to fetch from The Odds API (v4 historical), one week at a time;
responses cached to data/historical_odds_cache/, closing lines stored in data/odds.db (historical_odds table),
then merged into games_with_team_stats for edge calculation.
Run from repo root: python3 scripts/run_full_data_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MIN_ROWS = 500
SEASONS_COUNT = 2  # 2 full seasons


def main() -> int:
    argv = [a for a in sys.argv[1:] if a.startswith("--")]
    skip_espn = "--skip-espn" in argv
    max_games = None
    for a in argv:
        if a.startswith("--max-games="):
            try:
                max_games = int(a.split("=", 1)[1])
            except ValueError:
                pass
            break
    db_path = ROOT / "data" / "espn.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    seasons = [now.year - 1 - i for i in range(SEASONS_COUNT)]  # e.g. [2024, 2023]

    print("=" * 60)
    print("FULL DATA PIPELINE (ESPN + situational + merge)")
    print("=" * 60)
    print(f"Seasons: {seasons}")
    print(f"DB: {db_path}")
    if skip_espn:
        print("(Skipping ESPN fetch; using existing games in DB)")
    print()

    # 1) ESPN: pull 2 full seasons of NBA and NCAAB game results (unless --skip-espn)
    if not skip_espn:
        print("(1) Pulling NBA and NCAAB game results from ESPN (team schedules)...")
        from engine.espn_collector import collect_and_store_all

        result = collect_and_store_all(
            leagues=["nba", "ncaab"],
            seasons=seasons,
            db_path=db_path,
        )
        n_nba = result.get("nba", {}).get("games", None)
        n_ncaab = result.get("ncaab", {}).get("games", None)
        n_nba = len(n_nba) if n_nba is not None and hasattr(n_nba, "__len__") else 0
        n_ncaab = len(n_ncaab) if n_ncaab is not None and hasattr(n_ncaab, "__len__") else 0
        print(f"    NBA games stored: {n_nba}")
        print(f"    NCAAB games stored: {n_ncaab}")
    else:
        from engine.espn_collector import load_games_from_sqlite
        n_nba = len(load_games_from_sqlite(league_key="nba", db_path=db_path))
        n_ncaab = len(load_games_from_sqlite(league_key="ncaab", db_path=db_path))
        print(f"(1) Using existing games: NBA={n_nba}, NCAAB={n_ncaab}")
    print()

    # 2) Situational features (rest days, travel, ATS trends, back-to-backs)
    print("(2) Building situational features (rest, travel, B2B, win pct last 30)...")
    from engine.espn_collector import load_games_from_sqlite
    from engine.situational_features import build_and_save_situational_features

    for league in ["nba", "ncaab"]:
        games_df = load_games_from_sqlite(league_key=league, db_path=db_path)
        if games_df.empty:
            print(f"    No games for {league}; skipping situational.")
            continue
        built = build_and_save_situational_features(
            games_df, league=league, db_path=db_path, use_geopy=True
        )
        print(f"    {league}: {len(built)} situational rows")
    print()

    # 3) Merge: team stats (sportsreference or fallback from games) + games, advanced analytics, momentum, line movement
    skip_adv = "--skip-advanced-analytics" in sys.argv
    if skip_adv:
        print("(3) Merging (skip nba_api advanced analytics), momentum, line movement...")
    else:
        print("(3) Fetching team stats, merging, advanced analytics, momentum, line movement...")
    from engine.sportsref_stats import fetch_merge_and_save

    merged = fetch_merge_and_save(
        seasons=seasons,
        db_path=db_path,
        skip_advanced_analytics=skip_adv,
        max_games=max_games,
    )
    if merged.empty:
        print("    fetch_merge_and_save returned empty. Check games table and sportsreference/sportsreference install.")
        return 1
    print(f"    Merged rows: {len(merged)}")
    print()

    # 4) Historical odds: fetch from The Odds API (last 2 seasons, one week at a time), cache, store in odds.db
    import os
    odds_api_key = (os.environ.get("ODDS_API_KEY") or "").strip()
    odds_db_path = ROOT / "data" / "odds.db"
    if odds_api_key and "--skip-historical-odds" not in argv:
        print("(4) Fetching historical odds (The Odds API, 2 seasons, one week at a time, cache to disk)...")
        from engine.historical_odds import fetch_last_n_seasons
        totals = fetch_last_n_seasons(odds_api_key, seasons=2, db_path=odds_db_path, use_cache=True)
        for sk, count in totals.items():
            print(f"    {sk}: {count} rows inserted (cached responses in data/historical_odds_cache/)")
    else:
        if not odds_api_key:
            print("(4) Skipping historical odds (set ODDS_API_KEY to fetch).")
        else:
            print("(4) Skipping historical odds (--skip-historical-odds).")
    print()

    # 5) Join historical closing lines to games_with_team_stats (espn.db)
    print("(5) Merging historical closing lines into games_with_team_stats...")
    from engine.historical_odds import merge_historical_closing_into_games
    merged_with_odds = merge_historical_closing_into_games(espn_db_path=db_path, odds_db_path=odds_db_path)
    if not merged_with_odds.empty and "closing_home_spread" in merged_with_odds.columns:
        n_with_spread = merged_with_odds["closing_home_spread"].notna().sum()
        print(f"    Games with closing spread: {n_with_spread} / {len(merged_with_odds)}")
    print()

    # 6) Verify final table and print
    import sqlite3

    if not db_path.exists():
        print("DB not found after pipeline.")
        return 1
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute("SELECT COUNT(*) FROM games_with_team_stats")
        count = cur.fetchone()[0]
        if count == 0 and not merged_with_odds.empty:
            count = len(merged_with_odds)
        if count < MIN_ROWS:
            print(f"*** Only {count} rows in games_with_team_stats (target >= {MIN_ROWS}).")
        else:
            print(f"games_with_team_stats row count: {count} (>= {MIN_ROWS})")
        cur = conn.execute(
            "SELECT * FROM games_with_team_stats LIMIT 5"
        )
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    finally:
        conn.close()

    print()
    print("Sample of 5 rows (columns then values):")
    print("-" * 60)
    if cols and rows:
        # Print as a compact table: first few columns + key features
        key_cols = [c for c in cols if c in (
            "league", "game_id", "game_date", "home_team_name", "away_team_name",
            "home_score", "away_score", "home_offensive_rating", "away_offensive_rating",
            "home_days_rest", "away_days_rest", "home_games_in_last_5_days", "away_games_in_last_5_days",
            "closing_home_spread", "closing_total", "closing_ml_home", "closing_ml_away",
        )][:14]
        if not key_cols:
            key_cols = cols[:10]
        for i, row in enumerate(rows):
            d = dict(zip(cols, row))
            parts = [f"{c}={d.get(c)}" for c in key_cols if c in d]
            print(f"  Row {i+1}: " + " | ".join(parts))
    else:
        print("  (no rows)")
    print("=" * 60)
    return 0 if count >= MIN_ROWS else 1


if __name__ == "__main__":
    sys.exit(main())
