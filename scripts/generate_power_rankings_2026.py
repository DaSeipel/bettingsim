#!/usr/bin/env python3
"""
Generate data/ncaab/power_rankings_2026.csv for the Bracket Engine.

1. Load ncaab_team_season_stats (2026 season) from espn.db.
2. Compute "average opponent" as mean of all team stats.
3. For each team, build feature row (team vs average), get XGBoost predicted margin
   via get_spread_predicted_margin (same as get_spread_predicted_margin used elsewhere).
4. Rank teams 1..N by predicted margin descending → ModelRank.
5. Output CSV: Team (canonical per BRACKET_TEAM_ALIAS_MAP), Seed (from bracket or 0), ModelRank.
"""
from pathlib import Path
import sqlite3
import sys
import tempfile

# project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from engine.betting_models import (
    build_ncaab_feature_row_from_team_stats,
    get_spread_predicted_margin,
)
from engine.bracket_analysis import (
    BRACKET_TEAM_ALIAS_MAP,
    parse_bracket_csv,
    resolve_team_name,
)

AVERAGE_TEAM_PLACEHOLDER = "__Average__"
GAME_DATE = "2026-03-15"  # tournament time
BASE_STATS = ["ADJOE", "ADJDE", "BARTHAG", "EFG_O", "EFG_D", "TOR", "TORD", "ORB", "DRB", "FTR", "FTRD", "ADJ_T", "SEED"]


def db_team_to_canonical(team: str) -> str:
    """Output name that matches bracket resolution: use canonical form from BRACKET_TEAM_ALIAS_MAP."""
    # Map: if this DB name is an alias, use canonical; else use as-is (already canonical or unknown).
    return BRACKET_TEAM_ALIAS_MAP.get(team, team)


def main() -> int:
    db_path = ROOT / "data" / "espn.db"
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return 1

    conn = sqlite3.connect(db_path)
    try:
        stats = pd.read_sql_query("SELECT * FROM ncaab_team_season_stats", conn)
    finally:
        conn.close()

    if stats.empty or "TEAM" not in stats.columns or "season" not in stats.columns:
        print("ncaab_team_season_stats missing or empty.")
        return 1

    # 2026 season (fallback to 2025 if no 2026)
    stats["season"] = stats["season"].astype(int)
    season_2026 = stats[stats["season"] == 2026]
    if season_2026.empty:
        season_2026 = stats[stats["season"] == 2025]
    if season_2026.empty:
        print("No 2026 or 2025 season in ncaab_team_season_stats.")
        return 1

    teams = season_2026["TEAM"].astype(str).str.strip().dropna().unique().tolist()
    # Numeric columns that go into feature row
    numeric_cols = [c for c in BASE_STATS if c in season_2026.columns]
    avg_row = {"TEAM": AVERAGE_TEAM_PLACEHOLDER, "season": 2026}
    for c in numeric_cols:
        avg_row[c] = float(season_2026[c].mean())
    for c in season_2026.columns:
        if c in ("TEAM", "season") or c in numeric_cols:
            continue
        if pd.api.types.is_numeric_dtype(season_2026[c]):
            avg_row[c] = float(season_2026[c].mean())
        else:
            avg_row[c] = ""
    stats_with_avg = pd.concat([season_2026, pd.DataFrame([avg_row])], ignore_index=True)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db = f.name
    try:
        conn = sqlite3.connect(temp_db)
        stats_with_avg.to_sql("ncaab_team_season_stats", conn, index=False)
        conn.close()

        # Power rating = predicted margin (team as home vs Average as away)
        results = []
        for i, team in enumerate(teams):
            row = build_ncaab_feature_row_from_team_stats(
                team, AVERAGE_TEAM_PLACEHOLDER, game_date=GAME_DATE, db_path=Path(temp_db)
            )
            if row is None:
                continue
            margin = get_spread_predicted_margin(
                row, 0.0, league="ncaab", home_team=team, away_team=AVERAGE_TEAM_PLACEHOLDER
            )
            if margin is None:
                continue
            results.append((team, float(margin)))
            if (i + 1) % 80 == 0:
                print(f"  ... {i + 1}/{len(teams)} teams", flush=True)
    finally:
        Path(temp_db).unlink(missing_ok=True)

    # ModelRank 1 = best (highest margin)
    results.sort(key=lambda x: -x[1])
    model_rank_by_team = {}
    for rank, (team, _) in enumerate(results, start=1):
        model_rank_by_team[team] = rank

    # Bracket seeds (canonical name -> seed)
    bracket_path = ROOT / "data" / "ncaab" / "bracket_2026.csv"
    bracket_seed = {}
    if bracket_path.exists():
        bracket = parse_bracket_csv(bracket_path.read_text())
        for m in bracket:
            for name, seed in [(m["team_a"], m["seed_a"]), (m["team_b"], m["seed_b"])]:
                canonical = resolve_team_name(name)
                bracket_seed[canonical] = seed

    # Build CSV: Team (canonical), Seed, ModelRank. Include all teams with ModelRank; Seed=0 if not in bracket.
    out_path = ROOT / "data" / "ncaab" / "power_rankings_2026.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for team_name, _ in results:
        canonical = db_team_to_canonical(team_name)
        seed = bracket_seed.get(canonical, bracket_seed.get(team_name, 0))
        rank = model_rank_by_team[team_name]
        rows.append((canonical, seed, rank))
    # Dedupe by canonical (same team might appear under different DB names - keep best rank)
    seen_canonical = {}
    for canonical, seed, rank in rows:
        if canonical not in seen_canonical or rank < seen_canonical[canonical][1]:
            seen_canonical[canonical] = (seed, rank)
    # Sort by ModelRank
    final = sorted(seen_canonical.items(), key=lambda x: x[1][1])
    with open(out_path, "w") as f:
        f.write("Team,Seed,ModelRank\n")
        for canonical, (seed, rank) in final:
            f.write(f"{canonical},{seed},{rank}\n")

    print(f"Wrote {len(final)} teams to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
