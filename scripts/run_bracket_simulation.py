#!/usr/bin/env python3
"""
Run bracket analysis from data/ncaab/bracket_2026.csv.
Builds seed-based power rankings from the bracket if no rankings file is provided.
"""
from pathlib import Path
import sys

# project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.bracket_analysis import parse_bracket_csv, parse_power_rankings_csv, run_bracket_analysis


def main() -> int:
    bracket_path = ROOT / "data" / "ncaab" / "bracket_2026.csv"
    if not bracket_path.exists():
        print(f"Bracket file not found: {bracket_path}")
        return 1

    bracket_content = bracket_path.read_text()
    bracket = parse_bracket_csv(bracket_content)
    if len(bracket) < 32:
        print(f"Bracket has {len(bracket)} matchups; expected 32.")
        return 1

    # Power rankings: use seed-based ranks from bracket if no file
    rankings_path = ROOT / "data" / "ncaab" / "power_rankings_2026.csv"
    if rankings_path.exists():
        power_rankings_content = rankings_path.read_text()
    else:
        # Build minimal: Team, Seed, ModelRank (ModelRank = seed for all)
        seen = set()
        rows = []
        for m in bracket:
            for team, seed in [(m["team_a"], m["seed_a"]), (m["team_b"], m["seed_b"])]:
                if team not in seen:
                    seen.add(team)
                    rows.append(f"{team},{seed},{seed}")
        power_rankings_content = "Team,Seed,ModelRank\n" + "\n".join(rows)

    n_sims = 10_000
    print(f"Running {n_sims:,} bracket simulations...")
    out = run_bracket_analysis(
        bracket_content,
        power_rankings_content,
        n_sims=n_sims,
    )

    for err in out.get("errors", []):
        print(f"Error: {err}")

    print("\n--- Top 5 Value Sleepers ---")
    for s in out.get("value_sleepers", []):
        print(s)

    print("\n--- Final 4 probabilities (top 15) ---")
    for r in (out.get("final4_probabilities") or [])[:15]:
        print(f"  {r['team']}: {r['final4_pct']}%")

    print("\n--- Glitch teams ---")
    for g in out.get("glitch_teams", []):
        print(g)

    print("\n--- Walters plays (delta 2–7 pts only) ---")
    for w in out.get("walters_plays", []):
        print(w)
    errs = out.get("walters_data_errors", [])
    if errs:
        print("\n--- Excluded (model spread hit 10- or 22-pt floor anchor) ---")
        for e in errs:
            print(e)

    print(f"\nDone. n_bracket_games={out.get('n_bracket_games')}, n_sims={out.get('n_sims')}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
