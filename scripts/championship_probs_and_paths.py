#!/usr/bin/env python3
"""
Print championship win probabilities for all teams (sorted descending).
Print the most likely bracket path (opponent each round) for the top 5 championship favorites.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.bracket_analysis import parse_bracket_csv, parse_power_rankings_csv, run_bracket_analysis

ROUND_LABELS = ["R1 (First Round)", "R2 (Round of 32)", "R3 (Sweet 16)", "R4 (Elite 8)", "R5 (Final Four)", "R6 (Championship)"]


def main() -> int:
    bracket_path = ROOT / "data" / "ncaab" / "bracket_2026.csv"
    rankings_path = ROOT / "data" / "ncaab" / "power_rankings_2026.csv"

    if not bracket_path.exists():
        print(f"Bracket not found: {bracket_path}")
        return 1
    if not rankings_path.exists():
        print(f"Power rankings not found: {rankings_path}")
        return 1

    bracket_content = bracket_path.read_text()
    power_content = rankings_path.read_text()

    print("Running bracket analysis (10k sims, full bracket through championship)...")
    out = run_bracket_analysis(
        bracket_content,
        power_content,
        n_sims=10_000,
    )

    champion_pct = out.get("champion_pct") or {}
    bracket = parse_bracket_csv(bracket_content)
    all_teams = set()
    for m in bracket:
        all_teams.add(m["team_a"])
        all_teams.add(m["team_b"])
    for t in all_teams:
        if t not in champion_pct:
            champion_pct[t] = 0.0

    sorted_teams = sorted(champion_pct.items(), key=lambda x: -x[1])

    print("\n--- Championship win probability (all teams, descending) ---\n")
    print(f"{'Team':<28} {'Champ %':>8}")
    print("-" * 38)
    for team, pct in sorted_teams:
        print(f"{team:<28} {pct:>7.2f}%")

    print("\n--- Most likely bracket path for top 5 championship favorites ---\n")
    top5 = out.get("most_likely_paths_top5") or []
    for i, item in enumerate(top5, 1):
        team = item.get("team", "")
        pct = item.get("champion_pct", 0)
        path = item.get("path") or ["—"] * 6
        print(f"{i}. {team} ({pct}% champion)")
        for r, label in enumerate(ROUND_LABELS):
            opp = path[r] if r < len(path) else "—"
            print(f"   {label}: vs {opp}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
