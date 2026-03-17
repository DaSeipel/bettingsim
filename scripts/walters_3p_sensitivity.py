#!/usr/bin/env python3
"""
Sensitivity analysis on the 3 Walters Plays: simulate each matchup 20,000 times
with sigma = 0 (no 3P variance), 3, 6 (current), and 9. Print win probability
for each team at each sigma level to show how much 3P Variance is swinging outcomes.
"""
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.bracket_analysis import (
    parse_bracket_csv,
    parse_power_rankings_csv,
    load_march_stats,
    get_matchup_march_factor_breakdown,
    _potential_glitch_teams,
    resolve_team_name,
    BRACKET_TEAM_ALIAS_MAP,
)

WALTERS_MATCHUPS = [
    ("UCLA", "UCF"),
    ("Ohio State", "TCU"),
    ("Miami", "Missouri"),
]

N_SIMS = 20_000
SIGMA_LEVELS = [0, 3, 6, 9]  # 6 = current MARGIN_SIGMA_ELITE_3P


def simulate_win_prob(base_margin: float, sigma: float, n_sims: int, rng: random.Random) -> float:
    """P(team_a wins) when outcome = (base_margin + N(0, sigma)) > 0."""
    if sigma == 0:
        return 1.0 if base_margin > 0 else 0.0
    wins_a = sum(1 for _ in range(n_sims) if base_margin + rng.gauss(0, sigma) > 0)
    return wins_a / n_sims


def main() -> int:
    bracket_path = ROOT / "data" / "ncaab" / "bracket_2026.csv"
    rankings_path = ROOT / "data" / "ncaab" / "power_rankings_2026.csv"
    db_path = ROOT / "data" / "espn.db"

    if not bracket_path.exists() or not rankings_path.exists() or not db_path.exists():
        print("Missing data: bracket_2026.csv, power_rankings_2026.csv, or espn.db")
        return 1

    bracket = parse_bracket_csv(bracket_path.read_text())
    power_content = rankings_path.read_text()
    rankings = parse_power_rankings_csv(power_content)
    team_stats, rank_ft, rank_3p, rank_exp = load_march_stats(db_path)

    model_rank_by_canonical = {
        resolve_team_name(r["team"], BRACKET_TEAM_ALIAS_MAP): r["model_rank"]
        for r in rankings
    }
    team_to_seed = {}
    for m in bracket:
        team_to_seed[resolve_team_name(m["team_a"], BRACKET_TEAM_ALIAS_MAP)] = m["seed_a"]
        team_to_seed[resolve_team_name(m["team_b"], BRACKET_TEAM_ALIAS_MAP)] = m["seed_b"]
    potential_glitch = _potential_glitch_teams(
        bracket, rank_ft, rank_3p, rank_exp, BRACKET_TEAM_ALIAS_MAP
    )

    rng = random.Random(42)

    print("=" * 75)
    print("  3P Variance sensitivity: Walters Play matchups, 20,000 sims per (matchup, sigma)")
    print("  Base margin = adjusted margin + veteran edge (team_a perspective)")
    print("=" * 75)

    for team_a, team_b in WALTERS_MATCHUPS:
        b = get_matchup_march_factor_breakdown(
            team_a,
            team_b,
            db_path,
            None,
            team_stats,
            rank_ft,
            rank_3p,
            model_rank_by_canonical,
            team_to_seed,
            potential_glitch,
            BRACKET_TEAM_ALIAS_MAP,
        )
        raw = b.get("raw_margin")
        adj = b.get("adjusted_margin")
        ve = b.get("veteran_edge_pts") or 0
        if raw is None or adj is None:
            print(f"\n  {team_a} vs {team_b}: no model margin — skipping")
            continue
        base_margin = adj + ve

        print(f"\n  --- {team_a} vs {team_b} ---")
        print(f"  Base margin ({team_a}): {base_margin:+.2f} pts  (adjusted {adj:+.2f} + veteran edge {ve:+.1f})")
        print(f"  {'Sigma':<8} {team_a + ' win %':<20} {team_b + ' win %':<20}")
        print("  " + "-" * 52)

        for sigma in SIGMA_LEVELS:
            p_a = simulate_win_prob(base_margin, sigma, N_SIMS, rng)
            p_b = 1.0 - p_a
            print(f"  {sigma:<8} {p_a * 100:>6.2f}%             {p_b * 100:>6.2f}%")

    print("\n  (sigma=6 is current MARGIN_SIGMA_ELITE_3P in bracket_analysis)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
