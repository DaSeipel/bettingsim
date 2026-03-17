#!/usr/bin/env python3
"""
Print underlying factor breakdown for the 3 Walters Plays:
  UCLA vs UCF, Ohio State vs TCU, Miami vs Missouri.
Also for Illinois vs Penn and Virginia vs Wright St. (biggest model-vs-market disagreements among excluded).
Shows: raw/adjusted margin, Veteran Edge, Closer Factor, 3P Variance, power ranking delta.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.bracket_analysis import (
    parse_bracket_csv,
    parse_power_rankings_csv,
    load_march_stats,
    get_matchup_march_factor_breakdown,
    BRACKET_TEAM_ALIAS_MAP,
    resolve_team_name,
)

WALTERS_MATCHUPS = [
    ("UCLA", "UCF"),
    ("Ohio State", "TCU"),
    ("Miami", "Missouri"),
]

# Biggest model-vs-market disagreements among excluded (no anchor) matchups
EXCLUDED_DISAGREEMENT_MATCHUPS = [
    ("Illinois", "Penn"),
    ("Virginia", "Wright St."),
]


def main() -> int:
    bracket_path = ROOT / "data" / "ncaab" / "bracket_2026.csv"
    rankings_path = ROOT / "data" / "ncaab" / "power_rankings_2026.csv"
    db_path = ROOT / "data" / "espn.db"

    if not bracket_path.exists() or not rankings_path.exists() or not db_path.exists():
        print("Missing data: bracket_2026.csv, power_rankings_2026.csv, or espn.db")
        return 1

    bracket = parse_bracket_csv(bracket_path.read_text())
    rankings = parse_power_rankings_csv(rankings_path.read_text())
    team_stats, rank_ft, rank_3p, rank_exp = load_march_stats(db_path)

    model_rank_by_canonical = {resolve_team_name(r["team"], BRACKET_TEAM_ALIAS_MAP): r["model_rank"] for r in rankings}
    team_to_seed = {}
    for m in bracket:
        team_to_seed[resolve_team_name(m["team_a"], BRACKET_TEAM_ALIAS_MAP)] = m["seed_a"]
        team_to_seed[resolve_team_name(m["team_b"], BRACKET_TEAM_ALIAS_MAP)] = m["seed_b"]

    from engine.bracket_analysis import _potential_glitch_teams
    potential_glitch = _potential_glitch_teams(bracket, rank_ft, rank_3p, rank_exp, BRACKET_TEAM_ALIAS_MAP)

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

        print("=" * 60)
        print(f"  {team_a}  vs  {team_b}")
        print("=" * 60)

        if b["raw_margin"] is None:
            print("  (No model margin — feature row unavailable)\n")
            continue

        print(f"  Raw model margin (team_a - team_b):     {b['raw_margin']:+.2f} pts")
        print(f"  Adjusted margin (tier/anchor):          {b['adjusted_margin']:+.2f} pts")

        print("\n  --- Veteran Edge ---")
        print(f"  Experience (team_a): {b.get('exp_a')} yrs  (team_b): {b.get('exp_b')} yrs")
        ve = b["veteran_edge_pts"]
        if ve is not None:
            if ve > 0:
                print(f"  Veteran Edge score: +{ve} pts for {team_a} (veteran vs non-veteran)")
            elif ve < 0:
                print(f"  Veteran Edge score: +{-ve} pts for {team_b} (veteran vs non-veteran)")
            else:
                print(f"  Veteran Edge score: 0 (no veteran/non-veteran edge)")

        print("\n  --- Closer Factor ---")
        print(f"  {b['closer_factor']}")

        print("\n  --- 3P Variance ---")
        print(f"  Elite 3P (top 10%): {team_a}={b['elite_3p_team_a']}, {team_b}={b['elite_3p_team_b']}")
        if b["three_p_sigma_pts"] is not None:
            print(f"  3P Variance: sigma = {b['three_p_sigma_pts']} pts (margin noise when either is elite 3P)")
        else:
            print(f"  3P Variance: N/A (neither team elite 3P)")

        print("\n  --- Power ranking delta ---")
        ra, rb = b["power_rank_a"], b["power_rank_b"]
        delta = b["power_rank_delta"]
        if ra is not None and rb is not None:
            print(f"  {team_a} ModelRank: {ra}  |  {team_b} ModelRank: {rb}")
            if delta is not None:
                print(f"  Power rank delta (team_a - team_b): {delta:+d}  (lower rank = better; negative = {team_a} better ranked)")
        else:
            print("  (One or both teams missing from power_rankings_2026.csv)")

        print()

    # Same breakdown for Illinois vs Penn, Virginia vs Wright St. (excluded, big model-vs-market disagreement)
    print("=" * 60)
    print("  EXCLUDED MATCHUPS — biggest model vs market disagreement")
    print("=" * 60)

    for team_a, team_b in EXCLUDED_DISAGREEMENT_MATCHUPS:
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

        print()
        print("=" * 60)
        print(f"  {team_a}  vs  {team_b}")
        print("=" * 60)

        if b["raw_margin"] is None:
            print("  (No model margin — feature row unavailable)\n")
            continue

        print(f"  Raw model margin (team_a - team_b):     {b['raw_margin']:+.2f} pts")
        print(f"  Adjusted margin (tier/anchor):          {b['adjusted_margin']:+.2f} pts")

        print("\n  --- Veteran Edge ---")
        print(f"  Experience (team_a): {b.get('exp_a')} yrs  (team_b): {b.get('exp_b')} yrs")
        ve = b["veteran_edge_pts"]
        if ve is not None:
            if ve > 0:
                print(f"  Veteran Edge score: +{ve} pts for {team_a} (veteran vs non-veteran)")
            elif ve < 0:
                print(f"  Veteran Edge score: +{-ve} pts for {team_b} (veteran vs non-veteran)")
            else:
                print(f"  Veteran Edge score: 0 (no veteran/non-veteran edge)")

        print("\n  --- Closer Factor ---")
        print(f"  {b['closer_factor']}")

        print("\n  --- 3P Variance ---")
        print(f"  Elite 3P (top 10%): {team_a}={b['elite_3p_team_a']}, {team_b}={b['elite_3p_team_b']}")
        if b["three_p_sigma_pts"] is not None:
            print(f"  3P Variance: sigma = {b['three_p_sigma_pts']} pts (margin noise when either is elite 3P)")
        else:
            print(f"  3P Variance: N/A (neither team elite 3P)")

        print("\n  --- Power ranking delta ---")
        ra, rb = b["power_rank_a"], b["power_rank_b"]
        delta = b["power_rank_delta"]
        if ra is not None and rb is not None:
            print(f"  {team_a} ModelRank: {ra}  |  {team_b} ModelRank: {rb}")
            if delta is not None:
                print(f"  Power rank delta (team_a - team_b): {delta:+d}  (lower rank = better; negative = {team_a} better ranked)")
        else:
            print("  (One or both teams missing from power_rankings_2026.csv)")

        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
