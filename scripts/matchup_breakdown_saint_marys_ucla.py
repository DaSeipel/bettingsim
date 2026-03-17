#!/usr/bin/env python3
"""
Run get_matchup_march_factor_breakdown for Saint Mary's vs Texas A&M and UCLA vs UCF
(with rank-delta adjustment active). Print raw/adjusted margin, Veteran Edge, Closer Factor,
3P Variance, power ranking delta, and rank-delta adjustment amount.
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
    _potential_glitch_teams,
    BRACKET_TEAM_ALIAS_MAP,
    resolve_team_name,
)

MATCHUPS = [
    ("Saint Mary's", "Texas A&M"),
    ("UCLA", "UCF"),
]


def _print_breakdown(team_a: str, team_b: str, b: dict) -> None:
    print(f"  Raw margin (team_a - team_b):     {b['raw_margin']:+.2f} pts")
    print(f"  Adjusted margin (tier+anchor+rank_delta): {b['adjusted_margin']:+.2f} pts")
    print()
    print("  Veteran Edge:                    ", end="")
    ve = b.get("veteran_edge_pts")
    if ve is not None:
        if ve > 0:
            print(f"+{ve} pts for {team_a}")
        elif ve < 0:
            print(f"+{-ve} pts for {team_b}")
        else:
            print("0 (no edge)")
    else:
        print("—")
    print(f"  Closer Factor:                   {b.get('closer_factor', '—')}")
    print("  3P Variance:                     ", end="")
    sigma = b.get("three_p_sigma_pts")
    if sigma is not None:
        print(f"σ = {sigma} pts (elite 3P: {team_a}={b.get('elite_3p_team_a')}, {team_b}={b.get('elite_3p_team_b')})")
    else:
        print("no elite 3P in matchup")
    print(f"  Power rank delta (team_a - team_b): {b.get('power_rank_delta')}  (rank_a={b.get('power_rank_a')}, rank_b={b.get('power_rank_b')})")
    adj = b.get("rank_delta_adj_pts")
    print(f"  Rank-delta adjustment amount:    {adj:+.2f} pts" if adj is not None else "  Rank-delta adjustment amount:    —")
    print()


def main() -> int:
    bracket_path = ROOT / "data" / "ncaab" / "bracket_2026.csv"
    rankings_path = ROOT / "data" / "ncaab" / "power_rankings_2026.csv"
    db_path = ROOT / "data" / "espn.db"

    if not bracket_path.exists() or not rankings_path.exists() or not db_path.exists():
        print("Missing bracket_2026.csv, power_rankings_2026.csv, or espn.db")
        return 1

    bracket = parse_bracket_csv(bracket_path.read_text())
    rankings = parse_power_rankings_csv(rankings_path.read_text())
    team_stats, rank_ft, rank_3p, rank_exp = load_march_stats(db_path)
    model_rank_by_canonical = {resolve_team_name(r["team"], BRACKET_TEAM_ALIAS_MAP): r["model_rank"] for r in rankings}
    team_to_seed = {}
    for m in bracket:
        team_to_seed[resolve_team_name(m["team_a"], BRACKET_TEAM_ALIAS_MAP)] = m["seed_a"]
        team_to_seed[resolve_team_name(m["team_b"], BRACKET_TEAM_ALIAS_MAP)] = m["seed_b"]
    potential_glitch = _potential_glitch_teams(bracket, rank_ft, rank_3p, rank_exp, BRACKET_TEAM_ALIAS_MAP)

    print("Rank-delta adjustment: ACTIVE (tier+anchor then (rank_delta/10)*0.8 toward better-ranked when Δ>50, cap 15 pts)")
    print()

    for team_a, team_b in MATCHUPS:
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
        print("=" * 70)
        print(f"  {team_a}  vs  {team_b}")
        print("=" * 70)
        if b.get("raw_margin") is None:
            print("  No model margin (feature row unavailable).")
            print()
            continue
        _print_breakdown(team_a, team_b, b)

    return 0


if __name__ == "__main__":
    sys.exit(main())
