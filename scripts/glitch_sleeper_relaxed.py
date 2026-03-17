#!/usr/bin/env python3
"""
Re-run Glitch Sleeper filter with relaxed thresholds:
  - Seed 7+
  - Final Four rate > 3%
  - Top 75 in at least ONE of: FT%, 3P%, Experience

Print all teams that qualify. Then show details for Penn, VCU, Santa Clara, SMU, Wright St.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.bracket_analysis import (
    parse_bracket_csv,
    parse_power_rankings_csv,
    run_bracket_analysis,
    resolve_team_name,
    load_march_stats,
    get_model_spread_for_matchup,
    _apply_tier_and_anchor,
    _potential_glitch_teams,
    BRACKET_TEAM_ALIAS_MAP,
)

# Relaxed thresholds (vs original: F4 > 5%, Top 50 in at least 2)
RELAXED_F4_PCT_MIN = 3.0
RELAXED_TOP_N = 75
RELAXED_MIN_CATEGORIES = 1  # at least one of FT%, 3P%, Experience in top 75

CHECK_TEAMS = ["Penn", "VCU", "Santa Clara", "SMU", "Wright St."]

# Flag these: model spread disagrees sharply with market (seed 7+, F4 notable)
MODEL_VS_MARKET_FLAG = [
    ("Penn", 10, "Illinois", "Penn", -22.5),      # Illinois -22.5 market
    ("Wright St.", 10, "Virginia", "Wright St.", -17.5),  # Virginia -17.5 market
]


def main() -> int:
    bracket_path = ROOT / "data" / "ncaab" / "bracket_2026.csv"
    rankings_path = ROOT / "data" / "ncaab" / "power_rankings_2026.csv"
    db_path = ROOT / "data" / "espn.db"

    if not bracket_path.exists():
        print(f"Bracket not found: {bracket_path}")
        return 1
    if not rankings_path.exists():
        print(f"Power rankings not found: {rankings_path}")
        return 1
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        return 1

    bracket_content = bracket_path.read_text()
    power_content = rankings_path.read_text()

    print("Running bracket analysis (10k sims, rank-delta adjustment ON) for Final Four rates...")
    out = run_bracket_analysis(
        bracket_content,
        power_content,
        n_sims=10_000,
        db_path=db_path,
        apply_rank_delta=True,
    )

    bracket = parse_bracket_csv(bracket_content)
    r1_matchups = bracket
    # final4_probabilities only includes teams with >0%; build full final4_pct with zeros for rest
    prob_list = out.get("final4_probabilities") or []
    final4_pct = {p["team"]: float(p["final4_pct"]) for p in prob_list}
    all_teams = set()
    for m in bracket:
        all_teams.add(m["team_a"])
        all_teams.add(m["team_b"])
    for t in all_teams:
        if t not in final4_pct:
            final4_pct[t] = 0.0

    team_to_seed = {}
    for m in r1_matchups:
        team_to_seed[m["team_a"]] = m["seed_a"]
        team_to_seed[m["team_b"]] = m["seed_b"]

    team_stats, rank_ft, rank_3p, rank_exp = load_march_stats(db_path)

    # Relaxed glitch: seed >= 7, F4 > 3%, Top 75 in at least 1 category
    glitch = []
    for team, pct in final4_pct.items():
        seed = team_to_seed.get(team)
        if seed is None or seed < 7 or pct <= RELAXED_F4_PCT_MIN:
            continue
        team_c = resolve_team_name(team, BRACKET_TEAM_ALIAS_MAP)
        n_top75 = 0
        r_ft = rank_ft.get(team_c, 999)
        r_3p = rank_3p.get(team_c, 999)
        r_exp = rank_exp.get(team_c, 999)
        if r_ft <= RELAXED_TOP_N:
            n_top75 += 1
        if r_3p <= RELAXED_TOP_N:
            n_top75 += 1
        if r_exp <= RELAXED_TOP_N:
            n_top75 += 1
        if n_top75 >= RELAXED_MIN_CATEGORIES:
            categories = []
            if r_ft <= RELAXED_TOP_N:
                categories.append("FT%")
            if r_3p <= RELAXED_TOP_N:
                categories.append("3P%")
            if r_exp <= RELAXED_TOP_N:
                categories.append("Exp")
            glitch.append({
                "team": team,
                "seed": seed,
                "final4_pct": round(pct, 1),
                "rank_ft": r_ft if r_ft < 999 else None,
                "rank_3p": r_3p if r_3p < 999 else None,
                "rank_exp": r_exp if r_exp < 999 else None,
                "qualify_categories": categories,
            })

    glitch.sort(key=lambda x: -x["final4_pct"])

    print(f"\n--- Relaxed Glitch Sleepers (seed 7+, F4 > {RELAXED_F4_PCT_MIN}%, Top {RELAXED_TOP_N} in ≥1 of FT%/3P%/Exp) ---")
    print("Using updated (rank-delta) simulation results.\n")
    print(f"Qualifying teams: {len(glitch)}\n")
    if glitch:
        print(f"{'Team':<22} {'Seed':>5} {'F4%':>6}  Qualifies in (Top 75)")
        print("-" * 70)
        for g in glitch:
            cat_str = ", ".join(g.get("qualify_categories", []))
            print(f"{g['team']:<22} {g['seed']:>5} {g['final4_pct']:>5.1f}%  {cat_str}")

    # Specific check for Penn, VCU, Santa Clara, SMU, Wright St.
    print("\n--- Specific check: Penn, VCU, Santa Clara, SMU, Wright St. ---")
    print(f"{'Team':<18} {'Seed':>5} {'F4%':>6}  FT_rank  3P_rank  Exp_rank  Qualified?")
    print("-" * 75)
    for team in CHECK_TEAMS:
        seed = team_to_seed.get(team)
        pct = final4_pct.get(team, 0.0)
        team_c = resolve_team_name(team, BRACKET_TEAM_ALIAS_MAP)
        r_ft = rank_ft.get(team_c, 999)
        r_3p = rank_3p.get(team_c, 999)
        r_exp = rank_exp.get(team_c, 999)
        if r_ft == 999:
            r_ft = None
        if r_3p == 999:
            r_3p = None
        if r_exp == 999:
            r_exp = None
        rft_s = str(r_ft) if r_ft is not None else "—"
        r3p_s = str(r_3p) if r_3p is not None else "—"
        rexp_s = str(r_exp) if r_exp is not None else "—"
        n_top75 = sum(1 for r in (r_ft or 999, r_3p or 999, r_exp or 999) if r <= RELAXED_TOP_N)
        qualified = (
            seed is not None and seed >= 7
            and pct > RELAXED_F4_PCT_MIN
            and n_top75 >= RELAXED_MIN_CATEGORIES
        )
        q_str = "Yes" if qualified else "No"
        seed_s = str(seed) if seed is not None else "—"
        print(f"{team:<18} {seed_s:>5} {pct:>5.1f}%  {rft_s:>8}  {r3p_s:>8}  {rexp_s:>8}  {q_str}")

    # Flag Penn and Wright St.: model spread disagrees sharply with market
    rankings = parse_power_rankings_csv(power_content)
    model_rank_by_canonical = {
        resolve_team_name(r["team"], BRACKET_TEAM_ALIAS_MAP): r["model_rank"]
        for r in rankings
    }
    team_to_seed_c = {}
    for m in bracket:
        team_to_seed_c[resolve_team_name(m["team_a"], BRACKET_TEAM_ALIAS_MAP)] = m["seed_a"]
        team_to_seed_c[resolve_team_name(m["team_b"], BRACKET_TEAM_ALIAS_MAP)] = m["seed_b"]
    potential_glitch = _potential_glitch_teams(
        bracket, rank_ft, rank_3p, rank_exp, BRACKET_TEAM_ALIAS_MAP
    )

    print("\n--- FLAG: Model vs market spread disagreement (Penn, Wright St.) ---")
    print("  Seed 7+ teams where the model spread disagrees sharply with the market.\n")
    for team_flag, seed_flag, opponent, underdog, market_spread_home in MODEL_VS_MARKET_FLAG:
        pct = final4_pct.get(team_flag, 0.0)
        raw = get_model_spread_for_matchup(
            opponent, underdog, db_path=db_path, game_date=None, alias_map=BRACKET_TEAM_ALIAS_MAP
        )
        if raw is not None:
            o_c = resolve_team_name(opponent, BRACKET_TEAM_ALIAS_MAP)
            u_c = resolve_team_name(underdog, BRACKET_TEAM_ALIAS_MAP)
            model_spread_home = _apply_tier_and_anchor(
                raw, o_c, u_c, model_rank_by_canonical, team_to_seed_c, potential_glitch
            )
            delta = abs(model_spread_home - market_spread_home)
            print(f"  {underdog} ({seed_flag}-seed, {pct:.1f}% FF)")
            print(f"    Matchup: {opponent} vs {underdog}")
            print(f"    Market spread (favorite): {market_spread_home:+.1f}   Model spread (favorite): {model_spread_home:+.1f}   Delta: {delta:.1f} pts")
            print(f"    >>> Model disagrees sharply with market; consider as glitch/sleeper for spread value.")
        else:
            print(f"  {team_flag} ({seed_flag}-seed, {pct:.1f}% FF) — no model spread available")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
