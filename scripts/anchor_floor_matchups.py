#!/usr/bin/env python3
"""
Print every matchup where the model spread equals exactly 22.0 or 10.0 (floor anchors).
Show the raw model spread without the floor (after tier, before anchor) and power ranking gap.
Discuss scaling floors by rank gap.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.bracket_analysis import (
    parse_bracket_csv,
    parse_power_rankings_csv,
    load_march_stats,
    get_model_spread_for_matchup,
    _apply_tier_only,
    _apply_tier_and_anchor_with_flag,
    _potential_glitch_teams,
    BRACKET_TEAM_ALIAS_MAP,
    resolve_team_name,
    ANCHOR_MIN_ELITE_VS_LOW,
    ANCHOR_MIN_ELITE_VS_LOW_BIG_TIER,
    TIER_ANCHOR_RANK_DIFF_FOR_22,
)


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

    hit_22 = []
    hit_10 = []

    for m in bracket:
        ta, tb = m["team_a"], m["team_b"]
        ta_c = resolve_team_name(ta, BRACKET_TEAM_ALIAS_MAP)
        tb_c = resolve_team_name(tb, BRACKET_TEAM_ALIAS_MAP)

        raw = get_model_spread_for_matchup(
            ta, tb, db_path=db_path, game_date=None, alias_map=BRACKET_TEAM_ALIAS_MAP
        )
        if raw is None:
            continue

        spread_without_floor = _apply_tier_only(
            raw, ta_c, tb_c, model_rank_by_canonical
        )
        final, hit_anchor = _apply_tier_and_anchor_with_flag(
            raw, ta_c, tb_c, model_rank_by_canonical, team_to_seed, potential_glitch
        )

        if not hit_anchor:
            continue
        if abs(final) != 22.0 and abs(final) != 10.0:
            continue

        rank_a = model_rank_by_canonical.get(ta_c)
        rank_b = model_rank_by_canonical.get(tb_c)
        rank_gap = abs((rank_a or 0) - (rank_b or 0))

        row = {
            "team_a": ta,
            "team_b": tb,
            "seed_a": m["seed_a"],
            "seed_b": m["seed_b"],
            "rank_a": rank_a,
            "rank_b": rank_b,
            "rank_gap": rank_gap,
            "xg_raw": round(float(raw), 2),
            "spread_without_floor": round(spread_without_floor, 2),
            "final_spread": round(final, 1),
        }
        if abs(final) == 22.0:
            hit_22.append(row)
        else:
            hit_10.append(row)

    print("=" * 90)
    print("  Matchups where model spread = 22.0 (floor anchor)")
    print("=" * 90)
    print(f"  {'Team A':<20} {'Team B':<18} {'Seed':<6} {'Rank A':>7} {'Rank B':>7} {'Gap':>5}  {'XG raw':>8}  {'No floor':>9}  Final")
    print("-" * 90)
    for r in hit_22:
        seed = f"{r['seed_a']}-{r['seed_b']}"
        ra = r["rank_a"] if r["rank_a"] is not None else "—"
        rb = r["rank_b"] if r["rank_b"] is not None else "—"
        gap = r["rank_gap"] if r["rank_a"] is not None and r["rank_b"] is not None else "—"
        print(f"  {r['team_a']:<20} {r['team_b']:<18} {seed:<6} {str(ra):>7} {str(rb):>7} {str(gap):>5}  {r['xg_raw']:>+8.2f}  {r['spread_without_floor']:>+9.2f}  {r['final_spread']:+.1f}")

    print()
    print("=" * 90)
    print("  Matchups where model spread = 10.0 (floor anchor)")
    print("=" * 90)
    print(f"  {'Team A':<20} {'Team B':<18} {'Seed':<6} {'Rank A':>7} {'Rank B':>7} {'Gap':>5}  {'XG raw':>8}  {'No floor':>9}  Final")
    print("-" * 90)
    for r in hit_10:
        seed = f"{r['seed_a']}-{r['seed_b']}"
        ra = r["rank_a"] if r["rank_a"] is not None else "—"
        rb = r["rank_b"] if r["rank_b"] is not None else "—"
        gap = r["rank_gap"] if r["rank_a"] is not None and r["rank_b"] is not None else "—"
        print(f"  {r['team_a']:<20} {r['team_b']:<18} {seed:<6} {str(ra):>7} {str(rb):>7} {str(gap):>5}  {r['xg_raw']:>+8.2f}  {r['spread_without_floor']:>+9.2f}  {r['final_spread']:+.1f}")

    print()
    print("=" * 90)
    print("  Should the floors scale by power ranking gap?")
    print("=" * 90)
    print("  Current logic: 1-4 vs 13-16 use a fixed floor (10 pts if rank_diff <= 50, else 22 pts).")
    print("  So we use only two values regardless of whether the gap is 50 or 350.")
    print()
    print("  Option — scale floor with rank gap:")
    print("  - Linear: floor = min(22, max(10, 10 + (rank_gap - 50) * k)) for rank_gap > 50.")
    print("  - Or: floor = 10 + 12 * min(1, (rank_gap - 50) / 200) so 50→10, 250→22, smooth in between.")
    print("  That way a 1-seed vs 16-seed with rank_gap 350 still gets ~22, while a 4 vs 13 with")
    print("  rank_gap 80 might get ~12–14 instead of jumping to 22 when rank_diff > 50.")
    print()
    print("  Summary: rank_gap varies a lot (e.g. Duke vs Siena vs Gonzaga vs Kennesaw St.).")
    print("  Scaling floor by gap would make anchors less blunt and more aligned with team strength.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
