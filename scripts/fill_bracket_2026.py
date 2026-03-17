#!/usr/bin/env python3
"""
Using updated bracket simulation (rank-delta on, 20,000 iterations), fill the entire 2026
NCAA tournament bracket from Round of 64 through Championship. For each matchup, pick the
team with the higher win probability from the simulation. Print full bracket by round/region
and export to data/ncaab/bracket_filled_2026.json.
"""
from pathlib import Path
import sys
import json
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.bracket_analysis import (
    parse_bracket_csv,
    parse_power_rankings_csv,
    load_march_stats,
    get_model_spread_for_matchup,
    _apply_tier_and_anchor,
    _build_get_winner_with_march_factors,
    run_monte_carlo_bracket_with_game_probs,
    margin_to_win_prob,
    _potential_glitch_teams,
    BRACKET_TEAM_ALIAS_MAP,
    resolve_team_name,
)

N_SIMS = 20_000
REGIONS = ["East", "West", "South", "Midwest"]


def _region_for_game(round_idx: int, game_idx: int) -> str:
    """Region for a game: R1-R4 by bracket slot; R5/R6 = National."""
    if round_idx >= 4:
        return "National"
    # R1: 32 games -> 8 per region. R2: 16 -> 4 per region. R3: 8 -> 2. R4: 4 -> 1.
    games_per_region = 8 // (2 ** round_idx)
    return REGIONS[game_idx // games_per_region]


def main() -> int:
    import io
    bracket_path = ROOT / "data" / "ncaab" / "bracket_2026.csv"
    rankings_path = ROOT / "data" / "ncaab" / "power_rankings_2026.csv"
    db_path = ROOT / "data" / "espn.db"
    out_path = ROOT / "data" / "ncaab" / "bracket_filled_2026.json"

    if not bracket_path.exists() or not rankings_path.exists() or not db_path.exists():
        print("Missing bracket_2026.csv, power_rankings_2026.csv, or espn.db")
        return 1

    bracket_content = bracket_path.read_text()
    bracket = parse_bracket_csv(bracket_content)
    if len(bracket) < 32:
        print("Bracket has fewer than 32 first-round matchups.")
        return 1

    # Add region to each R1 game (from CSV first column, else East 0-7, West 8-15, etc.)
    import csv
    with open(bracket_path) as f:
        rows = list(csv.reader(f))
    for i, m in enumerate(bracket):
        if i + 1 < len(rows) and rows[i + 1] and rows[i + 1][0].strip() in REGIONS:
            m["region"] = rows[i + 1][0].strip()
        else:
            m["region"] = REGIONS[i // 8]

    rankings = parse_power_rankings_csv(rankings_path.read_text())
    team_stats, rank_ft, rank_3p, rank_exp = load_march_stats(db_path)
    model_rank_by_canonical = {resolve_team_name(r["team"], BRACKET_TEAM_ALIAS_MAP): r["model_rank"] for r in rankings}
    team_to_seed = {}
    for m in bracket:
        team_to_seed[resolve_team_name(m["team_a"], BRACKET_TEAM_ALIAS_MAP)] = m["seed_a"]
        team_to_seed[resolve_team_name(m["team_b"], BRACKET_TEAM_ALIAS_MAP)] = m["seed_b"]
    potential_glitch = _potential_glitch_teams(bracket, rank_ft, rank_3p, rank_exp, BRACKET_TEAM_ALIAS_MAP)
    margin_cache = {}

    get_winner = _build_get_winner_with_march_factors(
        db_path, None, team_stats, rank_ft, rank_3p, margin_cache, BRACKET_TEAM_ALIAS_MAP,
        model_rank_by_canonical=model_rank_by_canonical,
        team_to_seed=team_to_seed,
        potential_glitch=potential_glitch,
        apply_rank_delta=True,
    )

    print("Running 20,000-iteration bracket simulation (rank-delta ON)...")
    result = run_monte_carlo_bracket_with_game_probs(bracket, get_winner, n_sims=N_SIMS)
    r1_win_probs = result["r1_win_probs"]
    matchup_win_probs = result["matchup_win_probs"]

    def get_fallback_win_pct_a(ta: str, tb: str) -> Optional[float]:
        raw = get_model_spread_for_matchup(ta, tb, db_path=db_path, game_date=None, alias_map=BRACKET_TEAM_ALIAS_MAP)
        if raw is None:
            return None
        ta_c = resolve_team_name(ta, BRACKET_TEAM_ALIAS_MAP)
        tb_c = resolve_team_name(tb, BRACKET_TEAM_ALIAS_MAP)
        margin = _apply_tier_and_anchor(raw, ta_c, tb_c, model_rank_by_canonical, team_to_seed, potential_glitch, apply_rank_delta=True)
        return margin_to_win_prob(margin) * 100.0

    # Fill bracket round by round
    all_games = []
    winners_by_round = [None] * 7  # index 0 unused; 1=R1(32), 2=R2(16), ...
    winners_by_round[1] = []

    # Round 1 (Round of 64)
    for i in range(32):
        m = bracket[i]
        ta, tb = m["team_a"], m["team_b"]
        wp = r1_win_probs[i]
        pct_a, pct_b = wp["win_pct_a"], wp["win_pct_b"]
        winner = ta if pct_a >= 50.0 else tb
        winners_by_round[1].append(winner)
        all_games.append({
            "round": 1,
            "round_name": "Round of 64",
            "region": m["region"],
            "game_idx": i,
            "team_a": ta,
            "team_b": tb,
            "win_pct_a": pct_a,
            "win_pct_b": pct_b,
            "pick": winner,
        })

    # R2 (Round of 32)
    winners_by_round[2] = []
    for j in range(16):
        ta, tb = winners_by_round[1][2 * j], winners_by_round[1][2 * j + 1]
        key = (1, j, ta, tb)
        key_rev = (1, j, tb, ta)
        if key in matchup_win_probs:
            wins_a, total = matchup_win_probs[key]
            pct_a = (wins_a / total) * 100.0
        elif key_rev in matchup_win_probs:
            wins_b, total = matchup_win_probs[key_rev]
            pct_a = 100.0 - (wins_b / total) * 100.0
        else:
            pct_a = get_fallback_win_pct_a(ta, tb) or 50.0
        pct_b = 100.0 - pct_a
        winner = ta if pct_a >= 50.0 else tb
        winners_by_round[2].append(winner)
        all_games.append({
            "round": 2,
            "round_name": "Round of 32",
            "region": _region_for_game(1, j),
            "game_idx": j,
            "team_a": ta,
            "team_b": tb,
            "win_pct_a": round(pct_a, 1),
            "win_pct_b": round(pct_b, 1),
            "pick": winner,
        })

    # R3 (Sweet 16)
    winners_by_round[3] = []
    for j in range(8):
        ta, tb = winners_by_round[2][2 * j], winners_by_round[2][2 * j + 1]
        key = (2, j, ta, tb)
        key_rev = (2, j, tb, ta)
        if key in matchup_win_probs:
            wins_a, total = matchup_win_probs[key]
            pct_a = (wins_a / total) * 100.0
        elif key_rev in matchup_win_probs:
            wins_b, total = matchup_win_probs[key_rev]
            pct_a = 100.0 - (wins_b / total) * 100.0
        else:
            pct_a = get_fallback_win_pct_a(ta, tb) or 50.0
        pct_b = 100.0 - pct_a
        winner = ta if pct_a >= 50.0 else tb
        winners_by_round[3].append(winner)
        all_games.append({
            "round": 3,
            "round_name": "Sweet 16",
            "region": _region_for_game(2, j),
            "game_idx": j,
            "team_a": ta,
            "team_b": tb,
            "win_pct_a": round(pct_a, 1),
            "win_pct_b": round(pct_b, 1),
            "pick": winner,
        })

    # R4 (Elite 8 -> Final Four)
    winners_by_round[4] = []
    for j in range(4):
        ta, tb = winners_by_round[3][2 * j], winners_by_round[3][2 * j + 1]
        key = (3, j, ta, tb)
        key_rev = (3, j, tb, ta)
        if key in matchup_win_probs:
            wins_a, total = matchup_win_probs[key]
            pct_a = (wins_a / total) * 100.0
        elif key_rev in matchup_win_probs:
            wins_b, total = matchup_win_probs[key_rev]
            pct_a = 100.0 - (wins_b / total) * 100.0
        else:
            pct_a = get_fallback_win_pct_a(ta, tb) or 50.0
        pct_b = 100.0 - pct_a
        winner = ta if pct_a >= 50.0 else tb
        winners_by_round[4].append(winner)
        all_games.append({
            "round": 4,
            "round_name": "Elite 8",
            "region": _region_for_game(3, j),
            "game_idx": j,
            "team_a": ta,
            "team_b": tb,
            "win_pct_a": round(pct_a, 1),
            "win_pct_b": round(pct_b, 1),
            "pick": winner,
        })

    # R5 (National Semifinals): East vs South, West vs Midwest (2026 NCAA bracket)
    winners_by_round[5] = []
    semi_pairs = [(0, 2), (1, 3)]  # (East vs South), (West vs Midwest)
    for j, (idx_a, idx_b) in enumerate(semi_pairs):
        ta, tb = winners_by_round[4][idx_a], winners_by_round[4][idx_b]
        key = (4, j, ta, tb)
        key_rev = (4, j, tb, ta)
        if key in matchup_win_probs:
            wins_a, total = matchup_win_probs[key]
            pct_a = (wins_a / total) * 100.0
        elif key_rev in matchup_win_probs:
            wins_b, total = matchup_win_probs[key_rev]
            pct_a = 100.0 - (wins_b / total) * 100.0
        else:
            pct_a = get_fallback_win_pct_a(ta, tb) or 50.0
        pct_b = 100.0 - pct_a
        winner = ta if pct_a >= 50.0 else tb
        winners_by_round[5].append(winner)
        all_games.append({
            "round": 5,
            "round_name": "National Semifinals",
            "region": "National",
            "game_idx": j,
            "team_a": ta,
            "team_b": tb,
            "win_pct_a": round(pct_a, 1),
            "win_pct_b": round(pct_b, 1),
            "pick": winner,
        })

    # R6 (Championship)
    ta, tb = winners_by_round[5][0], winners_by_round[5][1]
    key = (5, 0, ta, tb)
    key_rev = (5, 0, tb, ta)
    if key in matchup_win_probs:
        wins_a, total = matchup_win_probs[key]
        pct_a = (wins_a / total) * 100.0
    elif key_rev in matchup_win_probs:
        wins_b, total = matchup_win_probs[key_rev]
        pct_a = 100.0 - (wins_b / total) * 100.0
    else:
        pct_a = get_fallback_win_pct_a(ta, tb) or 50.0
    pct_b = 100.0 - pct_a
    champion = ta if pct_a >= 50.0 else tb
    all_games.append({
        "round": 6,
        "round_name": "Championship",
        "region": "National",
        "game_idx": 0,
        "team_a": ta,
        "team_b": tb,
        "win_pct_a": round(pct_a, 1),
        "win_pct_b": round(pct_b, 1),
        "pick": champion,
    })

    # --- Print bracket view ---
    print()
    print("=" * 90)
    print("  2026 NCAA Tournament — Filled Bracket (rank-delta sim, 20k iterations)")
    print("  Each game: higher win-probability team from simulation is the pick.")
    print("=" * 90)

    for region in REGIONS:
        print(f"\n  --- {region} Region (Round of 64 → Elite 8) ---")
        for g in all_games:
            if g["region"] != region or g["round"] > 4:
                continue
            rname = g["round_name"]
            ta, tb = g["team_a"], g["team_b"]
            pa, pb = g["win_pct_a"], g["win_pct_b"]
            pick = g["pick"]
            print(f"    [{rname}] {ta} ({pa:.1f}%) vs {tb} ({pb:.1f}%)  →  Pick: {pick}")

    print("\n  --- Final Four (National Semifinals) ---")
    for g in all_games:
        if g["round"] != 5:
            continue
        ta, tb = g["team_a"], g["team_b"]
        pa, pb = g["win_pct_a"], g["win_pct_b"]
        print(f"    {ta} ({pa:.1f}%) vs {tb} ({pb:.1f}%)  →  Pick: {g['pick']}")

    print("\n  --- Championship ---")
    g = all_games[-1]
    ta, tb = g["team_a"], g["team_b"]
    pa, pb = g["win_pct_a"], g["win_pct_b"]
    print(f"    {ta} ({pa:.1f}%) vs {tb} ({pb:.1f}%)  →  Champion: {g['pick']}")

    # Round-by-round summary
    print("\n" + "=" * 90)
    print("  Round-by-round winners summary")
    print("=" * 90)
    print("  Round of 64 (32 winners): " + ", ".join(winners_by_round[1]))
    print("  Round of 32 (16 winners): " + ", ".join(winners_by_round[2]))
    print("  Sweet 16 (8 winners):     " + ", ".join(winners_by_round[3]))
    print("  Elite 8 / Final Four (4):  " + ", ".join(winners_by_round[4]))
    print("  National Semifinals (2):  " + ", ".join(winners_by_round[5]))
    print("  Champion:                 " + champion)

    # Export JSON
    export = {
        "n_sims": N_SIMS,
        "rank_delta_on": True,
        "games": all_games,
        "champion": champion,
        "final_four": list(winners_by_round[4]),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"\n  Exported to {out_path}")

    # Export flat winner list for ESPN bracket entry
    round_labels = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"]
    flat_path = out_path.parent / "bracket_winners_flat.txt"
    with open(flat_path, "w") as f:
        last_r = None
        for g in all_games:
            r = g.get("round")
            if r != last_r:
                if last_r is not None:
                    f.write("\n")
                f.write(round_labels[r - 1] + "\n")
                last_r = r
            f.write(g.get("pick", "") + "\n")
    print(f"  Exported to {flat_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
