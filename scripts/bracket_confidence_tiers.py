#!/usr/bin/env python3
"""
Read the filled bracket JSON and tag each of the 63 picks with a confidence tier:
  Lock:    win prob > 80%
  Strong:  65% < win prob <= 80%
  Lean:    55% < win prob <= 65%
  Coin Flip: 50% <= win prob <= 55%

Print summary (count per tier by round) and list every upset pick (lower seed winning) with its tier.
"""
from pathlib import Path
import sys
import json
import csv

ROOT = Path(__file__).resolve().parent.parent

# Confidence tiers: (min_pct, max_pct] for each tier (max_pct exclusive for Lock/Strong/Lean)
def _tier(pick_win_pct: float) -> str:
    if pick_win_pct > 80:
        return "Lock"
    if pick_win_pct > 65:
        return "Strong"
    if pick_win_pct > 55:
        return "Lean"
    return "Coin Flip"


def main() -> int:
    json_path = ROOT / "data" / "ncaab" / "bracket_filled_2026.json"
    bracket_path = ROOT / "data" / "ncaab" / "bracket_2026.csv"

    if not json_path.exists():
        print(f"Missing {json_path}. Run scripts/fill_bracket_2026.py first.")
        return 1
    if not bracket_path.exists():
        print(f"Missing {bracket_path}")
        return 1

    with open(json_path) as f:
        data = json.load(f)
    games = data.get("games", [])

    # Build team -> seed and first-round market spread from bracket CSV (64 teams)
    team_to_seed = {}
    matchup_to_market_spread = {}
    with open(bracket_path) as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 5 and row[0].strip() in ("East", "West", "South", "Midwest"):
                try:
                    seed_a, seed_b = int(float(row[1])), int(float(row[3]))
                    team_a, team_b = row[2].strip(), row[4].strip()
                    team_to_seed[team_a] = seed_a
                    team_to_seed[team_b] = seed_b
                    if len(row) >= 6:
                        try:
                            market_spread = float(row[5])
                        except ValueError:
                            market_spread = None
                        matchup_to_market_spread[(team_a, team_b)] = market_spread
                        if market_spread is not None:
                            matchup_to_market_spread[(team_b, team_a)] = -market_spread
                except (ValueError, IndexError):
                    pass

    round_names = {
        1: "Round of 64",
        2: "Round of 32",
        3: "Sweet 16",
        4: "Elite 8",
        5: "National Semifinals",
        6: "Championship",
    }

    # Tag each game with pick win pct, tier, upset flag, seed matchup, and market spread
    for g in games:
        pick = g.get("pick")
        pa, pb = g.get("win_pct_a"), g.get("win_pct_b")
        pick_win_pct = pa if pick == g.get("team_a") else pb
        g["pick_win_pct"] = pick_win_pct
        g["confidence_tier"] = _tier(pick_win_pct)
        loser = g["team_b"] if pick == g["team_a"] else g["team_a"]
        g["loser"] = loser
        winner_seed = team_to_seed.get(pick)
        loser_seed = team_to_seed.get(loser)
        g["winner_seed"] = winner_seed
        g["loser_seed"] = loser_seed
        g["is_upset"] = (winner_seed is not None and loser_seed is not None and winner_seed > loser_seed)
        seed_a = team_to_seed.get(g.get("team_a"))
        seed_b = team_to_seed.get(g.get("team_b"))
        if seed_a is not None and seed_b is not None:
            g["seed_matchup"] = f"{seed_a} vs {seed_b}"
        else:
            g["seed_matchup"] = None
        g["market_spread"] = matchup_to_market_spread.get(
            (g.get("team_a"), g.get("team_b"))
        )

    # Summary: count by round and tier
    tiers = ["Lock", "Strong", "Lean", "Coin Flip"]
    by_round_tier = {}  # round -> {tier: count}
    for g in games:
        r = g["round"]
        if r not in by_round_tier:
            by_round_tier[r] = {t: 0 for t in tiers}
        by_round_tier[r][g["confidence_tier"]] += 1

    upsets = [g for g in games if g.get("is_upset")]

    print("=" * 70)
    print("  Filled bracket: confidence tiers and upset picks")
    print("  Lock: >80%  |  Strong: 65–80%  |  Lean: 55–65%  |  Coin Flip: 50–55%")
    print("=" * 70)
    print("\n  --- Picks by confidence tier, by round ---")
    print(f"  {'Round':<22} {'Lock':>8} {'Strong':>8} {'Lean':>8} {'Coin Flip':>10}  Total")
    print("  " + "-" * 62)
    for r in sorted(by_round_tier.keys()):
        row = by_round_tier[r]
        total = sum(row.values())
        print(f"  {round_names.get(r, f'Round {r}'):<22} {row['Lock']:>8} {row['Strong']:>8} {row['Lean']:>8} {row['Coin Flip']:>10}  {total}")
    total_all = sum(sum(by_round_tier[r].values()) for r in by_round_tier)
    print("  " + "-" * 62)
    print(f"  {'Total':<22} {sum(by_round_tier[r]['Lock'] for r in by_round_tier):>8} {sum(by_round_tier[r]['Strong'] for r in by_round_tier):>8} {sum(by_round_tier[r]['Lean'] for r in by_round_tier):>8} {sum(by_round_tier[r]['Coin Flip'] for r in by_round_tier):>10}  {total_all}")

    print("\n  --- Upset picks (lower seed winning) ---")
    if not upsets:
        print("  None.")
    else:
        print(f"  {'Round':<20} {'Winner (seed)':<22} {'Loser (seed)':<22} {'Win%':>6}  Tier")
        print("  " + "-" * 82)
        for g in upsets:
            rname = round_names.get(g["round"], f"Round {g['round']}")
            winner = g["pick"]
            loser = g["loser"]
            ws = g.get("winner_seed")
            ls = g.get("loser_seed")
            w_str = f"{winner} ({ws})" if ws is not None else winner
            l_str = f"{loser} ({ls})" if ls is not None else loser
            pct = g.get("pick_win_pct")
            tier = g["confidence_tier"]
            print(f"  {rname:<20} {w_str:<22} {l_str:<22} {pct:>5.1f}%  {tier}")

    # Write display-ready JSON for app consumption
    display_path = ROOT / "data" / "ncaab" / "bracket_display_2026.json"
    with open(display_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nWrote display-ready bracket to {display_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
