#!/usr/bin/env python3
"""
After the power-ranking (rank-delta) adjustment: compare model spreads to market spreads
for all 32 first-round matchups. Print correlation, MAE overall, and MAE by group:
- Large seed mismatch (1-4 vs 13-16): goal within 5 pts of market on average
- Close matchups (4-5, 5-4, 6-7, 7-6, 8-9, 9-8): preserve existing accuracy
"""
from pathlib import Path
import sys
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from engine.bracket_analysis import (
    parse_bracket_csv,
    parse_power_rankings_csv,
    load_march_stats,
    get_model_spread_for_matchup,
    _apply_tier_and_anchor,
    _potential_glitch_teams,
    BRACKET_TEAM_ALIAS_MAP,
    resolve_team_name,
)


def _market_margin(market_spread: Optional[float]) -> Optional[float]:
    """Bracket CSV stores spread (negative when team_a favored). Convert to margin team_a - team_b."""
    if market_spread is None:
        return None
    # Convention: team_a -27.5 means team_a favored by 27.5 => margin +27.5
    return -float(market_spread)


def _is_large_seed_mismatch(seed_a: int, seed_b: int) -> bool:
    """1-4 vs 13-16."""
    lo, hi = min(seed_a, seed_b), max(seed_a, seed_b)
    return lo <= 4 and hi >= 13


def _is_close_matchup(seed_a: int, seed_b: int) -> bool:
    """Seeds 4-5, 5-4, 6-7, 7-6, 8-9, 9-8 (pair diff = 1, both in 4-9)."""
    lo, hi = min(seed_a, seed_b), max(seed_a, seed_b)
    return (hi - lo == 1) and (4 <= lo and hi <= 9)


def main() -> int:
    bracket_path = ROOT / "data" / "ncaab" / "bracket_2026.csv"
    rankings_path = ROOT / "data" / "ncaab" / "power_rankings_2026.csv"
    db_path = ROOT / "data" / "espn.db"

    if not bracket_path.exists() or not rankings_path.exists():
        print("Missing bracket_2026.csv or power_rankings_2026.csv")
        return 1
    if not db_path.exists():
        print("Missing espn.db")
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

    rows = []
    for m in bracket:
        ta, tb = m["team_a"], m["team_b"]
        seed_a, seed_b = m["seed_a"], m["seed_b"]
        ta_c = resolve_team_name(ta, BRACKET_TEAM_ALIAS_MAP)
        tb_c = resolve_team_name(tb, BRACKET_TEAM_ALIAS_MAP)
        market_raw = m.get("market_spread")
        market_margin = _market_margin(market_raw)

        raw = get_model_spread_for_matchup(
            ta, tb, db_path=db_path, game_date=None, alias_map=BRACKET_TEAM_ALIAS_MAP
        )
        model_margin = None
        if raw is not None:
            model_margin = _apply_tier_and_anchor(
                raw, ta_c, tb_c, model_rank_by_canonical, team_to_seed, potential_glitch
            )
        rows.append({
            "team_a": ta,
            "team_b": tb,
            "seed_a": seed_a,
            "seed_b": seed_b,
            "market_margin": market_margin,
            "model_margin": model_margin,
            "large_mismatch": _is_large_seed_mismatch(seed_a, seed_b),
            "close": _is_close_matchup(seed_a, seed_b),
        })

    # Drop rows missing market or model for correlation/MAE
    valid = [r for r in rows if r["market_margin"] is not None and r["model_margin"] is not None]
    if not valid:
        print("No matchups with both market and model spread.")
        return 1

    market_arr = np.array([r["market_margin"] for r in valid])
    model_arr = np.array([r["model_margin"] for r in valid])
    errors = model_arr - market_arr
    mae_all = float(np.mean(np.abs(errors)))
    corr = float(np.corrcoef(market_arr, model_arr)[0, 1]) if len(valid) > 1 else 0.0

    large = [r for r in valid if r["large_mismatch"]]
    close = [r for r in valid if r["close"]]
    other = [r for r in valid if not r["large_mismatch"] and not r["close"]]

    mae_large, n_large = (float(np.mean(np.abs([r["model_margin"] - r["market_margin"] for r in large]))), len(large)) if large else (float("nan"), 0)
    mae_close, n_close = (float(np.mean(np.abs([r["model_margin"] - r["market_margin"] for r in close]))), len(close)) if close else (float("nan"), 0)
    mae_other, n_other = (float(np.mean(np.abs([r["model_margin"] - r["market_margin"] for r in other]))), len(other)) if other else (float("nan"), 0)

    # Correlation table: model vs market (and summary stats)
    print("=" * 100)
    print("  Model vs market (32 first-round matchups) — UPDATED model spread (with rank-delta)")
    print("  Margin = team_a - team_b (positive = team_a favored). Model = tier+anchor+rank_delta.")
    print("=" * 100)
    print()
    print("  --- All 32 matchups: Market spread vs Updated model spread ---")
    print(f"  {'Matchup (Team A vs Team B)':<42} {'Seed':<8} {'Market':>8} {'Model':>8} {'Error':>8}  Group")
    print("-" * 100)
    for r in rows:
        mkt = r["market_margin"]
        mod = r["model_margin"]
        err = (mod - mkt) if (mkt is not None and mod is not None) else None
        seed = f"{r['seed_a']}-{r['seed_b']}"
        mkt_s = f"{mkt:+.1f}" if mkt is not None else "—"
        mod_s = f"{mod:+.1f}" if mod is not None else "—"
        err_s = f"{err:+.1f}" if err is not None else "—"
        grp = "large" if r["large_mismatch"] else ("close" if r["close"] else "other")
        matchup = f"{r['team_a']} vs {r['team_b']}"
        if len(matchup) > 40:
            matchup = matchup[:37] + "..."
        print(f"  {matchup:<42} {seed:<8} {mkt_s:>8} {mod_s:>8} {err_s:>8}  {grp}")
    print("-" * 100)
    print()
    print("  --- Correlation and average absolute error ---")
    print(f"  Correlation (model vs market):  {corr:.3f}")
    print(f"  MAE (all matchups):              {mae_all:.2f} pts  (n={len(valid)})")
    print()
    print("  --- MAE by group (goal: large mismatch ≤5 pts, preserve close accuracy) ---")
    print(f"  Large seed mismatch (1-4 vs 13-16):  MAE = {mae_large:.2f} pts  (n={n_large})  {'✓ ≤5' if n_large and mae_large <= 5 else ('—' if n_large else 'N/A')}")
    print(f"  Close matchups (4-5, 6-7, 8-9):      MAE = {mae_close:.2f} pts  (n={n_close})")
    print(f"  Other:                               MAE = {mae_other:.2f} pts  (n={n_other})")
    print()
    print("  --- Correlation table (model margin vs market margin) ---")
    print(f"  Pearson r (all valid): {corr:.3f}")
    if large:
        mkt_l = np.array([r["market_margin"] for r in large])
        mod_l = np.array([r["model_margin"] for r in large])
        r_large = float(np.corrcoef(mkt_l, mod_l)[0, 1]) if len(large) > 1 else float("nan")
        print(f"  Pearson r (large mismatch only): {r_large:.3f}")
    if close:
        mkt_c = np.array([r["market_margin"] for r in close])
        mod_c = np.array([r["model_margin"] for r in close])
        r_close = float(np.corrcoef(mkt_c, mod_c)[0, 1]) if len(close) > 1 else float("nan")
        print(f"  Pearson r (close matchups only): {r_close:.3f}")
    print()
    print("  --- Verdict: Big mismatches vs close games ---")
    if n_large and not np.isnan(mae_large):
        print(f"  Big mismatches (1-4 vs 13-16): MAE = {mae_large:.2f} pts. Goal ≤5 pts.")
    if n_close and not np.isnan(mae_close):
        print(f"  Close games (4-5, 6-7, 8-9):   MAE = {mae_close:.2f} pts. Goal: preserve disagreement (model can still find value).")
    if n_large and n_close and not np.isnan(mae_large) and not np.isnan(mae_close):
        if mae_large <= 5 and mae_close <= 6:
            print("  => Closer to market on big mismatches while preserving reasonable accuracy on close games.")
        elif mae_large < mae_close:
            print("  => Model is closer to market on big mismatches than on close games (rank-delta helps blowouts).")
        else:
            print("  => Large-mismatch MAE still above 5 pts; close-game MAE preserved.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
