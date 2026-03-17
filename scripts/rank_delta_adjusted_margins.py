#!/usr/bin/env python3
"""
Print adjusted margins for all first-round matchups excluded due to anchor (10/22-pt floor).
Shows XGB raw, tier+anchor margin, rank-delta adjustment, and final adjusted margin with the new layer.
Uses RANK_DELTA_ADJUSTMENT_* constants from bracket_analysis.
"""
from pathlib import Path
import sys
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.bracket_analysis import (
    parse_bracket_csv,
    parse_power_rankings_csv,
    run_bracket_analysis,
    resolve_team_name,
    get_model_spread_for_matchup,
    BRACKET_TEAM_ALIAS_MAP,
    RANK_DELTA_ADJUSTMENT_THRESHOLD,
    RANK_DELTA_ADJUSTMENT_SCALE,
    RANK_DELTA_ADJUSTMENT_CAP,
)


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

    bracket_content = bracket_path.read_text()
    power_content = rankings_path.read_text()

    out = run_bracket_analysis(
        bracket_content,
        power_content,
        n_sims=100,
        db_path=db_path,
    )
    excluded = out.get("walters_data_errors", [])  # excluded due to anchor

    rankings = parse_power_rankings_csv(power_content)
    model_rank_by_canonical = {
        resolve_team_name(r["team"], BRACKET_TEAM_ALIAS_MAP): r["model_rank"]
        for r in rankings
    }

    def rank_delta_adj_pts(rank_a: Optional[int], rank_b: Optional[int]) -> tuple[float, int]:
        """Adjustment points toward better-ranked team; rank_delta. Returns (adj_pts, rank_delta)."""
        if rank_a is None or rank_b is None:
            return 0.0, 0
        delta = abs(rank_a - rank_b)
        if delta <= RANK_DELTA_ADJUSTMENT_THRESHOLD:
            return 0.0, delta
        pts = min((delta / 10.0) * RANK_DELTA_ADJUSTMENT_SCALE, RANK_DELTA_ADJUSTMENT_CAP)
        return pts, delta

    def tier_anchor_from_final(final_margin: float, rank_a: Optional[int], rank_b: Optional[int]) -> float:
        """Back out tier+anchor margin from final (final = tier_anchor + sign*adj_pts)."""
        adj_pts, _ = rank_delta_adj_pts(rank_a, rank_b)
        if adj_pts == 0:
            return final_margin
        if rank_a is None or rank_b is None:
            return final_margin
        # We added +adj_pts if team_a better (rank_a < rank_b), else -adj_pts
        sign = 1 if rank_a < rank_b else -1
        return final_margin - sign * adj_pts

    print("=" * 115)
    print("  Adjusted margins with rank-delta layer (excluded anchor matchups)")
    print("  Rule: rank_delta > 50 => add (rank_delta/10)*0.8 toward better-ranked team, cap 15 pts")
    print("=" * 115)
    print(f"  {'Team A':<20} {'Team B':<18} {'Rk A':>5} {'Rk B':>5} {'Δ':>4}  {'XGB raw':>8}  {'Tier+Anch':>9}  {'Rank adj':>8}  {'Adjusted':>9}  {'Market':>7}")
    print("-" * 115)

    for e in excluded:
        ta, tb = e.get("team_a", ""), e.get("team_b", "")
        ta_c = resolve_team_name(ta, BRACKET_TEAM_ALIAS_MAP)
        tb_c = resolve_team_name(tb, BRACKET_TEAM_ALIAS_MAP)
        rank_a = model_rank_by_canonical.get(ta_c)
        rank_b = model_rank_by_canonical.get(tb_c)
        market = e.get("market_spread")
        model_spread = e.get("model_spread")  # final (tier+anchor+rank_delta)

        xg_raw = get_model_spread_for_matchup(
            ta, tb, db_path=db_path, game_date=None, alias_map=BRACKET_TEAM_ALIAS_MAP
        )
        adj_pts, rdelta = rank_delta_adj_pts(rank_a, rank_b)
        tier_anchor = tier_anchor_from_final(float(model_spread), rank_a, rank_b) if model_spread is not None else None

        ra_s = str(rank_a) if rank_a is not None else "—"
        rb_s = str(rank_b) if rank_b is not None else "—"
        xg_s = f"{xg_raw:+.2f}" if xg_raw is not None else "—"
        ta_s = f"{tier_anchor:+.2f}" if tier_anchor is not None else "—"
        adj_s = f"{adj_pts:+.1f}" if adj_pts else "0.0"
        final_s = f"{model_spread:+.1f}" if model_spread is not None else "—"
        mkt_s = f"{market:+.1f}" if market is not None else "—"

        print(f"  {ta:<20} {tb:<18} {ra_s:>5} {rb_s:>5} {rdelta:>4}  {xg_s:>8}  {ta_s:>9}  {adj_s:>8}  {final_s:>9}  {mkt_s:>7}")

    print("-" * 115)
    print(f"  Total excluded (anchor) matchups: {len(excluded)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
