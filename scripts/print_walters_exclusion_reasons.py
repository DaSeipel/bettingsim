#!/usr/bin/env python3
"""
Print the exclusion reason for each Walters-excluded first-round matchup.
Confirm that exclusions are ONLY due to anchor hit (model spread = 10.0 or 22.0), not delta > 15.
Flag if Illinois vs Penn or Virginia vs Wright St. are excluded (they are 3-10 matchups; anchor does not apply).
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.bracket_analysis import (
    parse_bracket_csv,
    parse_power_rankings_csv,
    run_bracket_analysis,
)

# 3-10 matchups: anchor does NOT apply (only 1-4 vs 13-16). Must never be excluded for "anchor".
MUST_NOT_BE_EXCLUDED = [
    ("Illinois", "Penn"),
    ("Virginia", "Wright St."),
]


def _norm_pair(ta: str, tb: str) -> tuple[str, str]:
    return (ta.strip(), tb.strip())


def main() -> int:
    bracket_path = ROOT / "data" / "ncaab" / "bracket_2026.csv"
    rankings_path = ROOT / "data" / "ncaab" / "power_rankings_2026.csv"
    db_path = ROOT / "data" / "espn.db"

    if not bracket_path.exists() or not rankings_path.exists():
        print("Missing bracket or power rankings.")
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
        apply_rank_delta=True,
    )
    excluded = out.get("walters_data_errors") or []

    print("=" * 95)
    print("  Walters excluded matchups: exclusion reason and anchor check")
    print("  Exclusion is ONLY when the floor was applied (hit_anchor); no delta > 15 filter.")
    print("  Model spread shown is final (after rank-delta); can be >10/22 because rank-delta is applied after the floor.")
    print("=" * 95)
    print(f"  {'Matchup (Team A vs Team B)':<42} {'Seed':<8} {'Model':>7}  =10/22?  Reason")
    print("-" * 95)

    must_not = {_norm_pair(a, b) for a, b in MUST_NOT_BE_EXCLUDED}
    bug_found = []

    for e in excluded:
        ta, tb = e.get("team_a", ""), e.get("team_b", "")
        seed_a, seed_b = e.get("seed_a"), e.get("seed_b")
        model = e.get("model_spread")
        reason = e.get("excluded_reason", "—")
        seed_str = f"{seed_a}-{seed_b}" if seed_a is not None and seed_b is not None else "—"
        model_abs = abs(float(model)) if model is not None else None
        is_anchor = (model_abs == 10.0 or model_abs == 22.0) if model_abs is not None else False
        anchor_str = "yes" if is_anchor else "no (floor+rank_adj)"
        matchup = f"{ta} vs {tb}"
        if len(matchup) > 40:
            matchup = matchup[:37] + "..."
        print(f"  {matchup:<42} {seed_str:<8} {model:>+7.1f}  {anchor_str:<10}  {reason}")

        # 3-10 games (e.g. Illinois vs Penn, Virginia vs Wright St.) must not be excluded for anchor
        pair = _norm_pair(ta, tb)
        pair_rev = _norm_pair(tb, ta)
        if pair in must_not or pair_rev in must_not:
            bug_found.append(f"{ta} vs {tb} (seed {seed_a}-{seed_b}): excluded with reason={reason}, model={model}. Anchor applies only to 1-4 vs 13-16.")

    print("-" * 95)
    print(f"  Total excluded: {len(excluded)}")
    print()

    # Confirm: all should have excluded_reason "anchor" and model spread 10 or 22
    non_anchor_reason = [e for e in excluded if e.get("excluded_reason") != "anchor"]
    model_not_10_22 = []
    for e in excluded:
        m = e.get("model_spread")
        if m is not None:
            a = abs(float(m))
            if a != 10.0 and a != 22.0:
                model_not_10_22.append(e)

    print("  --- Verification ---")
    if non_anchor_reason:
        print(f"  BUG: {len(non_anchor_reason)} matchup(s) have excluded_reason != 'anchor': {[e.get('excluded_reason') for e in non_anchor_reason]}")
    else:
        print("  All excluded matchups have excluded_reason = 'anchor' (no delta > 15 filter).")
    if model_not_10_22:
        print(f"  {len(model_not_10_22)} excluded matchup(s) have final model_spread other than ±10 or ±22 (floor was applied, then rank-delta added; still correctly excluded).")
    else:
        print("  All excluded matchups have final model_spread = ±10.0 or ±22.0.")
    if bug_found:
        print()
        print("  *** BUG: The following matchups are 3-10 (anchor does not apply) but were excluded ***")
        for b in bug_found:
            print(f"    - {b}")
    else:
        print("  Illinois vs Penn and Virginia vs Wright St. are NOT in the excluded list (correct).")
    print()
    return 1 if bug_found or non_anchor_reason else 0


if __name__ == "__main__":
    sys.exit(main())
