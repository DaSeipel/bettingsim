#!/usr/bin/env python3
"""
Re-run the full 20,000-iteration bracket simulation with the power-ranking adjustment in place.
Run once without rank-delta (original) and once with (updated), then print side-by-side:
  - Final Four probabilities
  - Walters Plays (with corrected exclusion filter: exclude only when model hit 10/22-pt anchor)
  - Glitch Sleepers (glitch_teams from run_bracket_analysis)
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

N_SIMS = 20_000


def _fmt_f4(rows: list, max_rows: int = 20) -> list[str]:
    out = []
    for r in (rows or [])[:max_rows]:
        out.append(f"  {r['team']}: {r['final4_pct']}%")
    return out


def _fmt_walters(rows: list) -> list[str]:
    out = []
    for w in rows or []:
        out.append(f"  {w.get('team_a')} vs {w.get('team_b')}  Mkt {w.get('market_spread')}  Model {w.get('model_spread')}  Δ{w.get('diff_pts')}")
    return out


def _fmt_glitch(rows: list) -> list[str]:
    out = []
    for g in rows or []:
        out.append(f"  {g.get('team')} (seed {g.get('seed')})  F4 {g.get('final4_pct')}%")
    return out


def _fmt_excluded(rows: list) -> list[str]:
    out = []
    for e in rows or []:
        out.append(f"  {e.get('team_a')} vs {e.get('team_b')}  model {e.get('model_spread')}  (anchor)")
    return out


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

    print("=" * 100)
    print("  Bracket simulation: Original (no rank-delta) vs Updated (with rank-delta adjustment)")
    print(f"  {N_SIMS:,} iterations each. Exclusion filter: Walters exclude only when model hit 10/22-pt anchor.")
    print("=" * 100)

    print("\nRunning ORIGINAL (no rank-delta adjustment)...")
    original = run_bracket_analysis(
        bracket_content,
        power_content,
        n_sims=N_SIMS,
        db_path=db_path,
        apply_rank_delta=False,
    )
    print("  Done.")

    print("\nRunning UPDATED (with rank-delta adjustment)...")
    updated = run_bracket_analysis(
        bracket_content,
        power_content,
        n_sims=N_SIMS,
        db_path=db_path,
        apply_rank_delta=True,
    )
    print("  Done.")

    # --- Final Four probabilities side-by-side ---
    print("\n" + "=" * 100)
    print("  Final Four probabilities (top 20)")
    print("=" * 100)
    o_f4 = _fmt_f4(original.get("final4_probabilities") or [], 20)
    u_f4 = _fmt_f4(updated.get("final4_probabilities") or [], 20)
    n = max(len(o_f4), len(u_f4))
    print(f"  {'ORIGINAL (no rank-delta)':<48}  {'UPDATED (with rank-delta)':<48}")
    print("  " + "-" * 46 + "  " + "-" * 46)
    for i in range(n):
        left = o_f4[i] if i < len(o_f4) else ""
        right = u_f4[i] if i < len(u_f4) else ""
        if not left:
            left = "  —"
        if not right:
            right = "  —"
        print(f"  {left:<48}  {right:<48}")
    print()

    # --- Walters Plays side-by-side ---
    print("=" * 100)
    print("  Walters Plays (delta 2–7 pts; exclude only when model hit anchor)")
    print("=" * 100)
    o_w = _fmt_walters(original.get("walters_plays") or [])
    u_w = _fmt_walters(updated.get("walters_plays") or [])
    nw = max(len(o_w), len(u_w), 1)
    print(f"  {'ORIGINAL':<52}  {'UPDATED':<52}")
    print("  " + "-" * 50 + "  " + "-" * 50)
    for i in range(nw):
        left = o_w[i] if i < len(o_w) else "  (none)"
        right = u_w[i] if i < len(u_w) else "  (none)"
        print(f"  {left:<52}  {right:<52}")
    print()

    # --- Excluded (anchor) count ---
    o_ex = original.get("walters_data_errors") or []
    u_ex = updated.get("walters_data_errors") or []
    print("  Excluded (model spread hit 10- or 22-pt floor anchor):")
    print(f"    ORIGINAL: {len(o_ex)} matchups excluded")
    print(f"    UPDATED:  {len(u_ex)} matchups excluded")
    print()

    # --- Glitch Sleepers side-by-side ---
    print("=" * 100)
    print("  Glitch Sleepers (seed 7+, F4 >5%, Top 50 in ≥2 of FT%/3P%/Exp)")
    print("=" * 100)
    o_g = _fmt_glitch(original.get("glitch_teams") or [])
    u_g = _fmt_glitch(updated.get("glitch_teams") or [])
    ng = max(len(o_g), len(u_g), 1)
    print(f"  {'ORIGINAL':<52}  {'UPDATED':<52}")
    print("  " + "-" * 50 + "  " + "-" * 50)
    for i in range(ng):
        left = o_g[i] if i < len(o_g) else "  —"
        right = u_g[i] if i < len(u_g) else "  —"
        print(f"  {left:<52}  {right:<52}")
    print()

    print("=" * 100)
    print(f"  Summary: n_sims={N_SIMS:,} each. Rank-delta: (rank_delta/10)*0.8 toward better-ranked when Δ>50, cap 15 pts.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
