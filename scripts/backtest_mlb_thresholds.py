#!/usr/bin/env python3
"""Read-only backtest for MLB edge-threshold scenarios."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "espn.db"
STAKE = 10.0


@dataclass
class PickRow:
    edge_pct: float
    odds: int
    result: str


def pick_profit(odds: int, result: str, stake: float = STAKE) -> float:
    if result == "L":
        return -stake
    if result != "W":
        return 0.0
    if odds >= 100:
        return stake * (odds / 100.0)
    if odds <= -100:
        return stake * (100.0 / abs(odds))
    return 0.0


def edge_tier(edge_pct: float) -> str:
    e = abs(edge_pct)
    if e < 3:
        return "< 3%"
    if e < 5:
        return "3-5%"
    if e < 10:
        return "5-10%"
    if e < 15:
        return "10-15%"
    return "15%+"


def load_rows() -> list[PickRow]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT my_edge_pct, market_odds_at_time, result
        FROM play_history
        WHERE sport='MLB'
          AND result IN ('W', 'L')
          AND spread_or_total = -999.0
        """
    ).fetchall()
    conn.close()
    return [PickRow(float(edge), int(odds), str(result)) for edge, odds, result in rows]


def scenario_stats(rows: list[PickRow], min_edge: float, max_edge: float) -> dict:
    passed = [r for r in rows if min_edge <= abs(r.edge_pct) < max_edge]
    n = len(passed)
    w = sum(1 for r in passed if r.result == "W")
    l = sum(1 for r in passed if r.result == "L")
    win_pct = (100.0 * w / (w + l)) if (w + l) else 0.0
    wagered = STAKE * n
    pl = round(sum(pick_profit(r.odds, r.result) for r in passed), 2)
    roi = (pl / wagered * 100.0) if wagered else 0.0
    return {
        "picks": n,
        "w": w,
        "l": l,
        "win_pct": win_pct,
        "wagered": wagered,
        "pl": pl,
        "roi": roi,
    }


def main() -> int:
    rows = load_rows()
    print(f"Loaded MLB moneyline resolved picks: {len(rows)}")

    scenarios = [
        ("Current (MIN=0.03, MAX=0.15)", 3.0, 15.0),
        ("Proposed A (MIN=0.05, MAX=0.15)", 5.0, 15.0),
        ("Proposed B (MIN=0.05, MAX=0.18)", 5.0, 18.0),
    ]
    stats = [(name, lo, hi, scenario_stats(rows, lo, hi)) for name, lo, hi in scenarios]

    print("\nSCENARIO_COMPARISON")
    print("Scenario|Picks|W-L|Win %|Total Wagered|P/L|ROI")
    for name, _lo, _hi, s in stats:
        print(
            f"{name}|{s['picks']}|{s['w']}-{s['l']}|{s['win_pct']:.1f}%|"
            f"${s['wagered']:,.0f}|${s['pl']:+.2f}|{s['roi']:+.1f}%"
        )

    # Tier breakdown for current settings.
    current_rows = [r for r in rows if 3.0 <= abs(r.edge_pct) < 15.0]
    tiers = ["< 3%", "3-5%", "5-10%", "10-15%", "15%+"]
    print("\nCURRENT_SETTINGS_TIER_CONTRIBUTION")
    print("Tier|Picks|W-L|Win %|Tier P/L|% of Total P/L")
    total_pl = sum(pick_profit(r.odds, r.result) for r in current_rows)
    for t in tiers:
        bucket = [r for r in current_rows if edge_tier(r.edge_pct) == t]
        n = len(bucket)
        w = sum(1 for r in bucket if r.result == "W")
        l = sum(1 for r in bucket if r.result == "L")
        win_pct = (100.0 * w / (w + l)) if (w + l) else 0.0
        pl = round(sum(pick_profit(r.odds, r.result) for r in bucket), 2)
        contrib = (pl / total_pl * 100.0) if total_pl else 0.0
        print(f"{t}|{n}|{w}-{l}|{win_pct:.1f}%|${pl:+.2f}|{contrib:+.1f}%")

    # Recommendation block.
    baseline = stats[0][3]
    best = max(stats, key=lambda x: x[3]["roi"])
    best_name, _lo, _hi, best_stats = best
    filtered_out = baseline["picks"] - best_stats["picks"]
    print("\nRECOMMENDATION")
    print(f"Best ROI scenario: {best_name}")
    print(
        f"Best ROI: {best_stats['roi']:+.1f}% on {best_stats['picks']} picks "
        f"({filtered_out} filtered out vs current baseline of {baseline['picks']})."
    )

    # Flag likely match between proposed A/B if no picks in [15,18).
    hi_band = sum(1 for r in rows if 15.0 <= abs(r.edge_pct) < 18.0)
    if hi_band == 0:
        print("Note: No historical picks in 15%-18% edge band; Proposed B is expected to match Proposed A currently.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
