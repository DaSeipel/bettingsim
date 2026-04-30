#!/usr/bin/env python3
"""Diagnose MLB 5-10% edge tier outcomes for common failure patterns.

Read-only: queries play_history and local CSV/JSON data; does not write DB.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "espn.db"
PITCHER_PATH = ROOT / "data" / "mlb" / "pitcher_stats.csv"
PARK_PATH = ROOT / "data" / "mlb" / "mlb_park_factors.json"
STAKE = 10.0


@dataclass
class PickDiag:
    date: str
    team: str
    odds: int
    edge_pct: float
    result: str
    favdog: str
    pitcher_quality: str
    park_mult: float
    park_bucket: str
    picked_pitcher: str


def profit(odds: int, result: str) -> float:
    if result == "L":
        return -STAKE
    if result != "W":
        return 0.0
    if odds >= 100:
        return STAKE * (odds / 100.0)
    return STAKE * (100.0 / abs(odds))


def wl_roi(rows: list[PickDiag]) -> tuple[int, int, float]:
    w = sum(1 for r in rows if r.result == "W")
    l = sum(1 for r in rows if r.result == "L")
    wagered = STAKE * len(rows)
    pl = sum(profit(r.odds, r.result) for r in rows)
    roi = (pl / wagered * 100.0) if wagered else 0.0
    return w, l, roi


def park_bucket(mult: float) -> str:
    if mult > 1.05:
        return "high (>1.05)"
    if mult < 0.95:
        return "low (<0.95)"
    return "neutral (0.95-1.05)"


def parse_sp(reasoning_summary: str) -> tuple[str | None, str | None]:
    text = str(reasoning_summary or "")
    m = re.search(r"SP:\s*(.*?)\s*@\s*(.*?)(?:\s*·|$)", text)
    if not m:
        return None, None
    away_sp = m.group(1).strip() or None
    home_sp = m.group(2).strip() or None
    return away_sp, home_sp


def main() -> int:
    pitchers = pd.read_csv(PITCHER_PATH)
    pitcher_ip = {
        str(r["odds_name"]).strip().lower(): float(r["innings_pitched"])
        for _, r in pitchers.iterrows()
        if pd.notna(r.get("odds_name"))
    }

    with open(PARK_PATH, encoding="utf-8") as fh:
        park_raw = json.load(fh)
    park_mult = {
        team: float(data.get("runs", 100)) / 100.0
        for team, data in park_raw.items()
        if isinstance(data, dict) and "runs" in data
    }

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT date_generated, recommended_side, market_odds_at_time, my_edge_pct, result,
               home_team, away_team, reasoning_summary
        FROM play_history
        WHERE sport='MLB'
          AND result IN ('W','L')
          AND spread_or_total = -999.0
          AND ABS(my_edge_pct) >= 5.0
          AND ABS(my_edge_pct) < 10.0
        ORDER BY date_generated ASC, recommended_side ASC
        """
    ).fetchall()
    conn.close()

    picks: list[PickDiag] = []
    for d, team, odds, edge, result, home, away, reasoning in rows:
        odds_i = int(odds)
        side = str(team).strip()
        home_t = str(home).strip()
        away_t = str(away).strip()
        away_sp, home_sp = parse_sp(reasoning)

        picked_sp = ""
        if side == home_t:
            picked_sp = home_sp or ""
        elif side == away_t:
            picked_sp = away_sp or ""

        key = picked_sp.lower().strip()
        ip = pitcher_ip.get(key)
        if not picked_sp or ip is None or ip < 30.0:
            pq = "default_or_incomplete"
        else:
            pq = "complete"

        pm = park_mult.get(home_t, 1.0)
        picks.append(
            PickDiag(
                date=str(d),
                team=side,
                odds=odds_i,
                edge_pct=float(edge),
                result=str(result),
                favdog="favorite" if odds_i < 0 else "underdog",
                pitcher_quality=pq,
                park_mult=pm,
                park_bucket=park_bucket(pm),
                picked_pitcher=picked_sp or "(unknown)",
            )
        )

    print("5_10_TIER_PICKS")
    print("Date|Team|Odds|Edge %|Result|Fav/Dog|Pitcher Data|Picked SP|Park Factor")
    for p in picks:
        print(
            f"{p.date}|{p.team}|{p.odds:+d}|{p.edge_pct:.2f}|{p.result}|{p.favdog}|"
            f"{p.pitcher_quality}|{p.picked_pitcher}|{p.park_mult:.2f}"
        )

    print("\nBREAKDOWN_FAVORITE_VS_UNDERDOG")
    print("Group|Picks|W-L|ROI")
    for g in ("favorite", "underdog"):
        sub = [p for p in picks if p.favdog == g]
        w, l, roi = wl_roi(sub)
        print(f"{g}|{len(sub)}|{w}-{l}|{roi:+.1f}%")

    print("\nBREAKDOWN_PITCHER_DATA")
    print("Group|Picks|W-L|ROI")
    for g in ("complete", "default_or_incomplete"):
        sub = [p for p in picks if p.pitcher_quality == g]
        w, l, roi = wl_roi(sub)
        print(f"{g}|{len(sub)}|{w}-{l}|{roi:+.1f}%")

    print("\nBREAKDOWN_PARK_FACTOR")
    print("Group|Picks|W-L|ROI")
    for g in ("high (>1.05)", "neutral (0.95-1.05)", "low (<0.95)"):
        sub = [p for p in picks if p.park_bucket == g]
        w, l, roi = wl_roi(sub)
        print(f"{g}|{len(sub)}|{w}-{l}|{roi:+.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
