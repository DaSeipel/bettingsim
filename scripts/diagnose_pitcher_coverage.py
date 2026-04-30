#!/usr/bin/env python3
"""Diagnose starter coverage quality for resolved MLB picks.

Read-only diagnostics using archived starter names in play_history.reasoning_summary:
- play_history resolved MLB moneyline picks
- data/mlb/pitcher_stats.csv coverage
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "espn.db"
PITCHER_CSV = ROOT / "data" / "mlb" / "pitcher_stats.csv"
@dataclass
class PickRow:
    date_generated: str
    home_team: str
    away_team: str
    recommended_side: str
    reasoning_summary: str


def norm_team(name: str) -> str:
    return " ".join(str(name or "").strip().lower().replace(".", "").split())


def norm_name(name: str) -> str:
    return " ".join(str(name or "").strip().lower().replace(".", "").replace("'", "").split())


def load_resolved_picks() -> list[PickRow]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT date_generated, home_team, away_team, recommended_side, COALESCE(reasoning_summary, '')
        FROM play_history
        WHERE sport='MLB'
          AND result IN ('W', 'L')
          AND spread_or_total = -999.0
        ORDER BY date_generated ASC, play_id ASC
        """
    ).fetchall()
    conn.close()
    return [PickRow(str(d), str(h), str(a), str(s), str(rs or "")) for d, h, a, s, rs in rows]


def parse_reasoning_starters(text: str) -> tuple[str | None, str | None]:
    # Expected pattern in archive: "SP: away_pitcher @ home_pitcher · event_id=..."
    s = str(text or "")
    if "SP:" not in s or "@" not in s:
        return None, None
    try:
        sp_part = s.split("SP:", 1)[1]
        if "·" in sp_part:
            sp_part = sp_part.split("·", 1)[0]
        if "|" in sp_part:
            sp_part = sp_part.split("|", 1)[0]
        if "@" not in sp_part:
            return None, None
        away_raw, home_raw = sp_part.split("@", 1)
        away_sp = away_raw.strip() or None
        home_sp = home_raw.strip() or None
        return away_sp, home_sp
    except Exception:
        return None, None


def main() -> int:
    picks = load_resolved_picks()
    print(f"Resolved MLB moneyline picks: {len(picks)}")
    print("\nSAMPLE_REASONING_SUMMARY (first 3)")
    for i, p in enumerate(picks[:3], start=1):
        print(f"{i}. {p.reasoning_summary}")

    pitch_df = pd.read_csv(PITCHER_CSV)
    csv_rows = len(pitch_df)
    csv_distinct = int(pitch_df["odds_name"].astype(str).str.strip().nunique()) if "odds_name" in pitch_df.columns else 0
    ip_vals = pd.to_numeric(pitch_df.get("innings_pitched"), errors="coerce").dropna().tolist()
    ip_min = min(ip_vals) if ip_vals else 0.0
    ip_med = median(ip_vals) if ip_vals else 0.0
    ip_max = max(ip_vals) if ip_vals else 0.0
    print(f"pitcher_stats.csv rows: {csv_rows}")
    print(f"pitcher_stats.csv distinct pitchers: {csv_distinct}")
    print(f"pitcher_stats.csv IP distribution: min={ip_min:.1f}, median={ip_med:.1f}, max={ip_max:.1f}")

    csv_mtime = datetime.fromtimestamp(PITCHER_CSV.stat().st_mtime).isoformat(sep=" ", timespec="seconds")
    print(f"pitcher_stats.csv last updated: {csv_mtime}")

    # Map normalized odds_name -> max IP seen for that name.
    csv_ip_by_name: dict[str, float] = {}
    for _, r in pitch_df.iterrows():
        nm = norm_name(r.get("odds_name", ""))
        if not nm:
            continue
        ip = pd.to_numeric(r.get("innings_pitched"), errors="coerce")
        if pd.isna(ip):
            continue
        csv_ip_by_name[nm] = max(float(ip), csv_ip_by_name.get(nm, float("-inf")))

    match = 0
    regressed = 0
    defaulted = 0
    default_examples: list[tuple[str, str]] = []
    distinct_referenced: set[str] = set()
    matched_referenced: set[str] = set()

    for p in picks:
        away_sp, home_sp = parse_reasoning_starters(p.reasoning_summary)
        picked_team = norm_team(p.recommended_side)
        home_team = norm_team(p.home_team)
        away_team = norm_team(p.away_team)

        relevant_starter = None
        if picked_team == home_team:
            relevant_starter = home_sp
        elif picked_team == away_team:
            relevant_starter = away_sp
        else:
            # Fallback for aliases like Athletics/Oakland Athletics using last token.
            pt = picked_team.split()[-1:] or [picked_team]
            ht = home_team.split()[-1:] or [home_team]
            at = away_team.split()[-1:] or [away_team]
            if pt == ht:
                relevant_starter = home_sp
            elif pt == at:
                relevant_starter = away_sp

        if not relevant_starter:
            defaulted += 1
            if len(default_examples) < 10:
                default_examples.append((f"(unparseable) team={p.recommended_side}", p.date_generated))
            continue

        starter_n = norm_name(relevant_starter)
        distinct_referenced.add(relevant_starter)
        ip = csv_ip_by_name.get(starter_n)
        if ip is None:
            defaulted += 1
            if len(default_examples) < 10:
                default_examples.append((f"{relevant_starter} | team={p.recommended_side}", p.date_generated))
            continue
        matched_referenced.add(relevant_starter)
        if ip < 30.0:
            regressed += 1
        else:
            match += 1

    print("\nCOVERAGE_BUCKETS")
    print(f"MATCH (found, IP>=30): {match}")
    print(f"REGRESSED (found, IP<30): {regressed}")
    print(f"DEFAULTED (not found): {defaulted}")

    print("\nDEFAULTED_EXAMPLES (up to 5)")
    if not default_examples:
        print("(none)")
    else:
        for nm, d in default_examples:
            print(f"{d} | {nm}")

    print("\nDISTINCT_PITCHERS_REFERENCED")
    for nm in sorted(distinct_referenced, key=lambda x: x.lower()):
        print(nm)
    print(f"Distinct referenced pitcher count: {len(distinct_referenced)}")
    print(f"Distinct referenced pitchers present in CSV: {len(matched_referenced)}")
    print(f"Distinct referenced pitchers missing from CSV: {len(distinct_referenced) - len(matched_referenced)}")

    print("\nCSV_PITCHER_SET")
    csv_names_sorted = sorted(
        [str(x).strip() for x in pitch_df["odds_name"].dropna().tolist() if str(x).strip()],
        key=lambda x: x.lower(),
    )
    for nm in csv_names_sorted:
        print(nm)
    print(f"CSV total pitcher count: {len(csv_names_sorted)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
