#!/usr/bin/env python3
"""
Comprehensive MLB model diagnostic (v1) — read-only analysis of archived picks.

Filters: sport=MLB, moneyline (spread_or_total=-999), edge >= 5%, date >= 2026-05-03,
resolved W/L/P. No DB or file writes.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "espn.db"
PITCHER_CSV = ROOT / "data" / "mlb" / "pitcher_stats.csv"
RECENT_FORM_CSV = ROOT / "data" / "mlb" / "recent_form.csv"
ODDS_ARCHIVE_DIR = ROOT / "data" / "odds" / "mlb_archive"
CACHE_ARCHIVE_DIR = ROOT / "data" / "cache" / "mlb_archive"

MLB_PIPELINE_START_DATE = "2026-05-03"
STAKE_USD = 10.0
MIN_PATTERN_PICKS = 8
CALIBRATION_FLAG_PP = 5.0

EDGE_TIERS = (
    ("5-10%", 5.0, 10.0),
    ("10-15%", 10.0, 15.0),
    ("15%+", 15.0, float("inf")),
)

PROB_BUCKETS = (
    ("40-45%", 0.40, 0.45),
    ("45-50%", 0.45, 0.50),
    ("50-55%", 0.50, 0.55),
    ("55-60%", 0.55, 0.60),
    ("60%+", 0.60, 1.01),
)

OPP_FIP_TIERS = (
    ("ace (<3.5)", 0.0, 3.5),
    ("average (3.5-4.5)", 3.5, 4.5),
    ("poor (>4.5)", 4.5, float("inf")),
)


@dataclass
class PatternStat:
    label: str
    n: int
    w: int
    l: int
    p: int
    pl: float
    roi: float
    win_pct: float

    @property
    def record(self) -> str:
        if self.p:
            return f"{self.w}-{self.l}-{self.p}"
        return f"{self.w}-{self.l}"


def _hr(char: str = "=", width: int = 72) -> None:
    print(char * width)


def _section(title: str) -> None:
    print()
    _hr()
    print(title.upper())
    _hr("-")
    print()


def _norm_team(name: str) -> str:
    return " ".join(str(name or "").strip().lower().replace(".", "").split())


def _norm_pitcher(name: str) -> str:
    return " ".join(str(name or "").strip().lower().replace(".", "").replace("'", "").split())


def flat_profit_usd(american_odds: Any, result: str, stake: float = STAKE_USD) -> float:
    r = str(result).strip().upper()
    if r == "P":
        return 0.0
    if r == "L":
        return -stake
    if r != "W":
        return 0.0
    try:
        a = float(american_odds)
    except (TypeError, ValueError):
        return 0.0
    if a >= 100:
        return stake * (a / 100.0)
    if a <= -100:
        return stake * (100.0 / abs(a))
    return 0.0


def implied_breakeven_win_pct(american_odds: float) -> float:
    """Break-even win rate (%) for a flat bet at American odds (no vig removal)."""
    try:
        a = float(american_odds)
    except (TypeError, ValueError):
        return 50.0
    if a >= 100:
        return 100.0 / (a + 100.0) * 100.0
    if a <= -100:
        return abs(a) / (abs(a) + 100.0) * 100.0
    return 50.0


def win_rate_se_pp(win_pct: float, n: int) -> float:
    """Standard error of win rate in percentage points (binomial)."""
    if n <= 0:
        return float("nan")
    p = win_pct / 100.0
    return math.sqrt(max(p * (1.0 - p), 0.0) / n) * 100.0


def parse_reasoning_starters(text: str) -> tuple[str | None, str | None]:
    s = str(text or "")
    m = re.search(r"SP:\s*(.*?)\s*@\s*(.*?)(?:\s*·|\s*\||$)", s)
    if not m:
        return None, None
    away_sp = m.group(1).strip() or None
    home_sp = m.group(2).strip() or None
    return away_sp, home_sp


def parse_event_id(text: str) -> str | None:
    m = re.search(r"event_id=(\d+)", str(text or ""))
    return m.group(1) if m else None


def dedupe_play_history(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "date_generated",
        "sport",
        "home_team",
        "away_team",
        "bet_type",
        "recommended_side",
        "spread_or_total",
    ]
    if df.empty or not all(c in df.columns for c in cols):
        return df
    return df.drop_duplicates(subset=cols, keep="last").reset_index(drop=True)


def load_picks() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT play_id, date_generated, sport, home_team, away_team, bet_type,
               recommended_side, spread_or_total, my_edge_pct, my_probability,
               market_odds_at_time, confidence_tier, reasoning_summary, result
        FROM play_history
        WHERE sport = 'MLB'
          AND spread_or_total = -999.0
          AND my_edge_pct >= 5.0
          AND date_generated >= ?
          AND result IN ('W', 'L', 'P')
        ORDER BY date_generated ASC, play_id ASC
        """,
        conn,
        params=(MLB_PIPELINE_START_DATE,),
    )
    conn.close()
    df = dedupe_play_history(df)
    df["result_clean"] = df["result"].astype(str).str.strip().str.upper()
    return df


def load_pitcher_fip_by_name() -> dict[str, float]:
    if not PITCHER_CSV.exists():
        return {}
    pdf = pd.read_csv(PITCHER_CSV)
    out: dict[str, float] = {}
    for _, row in pdf.iterrows():
        nm = _norm_pitcher(row.get("odds_name", ""))
        if not nm:
            continue
        fip = pd.to_numeric(row.get("fip"), errors="coerce")
        if pd.notna(fip):
            out[nm] = float(fip)
    return out


def load_recent_form_by_team() -> dict[str, float]:
    if not RECENT_FORM_CSV.exists():
        return {}
    rdf = pd.read_csv(RECENT_FORM_CSV)
    out: dict[str, float] = {}
    for _, row in rdf.iterrows():
        key = _norm_team(row.get("team_name", ""))
        rwp = pd.to_numeric(row.get("recent_win_pct"), errors="coerce")
        if key and pd.notna(rwp):
            out[key] = float(rwp)
    return out


def load_archive_jsons(directory: Path) -> dict[str, dict[str, Any]]:
    """date_iso -> parsed JSON."""
    out: dict[str, dict[str, Any]] = {}
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as fh:
                out[path.stem] = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
    return out


def summarize_df(df: pd.DataFrame) -> PatternStat:
    if df.empty:
        return PatternStat("(empty)", 0, 0, 0, 0, 0.0, 0.0, 0.0)
    w = int((df["result_clean"] == "W").sum())
    l = int((df["result_clean"] == "L").sum())
    p = int((df["result_clean"] == "P").sum())
    n = len(df)
    pl = sum(flat_profit_usd(r["market_odds_at_time"], r["result_clean"]) for _, r in df.iterrows())
    roi = (pl / (STAKE_USD * n) * 100.0) if n else 0.0
    wl = w + l
    win_pct = (100.0 * w / wl) if wl else 0.0
    return PatternStat("(aggregate)", n, w, l, p, pl, roi, win_pct)


def pattern_from_df(label: str, df: pd.DataFrame) -> PatternStat:
    s = summarize_df(df)
    s.label = label
    return s


def enrich_picks(df: pd.DataFrame) -> pd.DataFrame:
    pitcher_fip = load_pitcher_fip_by_name()
    form_by_team = load_recent_form_by_team()
    odds_arch = load_archive_jsons(ODDS_ARCHIVE_DIR)
    cache_arch = load_archive_jsons(CACHE_ARCHIVE_DIR)

    # Build per-date play lookup from cache archives (model_prob, edge, pitchers)
    cache_play_by_key: dict[tuple[str, str, str], dict] = {}
    for date_iso, payload in cache_arch.items():
        for play in payload.get("plays") or []:
            if not isinstance(play, dict):
                continue
            home = str(play.get("home_team", "")).strip()
            away = str(play.get("away_team", "")).strip()
            sel = str(play.get("selection", "")).strip()
            if home and away:
                cache_play_by_key[(date_iso, home, away, sel)] = play
                cache_play_by_key[(date_iso, _norm_team(home), _norm_team(away), _norm_team(sel))] = play

    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        d = str(r["date_generated"]).strip()
        home = str(r["home_team"]).strip()
        away = str(r["away_team"]).strip()
        side = str(r["recommended_side"]).strip()
        reasoning = str(r.get("reasoning_summary") or "")
        away_sp, home_sp = parse_reasoning_starters(reasoning)
        eid = parse_event_id(reasoning)

        picked_is_home = _norm_team(side) == _norm_team(home)
        picked_is_away = _norm_team(side) == _norm_team(away)
        if not picked_is_home and not picked_is_away:
            pt, ht, at = side.split()[-1:], home.split()[-1:], away.split()[-1:]
            picked_is_home = bool(pt and ht and pt[0].lower() == ht[0].lower())
            picked_is_away = bool(pt and at and pt[0].lower() == at[0].lower())

        opp_sp = None
        if picked_is_home:
            opp_sp = away_sp
        elif picked_is_away:
            opp_sp = home_sp

        opp_fip = pitcher_fip.get(_norm_pitcher(opp_sp or ""))
        if opp_fip is None:
            opp_fip_tier = "unknown"
        elif opp_fip < 3.5:
            opp_fip_tier = "ace (<3.5)"
        elif opp_fip <= 4.5:
            opp_fip_tier = "average (3.5-4.5)"
        else:
            opp_fip_tier = "poor (>4.5)"

        picked_rwp = form_by_team.get(_norm_team(side))
        if picked_is_home:
            opp_team = away
        elif picked_is_away:
            opp_team = home
        else:
            opp_team = ""
        opp_rwp = form_by_team.get(_norm_team(opp_team))
        contrarian = False
        if picked_rwp is not None and opp_rwp is not None:
            contrarian = opp_rwp > picked_rwp

        try:
            odds = float(r["market_odds_at_time"])
        except (TypeError, ValueError):
            odds = float("nan")
        favdog = "favorite" if odds < 0 else "underdog" if odds > 0 else "even"

        edge = float(r["my_edge_pct"])
        if edge < 10.0:
            edge_tier = "5-10%"
        elif edge < 15.0:
            edge_tier = "10-15%"
        else:
            edge_tier = "15%+"

        prob = float(r["my_probability"])
        if prob > 1.0:
            prob = prob / 100.0
        prob_bucket = "other"
        for label, lo, hi in PROB_BUCKETS:
            if lo <= prob < hi:
                prob_bucket = label
                break

        has_odds_arch = d in odds_arch
        has_cache_arch = d in cache_arch
        arch_play = cache_play_by_key.get((d, home, away, side))
        if arch_play is None:
            arch_play = cache_play_by_key.get(
                (d, _norm_team(home), _norm_team(away), _norm_team(side))
            )

        rows.append(
            {
                **r.to_dict(),
                "away_sp": away_sp,
                "home_sp": home_sp,
                "opp_sp": opp_sp,
                "opp_fip": opp_fip,
                "opp_fip_tier": opp_fip_tier,
                "picked_rwp": picked_rwp,
                "opp_rwp": opp_rwp,
                "contrarian": contrarian,
                "contrarian_label": "contrarian_fade" if contrarian else "non_contrarian",
                "favdog": favdog,
                "edge_tier": edge_tier,
                "prob_bucket": prob_bucket,
                "model_prob": prob,
                "has_odds_archive": has_odds_arch,
                "has_cache_archive": has_cache_arch,
                "event_id": eid,
                "arch_model_prob": arch_play.get("model_prob") if arch_play else None,
            }
        )
    return pd.DataFrame(rows)


def print_pattern_stat(s: PatternStat, extra: str = "") -> None:
    note = f"  {extra}" if extra else ""
    print(
        f"  {s.label:<32} n={s.n:3d}  {s.record:>7}  win%={s.win_pct:5.1f}  "
        f"P/L=${s.pl:+7.2f}  ROI={s.roi:+6.1f}%{note}"
    )


def print_sample_overview(df: pd.DataFrame) -> None:
    _section("1. Sample Overview")
    s = summarize_df(df)
    wl = s.w + s.l
    print(f"  Pipeline start date:     {MLB_PIPELINE_START_DATE}")
    print(f"  Total resolved picks:    {s.n}")
    print(f"  Record (W-L-P):          {s.record}")
    print(f"  Win % (excl. pushes):    {s.win_pct:.1f}%  ({s.w}/{wl})")
    print(f"  Flat ${STAKE_USD:.0f} P/L:            ${s.pl:+.2f}")
    print(f"  ROI:                     {s.roi:+.1f}%")

    breakevens = [
        implied_breakeven_win_pct(float(r["market_odds_at_time"]))
        for _, r in df.iterrows()
        if pd.notna(r.get("market_odds_at_time"))
    ]
    if breakevens:
        avg_be = float(np.mean(breakevens))
        print(f"  Avg break-even win %:    {avg_be:.1f}%  (implied from odds taken)")
        print(f"  Actual vs break-even:    {s.win_pct - avg_be:+.1f} pp")
        if s.win_pct < avg_be:
            print("  → Below break-even: model edge not realized in results yet.")
        else:
            print("  → Above break-even: positive edge vs posted odds (flat stakes).")


def print_data_coverage(df: pd.DataFrame, odds_arch: dict, cache_arch: dict) -> None:
    _section("Data Coverage")
    dates = sorted(df["date_generated"].astype(str).unique())
    print(f"  Pick dates in sample:    {len(dates)}  ({dates[0]} … {dates[-1]})")
    missing_odds = [d for d in dates if d not in odds_arch]
    missing_cache = [d for d in dates if d not in cache_arch]
    print(f"  odds/mlb_archive:        {len(dates) - len(missing_odds)}/{len(dates)} dates")
    if missing_odds:
        print(f"    missing: {', '.join(missing_odds)}")
    print(f"  cache/mlb_archive:       {len(dates) - len(missing_cache)}/{len(dates)} dates")
    if missing_cache:
        print(f"    missing: {', '.join(missing_cache)}")
    opp_known = int((df["opp_fip_tier"] != "unknown").sum()) if "opp_fip_tier" in df.columns else 0
    print(f"  Opposing SP FIP resolved: {opp_known}/{len(df)}")
    both = df["picked_rwp"].notna() & df["opp_rwp"].notna()
    print(f"  Contrarian (recent_form):  {int(both.sum())}/{len(df)} picks with both teams")
    print("    NOTE: recent_form.csv is current snapshot, not pick-date historical.")


def print_calibration(df: pd.DataFrame) -> None:
    _section("2. Calibration Check (model probability vs actual win rate)")
    print(f"  {'Bucket':<10} {'N':>4} {'Predicted':>10} {'Actual':>10} {'Gap':>8}  Flag")
    print(f"  {'-'*10} {'-'*4} {'-'*10} {'-'*10} {'-'*8}  ----")
    flags: list[str] = []
    for label, lo, hi in PROB_BUCKETS:
        sub = df[(df["model_prob"] >= lo) & (df["model_prob"] < hi)]
        if sub.empty:
            print(f"  {label:<10} {0:>4} {'—':>10} {'—':>10} {'—':>8}")
            continue
        pred = float(sub["model_prob"].mean()) * 100.0
        wl = sub[sub["result_clean"].isin(("W", "L"))]
        actual = (100.0 * (wl["result_clean"] == "W").sum() / len(wl)) if len(wl) else float("nan")
        gap = actual - pred if not math.isnan(actual) else float("nan")
        flag = ""
        if not math.isnan(gap):
            if gap <= -CALIBRATION_FLAG_PP:
                flag = "OVERCONFIDENT"
                flags.append(f"{label}: actual {actual:.1f}% vs predicted {pred:.1f}%")
            elif gap >= CALIBRATION_FLAG_PP:
                flag = "UNDERCONFIDENT"
                flags.append(f"{label}: actual {actual:.1f}% vs predicted {pred:.1f}%")
        print(
            f"  {label:<10} {len(sub):>4} {pred:>9.1f}% {actual:>9.1f}% {gap:>+7.1f}pp  {flag}"
        )
    if flags:
        print("\n  Flags (≥5pp gap):")
        for f in flags:
            print(f"    • {f}")
    else:
        print("\n  No bucket with ≥5pp systematic over/under-confidence.")


def print_edge_tiers(df: pd.DataFrame) -> None:
    _section("3. Edge Tier Breakdown")
    for label, lo, hi in EDGE_TIERS:
        sub = df[(df["my_edge_pct"] >= lo) & (df["my_edge_pct"] < hi)]
        s = pattern_from_df(label, sub)
        se = win_rate_se_pp(s.win_pct, s.w + s.l)
        reliability = "unreliable (n<10)" if s.n < 10 else f"SE(win%)≈±{se:.1f}pp"
        print_pattern_stat(s, extra=reliability)


def print_favdog(df: pd.DataFrame) -> None:
    _section("4. Favorite vs Underdog")
    for label in ("favorite", "underdog"):
        sub = df[df["favdog"] == label]
        print_pattern_stat(pattern_from_df(label, sub))
    fav = summarize_df(df[df["favdog"] == "favorite"])
    dog = summarize_df(df[df["favdog"] == "underdog"])
    if fav.pl < dog.pl:
        print("\n  → Losses driven more by FAVORITES (lower P/L).")
    elif dog.pl < fav.pl:
        print("\n  → Losses driven more by UNDERDOGS (lower P/L).")
    else:
        print("\n  → Similar P/L on both sides.")


def print_contrarian(df: pd.DataFrame) -> None:
    _section("5. Contrarian Fade Pattern")
    print("  Definition: pick team with LOWER recent_win_pct vs opponent (cold vs hot).")
    print("  Uses current recent_form.csv (historical pick-time form not archived).\n")
    for label in ("contrarian_fade", "non_contrarian"):
        sub = df[df["contrarian_label"] == label]
        print_pattern_stat(pattern_from_df(label.replace("_", " "), sub))
    unknown = df[df["picked_rwp"].isna() | df["opp_rwp"].isna()]
    if not unknown.empty:
        print_pattern_stat(pattern_from_df("form_unknown", unknown), extra="excluded from fade split")


def print_opp_pitcher(df: pd.DataFrame) -> None:
    _section("6. Quality of Opposing Pitcher (blended FIP from pitcher_stats.csv)")
    tier_order = [t[0] for t in OPP_FIP_TIERS] + ["unknown"]
    for label in tier_order:
        sub = df[df["opp_fip_tier"] == label]
        if sub.empty and label == "unknown":
            continue
        extra = ""
        if label != "unknown" and not sub.empty:
            med = float(sub["opp_fip"].median())
            extra = f"median FIP={med:.2f}"
        print_pattern_stat(pattern_from_df(label, sub), extra=extra)


def collect_all_patterns(df: pd.DataFrame) -> list[PatternStat]:
    patterns: list[PatternStat] = []

    for label, lo, hi in PROB_BUCKETS:
        sub = df[(df["model_prob"] >= lo) & (df["model_prob"] < hi)]
        if not sub.empty:
            patterns.append(pattern_from_df(f"prob {label}", sub))

    for label, lo, hi in EDGE_TIERS:
        sub = df[(df["my_edge_pct"] >= lo) & (df["my_edge_pct"] < hi)]
        if not sub.empty:
            patterns.append(pattern_from_df(f"edge {label}", sub))

    for label in ("favorite", "underdog"):
        patterns.append(pattern_from_df(f"{label}", df[df["favdog"] == label]))

    for label in ("contrarian_fade", "non_contrarian"):
        sub = df[df["contrarian_label"] == label]
        if not sub.empty:
            patterns.append(pattern_from_df(label.replace("_", " "), sub))

    for label in [t[0] for t in OPP_FIP_TIERS] + ["unknown"]:
        sub = df[df["opp_fip_tier"] == label]
        if not sub.empty:
            patterns.append(pattern_from_df(f"vs {label}", sub))

    # Combined high-signal splits
    patterns.append(
        pattern_from_df(
            "fav + 5-10% edge",
            df[(df["favdog"] == "favorite") & (df["edge_tier"] == "5-10%")],
        )
    )
    patterns.append(
        pattern_from_df(
            "dog + 15%+ edge",
            df[(df["favdog"] == "underdog") & (df["edge_tier"] == "15%+")],
        )
    )
    patterns.append(
        pattern_from_df(
            "contrarian + vs ace",
            df[(df["contrarian_label"] == "contrarian_fade") & (df["opp_fip_tier"] == "ace (<3.5)")],
        )
    )
    return patterns


def print_worst_pattern(patterns: list[PatternStat]) -> None:
    _section("7. Worst Single Pattern (n ≥ 8)")
    eligible = [p for p in patterns if p.n >= MIN_PATTERN_PICKS]
    if not eligible:
        print(f"  No pattern with at least {MIN_PATTERN_PICKS} picks.")
        return
    worst = min(eligible, key=lambda p: p.roi)
    print(
        f"\n  Consider filtering or down-weighting picks where: {worst.label}.\n"
        f"  Sample: {worst.n} picks, {worst.record} record, ROI: {worst.roi:+.1f}%.\n"
        f"  (Diagnostic only — no filter applied.)"
    )
    print("\n  All eligible patterns by ROI (n≥8):")
    for p in sorted(eligible, key=lambda x: x.roi):
        print_pattern_stat(p)


def main() -> int:
    print()
    _hr()
    print("MLB MODEL DIAGNOSTIC v1")
    print(f"Read-only · picks since {MLB_PIPELINE_START_DATE} · flat ${STAKE_USD:.0f} stakes")
    _hr()

    if not DB_PATH.exists():
        print(f"ERROR: play_history DB not found: {DB_PATH}")
        return 1

    raw = load_picks()
    if raw.empty:
        print("No picks matched filters.")
        return 0

    odds_arch = load_archive_jsons(ODDS_ARCHIVE_DIR)
    cache_arch = load_archive_jsons(CACHE_ARCHIVE_DIR)
    df = enrich_picks(raw)

    print_data_coverage(df, odds_arch, cache_arch)
    print_sample_overview(df)
    print_calibration(df)
    print_edge_tiers(df)
    print_favdog(df)
    print_contrarian(df)
    print_opp_pitcher(df)
    patterns = collect_all_patterns(df)
    print_worst_pattern(patterns)

    print()
    _hr()
    print("END DIAGNOSTIC")
    _hr()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
