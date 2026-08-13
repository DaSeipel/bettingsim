#!/usr/bin/env python3
"""
Repair pass for MLB decision_log edges contaminated by pre-2026-08-04 odds join.

Until the commence_time join fix, fetch_mlb_odds.py keyed Odds API events by team pair
only (last-write-wins), so some rows were logged against another day's line.

This script:
  - Does not change model params or recompute model probabilities
  - Re-pulls the line that should have applied from data/odds/mlb_archive/ via the same
    team-pair + nearest commence_time logic as scripts/fetch_mlb_odds.select_odds_candidate
  - Recomputes edge with engine.mlb_engine.value_summary_moneyline on the stored model prob
  - Writes data/cache/decision_log/all_decisions_repaired.csv (never overwrites all_decisions.csv)

Usage:
  python3 scripts/repair_decision_log_odds.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.mlb_engine import value_summary_moneyline  # noqa: E402
from scripts.fetch_mlb_odds import (  # noqa: E402
    _norm_team,
    _parse_iso_utc,
    select_odds_candidate,
)

DECISIONS_PATH = ROOT / "data" / "cache" / "decision_log" / "all_decisions.csv"
OUT_PATH = ROOT / "data" / "cache" / "decision_log" / "all_decisions_repaired.csv"
ARCHIVE_DIR = ROOT / "data" / "odds" / "mlb_archive"

ODDS_FIX_DATE = "2026-08-04"
EDGE_DELTA_FLAG = 0.02  # 2 percentage points (edges stored as decimals)


def _archive_capture_ts(path: Path, blob: dict) -> datetime | None:
    ts = _parse_iso_utc(blob.get("odds_captured_at") or blob.get("fetched_at_utc"))
    if ts is not None:
        return ts
    stem = path.stem
    if "_" in stem:
        day, hm = stem.split("_", 1)
        if len(hm) >= 6 and hm.isdigit():
            return _parse_iso_utc(f"{day}T{hm[0:2]}:{hm[2:4]}:{hm[4:6]}Z")
    if len(stem) == 10 and stem[4] == "-" and stem[7] == "-":
        return _parse_iso_utc(f"{stem}T12:00:00Z")
    return None


def _ml_pair(game: dict) -> tuple[float, float] | None:
    ml = game.get("moneyline") or {}
    try:
        away = float(ml["away_odds"])
        home = float(ml["home_odds"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(away) or not math.isfinite(home):
        return None
    return away, home


def _load_archives() -> list[tuple[datetime, Path, dict]]:
    out: list[tuple[datetime, Path, dict]] = []
    if not ARCHIVE_DIR.is_dir():
        return out
    for path in sorted(ARCHIVE_DIR.glob("*.json")):
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(blob, dict):
            continue
        ts = _archive_capture_ts(path, blob)
        if ts is None:
            continue
        out.append((ts, path, blob))
    out.sort(key=lambda x: x[0])
    return out


def _normalize_event_id(raw) -> str:
    s = str(raw or "").strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def _game_date_iso(game_date: str) -> str:
    return (game_date or "").strip()[:10]


def _commence_date_iso(ct: datetime) -> str:
    return ct.astimezone(timezone.utc).date().isoformat()


def _resolve_game_start(
    *,
    eid: str,
    away: str,
    home: str,
    game_date: str,
    nearest_blob: dict,
    archives: list[tuple[datetime, Path, dict]],
) -> datetime | None:
    """
    Resolve scheduled start for a decision row.

    Prefer the nearest archive snapshot (event_id, then team pair on game_date).
    Event IDs are not globally unique across days in historical archives, so never
    use a first-seen global event_id map.
    """
    gd = _game_date_iso(game_date)
    a_n, h_n = _norm_team(away), _norm_team(home)

    def from_blob(blob: dict) -> datetime | None:
        if eid:
            for g in blob.get("games") or []:
                if not isinstance(g, dict):
                    continue
                if _normalize_event_id(g.get("event_id")) != eid:
                    continue
                ct = _parse_iso_utc(g.get("commence_time"))
                if ct is None:
                    continue
                if gd and _commence_date_iso(ct) != gd:
                    # Same numeric id on another slate date — skip.
                    continue
                return ct
        for g in blob.get("games") or []:
            if not isinstance(g, dict):
                continue
            if _norm_team(g.get("away_team")) != a_n or _norm_team(g.get("home_team")) != h_n:
                continue
            ct = _parse_iso_utc(g.get("commence_time"))
            if ct is None:
                continue
            if gd and _commence_date_iso(ct) != gd:
                continue
            return ct
        return None

    hit = from_blob(nearest_blob)
    if hit is not None:
        return hit

    # Fall back across archives, still restricted to game_date when known.
    for _ts, _path, blob in archives:
        hit = from_blob(blob)
        if hit is not None:
            return hit
    return None


def _nearest_archive(
    archives: list[tuple[datetime, Path, dict]], run_ts: datetime
) -> tuple[datetime, Path, dict] | None:
    if not archives:
        return None
    return min(archives, key=lambda a: abs((a[0] - run_ts).total_seconds()))


def _candidates_from_blob(blob: dict, away: str, home: str) -> list[dict]:
    """Odds-API-style candidates for select_odds_candidate from one archive snapshot."""
    a_n, h_n = _norm_team(away), _norm_team(home)
    cands: list[dict] = []
    for g in blob.get("games") or []:
        if not isinstance(g, dict):
            continue
        if _norm_team(g.get("away_team")) != a_n or _norm_team(g.get("home_team")) != h_n:
            continue
        pair = _ml_pair(g)
        ct = _parse_iso_utc(g.get("commence_time"))
        if pair is None or ct is None:
            continue
        away_ml, home_ml = pair
        cands.append(
            {
                "commence_time_utc": ct,
                "home_ml": home_ml,
                "away_ml": away_ml,
                "event_id": _normalize_event_id(g.get("event_id")),
            }
        )
    return cands


def _pool_candidates_nearest_per_commence(
    archives: list[tuple[datetime, Path, dict]],
    run_ts: datetime,
    away: str,
    home: str,
    *,
    game_date: str = "",
) -> list[dict]:
    """
    Build a multi-commence candidate list like The Odds API would return for a matchup.

    For each distinct commence_time observed across archives for this team pair, keep the
    observation from the archive snapshot nearest to the decision timestamp. Then
    select_odds_candidate picks the commence closest to the game's scheduled start.

    When game_date is set, only keep commence times on that UTC date (plus adjacent-day
    evening slates that still belong to the decision's matchup window via select_odds_candidate).
    """
    a_n, h_n = _norm_team(away), _norm_team(home)
    gd = _game_date_iso(game_date)
    best_by_commence: dict[str, tuple[float, dict]] = {}
    for cap_ts, _path, blob in archives:
        delta = abs((cap_ts - run_ts).total_seconds())
        for g in blob.get("games") or []:
            if not isinstance(g, dict):
                continue
            if _norm_team(g.get("away_team")) != a_n or _norm_team(g.get("home_team")) != h_n:
                continue
            pair = _ml_pair(g)
            ct = _parse_iso_utc(g.get("commence_time"))
            if pair is None or ct is None:
                continue
            # Restrict pool to game_date ± 1 day so wrong-slate event_id reuse cannot
            # inject a distant day's line as a "candidate".
            if gd:
                cd = _commence_date_iso(ct)
                if abs(
                    (
                        datetime.fromisoformat(cd).date()
                        - datetime.fromisoformat(gd).date()
                    ).days
                ) > 1:
                    continue
            away_ml, home_ml = pair
            key = ct.isoformat()
            cand = {
                "commence_time_utc": ct,
                "home_ml": home_ml,
                "away_ml": away_ml,
                "event_id": _normalize_event_id(g.get("event_id")),
            }
            prev = best_by_commence.get(key)
            if prev is None or delta < prev[0]:
                best_by_commence[key] = (delta, cand)
    return [v[1] for v in best_by_commence.values()]


def _observations_for_commence(
    archives: list[tuple[datetime, Path, dict]],
    away: str,
    home: str,
    commence: datetime,
) -> list[tuple[datetime, float, float]]:
    """All (capture_ts, away_ml, home_ml) for this matchup+commence across archives."""
    a_n, h_n = _norm_team(away), _norm_team(home)
    target = commence.astimezone(timezone.utc)
    out: list[tuple[datetime, float, float]] = []
    for cap_ts, _path, blob in archives:
        for g in blob.get("games") or []:
            if not isinstance(g, dict):
                continue
            if _norm_team(g.get("away_team")) != a_n or _norm_team(g.get("home_team")) != h_n:
                continue
            ct = _parse_iso_utc(g.get("commence_time"))
            if ct is None or ct != target:
                continue
            pair = _ml_pair(g)
            if pair is None:
                continue
            away_ml, home_ml = pair
            out.append((cap_ts, away_ml, home_ml))
    return out


def _resolve_moneyline_for_commence(
    archives: list[tuple[datetime, Path, dict]],
    away: str,
    home: str,
    commence: datetime,
    run_ts: datetime,
    selected_away: float,
    selected_home: float,
) -> tuple[float, float]:
    """
    Given a time-matched commence, choose the moneyline that should have applied.

    Start from the nearest-snapshot selection. If other archives disagree on this same
    commence_time (last-write-wins contamination left different prices in later
    corrected snapshots), prefer the latest capture at or before game_start + 6h —
    i.e. the best available joined line for that event before/around first pitch.
    """
    obs = _observations_for_commence(archives, away, home, commence)
    if len(obs) <= 1:
        return selected_away, selected_home

    uniq = {(round(a), round(h)) for _c, a, h in obs}
    if len(uniq) <= 1:
        return selected_away, selected_home

    deadline = commence + timedelta(hours=6)
    before = [o for o in obs if o[0] <= deadline]
    pool = before if before else obs
    # Latest capture in the pool is the best available join for this commence.
    _cap, away_ml, home_ml = max(pool, key=lambda o: o[0])
    return float(away_ml), float(home_ml)


def _to_float(val) -> float | None:
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        x = float(s)
    except ValueError:
        return None
    if not math.isfinite(x):
        return None
    return x


def _logged_edge(row: dict) -> float | None:
    for key in ("edge", "pick_edge"):
        v = _to_float(row.get(key))
        if v is not None:
            return v
    return None


def _model_prob_for_pick(row: dict) -> float | None:
    p_home = _to_float(row.get("p_home"))
    if p_home is None:
        return None
    pick = str(row.get("pick_team") or "").strip()
    home = str(row.get("home_team") or "").strip()
    away = str(row.get("away_team") or "").strip()
    if not pick:
        return None
    if pick == home:
        return p_home
    if pick == away:
        return 1.0 - p_home
    # Fallback: compare normalized names
    if _norm_team(pick) == _norm_team(home):
        return p_home
    if _norm_team(pick) == _norm_team(away):
        return 1.0 - p_home
    return None


def _edge_tier(edge: float | None) -> str | None:
    if edge is None:
        return None
    if edge < 0.05:
        return "<5"
    if edge < 0.10:
        return "5-10"
    if edge < 0.15:
        return "10-15"
    return "15+"


def _data_regime(game_date: str) -> str:
    gd = (game_date or "").strip()[:10]
    if gd and gd < ODDS_FIX_DATE:
        return "pre_odds_fix"
    return "post_odds_fix"


def repair_row(
    row: dict,
    archives: list[tuple[datetime, Path, dict]],
) -> dict:
    out = dict(row)
    game_date = str(row.get("game_date") or "").strip()
    out["data_regime"] = _data_regime(game_date)
    logged = _logged_edge(row)
    out["logged_edge"] = "" if logged is None else logged
    out["recomputed_edge"] = ""
    out["edge_delta"] = ""
    out["line_was_wrong_day"] = ""

    run_ts = _parse_iso_utc(row.get("run_timestamp"))
    away = str(row.get("away_team") or "").strip()
    home = str(row.get("home_team") or "").strip()
    model_p = _model_prob_for_pick(row)
    if run_ts is None or not away or not home or model_p is None or logged is None:
        return out

    nearest = _nearest_archive(archives, run_ts)
    if nearest is None:
        return out
    _cap, _path, blob = nearest

    eid = _normalize_event_id(row.get("event_id"))
    game_start = _resolve_game_start(
        eid=eid,
        away=away,
        home=home,
        game_date=game_date,
        nearest_blob=blob,
        archives=archives,
    )

    # Candidates: nearest snapshot first, then pooled multi-commence fallback so
    # doubleheaders / adjacent-day matchups can be disambiguated like the live fix.
    cands = _candidates_from_blob(blob, away, home)
    selected = None
    if eid:
        eid_hits = [c for c in cands if str(c.get("event_id") or "") == eid]
        if len(eid_hits) == 1:
            selected = eid_hits[0]
        elif len(eid_hits) > 1 and game_start is not None:
            selected = select_odds_candidate(
                eid_hits, game_start, away=away, home=home, log=False
            )
    if selected is None:
        selected = select_odds_candidate(
            cands, game_start, away=away, home=home, log=False
        )
    if selected is None:
        cands = _pool_candidates_nearest_per_commence(
            archives, run_ts, away, home, game_date=game_date
        )
        if eid:
            eid_hits = [c for c in cands if str(c.get("event_id") or "") == eid]
            if len(eid_hits) == 1:
                selected = eid_hits[0]
            elif eid_hits:
                selected = select_odds_candidate(
                    eid_hits, game_start, away=away, home=home, log=False
                )
        if selected is None:
            selected = select_odds_candidate(
                cands, game_start, away=away, home=home, log=False
            )
    if selected is None:
        return out

    home_ml = float(selected["home_ml"])
    away_ml = float(selected["away_ml"])
    sel_commence = selected.get("commence_time_utc")
    if isinstance(sel_commence, datetime):
        away_ml, home_ml = _resolve_moneyline_for_commence(
            archives,
            away,
            home,
            sel_commence,
            run_ts,
            away_ml,
            home_ml,
        )

    pick = str(row.get("pick_team") or "").strip()
    if _norm_team(pick) == _norm_team(home) or pick == home:
        sel_ml, other_ml = home_ml, away_ml
    elif _norm_team(pick) == _norm_team(away) or pick == away:
        sel_ml, other_ml = away_ml, home_ml
    else:
        return out

    summ = value_summary_moneyline(float(model_p), float(sel_ml), float(other_ml))
    recomputed = float(summ["edge"])
    delta = recomputed - float(logged)
    wrong = abs(delta) > EDGE_DELTA_FLAG

    out["recomputed_edge"] = recomputed
    out["edge_delta"] = delta
    out["line_was_wrong_day"] = wrong
    return out


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def main() -> int:
    if not DECISIONS_PATH.exists():
        print(f"Missing {DECISIONS_PATH}", file=sys.stderr)
        return 1

    archives = _load_archives()
    if not archives:
        print(f"No odds archives under {ARCHIVE_DIR}", file=sys.stderr)
        return 1
    with DECISIONS_PATH.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    extra = [
        "data_regime",
        "logged_edge",
        "recomputed_edge",
        "edge_delta",
        "line_was_wrong_day",
    ]
    out_fields = fieldnames + [c for c in extra if c not in fieldnames]

    repaired: list[dict] = []
    deltas: list[float] = []
    wrong_n = 0
    tier_migrate_n = 0
    recomputed_n = 0
    tier_moves: dict[str, int] = defaultdict(int)

    for row in rows:
        out = repair_row(row, archives)
        repaired.append(out)
        rec = _to_float(out.get("recomputed_edge"))
        logged = _to_float(out.get("logged_edge"))
        if rec is None or logged is None:
            continue
        recomputed_n += 1
        delta = float(out["edge_delta"]) if out.get("edge_delta") != "" else rec - logged
        deltas.append(delta)
        if out.get("line_was_wrong_day") is True or str(out.get("line_was_wrong_day")).lower() == "true":
            wrong_n += 1
        t0, t1 = _edge_tier(logged), _edge_tier(rec)
        if t0 is not None and t1 is not None and t0 != t1:
            tier_migrate_n += 1
            tier_moves[f"{t0}->{t1}"] += 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        for row in repaired:
            # Serialize bools consistently for CSV
            if isinstance(row.get("line_was_wrong_day"), bool):
                row = dict(row)
                row["line_was_wrong_day"] = "true" if row["line_was_wrong_day"] else "false"
            writer.writerow(row)

    deltas_sorted = sorted(deltas)
    print(f"Wrote {OUT_PATH} ({len(repaired)} rows; original untouched).")
    print(f"Archives indexed: {len(archives)}")
    print(f"Rows with recomputed_edge: {recomputed_n}")
    print(f"Affected plays (line_was_wrong_day=true, |edge_delta|>{EDGE_DELTA_FLAG:.0%}): {wrong_n}")
    print(f"Tier-migration count (logged vs recomputed buckets): {tier_migrate_n}")
    if deltas_sorted:
        print(
            "edge_delta distribution: "
            f"n={len(deltas_sorted)} mean={mean(deltas_sorted):+.4f} "
            f"median={median(deltas_sorted):+.4f} "
            f"p05={_percentile(deltas_sorted, 0.05):+.4f} "
            f"p95={_percentile(deltas_sorted, 0.95):+.4f} "
            f"min={deltas_sorted[0]:+.4f} max={deltas_sorted[-1]:+.4f}"
        )
    if tier_moves:
        print("Tier migrations:")
        for k in sorted(tier_moves.keys()):
            print(f"  {k}: {tier_moves[k]}")
    else:
        print("Tier migrations: none")

    # Regime breakdown for wrong-day flags
    pre_wrong = sum(
        1
        for r in repaired
        if r.get("data_regime") == "pre_odds_fix"
        and str(r.get("line_was_wrong_day")).lower() == "true"
    )
    post_wrong = sum(
        1
        for r in repaired
        if r.get("data_regime") == "post_odds_fix"
        and str(r.get("line_was_wrong_day")).lower() == "true"
    )
    print(f"Wrong-day by regime: pre_odds_fix={pre_wrong} post_odds_fix={post_wrong}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
