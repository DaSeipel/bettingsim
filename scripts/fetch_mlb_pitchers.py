#!/usr/bin/env python3
"""
Fetch season pitching stats for probable starters listed in data/odds/live_mlb_odds.json
(home_pitcher / away_pitcher) from the MLB Stats API.

Writes data/mlb/pitcher_stats.csv with: odds_name, player_id, full_name, season, era, fip, xfip, whip,
k9, bb9, innings_pitched

xfip is an HR-regressed approximation: fip + (hr_per_9 - 1.2) * 0.3 (hr_per_9 from homeRuns / IP * 9).

Player lookup:
  GET https://statsapi.mlb.com/api/v1/people/search?names={name}&sportId=1
Stats:
  GET https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=season&group=pitching&season={year}

FIP uses the standard formula with constant 3.10 when not provided by the API.

Usage:
  python3 scripts/fetch_mlb_pitchers.py
  python3 scripts/fetch_mlb_pitchers.py --season 2025
  python3 scripts/fetch_mlb_pitchers.py --odds data/odds/live_mlb_odds.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*urllib3 v2 only supports OpenSSL.*")
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

REQUEST_TIMEOUT = 35
REQUEST_SLEEP_S = 0.15
BASE = "https://statsapi.mlb.com/api/v1"
APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ODDS_PATH = APP_ROOT / "data" / "odds" / "live_mlb_odds.json"
OUTPUT_PATH = APP_ROOT / "data" / "mlb" / "pitcher_stats.csv"
FIP_C_FGM = 3.10

# Blend weight for 2026 rate stats when both seasons are available.
PITCHER_BLEND_2026_WEIGHT = 0.60  # Increase toward 0.75-0.80 by mid-June as 2026 samples grow
_PITCHER_BLEND_RATE_COLS = ("era", "fip", "xfip", "whip", "k9", "bb9")


def _get_json(url: str) -> dict:
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def parse_innings_to_outs(ip) -> int:
    """MLB API returns innings as e.g. '1442.1' (thirds after the dot)."""
    if ip is None:
        return 0
    if isinstance(ip, (int, float)):
        whole = int(ip)
        frac = ip - whole
        thirds = int(round(frac * 10)) % 10
        if thirds > 2:
            thirds = thirds % 3
        return whole * 3 + thirds
    s = str(ip).strip()
    if not s:
        return 0
    if "." in s:
        whole, frac = s.split(".", 1)
        w = int(whole) if whole.isdigit() else 0
        f = int(frac[0]) if frac and frac[0].isdigit() else 0
        if f > 2:
            f = f % 3
        return w * 3 + f
    return int(s) * 3 if s.isdigit() else 0


def innings_pitched_to_float(ip) -> float:
    o = parse_innings_to_outs(ip)
    return o / 3.0 if o else 0.0


def _stat_float(d: dict, key: str) -> float | None:
    v = d.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def compute_fip(st: dict, ip_float: float) -> float | None:
    """FIP = ((13*HR) + 3*(BB+HBP) - 2*K) / IP + c. Returns None if IP <= 0."""
    if ip_float <= 0:
        return None
    hr = int(st.get("homeRuns") or 0)
    bb = int(st.get("baseOnBalls") or 0)
    hbp = int(st.get("hitByPitch") or st.get("hitBatsmen") or 0)
    k = int(st.get("strikeOuts") or st.get("strikeouts") or 0)
    num = (13 * hr) + (3 * (bb + hbp)) - (2 * k)
    return (num / ip_float) + FIP_C_FGM


def compute_xfip_approx(fip: float | None, st: dict, ip_float: float) -> float | None:
    """
    Simple xFIP-style nudge: regress HR/9 toward league ~1.2 HR/9.
    xFIP_approx = fip + (hr_per_9 - 1.2) * 0.3
    """
    if fip is None:
        return None
    if ip_float <= 0:
        return float(fip)
    hr = int(st.get("homeRuns") or 0)
    hr_per_9 = (hr / ip_float) * 9.0
    return float(fip) + (hr_per_9 - 1.2) * 0.3


def best_season_pitching_stat(splits: list, season: str) -> dict | None:
    """Pick the regular-season split with the most innings (handles partial/trade rows)."""
    best_st = None
    best_ip = -1.0
    for sp in splits:
        if not isinstance(sp, dict):
            continue
        if str(sp.get("season") or "") != str(season):
            continue
        gt = sp.get("gameType")
        if gt is not None and str(gt) not in ("R", ""):
            continue
        st = sp.get("stat") or {}
        if not isinstance(st, dict):
            continue
        ipf = innings_pitched_to_float(st.get("inningsPitched"))
        if ipf > best_ip:
            best_ip = ipf
            best_st = st
    return best_st


def _is_pitcher(p: dict) -> bool:
    pos = p.get("primaryPosition") or {}
    abbr = str(pos.get("abbreviation") or "")
    name = str(pos.get("name") or "").lower()
    return abbr == "P" or "pitcher" in name


def search_player_id(name: str) -> tuple[int | None, str | None]:
    """
    Search MLB people API; return (player_id, full_name) for the best pitcher match.
    """
    raw = " ".join(str(name).split()).strip()
    if not raw or raw.upper() in ("TBD", "T.B.D.", "—", "-"):
        return None, None
    # Try ASCII-ish fold for search (e.g. Jesús -> Jesus) if first search returns nothing
    trials = [raw]
    folded = "".join(
        c for c in unicodedata.normalize("NFD", raw) if unicodedata.category(c) != "Mn"
    )
    if folded != raw:
        trials.append(folded)

    people: list = []
    for q in trials:
        enc = quote(q, safe="")
        data = _get_json(f"{BASE}/people/search?names={enc}&sportId=1")
        people = data.get("people") or []
        if people:
            break
        time.sleep(REQUEST_SLEEP_S)  # before alternate spelling / rate limit

    pitchers = [p for p in people if isinstance(p, dict) and _is_pitcher(p)]
    candidates = pitchers if pitchers else [p for p in people if isinstance(p, dict)]

    if not candidates:
        return None, None

    raw_l = raw.lower()
    for p in candidates:
        fn = str(p.get("fullName") or "")
        if fn.lower() == raw_l:
            return (int(p["id"]), fn)

    p0 = candidates[0]
    return (int(p0["id"]), str(p0.get("fullName") or ""))


def fetch_pitcher_season_row(player_id: int, season: int, odds_name: str, full_name: str) -> dict | None:
    url = f"{BASE}/people/{player_id}/stats?stats=season&group=pitching&season={season}"
    data = _get_json(url)
    blocks = data.get("stats") or []
    splits: list = []
    for blk in blocks:
        if not isinstance(blk, dict):
            continue
        grp = (blk.get("group") or {}).get("displayName") if isinstance(blk.get("group"), dict) else ""
        if str(grp).lower() != "pitching":
            continue
        splits.extend(blk.get("splits") or [])

    st = best_season_pitching_stat(splits, str(season))
    if not st:
        return None

    ipf = innings_pitched_to_float(st.get("inningsPitched"))
    era = _stat_float(st, "era")
    whip = _stat_float(st, "whip")
    k9 = _stat_float(st, "strikeoutsPer9Inn")
    bb9 = _stat_float(st, "walksPer9Inn")
    if k9 is None and ipf > 0:
        k = float(int(st.get("strikeOuts") or st.get("strikeouts") or 0))
        k9 = (k * 9.0) / ipf
    if bb9 is None and ipf > 0:
        bb = float(int(st.get("baseOnBalls") or 0))
        bb9 = (bb * 9.0) / ipf

    fip = compute_fip(st, ipf)
    if fip is None and era is not None:
        fip = float(era)

    xfip = compute_xfip_approx(fip, st, ipf)
    if xfip is None and fip is not None:
        xfip = float(fip)

    return {
        "odds_name": odds_name,
        "player_id": player_id,
        "full_name": full_name,
        "season": season,
        "era": round(era, 3) if era is not None else None,
        "fip": round(fip, 3) if fip is not None else None,
        "xfip": round(xfip, 3) if xfip is not None else None,
        "whip": round(whip, 3) if whip is not None else None,
        "k9": round(k9, 3) if k9 is not None else None,
        "bb9": round(bb9, 3) if bb9 is not None else None,
        "innings_pitched": round(ipf, 3) if ipf else 0.0,
    }


def unique_pitcher_names_from_odds(odds_path: Path) -> list[str]:
    if not odds_path.exists():
        return []
    with open(odds_path, encoding="utf-8") as f:
        blob = json.load(f)
    seen: set[str] = set()
    order: list[str] = []
    for g in blob.get("games") or []:
        if not isinstance(g, dict):
            continue
        for key in ("home_pitcher", "away_pitcher"):
            nm = g.get(key)
            if nm is None:
                continue
            s = " ".join(str(nm).split()).strip()
            if not s or s.upper() in ("TBD", "T.B.D."):
                continue
            if s not in seen:
                seen.add(s)
                order.append(s)
    return order


def blend_pitcher_rows(
    row_2025: dict | None,
    row_2026: dict | None,
    w26: float = PITCHER_BLEND_2026_WEIGHT,
) -> dict | None:
    """Blend 2025 and 2026 rate stats for a single pitcher.
    Uses 2026 IP (current workload), blends ERA/FIP/WHIP/K9/BB9.
    If only one season exists, returns that row unchanged."""
    if row_2025 is None and row_2026 is None:
        return None
    if row_2025 is None:
        return row_2026
    if row_2026 is None:
        return row_2025
    w25 = 1.0 - w26
    out = dict(row_2025)
    out["season"] = "blend"
    out["innings_pitched"] = row_2026.get("innings_pitched") or row_2025.get("innings_pitched") or 0.0
    for col in _PITCHER_BLEND_RATE_COLS:
        v25 = row_2025.get(col)
        v26 = row_2026.get(col)
        if v25 is not None and v26 is not None:
            out[col] = round(w25 * float(v25) + w26 * float(v26), 3)
        elif v25 is not None:
            out[col] = v25
        elif v26 is not None:
            out[col] = v26
    return out


def infer_season_from_odds_blob(blob: dict | None, fallback: int) -> int:
    if not blob:
        return fallback
    gde = str(blob.get("games_date_et") or "").strip()
    if len(gde) >= 4 and gde[:4].isdigit():
        return int(gde[:4])
    return fallback


def main() -> int:
    p = argparse.ArgumentParser(description="Fetch MLB pitcher stats for starters in live_mlb_odds.json")
    p.add_argument("--odds", type=Path, default=DEFAULT_ODDS_PATH, help="Path to live_mlb_odds.json")
    p.add_argument("--season", type=int, default=None, help="Season year (default: from odds games_date_et or 2025)")
    args = p.parse_args()

    if not args.odds.exists():
        print(f"Missing odds file: {args.odds}", file=sys.stderr)
        return 1

    with open(args.odds, encoding="utf-8") as f:
        odds_blob = json.load(f)

    season = args.season if args.season is not None else infer_season_from_odds_blob(odds_blob, 2025)
    names = unique_pitcher_names_from_odds(args.odds)
    if not names:
        print("No pitcher names found in odds JSON.", file=sys.stderr)
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=[
                "odds_name",
                "player_id",
                "full_name",
                "season",
                "era",
                "fip",
                "xfip",
                "whip",
                "k9",
                "bb9",
                "innings_pitched",
            ]
        ).to_csv(OUTPUT_PATH, index=False)
        print(f"Wrote empty {OUTPUT_PATH}")
        return 0

    blend_2026 = season >= 2026
    rows: list[dict] = []
    for odds_name in names:
        time.sleep(REQUEST_SLEEP_S)
        try:
            pid, fn = search_player_id(odds_name)
        except requests.RequestException as e:
            print(f"Search failed for {odds_name!r}: {e}", file=sys.stderr)
            continue
        if pid is None:
            print(f"No MLB match for pitcher name: {odds_name!r}", file=sys.stderr)
            continue

        row_2025: dict | None = None
        row_2026: dict | None = None

        time.sleep(REQUEST_SLEEP_S)
        try:
            row_2025 = fetch_pitcher_season_row(pid, 2025, odds_name, fn or "")
        except requests.RequestException as e:
            print(f"  2025 stats failed for {odds_name!r}: {e}", file=sys.stderr)

        if blend_2026:
            time.sleep(REQUEST_SLEEP_S)
            try:
                row_2026 = fetch_pitcher_season_row(pid, 2026, odds_name, fn or "")
            except requests.RequestException as e:
                print(f"  2026 stats failed for {odds_name!r}: {e}", file=sys.stderr)

        if blend_2026:
            row = blend_pitcher_rows(row_2025, row_2026, w26=PITCHER_BLEND_2026_WEIGHT)
            tag = "blend" if row_2025 and row_2026 else ("2026-only" if row_2026 else "2025-only")
        else:
            row = row_2025
            tag = str(season)

        if row:
            rows.append(row)
            print(f"  {odds_name} -> {fn} ({pid}) IP={row.get('innings_pitched')} [{tag}]")
        else:
            print(f"No pitching data for {odds_name!r} (id={pid})", file=sys.stderr)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    label = f"blended 2025+2026 ({PITCHER_BLEND_2026_WEIGHT:.0%}/{1 - PITCHER_BLEND_2026_WEIGHT:.0%})" if blend_2026 else f"season={season}"
    print(f"Saved {len(df)} pitcher(s) to {OUTPUT_PATH} ({label})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
