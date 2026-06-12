#!/usr/bin/env python3
"""
Fetch today's MLB schedule + probable pitchers (MLB Stats API via statsapi)
and consensus moneylines / totals from The Odds API (https://the-odds-api.com).

Writes data/odds/live_mlb_odds.json with the same schema as the prior scraper version.
When the slate has >=1 game, archives to data/odds/mlb_archive/YYYY-MM-DD.json.

Freshness: if the file was modified in the last 30 minutes, skips the API call
unless --force is passed.

Consensus method: for each event, aggregates moneylines and totals across all
US bookmakers returned by the API. Takes the median of implied probabilities
per side and converts back to American odds. Totals lines are median-snapped to
the nearest half-run, and prices are medianed across books near that line.

Loud failures: prints HTTP errors, API credit usage on every run, and warns
if zero moneylines were retrieved.

Usage:
  python3 scripts/fetch_mlb_odds.py
  python3 scripts/fetch_mlb_odds.py --force
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*urllib3 v2 only supports OpenSSL.*")

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
os.chdir(APP_ROOT)

from engine.mlb_engine import MLB_TEAM_NAME_ALIASES, normalize_mlb_team_name_for_join

OUTPUT_PATH = APP_ROOT / "data" / "odds" / "live_mlb_odds.json"
ARCHIVE_DIR = APP_ROOT / "data" / "odds" / "mlb_archive"
SECRETS_PATH = APP_ROOT / ".streamlit" / "secrets.toml"
FRESHNESS_SECONDS = 30 * 60
_SLATE_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SPORT_KEY = "baseball_mlb"
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"


# ---------- API key loading ----------

def _read_odds_api_key() -> str:
    """Read The Odds API key from the [the_odds_api] section of secrets.toml.
    Falls back to ODDS_API_KEY env var."""
    if SECRETS_PATH.exists():
        text = SECRETS_PATH.read_text()
        m = re.search(
            r"\[the_odds_api\][^\[]*?api_key\s*=\s*['\"]([^'\"]+)['\"]",
            text,
            re.S,
        )
        if m:
            return m.group(1).strip()
    return os.environ.get("ODDS_API_KEY", "").strip()


# ---------- Odds math helpers ----------

def _american_to_prob(odds):
    if odds is None:
        return None
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return None
    if o == 0:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    return abs(o) / (abs(o) + 100.0)


def _prob_to_american(p):
    if p is None or p <= 0 or p >= 1:
        return None
    if p >= 0.5:
        return int(round(-(p / (1 - p)) * 100))
    return int(round((1 - p) / p * 100))


def _median_american(prices):
    probs = [_american_to_prob(p) for p in prices]
    probs = [p for p in probs if p is not None]
    if not probs:
        return None
    return _prob_to_american(statistics.median(probs))


# ---------- Freshness / archive ----------

def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return False
    return (datetime.now().timestamp() - mtime) < FRESHNESS_SECONDS


def _slate_date_iso_for_archive(payload: dict) -> str:
    raw = str(payload.get("games_date_et") or "").strip()
    if raw and _SLATE_DATE_RE.fullmatch(raw):
        return raw
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def _archive_live_odds_copy(payload: dict) -> None:
    games = payload.get("games") or []
    if not isinstance(games, list) or len(games) < 1:
        return
    slate = _slate_date_iso_for_archive(payload)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    dest = ARCHIVE_DIR / f"{slate}.json"
    shutil.copy2(OUTPUT_PATH, dest)
    print(f"Archived odds -> data/odds/mlb_archive/{slate}.json")


# ---------- Schedule (statsapi) ----------

def load_schedule_statsapi(date_et: datetime) -> list:
    import statsapi

    d = date_et.date().isoformat()
    games = statsapi.schedule(date=d) or []
    out = []
    for g in games:
        if not isinstance(g, dict):
            continue
        if str(g.get("status") or "") in ("Cancelled", "Postponed"):
            continue
        out.append(g)
    out.sort(key=lambda x: str(x.get("game_datetime") or ""))
    return out


# ---------- Team name matching ----------

def _norm_team(name: str) -> str:
    """Normalize a team name via the existing MLB alias layer so statsapi and
    Odds API names join cleanly even if minor drift exists."""
    n = (name or "").strip()
    if not n:
        return ""
    base = normalize_mlb_team_name_for_join(n)
    return MLB_TEAM_NAME_ALIASES.get(base, base).strip().lower()


# ---------- The Odds API fetch ----------

def fetch_odds_from_api(api_key: str):
    """Return (odds_by_matchup, bookmaker_label, x-requests-remaining, x-requests-used).
    odds_by_matchup key = (normalized_away, normalized_home) -> dict with away_ml/home_ml/total."""
    if not api_key:
        print(
            "[odds_api] No API key found in secrets.toml or ODDS_API_KEY env var.",
            file=sys.stderr,
        )
        return {}, "", "?", "?"

    params = {
        "regions": "us",
        "markets": "h2h,totals",
        "oddsFormat": "american",
        "apiKey": api_key,
    }
    try:
        resp = requests.get(ODDS_API_URL, params=params, timeout=20)
    except requests.RequestException as e:
        print(f"[odds_api] HTTP error: {e}", file=sys.stderr)
        return {}, "", "?", "?"

    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")

    if resp.status_code != 200:
        print(
            f"[odds_api] HTTP {resp.status_code}: {resp.text[:300]}",
            file=sys.stderr,
        )
        return {}, "", remaining, used

    try:
        events = resp.json() or []
    except json.JSONDecodeError:
        print("[odds_api] Failed to parse JSON response body.", file=sys.stderr)
        return {}, "", remaining, used

    if not isinstance(events, list):
        print(
            f"[odds_api] Unexpected response type: {type(events).__name__}",
            file=sys.stderr,
        )
        return {}, "", remaining, used

    odds_by_matchup = {}
    for event in events:
        away_raw = event.get("away_team") or ""
        home_raw = event.get("home_team") or ""
        if not away_raw or not home_raw:
            continue

        ml_away = []
        ml_home = []
        totals_rows = []  # (line, over_price, under_price)

        for bm in event.get("bookmakers") or []:
            for market in bm.get("markets") or []:
                mkey = market.get("key")
                outcomes = market.get("outcomes") or []
                if mkey == "h2h":
                    for o in outcomes:
                        nm = o.get("name") or ""
                        price = o.get("price")
                        if price is None:
                            continue
                        if nm == away_raw:
                            ml_away.append(float(price))
                        elif nm == home_raw:
                            ml_home.append(float(price))
                elif mkey == "totals":
                    line_v = None
                    over_p = None
                    under_p = None
                    for o in outcomes:
                        nm = (o.get("name") or "").lower()
                        if nm == "over":
                            over_p = int(o.get("price")) if o.get("price") is not None else None
                            if o.get("point") is not None:
                                line_v = float(o.get("point"))
                        elif nm == "under":
                            under_p = int(o.get("price")) if o.get("price") is not None else None
                            if line_v is None and o.get("point") is not None:
                                line_v = float(o.get("point"))
                    if line_v is not None and over_p is not None and under_p is not None:
                        totals_rows.append((line_v, over_p, under_p))

        agg_total = {"line": None, "over_odds": None, "under_odds": None}
        if totals_rows:
            try:
                med_line = statistics.median([t[0] for t in totals_rows])
            except statistics.StatisticsError:
                med_line = None
            if med_line is not None:
                near = [t for t in totals_rows if abs(t[0] - med_line) <= 0.5]
                if near:
                    agg_total["line"] = round(med_line * 2) / 2
                    agg_total["over_odds"] = _median_american([t[1] for t in near])
                    agg_total["under_odds"] = _median_american([t[2] for t in near])

        odds_by_matchup[(_norm_team(away_raw), _norm_team(home_raw))] = {
            "away_ml": _median_american(ml_away),
            "home_ml": _median_american(ml_home),
            "total": agg_total,
        }

    bm_label = "TheOddsAPI consensus (median, US books)"
    return odds_by_matchup, bm_label, remaining, used


# ---------- Build payload ----------

def build_payload():
    """Return (payload, remaining, used, matched_ml_count)."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    date_str = now_et.date().isoformat()
    sched = load_schedule_statsapi(now_et)

    api_key = _read_odds_api_key()
    odds_lookup, book_title, remaining, used = fetch_odds_from_api(api_key)

    games_out = []
    matched_ml = 0
    for g in sched:
        gid = str(g.get("game_id") or "")
        away = str(g.get("away_name") or "").strip()
        home = str(g.get("home_name") or "").strip()
        commence = str(g.get("game_datetime") or "")
        hp_val = g.get("home_probable_pitcher") or None
        ap_val = g.get("away_probable_pitcher") or None
        hp = str(hp_val).strip() if hp_val else None
        ap = str(ap_val).strip() if ap_val else None

        ml_data = odds_lookup.get((_norm_team(away), _norm_team(home)))
        if ml_data is None:
            ml = {"home_odds": None, "away_odds": None}
            total = {"line": None, "over_odds": None, "under_odds": None}
        else:
            ml = {"home_odds": ml_data["home_ml"], "away_odds": ml_data["away_ml"]}
            total = ml_data["total"]
            if ml_data["home_ml"] is not None and ml_data["away_ml"] is not None:
                matched_ml += 1

        games_out.append(
            {
                "event_id": gid,
                "commence_time": commence,
                "home_team": home,
                "away_team": away,
                "home_pitcher": hp,
                "away_pitcher": ap,
                "bookmaker_title": book_title,
                "moneyline": ml,
                "total": total,
            }
        )

    payload = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sport_key": SPORT_KEY,
        "source": "statsapi_schedule+the_odds_api",
        "bookmaker_title": book_title,
        "games_date_et": date_str,
        "games": games_out,
    }
    return payload, remaining, used, matched_ml


# ---------- Main ----------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="MLB schedule + The Odds API consensus odds -> live_mlb_odds.json"
    )
    parser.add_argument("--force", action="store_true", help="Ignore 30m freshness guard.")
    args = parser.parse_args()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not args.force and _is_fresh(OUTPUT_PATH):
        print("Data is fresh.")
        return 0

    try:
        payload, remaining, used, matched_ml = build_payload()
    except Exception as e:
        print(f"Fetch failed: {e}", file=sys.stderr)
        return 1

    n = len(payload.get("games") or [])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {n} game(s) to {OUTPUT_PATH}  [moneylines: {matched_ml}/{n}]")
    print(f"[odds_api] credits used: {used} | remaining: {remaining}")

    if n > 0 and matched_ml == 0:
        print(
            "WARNING: API returned no moneylines for any scheduled game. "
            "Check API key, team-name matching, or quota.",
            file=sys.stderr,
        )

    _archive_live_odds_copy(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
