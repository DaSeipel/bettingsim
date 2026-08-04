#!/usr/bin/env python3
"""
Fetch today's MLB schedule + probable pitchers (MLB Stats API via statsapi)
and consensus moneylines / totals from The Odds API (https://the-odds-api.com).

Writes data/odds/live_mlb_odds.json with the same schema as the prior scraper version.
When the slate has >=1 game, archives to data/odds/mlb_archive/YYYY-MM-DD_HHMMSS.json (UTC).

Every run calls The Odds API (no mtime short-circuit). Missing key / HTTP / empty
or unparseable responses raise and exit non-zero — never writes a silent null-ML slate.

Consensus method: for each event, aggregates moneylines and totals across all
US bookmakers returned by the API. Takes the median of implied probabilities
per side and converts back to American odds. Totals lines are median-snapped to
the nearest half-run, and prices are medianed across books near that line.

Usage:
  python3 scripts/fetch_mlb_odds.py
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

from datetime import datetime, timedelta, timezone
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
_SLATE_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SPORT_KEY = "baseball_mlb"
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"


class OddsApiError(RuntimeError):
    """Loud failure: API key missing, HTTP error, or unusable response."""


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


# ---------- Archive ----------

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
    stamp = datetime.now(timezone.utc).strftime("%H%M%S")
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    dest = ARCHIVE_DIR / f"{slate}_{stamp}.json"
    shutil.copy2(OUTPUT_PATH, dest)
    print(f"Archived odds -> data/odds/mlb_archive/{dest.name}")


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


# Match Odds API event to Stats API game only if commence times are within this window.
ODDS_COMMENCE_TOLERANCE = timedelta(hours=6)


def _parse_iso_utc(value) -> datetime | None:
    """Parse ISO-8601 (Z or offset) to timezone-aware UTC. Never returns naive."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def select_odds_candidate(
    candidates: list,
    game_start_utc: datetime | None,
    *,
    away: str = "",
    home: str = "",
    log: bool = True,
) -> dict | None:
    """
    Among Odds API candidates for a (away,home) matchup, pick the one whose
    commence_time is closest to game_start_utc. Reject if nearest is outside
    ODDS_COMMENCE_TOLERANCE (never attach a wrong-date line).
    """
    if not candidates:
        return None
    usable = [
        c
        for c in candidates
        if isinstance(c, dict)
        and c.get("commence_time_utc") is not None
        and c.get("home_ml") is not None
        and c.get("away_ml") is not None
    ]
    if not usable:
        return None
    if game_start_utc is None:
        if log:
            print(
                f"[odds_match] {away} @ {home}: missing Stats API commence_time — no odds match",
                flush=True,
            )
        return None
    if game_start_utc.tzinfo is None:
        game_start_utc = game_start_utc.replace(tzinfo=timezone.utc)
    else:
        game_start_utc = game_start_utc.astimezone(timezone.utc)

    ranked = sorted(
        usable,
        key=lambda c: abs((c["commence_time_utc"] - game_start_utc).total_seconds()),
    )
    best = ranked[0]
    best_delta = abs((best["commence_time_utc"] - game_start_utc).total_seconds())
    if best_delta > ODDS_COMMENCE_TOLERANCE.total_seconds():
        if log:
            print(
                f"[odds_match] {away} @ {home}: nearest Odds API event "
                f"commence={best['commence_time_utc'].isoformat()} "
                f"delta_h={best_delta / 3600:.2f} outside {ODDS_COMMENCE_TOLERANCE} — NO MATCH",
                flush=True,
            )
        return None

    in_window = [
        c
        for c in usable
        if abs((c["commence_time_utc"] - game_start_utc).total_seconds())
        <= ODDS_COMMENCE_TOLERANCE.total_seconds()
    ]
    if log and len(in_window) > 1:
        others = [c for c in in_window if c is not best]
        print(
            f"WARNING: DOUBLEHEADER_AMBIGUITY {away} @ {home} | "
            f"chosen={best['commence_time_utc'].isoformat()} "
            f"ml_away={best.get('away_ml')} ml_home={best.get('home_ml')} | "
            f"also_in_window={[c['commence_time_utc'].isoformat() for c in others]}",
            flush=True,
        )

    if log and len(usable) > 1:
        rejected = [c for c in usable if c is not best]
        for r in rejected:
            print(
                f"[odds_match] {away} @ {home}: chose commence={best['commence_time_utc'].isoformat()} "
                f"away_ml={best.get('away_ml')} home_ml={best.get('home_ml')} | "
                f"rejected commence={r['commence_time_utc'].isoformat()} "
                f"away_ml={r.get('away_ml')} home_ml={r.get('home_ml')}",
                flush=True,
            )

    return best


# ---------- The Odds API fetch ----------

def fetch_odds_from_api(api_key: str):
    """Return (odds_by_matchup, bookmaker_label, x-requests-remaining, x-requests-used).

    odds_by_matchup key = (normalized_away, normalized_home) -> LIST of candidate dicts:
      {away_ml, home_ml, total, commence_time_utc (aware UTC datetime), odds_api_event_id}

    Raises OddsApiError on missing key, HTTP failure, or empty/unparseable body.
    """
    if not api_key:
        raise OddsApiError(
            "[odds_api] No API key found in secrets.toml [the_odds_api] or ODDS_API_KEY env var."
        )

    params = {
        "regions": "us",
        "markets": "h2h,totals",
        "oddsFormat": "american",
        "apiKey": api_key,
    }
    try:
        resp = requests.get(ODDS_API_URL, params=params, timeout=20)
    except requests.RequestException as e:
        raise OddsApiError(f"[odds_api] HTTP error: {e}") from e

    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")

    if resp.status_code != 200:
        raise OddsApiError(
            f"[odds_api] HTTP {resp.status_code}: {resp.text[:300]} "
            f"(credits used={used} remaining={remaining})"
        )

    try:
        events = resp.json()
    except json.JSONDecodeError as e:
        raise OddsApiError("[odds_api] Failed to parse JSON response body.") from e

    if events is None:
        raise OddsApiError("[odds_api] Empty JSON body (null).")
    if not isinstance(events, list):
        raise OddsApiError(
            f"[odds_api] Unexpected response type: {type(events).__name__}"
        )
    if len(events) == 0:
        raise OddsApiError(
            f"[odds_api] Empty events list (credits used={used} remaining={remaining})."
        )

    odds_by_matchup: dict[tuple[str, str], list] = {}
    for event in events:
        away_raw = event.get("away_team") or ""
        home_raw = event.get("home_team") or ""
        if not away_raw or not home_raw:
            continue

        commence_utc = _parse_iso_utc(event.get("commence_time"))
        if commence_utc is None:
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

        key = (_norm_team(away_raw), _norm_team(home_raw))
        odds_by_matchup.setdefault(key, []).append(
            {
                "away_ml": _median_american(ml_away),
                "home_ml": _median_american(ml_home),
                "total": agg_total,
                "commence_time_utc": commence_utc,
                "odds_api_event_id": str(event.get("id") or ""),
            }
        )

    usable = sum(
        1
        for cands in odds_by_matchup.values()
        for c in cands
        if c.get("home_ml") is not None and c.get("away_ml") is not None
    )
    if usable == 0:
        raise OddsApiError(
            f"[odds_api] Response had {len(events)} event(s) but zero usable moneylines "
            f"(credits used={used} remaining={remaining})."
        )

    bm_label = "TheOddsAPI consensus (median, US books)"
    return odds_by_matchup, bm_label, remaining, used


# ---------- Build payload ----------

def build_payload():
    """Return (payload, remaining, used, matched_ml_count). Raises OddsApiError on API failure."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    date_str = now_et.date().isoformat()
    sched = load_schedule_statsapi(now_et)

    api_key = _read_odds_api_key()
    odds_lookup, book_title, remaining, used = fetch_odds_from_api(api_key)

    games_out = []
    matched_ml = 0
    unmatched = 0
    for g in sched:
        gid = str(g.get("game_id") or "")
        away = str(g.get("away_name") or "").strip()
        home = str(g.get("home_name") or "").strip()
        commence = str(g.get("game_datetime") or "")
        game_start_utc = _parse_iso_utc(commence)
        hp_val = g.get("home_probable_pitcher") or None
        ap_val = g.get("away_probable_pitcher") or None
        hp = str(hp_val).strip() if hp_val else None
        ap = str(ap_val).strip() if ap_val else None

        candidates = odds_lookup.get((_norm_team(away), _norm_team(home))) or []
        ml_data = select_odds_candidate(
            candidates,
            game_start_utc,
            away=away,
            home=home,
            log=True,
        )
        if ml_data is None:
            ml = {"home_odds": None, "away_odds": None}
            total = {"line": None, "over_odds": None, "under_odds": None}
            unmatched += 1
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

    print(
        f"[odds_match] summary: schedule_games={len(games_out)} "
        f"matched_moneylines={matched_ml} no_odds_match={unmatched}",
        flush=True,
    )

    if len(games_out) > 0 and matched_ml == 0:
        raise OddsApiError(
            f"[odds_api] Schedule has {len(games_out)} game(s) but zero moneylines matched. "
            "Check team-name matching, commence_time join, or API markets. "
            "Refusing to write null-ML slate."
        )

    captured = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    payload = {
        "fetched_at_utc": captured,
        "odds_captured_at": captured,
        "sport_key": SPORT_KEY,
        "source": "statsapi_schedule+the_odds_api",
        "bookmaker_title": book_title,
        "games_date_et": date_str,
        "games": games_out,
    }
    return payload, remaining, used, matched_ml


def write_live_odds(payload: dict) -> Path:
    """Write working live_mlb_odds.json and timestamped archive copy."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _archive_live_odds_copy(payload)
    return OUTPUT_PATH


def refresh_live_odds() -> dict:
    """Fetch fresh odds, write live file + archive, return payload. Raises OddsApiError."""
    payload, remaining, used, matched_ml = build_payload()
    n = len(payload.get("games") or [])
    write_live_odds(payload)
    print(f"Wrote {n} game(s) to {OUTPUT_PATH}  [moneylines: {matched_ml}/{n}]")
    print(f"[odds_api] credits used: {used} | remaining: {remaining}")
    return payload


# ---------- Main ----------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="MLB schedule + The Odds API consensus odds -> live_mlb_odds.json"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Deprecated no-op (every run always calls the API).",
    )
    args = parser.parse_args()
    if args.force:
        print("[odds_api] --force is deprecated (always-fresh); ignoring.", flush=True)

    try:
        refresh_live_odds()
    except OddsApiError as e:
        print(str(e), file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Fetch failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
