#!/usr/bin/env python3
"""
Fetch today's MLB schedule + probable pitchers (MLB Stats API via statsapi, no key)
and consensus moneylines / totals from public VegasInsider HTML (no paid API).

Writes data/odds/live_mlb_odds.json (same schema as before for the rest of the app).

If the file was modified within the last 30 minutes, skips network calls and prints
'Data is fresh.'

Fallback: if VegasInsider scrape yields no lines, games still include teams/times/pitchers
from statsapi with null odds.

Usage:
  python3 scripts/fetch_mlb_odds.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
# Must not import urllib3 before this — import triggers NotOpenSSLWarning on LibreSSL/macOS.
warnings.filterwarnings("ignore", message=r".*urllib3 v2 only supports OpenSSL.*")

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
os.chdir(APP_ROOT)

OUTPUT_PATH = APP_ROOT / "data" / "odds" / "live_mlb_odds.json"
FRESHNESS_SECONDS = 30 * 60
SPORT_KEY = "baseball_mlb"

VEGAS_INSIDER_MLB_ODDS = "https://www.vegasinsider.com/mlb/odds/las-vegas/"
ODDS_SHARK_MLB_ODDS = "https://www.oddsshark.com/mlb/odds"

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return False
    age = datetime.now().timestamp() - mtime
    return age < FRESHNESS_SECONDS


def _fetch_html(url: str) -> str | None:
    try:
        r = requests.get(url, headers=REQUEST_HEADERS, timeout=35)
        r.raise_for_status()
        return r.text
    except requests.RequestException:
        return None


def _consensus_int(cell_text: str) -> int | None:
    s = cell_text.strip().replace(" ", "")
    if s.lower() in ("even", "ev", "pk", "pick"):
        return 100
    m = re.fullmatch(r"([+-]?\d+)", s)
    if not m:
        return None
    try:
        v = int(m.group(1))
        if abs(v) < 100 and v != 0:
            return None
        return v
    except ValueError:
        return None


def _odds_cell_for_row(tds: list) -> str:
    """VegasInsider uses column 1 for consensus; older layouts used last column."""
    if len(tds) >= 2:
        c1 = tds[1].get_text(" ", strip=True)
        if c1:
            return c1
    return tds[-1].get_text(" ", strip=True) if tds else ""


def parse_vegasinsider_moneylines(html: str, max_pairs: int | None = None) -> list[tuple[int, int]]:
    """
    Parse VI Las Vegas MLB odds table: consensus moneyline is in the second <td> (or last).
    Returns list of (away_ml, home_ml) in page order (pairs of moneyline rows).
    Skips total (o9/u9) and run-line rows.
    """
    soup = BeautifulSoup(html, "html.parser")
    consensus_vals: list[int] = []
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue
        first = tds[0].get_text(" ", strip=True)
        fl = first.lower()
        if fl in ("time", "open") or first.startswith("›"):
            continue
        cell = _odds_cell_for_row(tds)
        cl = cell.lower()
        if re.match(r"^[ou]\d", cl):
            continue
        if re.search(r"\d+\.\d", cl):
            continue
        ci = _consensus_int(cell)
        if ci is not None:
            consensus_vals.append(ci)
    pairs: list[tuple[int, int]] = []
    for i in range(0, len(consensus_vals) - 1, 2):
        pairs.append((consensus_vals[i], consensus_vals[i + 1]))
    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
    return pairs


def parse_vegasinsider_totals(html: str, max_games: int | None = None) -> list[dict[str, float | int | None]]:
    """
    Parse o/u rows (consensus column: 'o9 -102', 'u9 -118', 'o8.5 +102', etc.).
    Returns one dict per game: line, over_odds, under_odds (consensus).
    """
    soup = BeautifulSoup(html, "html.parser")
    rows: list[tuple[str, float, int]] = []
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        cell = _odds_cell_for_row(tds)
        m = re.match(
            r"^([ou])(\d+(?:\.\d+)?)\s+([+-]?\d+|even|Even|EVEN|pick|Pick|PK|pk)\s*$",
            cell.replace(" ", " "),
            re.I,
        )
        if not m:
            m = re.match(r"^([ou])(\d+(?:\.\d+)?)\s*([+-]?\d+|even)\s*$", cell.strip(), re.I)
        if not m:
            continue
        side, line_s, odds_s = m.group(1).lower(), m.group(2), m.group(3)
        line = float(line_s)
        os_low = odds_s.strip().lower()
        if os_low in ("even", "ev", "pick", "pk"):
            oi = 100
        else:
            try:
                oi = int(odds_s)
            except ValueError:
                continue
        rows.append((side, line, oi))

    games: list[dict[str, float | int | None]] = []
    i = 0
    while i + 1 < len(rows):
        a, b = rows[i], rows[i + 1]
        if a[0] == "o" and b[0] == "u" and abs(a[1] - b[1]) < 0.01:
            games.append(
                {
                    "line": a[1],
                    "over_odds": a[2],
                    "under_odds": b[2],
                }
            )
            i += 2
        else:
            i += 1
    if max_games is not None and len(games) > max_games:
        games = games[:max_games]
    return games


def scrape_odds_pages(max_games: int) -> tuple[list[tuple[int, int]], list[dict[str, float | int | None]], str]:
    """
    Try VegasInsider first, then OddsShark HTML (same HTML parsers; sites change often).
    Returns (ml_pairs, totals_list, bookmaker_label). Caps lists to max_games (statsapi slate size).
    """
    cap = max_games if max_games > 0 else None
    html = _fetch_html(VEGAS_INSIDER_MLB_ODDS)
    if html:
        ml = parse_vegasinsider_moneylines(html, max_pairs=cap)
        tot = parse_vegasinsider_totals(html, max_games=cap)
        if ml:
            return ml, tot, "VegasInsider (consensus)"
    html2 = _fetch_html(ODDS_SHARK_MLB_ODDS)
    if html2:
        ml2 = parse_vegasinsider_moneylines(html2, max_pairs=cap)
        tot2 = parse_vegasinsider_totals(html2, max_games=cap)
        if ml2:
            return ml2, tot2, "OddsShark (consensus-style)"
    return [], [], ""


def load_schedule_statsapi(date_et: datetime) -> list[dict]:
    import statsapi

    d = date_et.date().isoformat()
    games = statsapi.schedule(date=d) or []
    out: list[dict] = []
    for g in games:
        if not isinstance(g, dict):
            continue
        st = str(g.get("status") or "")
        if st in ("Cancelled", "Postponed"):
            continue
        out.append(g)
    out.sort(key=lambda x: str(x.get("game_datetime") or ""))
    return out


def build_payload() -> dict:
    now_et = datetime.now(ZoneInfo("America/New_York"))
    date_str = now_et.date().isoformat()
    sched = load_schedule_statsapi(now_et)
    nsched = len(sched)

    ml_pairs, totals_list, book_title = scrape_odds_pages(max_games=nsched)

    games_out: list[dict] = []
    for i, g in enumerate(sched):
        gid = str(g.get("game_id") or "")
        away = str(g.get("away_name") or "").strip()
        home = str(g.get("home_name") or "").strip()
        commence = str(g.get("game_datetime") or "")
        hp = (g.get("home_probable_pitcher") or None) and str(g.get("home_probable_pitcher")).strip()
        ap = (g.get("away_probable_pitcher") or None) and str(g.get("away_probable_pitcher")).strip()

        ml = {"home_odds": None, "away_odds": None}
        if i < len(ml_pairs):
            away_ml, home_ml = ml_pairs[i]
            ml["away_odds"] = away_ml
            ml["home_odds"] = home_ml

        total: dict[str, float | int | None] = {"line": None, "over_odds": None, "under_odds": None}
        if i < len(totals_list):
            t = totals_list[i]
            total["line"] = t.get("line")
            total["over_odds"] = t.get("over_odds")
            total["under_odds"] = t.get("under_odds")

        games_out.append(
            {
                "event_id": gid,
                "commence_time": commence,
                "home_team": home,
                "away_team": away,
                "home_pitcher": hp or None,
                "away_pitcher": ap or None,
                "bookmaker_title": book_title,
                "moneyline": ml,
                "total": total,
            }
        )

    now = datetime.now(timezone.utc)
    return {
        "fetched_at_utc": now.isoformat().replace("+00:00", "Z"),
        "sport_key": SPORT_KEY,
        "source": "statsapi_schedule+web_scrape",
        "bookmaker_title": book_title,
        "games_date_et": date_str,
        "games": games_out,
    }


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if _is_fresh(OUTPUT_PATH):
        print("Data is fresh.")
        return 0

    try:
        payload = build_payload()
    except Exception as e:
        print(f"Fetch failed: {e}", file=sys.stderr)
        return 1

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    n = len(payload.get("games") or [])
    print(f"Wrote {n} game(s) to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
