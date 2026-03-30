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

import argparse
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

from engine.mlb_engine import MLB_TEAM_NAME_ALIASES, normalize_mlb_team_name_for_join

OUTPUT_PATH = APP_ROOT / "data" / "odds" / "live_mlb_odds.json"
FRESHNESS_SECONDS = 30 * 60
SPORT_KEY = "baseball_mlb"

VEGAS_INSIDER_MLB_ODDS = "https://www.vegasinsider.com/mlb/odds/las-vegas/"
ODDS_SHARK_MLB_ODDS = "https://www.oddsshark.com/mlb/odds"

# VegasInsider nicknames / abbreviations → expand for fuzzy match vs statsapi full names.
VI_TEAM_LABEL_HINTS: dict[str, str] = {
    "d-backs": "Arizona Diamondbacks",
    "dbacks": "Arizona Diamondbacks",
    "diamondbacks": "Arizona Diamondbacks",
    "jays": "Toronto Blue Jays",
    "m's": "Seattle Mariners",
    "ms": "Seattle Mariners",
}

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


def _vi_row_team_label(first_cell: str) -> str | None:
    """First column like '957 Twins' or '958 Orioles' → team nickname 'Twins'."""
    s = first_cell.strip()
    if not s or s.lower() in ("time", "open") or s.startswith("›"):
        return None
    m = re.match(r"^(\d+)\s+(.+)$", s)
    if m:
        return m.group(2).strip()
    return s


def _vi_label_matches_statsapi_team(vi_label: str, statsapi_full_name: str) -> bool:
    """
    True when VI's first-column label refers to the same club as statsapi's full name.
    VI often uses the bare nickname ('Rays', 'Brewers'); statsapi uses city + name.
    """
    sched = normalize_mlb_team_name_for_join(statsapi_full_name)
    alias = MLB_TEAM_NAME_ALIASES.get(sched, sched)
    raw = vi_label.strip()
    if not raw or not alias:
        return False
    expanded = VI_TEAM_LABEL_HINTS.get(raw.lower(), raw)
    v = " ".join(expanded.split()).lower()
    a = " ".join(alias.split()).lower()
    if v == a or v in a:
        return True
    toks = a.split()
    if len(toks) >= 2 and v == f"{toks[-2]} {toks[-1]}":
        return True
    if toks and v == toks[-1]:
        return True
    return False


def _team_match_score(vi_label: str, statsapi_full_name: str) -> int:
    """How well a VI row label matches statsapi away_name / home_name."""
    if _vi_label_matches_statsapi_team(vi_label, statsapi_full_name):
        return 100
    try:
        from thefuzz import fuzz
    except ImportError:
        fuzz = None
    sched = normalize_mlb_team_name_for_join(statsapi_full_name)
    alias = MLB_TEAM_NAME_ALIASES.get(sched, sched)
    vi = vi_label.strip()
    expanded = VI_TEAM_LABEL_HINTS.get(vi.lower(), vi)
    if not fuzz:
        a, b = expanded.lower(), alias.lower()
        if a in b or b.endswith(a.split()[-1].lower()):
            return 85
        return max(50, 100 - 10 * abs(len(a) - len(b)))
    return max(
        fuzz.token_sort_ratio(expanded, alias),
        fuzz.token_sort_ratio(vi, alias),
        fuzz.token_sort_ratio(expanded, sched),
    )


def _resolve_pair_to_away_home_ml(
    away_team: str, home_team: str, pair: tuple[tuple[str, int], tuple[str, int]]
) -> tuple[int, int, float] | None:
    """
    Map a scraped ((teamA, mlA), (teamB, mlB)) to (away_ml, home_ml) using name match.
    Returns (away_ml, home_ml, confidence) or None if match is too weak.
    """
    (t1, m1), (t2, m2) = pair
    s1a = _team_match_score(t1, away_team)
    s1h = _team_match_score(t1, home_team)
    s2a = _team_match_score(t2, away_team)
    s2h = _team_match_score(t2, home_team)
    o1 = s1a + s2h
    o2 = s1h + s2a
    min_side = 62

    if o1 >= o2 and s1a >= min_side and s2h >= min_side:
        return m1, m2, float(o1)
    if o2 > o1 and s1h >= min_side and s2a >= min_side:
        return m2, m1, float(o2)
    return None


def _assign_moneylines_and_total_indices(
    sched: list[dict],
    ml_pairs: list[tuple[tuple[str, int], tuple[str, int]]],
) -> list[tuple[int | None, int | None, int | None]]:
    """
    For each schedule game (statsapi order), pick an unused VI pair that best matches
    away_name / home_name. Returns parallel list: (away_ml, home_ml, vi_pair_index).

    vi_pair_index aligns with parse_vegasinsider_totals order (same table pairing).
    """
    n = len(sched)
    out: list[tuple[int | None, int | None, int | None]] = [(None, None, None)] * n
    used: set[int] = set()
    for i, g in enumerate(sched):
        away = str(g.get("away_name") or "").strip()
        home = str(g.get("home_name") or "").strip()
        if not away or not home:
            continue
        best_j: int | None = None
        best_sc = -1.0
        best_mls: tuple[int, int] = (0, 0)
        for j, pair in enumerate(ml_pairs):
            if j in used:
                continue
            resolved = _resolve_pair_to_away_home_ml(away, home, pair)
            if resolved is None:
                continue
            aml, hml, sc = resolved
            if sc > best_sc:
                best_sc = sc
                best_j = j
                best_mls = (aml, hml)
        if best_j is not None:
            used.add(best_j)
            out[i] = (best_mls[0], best_mls[1], best_j)
    return out


def _odds_cell_for_row(tds: list) -> str:
    """VegasInsider uses column 1 for consensus; older layouts used last column."""
    if len(tds) >= 2:
        c1 = tds[1].get_text(" ", strip=True)
        if c1:
            return c1
    return tds[-1].get_text(" ", strip=True) if tds else ""


def debug_vegasinsider_moneyline_rows(html: str) -> list[dict[str, str | int]]:
    """
    Each scraped ML table row before pairing: rotation+team cell, consensus cell, parsed label/ML.
    """
    soup = BeautifulSoup(html, "html.parser")
    out: list[dict[str, str | int]] = []
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue
        first = tds[0].get_text(" ", strip=True)
        label = _vi_row_team_label(first)
        if not label:
            continue
        cell = _odds_cell_for_row(tds)
        cl = cell.lower()
        if re.match(r"^[ou]\d", cl):
            continue
        if re.search(r"\d+\.\d", cl):
            continue
        ci = _consensus_int(cell)
        if ci is None:
            continue
        out.append(
            {
                "first_cell": first,
                "consensus_cell": cell.strip(),
                "label": label,
                "ml": ci,
            }
        )
    return out


def _pair_labels_lower(pair: tuple[tuple[str, int], tuple[str, int]]) -> set[str]:
    return {pair[0][0].lower(), pair[1][0].lower()}


def _pair_matches_substrs(
    pair: tuple[tuple[str, int], tuple[str, int]], a: str, b: str
) -> bool:
    labs = _pair_labels_lower(pair)
    return any(a in x for x in labs) and any(b in x for x in labs)


def print_raw_vi_moneylines_for_verification(html: str) -> None:
    """Stdout: numbered rows, sequential pairs, then spotlight NYM@STL and WSH@PHI (pre-schedule match)."""
    rows = debug_vegasinsider_moneyline_rows(html)
    pairs = parse_vegasinsider_moneyline_pairs(html, max_pairs=None)

    print("--- VegasInsider raw ML rows (consensus column, before schedule pairing) ---")
    for i, r in enumerate(rows):
        print(
            f"  row[{i:3d}] first_cell={r['first_cell']!r} | "
            f"consensus={r['consensus_cell']!r} → label={r['label']!r} ml={r['ml']}"
        )

    print("\n--- Sequential pairs (row[2k], row[2k+1]) — VI table order, not statsapi ---")
    for j, p in enumerate(pairs):
        (t1, m1), (t2, m2) = p
        print(f"  pair[{j:3d}] ({t1!r}, {m1})  |  ({t2!r}, {m2})")

    print("\n--- Spotlight: Mets @ Cardinals (look for Mets + Cardinals in same pair) ---")
    mets_stl = [
        (j, p)
        for j, p in enumerate(pairs)
        if _pair_matches_substrs(p, "mets", "cardinal")
    ]
    if not mets_stl:
        print("  (no pair with both Mets and Cardinals labels)")
    for j, p in mets_stl:
        (t1, m1), (t2, m2) = p
        print(f"  pair[{j}] order not away/home: {t1} {m1:+d}  |  {t2} {m2:+d}")

    print("\n--- Spotlight: Nationals @ Phillies (look for Nationals + Phillies in same pair) ---")
    was_phi = [
        (j, p)
        for j, p in enumerate(pairs)
        if _pair_matches_substrs(p, "national", "phillie")
    ]
    if not was_phi:
        print("  (no pair with both Nationals and Phillies labels)")
    for j, p in was_phi:
        (t1, m1), (t2, m2) = p
        print(f"  pair[{j}] order not away/home: {t1} {m1:+d}  |  {t2} {m2:+d}")

    print(
        "\n(Home/away on each side is determined only after matching this pair to statsapi "
        "away_name/home_name; first row in a pair is not necessarily away.)"
    )


def parse_vegasinsider_moneyline_pairs(
    html: str, max_pairs: int | None = None
) -> list[tuple[tuple[str, int], tuple[str, int]]]:
    """
    Parse VI MLB table: each moneyline row has rotation + team in col0, consensus ML in col1.
    Returns [((team_label, ml), (team_label, ml)), ...] — one tuple per game (two teams).
    Skips over/under and run-line rows (not plain integer ML in consensus cell).
    """
    soup = BeautifulSoup(html, "html.parser")
    rows: list[tuple[str, int]] = []
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue
        first = tds[0].get_text(" ", strip=True)
        label = _vi_row_team_label(first)
        if not label:
            continue
        cell = _odds_cell_for_row(tds)
        cl = cell.lower()
        if re.match(r"^[ou]\d", cl):
            continue
        if re.search(r"\d+\.\d", cl):
            continue
        ci = _consensus_int(cell)
        if ci is None:
            continue
        rows.append((label, ci))
    pairs: list[tuple[tuple[str, int], tuple[str, int]]] = []
    for i in range(0, len(rows) - 1, 2):
        pairs.append((rows[i], rows[i + 1]))
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


def scrape_odds_pages(
    max_pairs_hint: int,
) -> tuple[list[tuple[tuple[str, int], tuple[str, int]]], list[dict[str, float | int | None]], str]:
    """
    Try VegasInsider first, then OddsShark. ML pairs include team labels for schedule matching.
    Parse extra pairs (up to 2x slate) so reordering vs statsapi still matches; cap for safety.
    """
    cap = None
    if max_pairs_hint > 0:
        cap = min(60, max_pairs_hint * 2 + 8)
    html = _fetch_html(VEGAS_INSIDER_MLB_ODDS)
    if html:
        ml = parse_vegasinsider_moneyline_pairs(html, max_pairs=cap)
        tot = parse_vegasinsider_totals(html, max_games=cap)
        if ml:
            return ml, tot, "VegasInsider (consensus)"
    html2 = _fetch_html(ODDS_SHARK_MLB_ODDS)
    if html2:
        ml2 = parse_vegasinsider_moneyline_pairs(html2, max_pairs=cap)
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

    ml_pairs, totals_list, book_title = scrape_odds_pages(max_pairs_hint=nsched)
    ml_assign = _assign_moneylines_and_total_indices(sched, ml_pairs)

    games_out: list[dict] = []
    for i, g in enumerate(sched):
        gid = str(g.get("game_id") or "")
        away = str(g.get("away_name") or "").strip()
        home = str(g.get("home_name") or "").strip()
        commence = str(g.get("game_datetime") or "")
        hp = (g.get("home_probable_pitcher") or None) and str(g.get("home_probable_pitcher")).strip()
        ap = (g.get("away_probable_pitcher") or None) and str(g.get("away_probable_pitcher")).strip()

        ml = {"home_odds": None, "away_odds": None}
        away_ml, home_ml, vi_idx = ml_assign[i]
        if away_ml is not None:
            ml["away_odds"] = away_ml
        if home_ml is not None:
            ml["home_odds"] = home_ml

        total: dict[str, float | int | None] = {"line": None, "over_odds": None, "under_odds": None}
        if vi_idx is not None and vi_idx < len(totals_list):
            t = totals_list[vi_idx]
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
    parser = argparse.ArgumentParser(description="MLB schedule + scraped consensus odds → live_mlb_odds.json")
    parser.add_argument(
        "--print-raw-vi",
        action="store_true",
        help="Fetch VegasInsider, print raw ML rows and pairs (before statsapi matching), then exit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore 30m freshness guard and rewrite JSON.",
    )
    args = parser.parse_args()

    if args.print_raw_vi:
        html = _fetch_html(VEGAS_INSIDER_MLB_ODDS)
        if not html:
            print("Failed to fetch VegasInsider HTML.", file=sys.stderr)
            return 1
        print_raw_vi_moneylines_for_verification(html)
        return 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not args.force and _is_fresh(OUTPUT_PATH):
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
