#!/usr/bin/env python3
"""
Parse NBA historical odds from existing SBR files and load into historical_covers_odds.
No auto-download: place nba_YYYY_YY.html or nba_YYYY_YY.xls/.xlsx in data/raw_odds/ (e.g. from sportsbookreviewsonline.com).
Run from repo root: python3 scripts/download_sbr_nba_odds.py
"""

from __future__ import annotations

import io
import re
import sqlite3
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import requests

SBR_BASE = "https://www.sportsbookreviewsonline.com/scoresoddsarchives"
SBR_NBA_BASE = "https://www.sportsbookreviewsonline.com/scoresoddsarchives"
SEASONS = [
    ("2022-23", "nba-odds-2022-23"),
    ("2023-24", "nba-odds-2023-24"),
    ("2024-25", "nba-odds-2024-25"),
]
# Alternative URLs to try for Excel (tried in order; first real Excel wins).
# Trailing-slash URLs often return HTML; .xlsx/.xls are checked for binary Excel.
EXCEL_ALT_URLS = {
    "2023-24": [
        f"{SBR_NBA_BASE}/nba-odds-2023-24/",
        f"{SBR_NBA_BASE}/nba/nba-odds-2023-24.xlsx",
        f"{SBR_NBA_BASE}/nba-odds-2023-24.xlsx",
    ],
    "2024-25": [
        f"{SBR_NBA_BASE}/nba-odds-2024-25/",
        f"{SBR_NBA_BASE}/nba/nba-odds-2024-25.xlsx",
        f"{SBR_NBA_BASE}/nba-odds-2024-25.xlsx",
    ],
}
RAW_ODDS_DIR = ROOT / "data" / "raw_odds"
ODDS_DB = ROOT / "data" / "odds.db"
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}


def _ensure_raw_odds_dir() -> Path:
    RAW_ODDS_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_ODDS_DIR


def _fetch_url(url: str) -> str | None:
    try:
        r = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"  Fetch error: {e}")
        return None


def download_season_html(season_slug: str, season_label: str) -> Path | None:
    """Fetch season page HTML and save to data/raw_odds/nba_{season}.html. Returns path or None."""
    _ensure_raw_odds_dir()
    url = f"{SBR_BASE}/{season_slug}"
    if not url.endswith(".htm") and not url.endswith(".html"):
        url = url  # same URL
    html = _fetch_url(url)
    if not html:
        return None
    safe = season_label.replace("-", "_")
    path = RAW_ODDS_DIR / f"nba_{safe}.html"
    path.write_text(html, encoding="utf-8", errors="replace")
    print(f"  Saved {path}")
    return path


def _is_real_excel(content: bytes) -> bool:
    """True if content looks like Excel (OLE .xls or ZIP .xlsx), not HTML."""
    if len(content) < 8:
        return False
    raw = content[:200]
    if raw.startswith(b"<!") or raw.startswith(b"<html") or b"DOCTYPE" in raw:
        return False
    # .xls: D0 CF 11 E0 (OLE compound document)
    if content[:4] == b"\xD0\xCF\x11\xE0":
        return True
    # .xlsx: PK (ZIP)
    if content[:2] == b"PK":
        return True
    return False


def _try_excel_download(season_slug: str, season_label: str) -> tuple[Path | None, str | None]:
    """
    Try Excel URL patterns; save to data/raw_odds/ only if response is real Excel.
    Returns (saved_path, url_that_worked) or (None, None).
    For 2023-24 and 2024-25 uses EXCEL_ALT_URLS; otherwise tries SBR_BASE/slug.xls and .xlsx.
    """
    _ensure_raw_odds_dir()
    safe = season_label.replace("-", "_")

    urls_to_try: list[str] = []
    if season_label in EXCEL_ALT_URLS:
        urls_to_try = list(EXCEL_ALT_URLS[season_label])
    else:
        urls_to_try = [
            f"{SBR_BASE}/{season_slug}.xls",
            f"{SBR_BASE}/{season_slug}.xlsx",
        ]

    for url in urls_to_try:
        try:
            r = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
            if r.status_code != 200:
                continue
            if not _is_real_excel(r.content):
                continue
            # Choose extension from content
            if r.content[:2] == b"PK":
                ext = ".xlsx"
            else:
                ext = ".xls"
            path = RAW_ODDS_DIR / f"nba_{safe}{ext}"
            path.write_bytes(r.content)
            print(f"  Saved Excel: {path}")
            return path, url
        except Exception as e:
            continue
    return None, None


def _parse_mmdd_to_date(mmdd: str, season_label: str) -> str | None:
    """Convert MMDD and season (e.g. 2022-23) to YYYY-MM-DD. Oct=10->year 2022, Jan=01->year 2023."""
    s = str(mmdd).strip()
    if len(s) < 4:
        return None
    try:
        month = int(s[:2])
        day = int(s[2:4])
    except ValueError:
        return None
    if month < 1 or month > 12 or day < 1 or day > 31:
        return None
    # 2022-23: Oct-Dec = 2022, Jan-Jun = 2023
    start_year = int(season_label[:4])
    end_year = int(season_label[5:7])
    year = end_year if month <= 6 else start_year
    return f"{year:04d}-{month:02d}-{day:02d}"


def _parse_spread_or_total(val: str) -> float | None:
    """Parse Open/Close value; 'pk' -> 0."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip().lower()
    if s in ("pk", "pick", ""):
        return 0.0
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _looks_like_total(x: float | None) -> bool:
    return x is not None and 200 <= x <= 260


def _looks_like_spread(x: float | None) -> bool:
    return x is not None and -25 <= x <= 25


def _normalize_team_name(name: str) -> str:
    """e.g. LALakers -> Los Angeles Lakers, NewYork -> New York."""
    if not name or not isinstance(name, str):
        return ""
    s = name.strip()
    # Common SBR abbreviations
    fixes = {
        "LALakers": "Los Angeles Lakers",
        "LAClippers": "Los Angeles Clippers",
        "NewYork": "New York Knicks",
        "NewOrleans": "New Orleans Pelicans",
        "GoldenState": "Golden State Warriors",
        "SanAntonio": "San Antonio Spurs",
        "OklahomaCity": "Oklahoma City Thunder",
    }
    return fixes.get(s, s)


def parse_sbr_html(html: str, season_label: str) -> pd.DataFrame:
    """
    Parse SBR NBA odds HTML table. Rows are pairs (V, H). Columns: Date, Rot, VH, Team, 1st, 2nd, 3rd, 4th, Final, Open, Close, ML, 2H.
    V row: Open=opening total, Close=closing total, Final=away score.
    H row: Open=opening spread (home), Close=closing spread (home), Final=home score.
    """
    try:
        tables = pd.read_html(io.StringIO(html), flavor="lxml")
    except (ImportError, ValueError):
        try:
            tables = pd.read_html(io.StringIO(html))
        except (ValueError, ImportError):
            return pd.DataFrame()
    if not tables:
        return pd.DataFrame()
    return _df_from_sbr_table(tables[0], season_label)


def _df_from_sbr_table(df: pd.DataFrame, season_label: str) -> pd.DataFrame:
    """Shared logic: turn a DataFrame with columns Date, VH, Team, Final, Open, Close into game rows."""
    if df.shape[1] < 11:
        return pd.DataFrame()
    df = df.rename(columns={
        df.columns[0]: "Date",
        df.columns[1]: "Rot",
        df.columns[2]: "VH",
        df.columns[3]: "Team",
        df.columns[8]: "Final",
        df.columns[9]: "Open",
        df.columns[10]: "Close",
    })
    mask = df["Date"].astype(str).str.strip().str.match(r"^\d{4}$")
    df = df.loc[mask].copy()
    if df.empty:
        return pd.DataFrame()
    df["Date"] = df["Date"].astype(str).str.strip()
    df["VH"] = df["VH"].astype(str).str.strip().str.upper()
    df["Team"] = df["Team"].astype(str).str.strip()
    rows = []
    i = 0
    while i < len(df) - 1:
        rv = df.iloc[i]
        rh = df.iloc[i + 1]
        if rv["VH"] != "V" or rh["VH"] != "H":
            i += 1
            continue
        date_str = _parse_mmdd_to_date(str(rv["Date"]), season_label)
        if not date_str:
            i += 1
            continue
        away_team = _normalize_team_name(str(rv["Team"]))
        home_team = _normalize_team_name(str(rh["Team"]))
        try:
            away_score = float(rv["Final"]) if pd.notna(rv["Final"]) else None
            home_score = float(rh["Final"]) if pd.notna(rh["Final"]) else None
        except (TypeError, ValueError):
            away_score = home_score = None
        v_open = _parse_spread_or_total(rv.get("Open"))
        v_close = _parse_spread_or_total(rv.get("Close"))
        h_open = _parse_spread_or_total(rh.get("Open"))
        h_close = _parse_spread_or_total(rh.get("Close"))
        # Source usually: V=total, H=spread. Sometimes swapped (e.g. Washington @ Indiana).
        if _looks_like_total(v_open) and _looks_like_spread(h_open):
            open_total, close_total = v_open, v_close
            open_spread, close_spread = h_open, h_close
        elif _looks_like_spread(v_open) and _looks_like_total(h_open):
            open_total, close_total = h_open, h_close
            open_spread = -v_open if v_open is not None else None
            close_spread = -v_close if v_close is not None else None
        else:
            open_total, close_total = v_open, v_close
            open_spread, close_spread = h_open, h_close
        rows.append({
            "date": date_str,
            "home_team": home_team,
            "away_team": away_team,
            "opening_spread": open_spread,
            "closing_spread": close_spread,
            "opening_total": open_total,
            "closing_total": close_total,
            "away_score": away_score,
            "home_score": home_score,
            "season": season_label,
        })
        i += 2
    return pd.DataFrame(rows)


def parse_sbr_excel(path: Path, season_label: str) -> pd.DataFrame:
    """Parse SBR .xls or .xlsx file (same column layout as HTML)."""
    try:
        if path.suffix.lower() == ".xlsx":
            try:
                df = pd.read_excel(path, engine="openpyxl", header=None)
            except ImportError:
                df = pd.read_excel(path, header=None)
        else:
            df = pd.read_excel(path, engine="xlrd", header=None)
    except Exception as e:
        print(f"  Excel read error: {e}")
        return pd.DataFrame()
    return _df_from_sbr_table(df, season_label)


def _create_historical_covers_odds_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS historical_covers_odds (
            date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            opening_spread REAL,
            closing_spread REAL,
            opening_total REAL,
            closing_total REAL,
            away_score REAL,
            home_score REAL,
            season TEXT,
            PRIMARY KEY (date, home_team, away_team)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_historical_covers_odds_date ON historical_covers_odds(date)"
    )


def load_parsed_into_db(df: pd.DataFrame, db_path: Path) -> int:
    """Insert parsed games into historical_covers_odds. Returns count inserted."""
    if df.empty:
        return 0
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        _create_historical_covers_odds_table(conn)
        cur = conn.cursor()
        n = 0
        for _, row in df.iterrows():
            try:
                cur.execute(
                    """INSERT OR REPLACE INTO historical_covers_odds
                       (date, home_team, away_team, opening_spread, closing_spread, opening_total, closing_total, away_score, home_score, season)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        row["date"], row["home_team"], row["away_team"],
                        row.get("opening_spread"), row.get("closing_spread"),
                        row.get("opening_total"), row.get("closing_total"),
                        row.get("away_score"), row.get("home_score"),
                        row.get("season"),
                    ),
                )
                n += cur.rowcount
            except sqlite3.IntegrityError:
                pass
        conn.commit()
        return n
    finally:
        conn.close()


def main() -> int:
    print("SBR NBA odds parser (no auto-download; uses existing files in data/raw_odds/)")
    print("=" * 50)
    _ensure_raw_odds_dir()
    all_parsed = []
    for season_label, season_slug in SEASONS:
        print(f"\nSeason {season_label}")
        safe = season_label.replace("-", "_")
        html_path = RAW_ODDS_DIR / f"nba_{safe}.html"
        excel_path = None
        for ext in (".xls", ".xlsx"):
            p = RAW_ODDS_DIR / f"nba_{safe}{ext}"
            if p.exists():
                excel_path = p
                break
        df = pd.DataFrame()
        from_excel = False
        if html_path.exists():
            html = html_path.read_text(encoding="utf-8", errors="replace")
            df = parse_sbr_html(html, season_label)
        if df.empty and excel_path and excel_path.exists():
            df = parse_sbr_excel(excel_path, season_label)
            from_excel = not df.empty
        if df.empty:
            print(f"  No data for {season_label} (add nba_{safe}.html or nba_{safe}.xls/.xlsx to data/raw_odds/)")
            continue
        print(f"  Parsed {len(df)} games" + (" from Excel" if from_excel else " from HTML"))
        all_parsed.append(df)
    if not all_parsed:
        print("\nNo data parsed. Check raw HTML in data/raw_odds/.")
        return 1
    combined = pd.concat(all_parsed, ignore_index=True)
    n = load_parsed_into_db(combined, ODDS_DB)
    print(f"\nInserted/updated {n} rows into historical_covers_odds ({ODDS_DB})")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
