#!/usr/bin/env python3
"""
Fetch NCAAB historical game results and write data/ncaab/historical_games.csv.

Current focus: refresh the 2026 season, which is currently incomplete
(~200 games through Nov 5 only). We try Barttorvik's game-level CSV endpoint
first; if that fails, we log an error and exit without touching existing data.

Columns in historical_games.csv:
  date, home_team, away_team, home_score, away_score, margin, neutral_site, closing_spread

Note: closing_spread is not provided by Barttorvik and will remain empty here.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
import re
from typing import Any

import pandas as pd
import requests

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from engine.utils import game_season_from_date

NCAAB_DIR = APP_ROOT / "data" / "ncaab"
HISTORICAL_GAMES_PATH = NCAAB_DIR / "historical_games.csv"

BARTTORVIK_RESULTS_URLS = [
    # Primary guess for game-level 2026 results
    "https://barttorvik.com/2026_results.csv",
    # Fallback: some installations expose season results at this path
    "https://barttorvik.com/2026_team_results.csv",
]


def _fetch_barttorvik_2026_games() -> pd.DataFrame:
    """Try to fetch 2026 game results from Barttorvik. Returns DataFrame or empty on failure."""
    for url in BARTTORVIK_RESULTS_URLS:
        try:
            print(f"Trying Barttorvik URL: {url}", flush=True)
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            print(f"  Failed to fetch {url}: {e}", file=sys.stderr, flush=True)
            continue
        try:
            df = pd.read_csv(pd.compat.StringIO(resp.text), header=None)
        except Exception:
            # Fallback parser if pd.compat.StringIO not available
            try:
                from io import StringIO
                df = pd.read_csv(StringIO(resp.text), header=None)
            except Exception as e:
                print(f"  Failed to parse CSV from {url}: {e}", file=sys.stderr, flush=True)
                continue
        if df.empty:
            print(f"  {url} returned empty CSV.", flush=True)
            continue
        print(f"  Loaded {len(df)} rows from {url}", flush=True)
        return df
    print("ERROR: All Barttorvik 2026 URLs failed. Leaving historical_games.csv unchanged.", file=sys.stderr)
    return pd.DataFrame()


def _normalize_team_name(s: Any) -> str:
    if pd.isna(s):
        return ""
    return str(s).strip()


def _to_historical_games(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse Barttorvik 2026_results.csv (which has no header row) into the
    historical_games schema:

      date, home_team, away_team, home_score, away_score, margin, neutral_site, closing_spread

    Column semantics (0-indexed):
      0: two team names concatenated with the date digits (e.g. 'QueensWinthrop11-3')
      1: game date string (e.g. '11/3/25')
      2: favored team summary, including spread and score, e.g. 'Winthrop -6.5, 83-76 (72%)'
      5: first team score (integer)
      6: second team score (integer)

    We:
      - read with header=None so every row is data
      - parse column 1 as a date
      - use regex on column 2 to extract: favored team name, spread, and the two scores
      - use columns 5 and 6 (integer scores) plus the favored team's score to determine
        which team is home vs away
      - use column 0 plus the favored team name and trailing date digits to recover
        the other team name
      - derive closing_spread from the home team's perspective
    """
    records: list[dict[str, Any]] = []

    # Diagnostics for failed parses.
    fail_counts: dict[str, int] = {}
    fail_examples_printed = 0

    def _note_failure(reason: str, row: pd.Series) -> None:
        nonlocal fail_examples_printed
        fail_counts[reason] = fail_counts.get(reason, 0) + 1
        if fail_examples_printed < 10:
            fail_examples_printed += 1
            # Print the raw key columns for early failures so we can inspect format issues.
            try:
                col0 = row.iloc[0]
            except Exception:
                col0 = None
            try:
                col1 = row.iloc[1]
            except Exception:
                col1 = None
            try:
                col2 = row.iloc[2]
            except Exception:
                col2 = None
            try:
                col5 = row.iloc[5]
            except Exception:
                col5 = None
            try:
                col6 = row.iloc[6]
            except Exception:
                col6 = None
            print(
                f"[BARTTORVIK PARSE FAIL] reason={reason} "
                f"c0={col0!r} c1={col1!r} c2={col2!r} c5={col5!r} c6={col6!r}",
                flush=True,
            )

    for _, row in df.iterrows():
        # Access by position; many rows are slightly ragged so guard IndexErrors.
        try:
            col0 = row.iloc[0]
            col1 = row.iloc[1]
            col2 = row.iloc[2]
        except Exception:
            _note_failure("missing_core_columns_0_1_2", row)
            continue

        # Actual final scores are always in columns 5 (away) and 6 (home).
        try:
            away_score = int(row.iloc[5])
            home_score = int(row.iloc[6])
        except Exception:
            _note_failure("score_columns_5_6_invalid", row)
            continue

        # Parse date from column 1.
        date = pd.to_datetime(col1, errors="coerce")
        if pd.isna(date):
            _note_failure("date_parse_failed_col1", row)
            continue

        # Extract favored team and spread from column 2. Column 2 contains Torvik's
        # prediction, not the final score. Two main formats:
        #   "Winthrop -6.5, 83-76 (72%)"  -> fav="Winthrop", spread=6.5
        #   "East Texas A&M (100%)"       -> fav="East Texas A&M", spread=None
        text_c2 = str(col2)
        m_spread = re.search(
            r"^(?P<fav>.+?)\s+(?P<spread>[+-]?\d+(?:\.\d+)?),",
            text_c2,
        )
        m_nospread = None
        if not m_spread:
            m_nospread = re.search(
                r"^(?P<fav>.+?)\s*\(",
                text_c2,
            )

        if m_spread:
            fav_team = _normalize_team_name(m_spread.group("fav"))
            spread_str = m_spread.group("spread")
            try:
                fav_spread = float(spread_str) if spread_str not in (None, "", "NaN") else pd.NA
            except Exception:
                fav_spread = pd.NA
        elif m_nospread:
            fav_team = _normalize_team_name(m_nospread.group("fav"))
            fav_spread = pd.NA
        else:
            _note_failure("regex_no_match_col2", row)
            continue

        # Recover both team names from column 0, which looks like AwayHomeDATE
        # (e.g. 'QueensWinthrop11-3'). Strip the trailing date digits first.
        col0_str = str(col0)
        col0_str = re.sub(r"\d{1,2}-\d{1,2}$", "", col0_str).strip()

        if not fav_team:
            _note_failure("fav_team_empty_after_regex", row)
            continue

        col0_lower = col0_str.lower()
        fav_lower = fav_team.lower()
        idx = col0_lower.find(fav_lower)
        if idx == -1:
            _note_failure("team_split_no_fav_in_col0", row)
            continue

        # Column 0 is away-team first, home-team second. Use the favored team
        # position within this string to decide which side it's on.
        if idx == 0:
            # Favored team is at the start -> away team.
            away_raw = col0_str[: len(fav_team)]
            home_raw = col0_str[len(fav_team) :]
            fav_is_home = False
        else:
            # Favored team is the trailing part -> home team.
            away_raw = col0_str[:idx]
            home_raw = col0_str[idx : idx + len(fav_team)]
            fav_is_home = True

        away_team = _normalize_team_name(away_raw)
        home_team = _normalize_team_name(home_raw)

        if not home_team or not away_team:
            _note_failure("team_split_failed_empty_home_or_away", row)
            continue

        margin = home_score - away_score

        if pd.isna(fav_spread):
            home_spread = pd.NA
        else:
            # Store closing_spread from home team perspective: negative if
            # home is favored, positive if away is favored.
            home_spread = -fav_spread if fav_is_home else fav_spread

        records.append(
            {
                "date": date,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "margin": margin,
                "neutral_site": 0,
                "closing_spread": home_spread,
            }
        )

    total_rows = len(df)
    parsed_rows = len(records)
    failed_rows = total_rows - parsed_rows

    print(
        f"Barttorvik parsing summary: total_rows={total_rows}, "
        f"parsed_rows={parsed_rows}, failed_rows={failed_rows}",
        flush=True,
    )
    if fail_counts:
        # Report the most common failure reason.
        most_common_reason, most_common_count = max(
            fail_counts.items(), key=lambda kv: kv[1]
        )
        print(
            "Most common parse failure: "
            f"{most_common_reason} ({most_common_count} rows)",
            flush=True,
        )

    if not records:
        print("ERROR: No parsable rows from Barttorvik 2026_results.csv.", file=sys.stderr)
        return pd.DataFrame()

    out = pd.DataFrame.from_records(records)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])

    # Season filter
    out["season"] = out["date"].apply(game_season_from_date)
    out = out[out["season"] == 2026].copy()
    out = out.drop(columns=["season"])

    # Final formatting: date as YYYY-MM-DD string.
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def main() -> int:
    print("Inspecting existing historical_games.csv for 2026...", flush=True)
    existing = pd.DataFrame()
    if HISTORICAL_GAMES_PATH.exists():
        existing = pd.read_csv(HISTORICAL_GAMES_PATH)
        existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
        existing = existing.dropna(subset=["date"])
        existing["season"] = existing["date"].apply(game_season_from_date)
        g26 = existing[existing["season"] == 2026].copy()
        print(f"Existing 2026 games: {len(g26)}")
        if not g26.empty:
            print(f"  Earliest 2026: {g26['date'].min().strftime('%Y-%m-%d')}")
            print(f"  Latest   2026: {g26['date'].max().strftime('%Y-%m-%d')}")
    else:
        print("No existing historical_games.csv; will create new file.")

    print("\nFetching 2026 games from Barttorvik (best-effort)...", flush=True)
    df_raw = _fetch_barttorvik_2026_games()
    if df_raw.empty:
        return 1

    new_games = _to_historical_games(df_raw)
    if new_games.empty:
        print("No new 2026 games parsed from Barttorvik.", flush=True)
        return 1

    print(f"Parsed {len(new_games)} 2026 games from Barttorvik.", flush=True)

    # Diagnostics: total 2026 games from Barttorvik, date range, and a small sample.
    new_games_dates = pd.to_datetime(new_games["date"], errors="coerce")
    valid_dates = new_games_dates.dropna()
    if not valid_dates.empty:
        print(
            f"  2026 Barttorvik games: {len(new_games)} "
            f"(from {valid_dates.min().strftime('%Y-%m-%d')} "
            f"to {valid_dates.max().strftime('%Y-%m-%d')})",
            flush=True,
        )
    # Show a sample of the most recent 2026 games (typically February/March).
    sample = new_games.sort_values("date").tail(5)
    if not sample.empty:
        print("\nSample of 5 parsed 2026 games:", flush=True)
        for _, r in sample.iterrows():
            print(
                f"  {r['date']}: {r['home_team']} vs {r['away_team']} "
                f"{int(r['home_score'])}-{int(r['away_score'])} "
                f"margin={int(r['margin'])} "
                f"spread={r['closing_spread']}",
                flush=True,
            )

    if not existing.empty:
        cols = ["date", "home_team", "away_team", "home_score", "away_score", "margin", "neutral_site", "closing_spread"]
        existing = existing[cols]
        combined = pd.concat([existing, new_games[cols]], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
    else:
        combined = new_games

    # Ensure dates are all comparable before sorting.
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.sort_values("date").reset_index(drop=True)
    NCAAB_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(HISTORICAL_GAMES_PATH, index=False, quoting=csv.QUOTE_NONNUMERIC)

    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined["season"] = combined["date"].apply(game_season_from_date)

    # Per-season diagnostics for 2021–2026.
    print("\nGames per season (from historical_games.csv):", flush=True)
    for yr in range(2021, 2027):
        g_season = combined[combined["season"] == yr]
        print(f"  {yr}: {len(g_season)} games", flush=True)

    g26_new = combined[combined["season"] == 2026].copy()
    print(f"\nAfter fetch, 2026 games: {len(g26_new)}")
    if not g26_new.empty:
        print(f"  Earliest 2026: {g26_new['date'].min().strftime('%Y-%m-%d')}")
        print(f"  Latest   2026: {g26_new['date'].max().strftime('%Y-%m-%d')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

