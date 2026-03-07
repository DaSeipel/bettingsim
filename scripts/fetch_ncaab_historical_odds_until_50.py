#!/usr/bin/env python3
"""
Fetch historical NCAAB odds from The Odds API (v4 historical endpoint) one day at a time
starting from 60 days ago. Cache each response to disk. Insert into historical_odds in data/odds.db.
Stop when at least 50 completed NCAAB games have closing lines (after merge). Use API key from
.streamlit/secrets.toml. Print credits used per request; stop if remaining drops below 50.
Then run merge_historical_closing_into_games() and print how many NCAAB games have non-null closing_home_spread.
"""

import os
import re
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _get_api_key() -> str:
    key = (os.environ.get("ODDS_API_KEY") or "").strip()
    if key:
        return key
    secrets_path = ROOT / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            text = secrets_path.read_text(encoding="utf-8")
            m = re.search(r'\[the_odds_api\].*?api_key\s*=\s*["\']([^"\']+)["\']', text, re.DOTALL)
            if m:
                return m.group(1).strip()
        except (OSError, re.error):
            pass
    return ""


def _count_ncaab_with_closing(espn_db_path: Path) -> int:
    if not espn_db_path.exists():
        return 0
    conn = sqlite3.connect(espn_db_path)
    try:
        cur = conn.execute(
            "SELECT COUNT(*) FROM games_with_team_stats WHERE league = 'ncaab' AND closing_home_spread IS NOT NULL"
        )
        return cur.fetchone()[0]
    finally:
        conn.close()


def main() -> None:
    api_key = _get_api_key()
    if not api_key:
        print("Set ODDS_API_KEY or add [the_odds_api] api_key to .streamlit/secrets.toml")
        sys.exit(1)

    from engine.engine import BASKETBALL_NCAAB
    from engine.historical_odds import (
        fetch_historical_odds,
        load_cached_response,
        cache_response,
        _parse_historical_response,
        insert_historical_odds,
        merge_historical_closing_into_games,
    )
    from engine.odds_quota import get_quota_status

    db_path = ROOT / "data" / "odds.db"
    espn_db_path = ROOT / "data" / "espn.db"
    sport_key = BASKETBALL_NCAAB
    target_count = 50
    min_remaining = 50

    # Use NCAAB game date range from espn.db so we fetch dates that can match our games
    conn = sqlite3.connect(espn_db_path)
    try:
        cur = conn.execute(
            "SELECT MIN(game_date), MAX(game_date) FROM games_with_team_stats WHERE league = 'ncaab'"
        )
        row = cur.fetchone()
        min_date_s, max_date_s = (row[0], row[1]) if row and row[0] and row[1] else (None, None)
    finally:
        conn.close()
    if not max_date_s:
        print("No NCAAB games in games_with_team_stats.")
        sys.exit(1)
    try:
        end_date = datetime.fromisoformat(max_date_s.replace("Z", "+00:00"))
    except Exception:
        end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=60)
    current = start_date
    day_count = 0
    fetched_count = 0  # API calls made

    print(f"Fetching historical NCAAB odds from {current.date()} to {end_date.date()} (60 days before latest NCAAB game) until we have >={target_count} games with closing lines (or run out of days/credits).")
    print()

    while current <= end_date:
        date_iso = current.strftime("%Y-%m-%dT12:00:00Z")

        # Use cache if present (no API call)
        cached = load_cached_response(sport_key, date_iso)
        if cached is not None:
            rows = _parse_historical_response(cached, sport_key)
            inserted = insert_historical_odds(rows, db_path=db_path)
            if inserted > 0:
                day_count += 1
            current += timedelta(days=1)
            # Still run merge and check count
            merge_historical_closing_into_games(espn_db_path=espn_db_path, odds_db_path=db_path)
            n = _count_ncaab_with_closing(espn_db_path)
            if n >= target_count:
                print(f"Date {current.date()}: used cache (no API call). NCAAB games with closing_home_spread: {n}. Done.")
                break
            continue

        # No cache: check credits before fetching
        status = get_quota_status()
        remaining = status.get("requests_remaining")
        if remaining is not None and int(remaining) < min_remaining:
            print(f"Stopping: credits remaining ({remaining}) below {min_remaining}.")
            break

        resp = fetch_historical_odds(api_key, sport_key, date_iso)
        if resp is None:
            # API skipped (quota) or error (e.g. 401 = historical not on free tier)
            status = get_quota_status()
            rem = status.get("requests_remaining")
            print(f"Request for {date_iso[:10]} failed or skipped. Credits remaining: {rem}. (Historical odds require a paid Odds API plan on free tier.)")
            if rem is not None and int(rem) < min_remaining:
                print(f"Stopping: credits remaining below {min_remaining}.")
            break
        cache_response(sport_key, date_iso, resp)
        rows = _parse_historical_response(resp, sport_key)
        inserted = insert_historical_odds(rows, db_path=db_path)
        fetched_count += 1
        day_count += 1

        status = get_quota_status()
        used = status.get("requests_last")
        rem = status.get("requests_remaining")
        print(f"Date {date_iso[:10]}: inserted {inserted} rows. Credits used this request: {used}. Credits remaining: {rem}.")

        if rem is not None and int(rem) < min_remaining:
            print(f"Stopping: credits remaining ({rem}) below {min_remaining}.")
            break

        current += timedelta(days=1)

        # Merge and check if we have enough games with closing lines
        merge_historical_closing_into_games(espn_db_path=espn_db_path, odds_db_path=db_path)
        n = _count_ncaab_with_closing(espn_db_path)
        if n >= target_count:
            print(f"NCAAB games with closing_home_spread: {n}. Target reached.")
            break

    # Final merge and report
    print()
    print("Running final merge_historical_closing_into_games()...")
    merge_historical_closing_into_games(espn_db_path=espn_db_path, odds_db_path=db_path)
    n = _count_ncaab_with_closing(espn_db_path)
    print(f"NCAAB games with non-null closing_home_spread: {n}")


if __name__ == "__main__":
    main()
