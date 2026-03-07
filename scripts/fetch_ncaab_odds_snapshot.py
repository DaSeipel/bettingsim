#!/usr/bin/env python3
"""
Fetch current NCAAB odds from The Odds API and append the snapshot to data/odds.db
(odds table, odds_snapshots, and historical_odds). Run 2–3x daily (e.g. cron) to build
real closing lines for games; merge_historical_closing_into_games will then fill
closing_home_spread for completed games. API key: ODDS_API_KEY env or .streamlit/secrets.toml [the_odds_api] api_key.
"""

import os
import re
import sys
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


def main() -> None:
    api_key = _get_api_key()
    if not api_key:
        print("Set ODDS_API_KEY or add [the_odds_api] api_key to .streamlit/secrets.toml")
        sys.exit(1)
    from engine.engine import BASKETBALL_NCAAB
    from engine.odds_fetcher import fetch_and_store
    db_path = ROOT / "data" / "odds.db"
    result = fetch_and_store(
        api_key,
        sport_keys=[BASKETBALL_NCAAB],
        use_cache=False,
        db_path=db_path,
    )
    df = result.get(BASKETBALL_NCAAB)
    n = len(df) if df is not None and not df.empty else 0
    print(f"NCAAB odds snapshot stored: {n} rows (odds + odds_snapshots + historical_odds)")


if __name__ == "__main__":
    main()
