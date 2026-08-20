"""CFB Phase 0 configuration — API keys, paths, constants."""

from __future__ import annotations

import os
import re
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
ESPN_DB_PATH = APP_ROOT / "data" / "espn.db"
SECRETS_PATH = APP_ROOT / ".streamlit" / "secrets.toml"

CFBD_BASE_URL = "https://api.collegefootballdata.com"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
CFB_SPORT_KEY = "americanfootball_ncaaf"

BACKFILL_START_SEASON = 2015
BACKFILL_END_SEASON = 2025
SEASON_TYPES = ("regular", "postseason")
RATING_SCOPE_END_OF_SEASON = "end_of_season"
RATING_SCOPE_PRESEASON = "preseason"

REQUEST_SLEEP_S = 0.35
REQUEST_TIMEOUT_S = 90


def _read_section_key(text: str, section: str, key: str) -> str:
    pattern = rf"\[{re.escape(section)}\][^\[]*?{re.escape(key)}\s*=\s*['\"]([^'\"]+)['\"]"
    match = re.search(pattern, text, re.S | re.I)
    return match.group(1).strip() if match else ""


def get_cfbd_api_key() -> str:
    key = (os.environ.get("CFBD_API_KEY") or "").strip()
    if key:
        return key
    if SECRETS_PATH.exists():
        text = SECRETS_PATH.read_text(encoding="utf-8")
        key = _read_section_key(text, "cfbd", "api_key")
        if key:
            return key
    return ""


def get_odds_api_key() -> str:
    key = (os.environ.get("ODDS_API_KEY") or "").strip()
    if key:
        return key
    if SECRETS_PATH.exists():
        text = SECRETS_PATH.read_text(encoding="utf-8")
        key = _read_section_key(text, "the_odds_api", "api_key")
        if key:
            return key
    return ""
