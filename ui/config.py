"""Single-source dashboard sport and season configuration."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo


CFB_SHADOW_MODE = True
CFB_MODEL_STATUS = "failed_gate_2026_08_19"
EASTERN = ZoneInfo("America/New_York")
PIPELINE_FRESH_HOURS = 6
APP_ROOT = Path(__file__).resolve().parents[1]

PIPELINE_CACHE_PATHS = {
    "CFB": APP_ROOT / "data" / "cache" / "cfb_value_plays.json",
    "MLB": APP_ROOT / "data" / "cache" / "mlb_value_plays.json",
    "NCAAB": APP_ROOT / "data" / "cache" / "value_plays_cache.json",
}

PipelineStatus = Literal["fresh", "stale", "missing"]
logger = logging.getLogger(__name__)

SPORT_CONFIG = {
    "CFB": {
        "start": (8, 20),
        "end": (1, 20),
        "returns": "August 20",
        "accent": "#8e7cc3",
    },
    "MLB": {
        "start": (3, 15),
        "end": (11, 5),
        "returns": "March 15",
        "accent": "#42a5f5",
    },
    "NCAAB": {
        "start": (11, 1),
        "end": (4, 10),
        "returns": "November 1",
        "accent": "#ff9800",
    },
}


def sport_is_in_season(sport: str, on_date: date | None = None) -> bool:
    """Return whether ``on_date`` falls inside the configured season window."""
    on_date = on_date or date.today()
    config = SPORT_CONFIG[sport]
    start_month, start_day = config["start"]
    end_month, end_day = config["end"]
    marker = (on_date.month, on_date.day)
    start = (start_month, start_day)
    end = (end_month, end_day)
    if start <= end:
        return start <= marker <= end
    return marker >= start or marker <= end


def season_bounds(sport: str, on_date: date | None = None) -> tuple[date, date]:
    """Current active season, or the most recently completed season."""
    on_date = on_date or date.today()
    config = SPORT_CONFIG[sport]
    start_month, start_day = config["start"]
    end_month, end_day = config["end"]
    crosses_year = (start_month, start_day) > (end_month, end_day)

    if not crosses_year:
        start_year = on_date.year
        start = date(start_year, start_month, start_day)
        end = date(start_year, end_month, end_day)
        if on_date < start:
            start = date(start_year - 1, start_month, start_day)
            end = date(start_year - 1, end_month, end_day)
        return start, end

    if (on_date.month, on_date.day) >= (start_month, start_day):
        start_year = on_date.year
    else:
        start_year = on_date.year - 1
    return (
        date(start_year, start_month, start_day),
        date(start_year + 1, end_month, end_day),
    )


def season_note(sport: str) -> str:
    return f"Season complete — returns {SPORT_CONFIG[sport]['returns']}."


def active_sports(on_date: date | None = None) -> dict[str, bool]:
    return {
        sport: sport_is_in_season(sport, on_date)
        for sport in SPORT_CONFIG
    }


def _parse_timestamp(raw: object) -> datetime | None:
    if raw is None or not str(raw).strip():
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def load_pipeline_cache(
    sport: str,
    path: Path | None = None,
) -> dict:
    """Load one sport cache and normalize its odds timestamp at the read boundary."""
    sport = sport.upper()
    cache_path = path or PIPELINE_CACHE_PATHS[sport]
    if not cache_path.exists():
        return {"_exists": False, "_path": str(cache_path)}
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"_exists": False, "_path": str(cache_path)}
    if not isinstance(raw, dict):
        return {"_exists": True, "_path": str(cache_path), "_invalid": True}

    normalized = dict(raw)
    normalized["_exists"] = True
    normalized["_path"] = str(cache_path)
    if not normalized.get("odds_captured_at") and normalized.get("timestamp"):
        normalized["odds_captured_at"] = normalized["timestamp"]
        logger.warning(
            "%s pipeline cache uses legacy 'timestamp'; normalized to 'odds_captured_at'",
            sport,
        )
    return normalized


def _cache_entry_dates(cache: dict) -> set[date]:
    dates: set[date] = set()
    for key in ("card_date", "game_date", "date"):
        raw = cache.get(key)
        if raw:
            try:
                dates.add(date.fromisoformat(str(raw)[:10]))
            except ValueError:
                pass
    for key in ("plays", "value_plays"):
        rows = cache.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            for date_key in ("card_date", "game_date", "date"):
                raw = row.get(date_key)
                if raw:
                    try:
                        dates.add(date.fromisoformat(str(raw)[:10]))
                    except ValueError:
                        pass
    return dates


def pipeline_status(
    sport: str,
    *,
    cache: dict | None = None,
    now: datetime | None = None,
    path: Path | None = None,
) -> PipelineStatus:
    """Classify today's cache in ET, checking calendar date before elapsed age."""
    sport = sport.upper()
    cache = cache if cache is not None else load_pipeline_cache(sport, path)
    if not cache.get("_exists"):
        return "missing"

    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now_utc = now.astimezone(timezone.utc)
    today_et = now_utc.astimezone(EASTERN).date()
    captured = _parse_timestamp(cache.get("odds_captured_at"))
    entry_dates = _cache_entry_dates(cache)

    # Legacy NCAAB caches have no card_date. Their normalized capture date is
    # the only available indication that an entry exists for today's slate.
    if not entry_dates and captured is not None:
        entry_dates.add(captured.astimezone(EASTERN).date())
    if today_et not in entry_dates:
        return "missing"
    if captured is None:
        return "stale"
    captured_et = captured.astimezone(EASTERN)
    if captured_et.date() != today_et:
        return "stale"
    age_hours = (now_utc - captured.astimezone(timezone.utc)).total_seconds() / 3600
    return "fresh" if 0 <= age_hours <= PIPELINE_FRESH_HOURS else "stale"
