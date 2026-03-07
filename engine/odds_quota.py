"""
Quota management for The Odds API free tier.
Tracks x-requests-remaining and x-requests-used from response headers; skips calls when remaining < 10.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

MIN_REMAINING_BEFORE_SKIP = 3
QUOTA_HEADER_REMAINING = "x-requests-remaining"
QUOTA_HEADER_USED = "x-requests-used"
QUOTA_HEADER_LAST = "x-requests-last"


def _quota_file_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "odds_quota.json"


def _read_quota() -> dict[str, Any]:
    path = _quota_file_path()
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _write_quota(data: dict[str, Any]) -> None:
    path = _quota_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        logger.warning("Could not write quota file %s: %s", path, e)


def get_quota_status() -> dict[str, Any]:
    """Return current quota state: requests_remaining, requests_used, last_updated, requests_last (cost of last call)."""
    data = _read_quota()
    return {
        "requests_remaining": data.get("requests_remaining"),
        "requests_used": data.get("requests_used"),
        "requests_last": data.get("requests_last"),
        "last_updated": data.get("last_updated"),
    }


def update_quota_from_response(response: requests.Response) -> None:
    """Read x-requests-remaining and x-requests-used from response headers and persist to JSON."""
    headers = response.headers
    remaining = headers.get(QUOTA_HEADER_REMAINING)
    used = headers.get(QUOTA_HEADER_USED)
    last = headers.get(QUOTA_HEADER_LAST)
    if remaining is None and used is None:
        return
    data = _read_quota()
    if remaining is not None:
        try:
            data["requests_remaining"] = int(remaining)
        except ValueError:
            pass
    if used is not None:
        try:
            data["requests_used"] = int(used)
        except ValueError:
            pass
    if last is not None:
        try:
            data["requests_last"] = int(last)
        except ValueError:
            pass
    data["last_updated"] = datetime.now(timezone.utc).isoformat()
    _write_quota(data)


def should_skip_odds_api_call() -> bool:
    """Return True if we should skip the next Odds API call (remaining < MIN_REMAINING_BEFORE_SKIP)."""
    data = _read_quota()
    remaining = data.get("requests_remaining")
    if remaining is None:
        return False
    try:
        return int(remaining) < MIN_REMAINING_BEFORE_SKIP
    except (TypeError, ValueError):
        return False


def odds_api_get(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int | float = 15,
    session: requests.Session | None = None,
) -> requests.Response | None:
    """
    GET request for The Odds API with quota checks.
    If remaining credits < 10, skips the call, logs a warning, and returns None.
    Otherwise performs the request and updates quota from response headers.
    """
    if should_skip_odds_api_call():
        status = get_quota_status()
        logger.warning(
            "Odds API call skipped: remaining credits (%s) below threshold (%s). Run 'python quota.py status' for details.",
            status.get("requests_remaining"),
            MIN_REMAINING_BEFORE_SKIP,
        )
        return None
    if session is not None:
        resp = session.get(url, params=params, headers=headers or {}, timeout=timeout)
    else:
        resp = requests.get(url, params=params, headers=headers or {}, timeout=timeout)
    update_quota_from_response(resp)
    return resp
