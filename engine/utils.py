"""
Shared utilities for date/season parsing used across engine and scripts.
Consolidates _parse_date and _game_season_from_date to avoid duplication and drift.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional


def parse_date(s: Any) -> Optional[datetime]:
    """
    Parse game_date string or value to datetime. Returns None if invalid.
    Supports: YYYY-MM-DD, YYYY/MM/DD, and "Jan 15, 2024" (e.g. from nba_api).
    """
    if s is None:
        return None
    try:
        import pandas as pd
        if pd.isna(s):
            return None
    except ImportError:
        pass
    s = str(s).strip()
    if not s:
        return None
    if len(s) >= 10:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            try:
                return datetime.strptime(s[:10], "%Y/%m/%d")
            except ValueError:
                pass
    try:
        return datetime.strptime(s, "%b %d, %Y")
    except ValueError:
        pass
    return None


def game_season_from_date(game_date: Any) -> Optional[int]:
    """
    Derive season (end year) from game_date. Oct+ -> next year; else current year.
    E.g. 2024-11-15 -> 2025, 2024-03-01 -> 2024.
    """
    if game_date is None:
        return None
    try:
        import pandas as pd
        if pd.isna(game_date):
            return None
    except ImportError:
        pass
    s = str(game_date).strip()
    if not s or len(s) < 4:
        return None
    try:
        year = int(s[:4])
        if len(s) >= 7:
            month = int(s[5:7])
            if month >= 10:
                return year + 1
        return year
    except (ValueError, TypeError):
        return None


def effective_kenpom_season(
    game_date: Any,
    season: Optional[int],
    cutoff_month: int = 1,
    cutoff_day: int = 1,
) -> Optional[int]:
    """
    For KenPom lag (no end-of-season leakage): games before cutoff use prior season's ratings.
    E.g. cutoff (1, 1) = Jan 1 of season year: a Nov 2024 game (season 2025) uses season 2024 KenPom.
    Returns season or season-1; if season is None or game_date invalid, returns season unchanged.
    """
    if season is None:
        return None
    d = parse_date(game_date)
    if d is None:
        return season
    try:
        cutoff = date(season, cutoff_month, cutoff_day)
    except (ValueError, TypeError):
        cutoff = date(season, 1, 1)
    return season if d.date() >= cutoff else (season - 1)
