#!/usr/bin/env python3
"""
Fetch hourly weather (Open-Meteo, no API key) for each MLB game listed in live odds JSON.

Reads: data/cache/live_mlb_odds.json (falls back to data/odds/live_mlb_odds.json)
Writes: data/cache/mlb_weather.json — dict keyed by "Away Team @ Home Team" with
  temp_f, wind_speed_mph, wind_direction_deg, precip_prob

Weather row is the hourly slot closest to the game's commence_time (UTC); if missing,
uses games_date_et at 19:00 America/New_York.

Daily order (with MLB model):
  python3 scripts/fetch_mlb_odds.py
  python3 scripts/fetch_mlb_weather.py   # after odds; ~1 Open-Meteo call per game/venue
  python3 scripts/predict_mlb.py

Usage:
  python3 scripts/fetch_mlb_weather.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import requests

APP_ROOT = Path(__file__).resolve().parent.parent
ODDS_PATH_CACHE = APP_ROOT / "data" / "cache" / "live_mlb_odds.json"
ODDS_PATH_FALLBACK = APP_ROOT / "data" / "odds" / "live_mlb_odds.json"
OUT_PATH = APP_ROOT / "data" / "cache" / "mlb_weather.json"

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT = 25
REQUEST_SLEEP_S = 0.35

# Full MLB team display name -> ballpark name (home team in odds JSON).
TEAM_TO_VENUE: dict[str, str] = {
    "Arizona Diamondbacks": "Chase Field",
    "Atlanta Braves": "Truist Park",
    "Baltimore Orioles": "Oriole Park at Camden Yards",
    "Boston Red Sox": "Fenway Park",
    "Chicago Cubs": "Wrigley Field",
    "Chicago White Sox": "Rate Field",
    "Cincinnati Reds": "Great American Ball Park",
    "Cleveland Guardians": "Progressive Field",
    "Colorado Rockies": "Coors Field",
    "Detroit Tigers": "Comerica Park",
    "Houston Astros": "Daikin Park",
    "Kansas City Royals": "Kauffman Stadium",
    "Los Angeles Angels": "Angel Stadium",
    "Los Angeles Dodgers": "Dodger Stadium",
    "Miami Marlins": "loanDepot park",
    "Milwaukee Brewers": "American Family Field",
    "Minnesota Twins": "Target Field",
    "New York Mets": "Citi Field",
    "New York Yankees": "Yankee Stadium",
    "Athletics": "Sutter Health Park",
    "Oakland Athletics": "Sutter Health Park",
    "Philadelphia Phillies": "Citizens Bank Park",
    "Pittsburgh Pirates": "PNC Park",
    "San Diego Padres": "Petco Park",
    "San Francisco Giants": "Oracle Park",
    "Seattle Mariners": "T-Mobile Park",
    "St. Louis Cardinals": "Busch Stadium",
    "Tampa Bay Rays": "Tropicana Field",
    "Texas Rangers": "Globe Life Field",
    "Toronto Blue Jays": "Rogers Centre",
    "Washington Nationals": "Nationals Park",
}

# Ballpark name -> (latitude, longitude). All current MLB venues.
VENUE_COORDS: dict[str, tuple[float, float]] = {
    "Chase Field": (33.4453, -112.0667),
    "Truist Park": (33.8906, -84.4678),
    "Oriole Park at Camden Yards": (39.2839, -76.6217),
    "Fenway Park": (42.3467, -71.0972),
    "Wrigley Field": (41.9484, -87.6553),
    "Rate Field": (41.8299, -87.6338),
    "Great American Ball Park": (39.0974, -84.5066),
    "Progressive Field": (41.4962, -81.6852),
    "Coors Field": (39.7560, -104.9940),
    "Comerica Park": (42.3390, -83.0485),
    "Daikin Park": (29.7573, -95.3555),
    "Minute Maid Park": (29.7573, -95.3555),
    "Kauffman Stadium": (39.0514, -94.4805),
    "Angel Stadium": (33.8003, -117.8827),
    "Dodger Stadium": (34.0739, -118.2400),
    "loanDepot park": (25.7781, -80.2197),
    "American Family Field": (43.0280, -87.9712),
    "Target Field": (44.9817, -93.2776),
    "Citi Field": (40.7571, -73.8458),
    "Yankee Stadium": (40.8296, -73.9265),
    "Sutter Health Park": (38.5804, -121.5138),
    "Citizens Bank Park": (39.9061, -75.1665),
    "PNC Park": (40.4469, -80.0057),
    "Petco Park": (32.7073, -117.1570),
    "Oracle Park": (37.7786, -122.3893),
    "T-Mobile Park": (47.5914, -122.3325),
    "Busch Stadium": (38.6226, -90.1928),
    "Tropicana Field": (27.7682, -82.6534),
    "Globe Life Field": (32.7473, -97.0819),
    "Rogers Centre": (43.6414, -79.3894),
    "Nationals Park": (38.8730, -77.0075),
}


def _load_odds_path(cli_path: Optional[Path]) -> Path:
    if cli_path and cli_path.exists():
        return cli_path
    if ODDS_PATH_CACHE.exists():
        return ODDS_PATH_CACHE
    return ODDS_PATH_FALLBACK


def _matchup_key(away: str, home: str) -> str:
    return f"{away.strip()} @ {home.strip()}"


def _parse_game_time_utc(
    game: dict[str, Any],
    games_date_et: Optional[str],
) -> datetime:
    ct = game.get("commence_time")
    if ct:
        s = str(ct).strip()
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    # Default: 7 PM Eastern on slate date
    d = games_date_et or datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    local = datetime.strptime(d, "%Y-%m-%d").replace(hour=19, minute=0, second=0)
    local = local.replace(tzinfo=ZoneInfo("America/New_York"))
    return local.astimezone(timezone.utc)


def _fetch_forecast(lat: float, lon: float) -> dict[str, Any]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,precipitation_probability",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "UTC",
    }
    r = requests.get(OPEN_METEO, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _hour_index_closest_to(
    hourly_times: list[str],
    target_utc: datetime,
) -> int:
    """Return index of hourly time string closest to target_utc."""
    if not hourly_times:
        return 0
    target_ts = target_utc.timestamp()
    best_i = 0
    best_d = float("inf")
    for i, ts in enumerate(hourly_times):
        # Open-Meteo returns "2026-04-13T20:00" without Z — interpret as UTC
        tstr = str(ts)
        if len(tstr) == 16:
            dt = datetime.fromisoformat(tstr).replace(tzinfo=timezone.utc)
        else:
            dt = datetime.fromisoformat(tstr.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(timezone.utc)
        d = abs(dt.timestamp() - target_ts)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _extract_hour(
    payload: dict[str, Any],
    idx: int,
) -> dict[str, float | int]:
    h = payload.get("hourly") or {}
    temps = h.get("temperature_2m") or []
    winds = h.get("wind_speed_10m") or []
    dirs = h.get("wind_direction_10m") or []
    pcp = h.get("precipitation_probability") or []

    def _safe_float(seq: list, i: int) -> float:
        if i < len(seq) and seq[i] is not None:
            try:
                return float(seq[i])
            except (TypeError, ValueError):
                pass
        return float("nan")

    def _safe_int(seq: list, i: int) -> int:
        if i < len(seq) and seq[i] is not None:
            try:
                return int(round(float(seq[i])))
            except (TypeError, ValueError):
                pass
        return 0

    return {
        "temp_f": round(_safe_float(temps, idx), 1),
        "wind_speed_mph": round(_safe_float(winds, idx), 1),
        "wind_direction_deg": _safe_int(dirs, idx),
        "precip_prob": _safe_int(pcp, idx),
    }


def run(odds_path: Optional[Path] = None, dry_run: bool = False) -> dict[str, Any]:
    path = _load_odds_path(odds_path)
    if not path.exists():
        print(f"ERROR: odds file not found: {path}", file=sys.stderr)
        return {}

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    games = data.get("games") or []
    games_date_et = data.get("games_date_et")
    fetched_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    out: dict[str, Any] = {}
    errors: list[str] = []

    for game in games:
        if not isinstance(game, dict):
            continue
        home = str(game.get("home_team", "")).strip()
        away = str(game.get("away_team", "")).strip()
        if not home or not away:
            continue

        key = _matchup_key(away, home)
        venue = TEAM_TO_VENUE.get(home)
        if not venue:
            errors.append(f"No TEAM_TO_VENUE for home={home!r} ({key})")
            continue

        coords = VENUE_COORDS.get(venue)
        if not coords:
            errors.append(f"No VENUE_COORDS for venue={venue!r} ({key})")
            continue

        lat, lon = coords
        target_utc = _parse_game_time_utc(game, games_date_et)

        try:
            payload = _fetch_forecast(lat, lon)
            hourly = payload.get("hourly") or {}
            times = hourly.get("time") or []
            if not times:
                errors.append(f"Open-Meteo: no hourly times for {key}")
                continue
            idx = _hour_index_closest_to(times, target_utc)
            wx = _extract_hour(payload, idx)
            if wx["temp_f"] != wx["temp_f"]:  # NaN check
                errors.append(f"Bad weather values for {key}")
                continue
            out[key] = {
                "venue": venue,
                "latitude": lat,
                "longitude": lon,
                "forecast_hour_utc": times[idx] if idx < len(times) else None,
                "commence_time_utc": game.get("commence_time"),
                **wx,
            }
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{key}: {exc}")

        time.sleep(REQUEST_SLEEP_S)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "fetched_at_utc": fetched_at,
        "odds_source": str(path),
        "games_date_et": games_date_et,
        "weather": out,
        "errors": errors,
    }
    if not dry_run:
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(document, f, indent=2)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch MLB game weather via Open-Meteo.")
    parser.add_argument(
        "--odds",
        type=Path,
        default=None,
        help=f"Path to live MLB odds JSON (default: {ODDS_PATH_CACHE} or fallback)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print summary only; do not write JSON.")
    args = parser.parse_args()

    doc = run(odds_path=args.odds, dry_run=args.dry_run)
    n = len(doc.get("weather") or {})
    print(f"Weather rows: {n}")
    for err in doc.get("errors") or []:
        print(f"  WARN: {err}", file=sys.stderr)
    if not args.dry_run:
        print(f"Wrote {OUT_PATH}")
    return 0 if not doc.get("errors") or n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
