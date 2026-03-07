"""
The Rundown API (RapidAPI) as an odds source.
Base URL: https://therundown-therundown-v1.p.rapidapi.com
NCAAB sport_id=6, NBA sport_id=2.
Maps to live_odds_df schema: sport_key, league, event_id, commence_time, home_team, away_team,
event_name, market_type, selection, point, odds (American).
"""

from __future__ import annotations

import os
import re
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any

import pandas as pd
import requests

from .engine import BASKETBALL_NBA, BASKETBALL_NCAAB, SPORT_KEY_TO_LEAGUE

RUNDOWN_BASE = "https://therundown-therundown-v1.p.rapidapi.com"
REQUEST_TIMEOUT = 15


def _debug_rundown_odds() -> bool:
    return os.environ.get("DEBUG_RUNDOWN_ODDS", "") == "1"

# sport_id -> (sport_key, league). The Rundown: NCAAB=5, NBA=2 (RapidAPI v1 may differ; 6=NHL)
RUNDOWN_SPORT_IDS = {
    5: (BASKETBALL_NCAAB, "NCAAB"),
    2: (BASKETBALL_NBA, "NBA"),
}


def _load_rundown_secrets() -> tuple[str, str]:
    """Load api_key and host from .streamlit/secrets.toml [therundown] or env."""
    api_key = (os.environ.get("RUNDOWN_API_KEY") or "").strip()
    host = (os.environ.get("RUNDOWN_HOST") or "").strip()
    if api_key and host:
        return (api_key, host)
    try:
        root = Path(__file__).resolve().parent.parent
        secrets_path = root / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            text = secrets_path.read_text()
            in_therundown = False
            for line in text.splitlines():
                if line.strip() == "[therundown]":
                    in_therundown = True
                    continue
                if in_therundown and line.strip().startswith("["):
                    break
                if in_therundown:
                    m = re.match(r'api_key\s*=\s*["\']([^"\']+)["\']', line.strip())
                    if m:
                        api_key = m.group(1).strip()
                    m = re.match(r'host\s*=\s*["\']([^"\']+)["\']', line.strip())
                    if m:
                        host = m.group(1).strip()
    except Exception:
        pass
    return (api_key or "", host or "therundown-therundown-v1.p.rapidapi.com")


def _fetch_rundown_events(
    sport_id: int,
    on_date: date,
    api_key: str,
    host: str,
    session: requests.Session,
) -> list[dict] | None:
    """GET /sports/{sport_id}/events/{YYYY-MM-DD}?include=scores&affiliate_ids=1,2,3&offset=0"""
    if not api_key.strip():
        if _debug_rundown_odds():
            print("[DEBUG_RUNDOWN] (3) _fetch_rundown_events skipped: no api_key")
        return None
    date_str = on_date.strftime("%Y-%m-%d")
    url = f"{RUNDOWN_BASE}/sports/{sport_id}/events/{date_str}"
    params = {"include": "scores", "affiliate_ids": "1,2,3", "offset": 0}
    headers = {
        "x-rapidapi-host": host.strip(),
        "x-rapidapi-key": api_key.strip(),
    }
    if _debug_rundown_odds() and sport_id == 5:
        print(f"[DEBUG_RUNDOWN] (2) Request URL: {url}")
        print(f"[DEBUG_RUNDOWN] (2) Params: {params}")
    try:
        r = session.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        if _debug_rundown_odds():
            print(f"[DEBUG_RUNDOWN] (3) The Rundown API error: {type(e).__name__}: {e}")
            if hasattr(e, "response") and e.response is not None:
                try:
                    print(f"[DEBUG_RUNDOWN] (3) Response status: {e.response.status_code}")
                    print(f"[DEBUG_RUNDOWN] (3) Response body (first 1500 chars): {e.response.text[:1500]}")
                except Exception:
                    pass
        return None
    if _debug_rundown_odds() and sport_id == 5:
        import json
        try:
            raw_str = json.dumps(data, indent=2)
            print(f"[DEBUG_RUNDOWN] (2) Raw response for NCAAB (sport_id=5) today:\n{raw_str[:8000]}{'...' if len(raw_str) > 8000 else ''}")
        except Exception as ser:
            print(f"[DEBUG_RUNDOWN] (2) Raw response (could not serialize): {type(data)} {str(data)[:1500]}")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("events") or data.get("event") or []
    return None


def _commence_iso(ev: dict, on_date: date) -> str:
    """Get commence time as ISO string (UTC)."""
    for key in ("commence_time", "start_date", "event_date", "date", "start_datetime"):
        val = ev.get(key)
        if val is None:
            continue
        if isinstance(val, str) and "T" in val:
            if not val.endswith("Z") and "+" not in val and "-" not in val[-6:]:
                val = val + "Z" if val[-1] != "Z" else val
            return val.replace("Z", "+00:00")
        if isinstance(val, (int, float)):
            try:
                dt = datetime.fromtimestamp(val, tz=timezone.utc)
                return dt.isoformat()
            except (OSError, ValueError):
                pass
    # Fallback: noon UTC on event date
    return f"{on_date.isoformat()}T12:00:00+00:00"


def _teams_from_event(ev: dict) -> tuple[str, str]:
    """Extract (away_team, home_team) from event."""
    teams = ev.get("teams")
    if isinstance(teams, list) and len(teams) >= 2:
        t0 = teams[0].get("name") if isinstance(teams[0], dict) else str(teams[0])
        t1 = teams[1].get("name") if isinstance(teams[1], dict) else str(teams[1])
        if t0 and t1:
            return (str(t0).strip(), str(t1).strip())
    if isinstance(teams, list) and len(teams) == 1:
        t0 = teams[0].get("name") if isinstance(teams[0], dict) else str(teams[0])
        if t0:
            return (str(t0).strip(), "")
    for key in ("away_team", "team_away", "away"):
        a = ev.get(key)
        if isinstance(a, str) and a.strip():
            h = ev.get("home_team") or ev.get("team_home") or ev.get("home") or ""
            if isinstance(h, str) and h.strip():
                return (a.strip(), h.strip())
    name = ev.get("event_name") or ev.get("name") or ""
    if " at " in str(name):
        parts = str(name).split(" at ", 1)
        return (parts[0].strip(), parts[1].strip() if len(parts) > 1 else "")
    if " @ " in str(name):
        parts = str(name).split(" @ ", 1)
        return (parts[0].strip(), parts[1].strip() if len(parts) > 1 else "")
    return ("", "")


# Sentinel: line off the board (do not use in odds)
OFF_BOARD_PRICE = 0.0001


def _price_american(raw: Any) -> int | None:
    """Convert to American odds int; ignore off-board."""
    if raw is None:
        return None
    try:
        p = float(raw)
    except (TypeError, ValueError):
        return None
    if abs(p - OFF_BOARD_PRICE) < 1e-6 or abs(p) < 100:
        return None
    return int(round(p))


def _rows_from_event_v2_style(
    ev: dict,
    sport_key: str,
    league: str,
    on_date: date,
) -> list[dict]:
    """Parse V2-style event: markets[] with participants[].lines[].prices. market_id 1=ML, 2=spread, 3=total."""
    event_id = str(ev.get("event_id") or ev.get("id") or "")
    away_team, home_team = _teams_from_event(ev)
    if not home_team or not away_team:
        return []
    commence = _commence_iso(ev, on_date)
    event_name = f"{away_team} @ {home_team}"
    rows: list[dict] = []
    markets = ev.get("markets") or []
    for mkt in markets:
        if not isinstance(mkt, dict):
            continue
        market_id = mkt.get("market_id")
        if market_id is None:
            market_id = mkt.get("market_type")
        mkt_name = (mkt.get("name") or "").lower()
        market_type = "h2h"
        if market_id == 2 or "spread" in mkt_name or "handicap" in mkt_name:
            market_type = "spreads"
        elif market_id == 3 or "total" in mkt_name or "over" in mkt_name or "under" in mkt_name:
            market_type = "totals"
        elif market_id == 1 or "money" in mkt_name or "h2h" in mkt_name:
            market_type = "h2h"
        for part in mkt.get("participants") or []:
            if not isinstance(part, dict):
                continue
            name = (part.get("name") or "").strip()
            if not name:
                continue
            for line in part.get("lines") or []:
                if not isinstance(line, dict):
                    continue
                prices = line.get("prices") or {}
                if not isinstance(prices, dict):
                    continue
                # Use first available price (e.g. first affiliate)
                for _aff_id, price_obj in prices.items():
                    if not isinstance(price_obj, dict):
                        continue
                    price = price_obj.get("price")
                    odds = _price_american(price)
                    if odds is None:
                        continue
                    value = line.get("value")
                    point = None
                    if value not in (None, ""):
                        try:
                            point = float(value)
                        except (TypeError, ValueError):
                            pass
                    selection = name
                    if market_type == "spreads":
                        if point is None:
                            break
                        rows.append({
                            "sport_key": sport_key, "league": league, "event_id": event_id,
                            "commence_time": commence, "home_team": home_team, "away_team": away_team,
                            "event_name": event_name, "market_type": market_type, "selection": name,
                            "point": point, "odds": odds,
                        })
                    elif market_type == "h2h":
                        rows.append({
                            "sport_key": sport_key, "league": league, "event_id": event_id,
                            "commence_time": commence, "home_team": home_team, "away_team": away_team,
                            "event_name": event_name, "market_type": market_type, "selection": name,
                            "point": None, "odds": odds,
                        })
                    elif market_type == "totals" and point is not None:
                        sel = f"Over {point}" if "over" in name.lower() or name == "Over" else f"Under {point}"
                        rows.append({
                            "sport_key": sport_key, "league": league, "event_id": event_id,
                            "commence_time": commence, "home_team": home_team, "away_team": away_team,
                            "event_name": event_name, "market_type": market_type, "selection": sel,
                            "point": point, "odds": odds,
                        })
                    break
                break
    return rows


def _rows_from_event_lines_dict(
    ev: dict,
    sport_key: str,
    league: str,
    on_date: date,
) -> list[dict]:
    """Parse event when lines is a dict: affiliate_id -> { moneyline: {}, spread: {}, total: {} } (RapidAPI v1)."""
    event_id = str(ev.get("event_id") or ev.get("id") or "")
    away_team, home_team = _teams_from_event(ev)
    if not home_team or not away_team:
        return []
    commence = _commence_iso(ev, on_date)
    event_name = f"{away_team} @ {home_team}"
    rows: list[dict] = []
    lines = ev.get("lines")
    if not isinstance(lines, dict):
        return []
    for _aff_id, line_block in lines.items():
        if not isinstance(line_block, dict):
            continue
        ml = line_block.get("moneyline") or {}
        sp = line_block.get("spread") or {}
        tot = line_block.get("total") or {}
        added = 0
        # Moneyline
        for key, sel in (("moneyline_away", away_team), ("moneyline_home", home_team)):
            odds = _price_american(ml.get(key))
            if odds is not None:
                rows.append({
                    "sport_key": sport_key, "league": league, "event_id": event_id,
                    "commence_time": commence, "home_team": home_team, "away_team": away_team,
                    "event_name": event_name, "market_type": "h2h", "selection": sel,
                    "point": None, "odds": odds,
                })
                added += 1
        # Spread
        ps_away = sp.get("point_spread_away")
        ps_home = sp.get("point_spread_home")
        money_away = _price_american(sp.get("point_spread_away_money"))
        money_home = _price_american(sp.get("point_spread_home_money"))
        if ps_away is not None and money_away is not None:
            try:
                point = float(ps_away)
                if abs(point - OFF_BOARD_PRICE) >= 1e-6:
                    rows.append({
                        "sport_key": sport_key, "league": league, "event_id": event_id,
                        "commence_time": commence, "home_team": home_team, "away_team": away_team,
                        "event_name": event_name, "market_type": "spreads", "selection": away_team,
                        "point": point, "odds": money_away,
                    })
                    added += 1
            except (TypeError, ValueError):
                pass
        if ps_home is not None and money_home is not None:
            try:
                point = float(ps_home)
                if abs(point - OFF_BOARD_PRICE) >= 1e-6:
                    rows.append({
                        "sport_key": sport_key, "league": league, "event_id": event_id,
                        "commence_time": commence, "home_team": home_team, "away_team": away_team,
                        "event_name": event_name, "market_type": "spreads", "selection": home_team,
                        "point": point, "odds": money_home,
                    })
                    added += 1
            except (TypeError, ValueError):
                pass
        # Total
        over_pt = tot.get("total_over")
        under_pt = tot.get("total_under")
        over_money = _price_american(tot.get("total_over_money"))
        under_money = _price_american(tot.get("total_under_money"))
        if over_pt is not None and over_money is not None:
            try:
                point = float(over_pt)
                if abs(point - OFF_BOARD_PRICE) >= 1e-6:
                    rows.append({
                        "sport_key": sport_key, "league": league, "event_id": event_id,
                        "commence_time": commence, "home_team": home_team, "away_team": away_team,
                        "event_name": event_name, "market_type": "totals", "selection": f"Over {point}",
                        "point": point, "odds": over_money,
                    })
                    added += 1
            except (TypeError, ValueError):
                pass
        if under_pt is not None and under_money is not None:
            try:
                point = float(under_pt)
                if abs(point - OFF_BOARD_PRICE) >= 1e-6:
                    rows.append({
                        "sport_key": sport_key, "league": league, "event_id": event_id,
                        "commence_time": commence, "home_team": home_team, "away_team": away_team,
                        "event_name": event_name, "market_type": "totals", "selection": f"Under {point}",
                        "point": point, "odds": under_money,
                    })
                    added += 1
            except (TypeError, ValueError):
                pass
        if added > 0:
            break
    return rows


def _rows_from_event_lines_array(
    ev: dict,
    sport_key: str,
    league: str,
    on_date: date,
) -> list[dict]:
    """Parse event with top-level 'lines' array (V1 / RapidAPI style)."""
    event_id = str(ev.get("event_id") or ev.get("id") or "")
    away_team, home_team = _teams_from_event(ev)
    if not home_team or not away_team:
        return []
    commence = _commence_iso(ev, on_date)
    event_name = f"{away_team} @ {home_team}"
    rows: list[dict] = []
    lines = ev.get("lines") or []
    for line in lines:
        if not isinstance(line, dict):
            continue
        line_type = (line.get("line_type") or line.get("type") or "").lower()
        price = _price_american(line.get("price") or line.get("odds"))
        if price is None:
            continue
        point = line.get("point") or line.get("line") or line.get("value")
        if point is not None:
            try:
                point = float(point)
            except (TypeError, ValueError):
                point = None
        selection = (line.get("team_name") or line.get("selection") or line.get("participant") or "").strip()
        if not selection and "team" in line:
            selection = (line.get("team") or {}).get("name", "") if isinstance(line.get("team"), dict) else str(line.get("team", ""))
        if not selection:
            selection = home_team if (line.get("is_home") or line.get("home")) else away_team
        if not selection:
            continue
        market_type = "h2h"
        if "spread" in line_type or "handicap" in line_type:
            market_type = "spreads"
            if point is None:
                continue
            rows.append({"sport_key": sport_key, "league": league, "event_id": event_id, "commence_time": commence, "home_team": home_team, "away_team": away_team, "event_name": event_name, "market_type": market_type, "selection": selection, "point": point, "odds": price})
        elif "money" in line_type or "ml" in line_type or line_type == "h2h":
            market_type = "h2h"
            rows.append({"sport_key": sport_key, "league": league, "event_id": event_id, "commence_time": commence, "home_team": home_team, "away_team": away_team, "event_name": event_name, "market_type": market_type, "selection": selection, "point": None, "odds": price})
        elif "total" in line_type or "over" in line_type or "under" in line_type:
            market_type = "totals"
            if point is None:
                continue
            sel = f"Over {point}" if "over" in line_type or "over" in str(selection).lower() else f"Under {point}"
            rows.append({"sport_key": sport_key, "league": league, "event_id": event_id, "commence_time": commence, "home_team": home_team, "away_team": away_team, "event_name": event_name, "market_type": market_type, "selection": sel, "point": point, "odds": price})
    return rows


def _event_to_rows(ev: dict, sport_key: str, league: str, on_date: date) -> list[dict]:
    """Dispatch to V2-style, lines-dict (RapidAPI v1), or lines-array parser."""
    if ev.get("markets"):
        return _rows_from_event_v2_style(ev, sport_key, league, on_date)
    if isinstance(ev.get("lines"), dict):
        return _rows_from_event_lines_dict(ev, sport_key, league, on_date)
    return _rows_from_event_lines_array(ev, sport_key, league, on_date)


def get_rundown_live_odds(
    sport_keys: list[str] | None = None,
    commence_on_date: date | None = None,
    api_key: str | None = None,
    host: str | None = None,
) -> pd.DataFrame:
    """
    Fetch live odds from The Rundown (RapidAPI). NCAAB sport_id=6, NBA sport_id=2.
    Returns DataFrame with live_odds_df schema. Uses today when commence_on_date is None.
    """
    if _debug_rundown_odds():
        print("[DEBUG_RUNDOWN] (1) get_rundown_live_odds() is being called.")
    key, h = _load_rundown_secrets()
    api_key = (api_key or key).strip()
    host = (host or h).strip()
    if _debug_rundown_odds():
        print(f"[DEBUG_RUNDOWN] (1) api_key loaded: {'yes' if api_key else 'no'}, host={host[:40]}...")
    if not api_key:
        return pd.DataFrame(
            columns=[
                "sport_key", "league", "event_id", "commence_time", "home_team", "away_team",
                "event_name", "market_type", "selection", "point", "odds",
            ]
        )
    sport_keys = sport_keys or [BASKETBALL_NCAAB, BASKETBALL_NBA]
    tz = ZoneInfo("America/New_York")
    on_date = commence_on_date or datetime.now(tz).date()
    session = requests.Session()
    all_rows: list[dict] = []
    for sport_id, (sport_key, league) in RUNDOWN_SPORT_IDS.items():
        if sport_key not in sport_keys:
            continue
        events = _fetch_rundown_events(sport_id, on_date, api_key, host, session)
        if not events:
            continue
        for ev in events:
            all_rows.extend(_event_to_rows(ev, sport_key, league, on_date))
    if not all_rows:
        return pd.DataFrame(
            columns=[
                "sport_key", "league", "event_id", "commence_time", "home_team", "away_team",
                "event_name", "market_type", "selection", "point", "odds",
            ]
        )
    return pd.DataFrame(all_rows)


def get_rundown_live_odds_with_stats(
    sport_keys: list[str] | None = None,
    commence_on_date: date | None = None,
    api_key: str | None = None,
    host: str | None = None,
) -> tuple[pd.DataFrame, int, int]:
    """Same as get_rundown_live_odds but returns (df, n_games, n_odds_rows)."""
    df = get_rundown_live_odds(
        sport_keys=sport_keys,
        commence_on_date=commence_on_date,
        api_key=api_key,
        host=host,
    )
    n_rows = len(df)
    n_games = df["event_id"].nunique() if not df.empty else 0
    return (df, n_games, n_rows)
