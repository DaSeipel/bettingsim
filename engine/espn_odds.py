"""
ESPN unofficial odds API (no API key).
NCAAB: uses site scoreboard endpoint (full daily slate), then core API for odds when scoreboard has none.
NBA: uses sports.core.api.espn.com v2 events + per-event odds.
Maps to the same DataFrame schema as get_live_odds (live_odds_df) for pipeline compatibility.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

def _debug_espn_odds() -> bool:
    return os.environ.get("DEBUG_ESPN_ODDS", "") == "1"

from .engine import BASKETBALL_NBA, BASKETBALL_NCAAB, SPORT_KEY_TO_LEAGUE

ESPN_BASE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues"
# Site API scoreboard returns full daily slate (used for NCAAB)
ESPN_SITE_SCOREBOARD_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball"
REQUEST_TIMEOUT = 12
REQUEST_HEADERS = {"Accept": "application/json", "User-Agent": "bettingsim/1.0"}

# ESPN league path segment -> (sport_key, league display)
ESPN_LEAGUE_CONFIG = {
    "nba": (BASKETBALL_NBA, "NBA"),
    "mens-college-basketball": (BASKETBALL_NCAAB, "NCAAB"),
}


def _events_url(league_slug: str) -> str:
    return f"{ESPN_BASE}/{league_slug}/events"


def _event_url(league_slug: str, event_id: str) -> str:
    return f"{ESPN_BASE}/{league_slug}/events/{event_id}"


def _odds_url(league_slug: str, event_id: str, competition_id: str) -> str:
    return f"{ESPN_BASE}/{league_slug}/events/{event_id}/competitions/{competition_id}/odds"


def _parse_event_id_from_ref(ref: str) -> str | None:
    """Extract event id from $ref like .../events/401810763?lang=en&region=us"""
    if not ref:
        return None
    m = re.search(r"/events/(\d+)(?:\?|$)", ref)
    return m.group(1) if m else None


def _parse_commence_datetime(commence_time: str) -> datetime | None:
    if not commence_time:
        return None
    try:
        dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        return dt if dt.tzinfo else None
    except (ValueError, TypeError):
        return None


def _event_date_in_tz(commence_time: str, tz: ZoneInfo) -> date | None:
    dt = _parse_commence_datetime(commence_time)
    if dt is None:
        return None
    return dt.astimezone(tz).date()


def _parse_commence_date_utc(commence_time: str) -> date | None:
    dt = _parse_commence_datetime(commence_time)
    if dt is None:
        return None
    return dt.date()


def _is_future(commence_time: str, now_utc: datetime | None = None) -> bool:
    """True if game has not yet started (compare in UTC)."""
    dt = _parse_commence_datetime(commence_time)
    if dt is None:
        return False
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    return dt > now_utc


def _parse_home_away_from_name(name: str) -> tuple[str, str]:
    """ESPN event name is 'Away at Home'. Return (home_team, away_team)."""
    if not name or " at " not in name:
        return ("", "")
    parts = name.split(" at ", 1)
    away = (parts[0] or "").strip()
    home = (parts[1] or "").strip()
    return (home, away)


def _american_from_espn(value: float | int | None) -> int | None:
    """ESPN odds are already American. Ensure int and |value| >= 100 for consistency with pipeline."""
    if value is None:
        return None
    try:
        v = int(round(float(value)))
        if abs(v) < 100:
            return None
        return v
    except (TypeError, ValueError):
        return None


def _fetch_json(url: str, session: requests.Session) -> dict | list | None:
    try:
        r = session.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _scoreboard_url(league_slug: str, on_date: date, limit: int = 200) -> str:
    """Scoreboard endpoint. NCAAB: groups=50 = all conferences for full daily slate (otherwise capped at 15)."""
    dates_param = on_date.strftime("%Y%m%d")
    if league_slug == "mens-college-basketball":
        return f"{ESPN_SITE_SCOREBOARD_BASE}/{league_slug}/scoreboard?groups=50&limit={limit}&dates={dates_param}"
    return f"{ESPN_SITE_SCOREBOARD_BASE}/{league_slug}/scoreboard?dates={dates_param}&limit={limit}"


def _fetch_scoreboard(league_slug: str, session: requests.Session, on_date: date) -> dict | None:
    """Fetch scoreboard JSON for one date. Returns None on error."""
    url = _scoreboard_url(league_slug, on_date)
    return _fetch_json(url, session)


def _parse_competitors_home_away(competitors: list) -> tuple[str, str]:
    """From scoreboard competition.competitors (homeAway + team.displayName), return (home_team, away_team)."""
    home_team = ""
    away_team = ""
    for c in competitors or []:
        if not isinstance(c, dict):
            continue
        ha = (c.get("homeAway") or "").strip().lower()
        team = c.get("team") if isinstance(c.get("team"), dict) else {}
        name = (team.get("displayName") or team.get("name") or "").strip()
        if not name:
            continue
        if ha == "home":
            home_team = name
        elif ha == "away":
            away_team = name
    return (home_team, away_team)


def _odds_items_from_scoreboard_competition(comp: dict) -> list[dict]:
    """Parse competitions[0].odds from scoreboard. ESPN may return odds as list of provider items."""
    odds = comp.get("odds")
    if odds is None:
        return []
    if isinstance(odds, list):
        return [o for o in odds if isinstance(o, dict)]
    if isinstance(odds, dict):
        return [odds]
    return []


def _fetch_events_list(
    league_slug: str,
    session: requests.Session,
    on_date: date,
    limit: int = 200,
) -> list[str]:
    """Return list of event IDs for on_date (YYYYMMDD). Fetches all pages. Caller filters by future commence_time."""
    url = _events_url(league_slug)
    dates_param = on_date.strftime("%Y%m%d")
    event_ids: list[str] = []
    page_index = 1
    first_response_printed = False
    while True:
        full_url = f"{url}?limit={limit}&dates={dates_param}&pageIndex={page_index}"
        if _debug_espn_odds() and page_index == 1:
            print(f"[DEBUG_ESPN_ODDS] (1) Raw URL for NCAAB events: {full_url}")
        data = _fetch_json(full_url, session)
        if _debug_espn_odds() and data is not None and not first_response_printed:
            try:
                raw_str = json.dumps(data, indent=2)
                print(f"[DEBUG_ESPN_ODDS] (2) Full raw response (first API call, events list):\n{raw_str[:12000]}{'...' if len(raw_str) > 12000 else ''}")
                first_response_printed = True
            except Exception:
                print("[DEBUG_ESPN_ODDS] (2) Raw response (could not serialize):", type(data), str(data)[:2000])
        if not data or not isinstance(data, dict):
            break
        items = data.get("items") or []
        for item in items:
            ref = item.get("$ref") if isinstance(item, dict) else None
            if not ref:
                continue
            eid = _parse_event_id_from_ref(ref)
            if eid:
                event_ids.append(eid)
        page_count = data.get("pageCount") or 1
        if page_index >= page_count:
            break
        page_index += 1
    return event_ids


def _fetch_event_detail(league_slug: str, event_id: str, session: requests.Session) -> dict | None:
    data = _fetch_json(_event_url(league_slug, event_id), session)
    if not data or not isinstance(data, dict):
        return None
    return data


def _fetch_odds(league_slug: str, event_id: str, competition_id: str, session: requests.Session) -> list[dict] | None:
    data = _fetch_json(_odds_url(league_slug, event_id, competition_id), session)
    if not data or not isinstance(data, dict):
        return None
    return data.get("items")


def _rows_from_espn_odds(
    event_id: str,
    commence_time: str,
    home_team: str,
    away_team: str,
    sport_key: str,
    league: str,
    odds_items: list[dict],
) -> list[dict]:
    """Convert ESPN odds items to live_odds_df-style rows (market_type, selection, point, odds)."""
    rows: list[dict] = []
    event_name = f"{away_team} @ {home_team}"

    for item in odds_items or []:
        if not isinstance(item, dict):
            continue
        # One provider per item: spread, overUnder, awayTeamOdds, homeTeamOdds
        spread = item.get("spread")  # home spread (e.g. -15.5)
        over_under = item.get("overUnder")
        away_odds = item.get("awayTeamOdds") or {}
        home_odds = item.get("homeTeamOdds") or {}

        # Prefer current, fallback to close
        a_cur = (away_odds.get("current") or away_odds.get("close") or {})
        h_cur = (home_odds.get("current") or home_odds.get("close") or {})

        # Moneyline (h2h)
        ml_away = _american_from_espn(away_odds.get("moneyLine"))
        ml_home = _american_from_espn(home_odds.get("moneyLine"))
        if ml_away is not None:
            rows.append({
                "sport_key": sport_key,
                "league": league,
                "event_id": event_id,
                "commence_time": commence_time,
                "home_team": home_team,
                "away_team": away_team,
                "event_name": event_name,
                "market_type": "h2h",
                "selection": away_team,
                "point": None,
                "odds": ml_away,
            })
        if ml_home is not None:
            rows.append({
                "sport_key": sport_key,
                "league": league,
                "event_id": event_id,
                "commence_time": commence_time,
                "home_team": home_team,
                "away_team": away_team,
                "event_name": event_name,
                "market_type": "h2h",
                "selection": home_team,
                "point": None,
                "odds": ml_home,
            })

        # Spreads: home spread (e.g. -15.5), away +15.5
        if spread is not None:
            try:
                spread_f = float(spread)
            except (TypeError, ValueError):
                spread_f = None
            if spread_f is not None:
                away_spread_odds = _american_from_espn(away_odds.get("spreadOdds"))
                home_spread_odds = _american_from_espn(home_odds.get("spreadOdds"))
                away_point = -spread_f  # away +15.5 when home -15.5
                if away_spread_odds is not None:
                    rows.append({
                        "sport_key": sport_key,
                        "league": league,
                        "event_id": event_id,
                        "commence_time": commence_time,
                        "home_team": home_team,
                        "away_team": away_team,
                        "event_name": event_name,
                        "market_type": "spreads",
                        "selection": away_team,
                        "point": away_point,
                        "odds": away_spread_odds,
                    })
                if home_spread_odds is not None:
                    rows.append({
                        "sport_key": sport_key,
                        "league": league,
                        "event_id": event_id,
                        "commence_time": commence_time,
                        "home_team": home_team,
                        "away_team": away_team,
                        "event_name": event_name,
                        "market_type": "spreads",
                        "selection": home_team,
                        "point": spread_f,
                        "odds": home_spread_odds,
                    })

        # Totals
        if over_under is not None:
            try:
                ou_f = float(over_under)
            except (TypeError, ValueError):
                ou_f = None
            if ou_f is not None:
                over_odds = _american_from_espn(item.get("overOdds"))
                under_odds = _american_from_espn(item.get("underOdds"))
                if over_odds is not None:
                    rows.append({
                        "sport_key": sport_key,
                        "league": league,
                        "event_id": event_id,
                        "commence_time": commence_time,
                        "home_team": home_team,
                        "away_team": away_team,
                        "event_name": event_name,
                        "market_type": "totals",
                        "selection": f"Over {ou_f}",
                        "point": ou_f,
                        "odds": over_odds,
                    })
                if under_odds is not None:
                    rows.append({
                        "sport_key": sport_key,
                        "league": league,
                        "event_id": event_id,
                        "commence_time": commence_time,
                        "home_team": home_team,
                        "away_team": away_team,
                        "event_name": event_name,
                        "market_type": "totals",
                        "selection": f"Under {ou_f}",
                        "point": ou_f,
                        "odds": under_odds,
                    })
    return rows


def _ncaab_odds_from_scoreboard(
    session: requests.Session,
    sport_key: str,
    league: str,
    today: date,
    now_utc: datetime,
    tz: ZoneInfo,
    use_tz_for_today: bool,
    include_all_today: bool = False,
) -> list[dict]:
    """
    Fetch NCAAB events and odds using the site scoreboard endpoint (full daily slate).
    Parses home, away, commence time, and when present competitions[0].odds (spread, moneyline, total).
    When scoreboard has no odds for an event, fetches odds from core API for that event.
    If include_all_today=False, filters to today (ET) and future games only.
    If include_all_today=True, includes all events for today (no future filter) for use as primary source.
    """
    league_slug = "mens-college-basketball"
    if _debug_espn_odds():
        now_et = now_utc.astimezone(tz)
        print(f"[DEBUG_ESPN_ODDS] Current system time UTC: {now_utc.isoformat()}")
        print(f"[DEBUG_ESPN_ODDS] Current system time ET:  {now_et.isoformat()}")
        data0 = _fetch_scoreboard(league_slug, session, today)
        events0 = (data0 or {}).get("events") or []
        print(f"[DEBUG_ESPN_ODDS] First 5 scoreboard events (before any filtering), total today={len(events0)}:")
        for i, ev in enumerate(events0[:5]):
            commence = ev.get("date") or "(none)"
            name = ev.get("name") or "(no name)"
            print(f"[DEBUG_ESPN_ODDS]   {i+1}. commence_time={commence}  |  {name}")
    all_rows: list[dict] = []
    seen_events: set[str] = set()
    for day_offset in (0, 1):
        fetch_date = today + timedelta(days=day_offset) if use_tz_for_today else today
        data = _fetch_scoreboard(league_slug, session, fetch_date)
        if not data or not isinstance(data, dict):
            continue
        events = data.get("events") or []
        for ev in events:
            event_id = str(ev.get("id") or "").strip()
            if not event_id or event_id in seen_events:
                continue
            seen_events.add(event_id)
            commence_time = ev.get("date") or ""
            if not commence_time:
                continue
            event_date_et = _event_date_in_tz(commence_time, tz) if use_tz_for_today else _parse_commence_date_utc(commence_time)
            if event_date_et is None or event_date_et != today:
                continue
            if not include_all_today and not _is_future(commence_time, now_utc=now_utc):
                continue
            comps = ev.get("competitions") or []
            if not comps:
                continue
            comp = comps[0] if isinstance(comps[0], dict) else {}
            home_team, away_team = _parse_competitors_home_away(comp.get("competitors") or [])
            if not home_team or not away_team:
                name = ev.get("name") or ""
                home_team, away_team = _parse_home_away_from_name(name)
            if not home_team or not away_team:
                continue
            odds_items = _odds_items_from_scoreboard_competition(comp)
            if not odds_items:
                comp_id = str(comp.get("id") or event_id)
                odds_items = _fetch_odds(league_slug, event_id, comp_id, session) or []
            rows = _rows_from_espn_odds(
                event_id, commence_time, home_team, away_team, sport_key, league, odds_items
            )
            all_rows.extend(rows)
    return all_rows


def get_espn_live_odds(
    sport_keys: list[str] | None = None,
    display_timezone: str = "America/New_York",
    commence_on_date: date | None = None,
    ncaab_include_all_today: bool = False,
) -> pd.DataFrame:
    """
    Fetch live odds from ESPN's free API for NBA and/or NCAAB.
    NCAAB uses the site scoreboard endpoint (full daily slate); NBA uses core events endpoint.
    Returns a DataFrame with the same columns as get_live_odds: sport_key, league, event_id,
    commence_time, home_team, away_team, event_name, market_type, selection, point, odds (American).
    Only includes events for today (ET) and commence_time in the future, unless ncaab_include_all_today=True.
    """
    sport_keys = sport_keys or [BASKETBALL_NBA, BASKETBALL_NCAAB]
    tz = ZoneInfo(display_timezone)
    use_tz_for_today = commence_on_date is None
    today = datetime.now(tz).date() if use_tz_for_today else commence_on_date

    empty = pd.DataFrame(
        columns=[
            "sport_key", "league", "event_id", "commence_time", "home_team", "away_team",
            "event_name", "market_type", "selection", "point", "odds",
        ]
    )

    league_slugs = []
    for sk in sport_keys:
        if sk == BASKETBALL_NBA:
            league_slugs.append(("nba", sk, SPORT_KEY_TO_LEAGUE.get(sk, "NBA")))
        elif sk == BASKETBALL_NCAAB:
            league_slugs.append(("mens-college-basketball", sk, SPORT_KEY_TO_LEAGUE.get(sk, "NCAAB")))
        else:
            continue

    all_rows: list[dict] = []
    session = requests.Session()
    now_utc = datetime.now(timezone.utc)

    for league_slug, sport_key, league in league_slugs:
        if league_slug == "mens-college-basketball":
            rows = _ncaab_odds_from_scoreboard(
                session, sport_key, league, today, now_utc, tz, use_tz_for_today,
                include_all_today=ncaab_include_all_today,
            )
            all_rows.extend(rows)
            continue
        # NBA: use core API events list + per-event detail + odds
        event_ids_seen: set[str] = set()
        for day_offset in (0, 1):
            fetch_date = today + timedelta(days=day_offset) if use_tz_for_today else today
            for eid in _fetch_events_list(league_slug, session, fetch_date):
                if eid not in event_ids_seen:
                    event_ids_seen.add(eid)
        event_ids = list(event_ids_seen)
        for event_id in event_ids:
            ev = _fetch_event_detail(league_slug, event_id, session)
            if not ev:
                continue
            commence_time = ev.get("date") or ""
            if not commence_time:
                continue
            event_date_et = _event_date_in_tz(commence_time, tz) if use_tz_for_today else _parse_commence_date_utc(commence_time)
            if event_date_et is None or event_date_et != today:
                continue
            if not _is_future(commence_time, now_utc=now_utc):
                continue
            name = ev.get("name") or ""
            home_team, away_team = _parse_home_away_from_name(name)
            if not home_team or not away_team:
                continue
            comps = ev.get("competitions") or []
            comp_id = None
            for c in comps:
                if isinstance(c, dict) and c.get("id"):
                    comp_id = str(c.get("id"))
                    break
                if isinstance(c, dict) and c.get("$ref"):
                    m = re.search(r"/competitions/(\d+)(?:\?|$)", c.get("$ref", ""))
                    if m:
                        comp_id = m.group(1)
                        break
            if not comp_id:
                comp_id = event_id
            odds_items = _fetch_odds(league_slug, event_id, comp_id, session)
            rows = _rows_from_espn_odds(
                event_id, commence_time, home_team, away_team, sport_key, league, odds_items or []
            )
            all_rows.extend(rows)

    if not all_rows:
        return empty
    return pd.DataFrame(all_rows)


def get_espn_live_odds_with_stats(
    sport_keys: list[str] | None = None,
    display_timezone: str = "America/New_York",
    commence_on_date: date | None = None,
) -> tuple[pd.DataFrame, int, int]:
    """
    Same as get_espn_live_odds but returns (df, n_games, n_odds_rows).
    n_games = number of distinct events; n_odds_rows = len(df).
    """
    df = get_espn_live_odds(
        sport_keys=sport_keys,
        display_timezone=display_timezone,
        commence_on_date=commence_on_date,
    )
    n_rows = len(df)
    n_games = df["event_id"].nunique() if not df.empty else 0
    return (df, n_games, n_rows)
