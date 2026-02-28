"""
Betting simulation engine.
Runs a strategy over a historical dataset and tracks bankroll, wins/losses, ROI.
Also provides get_daily_fixtures (football-data.co.uk CSV) and Scraper for odds.
"""

from __future__ import annotations

import io
import re
from datetime import date, datetime, timezone, timedelta
import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import Callable, Any

from strategies.strategies import american_to_decimal, decimal_to_american

# Football-Data.co.uk fixtures CSV (no API key required)
FIXTURES_CSV_URL = "https://www.football-data.co.uk/fixtures.csv"

# The Odds API (https://api.the-odds-api.com) — sport keys
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
BASKETBALL_NBA = "basketball_nba"
BASKETBALL_NCAAB = "basketball_ncaab"
BASKETBALL_SPORT_KEYS = [BASKETBALL_NBA, BASKETBALL_NCAAB]

# Display names for sport_key (for Best Value Plays table)
SPORT_KEY_TO_LEAGUE: dict[str, str] = {
    BASKETBALL_NBA: "NBA",
    BASKETBALL_NCAAB: "NCAAB",
}

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/csv,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def get_daily_fixtures(source_url: str | None = None) -> pd.DataFrame:
    """
    Load upcoming fixtures (and odds) from football-data.co.uk CSV.
    No API key required. Data is updated Fridays (weekend) and Tuesdays (midweek).

    Uses source_url if provided, otherwise the default fixtures.csv URL.
    Returns a DataFrame with columns: date, time, league, home_team, away_team,
    event_name, odds_home, odds_draw, odds_away (from first available bookmaker).
    """
    url = source_url or FIXTURES_CSV_URL
    try:
        # Prefer pandas direct read; fallback to requests if blocked or error
        try:
            df = pd.read_csv(url, encoding="utf-8", on_bad_lines="skip")
        except Exception:
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text), encoding="utf-8", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame(
            columns=["date", "time", "league", "home_team", "away_team", "event_name", "odds_home", "odds_draw", "odds_away"]
        )

    if df.empty:
        return df

    # Normalize column names (football-data uses mixed case)
    df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
    # Prefer Bet365 1X2; else first of PSH/PH, MaxH/AvgH (avoid "Series or ..." — truth value ambiguous)
    def _first_col(df: pd.DataFrame, names: list[str]) -> pd.Series | None:
        for n in names:
            if n in df.columns:
                return df[n]
        return None
    odds_home = _first_col(df, ["B365H", "PSH", "PH", "MaxH", "AvgH"])
    odds_draw = _first_col(df, ["B365D", "PSD", "PD", "MaxD", "AvgD"])
    odds_away = _first_col(df, ["B365A", "PSA", "PA", "MaxA", "AvgA"])

    out = pd.DataFrame({
        "date": df.get("Date", pd.Series(dtype=object)),
        "time": df.get("Time", pd.Series(dtype=object)),
        "league": df.get("Div", pd.Series(dtype=object)),
        "home_team": df.get("HomeTeam", pd.Series(dtype=object)),
        "away_team": df.get("AwayTeam", pd.Series(dtype=object)),
    })
    out["event_name"] = out["home_team"].astype(str) + " vs " + out["away_team"].astype(str)
    # Football-data.co.uk uses decimal odds; convert to American for project standard
    def _dec_to_am(s: pd.Series | None) -> pd.Series | None:
        if s is None:
            return None
        num = pd.to_numeric(s, errors="coerce")
        return num.apply(lambda x: decimal_to_american(x) if pd.notna(x) and x > 1 else None)
    out["odds_home"] = _dec_to_am(odds_home)
    out["odds_draw"] = _dec_to_am(odds_draw)
    out["odds_away"] = _dec_to_am(odds_away)
    return out.dropna(how="all", subset=["home_team", "away_team"]).reset_index(drop=True)


def get_basketball_odds(
    api_key: str,
    sports: list[str] | None = None,
    regions: str = "us",
    markets: list[str] | None = None,
    odds_format: str = "american",
) -> pd.DataFrame:
    """
    Fetch NBA and NCAA basketball odds from The Odds API (v4).
    Uses sport keys basketball_nba and basketball_ncaab.
    Returns spreads and totals (Over/Under) as well as moneyline (h2h).
    Odds are returned in American format (+150, -110).

    Requires an API key from https://the-odds-api.com (free tier available).
    Returns a DataFrame with columns: sport_key, league, event_id, commence_time,
    home_team, away_team, event_name, market_type (h2h|spreads|totals),
    selection, point (spread or total line; NaN for h2h), odds (American).
    """
    if not (api_key or "").strip():
        return pd.DataFrame(
            columns=[
                "sport_key", "league", "event_id", "commence_time", "home_team", "away_team",
                "event_name", "market_type", "selection", "point", "odds",
            ]
        )
    sport_keys = sports if sports is not None else BASKETBALL_SPORT_KEYS.copy()
    markets = markets if markets is not None else ["h2h", "spreads", "totals"]
    markets_param = ",".join(markets)
    rows: list[dict[str, Any]] = []
    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)

    for sport_key in sport_keys:
        url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
        params = {
            "regions": regions,
            "markets": markets_param,
            "oddsFormat": odds_format,
            "apiKey": api_key.strip(),
        }
        try:
            resp = session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        league = "NBA" if sport_key == BASKETBALL_NBA else "NCAAB"
        for event in data:
            if not isinstance(event, dict):
                continue
            event_id = event.get("id", "")
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            commence = event.get("commence_time", "")
            event_name = f"{away} @ {home}"
            bookmakers = event.get("bookmakers") or []
            seen_outcomes: set[tuple[str, str, str | float | None]] = set()
            for bm in bookmakers:
                for mkt in bm.get("markets") or []:
                    mkt_key = mkt.get("key", "")
                    if mkt_key not in ("h2h", "spreads", "totals"):
                        continue
                    market_type = "h2h" if mkt_key == "h2h" else mkt_key
                    for out in mkt.get("outcomes") or []:
                        name = out.get("name", "")
                        price = out.get("price")
                        if price is None:
                            continue
                        try:
                            p = float(price)
                        except (TypeError, ValueError):
                            continue
                        # Store American: API returns American when oddsFormat=american; else convert decimal to American
                        if isinstance(price, (int, float)) and (price >= 100 or price <= -100):
                            odds_american = int(p)
                        else:
                            odds_american = int(round(decimal_to_american(p)))
                        if abs(odds_american) < 100:
                            continue
                        point = out.get("point")
                        if point is not None:
                            try:
                                point = float(point)
                            except (TypeError, ValueError):
                                point = None
                        selection = name
                        if market_type == "totals" and point is not None:
                            selection = f"{name} {point}"
                        key = (event_id, market_type, selection, point)
                        if key in seen_outcomes:
                            continue
                        seen_outcomes.add(key)
                        rows.append({
                            "sport_key": sport_key,
                            "league": league,
                            "event_id": event_id,
                            "commence_time": commence,
                            "home_team": home,
                            "away_team": away,
                            "event_name": event_name,
                            "market_type": market_type,
                            "selection": selection,
                            "point": point,
                            "odds": odds_american,
                        })

    if not rows:
        return pd.DataFrame(
            columns=[
                "sport_key", "league", "event_id", "commence_time", "home_team", "away_team",
                "event_name", "market_type", "selection", "point", "odds",
            ]
        )
    return pd.DataFrame(rows)


def _parse_commence_date(commence_time: str) -> date | None:
    """Parse ISO 8601 commence_time to a date (UTC). Returns None if invalid."""
    if not commence_time:
        return None
    try:
        # API returns e.g. 2026-02-28T20:30:00Z
        dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        return dt.date() if dt.tzinfo else None
    except (ValueError, TypeError):
        return None


def get_live_odds(
    api_key: str,
    sport_keys: list[str],
    regions: str = "us",
    markets: list[str] | None = None,
    odds_format: str = "american",
    commence_on_date: date | None = None,
) -> pd.DataFrame:
    """
    Fetch live odds from The Odds API (v4) for multiple sports.
    Uses requests to GET https://api.the-odds-api.com/v4/sports/{sport_key}/odds for each key.
    If commence_on_date is set, only events whose commence_time falls on that date (UTC) are included.
    Returns a single DataFrame with columns: sport_key, league, event_id, commence_time,
    home_team, away_team, event_name, market_type, selection, point, odds (American).
    League name is derived from SPORT_KEY_TO_LEAGUE or sport_key.
    """
    if not (api_key or "").strip() or not sport_keys:
        return pd.DataFrame(
            columns=[
                "sport_key", "league", "event_id", "commence_time", "home_team", "away_team",
                "event_name", "market_type", "selection", "point", "odds",
            ]
        )
    today = commence_on_date if commence_on_date is not None else date.today()
    markets = markets if markets is not None else ["h2h", "spreads", "totals"]
    markets_param = ",".join(markets)
    rows: list[dict[str, Any]] = []
    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)

    for sport_key in sport_keys:
        url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
        params = {
            "regions": regions,
            "markets": markets_param,
            "oddsFormat": odds_format,
            "apiKey": api_key.strip(),
        }
        try:
            resp = session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        league = SPORT_KEY_TO_LEAGUE.get(sport_key, sport_key.replace("_", " ").title())
        for event in data:
            if not isinstance(event, dict):
                continue
            event_id = event.get("id", "")
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            commence = event.get("commence_time", "")
            event_date = _parse_commence_date(commence)
            if event_date is None or event_date != today:
                continue
            event_name = f"{away} @ {home}"
            bookmakers = event.get("bookmakers") or []
            seen_outcomes: set[tuple[str, str, str | float | None]] = set()
            for bm in bookmakers:
                for mkt in bm.get("markets") or []:
                    mkt_key = mkt.get("key", "")
                    if mkt_key not in ("h2h", "spreads", "totals"):
                        continue
                    market_type = "h2h" if mkt_key == "h2h" else mkt_key
                    for out in mkt.get("outcomes") or []:
                        name = out.get("name", "")
                        price = out.get("price")
                        if price is None:
                            continue
                        try:
                            p = float(price)
                        except (TypeError, ValueError):
                            continue
                        if isinstance(price, (int, float)) and (price >= 100 or price <= -100):
                            odds_american = int(p)
                        else:
                            odds_american = int(round(decimal_to_american(p)))
                        if abs(odds_american) < 100:
                            continue
                        point = out.get("point")
                        if point is not None:
                            try:
                                point = float(point)
                            except (TypeError, ValueError):
                                point = None
                        selection = name
                        if market_type == "totals" and point is not None:
                            selection = f"{name} {point}"
                        key = (event_id, market_type, selection, point)
                        if key in seen_outcomes:
                            continue
                        seen_outcomes.add(key)
                        rows.append({
                            "sport_key": sport_key,
                            "league": league,
                            "event_id": event_id,
                            "commence_time": commence,
                            "home_team": home,
                            "away_team": away,
                            "event_name": event_name,
                            "market_type": market_type,
                            "selection": selection,
                            "point": point,
                            "odds": odds_american,
                        })

    if not rows:
        return pd.DataFrame(
            columns=[
                "sport_key", "league", "event_id", "commence_time", "home_team", "away_team",
                "event_name", "market_type", "selection", "point", "odds",
            ]
        )
    return pd.DataFrame(rows)


# NBA pace-adjusted totals: league averages when per-team data is unavailable
NBA_LEAGUE_AVG_PACE = 100.0
NBA_LEAGUE_AVG_OFF_RATING = 115.0

# Back-to-back fatigue: reduce Pace by 1.5%, Off Rating by 2%
B2B_PACE_MULTIPLIER = 0.985
B2B_OFF_RATING_MULTIPLIER = 0.98


def get_nba_teams_back_to_back(api_key: str, as_of_date: date) -> set[str]:
    """
    Return set of NBA team names that played the night before as_of_date (B2B fatigue).
    Fetches yesterday's events from The Odds API; returns empty set if unavailable.
    """
    if not (api_key or "").strip():
        return set()
    yesterday = as_of_date - timedelta(days=1)
    df = get_live_odds(
        api_key=api_key.strip(),
        sport_keys=[BASKETBALL_NBA],
        commence_on_date=yesterday,
    )
    teams: set[str] = set()
    for _, row in df.iterrows():
        h = row.get("home_team")
        a = row.get("away_team")
        if h and str(h).strip():
            teams.add(str(h).strip())
        if a and str(a).strip():
            teams.add(str(a).strip())
    return teams


def get_nba_team_pace_stats() -> dict[str, dict[str, float]]:
    """
    Fetch or provide NBA team pace stats (possessions/48m, offensive rating).
    Returns dict: team_name -> {"pace": float, "off_rating": float}.
    When live data is unavailable, returns empty dict; callers use NBA_LEAGUE_AVG_*.
    """
    # TODO: fetch from NBA stats API when available
    return {}


class Scraper:
    """
    Scrape today's odds from a public odds comparison (or similar) HTML page
    when no API key is available. Uses requests + BeautifulSoup.
    Configure base_url and optionally table/row selectors for your target site.
    """

    def __init__(
        self,
        base_url: str | None = None,
        table_selector: str = "table",
        row_selector: str = "tr",
        timeout: int = 15,
    ) -> None:
        self.base_url = base_url or ""
        self.table_selector = table_selector
        self.row_selector = row_selector
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(REQUEST_HEADERS)

    def _fetch(self, url: str) -> str:
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.text

    def _parse_odds_cell(self, cell: str) -> float | None:
        """Try to extract a decimal odds value from a cell string."""
        if not cell or not isinstance(cell, str):
            return None
        # Match decimal odds (e.g. 2.50, 1.95, 11/4 as fraction later if needed)
        m = re.search(r"\d+\.\d{2}", cell.strip())
        if m:
            return float(m.group())
        m = re.search(r"\d+\.\d+", cell.strip())
        if m:
            return float(m.group())
        return None

    def scrape_odds(self, url: str | None = None) -> pd.DataFrame:
        """
        Fetch the given URL (or base_url), parse HTML with BeautifulSoup,
        find tables and extract rows into a DataFrame with columns:
        event_name, odds_home, odds_draw, odds_away (where available).
        Sites that load odds via JavaScript may need a different approach;
        adjust table_selector / row logic for your target page.
        """
        target = (url or self.base_url).strip()
        if not target:
            return pd.DataFrame(columns=["event_name", "odds_home", "odds_draw", "odds_away"])

        html = self._fetch(target)
        soup = BeautifulSoup(html, "html.parser")
        tables = soup.select(self.table_selector)
        rows_data: list[dict[str, Any]] = []

        for table in tables:
            rows = table.select(self.row_selector)
            for tr in rows:
                cells = [td.get_text(separator=" ", strip=True) for td in tr.find_all(["td", "th"])]
                if len(cells) < 2:
                    continue
                # Try to interpret: first columns often match name, then odds
                odds_values = [self._parse_odds_cell(c) for c in cells]
                odds_nums = [x for x in odds_values if x is not None and 1.0 < x < 50.0]
                if len(odds_nums) >= 2:
                    # Heuristic: assume order H, D, A if we have 3; convert decimal to American
                    def _to_american(d: float) -> int:
                        return int(round(decimal_to_american(d)))
                    event_name = " ".join(c for c in cells[:3] if c and not re.match(r"^\d+\.\d+", c)) or "Unknown"
                    if len(odds_nums) >= 3:
                        rows_data.append({
                            "event_name": event_name[:80],
                            "odds_home": _to_american(odds_nums[0]),
                            "odds_draw": _to_american(odds_nums[1]),
                            "odds_away": _to_american(odds_nums[2]),
                        })
                    else:
                        rows_data.append({
                            "event_name": event_name[:80],
                            "odds_home": _to_american(odds_nums[0]),
                            "odds_draw": None,
                            "odds_away": _to_american(odds_nums[1]) if len(odds_nums) > 1 else None,
                        })

        if not rows_data:
            return pd.DataFrame(columns=["event_name", "odds_home", "odds_draw", "odds_away"])
        return pd.DataFrame(rows_data)

    def get_today_odds(self, url: str | None = None) -> pd.DataFrame:
        """Alias for scrape_odds; fetches and returns today's odds from the given or default URL."""
        return self.scrape_odds(url=url)


# Row dict passed to strategy: at least odds, model_prob, result; optionally event_id, event_name.
StrategyFn = Callable[[pd.Series, float], float]


class BettingEngine:
    """
    Simulates a series of bets over a historical dataset using a given strategy.
    Tracks bankroll, wins, losses, and ROI.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        strategy: StrategyFn,
        starting_bankroll: float,
    ) -> None:
        """
        Args:
            df: Historical dataset with columns [odds, model_prob, result].
                result: 1 = win, 0 = loss.
                Optional: event_id, event_name for display.
            strategy: Function (row: pd.Series, bankroll: float) -> stake (float). 0 = no bet.
            starting_bankroll: Initial bankroll.
        """
        self.df = df.reset_index(drop=True)
        self.strategy = strategy
        self.starting_bankroll = starting_bankroll
        self._bet_history: list[dict[str, Any]] = []
        self._bankroll_curve: list[float] = []

    def run(self) -> dict[str, Any]:
        """
        Run the simulation over all rows.
        Returns summary and bet history for display.
        """
        self._bet_history = []
        self._bankroll_curve = []
        bankroll = self.starting_bankroll
        self._bankroll_curve.append(bankroll)

        wins = 0
        losses = 0
        total_staked = 0.0
        total_profit = 0.0

        for step, (idx, row) in enumerate(self.df.iterrows()):
            stake = self.strategy(row, bankroll)
            if stake <= 0 or stake > bankroll:
                self._bankroll_curve.append(bankroll)
                continue

            odds_american = float(row["odds"])
            result = int(row["result"])
            total_staked += stake
            odds_decimal = american_to_decimal(odds_american)

            if result == 1:
                profit = stake * (odds_decimal - 1.0)
                wins += 1
                result_str = "Won"
            else:
                profit = -stake
                losses += 1
                result_str = "Lost"

            total_profit += profit
            bankroll += profit
            self._bankroll_curve.append(bankroll)

            record = {
                "step": step + 1,
                "event_id": row.get("event_id", idx),
                "event_name": row.get("event_name", f"Event {idx + 1}"),
                "odds": odds_american,
                "model_prob": row.get("model_prob", 1.0 / odds_decimal),
                "stake": round(stake, 2),
                "result": result_str,
                "profit": round(profit, 2),
                "bankroll_after": round(bankroll, 2),
            }
            self._bet_history.append(record)

        roi = (total_profit / total_staked * 100.0) if total_staked > 0 else 0.0
        return {
            "bet_history": self._bet_history,
            "bankroll_curve": self._bankroll_curve,
            "final_bankroll": round(bankroll, 2),
            "wins": wins,
            "losses": losses,
            "total_bets": wins + losses,
            "total_staked": round(total_staked, 2),
            "total_profit": round(total_profit, 2),
            "roi_pct": round(roi, 2),
        }
