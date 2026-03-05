"""
Auto-result job: resolve play_history results from ESPN scoreboard.
Runs at 8am ET daily. For each unresolved play from the previous day, fetches final score
from ESPN and sets W/L/P. If ESPN has no result, sets manual_review_flag for dashboard badge.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .espn_collector import fetch_scoreboard
from .play_history import (
    load_play_history,
    set_manual_review_flag,
    update_play_result,
    _default_db_path,
)

# Sport display name (from odds/play_history) -> ESPN LEAGUES key
SPORT_TO_LEAGUE: dict[str, str] = {
    "nba": "nba",
    "nfl": "nfl",
    "mlb": "mlb",
    "nhl": "nhl",
    "ncaab": "ncaab",  # may not be in ESPN LEAGUES; handle below
}


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _team_match(ph_team: str, espn_team: str) -> bool:
    a, b = _norm(ph_team), _norm(espn_team)
    if not a or not b:
        return False
    return a == b or a in b or b in a


def _match_game(play: dict[str, Any], games: list[dict[str, Any]]) -> dict[str, Any] | None:
    ph_home = (play.get("home_team") or "").strip()
    ph_away = (play.get("away_team") or "").strip()
    for g in games:
        espn_home = (g.get("home_team_name") or "").strip()
        espn_away = (g.get("away_team_name") or "").strip()
        if _team_match(ph_home, espn_home) and _team_match(ph_away, espn_away):
            return g
    return None


def _resolve_moneyline(play: dict[str, Any], game: dict[str, Any]) -> str | None:
    side = (play.get("recommended_side") or "").strip()
    home_score = game.get("home_score")
    away_score = game.get("away_score")
    if home_score is None or away_score is None:
        return None
    home_name = (game.get("home_team_name") or "").strip()
    away_name = (game.get("away_team_name") or "").strip()
    we_bet_home = _team_match(side, home_name)
    we_bet_away = _team_match(side, away_name)
    if we_bet_home:
        if home_score > away_score:
            return "W"
        if home_score < away_score:
            return "L"
        return "P"
    if we_bet_away:
        if away_score > home_score:
            return "W"
        if away_score < home_score:
            return "L"
        return "P"
    return None


def _resolve_spread(play: dict[str, Any], game: dict[str, Any]) -> str | None:
    line = play.get("spread_or_total")
    if line is None or (isinstance(line, float) and pd.isna(line)):
        return None
    try:
        line = float(line)
    except (TypeError, ValueError):
        return None
    home_score = game.get("home_score")
    away_score = game.get("away_score")
    if home_score is None or away_score is None:
        return None
    side = (play.get("recommended_side") or "").strip()
    home_name = (game.get("home_team_name") or "").strip()
    away_name = (game.get("away_team_name") or "").strip()
    if _team_match(side, home_name):
        our_score, opp_score = home_score, away_score
    elif _team_match(side, away_name):
        our_score, opp_score = away_score, home_score
    else:
        return None
    margin_plus_line = (our_score - opp_score) + line
    if margin_plus_line > 0:
        return "W"
    if margin_plus_line < 0:
        return "L"
    return "P"


def _resolve_total(play: dict[str, Any], game: dict[str, Any]) -> str | None:
    line = play.get("spread_or_total")
    if line is None or (isinstance(line, float) and pd.isna(line)):
        return None
    try:
        line = float(line)
    except (TypeError, ValueError):
        return None
    home_score = game.get("home_score")
    away_score = game.get("away_score")
    if home_score is None or away_score is None:
        return None
    total = home_score + away_score
    side = (play.get("recommended_side") or "").strip().lower()
    if "over" in side:
        if total > line:
            return "W"
        if total < line:
            return "L"
        return "P"
    if "under" in side:
        if total < line:
            return "W"
        if total > line:
            return "L"
        return "P"
    return None


def _resolve_play(play: dict[str, Any], game: dict[str, Any]) -> str | None:
    bet_type = (play.get("bet_type") or "").strip().lower()
    if "moneyline" in bet_type or "winner" in bet_type:
        return _resolve_moneyline(play, game)
    if "spread" in bet_type:
        return _resolve_spread(play, game)
    if "over" in bet_type or "under" in bet_type or "total" in bet_type:
        return _resolve_total(play, game)
    return None


def run_auto_result(as_of_date: date | None = None, db_path: Path | None = None) -> dict[str, int]:
    """
    Resolve yesterday's unresolved plays using ESPN scoreboard. Updates result and
    actual_payout when a final score is found; sets manual_review_flag when not.
    Call from APScheduler at 8am ET.
    Returns dict with keys: resolved, flagged, skipped_league, error.
    """
    path = db_path or _default_db_path()
    if not path.exists():
        return {"resolved": 0, "flagged": 0, "skipped_league": 0, "error": 0}
    as_of_date = as_of_date or date.today()
    yesterday = as_of_date - timedelta(days=1)
    df = load_play_history(from_date=yesterday, to_date=yesterday, db_path=path)
    if df.empty:
        return {"resolved": 0, "flagged": 0, "skipped_league": 0, "error": 0}
    unresolved = df[df["result"].isna()].copy()
    if unresolved.empty:
        return {"resolved": 0, "flagged": 0, "skipped_league": 0, "error": 0}

    from .espn_collector import LEAGUES

    resolved = 0
    flagged = 0
    skipped_league = 0
    error = 0

    for league_key in set(_norm(str(s)) for s in unresolved["sport"].unique()):
        if league_key not in LEAGUES:
            # #region agent log
            try:
                n_skipped = len(unresolved[unresolved["sport"].astype(str).str.strip().str.lower() == league_key])
                with open(Path(__file__).resolve().parent.parent / ".cursor" / "debug-a60dbe.log", "a") as f:
                    f.write(json.dumps({"sessionId": "a60dbe", "location": "auto_result_job.py:run_auto_result", "message": "league skipped", "data": {"league_key": league_key, "n_skipped": n_skipped}, "hypothesisId": "C", "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000)}, default=str) + "\n")
            except Exception:
                pass
            # #endregion
            skipped_league += len(unresolved[unresolved["sport"].astype(str).str.strip().str.lower() == league_key])
            continue
        games = fetch_scoreboard(league_key, yesterday)
        plays_in_league = unresolved[unresolved["sport"].astype(str).str.strip().str.lower() == league_key]
        for _, row in plays_in_league.iterrows():
            play_id = int(row.get("play_id", 0))
            play = row.to_dict()
            game = _match_game(play, games)
            if game is None:
                set_manual_review_flag(play_id, True, path)
                flagged += 1
                continue
            result = _resolve_play(play, game)
            if result is None:
                set_manual_review_flag(play_id, True, path)
                flagged += 1
                continue
            if update_play_result(play_id, result, path):
                resolved += 1
                set_manual_review_flag(play_id, False, path)
            else:
                error += 1

    return {"resolved": resolved, "flagged": flagged, "skipped_league": skipped_league, "error": error}
