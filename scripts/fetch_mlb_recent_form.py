#!/usr/bin/env python3
"""
Fetch each team's last up-to-7 completed games from the MLB Stats API schedule
and write rolling recent form to data/mlb/recent_form.csv.

Uses one GET per team (30 total) plus one teams list call:
  GET /api/v1/teams?sportId=1&season=2026
  GET /api/v1/schedule?sportId=1&teamId={id}&startDate={start}&endDate={end}

Date window: last 7 calendar days through today (America/New_York) so the
single schedule call matches the spec. Completed games in that window are
sorted by game time (newest first); we keep at most 7 for the aggregates.

Columns: team_id, team_name, recent_win_pct, recent_rs_avg, recent_ra_avg, games_counted

Usage (run once per day before scripts/predict_mlb.py):
  python3 scripts/fetch_mlb_recent_form.py
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

REQUEST_TIMEOUT = 30
REQUEST_SLEEP_S = 0.12
BASE = "https://statsapi.mlb.com/api/v1"
APP_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = APP_ROOT / "data" / "mlb" / "recent_form.csv"

MAX_GAMES = 7


def _get_json(url: str) -> dict:
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _is_final(g: dict) -> bool:
    st = g.get("status") or {}
    return st.get("abstractGameState") == "Final"


def _games_for_team(schedule: dict) -> list[dict]:
    out: list[dict] = []
    for d in schedule.get("dates") or []:
        for g in d.get("games") or []:
            if _is_final(g):
                out.append(g)
    return out


def _sort_key_game(g: dict) -> str:
    return str(g.get("gameDate") or "")


def _line_for_team_in_game(g: dict, team_id: int) -> tuple[int, int, bool] | None:
    """
    Returns (runs_scored, runs_allowed, won) for this team in a final game.
    """
    th = g.get("teams") or {}
    home = th.get("home") or {}
    away = th.get("away") or {}
    ht = (home.get("team") or {}).get("id")
    at = (away.get("team") or {}).get("id")
    hs = home.get("score")
    aw = away.get("score")
    if hs is None or aw is None:
        return None
    try:
        hs_i = int(hs)
        aw_i = int(aw)
    except (TypeError, ValueError):
        return None

    if ht == team_id:
        won = home.get("isWinner") is True
        return hs_i, aw_i, won
    if at == team_id:
        won = away.get("isWinner") is True
        return aw_i, hs_i, won
    return None


def fetch_team_rows(
    season: int,
    days_back: int,
    tz_name: str,
) -> list[dict]:
    teams_url = f"{BASE}/teams?sportId=1&season={season}"
    teams_blob = _get_json(teams_url)
    time.sleep(REQUEST_SLEEP_S)

    tz = ZoneInfo(tz_name)
    end_d = datetime.now(tz).date()
    start_d = end_d - timedelta(days=days_back)

    rows: list[dict] = []
    for t in teams_blob.get("teams") or []:
        tid = int(t["id"])
        name = str(t.get("name") or "").strip()

        sched_url = (
            f"{BASE}/schedule?sportId=1&teamId={tid}"
            f"&startDate={start_d.isoformat()}&endDate={end_d.isoformat()}"
        )
        sched = _get_json(sched_url)
        time.sleep(REQUEST_SLEEP_S)

        games = _games_for_team(sched)
        games.sort(key=_sort_key_game, reverse=True)
        take = games[:MAX_GAMES]

        lines: list[tuple[int, int, bool]] = []
        for g in take:
            line = _line_for_team_in_game(g, tid)
            if line is not None:
                lines.append(line)

        if not lines:
            rows.append(
                {
                    "team_id": tid,
                    "team_name": name,
                    "recent_win_pct": 0.5,
                    "recent_rs_avg": float("nan"),
                    "recent_ra_avg": float("nan"),
                    "games_counted": 0,
                }
            )
            continue

        wins = sum(1 for _rs, _ra, won in lines if won)
        n = len(lines)
        rs_sum = sum(rs for rs, _ra, _ in lines)
        ra_sum = sum(ra for _rs, ra, _ in lines)

        rows.append(
            {
                "team_id": tid,
                "team_name": name,
                "recent_win_pct": wins / n,
                "recent_rs_avg": rs_sum / n,
                "recent_ra_avg": ra_sum / n,
                "games_counted": n,
            }
        )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch MLB recent form (last 7 games in window).")
    parser.add_argument("--season", type=int, default=2026, help="Season for /teams (default 2026)")
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Schedule startDate = today minus this many days (default 7)",
    )
    parser.add_argument(
        "--tz",
        default="America/New_York",
        help="Timezone for 'today' (default America/New_York)",
    )
    args = parser.parse_args()

    rows = fetch_team_rows(season=args.season, days_back=args.days, tz_name=args.tz)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df = df.sort_values("team_id").reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH} ({len(df)} teams).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
