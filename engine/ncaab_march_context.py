"""
NCAAB late-season and conference tournament context features for March betting.
Flags: conference tournament game, seeds, days since last tournament game,
clinched NCAA bid (reduced motivation), bubble team (increased desperation).

CSV data (maintain from free bracket/seed sources): data/ncaab_seeds.csv (team, seed),
data/ncaab_clinched_bid.csv, data/ncaab_bubble_teams.csv.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# Conference tournament season: early March through Selection Sunday (typically 2nd Sunday of March)
MARCH_CONTEXT_START_MONTH = 3
MARCH_CONTEXT_START_DAY = 1
# Selection Sunday is the Sunday when the bracket is revealed (e.g. 2025 = March 16 = 3rd Sunday)
def _selection_sunday(year: int) -> date:
    """Approximate Selection Sunday: 3rd Sunday of March."""
    d = date(year, 3, 1)
    days_to_first_sun = (6 - d.weekday()) % 7
    first_sun_day = 1 + days_to_first_sun
    third_sun_day = first_sun_day + 14
    return date(year, 3, third_sun_day)


def is_after_selection_sunday(d: date) -> bool:
    """True if d is on or after Selection Sunday (bracket revealed); used to auto-enable March Madness mode."""
    return d >= _selection_sunday(d.year)

# Event name patterns that suggest conference tournament
CONF_TOURNEY_PATTERNS = [
    "tournament",
    "championship",
    "champ ",
    " semifinal",
    "semifinals",
    " final",
    "quarterfinal",
    "quarterfinals",
    "first round",
    "second round",
    "conference tournament",
    "conf. tournament",
    "conf tourney",
    "automatic bid",
]


def _game_date_from_commence(commence_time: Any) -> Optional[date]:
    """Parse commence_time to date (UTC or naive)."""
    if commence_time is None or pd.isna(commence_time):
        return None
    s = str(commence_time).strip()
    if not s:
        return None
    try:
        s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt.date() if hasattr(dt, "date") else date(dt.year, dt.month, dt.day)
    except Exception:
        return None


def is_conference_tournament_season(game_date: Optional[date]) -> bool:
    """True if game_date falls in conference tournament / March context window."""
    if game_date is None:
        return False
    if game_date.month != MARCH_CONTEXT_START_MONTH:
        return False
    sel_sun = _selection_sunday(game_date.year)
    return MARCH_CONTEXT_START_DAY <= game_date.day and game_date <= sel_sun


def is_conference_tournament_game(game_date: Optional[date], event_name: str) -> bool:
    """
    Heuristic: game is in March context window and event name suggests conference tournament.
    """
    if not is_conference_tournament_season(game_date):
        return False
    name = (event_name or "").strip().lower()
    return any(p in name for p in CONF_TOURNEY_PATTERNS)


def get_ncaab_march_context(
    event_name: str,
    home_team: str,
    away_team: str,
    commence_time: Any,
    *,
    seeds_csv_path: Optional[Path] = None,
    clinched_bid_csv_path: Optional[Path] = None,
    bubble_teams_csv_path: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Return a dict of NCAAB March context flags for one game.

    - is_conference_tournament: bool (from date + event name heuristic)
    - home_seed, away_seed: int or None (from seeds_csv or None; 1–16 typical)
    - home_days_since_tournament_game, away_days_since_tournament_game: int or None
      (placeholder None without tournament game log)
    - home_clinched_ncaa_bid, away_clinched_ncaa_bid: bool (from CSV or False)
    - home_is_bubble_team, away_is_bubble_team: bool (from CSV or False)

    Optional CSVs (team name in first column, then value column):
    - seeds_csv_path: team,seed
    - clinched_bid_csv_path: team (one per line or team,1)
    - bubble_teams_csv_path: team (one per line or team,1)
    """
    game_date = _game_date_from_commence(commence_time)
    conf_tourney = is_conference_tournament_game(game_date, event_name or "")

    home = (home_team or "").strip()
    away = (away_team or "").strip()

    def _seed(team: str) -> Optional[int]:
        if not team or not seeds_csv_path or not seeds_csv_path.exists():
            return None
        try:
            df = pd.read_csv(seeds_csv_path)
            col = "team" if "team" in df.columns else df.columns[0]
            seed_col = "seed" if "seed" in df.columns else (df.columns[1] if len(df.columns) > 1 else None)
            if seed_col is None:
                return None
            row = df[df[col].astype(str).str.strip().str.lower() == team.lower()]
            if row.empty:
                return None
            v = row[seed_col].iloc[0]
            return int(v) if pd.notna(v) else None
        except Exception:
            return None

    def _in_list(team: str, p: Optional[Path], col_name: str = "team") -> bool:
        if not team or not p or not p.exists():
            return False
        try:
            df = pd.read_csv(p)
            col = col_name if col_name in df.columns else df.columns[0]
            return bool((df[col].astype(str).str.strip().str.lower() == team.lower()).any())
        except Exception:
            return False

    data_dir = Path(__file__).resolve().parent.parent / "data"
    seeds_path = seeds_csv_path or data_dir / "ncaab_seeds.csv"
    clinched_path = clinched_bid_csv_path or data_dir / "ncaab_clinched_bid.csv"
    bubble_path = bubble_teams_csv_path or data_dir / "ncaab_bubble_teams.csv"

    return {
        "is_conference_tournament": conf_tourney,
        "home_seed": _seed(home),
        "away_seed": _seed(away),
        "home_days_since_tournament_game": None,  # placeholder: needs tournament game log
        "away_days_since_tournament_game": None,
        "home_clinched_ncaa_bid": _in_list(home, clinched_path),
        "away_clinched_ncaa_bid": _in_list(away, clinched_path),
        "home_is_bubble_team": _in_list(home, bubble_path),
        "away_is_bubble_team": _in_list(away, bubble_path),
    }


def add_ncaab_march_context_to_df(
    df: pd.DataFrame,
    event_col: str = "Event",
    home_col: str = "home_team",
    away_col: str = "away_team",
    commence_col: str = "commence_time",
    league_col: str = "League",
) -> pd.DataFrame:
    """
    Add March context columns to rows where league_col == "NCAAB".
    New columns: is_conference_tournament, home_seed, away_seed,
    home_days_since_tournament_game, away_days_since_tournament_game,
    home_clinched_ncaa_bid, away_clinched_ncaa_bid, home_is_bubble_team, away_is_bubble_team.
    """
    if df.empty or league_col not in df.columns:
        return df
    ncaab = df[df[league_col].astype(str).str.strip().str.upper() == "NCAAB"]
    if ncaab.empty:
        return df
    out = df.copy()
    for col in [
        "is_conference_tournament", "home_seed", "away_seed",
        "home_days_since_tournament_game", "away_days_since_tournament_game",
        "home_clinched_ncaa_bid", "away_clinched_ncaa_bid",
        "home_is_bubble_team", "away_is_bubble_team",
    ]:
        if col not in out.columns:
            out[col] = None
    event_c = out[event_col] if event_col in out.columns else (out["event_name"] if "event_name" in out.columns else None)
    home_c = out[home_col] if home_col in out.columns else None
    away_c = out[away_col] if away_col in out.columns else None
    commence_c = out[commence_col] if commence_col in out.columns else None
    if home_c is None or away_c is None:
        return out
    for idx in ncaab.index:
        ctx = get_ncaab_march_context(
            event_c.iloc[idx] if event_c is not None else "",
            home_c.iloc[idx],
            away_c.iloc[idx],
            commence_c.iloc[idx] if commence_c is not None else None,
        )
        for k, v in ctx.items():
            out.at[idx, k] = v
    return out


def merge_ncaab_march_seeds_into_feature_matrix(
    df: pd.DataFrame,
    league_col: str = "league",
    home_col: str = "home_team_name",
    away_col: str = "away_team_name",
    date_col: str = "game_date",
) -> pd.DataFrame:
    """
    Add home_seed, away_seed (and other March context) to NCAAB rows in games_with_team_stats-style DataFrame.
    For rows where league is ncaab and game_date is in March, look up seeds from ncaab_seeds.csv.
    Other columns (is_conference_tournament, clinched, bubble) are also set when applicable.
    """
    if df.empty or league_col not in df.columns or home_col not in df.columns or away_col not in df.columns:
        return df
    ncaab_mask = df[league_col].astype(str).str.strip().str.lower() == "ncaab"
    if not ncaab_mask.any():
        return df
    out = df.copy()
    for col in [
        "home_seed", "away_seed", "is_conference_tournament",
        "home_clinched_ncaa_bid", "away_clinched_ncaa_bid",
        "home_is_bubble_team", "away_is_bubble_team",
    ]:
        if col not in out.columns:
            out[col] = None
    date_ser = out[date_col] if date_col in out.columns else None
    for idx in out.index:
        if not ncaab_mask.loc[idx]:
            continue
        try:
            gd = date_ser.loc[idx] if date_ser is not None else None
            if gd is not None and pd.notna(gd):
                gd_str = str(gd)[:10]
                if len(gd_str) >= 10 and gd_str[5:7] != "03":
                    continue
                commence = f"{gd_str}T12:00:00" if gd_str else None
            else:
                commence = None
            ctx = get_ncaab_march_context(
                "",
                out.at[idx, home_col],
                out.at[idx, away_col],
                commence,
            )
            for k, v in ctx.items():
                if k in out.columns:
                    out.at[idx, k] = v
        except Exception:
            continue
    return out
