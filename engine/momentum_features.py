"""
Momentum-based features for NBA and NCAAB: streaks, ATS/O/U records, point-diff trend, home/road ATS.
All computed with no lookahead (only games before each game_date). Used for spread predictions.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from .utils import parse_date

# League default totals when we don't have ratings/pace to estimate (for O/U record)
NBA_DEFAULT_TOTAL = 220.0
NCAAB_DEFAULT_TOTAL = 140.0
# Scale offensive rating diff to approximate point spread: (home_off - away_off) * (pace/400)
RATING_TO_SPREAD_SCALE = 1.0 / 400.0


def _estimated_home_spread(row: pd.Series, league: str) -> float:
    """Estimated home spread (negative = home favored) from ratings/pace if present."""
    ho = row.get("home_offensive_rating")
    ao = row.get("away_offensive_rating")
    hp = row.get("home_pace")
    ap = row.get("away_pace")
    if pd.notna(ho) and pd.notna(ao):
        pace = 100.0
        if pd.notna(hp) and pd.notna(ap):
            pace = (float(hp) + float(ap)) / 2.0
        return (float(ho) - float(ao)) * pace * RATING_TO_SPREAD_SCALE
    return 0.0


def _estimated_total(row: pd.Series, league: str) -> float:
    """Estimated total points for O/U from ratings/pace if present."""
    ho = row.get("home_offensive_rating")
    ao = row.get("away_offensive_rating")
    hp = row.get("home_pace")
    ap = row.get("away_pace")
    if pd.notna(ho) and pd.notna(ao):
        pace = 100.0
        if pd.notna(hp) and pd.notna(ap):
            pace = (float(hp) + float(ap)) / 2.0
        return (float(ho) + float(ao)) / 100.0 * pace
    return NBA_DEFAULT_TOTAL if str(league).strip().lower() == "nba" else NCAAB_DEFAULT_TOTAL


def _team_margin_and_covered(team: str, home: str, away: str, home_pts: float, away_pts: float, home_spread: float) -> tuple[float, bool]:
    """Margin from team POV (positive = team won by that much). Covered = team beat the spread."""
    if team == home:
        margin = home_pts - away_pts
        spread_team = home_spread  # home spread
    else:
        margin = away_pts - home_pts
        spread_team = -home_spread  # away spread
    covered = (margin + spread_team) > 0
    return margin, covered


def _streak(team: str, game_date: pd.Timestamp, games_before: pd.DataFrame) -> int:
    """Current streak: positive = wins, negative = losses. Uses only games before game_date."""
    if games_before.empty:
        return 0
    team_games = games_before[
        (games_before["home_team_name"].astype(str).str.strip() == team)
        | (games_before["away_team_name"].astype(str).str.strip() == team)
    ].copy()
    team_games["_dt"] = team_games["game_date"].apply(parse_date)
    team_games = team_games.dropna(subset=["_dt"]).sort_values("_dt", ascending=False)
    if team_games.empty:
        return 0
    sign = None
    count = 0
    for _, r in team_games.iterrows():
        h = str(r["home_team_name"]).strip()
        a = str(r["away_team_name"]).strip()
        sh = float(r.get("home_score") or 0)
        sa = float(r.get("away_score") or 0)
        won = (h == team and sh > sa) or (a == team and sa > sh)
        s = 1 if won else -1
        if sign is None:
            sign = s
            count = 1
        elif s == sign:
            count += 1
        else:
            break
    return count * sign if sign else 0


def _ats_ou_last_n(
    team: str,
    game_date: pd.Timestamp,
    games_before: pd.DataFrame,
    n: int,
    league: str,
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """(ats_wins, ats_games, over_wins, over_games) over last n games. None if no games."""
    if games_before.empty:
        return (None, None, None, None)
    team_games = games_before[
        (games_before["home_team_name"].astype(str).str.strip() == team)
        | (games_before["away_team_name"].astype(str).str.strip() == team)
    ].copy()
    team_games["_dt"] = team_games["game_date"].apply(parse_date)
    team_games = team_games.dropna(subset=["_dt"]).sort_values("_dt", ascending=False).head(n)
    if team_games.empty:
        return (None, None, None, None)
    ats_w, over_w = 0, 0
    for _, r in team_games.iterrows():
        home_spread = _estimated_home_spread(r, league)
        total_line = _estimated_total(r, league)
        h = str(r["home_team_name"]).strip()
        a = str(r["away_team_name"]).strip()
        sh = float(r.get("home_score") or 0)
        sa = float(r.get("away_score") or 0)
        _, covered = _team_margin_and_covered(team, h, a, sh, sa, home_spread)
        if covered:
            ats_w += 1
        total = sh + sa
        if total > total_line:
            over_w += 1
    return (ats_w, len(team_games), over_w, len(team_games))


def _point_diff_trend_last5(team: str, game_date: pd.Timestamp, games_before: pd.DataFrame) -> Optional[int]:
    """Point differential trend: 1 = improving, -1 = declining, 0 = flat. Last 5 games margins, compare first 3 avg to last 2 avg."""
    if games_before.empty:
        return None
    team_games = games_before[
        (games_before["home_team_name"].astype(str).str.strip() == team)
        | (games_before["away_team_name"].astype(str).str.strip() == team)
    ].copy()
    team_games["_dt"] = team_games["game_date"].apply(parse_date)
    team_games = team_games.dropna(subset=["_dt"]).sort_values("_dt", ascending=False).head(5)
    if len(team_games) < 3:
        return None
    margins = []
    for _, r in team_games.iterrows():
        h = str(r["home_team_name"]).strip()
        a = str(r["away_team_name"]).strip()
        sh = float(r.get("home_score") or 0)
        sa = float(r.get("away_score") or 0)
        margin, _ = _team_margin_and_covered(team, h, a, sh, sa, 0.0)
        margins.append(margin)
    # margins are ordered most recent first (game 0 = most recent)
    if len(margins) < 3:
        return None
    recent_avg = sum(margins[:2]) / 2.0 if len(margins) >= 2 else margins[0]
    older_avg = sum(margins[2:5]) / min(3, len(margins) - 2)
    if recent_avg > older_avg + 0.5:
        return 1
    if recent_avg < older_avg - 0.5:
        return -1
    return 0


def _home_road_ats_season(
    team: str,
    game_date: pd.Timestamp,
    games_before: pd.DataFrame,
    league: str,
) -> tuple[Optional[float], Optional[float]]:
    """(ats_pct_at_home, ats_pct_on_road) for season so far. None if no games in that split."""
    if games_before.empty:
        return (None, None)
    team_games = games_before[
        (games_before["home_team_name"].astype(str).str.strip() == team)
        | (games_before["away_team_name"].astype(str).str.strip() == team)
    ].copy()
    team_games["_dt"] = team_games["game_date"].apply(parse_date)
    team_games = team_games.dropna(subset=["_dt"])
    if team_games.empty:
        return (None, None)
    home_ats_w, home_ats_n = 0, 0
    road_ats_w, road_ats_n = 0, 0
    for _, r in team_games.iterrows():
        home_spread = _estimated_home_spread(r, league)
        h = str(r["home_team_name"]).strip()
        a = str(r["away_team_name"]).strip()
        sh = float(r.get("home_score") or 0)
        sa = float(r.get("away_score") or 0)
        _, covered = _team_margin_and_covered(team, h, a, sh, sa, home_spread)
        if h == team:
            home_ats_n += 1
            if covered:
                home_ats_w += 1
        else:
            road_ats_n += 1
            if covered:
                road_ats_w += 1
    home_pct = (home_ats_w / home_ats_n) if home_ats_n else None
    road_pct = (road_ats_w / road_ats_n) if road_ats_n else None
    return (home_pct, road_pct)


def build_momentum_features(
    games_df: pd.DataFrame,
    league: str,
) -> pd.DataFrame:
    """
    Build momentum features for each game. No lookahead: only games with game_date < row's game_date.

    Returns DataFrame with columns: league, game_id, plus for home/away:
    - home_streak, away_streak (signed: + wins, - losses)
    - home_ats_pct_last10, away_ats_pct_last10 (0-1, None if < 1 game)
    - home_over_pct_last10, away_over_pct_last10 (0-1)
    - home_pt_diff_trend_last5, away_pt_diff_trend_last5 (1 improving, -1 declining, 0 flat)
    - home_ats_pct_home_season, home_ats_pct_road_season, away_ats_pct_home_season, away_ats_pct_road_season (0-1)
    """
    if games_df.empty or "game_date" not in games_df.columns:
        return pd.DataFrame()
    required = ["game_id", "game_date", "home_team_name", "away_team_name"]
    if not all(c in games_df.columns for c in required):
        return pd.DataFrame()
    g = games_df.sort_values("game_date").reset_index(drop=True)
    league_str = str(league).strip().lower()
    league_val = str(g.iloc[0].get("league", league)).strip() if not g.empty else league_str
    rows = []
    for i, row in g.iterrows():
        game_id = row["game_id"]
        game_date = parse_date(row["game_date"])
        if game_date is None:
            continue
        home = str(row["home_team_name"]).strip()
        away = str(row["away_team_name"]).strip()
        before = g[g["game_date"].apply(lambda x: parse_date(x) is not None and parse_date(x) < game_date)]

        home_streak = _streak(home, game_date, before)
        away_streak = _streak(away, game_date, before)

        home_ats_w, home_ats_n, home_over_w, home_over_n = _ats_ou_last_n(home, game_date, before, 10, league_str)
        away_ats_w, away_ats_n, away_over_w, away_over_n = _ats_ou_last_n(away, game_date, before, 10, league_str)

        home_ats_pct = (home_ats_w / home_ats_n) if home_ats_n and home_ats_n > 0 else None
        away_ats_pct = (away_ats_w / away_ats_n) if away_ats_n and away_ats_n > 0 else None
        home_over_pct = (home_over_w / home_over_n) if home_over_n and home_over_n > 0 else None
        away_over_pct = (away_over_w / away_over_n) if away_over_n and away_over_n > 0 else None

        home_trend = _point_diff_trend_last5(home, game_date, before)
        away_trend = _point_diff_trend_last5(away, game_date, before)

        home_ats_home, home_ats_road = _home_road_ats_season(home, game_date, before, league_str)
        away_ats_home, away_ats_road = _home_road_ats_season(away, game_date, before, league_str)

        rows.append({
            "league": league_val,
            "game_id": game_id,
            "home_streak": home_streak,
            "away_streak": away_streak,
            "home_ats_pct_last10": home_ats_pct,
            "away_ats_pct_last10": away_ats_pct,
            "home_over_pct_last10": home_over_pct,
            "away_over_pct_last10": away_over_pct,
            "home_pt_diff_trend_last5": home_trend,
            "away_pt_diff_trend_last5": away_trend,
            "home_ats_pct_home_season": home_ats_home,
            "home_ats_pct_road_season": home_ats_road,
            "away_ats_pct_home_season": away_ats_home,
            "away_ats_pct_road_season": away_ats_road,
        })
    return pd.DataFrame(rows)


def merge_momentum_into_feature_matrix(
    games_df: pd.DataFrame,
    league_col: str = "league",
) -> pd.DataFrame:
    """
    Add momentum features to a games DataFrame. Builds per league and merges on (league, game_id).
    """
    if games_df.empty or "game_id" not in games_df.columns or league_col not in games_df.columns:
        return games_df
    out = games_df.copy()
    momentum_cols = [
        "home_streak", "away_streak",
        "home_ats_pct_last10", "away_ats_pct_last10",
        "home_over_pct_last10", "away_over_pct_last10",
        "home_pt_diff_trend_last5", "away_pt_diff_trend_last5",
        "home_ats_pct_home_season", "home_ats_pct_road_season",
        "away_ats_pct_home_season", "away_ats_pct_road_season",
    ]
    out = out.drop(columns=[c for c in momentum_cols if c in out.columns], errors="ignore")
    leagues = out[league_col].astype(str).str.strip().str.lower().unique().tolist()
    all_mom = []
    for league in leagues:
        sub = out[out[league_col].astype(str).str.strip().str.lower() == league]
        if sub.empty:
            continue
        mom = build_momentum_features(sub, league=league)
        if not mom.empty:
            all_mom.append(mom)
    if not all_mom:
        return out
    combined = pd.concat(all_mom, ignore_index=True)
    out = out.merge(combined, on=[league_col, "game_id"], how="left")
    return out
