"""CFB point-in-time team ratings and margin projections (Phase 1)."""

from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from engine.cfb_config import ESPN_DB_PATH

FBS_DIVISION = "fbs"
PRIOR_DECAY = 0.35
BLEND_K = 4
OPP_ADJ_PASSES = 4
RIDGE_LAMBDA = 0.25
EXCLUDED_TRAINING_SEASONS = {2020}


def assert_no_sp_eos_inputs() -> None:
    """cfb_ratings_sp end_of_season is never queried by this module."""
    forbidden = "cfb_ratings_sp"
    for name, obj in globals().items():
        if isinstance(obj, str) and forbidden in obj and "end_of_season" in obj:
            raise AssertionError("Forbidden SP+ end_of_season reference detected")
    return None


def _parse_ratio(stat: str | None) -> tuple[float, float]:
    if not stat or "-" not in stat:
        return 0.0, 0.0
    left, right = stat.split("-", 1)
    try:
        return float(left), float(right)
    except ValueError:
        return 0.0, 0.0


def _stat_map(stats_json: str) -> dict[str, str]:
    return {item["category"]: item["stat"] for item in json.loads(stats_json)}


def _plays_from_stats(stats: dict[str, str]) -> float:
    rush_att, _ = _parse_ratio(stats.get("rushingAttempts"))
    _, pass_att = _parse_ratio(stats.get("completionAttempts"))
    return max(rush_att + pass_att, 1.0)


@dataclass
class GameEfficiency:
    game_id: int
    season: int
    week: int
    team: str
    opponent: str
    team_is_fbs: bool
    opp_is_fbs: bool
    off_score: float
    def_score: float


def _game_efficiency_advanced(
    off_ppa: float | None,
    off_sr: float | None,
    off_expl: float | None,
    off_line: float | None,
    def_ppa: float | None,
    def_sr: float | None,
    def_expl: float | None,
    def_line: float | None,
) -> tuple[float, float]:
    """Per-game efficiency from CFBD game/advanced (2021+)."""
    off = (
        0.35 * (off_ppa or 0.0)
        + 0.25 * (off_sr or 0.0)
        + 0.25 * (off_expl or 0.0)
        + 0.15 * ((off_line or 0.0) / 10.0)
    )
    def_score = (
        0.35 * (def_ppa or 0.0)
        + 0.25 * (def_sr or 0.0)
        + 0.25 * (def_expl or 0.0)
        + 0.15 * ((def_line or 0.0) / 10.0)
    )
    return off, def_score


def _game_efficiency(
    points: int,
    stats: dict[str, str],
    opp_points: int,
    opp_stats: dict[str, str],
) -> tuple[float, float]:
    plays = _plays_from_stats(stats)
    opp_plays = _plays_from_stats(opp_stats)
    total_yards = float(stats.get("totalYards") or 0)
    opp_yards = float(opp_stats.get("totalYards") or 0)
    third_made, third_att = _parse_ratio(stats.get("thirdDownEff"))
    third_rate = third_made / max(third_att, 1.0)
    ypr = float(stats.get("yardsPerRushAttempt") or 0)
    ypp_pass = float(stats.get("yardsPerPass") or 0)
    explosiveness = 0.5 * (ypr + ypp_pass)
    turnovers = float(stats.get("turnovers") or 0)
    havoc_created = (
        float(opp_stats.get("sacks") or 0)
        + float(opp_stats.get("tacklesForLoss") or 0)
        + float(opp_stats.get("interceptions") or 0)
    ) / max(opp_plays, 1.0)
    off = (
        0.35 * (points / plays)
        + 0.25 * (total_yards / plays)
        + 0.15 * third_rate
        + 0.15 * explosiveness
        - 0.10 * (turnovers / plays)
    )
    def_score = (
        0.35 * (opp_points / opp_plays)
        + 0.25 * (opp_yards / opp_plays)
        + 0.15 * _parse_ratio(opp_stats.get("thirdDownEff"))[0]
        / max(_parse_ratio(opp_stats.get("thirdDownEff"))[1], 1.0)
        + 0.15
        * (
            float(opp_stats.get("yardsPerRushAttempt") or 0)
            + float(opp_stats.get("yardsPerPass") or 0)
        )
        * 0.5
        - 0.10 * havoc_created
    )
    return off, def_score


def _load_game_efficiencies(conn: sqlite3.Connection) -> list[GameEfficiency]:
    adv_by_game: dict[int, list] = defaultdict(list)
    for row in conn.execute(
        """
        SELECT ga.game_id, ga.team, ga.is_home,
               ga.off_ppa, ga.off_success_rate, ga.off_explosiveness, ga.off_line_yards,
               ga.def_ppa, ga.def_success_rate, ga.def_explosiveness, ga.def_line_yards,
               g.season, g.week, g.home_division, g.away_division
        FROM cfb_game_stats_adv ga
        JOIN cfb_games g ON g.game_id = ga.game_id
        WHERE g.home_points IS NOT NULL
        """
    ).fetchall():
        adv_by_game[row[0]].append(row)

    basic_by_game: dict[int, list] = defaultdict(list)
    for row in conn.execute(
        """
        SELECT gs.game_id, gs.team, gs.is_home, gs.points, gs.stats_json,
               g.season, g.week, g.home_team, g.away_team,
               g.home_division, g.away_division
        FROM cfb_game_stats gs
        JOIN cfb_games g ON g.game_id = gs.game_id
        WHERE g.home_points IS NOT NULL
        """
    ).fetchall():
        basic_by_game[row[0]].append(row)

    out: list[GameEfficiency] = []
    for game_id in set(adv_by_game) | set(basic_by_game):
        if game_id in adv_by_game and len(adv_by_game[game_id]) == 2:
            t0, t1 = adv_by_game[game_id]
            home_div = (t0[13] or "").lower()
            away_div = (t0[14] or "").lower()
            for team_row, opp_row in ((t0, t1), (t1, t0)):
                team_div = home_div if team_row[2] else away_div
                opp_div = home_div if opp_row[2] else away_div
                off, def_score = _game_efficiency_advanced(
                    team_row[3], team_row[4], team_row[5], team_row[6],
                    team_row[7], team_row[8], team_row[9], team_row[10],
                )
                out.append(
                    GameEfficiency(
                        game_id=game_id,
                        season=team_row[11],
                        week=int(team_row[12] or 0),
                        team=team_row[1],
                        opponent=opp_row[1],
                        team_is_fbs=team_div == FBS_DIVISION,
                        opp_is_fbs=opp_div == FBS_DIVISION,
                        off_score=off,
                        def_score=def_score,
                    )
                )
            continue

        teams = basic_by_game.get(game_id, [])
        if len(teams) != 2:
            continue
        home_row = next((t for t in teams if t[2] == 1), None)
        away_row = next((t for t in teams if t[2] == 0), None)
        if not home_row or not away_row:
            continue
        home_div = (home_row[9] or "").lower()
        away_div = (away_row[10] or "").lower()
        for team_row, opp_row, team_div, opp_div in (
            (home_row, away_row, home_div, away_div),
            (away_row, home_row, away_div, home_div),
        ):
            off, def_score = _game_efficiency(
                int(team_row[3] or 0),
                _stat_map(team_row[4]),
                int(opp_row[3] or 0),
                _stat_map(opp_row[4]),
            )
            out.append(
                GameEfficiency(
                    game_id=game_id,
                    season=team_row[5],
                    week=int(team_row[6] or 0),
                    team=team_row[1],
                    opponent=opp_row[1],
                    team_is_fbs=team_div == FBS_DIVISION,
                    opp_is_fbs=opp_div == FBS_DIVISION,
                    off_score=off,
                    def_score=def_score,
                )
            )
    return out


def _decayed_prior(final_rating: float) -> float:
    return (1.0 - PRIOR_DECAY) * final_rating


def _iterative_team_ratings(
    games: list[GameEfficiency],
    priors: dict[str, float],
) -> dict[str, tuple[float, float, float, int, float]]:
    by_team: dict[str, list[GameEfficiency]] = defaultdict(list)
    for g in games:
        by_team[g.team].append(g)

    off_ratings = {t: priors.get(t, 0.0) for t in by_team}
    def_ratings = {t: priors.get(t, 0.0) for t in by_team}

    for _ in range(OPP_ADJ_PASSES):
        new_off: dict[str, float] = {}
        new_def: dict[str, float] = {}
        for team, tg in by_team.items():
            adj_off: list[float] = []
            adj_def: list[float] = []
            for g in tg:
                if g.opp_is_fbs:
                    adj_off.append(g.off_score - def_ratings.get(g.opponent, 0.0))
                    adj_def.append(g.def_score - off_ratings.get(g.opponent, 0.0))
                else:
                    adj_off.append(g.off_score)
                    adj_def.append(g.def_score)
            prior = priors.get(team, 0.0)
            n = len(tg)
            raw_off = float(np.mean(adj_off)) if adj_off else prior
            raw_def = float(np.mean(adj_def)) if adj_def else prior
            reg_off = (n * raw_off + RIDGE_LAMBDA * prior) / (n + RIDGE_LAMBDA)
            reg_def = (n * raw_def + RIDGE_LAMBDA * prior) / (n + RIDGE_LAMBDA)
            new_off[team] = reg_off
            new_def[team] = reg_def
        off_ratings = new_off
        def_ratings = new_def

    result: dict[str, tuple[float, float, float, int, float]] = {}
    for team, tg in by_team.items():
        n = len(tg)
        w_prior = BLEND_K / (BLEND_K + n)
        prior = priors.get(team, 0.0)
        off = w_prior * prior + (1.0 - w_prior) * off_ratings[team]
        def_r = w_prior * prior + (1.0 - w_prior) * def_ratings[team]
        result[team] = (off - def_r, off, def_r, n, w_prior)
    return result


def build_pit_ratings(conn: sqlite3.Connection) -> int:
    assert_no_sp_eos_inputs()
    all_eff = _load_game_efficiencies(conn)
    by_sw: dict[tuple[int, int], list[GameEfficiency]] = defaultdict(list)
    for g in all_eff:
        by_sw[(g.season, g.week)].append(g)

    seasons = sorted({g.season for g in all_eff})
    season_final: dict[int, dict[str, float]] = {}
    rows: list[tuple] = []

    for season in seasons:
        prior_map = {
            t: _decayed_prior(r) for t, r in season_final.get(season - 1, {}).items()
        }
        weeks = sorted({w for (s, w) in by_sw if s == season})
        if not weeks:
            continue
        max_week = max(weeks)
        for week in range(min(weeks), max_week + 2):
            through = week - 1
            hist = [g for w in weeks if w <= through for g in by_sw[(season, w)]]
            ratings = _iterative_team_ratings(hist, prior_map)
            for team, (rating, off_r, def_r, games_used, w_prior) in ratings.items():
                if not any(g.team_is_fbs for g in hist if g.team == team):
                    continue
                rows.append(
                    (season, week, team, rating, off_r, def_r, games_used, w_prior, through)
                )
        full_season = [g for w in weeks for g in by_sw[(season, w)]]
        season_final[season] = {
            t: v[0] for t, v in _iterative_team_ratings(full_season, prior_map).items()
        }

    conn.execute("DELETE FROM cfb_team_ratings_pit")
    conn.executemany(
        """
        INSERT INTO cfb_team_ratings_pit (
            season, week, team, rating, off_rating, def_rating,
            games_used, prior_weight, computed_through_week
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00").replace("+00:00", ""))
    except ValueError:
        return None


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


@dataclass
class ModelConstants:
    hfa: float
    rest_coef: float
    travel_coef: float


@dataclass
class MarginCalibration:
    """Points-scale margin calibration fit on training seasons only."""

    intercept: float
    rating_coef: float
    hfa: float
    rest_coef: float
    travel_coef: float
    train_seasons: tuple[int, ...]
    r2: float
    n: int
    se_intercept: float
    se_rating_coef: float
    se_hfa: float
    se_rest_coef: float
    se_travel_coef: float


@dataclass
class GameProjectionFeatures:
    game_id: int
    season: int
    week: int
    actual_margin: int | None
    neutral_site: bool
    home_rating: float
    away_rating: float
    home_rest: float
    away_rest: float
    away_travel_miles: float


def _load_game_features(conn: sqlite3.Connection, seasons: tuple[int, ...] | None, min_week: int) -> list[GameProjectionFeatures]:
    season_filter = ""
    params: list = [min_week]
    if seasons:
        season_filter = f"AND g.season IN ({','.join('?' * len(seasons))})"
        params = list(seasons) + params

    rows = conn.execute(
        f"""
        SELECT g.game_id, g.season, g.week, g.home_points, g.away_points,
               g.neutral_site, g.start_date, g.home_team, g.away_team,
               rh.rating, ra.rating, v.latitude, v.longitude
        FROM cfb_games g
        JOIN cfb_team_ratings_pit rh
          ON rh.season = g.season AND rh.week = g.week AND rh.team = g.home_team
        JOIN cfb_team_ratings_pit ra
          ON ra.season = g.season AND ra.week = g.week AND ra.team = g.away_team
        LEFT JOIN cfb_venues v ON v.venue_id = g.venue_id
        WHERE lower(coalesce(g.home_division,'')) = 'fbs'
          AND lower(coalesce(g.away_division,'')) = 'fbs'
          AND g.home_points IS NOT NULL
          {season_filter}
          AND g.week >= ?
        ORDER BY g.season, g.start_date, g.game_id
        """,
        params,
    ).fetchall()

    last_date: dict[tuple[int, str], datetime] = {}
    last_venue: dict[tuple[int, str], tuple[float, float]] = {}
    out: list[GameProjectionFeatures] = []

    for row in rows:
        (
            gid,
            season,
            week,
            hp,
            ap,
            neutral,
            start_date,
            home,
            away,
            hr,
            ar,
            lat,
            lon,
        ) = row
        dt = _parse_date(start_date)
        home_rest = away_rest = 7.0
        away_travel = 0.0
        if dt:
            hk, ak = (season, home), (season, away)
            if hk in last_date:
                home_rest = float((dt - last_date[hk]).days)
            if ak in last_date:
                away_rest = float((dt - last_date[ak]).days)
            last_date[hk] = dt
            last_date[ak] = dt
        if lat is not None and lon is not None:
            ak = (season, away)
            if ak in last_venue:
                away_travel = _haversine_miles(last_venue[ak][0], last_venue[ak][1], lat, lon)
            last_venue[ak] = (lat, lon)
            last_venue[(season, home)] = (lat, lon)

        out.append(
            GameProjectionFeatures(
                game_id=gid,
                season=season,
                week=week,
                actual_margin=hp - ap,
                neutral_site=bool(neutral),
                home_rating=float(hr or 0),
                away_rating=float(ar or 0),
                home_rest=home_rest,
                away_rest=away_rest,
                away_travel_miles=away_travel,
            )
        )
    return out


def fit_margin_calibration(
    conn: sqlite3.Connection,
    train_seasons: tuple[int, ...],
    *,
    min_week: int = 4,
) -> MarginCalibration:
    """Fit points-scale calibration on training seasons only."""
    games = _load_game_features(conn, train_seasons, min_week)
    if len(games) < 50:
        raise ValueError(f"Insufficient training games: {len(games)}")

    y = np.array([g.actual_margin for g in games], float)
    rating_diff = np.array([g.home_rating - g.away_rating for g in games])
    hfa_term = np.array([0.0 if g.neutral_site else 1.0 for g in games])
    rest_term = np.array([g.home_rest - g.away_rest for g in games])
    travel_term = np.array([g.away_travel_miles for g in games])

    X = np.column_stack([np.ones(len(games)), rating_diff, hfa_term, rest_term, travel_term])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(games) - X.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())

    return MarginCalibration(
        intercept=float(beta[0]),
        rating_coef=float(beta[1]),
        hfa=float(beta[2]),
        rest_coef=float(beta[3]),
        travel_coef=float(beta[4]),
        train_seasons=train_seasons,
        r2=1 - ss_res / ss_tot if ss_tot else 0.0,
        n=len(games),
        se_intercept=float(se[0]),
        se_rating_coef=float(se[1]),
        se_hfa=float(se[2]),
        se_rest_coef=float(se[3]),
        se_travel_coef=float(se[4]),
    )


def projected_margin_pts_from_cal(
    cal: MarginCalibration,
    home_rating: float,
    away_rating: float,
    neutral_site: bool,
    home_rest: float = 7.0,
    away_rest: float = 7.0,
    away_travel_miles: float = 0.0,
) -> float:
    hfa_term = 0.0 if neutral_site else cal.hfa
    rest_adj = cal.rest_coef * (home_rest - away_rest)
    travel_adj = -cal.travel_coef * away_travel_miles
    return (
        cal.intercept
        + cal.rating_coef * (home_rating - away_rating)
        + hfa_term
        + rest_adj
        + travel_adj
    )


def projected_margin_pts_from_features(
    cal: MarginCalibration,
    feat: GameProjectionFeatures,
) -> float:
    return projected_margin_pts_from_cal(
        cal,
        feat.home_rating,
        feat.away_rating,
        feat.neutral_site,
        feat.home_rest,
        feat.away_rest,
        feat.away_travel_miles,
    )


def populate_game_projections(
    conn: sqlite3.Connection,
    cal: MarginCalibration,
    constants: ModelConstants,
) -> int:
    """Store projected_margin (raw) and projected_margin_pts (calibrated) per game."""
    games = _load_game_features(conn, None, min_week=1)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cfb_game_projections (
            game_id INTEGER PRIMARY KEY,
            projected_margin REAL,
            projected_margin_pts REAL,
            calibration_train_seasons TEXT
        )
        """
    )
    conn.execute("DELETE FROM cfb_game_projections")
    train_label = ",".join(str(s) for s in cal.train_seasons)
    rows = []
    for g in games:
        raw = projected_margin(
            g.home_rating,
            g.away_rating,
            g.neutral_site,
            constants,
            g.home_rest,
            g.away_rest,
            g.away_travel_miles,
        )
        pts = projected_margin_pts_from_features(cal, g)
        rows.append((g.game_id, raw, pts, train_label))
    conn.executemany(
        """
        INSERT INTO cfb_game_projections (game_id, projected_margin, projected_margin_pts, calibration_train_seasons)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def estimate_model_constants(conn: sqlite3.Connection) -> ModelConstants:
    rows = conn.execute(
        """
        SELECT g.game_id, g.season, g.week, g.home_team, g.away_team,
               g.home_points, g.away_points, g.neutral_site, g.start_date,
               rh.rating, ra.rating, v.latitude, v.longitude
        FROM cfb_games g
        JOIN cfb_team_ratings_pit rh
          ON rh.season = g.season AND rh.week = g.week AND rh.team = g.home_team
        JOIN cfb_team_ratings_pit ra
          ON ra.season = g.season AND ra.week = g.week AND ra.team = g.away_team
        LEFT JOIN cfb_venues v ON v.venue_id = g.venue_id
        WHERE g.home_points IS NOT NULL
          AND lower(coalesce(g.home_division,'')) = 'fbs'
          AND lower(coalesce(g.away_division,'')) = 'fbs'
          AND g.season NOT IN (2020)
          AND g.season BETWEEN 2015 AND 2025
          AND g.week >= 4
        """
    ).fetchall()

    hfa_residuals: list[float] = []
    rest_x: list[float] = []
    rest_y: list[float] = []
    travel_x: list[float] = []
    travel_y: list[float] = []
    last_date: dict[tuple[int, str], datetime] = {}
    last_venue: dict[tuple[int, str], tuple[float, float]] = {}

    for row in rows:
        (
            _gid,
            season,
            _week,
            home,
            away,
            hp,
            ap,
            neutral,
            start_date,
            hr,
            ar,
            lat,
            lon,
        ) = row
        margin = hp - ap
        rating_diff = (hr or 0) - (ar or 0)
        resid = margin - rating_diff
        if not neutral:
            hfa_residuals.append(resid)

        dt = _parse_date(start_date)
        if dt:
            hk, ak = (season, home), (season, away)
            if hk in last_date:
                rest_x.append((dt - last_date[hk]).days)
                rest_y.append(resid)
            if ak in last_date:
                rest_x.append(-(dt - last_date[ak]).days)
                rest_y.append(resid)
            last_date[hk] = dt
            last_date[ak] = dt

        if lat is not None and lon is not None:
            ak = (season, away)
            if ak in last_venue:
                travel_x.append(_haversine_miles(last_venue[ak][0], last_venue[ak][1], lat, lon))
                travel_y.append(resid)
            last_venue[ak] = (lat, lon)
            last_venue[(season, home)] = (lat, lon)

    hfa = float(np.mean(hfa_residuals)) if hfa_residuals else 2.5
    rest_coef = (
        float(np.cov(rest_x, rest_y, bias=True)[0, 1] / (np.var(rest_x) + 1e-6))
        if len(rest_x) > 30
        else 0.0
    )
    travel_coef = (
        float(np.cov(travel_x, travel_y, bias=True)[0, 1] / (np.var(travel_x) + 1e-6))
        if len(travel_x) > 30
        else 0.0
    )
    return ModelConstants(hfa=hfa, rest_coef=rest_coef, travel_coef=travel_coef)


def projected_margin(
    home_rating: float,
    away_rating: float,
    neutral_site: bool,
    constants: ModelConstants,
    home_rest: float = 0.0,
    away_rest: float = 0.0,
    away_travel_miles: float = 0.0,
) -> float:
    hfa = 0.0 if neutral_site else constants.hfa
    rest_adj = constants.rest_coef * (home_rest - away_rest)
    travel_adj = -constants.travel_coef * away_travel_miles
    return (home_rating - away_rating) + hfa + rest_adj + travel_adj


def run_phase1_pipeline(db_path: str | None = None) -> ModelConstants:
    path = db_path or str(ESPN_DB_PATH)
    conn = sqlite3.connect(path)
    try:
        from engine.cfb_schema import ensure_cfb_schema

        ensure_cfb_schema(conn)
        build_pit_ratings(conn)
        return estimate_model_constants(conn)
    finally:
        conn.close()
