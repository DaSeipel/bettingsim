#!/usr/bin/env python3
"""CFB totals fast signal check — crude PIT projection vs closing/opening totals."""

from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.cfb_config import ESPN_DB_PATH
from engine.cfb_schema import ensure_cfb_schema

TEST_SEASONS = (2021, 2022, 2023, 2024, 2025)
MIN_TEST_WEEK = 4

P5_CONFERENCES = frozenset({"ACC", "Big Ten", "Big 12", "SEC", "Pac-12"})
P5_TEAMS = frozenset({"Notre Dame"})


@dataclass
class GameRow:
    game_id: int
    season: int
    week: int
    home_team: str
    away_team: str
    home_conference: str | None
    away_conference: str | None
    conference_game: int
    actual_total: float
    market_total: float
    home_plays_avg: float
    away_plays_avg: float
    home_off_ppa_avg: float
    home_def_ppa_avg: float
    away_off_ppa_avg: float
    away_def_ppa_avg: float


def is_p5(conference: str | None, team: str) -> bool:
    if team in P5_TEAMS:
        return True
    return conference in P5_CONFERENCES


def ols_with_inference(y: np.ndarray, x_cols: list[np.ndarray]) -> dict:
    n = len(y)
    X = np.column_stack([np.ones(n)] + x_cols)
    k = X.shape[1]
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = beta / np.where(se > 0, se, np.nan)
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
    return {"beta": beta, "se": se, "p": p_vals}


def corr_with_p(x: np.ndarray, y: np.ndarray) -> tuple[float, int, float]:
    n = len(x)
    if n < 3:
        return float("nan"), n, float("nan")
    r, p = stats.pearsonr(x, y)
    return float(r), n, float(p)


def rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def raw_projection(row: GameRow) -> float:
    expected_plays = 0.5 * (row.home_plays_avg + row.away_plays_avg)
    home_match = 0.5 * (row.home_off_ppa_avg + row.away_def_ppa_avg)
    away_match = 0.5 * (row.away_off_ppa_avg + row.home_def_ppa_avg)
    expected_ppp = 0.5 * (home_match + away_match)
    return expected_plays * expected_ppp


def fit_scaling(train_rows: list[GameRow]) -> tuple[float, float]:
    if not train_rows:
        return 0.0, 1.0
    raw = np.array([raw_projection(r) for r in train_rows], float)
    actual = np.array([r.actual_total for r in train_rows], float)
    fit = ols_with_inference(actual, [raw])
    return float(fit["beta"][0]), float(fit["beta"][1])


def projected_total(row: GameRow, intercept: float, slope: float) -> float:
    return intercept + slope * raw_projection(row)


def load_team_game_adv(conn: sqlite3.Connection) -> dict[tuple[int, str], list[tuple[int, float, float, float]]]:
    """Map (season, team) -> list of (week, plays, off_ppa, def_ppa) sorted by week."""
    rows = conn.execute(
        """
        SELECT g.season, g.week, ga.team, ga.off_ppa, ga.def_ppa, ga.stats_json
        FROM cfb_game_stats_adv ga
        JOIN cfb_games g ON g.game_id = ga.game_id
        WHERE g.season BETWEEN 2021 AND 2025
          AND g.week IS NOT NULL
          AND lower(coalesce(g.home_division, '')) = 'fbs'
          AND lower(coalesce(g.away_division, '')) = 'fbs'
          AND g.home_points IS NOT NULL
        ORDER BY g.season, ga.team, g.week
        """
    ).fetchall()
    out: dict[tuple[int, str], list[tuple[int, float, float, float]]] = {}
    for season, week, team, off_ppa, def_ppa, stats_json in rows:
        plays = None
        if stats_json:
            try:
                payload = json.loads(stats_json)
                plays = (payload.get("offense") or {}).get("plays")
            except json.JSONDecodeError:
                plays = None
        if plays is None or off_ppa is None or def_ppa is None:
            continue
        out.setdefault((season, team), []).append((week, float(plays), float(off_ppa), float(def_ppa)))
    return out


def pit_averages(
    team_history: list[tuple[int, float, float, float]], before_week: int
) -> tuple[float, float, float] | None:
    prior = [t for t in team_history if t[0] < before_week]
    if not prior:
        return None
    plays = np.mean([t[1] for t in prior])
    off_ppa = np.mean([t[2] for t in prior])
    def_ppa = np.mean([t[3] for t in prior])
    return float(plays), float(off_ppa), float(def_ppa)


def load_games(conn: sqlite3.Connection, market_col: str) -> list[GameRow]:
    team_adv = load_team_game_adv(conn)
    sql = f"""
        SELECT g.game_id, g.season, g.week, g.home_team, g.away_team,
               g.home_conference, g.away_conference, g.conference_game,
               g.home_points + g.away_points AS actual_total,
               l.{market_col} AS market_total
        FROM cfb_games g
        JOIN cfb_lines l ON l.game_id = g.game_id AND l.is_backtest_reference = 1
        WHERE g.season BETWEEN 2021 AND 2025
          AND g.week >= ?
          AND lower(coalesce(g.home_division, '')) = 'fbs'
          AND lower(coalesce(g.away_division, '')) = 'fbs'
          AND g.home_points IS NOT NULL
          AND l.{market_col} IS NOT NULL
        ORDER BY g.season, g.week, g.game_id
    """
    rows = conn.execute(sql, (MIN_TEST_WEEK,)).fetchall()
    games: list[GameRow] = []
    for (
        game_id,
        season,
        week,
        home_team,
        away_team,
        home_conf,
        away_conf,
        conf_game,
        actual_total,
        market_total,
    ) in rows:
        home_hist = team_adv.get((season, home_team), [])
        away_hist = team_adv.get((season, away_team), [])
        home_avg = pit_averages(home_hist, week)
        away_avg = pit_averages(away_hist, week)
        if home_avg is None or away_avg is None:
            continue
        games.append(
            GameRow(
                game_id=int(game_id),
                season=int(season),
                week=int(week),
                home_team=home_team,
                away_team=away_team,
                home_conference=home_conf,
                away_conference=away_conf,
                conference_game=int(conf_game or 0),
                actual_total=float(actual_total),
                market_total=float(market_total),
                home_plays_avg=home_avg[0],
                away_plays_avg=away_avg[0],
                home_off_ppa_avg=home_avg[1],
                home_def_ppa_avg=home_avg[2],
                away_off_ppa_avg=away_avg[1],
                away_def_ppa_avg=away_avg[2],
            )
        )
    return games


def apply_walk_forward(games: list[GameRow]) -> list[tuple[GameRow, float]]:
    out: list[tuple[GameRow, float]] = []
    for season in TEST_SEASONS:
        train = [g for g in games if g.season < season and g.week >= MIN_TEST_WEEK]
        intercept, slope = fit_scaling(train)
        test = [g for g in games if g.season == season]
        for row in test:
            out.append((row, projected_total(row, intercept, slope)))
    return out


def report_signal_block(
    title: str,
    scored: list[tuple[GameRow, float]],
    *,
    market_label: str = "over_under",
) -> dict[str, float]:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    key_rs: list[float] = []
    pooled_actual = np.array([r.actual_total for r, _ in scored], float)
    pooled_proj = np.array([p for _, p in scored], float)
    pooled_market = np.array([r.market_total for r, _ in scored], float)

    def block(label: str, idx: np.ndarray) -> None:
        if len(idx) < 3:
            print(f"\n{label}: n={len(idx)} (too few)")
            return
        actual = pooled_actual[idx]
        proj = pooled_proj[idx]
        market = pooled_market[idx]
        ra, n, pa = corr_with_p(proj, actual)
        rb, _, pb = corr_with_p(proj, market)
        rc, _, pc = corr_with_p(proj - market, actual - market)
        if label == "POOLED":
            key_rs.append(rc)
        print(f"\n{label} (n={n})")
        print(f"  (a) corr(projected_total, actual_total)     = {ra:+.4f}  p={pa:.4g}")
        print(f"  (b) corr(projected_total, {market_label})   = {rb:+.4f}  p={pb:.4g}")
        print(f"  (c) corr(proj-{market_label}, actual-{market_label}) = {rc:+.4f}  p={pc:.4g}")
        fit = ols_with_inference(actual, [market, proj])
        b0, b1, b2 = fit["beta"]
        print(
            f"  OLS actual ~ b1*{market_label} + b2*projected_total:"
        )
        print(
            f"      intercept={b0:+.3f} (se={fit['se'][0]:.3f}, p={fit['p'][0]:.4g})"
        )
        print(
            f"      b1={b1:+.4f} (se={fit['se'][1]:.4f}, p={fit['p'][1]:.4g})"
        )
        print(
            f"      b2={b2:+.4f} (se={fit['se'][2]:.4f}, p={fit['p'][2]:.4g})"
        )
        print(
            f"  RMSE projected={rmse(proj, actual):.2f}  "
            f"RMSE {market_label}={rmse(market, actual):.2f}"
        )

    block("POOLED", np.arange(len(scored)))
    for season in TEST_SEASONS:
        idx = np.array([i for i, (r, _) in enumerate(scored) if r.season == season])
        block(str(season), idx)

    rc_pooled, _, pc_pooled = corr_with_p(pooled_proj - pooled_market, pooled_actual - pooled_market)
    return {"rc": rc_pooled, "pc": pc_pooled, "n": len(scored)}


def report_subset_scan(scored: list[tuple[GameRow, float]]) -> None:
    print("\n" + "=" * 72)
    print("SECTION 4 — Subset scan (key corr (c) only)")
    print("=" * 72)
    actual = np.array([r.actual_total for r, _ in scored], float)
    proj = np.array([p for _, p in scored], float)
    market = np.array([r.market_total for r, _ in scored], float)
    resid_proj = proj - market
    resid_actual = actual - market

    def scan(label: str, mask: np.ndarray) -> None:
        idx = np.where(mask)[0]
        if len(idx) < 3:
            print(f"{label:<28} n={len(idx):>4}  corr(c)=nan")
            return
        rc, n, pc = corr_with_p(resid_proj[idx], resid_actual[idx])
        print(f"{label:<28} n={n:>4}  corr(c)={rc:+.4f}  p={pc:.4g}")

    both_p5 = np.array(
        [
            is_p5(r.home_conference, r.home_team) and is_p5(r.away_conference, r.away_team)
            for r, _ in scored
        ]
    )
    both_g5 = np.array(
        [
            not is_p5(r.home_conference, r.home_team)
            and not is_p5(r.away_conference, r.away_team)
            for r, _ in scored
        ]
    )
    pace = 0.5 * (np.array([r.home_plays_avg + r.away_plays_avg for r, _ in scored]))
    q25, q75 = np.quantile(pace, [0.25, 0.75])
    high_pace = pace >= q75
    low_pace = pace <= q25
    weeks_early = np.array([r.week <= 8 for r, _ in scored])
    weeks_late = np.array([r.week >= 9 for r, _ in scored])
    conf = np.array([bool(r.conference_game) for r, _ in scored])
    non_conf = ~conf

    scan("Both P5", both_p5)
    scan("Both G5", both_g5)
    scan("Highest pace quartile", high_pace)
    scan("Lowest pace quartile", low_pace)
    scan("Weeks 4-8", weeks_early)
    scan("Weeks 9-14", weeks_late)
    scan("Conference games", conf)
    scan("Non-conference games", non_conf)


def main() -> None:
    conn = sqlite3.connect(ESPN_DB_PATH)
    ensure_cfb_schema(conn)

    print("=" * 72)
    print("SECTION 1 — Crude PIT totals projection (one pass, no tuning)")
    print("=" * 72)
    print(
        "expected_plays = mean(home/away season-to-date plays through week W-1)\n"
        "expected_ppp   = avg( home_off vs away_def , away_off vs home_def )\n"
        "raw            = expected_plays * expected_ppp\n"
        "projected_total = intercept + slope * raw  (walk-forward: prior seasons only)\n"
        "Pool: FBS vs FBS, weeks 4+, 2021-2025, Bovada reference line"
    )

    close_games = load_games(conn, "over_under")
    close_scored = apply_walk_forward(close_games)
    print(f"\nGames with projection + closing total: {len(close_scored)}")
    train_note = (
        "2021 uses identity scaling (no prior-season advanced game stats for calibration)."
    )
    print(train_note)

    close_stats = report_signal_block(
        "SECTION 2 — Signal test vs CLOSING total (over_under)",
        close_scored,
        market_label="over_under",
    )

    print("\n" + "=" * 72)
    print("SECTION 3 — Opening totals backfill status")
    print("=" * 72)
    est_calls = conn.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT season, season_type, week
            FROM cfb_games
            WHERE season BETWEEN 2021 AND 2025 AND week IS NOT NULL
            GROUP BY season, season_type, week
        )
        """
    ).fetchone()[0]
    open_cov = conn.execute(
        """
        SELECT COUNT(*)
        FROM cfb_games g
        JOIN cfb_lines l ON l.game_id = g.game_id AND l.is_backtest_reference = 1
        WHERE g.season BETWEEN 2021 AND 2025 AND g.week >= 4
          AND lower(coalesce(g.home_division, '')) = 'fbs'
          AND lower(coalesce(g.away_division, '')) = 'fbs'
          AND l.over_under_open IS NOT NULL
        """
    ).fetchone()[0]
    print(f"Estimated /lines calls for backfill (if needed): {est_calls}")
    print(f"Reserve target after backfill: 150+ calls")
    print(f"Test-pool games with over_under_open present: {open_cov}")

    open_games = load_games(conn, "over_under_open")
    if open_games:
        open_scored = apply_walk_forward(open_games)
        open_stats = report_signal_block(
            "SECTION 3b — Signal test vs OPENING total (over_under_open)",
            open_scored,
            market_label="over_under_open",
        )
    else:
        print("\nNo over_under_open in DB — run scripts/cfb_backfill_ou_open.py first.")
        open_stats = None

    rc = close_stats["rc"]
    pc = close_stats["pc"]
    sig_threshold = 0.05
    has_signal = pc < sig_threshold and rc > 0
    print("\n" + "=" * 72)
    print("GATE — Section 4 eligibility")
    print("=" * 72)
    print(
        f"Pooled key corr (c) vs close = {rc:+.4f} (p={pc:.4g}, n={close_stats['n']})"
    )
    if has_signal:
        print("Key corr significantly positive vs close — running subset scan.")
        report_subset_scan(close_scored)
        if open_stats is not None:
            print("\nSubset scan vs OPEN (same subsets, opening line):")
            # Re-use opening market totals in scored rows
            open_scored = apply_walk_forward(open_games)
            report_subset_scan(open_scored)
    else:
        print(
            "Key corr (c) not significantly positive vs close — "
            "subset scan skipped (no signal to hunt)."
        )

    conn.close()


if __name__ == "__main__":
    main()
