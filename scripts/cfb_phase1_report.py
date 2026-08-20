#!/usr/bin/env python3
"""CFB Phase 1 report: spread diagnosis, spread_home, PIT ratings, walk-forward test."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.cfb_config import ESPN_DB_PATH
from engine.cfb_lines import populate_opener_suspect, populate_spread_home
from engine.cfb_ratings import (
    ModelConstants,
    build_pit_ratings,
    estimate_model_constants,
    projected_margin,
)
from engine.cfb_schema import ensure_cfb_schema

TEST_SEASONS = (2021, 2022, 2023, 2024, 2025)
MIN_TEST_WEEK = 4


def ols_with_inference(y: np.ndarray, x_cols: list[np.ndarray]) -> dict:
    """OLS of y on intercept + x_cols. Returns coefs, SEs, p-values, R²."""
    n = len(y)
    X = np.column_stack([np.ones(n)] + x_cols)
    k = X.shape[1]
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        cov = np.full((k, k), np.nan)
    se = np.sqrt(np.diag(cov))
    t_stats = beta / np.where(se > 0, se, np.nan)
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    names = ["intercept"] + [f"x{i}" for i in range(1, k)]
    return {
        "names": names,
        "beta": beta,
        "se": se,
        "p": p_vals,
        "r2": r2,
        "n": n,
        "resid": resid,
    }


def section1_disagreements(conn: sqlite3.Connection) -> bool:
    print("=" * 72)
    print("SECTION 1 — Bovada vs teamrankings spread disagreements (>1.0 pt, 2021–2022 FBS)")
    print("=" * 72)
    rows = conn.execute(
        """
        SELECT g.game_id, g.season, g.week, g.away_team, g.home_team,
               g.neutral_site, COALESCE(v.name, '') AS venue,
               b.spread AS bovada_spread, t.spread AS tr_spread,
               ABS(b.spread - t.spread) AS diff,
               CASE WHEN ABS(b.spread + t.spread) < 0.01 THEN 1 ELSE 0 END AS sign_flip
        FROM cfb_games g
        LEFT JOIN cfb_venues v ON v.venue_id = g.venue_id
        JOIN cfb_lines b ON b.game_id = g.game_id AND b.provider = 'Bovada'
        JOIN cfb_lines t ON t.game_id = g.game_id AND t.provider = 'teamrankings'
        WHERE g.season IN (2021, 2022)
          AND lower(coalesce(g.home_division,'')) = 'fbs'
          AND lower(coalesce(g.away_division,'')) = 'fbs'
          AND b.spread IS NOT NULL AND t.spread IS NOT NULL
          AND ABS(b.spread - t.spread) > 1.0
        ORDER BY diff DESC
        """
    ).fetchall()
    print(
        f"{'game_id':<12} {'season':>6} {'wk':>3} {'matchup':<35} {'neut':>4} "
        f"{'bovada':>7} {'tr':>7} {'diff':>6} {'flip':>4}"
    )
    print("-" * 95)
    for r in rows:
        gid, season, week, away, home, neut, venue, bov, tr, diff, flip = r
        matchup = f"{away} @ {home}"[:35]
        print(
            f"{gid:<12} {season:>6} {week:>3} {matchup:<35} {neut:>4} "
            f"{bov:>7.1f} {tr:>7.1f} {diff:>6.1f} {flip:>4}"
        )
        if venue:
            print(f"             venue: {venue}")

    for ns in (0, 1):
        sub = [r for r in rows if r[5] == ns]
        if sub:
            mean_diff = sum(r[9] for r in sub) / len(sub)
            flips = sum(r[10] for r in sub)
            print(
                f"\nneutral_site={ns}: n={len(sub)}, "
                f"mean_abs_diff={mean_diff:.3f}, sign_flips={flips}"
            )

    total_flips = sum(r[10] for r in rows)
    non_neutral_flips = sum(r[10] for r in rows if r[5] == 0)
    neutral_flips = sum(r[10] for r in rows if r[5] == 1)

    print("\n--- Hypothesis: large disagreements = neutral-site home/away mismatch ---")
    if neutral_flips > 0 and non_neutral_flips == 0:
        print("CONFIRMED: all sign flips are neutral-site games.")
        confirmed = True
    elif non_neutral_flips > 0:
        print(
            f"PARTIALLY REFUTED: {non_neutral_flips} non-neutral sign-flip(s) among "
            f"{len(rows)} games >1 pt (plus {len(rows) - total_flips} magnitude-only diffs)."
        )
        print("Non-neutral sign-flip games (likely mis-flagged neutral or provider swap):")
        for r in rows:
            if r[5] == 0 and r[10]:
                print(f"  {r[0]} {r[3]} @ {r[4]} week {r[2]} bovada={r[7]} tr={r[8]}")
        confirmed = False
    else:
        print("REFUTED: no pure sign flips; disagreements are magnitude-only.")
        confirmed = False

    print("\nNeutral-site games in backtest reference set (FBS-vs-FBS) per season:")
    ref_neutral = conn.execute(
        """
        SELECT g.season, COUNT(*),
               SUM(CASE WHEN g.neutral_site = 1 THEN 1 ELSE 0 END)
        FROM cfb_games g
        JOIN cfb_lines l ON l.game_id = g.game_id AND l.is_backtest_reference = 1
        WHERE lower(coalesce(g.home_division,'')) = 'fbs'
          AND lower(coalesce(g.away_division,'')) = 'fbs'
        GROUP BY g.season ORDER BY g.season
        """
    ).fetchall()
    for season, total, neutral in ref_neutral:
        print(f"  {season}: {neutral} neutral / {total} FBS-vs-FBS reference games")

    if not confirmed and non_neutral_flips > 0:
        print(
            "\nNOTE: Backtest uses a single reference provider per game (teamrankings "
            "2015–2020, Bovada 2021+), so cross-provider sign flips affect overlap QA "
            "only. Proceeding with spread_home normalization per Section 2."
        )
    return confirmed


def section2_spread_convention(conn: sqlite3.Connection) -> None:
    print("\n" + "=" * 72)
    print("SECTION 2 — Spread convention verification (20 blowouts, margin ≥ 21)")
    print("=" * 72)
    rows = conn.execute(
        """
        SELECT g.away_team, g.home_team, g.home_points, g.away_points,
               (g.home_points - g.away_points) AS margin, l.spread, l.spread_home,
               gs.is_home, l.provider, g.neutral_site
        FROM cfb_games g
        JOIN cfb_lines l ON l.game_id = g.game_id AND l.is_backtest_reference = 1
        LEFT JOIN cfb_game_stats gs ON gs.game_id = g.game_id AND gs.team = g.home_team
        WHERE g.season BETWEEN 2021 AND 2025
          AND lower(coalesce(g.home_division,'')) = 'fbs'
          AND lower(coalesce(g.away_division,'')) = 'fbs'
          AND g.home_points IS NOT NULL
          AND ABS(g.home_points - g.away_points) >= 21
        ORDER BY ABS(g.home_points - g.away_points) DESC
        LIMIT 20
        """
    ).fetchall()
    home_perspective_ok = 0
    for r in rows:
        away, home, hp, ap, margin, spread, spread_home, is_home, provider, neut = r
        home_fav_spread = spread < 0 if spread is not None else None
        home_won_big = margin >= 21
        away_won_big = margin <= -21
        obvious_home_fav = home_won_big
        obvious_away_fav = away_won_big
        consistent = (
            (obvious_home_fav and home_fav_spread)
            or (obvious_away_fav and not home_fav_spread)
            or (not obvious_home_fav and not obvious_away_fav)
        )
        if consistent:
            home_perspective_ok += 1
        print(
            f"{away} @ {home}  {hp}-{ap}  margin={margin:+d}  "
            f"spread={spread} spread_home={spread_home}  provider={provider}  "
            f"is_home={is_home}  neutral={neut}"
        )
    print(
        f"\nConvention: spread is from HOME team perspective (negative = home favored). "
        f"Reference provider consistent with obvious favorite in {home_perspective_ok}/20 blowouts."
    )
    ml_aligned = conn.execute(
        """
        SELECT COUNT(*) FROM cfb_lines l
        WHERE l.spread IS NOT NULL
          AND l.home_moneyline IS NOT NULL AND l.away_moneyline IS NOT NULL
          AND ABS(l.spread + l.spread_home) < 0.01 AND ABS(l.spread) > 0.01
        """
    ).fetchone()[0]
    print(
        f"ML alignment sign-flips (betting spread): {ml_aligned} rows; "
        f"spread_home stored on margin scale (positive = home favored)."
    )


def section3_opener_suspect(conn: sqlite3.Connection) -> None:
    print("\n" + "=" * 72)
    print("SECTION 3 — opener_suspect flag")
    print("=" * 72)
    n = populate_opener_suspect(conn)
    print(f"Marked opener_suspect=1 on {n} DraftKings 2023 rows.")
    print("Reason: DraftKings 2023 opener spreads flagged in Phase 0 provider audit.")


def _load_evaluation_rows(conn: sqlite3.Connection, constants: ModelConstants) -> list[dict]:
    rows = conn.execute(
        """
        SELECT g.game_id, g.season, g.week, g.home_team, g.away_team,
               g.home_points, g.away_points, g.neutral_site,
               l.spread_home, rh.rating, ra.rating
        FROM cfb_games g
        JOIN cfb_lines l ON l.game_id = g.game_id AND l.is_backtest_reference = 1
        JOIN cfb_team_ratings_pit rh
          ON rh.season = g.season AND rh.week = g.week AND rh.team = g.home_team
        JOIN cfb_team_ratings_pit ra
          ON ra.season = g.season AND ra.week = g.week AND ra.team = g.away_team
        WHERE lower(coalesce(g.home_division,'')) = 'fbs'
          AND lower(coalesce(g.away_division,'')) = 'fbs'
          AND g.home_points IS NOT NULL
          AND l.spread_home IS NOT NULL
          AND g.week >= ?
        """,
        (MIN_TEST_WEEK,),
    ).fetchall()
    out = []
    for r in rows:
        margin = r[5] - r[6]
        proj = projected_margin(r[9], r[10], bool(r[7]), constants)
        out.append(
            {
                "game_id": r[0],
                "season": r[1],
                "week": r[2],
                "actual_margin": margin,
                "spread_home": r[8],
                "projected_margin": proj,
            }
        )
    return out


def _ats_stats(rows: list[dict], threshold: float) -> tuple[float, int]:
    wins = 0
    n = 0
    for r in rows:
        edge = r["projected_margin"] - r["spread_home"]
        if abs(edge) < threshold:
            continue
        n += 1
        pick_home = edge > 0
        actual = r["actual_margin"]
        line = r["spread_home"]
        if pick_home:
            if actual > line:
                wins += 1
            elif actual == line:
                n -= 1
        else:
            if actual < line:
                wins += 1
            elif actual == line:
                n -= 1
    rate = wins / n if n else float("nan")
    return rate, n


def section45_walkforward(conn: sqlite3.Connection, constants: ModelConstants) -> None:
    print("\n" + "=" * 72)
    print("SECTION 4 — PIT ratings built (cfb_team_ratings_pit)")
    print("=" * 72)
    n = conn.execute("SELECT COUNT(*) FROM cfb_team_ratings_pit").fetchone()[0]
    teams = conn.execute("SELECT COUNT(DISTINCT team) FROM cfb_team_ratings_pit").fetchone()[0]
    print(f"Rows: {n:,}  |  Teams: {teams}  |  HFA={constants.hfa:.2f}  "
          f"rest_coef={constants.rest_coef:.4f}  travel_coef={constants.travel_coef:.6f}")

    print("\n" + "=" * 72)
    print(f"SECTION 5 — Walk-forward test (weeks {MIN_TEST_WEEK}+ only)")
    print("=" * 72)
    print("Regression: actual_margin ~ b1*spread_home + b2*projected_margin")

    all_rows = _load_evaluation_rows(conn, constants)
    prior_seasons = {s for s in range(2015, 2026) if s not in (2020,)}

    pooled_test: list[dict] = []
    for test_season in TEST_SEASONS:
        train = [
            r
            for r in all_rows
            if r["season"] in prior_seasons
            and r["season"] < test_season
        ]
        test = [r for r in all_rows if r["season"] == test_season]
        pooled_test.extend(test)
        if len(train) < 50 or len(test) < 20:
            print(f"\n{test_season}: insufficient data (train={len(train)}, test={len(test)})")
            continue

        y_train = np.array([r["actual_margin"] for r in train])
        x1_train = np.array([r["spread_home"] for r in train])
        x2_train = np.array([r["projected_margin"] for r in train])
        fit = ols_with_inference(y_train, [x1_train, x2_train])

        y_test = np.array([r["actual_margin"] for r in test])
        x1_test = np.array([r["spread_home"] for r in test])
        x2_test = np.array([r["projected_margin"] for r in test])
        X_test = np.column_stack([np.ones(len(test)), x1_test, x2_test])
        y_hat = X_test @ fit["beta"]
        test_resid = y_test - y_hat
        ss_res = float(test_resid @ test_resid)
        ss_tot = float(((y_test - y_test.mean()) ** 2).sum())
        test_r2 = 1 - ss_res / ss_tot if ss_tot else 0.0

        proj_rmse = float(np.sqrt(np.mean((y_test - x2_test) ** 2)))
        proj_mae = float(np.mean(np.abs(y_test - x2_test)))
        spread_rmse = float(np.sqrt(np.mean((y_test - x1_test) ** 2)))
        spread_mae = float(np.mean(np.abs(y_test - x1_test)))
        sigma = float(np.std(y_test - x1_test, ddof=1))

        print(f"\n--- Test season {test_season} (n_test={len(test)}, n_train={len(train)}) ---")
        print(
            f"  b1(spread_home)={fit['beta'][1]:+.3f} (SE={fit['se'][1]:.3f}, p={fit['p'][1]:.4f})"
        )
        print(
            f"  b2(projected_margin)={fit['beta'][2]:+.3f} "
            f"(SE={fit['se'][2]:.3f}, p={fit['p'][2]:.4f})"
        )
        print(f"  R² (in-sample train)={fit['r2']:.4f}  |  R² (holdout test)={test_r2:.4f}")
        print(f"  RMSE projected={proj_rmse:.2f}  MAE projected={proj_mae:.2f}")
        print(f"  RMSE spread_home={spread_rmse:.2f}  MAE spread_home={spread_mae:.2f}  sigma={sigma:.2f}")
        for thr in (1, 2, 3, 4, 5):
            rate, cnt = _ats_stats(test, thr)
            print(f"  ATS edge≥{thr}pt: {rate:.1%} ({cnt} picks)")

    if pooled_test:
        print(f"\n--- Pooled test 2021–2025 weeks {MIN_TEST_WEEK}+ (n={len(pooled_test)}) ---")
        train_all = [r for r in all_rows if r["season"] < min(TEST_SEASONS)]
        y_train = np.array([r["actual_margin"] for r in train_all])
        fit = ols_with_inference(
            y_train,
            [
                np.array([r["spread_home"] for r in train_all]),
                np.array([r["projected_margin"] for r in train_all]),
            ],
        )
        y_test = np.array([r["actual_margin"] for r in pooled_test])
        x1 = np.array([r["spread_home"] for r in pooled_test])
        x2 = np.array([r["projected_margin"] for r in pooled_test])
        proj_rmse = float(np.sqrt(np.mean((y_test - x2) ** 2)))
        spread_rmse = float(np.sqrt(np.mean((y_test - x1) ** 2)))
        sigma = float(np.std(y_test - x1, ddof=1))
        print(
            f"  b1={fit['beta'][1]:+.3f} (p={fit['p'][1]:.4f})  "
            f"b2={fit['beta'][2]:+.3f} (p={fit['p'][2]:.4f})  R²_train={fit['r2']:.4f}"
        )
        print(
            f"  RMSE projected={proj_rmse:.2f}  RMSE spread={spread_rmse:.2f}  sigma={sigma:.2f}"
        )
        b2_sig = fit["p"][2] < 0.05 and fit["beta"][2] > 0
        if b2_sig:
            print("\n  b2 is significantly positive — model adds information beyond the closing line.")
        else:
            print(
                "\n  b2 is NOT significantly positive — model does not beat spread_home "
                "in pooled regression; stop before live use."
            )


def main() -> None:
    conn = sqlite3.connect(ESPN_DB_PATH)
    ensure_cfb_schema(conn)
    section1_disagreements(conn)
    n_spread = populate_spread_home(conn)
    print(f"\nComputed spread_home for {n_spread} cfb_lines rows.")
    section2_spread_convention(conn)
    section3_opener_suspect(conn)
    build_pit_ratings(conn)
    constants = estimate_model_constants(conn)
    section45_walkforward(conn, constants)
    conn.close()


if __name__ == "__main__":
    main()
