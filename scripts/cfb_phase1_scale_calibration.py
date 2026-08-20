#!/usr/bin/env python3
"""CFB Phase 1 scale calibration and re-test."""

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
from engine.cfb_lines import (
    align_betting_spread_to_home,
    betting_spread_to_line_margin,
    populate_spread_home,
)
from engine.cfb_ratings import (
    MarginCalibration,
    build_pit_ratings,
    estimate_model_constants,
    fit_margin_calibration,
    populate_game_projections,
    projected_margin,
    projected_margin_pts_from_cal,
    _load_game_features,
)
from engine.cfb_schema import ensure_cfb_schema

MIN_TEST_WEEK = 4
ADVANCED_TRAIN_SEASONS = (2021, 2022, 2023, 2024, 2025)
WALK_FORWARD = (
    ((2021, 2022), 2023),
    ((2021, 2022, 2023), 2024),
    ((2021, 2022, 2023, 2024), 2025),
)


def ols_plain(y: np.ndarray, x_cols: list[np.ndarray]) -> dict:
    n = len(y)
    X = np.column_stack([np.ones(n)] + x_cols)
    k = X.shape[1]
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t_stats = beta / np.where(se > 0, se, np.nan)
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "beta": beta,
        "se": se,
        "t": t_stats,
        "p": p_vals,
        "r2": 1 - ss_res / ss_tot if ss_tot else 0.0,
        "n": n,
    }


def _dist_stats(arr: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr, ddof=1)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


def _print_dist(label: str, arr: np.ndarray) -> None:
    s = _dist_stats(arr)
    print(
        f"  {label}: mean={s['mean']:+.2f} SD={s['sd']:.2f} "
        f"min={s['min']:+.1f} max={s['max']:+.1f} "
        f"p5={s['p5']:+.1f} p95={s['p95']:+.1f}"
    )


def _spread_open_margin(spread_open, home_ml, away_ml):
    return betting_spread_to_line_margin(align_betting_spread_to_home(spread_open, home_ml, away_ml))


def load_test_rows(conn, constants, cal: MarginCalibration) -> list[dict]:
    rows = conn.execute(
        """
        SELECT g.game_id, g.season, g.week, g.home_points, g.away_points,
               g.neutral_site,
               l.spread_home,
               rh.rating, ra.rating,
               b.spread_open, b.spread, b.home_moneyline, b.away_moneyline,
               b.opener_suspect,
               gp.projected_margin, gp.projected_margin_pts
        FROM cfb_games g
        JOIN cfb_lines l ON l.game_id = g.game_id AND l.is_backtest_reference = 1
        JOIN cfb_team_ratings_pit rh
          ON rh.season = g.season AND rh.week = g.week AND rh.team = g.home_team
        JOIN cfb_team_ratings_pit ra
          ON ra.season = g.season AND ra.week = g.week AND ra.team = g.away_team
        LEFT JOIN cfb_lines b ON b.game_id = g.game_id AND b.provider = 'Bovada'
        LEFT JOIN cfb_game_projections gp ON gp.game_id = g.game_id
        WHERE lower(coalesce(g.home_division,'')) = 'fbs'
          AND lower(coalesce(g.away_division,'')) = 'fbs'
          AND g.home_points IS NOT NULL
          AND l.spread_home IS NOT NULL
          AND g.week >= ?
          AND g.season BETWEEN 2021 AND 2025
        """,
        (MIN_TEST_WEEK,),
    ).fetchall()

    feat_map = {g.game_id: g for g in _load_game_features(conn, (2021, 2022, 2023, 2024, 2025), MIN_TEST_WEEK)}

    out = []
    for r in rows:
        gid = r[0]
        feat = feat_map.get(gid)
        margin = r[3] - r[4]
        raw = r[14] if r[14] is not None else (
            projected_margin(r[7], r[8], bool(r[5]), constants,
                             feat.home_rest if feat else 7, feat.away_rest if feat else 7,
                             feat.away_travel_miles if feat else 0)
            if feat else projected_margin(r[7], r[8], bool(r[5]), constants)
        )
        pts = r[15] if r[15] is not None else (
            projected_margin_pts_from_cal(cal, r[7], r[8], bool(r[5]),
                                          feat.home_rest if feat else 7,
                                          feat.away_rest if feat else 7,
                                          feat.away_travel_miles if feat else 0)
            if feat else 0
        )
        spread_open = _spread_open_margin(r[9], r[12], r[13])
        spread_close = betting_spread_to_line_margin(align_betting_spread_to_home(r[10], r[12], r[13]))
        out.append({
            "game_id": gid,
            "season": r[1],
            "week": r[2],
            "actual_margin": margin,
            "spread_home": r[6],
            "spread_open": spread_open,
            "spread_close_bovada": spread_close,
            "projected_margin": raw,
            "projected_margin_pts": pts,
            "opener_suspect": bool(r[13]),
        })
    return out


def section1_compression(rows: list[dict]) -> None:
    print("=" * 72)
    print("SECTION 1 — Confirm compression (2,995 test-pool games)")
    print("=" * 72)
    test = rows
    print(f"n={len(test)}")
    spread = np.array([r["spread_home"] for r in test])
    proj = np.array([r["projected_margin"] for r in test])
    actual = np.array([r["actual_margin"] for r in test])
    _print_dist("spread_home", spread)
    _print_dist("projected_margin (raw)", proj)
    ratio = float(np.std(spread, ddof=1) / np.std(proj, ddof=1))
    print(f"\n  SD(spread_home)/SD(projected_margin) = {ratio:.2f}x")
    corr = float(np.corrcoef(proj, actual)[0, 1])
    _print_dist("actual_margin", actual)
    rmse_proj = float(np.sqrt(np.mean((actual - proj) ** 2)))
    print(f"\n  corr(projected_margin, actual_margin) = {corr:.4f}")
    print(f"  RMSE(projected_margin) = {rmse_proj:.2f}")
    print(f"  SD(actual_margin) = {np.std(actual, ddof=1):.2f}")
    if abs(rmse_proj - np.std(actual, ddof=1)) < 2:
        print("  → RMSE ≈ SD(actual): raw projection has near-zero standalone predictive power.")


def section2_calibration(conn, master_cal: MarginCalibration, test_rows: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("SECTION 2 — Calibrate rating scale (training seasons only)")
    print("=" * 72)
    pools = (
        ((2021, 2022), "2021-2022"),
        ((2021, 2022, 2023), "2021-2023"),
        ((2021, 2022, 2023, 2024), "2021-2024"),
        ((2015, 2016, 2017, 2018, 2019), "2015-2019 (legacy)"),
    )
    master_cal = None
    for seasons, label in pools:
        try:
            cal = fit_margin_calibration(conn, seasons, min_week=MIN_TEST_WEEK)
        except ValueError:
            continue
        if label.startswith("2021-2024"):
            master_cal = cal
        print(f"\n  Train pool {label} (n={cal.n}):")
        print(f"    intercept a = {cal.intercept:+.3f} (SE={cal.se_intercept:.3f})")
        print(f"    B (rating_coef) = {cal.rating_coef:+.3f} (SE={cal.se_rating_coef:.3f})")
        print(f"    HFA = {cal.hfa:+.3f} (SE={cal.se_hfa:.3f})")
        print(f"    rest_coef = {cal.rest_coef:+.4f} (SE={cal.se_rest_coef:.4f})")
        print(f"    travel_coef = {cal.travel_coef:+.6f} (SE={cal.se_travel_coef:.6f})")
        print(f"    R² = {cal.r2:.4f}")

    if master_cal is None:
        master_cal = fit_margin_calibration(conn, (2021, 2022, 2023, 2024), min_week=MIN_TEST_WEEK)

    pts = np.array([r["projected_margin_pts"] for r in test_rows])
    spread = np.array([r["spread_home"] for r in test_rows])
    _print_dist("\n  projected_margin_pts (test pool)", pts)
    _print_dist("  spread_home (test pool)", spread)
    sd_ratio = float(np.std(pts, ddof=1) / np.std(spread, ddof=1))
    print(f"  SD(projected_margin_pts)/SD(spread_home) = {sd_ratio:.2f}")
    if 0.7 <= sd_ratio <= 1.3:
        print("  → Calibrated projection SD is comparable to spread SD.")
    else:
        print("  → SD still mismatched; calibration may need refinement.")


def section3_walkforward(conn, feat_by_id: dict) -> list[dict]:
    print("\n" + "=" * 72)
    print("SECTION 3 — Restricted walk-forward (2021+ advanced-stats era only)")
    print("=" * 72)
    print("Plain OLS on test pools. projected_margin_pts used in regression.\n")
    constants = estimate_model_constants(conn)
    all_test: list[dict] = []

    for train_seasons, test_season in WALK_FORWARD:
        cal_fold = fit_margin_calibration(conn, train_seasons, min_week=MIN_TEST_WEEK)
        rows = load_test_rows(conn, constants, cal_fold)
        test = [r for r in rows if r["season"] == test_season]
        train = [r for r in rows if r["season"] in train_seasons]
        for r in test + train:
            feat = feat_by_id.get(r["game_id"])
            if feat:
                r["projected_margin_pts"] = projected_margin_pts_from_cal(
                    cal_fold,
                    feat.home_rating,
                    feat.away_rating,
                    feat.neutral_site,
                    feat.home_rest,
                    feat.away_rest,
                    feat.away_travel_miles,
                )
        all_test.extend(test)

        y_tr = np.array([r["actual_margin"] for r in train])
        fit = ols_plain(y_tr, [
            np.array([r["spread_home"] for r in train]),
            np.array([r["projected_margin_pts"] for r in train]),
        ])
        y_te = np.array([r["actual_margin"] for r in test])
        print(f"--- Train {train_seasons[0]}-{train_seasons[-1]}, test {test_season} ---")
        print(f"  n_train={len(train)} n_test={len(test)}  B={cal_fold.rating_coef:.2f}  R²_cal={cal_fold.r2:.3f}")
        print(
            f"  b1={fit['beta'][1]:+.3f} (SE={fit['se'][1]:.3f}, p={fit['p'][1]:.4f})  "
            f"b2={fit['beta'][2]:+.3f} (SE={fit['se'][2]:.3f}, p={fit['p'][2]:.4f})  "
            f"R²_train={fit['r2']:.4f}"
        )

    print(f"\n--- Pooled test OLS (n={len(all_test)}) ---")
    y = np.array([r["actual_margin"] for r in all_test])
    fit = ols_plain(y, [
        np.array([r["spread_home"] for r in all_test]),
        np.array([r["projected_margin_pts"] for r in all_test]),
    ])
    print(
        f"  b1={fit['beta'][1]:+.3f} (SE={fit['se'][1]:.3f}, p={fit['p'][1]:.4f})  "
        f"b2={fit['beta'][2]:+.3f} (SE={fit['se'][2]:.3f}, p={fit['p'][2]:.4f})  "
        f"R²={fit['r2']:.4f}"
    )
    return all_test


def _economic_buckets(rows: list[dict], line_key: str, proj_key: str, title: str) -> None:
    eligible = [
        r for r in rows
        if r.get(line_key) is not None and not r.get("opener_suspect")
    ]
    edges = np.array([r[proj_key] - r[line_key] for r in eligible])
    print(f"\n{title} (n={len(eligible)})")
    _print_dist("edge", edges)

    buckets = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 999)]
    print(f"\n{'bucket':<8} {'n':>5} {'ATS':>12} {'win%':>7} {'SE':>6} {'ROI':>8} {'CLV%':>7} {'|line|':>7} {'dog%':>6}")
    all_picks_dog = 0
    all_picks = 0
    for lo, hi in buckets:
        sub = [r for r in eligible if lo <= abs(r[proj_key] - r[line_key]) < hi]
        wins = losses = pushes = picks_dog = 0
        toward = clv_n = 0
        line_abs = []
        for r in sub:
            edge = r[proj_key] - r[line_key]
            if abs(edge) < 0.001:
                continue
            pick_home = edge > 0
            line = r[line_key]
            m = r["actual_margin"]
            all_picks += 1
            is_dog_pick = (pick_home and line < 0) or (not pick_home and line > 0)
            if is_dog_pick:
                picks_dog += 1
                all_picks_dog += 1
            line_abs.append(abs(line))
            if pick_home:
                if m > line:
                    wins += 1
                elif m < line:
                    losses += 1
                else:
                    pushes += 1
            else:
                if m < line:
                    wins += 1
                elif m > line:
                    losses += 1
                else:
                    pushes += 1
            if r.get("spread_close_bovada") is not None and line_key == "spread_open":
                move = r["spread_close_bovada"] - r["spread_open"]
                if edge > 0 and move > 0:
                    toward += 1
                    clv_n += 1
                elif edge < 0 and move < 0:
                    toward += 1
                    clv_n += 1
                elif edge != 0:
                    clv_n += 1
        n_bets = wins + losses
        wp = wins / n_bets if n_bets else float("nan")
        wp_se = float(np.sqrt(wp * (1 - wp) / n_bets)) if n_bets else float("nan")
        roi = ((wins * (100 / 110) - losses) / n_bets) if n_bets else float("nan")
        clv = toward / clv_n if clv_n else float("nan")
        mean_line = float(np.mean(line_abs)) if line_abs else float("nan")
        dog_pct = picks_dog / (wins + losses + pushes) if (wins + losses + pushes) else float("nan")
        label = f"{lo}-{hi}" if hi < 999 else "5+"
        print(
            f"{label:<8} {len(sub):>5} {wins}-{losses}-{pushes:>3} "
            f"{wp:>6.1%} {wp_se:>6.3f} {roi:>7.1%} {clv:>6.1%} {mean_line:>7.1f} {dog_pct:>5.1%}"
        )
    overall_dog = all_picks_dog / all_picks if all_picks else float("nan")
    print(f"  Overall underdog pick rate: {overall_dog:.1%} (expect ~50% on correct scale)")


def section4_opener(rows: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("SECTION 4 — Economics on calibrated scale vs OPENER")
    print("=" * 72)
    _economic_buckets(rows, "spread_open", "projected_margin_pts", "edge_open = projected_margin_pts - spread_open")


def section5_closing(rows: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("SECTION 5 — Closing-line gate (calibrated scale)")
    print("=" * 72)
    test = [r for r in rows if r["season"] in (2023, 2024, 2025)]
    y = np.array([r["actual_margin"] for r in test])
    fit = ols_plain(y, [
        np.array([r["spread_home"] for r in test]),
        np.array([r["projected_margin_pts"] for r in test]),
    ])
    print(f"Pooled OLS restricted test 2023-2025 (n={len(test)}):")
    print(
        f"  b1={fit['beta'][1]:+.3f} (SE={fit['se'][1]:.3f}, p={fit['p'][1]:.4f})  "
        f"b2={fit['beta'][2]:+.3f} (SE={fit['se'][2]:.3f}, p={fit['p'][2]:.4f})  "
        f"R²={fit['r2']:.4f}"
    )
    _economic_buckets(test, "spread_home", "projected_margin_pts", "edge_close = projected_margin_pts - spread_home")


def main() -> None:
    conn = sqlite3.connect(ESPN_DB_PATH)
    ensure_cfb_schema(conn)
    populate_spread_home(conn)
    build_pit_ratings(conn)
    constants = estimate_model_constants(conn)

    master_cal = fit_margin_calibration(conn, (2021, 2022, 2023, 2024), min_week=MIN_TEST_WEEK)
    n_proj = populate_game_projections(conn, master_cal, constants)
    print(f"Stored {n_proj} rows in cfb_game_projections (projected_margin + projected_margin_pts)")

    rows = load_test_rows(conn, constants, master_cal)
    section1_compression(rows)
    section2_calibration(conn, master_cal, rows)
    feat_by_id = {
        g.game_id: g
        for g in _load_game_features(conn, ADVANCED_TRAIN_SEASONS, MIN_TEST_WEEK)
    }
    section3_rows = section3_walkforward(conn, feat_by_id)
    section4_opener(rows)
    section5_closing(section3_rows if section3_rows else rows)
    conn.close()


if __name__ == "__main__":
    main()
