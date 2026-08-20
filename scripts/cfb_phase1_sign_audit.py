#!/usr/bin/env python3
"""CFB Phase 1 sign audit — validate spread convention before accepting model results."""

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
from engine.cfb_lines import align_betting_spread_to_home, populate_spread_home
from engine.cfb_ratings import (
    ModelConstants,
    build_pit_ratings,
    estimate_model_constants,
    projected_margin,
)
from engine.cfb_schema import ensure_cfb_schema

ACTUAL_MARGIN_SQL = "(g.home_points - g.away_points)"
TEST_SEASONS = (2021, 2022, 2023, 2024, 2025)
MIN_TEST_WEEK = 4


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
    p_vals = 2 * (1 - stats.t.cdf(np.abs(beta / np.where(se > 0, se, np.nan)), dof))
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"beta": beta, "se": se, "p": p_vals, "r2": 1 - ss_res / ss_tot if ss_tot else 0.0}


def section1_raw_values(conn: sqlite3.Connection, constants: ModelConstants) -> None:
    print("=" * 72)
    print("SECTION 1 — Raw values (2024 weeks 6–10, home clear favorite |spread| ≥ 7)")
    print("=" * 72)
    print(f"actual_margin expression: {ACTUAL_MARGIN_SQL}")
    print("  (= home_points - away_points; positive when home outscores away)")
    rows = conn.execute(
        f"""
        SELECT g.home_team, g.away_team, g.home_points, g.away_points,
               {ACTUAL_MARGIN_SQL} AS actual_margin,
               l.spread_home, g.neutral_site, rh.rating, ra.rating, l.spread AS raw_spread
        FROM cfb_games g
        JOIN cfb_lines l ON l.game_id = g.game_id AND l.is_backtest_reference = 1
        JOIN cfb_team_ratings_pit rh
          ON rh.season = g.season AND rh.week = g.week AND rh.team = g.home_team
        JOIN cfb_team_ratings_pit ra
          ON ra.season = g.season AND ra.week = g.week AND ra.team = g.away_team
        WHERE g.season = 2024 AND g.week BETWEEN 6 AND 10
          AND lower(coalesce(g.home_division,'')) = 'fbs'
          AND lower(coalesce(g.away_division,'')) = 'fbs'
          AND g.home_points IS NOT NULL
          AND l.spread_home IS NOT NULL AND l.spread_home >= 7
        ORDER BY l.spread_home DESC
        LIMIT 20
        """
    ).fetchall()
    print(
        f"\n{'home':<16} {'away':<16} {'hp':>3} {'ap':>3} {'margin':>7} "
        f"{'spread_home':>11} {'proj':>7} {'neut':>4} {'raw':>7}"
    )
    print("-" * 88)
    for r in rows:
        proj = projected_margin(r[7], r[8], bool(r[6]), constants)
        print(
            f"{r[0]:<16} {r[1]:<16} {r[2]:>3} {r[3]:>3} {r[4]:>7.1f} "
            f"{r[5]:>11.1f} {proj:>7.2f} {r[6]:>4} {r[9]:>7.1f}"
        )
    pos_margin = sum(1 for r in rows if r[4] > 0)
    pos_spread = sum(1 for r in rows if r[5] > 0)
    pos_proj = sum(
        1
        for r in rows
        if projected_margin(r[7], r[8], bool(r[6]), constants) > 0
    )
    print(
        f"\nSign check (home clear favorites): "
        f"actual_margin>0: {pos_margin}/20, spread_home>0: {pos_spread}/20, "
        f"projected_margin>0: {pos_proj}/20"
    )
    print("spread_home: POSITIVE when home is favored (margin scale, = -betting_spread).")
    print("projected_margin: POSITIVE when model favors home.")


def section2_sigma(conn: sqlite3.Connection) -> None:
    print("\n" + "=" * 72)
    print("SECTION 2 — Sigma three ways (2021–2025 FBS reference, weeks 4+)")
    print("=" * 72)
    rows = conn.execute(
        f"""
        SELECT {ACTUAL_MARGIN_SQL} AS m, l.spread_home AS s
        FROM cfb_games g
        JOIN cfb_lines l ON l.game_id = g.game_id AND l.is_backtest_reference = 1
        WHERE g.season BETWEEN 2021 AND 2025 AND g.week >= 4
          AND lower(coalesce(g.home_division,'')) = 'fbs'
          AND lower(coalesce(g.away_division,'')) = 'fbs'
          AND g.home_points IS NOT NULL AND l.spread_home IS NOT NULL
        """
    ).fetchall()
    m = np.array([r[0] for r in rows], float)
    s = np.array([r[1] for r in rows], float)
    results = [
        ("(a) std(actual_margin - spread_home)", m - s),
        ("(b) std(actual_margin + spread_home)", m + s),
        ("(c) std(actual_margin - (-spread_home))", m + s),
    ]
    for label, arr in results:
        print(f"  {label}: std={np.std(arr, ddof=1):.3f}, mean={np.mean(arr):.3f}, n={len(arr)}")
    print(
        "\nWith spread_home on margin scale (positive = home favored), "
        "(a) should be ~13.5 with mean ~0."
    )


def section3_ml_flips(conn: sqlite3.Connection) -> None:
    print("\n" + "=" * 72)
    print("SECTION 3 — ML alignment flips (betting spread, pre-negation)")
    print("=" * 72)
    rows = conn.execute(
        """
        SELECT l.game_id, g.away_team, g.home_team, g.neutral_site,
               l.home_moneyline, l.away_moneyline, l.spread AS orig,
               l.spread_home, g.home_points, g.away_points,
               l.is_backtest_reference, COALESCE(v.name, '')
        FROM cfb_lines l
        JOIN cfb_games g ON g.game_id = l.game_id
        LEFT JOIN cfb_venues v ON v.venue_id = g.venue_id
        WHERE l.spread IS NOT NULL AND l.spread_home IS NOT NULL
          AND l.home_moneyline IS NOT NULL AND l.away_moneyline IS NOT NULL
        """
    ).fetchall()
    flipped_rows = []
    for r in rows:
        aligned = align_betting_spread_to_home(r[6], r[4], r[5])
        if aligned is not None and abs(r[6] + aligned) < 0.01 and abs(r[6]) > 0.01:
            flipped_rows.append((*r, aligned))
    ref = [r for r in flipped_rows if r[10]]
    print(f"Total ML sign-alignment flips: {len(flipped_rows)} (reference set: {len(ref)})")
    print(
        f"  neutral_site=1: {sum(r[3] for r in flipped_rows)} | "
        f"Army/Navy games: {sum(1 for r in flipped_rows if 'Army' in (r[1], r[2]) or 'Navy' in (r[1], r[2]))}"
    )
    known_neutral = [
        r for r in flipped_rows
        if r[3] == 0
        and any(x in (r[11] or "").lower() for x in ("metlife", "jerry", "at&t", "lucas oil", "mercedes"))
    ]
    print(f"  Known-neutral venue but neutral_site=0: {len(known_neutral)}")

    ml_agree_orig = ml_agree_flip = 0
    same_sign_margin = 0
    scored = 0
    for r in ref:
        _, away, home, neut, hml, aml, orig, sh, hp, ap, is_ref, venue, aligned = r
        hf_ml = hml < aml
        if hf_ml == (orig < 0):
            ml_agree_orig += 1
        if hf_ml == (aligned < 0):
            ml_agree_flip += 1
        if hp is not None and ap is not None:
            margin = hp - ap
            scored += 1
            if margin != 0 and sh != 0 and (margin > 0) == (sh > 0):
                same_sign_margin += 1

    print(
        f"Reference flips: ML agrees with orig betting spread {ml_agree_orig}/{len(ref)}; "
        f"ML agrees after flip {ml_agree_flip}/{len(ref)}"
    )
    print(
        f"Reference flips with final scores: spread_home same sign as margin "
        f"{same_sign_margin}/{scored} (expected ~50% — favorites lose often)"
    )
    print("\nAll reference-set ML flips:")
    for r in ref:
        gid, away, home, neut, hml, aml, orig, sh, hp, ap, _, venue, aligned = r
        margin = (hp - ap) if hp is not None and ap is not None else None
        print(
            f"  {gid} {away} @ {home} neut={neut} ml={hml}/{aml} "
            f"orig={orig} aligned={aligned} spread_home={sh} "
            f"score={hp}-{ap} margin={margin} venue={venue[:40] if venue else ''}"
        )
    if ml_agree_flip == len(ref) and ml_agree_orig == 0:
        print("\nVerdict: KEEP ML alignment — all reference flips match moneyline favorite.")
    elif ml_agree_orig > ml_agree_flip:
        print("\nVerdict: REVERT ML heuristic — original spread matches ML better.")


def _load_eval_rows(conn: sqlite3.Connection, constants: ModelConstants) -> list[dict]:
    rows = conn.execute(
        f"""
        SELECT g.season, g.week, {ACTUAL_MARGIN_SQL}, l.spread_home,
               rh.rating, ra.rating, g.neutral_site
        FROM cfb_games g
        JOIN cfb_lines l ON l.game_id = g.game_id AND l.is_backtest_reference = 1
        JOIN cfb_team_ratings_pit rh
          ON rh.season = g.season AND rh.week = g.week AND rh.team = g.home_team
        JOIN cfb_team_ratings_pit ra
          ON ra.season = g.season AND ra.week = g.week AND ra.team = g.away_team
        WHERE lower(coalesce(g.home_division,'')) = 'fbs'
          AND lower(coalesce(g.away_division,'')) = 'fbs'
          AND g.home_points IS NOT NULL AND l.spread_home IS NOT NULL
          AND g.week >= ?
        """,
        (MIN_TEST_WEEK,),
    ).fetchall()
    out = []
    for season, week, margin, sh, hr, ar, neut in rows:
        out.append(
            {
                "season": season,
                "actual_margin": margin,
                "spread_home": sh,
                "projected_margin": projected_margin(hr, ar, bool(neut), constants),
            }
        )
    return out


def _ats(rows: list[dict], thr: float) -> tuple[float, int]:
    wins = n = 0
    for r in rows:
        edge = r["projected_margin"] - r["spread_home"]
        if abs(edge) < thr:
            continue
        n += 1
        pick_home = edge > 0
        if pick_home:
            if r["actual_margin"] > r["spread_home"]:
                wins += 1
            elif r["actual_margin"] == r["spread_home"]:
                n -= 1
        else:
            if r["actual_margin"] < r["spread_home"]:
                wins += 1
            elif r["actual_margin"] == r["spread_home"]:
                n -= 1
    return (wins / n if n else float("nan")), n


def section4_gate(conn: sqlite3.Connection, constants: ModelConstants) -> None:
    print("\n" + "=" * 72)
    print(f"SECTION 4 — Walk-forward gate (weeks {MIN_TEST_WEEK}+, signs aligned)")
    print("=" * 72)
    print("Regression: actual_margin ~ b1*spread_home + b2*projected_margin")
    all_rows = _load_eval_rows(conn, constants)
    prior = {s for s in range(2015, 2026) if s != 2020}
    pooled: list[dict] = []

    for test_season in TEST_SEASONS:
        train = [r for r in all_rows if r["season"] in prior and r["season"] < test_season]
        test = [r for r in all_rows if r["season"] == test_season]
        pooled.extend(test)
        if len(train) < 50 or len(test) < 20:
            continue
        y_tr = np.array([r["actual_margin"] for r in train])
        fit = ols_with_inference(
            y_tr,
            [
                np.array([r["spread_home"] for r in train]),
                np.array([r["projected_margin"] for r in train]),
            ],
        )
        y_te = np.array([r["actual_margin"] for r in test])
        x1 = np.array([r["spread_home"] for r in test])
        x2 = np.array([r["projected_margin"] for r in test])
        sigma = float(np.std(y_te - x1, ddof=1))
        print(f"\n--- {test_season} (train={len(train)}, test={len(test)}) ---")
        print(
            f"  b1={fit['beta'][1]:+.3f} (SE={fit['se'][1]:.3f}, p={fit['p'][1]:.4f})  "
            f"b2={fit['beta'][2]:+.3f} (SE={fit['se'][2]:.3f}, p={fit['p'][2]:.4f})  "
            f"R²={fit['r2']:.4f}"
        )
        print(
            f"  RMSE proj={float(np.sqrt(np.mean((y_te-x2)**2))):.2f}  "
            f"RMSE spread={float(np.sqrt(np.mean((y_te-x1)**2))):.2f}  sigma={sigma:.2f}"
        )
        for thr in (1, 2, 3, 4, 5):
            rate, cnt = _ats(test, thr)
            print(f"  ATS edge≥{thr}: {rate:.1%} ({cnt})")

    if pooled:
        train_all = [r for r in all_rows if r["season"] < min(TEST_SEASONS)]
        fit = ols_with_inference(
            np.array([r["actual_margin"] for r in train_all]),
            [
                np.array([r["spread_home"] for r in train_all]),
                np.array([r["projected_margin"] for r in train_all]),
            ],
        )
        y = np.array([r["actual_margin"] for r in pooled])
        x1 = np.array([r["spread_home"] for r in pooled])
        x2 = np.array([r["projected_margin"] for r in pooled])
        print(f"\n--- Pooled 2021–2025 (n={len(pooled)}) ---")
        print(
            f"  b1={fit['beta'][1]:+.3f} (p={fit['p'][1]:.4f})  "
            f"b2={fit['beta'][2]:+.3f} (p={fit['p'][2]:.4f})  "
            f"sigma={float(np.std(y-x1, ddof=1)):.2f}"
        )
        b1_ok = 0.85 <= fit["beta"][1] <= 1.15
        sigma_ok = 12.0 <= float(np.std(y - x1, ddof=1)) <= 15.0
        print(f"  b1 near +1.0: {'YES' if b1_ok else 'NO'}  |  sigma near 13.5: {'YES' if sigma_ok else 'NO'}")
        if not b1_ok:
            print("  STOP — b1 still far from +1.0 after alignment.")


def main() -> None:
    conn = sqlite3.connect(ESPN_DB_PATH)
    ensure_cfb_schema(conn)
    populate_spread_home(conn)
    build_pit_ratings(conn)
    constants = estimate_model_constants(conn)
    section1_raw_values(conn, constants)
    section2_sigma(conn)
    section3_ml_flips(conn)
    section4_gate(conn, constants)
    conn.close()


if __name__ == "__main__":
    main()
