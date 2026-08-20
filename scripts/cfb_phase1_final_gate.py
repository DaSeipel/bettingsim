#!/usr/bin/env python3
"""CFB Phase 1 final gate tests — pooled regression, opener test, subsets, rating quality."""

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
from engine.cfb_ratings import build_pit_ratings, estimate_model_constants, projected_margin
from engine.cfb_schema import ensure_cfb_schema

TEST_SEASONS = (2021, 2022, 2023, 2024, 2025)
MIN_TEST_WEEK = 4

G5_CONFERENCES = frozenset({
    "American Athletic",
    "Conference USA",
    "Mid-American",
    "Mountain West",
    "Sun Belt",
})

P5_CONFERENCES = frozenset({
    "SEC",
    "Big Ten",
    "Big 12",
    "ACC",
    "Pac-12",
    "FBS Independents",
})


def ols_plain(y: np.ndarray, x_cols: list[np.ndarray]) -> dict:
    """Plain OLS with homoskedastic standard errors."""
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
    names = ["intercept", "spread", "projected"][:k]
    return {
        "beta": beta,
        "se": se,
        "t": t_stats,
        "p": p_vals,
        "r2": 1 - ss_res / ss_tot if ss_tot else 0.0,
        "n": n,
        "dof": dof,
    }


def _spread_open_margin(spread_open: float | None, home_ml, away_ml) -> float | None:
    betting = align_betting_spread_to_home(spread_open, home_ml, away_ml)
    return betting_spread_to_line_margin(betting)


def load_eval_rows(conn: sqlite3.Connection, constants) -> list[dict]:
    """FBS-vs-FBS reference closing-line eval rows, weeks 4+."""
    rows = conn.execute(
        """
        SELECT g.game_id, g.season, g.week, g.home_points, g.away_points,
               g.neutral_site, g.conference_game,
               g.home_conference, g.away_conference,
               l.spread_home, rh.rating, ra.rating,
               b.spread_open, b.spread AS bovada_close, b.home_moneyline, b.away_moneyline,
               b.opener_suspect
        FROM cfb_games g
        JOIN cfb_lines l ON l.game_id = g.game_id AND l.is_backtest_reference = 1
        JOIN cfb_team_ratings_pit rh
          ON rh.season = g.season AND rh.week = g.week AND rh.team = g.home_team
        JOIN cfb_team_ratings_pit ra
          ON ra.season = g.season AND ra.week = g.week AND ra.team = g.away_team
        LEFT JOIN cfb_lines b ON b.game_id = g.game_id AND b.provider = 'Bovada'
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
        margin = r[3] - r[4]
        proj = projected_margin(r[10], r[11], bool(r[5]), constants)
        spread_open = _spread_open_margin(r[12], r[14], r[15])
        spread_close_bovada = betting_spread_to_line_margin(
            align_betting_spread_to_home(r[13], r[14], r[15])
        )
        hc, ac = r[7] or "", r[8] or ""
        out.append(
            {
                "game_id": r[0],
                "season": r[1],
                "week": r[2],
                "actual_margin": margin,
                "spread_home": r[9],
                "projected_margin": proj,
                "spread_open": spread_open,
                "spread_close_bovada": spread_close_bovada,
                "conference_game": bool(r[6]),
                "home_conference": hc,
                "away_conference": ac,
                "opener_suspect": bool(r[16]) if r[16] is not None else False,
                "both_g5": hc in G5_CONFERENCES and ac in G5_CONFERENCES,
                "any_g5": hc in G5_CONFERENCES or ac in G5_CONFERENCES,
                "both_p5": hc in P5_CONFERENCES and ac in P5_CONFERENCES,
            }
        )
    return out


def _fmt_ols(fit: dict, label_spread: str = "spread") -> str:
    b0, b1, b2 = fit["beta"]
    lines = [
        f"  n={fit['n']}, R²={fit['r2']:.4f}, dof={fit['dof']}",
        f"  intercept={b0:+.3f} (SE={fit['se'][0]:.3f}, t={fit['t'][0]:+.2f}, p={fit['p'][0]:.4f})",
        f"  b1({label_spread})={b1:+.3f} (SE={fit['se'][1]:.3f}, t={fit['t'][1]:+.2f}, p={fit['p'][1]:.4f})",
        f"  b2(projected)={b2:+.3f} (SE={fit['se'][2]:.3f}, t={fit['t'][2]:+.2f}, p={fit['p'][2]:.4f})",
    ]
    return "\n".join(lines)


def section1_pooled_closing(rows: list[dict]) -> None:
    print("=" * 72)
    print("SECTION 1 — Pooled regression on 2,995 walk-forward test games (closing)")
    print("=" * 72)
    test = [r for r in rows if r["season"] in TEST_SEASONS]
    print(f"Test pool size: {len(test)} (expected 2,995)")
    print("Method: plain OLS (homoskedastic SEs). NOT cluster-robust — 5 season clusters")
    print("        cannot support reliable cluster inference.")
    print("Regression: actual_margin ~ b1*spread_home + b2*projected_margin\n")

    y = np.array([r["actual_margin"] for r in test])
    x1 = np.array([r["spread_home"] for r in test])
    x2 = np.array([r["projected_margin"] for r in test])
    fit = ols_plain(y, [x1, x2])
    print(_fmt_ols(fit, "spread_home"))

    corr = float(np.corrcoef(x1, x2)[0, 1])
    print(f"\nCorrelation(spread_home, projected_margin) = {corr:.4f}")
    if corr > 0.95:
        print("  WARNING: collinearity > 0.95 may inflate SE on b2.")
    else:
        print("  Collinearity is moderate; SE on b2 is not obviously inflated by r alone.")

    # Contrast: prior walk-forward TRAIN pooled (pre-2021 train coef applied to test)
    train = [r for r in rows if r["season"] < min(TEST_SEASONS)]
    fit_train = ols_plain(
        np.array([r["actual_margin"] for r in train]),
        [
            np.array([r["spread_home"] for r in train]),
            np.array([r["projected_margin"] for r in train]),
        ],
    )
    print(
        f"\nFor comparison — OLS on pre-2021 training pool (n={fit_train['n']}), "
        "which is what earlier reports labeled 'pooled train':"
    )
    print(
        f"  b1={fit_train['beta'][1]:+.3f} (p={fit_train['p'][1]:.4f})  "
        f"b2={fit_train['beta'][2]:+.3f} (p={fit_train['p'][2]:.4f})  R²={fit_train['r2']:.4f}"
    )
    print(
        f"Fresh test-pool b2={fit['beta'][2]:+.3f} vs train-pool b2={fit_train['beta'][2]:+.3f} "
        f"— {'identical rounding' if abs(fit['beta'][2]-fit_train['beta'][2])<0.001 else 'different pools'}"
    )


def _ats_open(row: dict, pick_home: bool) -> bool | None:
    m, line = row["actual_margin"], row["spread_open"]
    if m == line:
        return None
    return m > line if pick_home else m < line


def section2_opener(rows: list[dict], constants) -> None:
    print("\n" + "=" * 72)
    print("SECTION 2 — Opener test (Bovada spread_open, weeks 4+, opener_suspect excluded)")
    print("=" * 72)
    eligible = [
        r
        for r in rows
        if r["season"] in TEST_SEASONS
        and r["spread_open"] is not None
        and not r["opener_suspect"]
    ]
    print(f"Eligible opener games: {len(eligible)}")

    prior = {s for s in range(2015, 2026) if s != 2020}
    pooled_open: list[dict] = []

    for season in TEST_SEASONS:
        train = [
            r
            for r in eligible
            if r["season"] in prior and r["season"] < season
        ]
        test = [r for r in eligible if r["season"] == season]
        pooled_open.extend(test)
        if len(train) < 30 or len(test) < 10:
            print(f"\n{season}: skipped (train={len(train)}, test={len(test)})")
            continue
        fit = ols_plain(
            np.array([r["actual_margin"] for r in train]),
            [
                np.array([r["spread_open"] for r in train]),
                np.array([r["projected_margin"] for r in train]),
            ],
        )
        y_te = np.array([r["actual_margin"] for r in test])
        x1 = np.array([r["spread_open"] for r in test])
        x2 = np.array([r["projected_margin"] for r in test])
        sigma = float(np.std(y_te - x1, ddof=1))
        print(f"\n--- {season} (train={len(train)}, test={len(test)}) walk-forward fit ---")
        print(_fmt_ols(fit, "spread_open"))
        print(
            f"  Test RMSE proj={float(np.sqrt(np.mean((y_te-x2)**2))):.2f}  "
            f"RMSE open={float(np.sqrt(np.mean((y_te-x1)**2))):.2f}  sigma={sigma:.2f}"
        )

    print(f"\n--- Pooled opener test (n={len(pooled_open)}) plain OLS on test pool ---")
    y = np.array([r["actual_margin"] for r in pooled_open])
    x1 = np.array([r["spread_open"] for r in pooled_open])
    x2 = np.array([r["projected_margin"] for r in pooled_open])
    fit = ols_plain(y, [x1, x2])
    print(_fmt_ols(fit, "spread_open"))
    print(
        f"  RMSE proj={float(np.sqrt(np.mean((y-x2)**2))):.2f}  "
        f"RMSE open={float(np.sqrt(np.mean((y-x1)**2))):.2f}  "
        f"sigma={float(np.std(y-x1, ddof=1)):.2f}"
    )

    # Economic buckets
    print("\n--- Economic buckets (edge_open = projected_margin - spread_open) ---")
    print(f"{'bucket':<10} {'n':>5} {'ATS W-L':>10} {'win%':>7} {'win% SE':>8} {'ROI@-110':>9} {'CLV%toward':>11}")
    buckets = [(0, 3), (3, 6), (6, 10), (10, 15), (15, 999)]
    for lo, hi in buckets:
        sub = [
            r
            for r in pooled_open
            if lo <= abs(r["projected_margin"] - r["spread_open"]) < hi
            and r["spread_close_bovada"] is not None
        ]
        wins = losses = pushes = 0
        toward = clv_n = 0
        for r in sub:
            edge = r["projected_margin"] - r["spread_open"]
            if abs(edge) < 0.01:
                continue
            pick_home = edge > 0
            result = _ats_open(r, pick_home)
            if result is None:
                pushes += 1
            elif result:
                wins += 1
            else:
                losses += 1
            close_move = r["spread_close_bovada"] - r["spread_open"]
            if edge > 0 and close_move > 0:
                toward += 1
                clv_n += 1
            elif edge < 0 and close_move < 0:
                toward += 1
                clv_n += 1
            elif edge != 0:
                clv_n += 1
        n_bets = wins + losses
        wp = wins / n_bets if n_bets else float("nan")
        wp_se = float(np.sqrt(wp * (1 - wp) / n_bets)) if n_bets else float("nan")
        roi = ((wins * (100 / 110) - losses) / n_bets) if n_bets else float("nan")
        clv_pct = toward / clv_n if clv_n else float("nan")
        label = f"{lo}-{hi}" if hi < 999 else "15+"
        print(
            f"{label:<10} {len(sub):>5} {wins}-{losses}-{pushes:>3} "
            f"{wp:>6.1%} {wp_se:>8.3f} {roi:>8.1%} {clv_pct:>10.1%}"
        )
    print("CLV%toward = share of games where closing line moved toward model side vs open.")


def section3_subsets(rows: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("SECTION 3 — Subset tests (closing lines, pooled test pool plain OLS)")
    print("=" * 72)
    test = [r for r in rows if r["season"] in TEST_SEASONS]

    def subset_b2(label: str, sub: list[dict]) -> None:
        if len(sub) < 30:
            print(f"\n{label}: n={len(sub)} — too few")
            return
        fit = ols_plain(
            np.array([r["actual_margin"] for r in sub]),
            [
                np.array([r["spread_home"] for r in sub]),
                np.array([r["projected_margin"] for r in sub]),
            ],
        )
        print(
            f"\n{label}: n={fit['n']}  b2={fit['beta'][2]:+.3f}  "
            f"SE={fit['se'][2]:.3f}  t={fit['t'][2]:+.2f}  p={fit['p'][2]:.4f}"
        )

    subset_b2("(a) Both G5", [r for r in test if r["both_g5"]])
    subset_b2("(b) At least one G5", [r for r in test if r["any_g5"]])
    subset_b2("(c) Both Power conference", [r for r in test if r["both_p5"]])
    subset_b2(
        "(d) Total line movement >= 2 pts (Bovada open→close)",
        [
            r
            for r in test
            if r["spread_open"] is not None
            and r["spread_close_bovada"] is not None
            and abs(r["spread_close_bovada"] - r["spread_open"]) >= 2
        ],
    )
    subset_b2("(e) Weeks 4-8", [r for r in test if MIN_TEST_WEEK <= r["week"] <= 8])
    subset_b2("(f) Weeks 9-14", [r for r in test if 9 <= r["week"] <= 14])
    subset_b2("(g) Non-conference", [r for r in test if not r["conference_game"]])
    subset_b2("(h) Conference games", [r for r in test if r["conference_game"]])


def section4_rating_quality(conn: sqlite3.Connection) -> None:
    print("\n" + "=" * 72)
    print("SECTION 4 — PIT rating quality vs SP+ end_of_season (validation only)")
    print("=" * 72)
    seasons = conn.execute(
        "SELECT DISTINCT season FROM cfb_ratings_sp WHERE rating_scope='end_of_season' ORDER BY season"
    ).fetchall()
    for (season,) in seasons:
        if season < 2015:
            continue
        pit = conn.execute(
            """
            SELECT p.team, p.rating
            FROM cfb_team_ratings_pit p
            JOIN (
                SELECT team, MAX(week) AS max_week
                FROM cfb_team_ratings_pit
                WHERE season = ?
                GROUP BY team
            ) m ON m.team = p.team AND m.max_week = p.week AND p.season = ?
            """,
            (season, season),
        ).fetchall()
        sp = conn.execute(
            """
            SELECT team, rating FROM cfb_ratings_sp
            WHERE season = ? AND rating_scope = 'end_of_season'
            """,
            (season,),
        ).fetchall()
        pit_map = {t: r for t, r in pit}
        sp_map = {t: r for t, r in sp}
        common = sorted(set(pit_map) & set(sp_map))
        if len(common) < 10:
            print(f"{season}: insufficient overlap ({len(common)} teams)")
            continue
        x = np.array([pit_map[t] for t in common])
        y = np.array([sp_map[t] for t in common])
        r = float(np.corrcoef(x, y)[0, 1])
        flag = "OK" if r >= 0.85 else "LOW"
        print(f"  {season}: r={r:.4f}  n={len(common)} teams  [{flag}]")


def main() -> None:
    conn = sqlite3.connect(ESPN_DB_PATH)
    ensure_cfb_schema(conn)
    populate_spread_home(conn)
    build_pit_ratings(conn)
    constants = estimate_model_constants(conn)
    rows = load_eval_rows(conn, constants)
    conn.close()

    section1_pooled_closing(rows)
    section2_opener(rows, constants)
    section3_subsets(rows)
    conn2 = sqlite3.connect(ESPN_DB_PATH)
    section4_rating_quality(conn2)
    conn2.close()


if __name__ == "__main__":
    main()
