#!/usr/bin/env python3
"""
Backtest NCAAB spread strategies on 2025 season.

Replays 2025 season game-by-game: for each game with closing spread and features,
computes predictions (BARTHAG*50 and XGBoost), edges vs closing spread, and
simulates $100 flat bets at -110 on value plays. Uses team stats from
training_data (season-level stats merged per game; no rolling point-in-time stats).
Compares:
  (1) BARTHAG*50 model, 3-pt threshold
  (2) XGBoost model, 3-pt threshold
  (3) XGBoost model, calibrated threshold (Step 6; configurable CALIBRATED_THRESHOLD)

Outputs: summary table (picks, win%, ROI, max drawdown, Sharpe-equivalent),
detailed results to data/backtest_results.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import numpy as np
import pandas as pd

from engine.utils import game_season_from_date

DATA_DIR = APP_ROOT / "data"
NCAAB_DIR = DATA_DIR / "ncaab"
MODELS_DIR = DATA_DIR / "models"
HISTORICAL_GAMES_PATH = NCAAB_DIR / "historical_games.csv"
TRAINING_DATA_PATH = NCAAB_DIR / "training_data.csv"
MODEL_PATH = MODELS_DIR / "xgboost_spread_ncaab.pkl"
OUTPUT_CSV = DATA_DIR / "backtest_results.csv"

# Strategy config
BARTHAG_SCALE = 50.0
NEUTRAL_ADJUSTMENT = 3.0
THRESHOLD_3PT = 3.0
# Calibrated threshold from Step 6 (tune to maximize ROI on holdout if available)
CALIBRATED_THRESHOLD = 2.5
STAKE_USD = 100.0
ODDS_AMERICAN = -110  # win 100/110 * stake
TEST_SEASON = 2025


def _load_2025_games() -> pd.DataFrame:
    """Load 2025 season games with closing spread; merge training_data features. One row per game."""
    if not HISTORICAL_GAMES_PATH.exists():
        raise FileNotFoundError(f"Missing {HISTORICAL_GAMES_PATH}")
    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {TRAINING_DATA_PATH}. Run scripts/build_training_data.py first.")

    games = pd.read_csv(HISTORICAL_GAMES_PATH)
    games["date"] = pd.to_datetime(games["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    games = games.dropna(subset=["date", "home_team", "away_team"])
    games["season"] = games["date"].apply(lambda d: game_season_from_date(d))
    games = games[games["season"] == TEST_SEASON].copy()
    games = games.dropna(subset=["closing_spread"])
    games["closing_spread"] = pd.to_numeric(games["closing_spread"], errors="coerce")
    games = games.dropna(subset=["closing_spread"])
    if "margin" not in games.columns and "home_score" in games.columns and "away_score" in games.columns:
        games["margin"] = games["home_score"].astype(float) - games["away_score"].astype(float)
    games = games.dropna(subset=["margin"])

    train = pd.read_csv(TRAINING_DATA_PATH)
    train["date"] = pd.to_datetime(train["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    train = train[train["season"] == TEST_SEASON].copy()
    # Merge on date + home_team + away_team (training_data has actual_margin; we keep closing_spread from games)
    merge_cols = ["date", "home_team", "away_team"]
    for c in merge_cols:
        if c not in train.columns or c not in games.columns:
            return pd.DataFrame()
    # Keep games columns: date, home_team, away_team, margin, closing_spread; add feature columns from train
    train_sub = train.drop(columns=["actual_margin"], errors="ignore")
    merged = games.merge(
        train_sub,
        on=merge_cols,
        how="inner",
        suffixes=("", "_y"),
    )
    # Drop duplicate columns from merge
    merged = merged[[c for c in merged.columns if not c.endswith("_y")]]
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def _edge_pts(pred_margin: float, closing_spread: float) -> float:
    """Edge in points. closing_spread = home spread (negative = home favored). Same convention as predict_games."""
    return pred_margin + float(closing_spread)


def _barthag_pred(row: pd.Series) -> float | None:
    """Predicted margin (home - away) from BARTHAG*scale; subtract NEUTRAL_ADJUSTMENT if neutral."""
    h = row.get("home_BARTHAG")
    a = row.get("away_BARTHAG")
    if pd.isna(h) or pd.isna(a):
        return None
    try:
        pred = (float(h) - float(a)) * BARTHAG_SCALE
    except (TypeError, ValueError):
        return None
    if row.get("is_neutral") in (1, True, "1", "yes") or (isinstance(row.get("is_neutral"), (int, float)) and row.get("is_neutral") not in (0, 0.0)):
        pred -= NEUTRAL_ADJUSTMENT
    return pred


def _xgboost_pred(row: pd.Series, model, feature_columns: list[str], calibration_shift: float) -> float | None:
    """Predicted margin from XGBoost model. Returns None if features missing."""
    try:
        vals = [float(row.get(c, 0) if pd.notna(row.get(c)) else 0.0) for c in feature_columns]
    except (TypeError, ValueError, KeyError):
        return None
    X_arr = np.array(vals, dtype=np.float64).reshape(1, -1)
    raw = float(model.predict(X_arr)[0])
    return raw - calibration_shift


def _value_play(edge: float, threshold: float) -> bool:
    return abs(edge) > threshold


def _pick_side(edge: float, threshold: float) -> str | None:
    """Return 'Home' or 'Away' if value play, else None."""
    if not _value_play(edge, threshold):
        return None
    return "Home" if edge > threshold else "Away"


def _covered_home(actual_margin: float, closing_spread: float) -> bool:
    """True if home covered. closing_spread is home spread (e.g. -7 = home favored by 7); home covers if margin > 7."""
    return float(actual_margin) > -float(closing_spread)


def _covered_away(actual_margin: float, closing_spread: float) -> bool:
    """True if away covered (margin < -closing_spread)."""
    return float(actual_margin) < -float(closing_spread)


def _won(pick: str, actual_margin: float, closing_spread: float) -> bool:
    if pick == "Home":
        return _covered_home(actual_margin, closing_spread)
    return _covered_away(actual_margin, closing_spread)


def _profit(won: bool) -> float:
    if won:
        return STAKE_USD * (100.0 / abs(ODDS_AMERICAN))
    return -STAKE_USD


def run_backtest() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Load 2025 games, run three strategies, return (games_df, summary_df, detail_rows)."""
    df = _load_2025_games()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    # Load XGBoost model
    import joblib
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Run scripts/train_spread_model.py first.")
    payload = joblib.load(MODEL_PATH)
    model = payload["model"]
    feature_columns = payload.get("feature_columns", [])
    calibration_shift = payload.get("margin_calibration_shift", 0.0)
    # Ensure we only use columns present in df
    feature_columns = [c for c in feature_columns if c in df.columns]
    if not feature_columns:
        raise ValueError("No XGBoost feature columns found in training data.")

    # Ensure numeric
    for c in feature_columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    detail_rows = []
    strategies = [
        ("BARTHAG_3pt", "barthag", THRESHOLD_3PT),
        ("XGBoost_3pt", "xgb", THRESHOLD_3PT),
        ("XGBoost_calibrated", "xgb", CALIBRATED_THRESHOLD),
    ]

    for strat_name, model_type, thresh in strategies:
        cum_pl = 0.0
        peak = 0.0
        returns_list = []

        for idx, row in df.iterrows():
            date = row["date"]
            home = row["home_team"]
            away = row["away_team"]
            spread = float(row["closing_spread"])
            actual_margin = float(row["margin"])

            if model_type == "barthag":
                pred = _barthag_pred(row)
            else:
                pred = _xgboost_pred(row, model, feature_columns, calibration_shift)

            if pred is None:
                continue

            edge = _edge_pts(pred, spread)
            pick = _pick_side(edge, thresh)
            if pick is None:
                continue

            won = _won(pick, actual_margin, spread)
            pl = _profit(won)
            cum_pl += pl
            peak = max(peak, cum_pl)
            drawdown = peak - cum_pl
            ret = pl / STAKE_USD
            returns_list.append(ret)

            detail_rows.append({
                "strategy": strat_name,
                "date": date,
                "home_team": home,
                "away_team": away,
                "closing_spread": spread,
                "actual_margin": actual_margin,
                "pred_margin": pred,
                "edge_pts": round(edge, 2),
                "pick": pick,
                "won": won,
                "profit": round(pl, 2),
                "cumulative_pl": round(cum_pl, 2),
                "drawdown": round(drawdown, 2),
            })

    # Build summary per strategy
    summary_rows = []
    for strat_name, _, thresh in strategies:
        rows = [r for r in detail_rows if r["strategy"] == strat_name]
        n = len(rows)
        if n == 0:
            summary_rows.append({
                "strategy": strat_name,
                "total_picks": 0,
                "wins": 0,
                "win_pct": None,
                "total_staked": 0.0,
                "total_profit": 0.0,
                "roi_pct": None,
                "max_drawdown": 0.0,
                "sharpe_equiv": None,
            })
            continue
        wins = sum(1 for r in rows if r["won"])
        total_staked = n * STAKE_USD
        total_profit = sum(r["profit"] for r in rows)
        roi = (total_profit / total_staked * 100) if total_staked else 0.0
        max_dd = max(r["drawdown"] for r in rows) if rows else 0.0
        returns = [r["profit"] / STAKE_USD for r in rows]
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret and std_ret > 1e-10:
            sharpe_equiv = (mean_ret / std_ret) * np.sqrt(len(returns))
        else:
            sharpe_equiv = 0.0 if mean_ret == 0 else (np.inf if mean_ret > 0 else -np.inf)

        summary_rows.append({
            "strategy": strat_name,
            "total_picks": n,
            "wins": wins,
            "win_pct": round(100.0 * wins / n, 1),
            "total_staked": total_staked,
            "total_profit": round(total_profit, 2),
            "roi_pct": round(roi, 1),
            "max_drawdown": round(max_dd, 2),
            "sharpe_equiv": round(sharpe_equiv, 3),
        })

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)
    return df, summary_df, detail_rows


def main() -> int:
    print("NCAAB spread backtest — 2025 season")
    print("Strategies: (1) BARTHAG*50 @ 3pt (2) XGBoost @ 3pt (3) XGBoost @ calibrated threshold")
    print()

    try:
        games_df, summary_df, detail_rows = run_backtest()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if games_df.empty:
        print("No 2025 games with closing spread and features. Check historical_games.csv and training_data.csv.")
        return 1

    print(f"Games in backtest: {len(games_df)} (2025 season, with closing spread + features)\n")
    print("Summary")
    print("=" * 90)
    print(f"{'Strategy':<22} | {'Picks':>6} | {'Win%':>6} | {'ROI%':>7} | {'MaxDD':>8} | {'Sharpe':>8}")
    print("=" * 90)
    for _, r in summary_df.iterrows():
        wp = f"{r['win_pct']:.1f}" if r["win_pct"] is not None else "—"
        roi = f"{r['roi_pct']:.1f}" if r["roi_pct"] is not None else "—"
        sh = f"{r['sharpe_equiv']:.3f}" if r["sharpe_equiv"] is not None else "—"
        print(f"{r['strategy']:<22} | {r['total_picks']:>6} | {wp:>6} | {roi:>7} | {r['max_drawdown']:>8.2f} | {sh:>8}")
    print("=" * 90)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if detail_rows:
        pd.DataFrame(detail_rows).to_csv(OUTPUT_CSV, index=False)
        print(f"\nDetailed results saved to {OUTPUT_CSV} ({len(detail_rows)} rows)")
    else:
        print("\nNo value-play rows to save.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
