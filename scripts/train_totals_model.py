#!/usr/bin/env python3
"""
Train XGBoost totals model on NCAAB training_data.csv with walk-forward CV.

Features: home_ADJ_T, away_ADJ_T, home_ADJOE, home_ADJDE, away_ADJOE, away_ADJDE,
tempo_diff, combined_tempo, combined_offense, combined_defense, is_neutral,
is_conference_tourney, is_ncaa_tourney.
Target: actual_total (= home_score + away_score).

Walk-forward folds (same as Step 14 / train_spread_model):
  Fold 1: train 2021-2022, test 2023
  Fold 2: train 2021-2023, test 2024
  Fold 3: train 2021-2024, test 2025

Prints MAE and RMSE per fold and average. If historical_games has closing O/U,
computes totals edge vs closing over/under; otherwise skips that comparison.
Saves model to data/models/xgboost_totals_ncaab.pkl.
"""
from __future__ import annotations

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import numpy as np
import pandas as pd

TRAINING_DATA_PATH = APP_ROOT / "data" / "ncaab" / "training_data.csv"
HISTORICAL_GAMES_PATH = APP_ROOT / "data" / "ncaab" / "historical_games.csv"
MODEL_PATH = APP_ROOT / "data" / "models" / "xgboost_totals_ncaab.pkl"

TOTALS_FEATURE_COLUMNS = [
    "home_ADJ_T", "away_ADJ_T",
    "home_ADJOE", "home_ADJDE", "away_ADJOE", "away_ADJDE",
    "tempo_diff", "combined_tempo", "combined_offense", "combined_defense",
    "is_neutral", "is_conference_tourney", "is_ncaa_tourney",
]
TARGET = "actual_total"
TOTALS_EDGE_THRESHOLD = 5.0  # Totals markets are tighter; need larger edge for meaningful value play
BIAS_OVERPREDICTION_THRESHOLD = 2.0  # If mean(pred) - mean(actual) > this, confirm over-prediction bias
DIRECTIONAL_BIAS_OVER_PCT = 65.0  # If % Overs among picks > this, directional bias

CV_FOLDS = [
    (2023, (2021, 2022)),
    (2024, (2021, 2022, 2023)),
    (2025, (2021, 2022, 2023, 2024)),
]


def _ensure_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add combined_tempo, combined_offense, combined_defense if missing."""
    df = df.copy()
    if "combined_tempo" not in df.columns and "home_ADJ_T" in df.columns and "away_ADJ_T" in df.columns:
        df["combined_tempo"] = df["home_ADJ_T"].fillna(0) + df["away_ADJ_T"].fillna(0)
    if "combined_offense" not in df.columns and "home_ADJOE" in df.columns and "away_ADJOE" in df.columns:
        df["combined_offense"] = df["home_ADJOE"].fillna(0) + df["away_ADJOE"].fillna(0)
    if "combined_defense" not in df.columns and "home_ADJDE" in df.columns and "away_ADJDE" in df.columns:
        df["combined_defense"] = df["home_ADJDE"].fillna(0) + df["away_ADJDE"].fillna(0)
    if "tempo_diff" not in df.columns and "home_ADJ_T" in df.columns and "away_ADJ_T" in df.columns:
        df["tempo_diff"] = df["home_ADJ_T"].fillna(0) - df["away_ADJ_T"].fillna(0)
    return df


def _merge_closing_total(df: pd.DataFrame) -> pd.DataFrame:
    """Merge closing over/under from historical_games if column exists."""
    if not HISTORICAL_GAMES_PATH.exists():
        return df
    try:
        games = pd.read_csv(HISTORICAL_GAMES_PATH)
    except Exception:
        return df
    ou_col = None
    for c in ("closing_total", "over_under", "closing_ou", "total"):
        if c in games.columns:
            ou_col = c
            break
    if ou_col is None:
        return df
    games = games[["date", "home_team", "away_team", ou_col]].copy()
    games = games.rename(columns={ou_col: "closing_total"})
    games["date"] = pd.to_datetime(games["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.copy()
    if "date" not in df.columns:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    merged = df.merge(
        games,
        on=["date", "home_team", "away_team"],
        how="left",
        suffixes=("", "_y"),
    )
    return merged


def _totals_edge_metrics(
    y_true: np.ndarray,
    pred_total: np.ndarray,
    closing_total: np.ndarray,
) -> tuple[float, float, int]:
    """Over/Under at TOTALS_EDGE_THRESHOLD: win%, ROI%, n_picks. Edge = pred - line; Over if edge > 3, Under if edge < -3."""
    mask = ~np.isnan(closing_total)
    y = y_true[mask]
    pred = pred_total[mask]
    line = closing_total[mask]
    if y.size == 0:
        return float("nan"), float("nan"), 0
    edge = pred - line
    pick_over = edge > TOTALS_EDGE_THRESHOLD
    pick_under = edge < -TOTALS_EDGE_THRESHOLD
    is_pick = pick_over | pick_under
    if not np.any(is_pick):
        return float("nan"), float("nan"), 0
    y = y[is_pick]
    line = line[is_pick]
    pick_over = pick_over[is_pick]
    over_hit = y > line
    under_hit = y < line
    won = np.where(pick_over, over_hit, under_hit)
    n_picks = won.size
    wins = won.sum()
    stake = 100.0
    win_profit = stake * (100.0 / 110.0)
    pl = np.where(won, win_profit, -stake)
    total_profit = pl.sum()
    total_staked = stake * n_picks
    win_pct = wins / n_picks if n_picks else float("nan")
    roi_pct = (total_profit / total_staked * 100.0) if total_staked else float("nan")
    return win_pct, roi_pct, int(n_picks)


def main() -> int:
    if not TRAINING_DATA_PATH.exists():
        print(f"Missing {TRAINING_DATA_PATH}. Run scripts/build_training_data.py first.", file=sys.stderr)
        return 1
    try:
        import xgboost as xgb
        import joblib
        from sklearn.metrics import mean_absolute_error, mean_squared_error
    except ImportError as e:
        print(f"Install dependencies: {e}", file=sys.stderr)
        return 1

    df = pd.read_csv(TRAINING_DATA_PATH)
    if "season" not in df.columns or TARGET not in df.columns:
        print(f"training_data.csv must have 'season' and '{TARGET}'.", file=sys.stderr)
        return 1

    df = _ensure_derived_columns(df)
    use_cols = [c for c in TOTALS_FEATURE_COLUMNS if c in df.columns]
    if len(use_cols) < len(TOTALS_FEATURE_COLUMNS):
        missing = set(TOTALS_FEATURE_COLUMNS) - set(use_cols)
        print(f"Missing feature columns: {missing}", file=sys.stderr)
        return 1

    df = _merge_closing_total(df)
    has_ou = "closing_total" in df.columns and df["closing_total"].notna().any()

    fold_metrics = []
    all_pred_test = []
    all_y_test = []
    all_over_count = []
    all_under_count = []
    print("Walk-forward CV (3 folds, same as Step 14)")
    print("=" * 60)
    for test_season, train_seasons in CV_FOLDS:
        train_df = df[df["season"].isin(train_seasons)].copy()
        test_df = df[df["season"] == test_season].copy()
        if train_df.empty or test_df.empty:
            print(f"Fold {train_seasons} -> {test_season}: SKIPPED (no data)")
            continue
        X_train = train_df[use_cols].fillna(0.0)
        y_train = train_df[TARGET].astype(float).values
        X_test = test_df[use_cols].fillna(0.0)
        y_test = test_df[TARGET].astype(float).values

        model = xgb.XGBRegressor(
            max_depth=4,
            n_estimators=300,
            learning_rate=0.05,
            random_state=42,
            early_stopping_rounds=20,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False,
        )
        pred_test = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred_test))

        # Diagnostics: mean pred vs mean actual; % Overs among hypothetical picks
        mean_pred = float(np.mean(pred_test))
        mean_actual = float(np.mean(y_test))
        bias_pts = mean_pred - mean_actual
        all_pred_test.append(pred_test)
        all_y_test.append(y_test)
        # Use test-set mean actual as proxy line for directional diagnostic when no closing_total
        line_proxy = np.mean(y_test)
        edge_proxy = pred_test - line_proxy
        over_picks = np.sum(edge_proxy > TOTALS_EDGE_THRESHOLD)
        under_picks = np.sum(edge_proxy < -TOTALS_EDGE_THRESHOLD)
        all_over_count.append(over_picks)
        all_under_count.append(under_picks)

        closing = test_df.get("closing_total")
        if has_ou and closing is not None:
            closing_arr = pd.to_numeric(closing, errors="coerce").to_numpy(dtype=float)
            win_pct, roi_pct, n_picks = _totals_edge_metrics(y_test, pred_test, closing_arr)
        else:
            win_pct, roi_pct, n_picks = float("nan"), float("nan"), 0

        print(f"Fold train {train_seasons} -> test {test_season}:")
        print(f"  MAE  = {mae:.4f}, RMSE = {rmse:.4f}")
        print(f"  Mean predicted = {mean_pred:.2f}, Mean actual = {mean_actual:.2f}  (bias = {bias_pts:+.2f} pts)")
        if over_picks + under_picks > 0:
            pct_over = 100.0 * over_picks / (over_picks + under_picks)
            print(f"  Picks @ |edge|>{TOTALS_EDGE_THRESHOLD}: {over_picks} Over, {under_picks} Under  ({pct_over:.1f}% Over)")
        if has_ou and n_picks > 0:
            print(f"  Totals vs closing O/U: n={n_picks}, win%={win_pct*100:.1f}%, ROI={roi_pct:.1f}%")
        elif has_ou:
            print(f"  Totals vs closing O/U: n=0 (no picks)")
        else:
            print("  Totals vs closing O/U: skipped (no closing total in historical_games)")

        fold_metrics.append({"mae": mae, "rmse": rmse})

    bias_correction = 0.0
    if fold_metrics and all_pred_test and all_y_test:
        maes = np.array([m["mae"] for m in fold_metrics])
        rmses = np.array([m["rmse"] for m in fold_metrics])
        print("\nCV summary across folds")
        print("=" * 60)
        print(f"MAE  mean={maes.mean():.4f}, std={maes.std(ddof=0):.4f}")
        print(f"RMSE mean={rmses.mean():.4f}, std={rmses.std(ddof=0):.4f}")

        # Aggregate bias and directional diagnostics across folds
        concat_pred = np.concatenate(all_pred_test)
        concat_actual = np.concatenate(all_y_test)
        mean_pred_overall = float(np.mean(concat_pred))
        mean_actual_overall = float(np.mean(concat_actual))
        bias_pts_overall = mean_pred_overall - mean_actual_overall
        total_over = sum(all_over_count)
        total_under = sum(all_under_count)
        total_picks = total_over + total_under
        pct_over_overall = (100.0 * total_over / total_picks) if total_picks > 0 else 0.0

        print("\nTotals bias diagnostics (test sets)")
        print("=" * 60)
        print(f"  Mean predicted total = {mean_pred_overall:.2f}, Mean actual total = {mean_actual_overall:.2f}")
        print(f"  Bias (pred - actual) = {bias_pts_overall:+.2f} pts")
        if bias_pts_overall > BIAS_OVERPREDICTION_THRESHOLD:
            print(f"  >>> Systematic over-prediction bias detected (>{BIAS_OVERPREDICTION_THRESHOLD} pts). Calibration will be applied.")
        print(f"  Picks @ |edge|>{TOTALS_EDGE_THRESHOLD}: {total_over} Over, {total_under} Under  ({pct_over_overall:.1f}% Over)")
        if total_picks > 0 and pct_over_overall > DIRECTIONAL_BIAS_OVER_PCT:
            print(f"  >>> Directional bias: {pct_over_overall:.1f}% Overs (>{DIRECTIONAL_BIAS_OVER_PCT}%). Bias correction will reduce mean prediction.")

        bias_correction = bias_pts_overall

    # Train final model on 2021-2025
    print("\nTraining final production model on 2021-2025...")
    final_df = df[df["season"].between(2021, 2025)].copy()
    X_final = final_df[use_cols].fillna(0.0)
    y_final = final_df[TARGET].astype(float).values

    final_model = xgb.XGBRegressor(
        max_depth=4,
        n_estimators=300,
        learning_rate=0.05,
        random_state=42,
        early_stopping_rounds=20,
    )
    final_model.fit(
        X_final,
        y_final,
        eval_set=[(X_final, y_final)],
        verbose=False,
    )

    pred_final = final_model.predict(X_final)
    if bias_correction != 0.0:
        pred_final_calibrated = pred_final - bias_correction
        mae_full = mean_absolute_error(y_final, pred_final_calibrated)
        rmse_full = np.sqrt(mean_squared_error(y_final, pred_final_calibrated))
        print(f"  (After bias correction: {bias_correction:+.2f} pts)")
    else:
        mae_full = mean_absolute_error(y_final, pred_final)
        rmse_full = np.sqrt(mean_squared_error(y_final, pred_final))
    print("\nFull-data metrics (2021-2025, in-sample):")
    print(f"  MAE  = {mae_full:.4f}")
    print(f"  RMSE = {rmse_full:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": final_model,
        "feature_columns": use_cols,
        "target": TARGET,
        "bias_correction": bias_correction,
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"\nSaved production model to {MODEL_PATH}" + (f" (bias_correction={bias_correction:+.2f})" if bias_correction != 0.0 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
