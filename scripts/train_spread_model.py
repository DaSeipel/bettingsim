#!/usr/bin/env python3
"""
Train XGBoost spread model on NCAAB training_data.csv with walk-forward CV.

Features: raw stats, diffs, home_hca, is_neutral, form (last 5/10 margin, win%, diffs),
tournament flags, seed_diff, and days_rest.

Walk-forward folds:
  Fold 1: train 2021-2022, test 2023
  Fold 2: train 2021-2023, test 2024
  Fold 3: train 2021-2024, test 2025

For each fold, prints MAE/RMSE and ATS performance at a 3-pt edge threshold using
closing_spread from historical_games.csv. Also prints average metrics and flags
low-importance features. Finally trains a production model on 2021-2025 and saves to
data/models/xgboost_spread_ncaab.pkl, reporting full-data metrics separately.
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
MODEL_PATH = APP_ROOT / "data" / "models" / "xgboost_spread_ncaab.pkl"

STAT_COLS = [
    "ADJOE", "ADJDE", "BARTHAG", "ADJ_T", "EFG_O", "EFG_D",
    "TOR", "ORB", "FTR", "THREE_PT_RATE",
]
FORM_COLS = [
    "home_last5_margin", "home_last10_margin", "home_last5_winpct",
    "away_last5_margin", "away_last10_margin", "away_last5_winpct",
    "last5_margin_diff", "last10_margin_diff", "last5_winpct_diff",
]
FEATURE_COLUMNS = (
    [f"home_{c}" for c in STAT_COLS]
    + [f"away_{c}" for c in STAT_COLS]
    + ["BARTHAG_diff", "ADJOE_diff", "ADJDE_diff", "tempo_diff", "home_hca", "is_neutral"]
    + FORM_COLS
    + ["is_conference_tourney", "is_ncaa_tourney", "seed_diff", "days_rest_home", "days_rest_away"]
)
TARGET = "actual_margin"
TRAIN_SEASONS = (2021, 2022, 2023, 2024)
TEST_SEASON = 2025
BARTHAG_BASELINE_SCALE = 50.0
EDGE_THRESHOLD_ATS = 3.0


def _merge_closing_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Merge closing_spread from historical_games.csv into training_data on (date, home_team, away_team)."""
    if not HISTORICAL_GAMES_PATH.exists():
        return df
    try:
        games = pd.read_csv(HISTORICAL_GAMES_PATH)
    except Exception:
        return df
    if "closing_spread" not in games.columns:
        return df
    games = games[["date", "home_team", "away_team", "closing_spread"]].copy()
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


def _ats_metrics(y_true: np.ndarray, pred_margin: np.ndarray, closing_spread: np.ndarray) -> tuple[float, float, int]:
    """Compute ATS win% and ROI at EDGE_THRESHOLD_ATS with -110, $100 flat bets. Returns (win_pct, roi_pct, n_picks)."""
    mask = ~np.isnan(closing_spread)
    y = y_true[mask]
    pred = pred_margin[mask]
    spread = closing_spread[mask]
    if y.size == 0:
        return float("nan"), float("nan"), 0
    edge = pred + spread  # home edge in points
    pick_home = edge > EDGE_THRESHOLD_ATS
    pick_away = edge < -EDGE_THRESHOLD_ATS
    is_pick = pick_home | pick_away
    if not np.any(is_pick):
        return float("nan"), float("nan"), 0
    y = y[is_pick]
    spread = spread[is_pick]
    pick_home = pick_home[is_pick]
    # Home covers if margin > -spread; away covers if margin < -spread
    home_cover = y > -spread
    away_cover = y < -spread
    won = np.where(pick_home, home_cover, away_cover)
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


def main() -> None:
    if not TRAINING_DATA_PATH.exists():
        print(f"Missing {TRAINING_DATA_PATH}. Run scripts/build_training_data.py first.", file=sys.stderr)
        sys.exit(1)
    try:
        import xgboost as xgb
        import joblib
        from sklearn.metrics import mean_absolute_error, mean_squared_error
    except ImportError as e:
        print(f"Install dependencies: {e}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(TRAINING_DATA_PATH)
    if "season" not in df.columns or TARGET not in df.columns:
        print("training_data.csv must have 'season' and 'actual_margin'.", file=sys.stderr)
        sys.exit(1)

    use_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not use_cols:
        print("No feature columns found in CSV.", file=sys.stderr)
        sys.exit(1)

    # Merge closing_spread for ATS backtest
    df = _merge_closing_spread(df)

    cv_folds = [
        (2023, (2021, 2022)),
        (2024, (2021, 2022, 2023)),
        (2025, (2021, 2022, 2023, 2024)),
    ]
    fold_metrics = []
    feature_importances_per_fold: list[np.ndarray] = []

    print("Walk-forward CV (3 folds)")
    print("=" * 60)
    for test_season, train_seasons in cv_folds:
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

        closing = test_df.get("closing_spread")
        if closing is not None:
            closing_arr = pd.to_numeric(closing, errors="coerce").to_numpy(dtype=float)
        else:
            closing_arr = np.full_like(y_test, np.nan, dtype=float)
        win_pct, roi_pct, n_picks = _ats_metrics(y_test, pred_test, closing_arr)

        print(f"Fold train {train_seasons} -> test {test_season}:")
        print(f"  MAE  = {mae:.4f}, RMSE = {rmse:.4f}")
        if n_picks > 0:
            print(f"  ATS @ {EDGE_THRESHOLD_ATS:.1f} pts: n={n_picks}, win%={win_pct*100:.1f}%, ROI={roi_pct:.1f}%")
        else:
            print(f"  ATS @ {EDGE_THRESHOLD_ATS:.1f} pts: n=0 (no picks)")

        fold_metrics.append(
            {
                "test_season": test_season,
                "mae": mae,
                "rmse": rmse,
                "win_pct": win_pct,
                "roi_pct": roi_pct,
            }
        )
        feature_importances_per_fold.append(model.feature_importances_)

    if fold_metrics:
        maes = np.array([m["mae"] for m in fold_metrics])
        rmses = np.array([m["rmse"] for m in fold_metrics])
        win_pcts = np.array([m["win_pct"] for m in fold_metrics if not np.isnan(m["win_pct"])])
        rois = np.array([m["roi_pct"] for m in fold_metrics if not np.isnan(m["roi_pct"])])

        print("\nCV summary across folds")
        print("=" * 60)
        print(f"MAE  mean={maes.mean():.4f}, std={maes.std(ddof=0):.4f}")
        print(f"RMSE mean={rmses.mean():.4f}, std={rmses.std(ddof=0):.4f}")
        if win_pcts.size:
            print(f"ATS win% mean={win_pcts.mean()*100:.1f}%, std={win_pcts.std(ddof=0)*100:.1f}%")
        if rois.size:
            print(f"ATS ROI mean={rois.mean():.1f}%, std={rois.std(ddof=0):.1f}%")
            if rois.max() - rois.min() > 15.0:
                print("WARNING: ATS ROI varies by more than 15 percentage points across folds (instability).")

        # Features with importance < 0.005 in ALL folds
        if feature_importances_per_fold:
            imps = np.vstack(feature_importances_per_fold)
            low_mask = np.all(imps < 0.005, axis=0)
            low_feats = [use_cols[i] for i, low in enumerate(low_mask) if low]
            if low_feats:
                print("\nFeatures with importance < 0.005 in ALL folds (removal candidates):")
                for name in low_feats:
                    print(f"  {name}")

    # Train final production model on all seasons 2021-2025
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

    # Full-data metrics (in-sample) for reference
    pred_final = final_model.predict(X_final)
    mae_full = mean_absolute_error(y_final, pred_final)
    rmse_full = np.sqrt(mean_squared_error(y_final, pred_final))
    print("\nFull-data metrics (2021-2025, in-sample):")
    print(f"  MAE  = {mae_full:.4f}")
    print(f"  RMSE = {rmse_full:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": final_model,
        "feature_columns": use_cols,
        "margin_home_minus_away": True,
        "margin_calibration_shift": 0.0,
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"\nSaved production model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
