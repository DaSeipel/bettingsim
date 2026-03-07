#!/usr/bin/env python3
"""
Train NCAAB spread as MARGIN REGRESSION (not classifier): predict actual point margin.
Value = model_margin - closing_spread. If > 3 pts → underdog value; if < -3 pts → favorite value.
- Target: margin = home_score - away_score (continuous).
- No SMOTE; standard train/test split.
- Saves regressor to SPREAD_MODEL_PATH_NCAAB with is_underdog_cover_classifier=False.
- Prints underdog recommendation rate on test set (recommend when |line_error| > 3 pts).
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from engine.betting_models import (
    NCAAB_KENPOM_SPREAD_FEATURE_COLUMNS,
    SPREAD_MODEL_PATH_NCAAB,
    _ensure_models_dir,
    _select_features,
    get_training_data,
)

# Line-error threshold: recommend when |line_error| > this (model must disagree with market by at least this many points)
LINE_ERROR_THRESHOLD_PTS = 10.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train NCAAB spread as margin regression; recommend by line error (model_margin - spread)."
    )
    parser.add_argument(
        "--allow-proxy",
        action="store_true",
        help="If set, use KenPom proxy when <50 real closing lines (testing only).",
    )
    args = parser.parse_args()

    print("Loading NCAAB training data (games_with_team_stats + situational)...")
    df, _, _ = get_training_data(league="ncaab", merge_situational=True)
    if df.empty:
        print("No NCAAB training data.")
        sys.exit(1)

    if "closing_home_spread" not in df.columns:
        df["closing_home_spread"] = np.nan
    df["closing_home_spread"] = pd.to_numeric(df["closing_home_spread"], errors="coerce")
    has_real = df["closing_home_spread"].notna()
    if has_real.sum() < 50 and args.allow_proxy:
        proxy = -(df["home_ADJOE"].astype(float) - df["away_ADJOE"].astype(float))
        df.loc[~has_real, "closing_home_spread"] = proxy
        print("Using proxy closing_home_spread (--allow-proxy) for testing only.")
    df = df.dropna(subset=["closing_home_spread"]).copy()
    if "home_ADJOE" not in df.columns or "away_ADJOE" not in df.columns:
        print("KenPom columns missing. Run merge_ncaab_kenpom_into_games.py first.")
        sys.exit(1)
    df = df.dropna(subset=["home_ADJOE", "away_ADJOE", "home_BARTHAG", "away_BARTHAG"])
    if len(df) < 50:
        print(
            f"Only {len(df)} NCAAB games have closing lines and KenPom stats. Need at least 50."
        )
        sys.exit(1)

    n_total = len(df)
    print(f"Games with KenPom + real closing_home_spread: {n_total}")

    # Target: actual margin = home_score - away_score (home - away). Positive = home wins by that many.
    df = df.copy()
    df["margin"] = df["home_score"].astype(float) - df["away_score"].astype(float)
    spread = df["closing_home_spread"].astype(float)
    print("Margin target definition: home_score - away_score (home - away); positive = home wins by that many.")
    print(f"Margin (home-away) range: [{df['margin'].min():.1f}, {df['margin'].max():.1f}]")
    print(f"Sample (home_score, away_score, margin): {list(zip(df['home_score'].head(3).tolist(), df['away_score'].head(3).tolist(), df['margin'].head(3).tolist()))}")

    feature_cols = list(NCAAB_KENPOM_SPREAD_FEATURE_COLUMNS)
    X, used = _select_features(df, feature_cols)
    if X.empty:
        print("No features after selection.")
        sys.exit(1)

    y = df["margin"].values

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Align spread with train/test (same indices as X_train / X_test)
    spread_train = df.loc[X_train.index, "closing_home_spread"].values
    spread_test = df.loc[X_test.index, "closing_home_spread"].values

    try:
        import xgboost as xgb
    except ImportError:
        print("XGBoost required.")
        sys.exit(1)

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.08,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Save as margin model. No calibration shift: raw predicted margin vs closing spread for natural edge.
    payload = {
        "model": model,
        "feature_columns": used,
        "is_underdog_cover_classifier": False,
        "margin_home_minus_away": True,
        "margin_calibration_shift": 0.0,
    }
    _ensure_models_dir()
    with open(SPREAD_MODEL_PATH_NCAAB, "wb") as f:
        pickle.dump(payload, f)
    print(f"Model saved to {SPREAD_MODEL_PATH_NCAAB} (no calibration shift)")

    # Test set: line error = pred_margin - closing_spread (no offset)
    pred_margin_test = model.predict(X_test)
    line_error_test = pred_margin_test - spread_test

    recommend_underdog = line_error_test > LINE_ERROR_THRESHOLD_PTS
    recommend_favorite = line_error_test < -LINE_ERROR_THRESHOLD_PTS
    total_recommendations = int(recommend_underdog.sum() + recommend_favorite.sum())

    if total_recommendations > 0:
        underdog_rate = 100.0 * recommend_underdog.sum() / total_recommendations
        print()
        print(
            f"Underdog spread recommendation rate (test set, |line_error| > {LINE_ERROR_THRESHOLD_PTS} pts, no shift): {underdog_rate:.1f}%"
        )
        print(f"  Underdog: {int(recommend_underdog.sum())}, Favorite: {int(recommend_favorite.sum())}, total: {total_recommendations}")
        print(f"  Target: underdog rate 30–50%, MAE < 12 pts")
    else:
        print()
        print("No test recommendations (|line_error| > threshold).")

    mae = np.abs(pred_margin_test - y_test).mean()
    print()
    print(f"Test MAE (margin): {mae:.2f} pts (target: < 12 pts)")


if __name__ == "__main__":
    main()
