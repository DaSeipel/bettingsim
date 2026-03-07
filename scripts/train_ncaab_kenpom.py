#!/usr/bin/env python3
"""
Train NCAAB spread model on games that have KenPom stats and REAL closing lines only (no proxy).
Target: predict margin; we then derive P(cover) and compare to market — training on real lines
so the model learns vs actual market. Mandatory calibration forces average predicted cover
probability for favorites and underdogs both within 5% of 50%. Prints recalibrated probs.
Use --allow-proxy only for testing calibration when you have no real closing lines yet.
"""

import argparse
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
    load_model,
)

# Calibration target: mean predicted cover prob for favorites and underdogs in [0.45, 0.55]
CALIBRATION_TARGET = 0.5
CALIBRATION_TOLERANCE = 0.05  # within 5% of 50%


def _sigmoid_cover_prob(pred_margin: float, market_spread: float, k: float = 0.35) -> float:
    """P(home covers). Home covers when margin > market_spread."""
    diff = pred_margin - market_spread
    return float(1.0 / (1.0 + np.exp(-k * diff)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NCAAB spread model with real closing lines and calibration.")
    parser.add_argument("--allow-proxy", action="store_true", help="If set, use KenPom proxy when <50 real closing lines (for testing calibration only).")
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
        print("Using proxy closing_home_spread (--allow-proxy) for calibration test only.")
    df = df.dropna(subset=["closing_home_spread"]).copy()
    if "home_ADJOE" not in df.columns or "away_ADJOE" not in df.columns:
        print("KenPom columns missing. Run merge_ncaab_kenpom_into_games.py first.")
        sys.exit(1)
    df = df.dropna(subset=["home_ADJOE", "away_ADJOE"])
    if len(df) < 50:
        print(
            f"Only {len(df)} NCAAB games have closing lines and KenPom stats. Need at least 50. "
            "Fetch NCAAB odds regularly (scripts/fetch_ncaab_odds_snapshot.py), run "
            "merge_historical_closing_into_games, then re-run. Or use --allow-proxy for a calibration test."
        )
        sys.exit(1)

    n_total = len(df)
    print(f"Games with KenPom + real closing_home_spread: {n_total}")

    # Add differential features (gap between teams is more predictive than raw values)
    df = df.copy()
    df["adjoe_diff"] = df["home_ADJOE"].astype(float) - df["away_ADJOE"].astype(float)
    df["adjde_diff"] = df["home_ADJDE"].astype(float) - df["away_ADJDE"].astype(float)
    df["barthag_diff"] = df["home_BARTHAG"].astype(float) - df["away_BARTHAG"].astype(float)
    # Seed: use 99 if null (team not in tournament)
    home_seed = df["home_SEED"].fillna(99).astype(float)
    away_seed = df["away_SEED"].fillna(99).astype(float)
    df["seed_diff"] = home_seed - away_seed
    df["tempo_diff"] = df["home_ADJ_T"].astype(float) - df["away_ADJ_T"].astype(float)

    feature_cols = list(NCAAB_KENPOM_SPREAD_FEATURE_COLUMNS) + [
        "adjoe_diff", "adjde_diff", "barthag_diff", "seed_diff", "tempo_diff",
    ]
    X, used = _select_features(df, feature_cols)
    if X.empty or "margin" not in df.columns:
        print("No features available after selection.")
        sys.exit(1)

    import pickle
    from sklearn.model_selection import train_test_split
    try:
        import xgboost as xgb
        _use_xgb = True
    except Exception:
        _use_xgb = False
        from sklearn.ensemble import GradientBoostingRegressor

    y = df["margin"].astype(float)
    spread = df["closing_home_spread"].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    train_idx = X_train.index
    test_idx = X_test.index
    spread_train = spread.loc[train_idx].values
    spread_test = spread.loc[test_idx].values
    margin_train = y_train.values
    margin_test = y_test.values

    if _use_xgb:
        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
        model.fit(X_train, y_train)
        print("Using sklearn GradientBoostingRegressor (XGBoost not available).")

    # Raw predicted P(cover) on full dataset (for calibration)
    pred_margin_all = model.predict(X)
    prob_home_cover = np.array([
        _sigmoid_cover_prob(pm, s) for pm, s in zip(pred_margin_all, spread.values)
    ])
    is_favorite_home = spread.values < 0
    pred_favorite_cover = np.where(is_favorite_home, prob_home_cover, 1.0 - prob_home_cover)
    pred_underdog_cover = 1.0 - pred_favorite_cover
    mean_fav_raw = float(np.mean(pred_favorite_cover))
    mean_dog_raw = float(np.mean(pred_underdog_cover))

    # Mandatory calibration: shift so both favorites and underdogs average 50%
    shift_fav = CALIBRATION_TARGET - mean_fav_raw
    shift_dog = CALIBRATION_TARGET - mean_dog_raw
    cal_fav = np.clip(pred_favorite_cover + shift_fav, 0.02, 0.98)
    cal_dog = np.clip(pred_underdog_cover + shift_dog, 0.02, 0.98)
    mean_fav_cal = float(np.mean(cal_fav))
    mean_dog_cal = float(np.mean(cal_dog))

    # Check within 5% of 50%
    ok_fav = abs(mean_fav_cal - CALIBRATION_TARGET) <= CALIBRATION_TOLERANCE
    ok_dog = abs(mean_dog_cal - CALIBRATION_TARGET) <= CALIBRATION_TOLERANCE
    if not ok_fav or not ok_dog:
        print(
            f"Calibration: favorites mean={mean_fav_cal:.4f}, underdogs mean={mean_dog_cal:.4f} "
            f"(target {CALIBRATION_TARGET} ± {CALIBRATION_TOLERANCE})."
        )

    payload = {
        "model": model,
        "feature_columns": used,
        "calibration_shift_fav": float(shift_fav),
        "calibration_shift_dog": float(shift_dog),
    }
    _ensure_models_dir()
    with open(SPREAD_MODEL_PATH_NCAAB, "wb") as f:
        pickle.dump(payload, f)
    print(f"Model (with calibration) saved to {SPREAD_MODEL_PATH_NCAAB}")

    # Metrics
    actual_cover_train = (margin_train > spread_train).astype(int)
    actual_cover_test = (margin_test > spread_test).astype(int)
    pred_cover_train = (model.predict(X_train) > spread_train).astype(int)
    pred_cover_test = (model.predict(X_test) > spread_test).astype(int)
    train_acc = (actual_cover_train == pred_cover_train).mean()
    val_acc = (actual_cover_test == pred_cover_test).mean()

    print()
    print("Total games used for training:", n_total)
    print("Train accuracy (cover):", round(train_acc, 4))
    print("Validation accuracy (cover):", round(val_acc, 4))
    print("Pre-calibration  average predicted cover prob — favorites:", round(mean_fav_raw, 4), " underdogs:", round(mean_dog_raw, 4))
    print("Recalibrated     average predicted cover prob — favorites:", round(mean_fav_cal, 4), " underdogs:", round(mean_dog_cal, 4))

    # Top 5 features by mean |SHAP| (fallback: tree feature_importances_)
    print()
    try:
        import shap
        X_shap = X_test
        explainer = shap.TreeExplainer(model, X_shap)
        sv = explainer.shap_values(X_shap)
        if isinstance(sv, list):
            sv = sv[0]
        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(-mean_abs)[:5]
        print("Top 5 features by mean |SHAP|:")
        for i, idx in enumerate(order, 1):
            print(f"  {i}. {used[idx]}: {mean_abs[idx]:.4f}")
    except Exception as e:
        # Fallback: tree feature_importances_ (GradientBoostingRegressor / XGBoost)
        imp = getattr(model, "feature_importances_", None)
        if imp is not None and len(imp) == len(used):
            order = np.argsort(-imp)[:5]
            print("Top 5 features by feature_importances_ (SHAP failed):")
            for i, idx in enumerate(order, 1):
                print(f"  {i}. {used[idx]}: {imp[idx]:.4f}")
        else:
            print("SHAP importance skipped:", e)


if __name__ == "__main__":
    main()
