#!/usr/bin/env python3
"""
Full favorite bias diagnostic: SHAP top features + direction, avg predicted prob
(favorite vs underdog), and correlation(edge, is_favorite). No model changes — print only.
Run from repo root: python3 scripts/favorite_bias_diagnostic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from engine.betting_models import (
    get_training_data,
    load_model,
    SPREAD_MODEL_PATH,
    predict_spread_prob,
    _select_features,
)

# Cap rows for SHAP to keep runtime reasonable
SHAP_SAMPLE = 800
CORRELATION_BIAS_THRESHOLD = 0.3


def _in_house_spread_home_minus_away(row: pd.Series) -> float:
    """Home spread in 'market' convention: negative when home favored. Uses offensive rating diff."""
    ho = row.get("home_offensive_rating", 0) or 0
    ao = row.get("away_offensive_rating", 0) or 0
    diff = float(pd.to_numeric(ho, errors="coerce") or 0) - float(pd.to_numeric(ao, errors="coerce") or 0)
    return -diff  # market_spread: home -3.5 => -3.5


def run_shap_top15(payload: dict, df: pd.DataFrame) -> list[dict]:
    """Top 15 features by mean |SHAP|, with mean(SHAP) for direction (positive = pushes margin up = home/favorite)."""
    try:
        import shap
    except ImportError:
        return []
    model = payload.get("model")
    used = payload.get("feature_columns") or []
    if model is None or not used:
        return []
    X, used = _select_features(df, used)
    if X.empty:
        return []
    if len(X) > SHAP_SAMPLE:
        X = X.sample(n=SHAP_SAMPLE, random_state=42)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[0] if len(sv) else sv
    sv = np.asarray(sv)
    mean_abs = np.abs(sv).mean(axis=0)
    mean_direction = sv.mean(axis=0)
    order = np.argsort(-mean_abs)[:15]
    out = []
    favorite_related = {"home_is_favorite", "home_offensive_rating", "away_offensive_rating"}
    for i in order:
        name = used[i]
        out.append({
            "feature": name,
            "mean_abs_shap": float(mean_abs[i]),
            "mean_shap": float(mean_direction[i]),
            "favorite_related": name in favorite_related,
        })
    return out


def run_avg_prob_favorite_vs_underdog(df: pd.DataFrame) -> tuple[float, float, int]:
    """For each game, in-house spread and model prob for favorite vs underdog cover. Return (avg_fav, avg_dog, n)."""
    payload = load_model(SPREAD_MODEL_PATH)
    if payload is None:
        return 0.0, 0.0, 0
    cols = payload.get("feature_columns", [])
    prob_fav_list = []
    prob_dog_list = []
    for idx, row in df.iterrows():
        row_ser = row
        X, _ = _select_features(pd.DataFrame([row_ser]), cols)
        if X.empty:
            continue
        market_spread = _in_house_spread_home_minus_away(row_ser)
        # we_cover_favorite True => probability that the favorite covers
        p_fav = predict_spread_prob(row_ser, market_spread, we_cover_favorite=True, fallback_prob=0.5)
        p_dog = predict_spread_prob(row_ser, market_spread, we_cover_favorite=False, fallback_prob=0.5)
        prob_fav_list.append(p_fav)
        prob_dog_list.append(p_dog)
    if not prob_fav_list:
        return 0.0, 0.0, 0
    n = len(prob_fav_list)
    return float(np.mean(prob_fav_list)), float(np.mean(prob_dog_list)), n


def run_correlation_edge_is_favorite(df: pd.DataFrame) -> tuple[float, int]:
    """Correlation between model probability (for the side) and is_favorite (1=favorite, 0=underdog)."""
    payload = load_model(SPREAD_MODEL_PATH)
    if payload is None:
        return 0.0, 0
    cols = payload.get("feature_columns", [])
    probs = []
    is_fav = []
    for idx, row in df.iterrows():
        row_ser = row
        X, _ = _select_features(pd.DataFrame([row_ser]), cols)
        if X.empty:
            continue
        market_spread = _in_house_spread_home_minus_away(row_ser)
        p_fav = predict_spread_prob(row_ser, market_spread, we_cover_favorite=True, fallback_prob=0.5)
        p_dog = predict_spread_prob(row_ser, market_spread, we_cover_favorite=False, fallback_prob=0.5)
        probs.append(p_fav)
        is_fav.append(1)
        probs.append(p_dog)
        is_fav.append(0)
    if len(probs) < 10:
        return 0.0, 0
    probs = np.array(probs)
    is_fav = np.array(is_fav)
    r = np.corrcoef(probs, is_fav)[0, 1]
    return float(r) if not np.isnan(r) else 0.0, len(probs) // 2


def main() -> None:
    db_path = ROOT / "data" / "espn.db"
    print("Loading historical training data...")
    df, _, _ = get_training_data(db_path=db_path, league=None, merge_situational=True)
    if df.empty or len(df) < 5:
        print("Not enough training data. Need at least 5 games with features.")
        print("Run your data pipeline (fetch/merge + situational features) and train_models.py first.")
        return

    payload = load_model(SPREAD_MODEL_PATH)
    if payload is None:
        print("Spread model not found. Train with: python3 train_models.py")
        return

    print()
    print("=" * 70)
    print("FAVORITE BIAS DIAGNOSTIC (report only — no changes made)")
    print("=" * 70)

    # (1) Top 15 SHAP + direction
    print()
    print("(1) TOP 15 FEATURES BY SHAP IMPORTANCE (mean |SHAP|) + IMPACT DIRECTION")
    print("-" * 70)
    print("    Direction: mean(SHAP) > 0 => pushes predicted margin UP (toward home/favorite).")
    print("    'Favorite-related' = direct encoding of who is favored (may indicate bias).")
    print()
    shap_results = run_shap_top15(payload, df)
    if not shap_results:
        print("    Could not compute SHAP (install: pip install shap).")
    else:
        for i, r in enumerate(shap_results, 1):
            direction = "toward favorite/home" if r["mean_shap"] > 0 else "toward underdog/away"
            tag = " [FAVORITE-RELATED]" if r["favorite_related"] else ""
            print(f"    {i:2}. {r['feature']}")
            print(f"        mean |SHAP| = {r['mean_abs_shap']:.4f}  |  mean SHAP = {r['mean_shap']:+.4f}  =>  {direction}{tag}")
        print()

    # (2) Average predicted win probability: favorite vs underdog
    print("(2) AVERAGE PREDICTED WIN (COVER) PROBABILITY: FAVORITE vs UNDERDOG")
    print("-" * 70)
    avg_fav, avg_dog, n_games = run_avg_prob_favorite_vs_underdog(df)
    print(f"    Games in historical set: {n_games}")
    if n_games < 30:
        print("    (Small sample — averages may be noisy.)")
    print(f"    Average probability model assigns to FAVORITE covering:  {avg_fav:.4f} ({avg_fav*100:.2f}%)")
    print(f"    Average probability model assigns to UNDERDOG covering: {avg_dog:.4f} ({avg_dog*100:.2f}%)")
    if avg_fav > 0.55 and avg_dog < 0.45:
        print("    => Model systematically favors the favorite side.")
    print()

    # (3) Correlation(edge, is_favorite)
    print("(3) CORRELATION: MODEL PROBABILITY vs IS_FAVORITE (1=favorite, 0=underdog)")
    print("-" * 70)
    corr, n_obs = run_correlation_edge_is_favorite(df)
    print(f"    Observations (each game contributes 2: favorite side + underdog side): {n_obs * 2}")
    print(f"    Correlation(model_prob, is_favorite) = {corr:.4f}")
    if corr > CORRELATION_BIAS_THRESHOLD:
        print(f"    *** BIAS FLAG: Correlation > {CORRELATION_BIAS_THRESHOLD} — model is biased toward favorites.")
    else:
        print(f"    OK: Correlation <= {CORRELATION_BIAS_THRESHOLD}.")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
