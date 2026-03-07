#!/usr/bin/env python3
"""
Favorite bias audit: breakdown of spread picks (favorite vs underdog), implied prob check,
underdog value detection, and is_favorite SHAP importance.
Run from repo root: python scripts/audit_favorite_bias.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from datetime import date, timedelta
from engine.play_history import load_play_history

FAVORITE_BIAS_THRESHOLD_PCT = 65.0


def _is_spread_favorite(recommended_side: str) -> bool:
    """True if the recommended side is the favorite (minus spread, e.g. 'Lakers -7.0')."""
    s = (recommended_side or "").strip()
    # Favorite = minus spread; underdog = plus spread
    if "+" in s:
        return False
    return "-" in s and any(c.isdigit() for c in s)


def run_favorite_underdog_report(db_path: Path | None = None, days: int = 90) -> dict:
    """Load play_history spread plays and return favorite vs underdog counts."""
    to_date = date.today()
    from_date = to_date - timedelta(days=days)
    hist = load_play_history(from_date=from_date, to_date=to_date, db_path=db_path)
    if hist.empty:
        return {"n_spread": 0, "n_favorite": 0, "n_underdog": 0, "pct_favorite": 0.0, "flag": False}

    spread = hist[hist["bet_type"].astype(str).str.strip().str.lower().str.contains("spread", na=False)]
    if spread.empty:
        return {"n_spread": 0, "n_favorite": 0, "n_underdog": 0, "pct_favorite": 0.0, "flag": False}

    n_fav = spread["recommended_side"].apply(_is_spread_favorite).sum()
    n_dog = len(spread) - n_fav
    pct_fav = (n_fav / len(spread) * 100.0) if len(spread) else 0.0
    return {
        "n_spread": len(spread),
        "n_favorite": int(n_fav),
        "n_underdog": int(n_dog),
        "pct_favorite": round(pct_fav, 1),
        "flag": pct_fav > FAVORITE_BIAS_THRESHOLD_PCT,
    }


def main() -> None:
    db_path = ROOT / "data" / "espn.db"
    print("=" * 60)
    print("FAVORITE BIAS AUDIT")
    print("=" * 60)

    # 1. Favorite vs underdog report (spread picks)
    report = run_favorite_underdog_report(db_path=db_path, days=365)
    print("\n1. SPREAD PICKS BREAKDOWN (last 365 days from play_history)")
    print("-" * 50)
    print(f"   Total spread plays: {report['n_spread']}")
    print(f"   Favorite: {report['n_favorite']}")
    print(f"   Underdog: {report['n_underdog']}")
    if report["n_spread"] > 0:
        print(f"   % on favorite: {report['pct_favorite']}%")
        if report["flag"]:
            print(f"   *** FLAG: More than {FAVORITE_BIAS_THRESHOLD_PCT}% on favorite — possible favorite bias.")
        else:
            print(f"   OK: Favorite share below {FAVORITE_BIAS_THRESHOLD_PCT}%.")
    else:
        print("   No spread plays in history yet.")

    # 2. Implied probability check (done in strategies.py)
    print("\n2. IMPLIED PROBABILITY (strategies.strategies)")
    print("-" * 50)
    from strategies.strategies import implied_probability, american_to_decimal
    dec = american_to_decimal(-110)
    impl = implied_probability(-110)
    print(f"   American -110 -> decimal {dec:.4f} -> implied = 1/decimal = {impl:.4f}")
    print("   Standard formula: implied = 1 / decimal_odds. See implied_probability_fair_two_sided() for two-outcome vig removal.")

    # 3. Underdog value detector (logic lives in app/engine; report here)
    print("\n3. UNDERDOG VALUE DETECTOR")
    print("-" * 50)
    print("   Implemented in engine: underdog value = model_prob(underdog) > market fair implied(underdog).")
    print("   Use fair implied from implied_probability_fair_two_sided(fav_odds, dog_odds) when both sides available.")

    # 4. SHAP importance for is_favorite (if model exists)
    print("\n4. IS_FAVORITE SHAP IMPORTANCE (spread model)")
    print("-" * 50)
    try:
        from engine.betting_models import (
            load_model,
            SPREAD_MODEL_PATH,
            SPREAD_FEATURE_COLUMNS,
            get_training_data,
            _select_features,
        )
        import numpy as np
        payload = load_model(SPREAD_MODEL_PATH)
        if payload and "home_is_favorite" in (payload.get("feature_columns") or []):
            # Run SHAP
            try:
                import shap
                df, _, _ = get_training_data(db_path=db_path, merge_situational=True)
                if not df.empty and "home_is_favorite" in df.columns:
                    X, used = _select_features(df, payload["feature_columns"])
                    if not X.empty and "home_is_favorite" in used:
                        model = payload["model"]
                        explainer = shap.TreeExplainer(model)
                        sv = explainer.shap_values(X)
                        if isinstance(sv, list):
                            sv = sv[0] if len(sv) else sv
                        mean_abs = np.abs(sv).mean(axis=0)
                        idx = used.index("home_is_favorite")
                        rank = int((mean_abs >= mean_abs[idx]).sum())
                        print(f"   home_is_favorite mean |SHAP| rank: {rank} of {len(used)}")
                        if rank <= 3:
                            print("   *** FLAG: home_is_favorite has disproportionate importance. Remove from features and retrain.")
                        else:
                            print("   OK: home_is_favorite not in top 3 by SHAP.")
                    else:
                        print("   home_is_favorite not in model features.")
                else:
                    print("   No training data or home_is_favorite not in data.")
            except ImportError:
                print("   SHAP not installed; pip install shap to run importance check.")
        else:
            print("   Spread model not found or home_is_favorite not in feature list.")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
