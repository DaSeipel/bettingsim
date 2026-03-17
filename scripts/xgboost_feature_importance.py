#!/usr/bin/env python3
"""
Inspect XGBoost NCAAB spread model: print feature list and feature importance.
Flag whether ModelRank or any power-ranking-derived feature is in the model or in the top 20.
Explains why flat margins for large rank-gap matchups (e.g. Illinois vs Penn) if rank is not a feature.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "data" / "models" / "xgboost_spread_ncaab.pkl"

# Substrings that indicate a ranking / power-rank-derived feature
RANK_FEATURE_KEYWORDS = ("rank", "model_rank", "ModelRank", "power_rank", "rating_rank", "barthag_rank")


def _is_ranking_feature(name: str) -> bool:
    n = (name or "").strip().lower()
    return any(k.lower() in n for k in RANK_FEATURE_KEYWORDS)


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        return 1

    try:
        import joblib
        payload = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return 1

    model = payload.get("model")
    feature_columns = payload.get("feature_columns") or []

    if not model or not feature_columns:
        print("Model or feature_columns missing in payload.")
        return 1

    # XGBoost feature importance (gain or weight; regressor has feature_importances_)
    try:
        importances = model.feature_importances_
    except Exception:
        importances = []

    if len(importances) != len(feature_columns):
        print(f"Feature count mismatch: {len(feature_columns)} columns vs {len(importances)} importance values.")
        importances = [0.0] * len(feature_columns)

    # Sort by importance descending
    indexed = list(zip(feature_columns, importances))
    indexed.sort(key=lambda x: -x[1])

    print("=" * 70)
    print("  XGBoost NCAAB spread model: feature set and importance")
    print("  Model path:", MODEL_PATH)
    print("=" * 70)

    print("\n  --- Is ModelRank or any power-ranking feature in the model? ---")
    ranking_features = [f for f in feature_columns if _is_ranking_feature(f)]
    if ranking_features:
        print("  YES. Ranking-related features found:", ranking_features)
    else:
        print("  NO. No feature name contains 'rank', 'ModelRank', or 'power_rank'.")
        print("  => Power rankings are NOT an input to the spread model.")
        print("  => That explains flat margins for large rank-gap matchups (e.g. Illinois rank 7 vs Penn rank 134):")
        print("     the model only sees KenPom/stat and situational features, not overall team rank.")

    print("\n  --- Full feature list (in model order) ---")
    for i, col in enumerate(feature_columns):
        imp = importances[i] if i < len(importances) else 0
        flag = " [RANKING]" if _is_ranking_feature(col) else ""
        print(f"    {i+1:2}. {col:<45} importance={imp:.4f}{flag}")

    print("\n  --- Top 20 features by importance ---")
    for r, (name, imp) in enumerate(indexed[:20], 1):
        flag = "  <-- RANKING FEATURE" if _is_ranking_feature(name) else ""
        print(f"    {r:2}. {name:<45} {imp:.4f}{flag}")

    print("\n  --- Ranking-related in top 20? ---")
    top20_names = [x[0] for x in indexed[:20]]
    ranking_in_top20 = [f for f in top20_names if _is_ranking_feature(f)]
    if ranking_in_top20:
        print("  YES:", ranking_in_top20)
    else:
        print("  NO. No ranking-based feature appears in the top 20 by importance.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
