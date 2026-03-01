#!/usr/bin/env python3
"""
Train the three specialized XGBoost models (spreads, totals, moneylines) on historical
games_with_team_stats + situational features. Saves models to data/models/ and metrics to metrics.json.
Run after fetching and merging data (fetch_merge_and_save, situational features).
"""

import sys
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.betting_models import (
    get_training_data,
    train_all_models,
    load_metrics,
    MODELS_DIR,
)


def main() -> None:
    print("Loading training data (games_with_team_stats + situational)...")
    df, _, _ = get_training_data(league=None, merge_situational=True)
    if df.empty:
        print("No training data. Run fetch_merge_and_save and build situational features first.")
        sys.exit(1)
    n = len(df)
    print(f"Found {n} completed games with features.")
    if n < 50:
        print("Recommend at least 50 games for training. Proceeding anyway.")
    print("Training spread, totals, and moneyline models (XGBoost)...")
    metrics = train_all_models(league=None, test_size=0.2, random_state=42)
    if not metrics:
        print("Training produced no metrics (check feature columns and targets).")
        sys.exit(1)
    print(f"Models saved to {MODELS_DIR}")
    print("Evaluation (test set):")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v}")
    saved = load_metrics()
    print(f"Metrics written to {MODELS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
