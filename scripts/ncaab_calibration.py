#!/usr/bin/env python3
"""
NCAAB spread model calibration and simple backtest.
Loads games_with_team_stats (NCAAB, with closing_home_spread and scores), computes model P(home covers),
buckets predictions, and prints actual cover rate per bucket (calibration).
Optionally prints a CLV note: with opening lines you could count correct only when bet price was >= closing.
Usage: python scripts/ncaab_calibration.py [--espn-db path] [--buckets 10]
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
    load_model,
    predict_spread_prob,
    _spread_model_path_for_league,
    _select_features,
    SPREAD_FEATURE_COLUMNS,
)
from engine.sportsref_stats import load_merged_games_from_sqlite

DATA_DIR = ROOT / "data"
ESPN_DB = DATA_DIR / "espn.db"
BUCKET_EDGES = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]


def run_calibration(espn_db_path: Path | None = None, n_buckets: int = 10) -> None:
    espn_db_path = espn_db_path or ESPN_DB
    if not espn_db_path.exists():
        print(f"DB not found: {espn_db_path}")
        return
    df = load_merged_games_from_sqlite(league="ncaab", db_path=espn_db_path)
    if df.empty:
        print("No NCAAB games in games_with_team_stats.")
        return
    for col in ("closing_home_spread", "home_score", "away_score"):
        if col not in df.columns:
            print(f"Missing column: {col}. Run full pipeline with historical odds merge to add closing lines.")
            return
    df = df.dropna(subset=["closing_home_spread", "home_score", "away_score"]).copy()
    if df.empty:
        print("No NCAAB games with closing spread and scores.")
        return
    df["margin"] = df["home_score"].astype(float) - df["away_score"].astype(float)
    df["home_covered"] = (df["margin"] > df["closing_home_spread"].astype(float)).astype(int)

    path = _spread_model_path_for_league("ncaab")
    payload = load_model(path)
    if payload is None:
        path = _spread_model_path_for_league(None)
        payload = load_model(path)
    if payload is None:
        print("No spread model found. Train with train_ncaab_models() or train_all_models().")
        return
    model = payload["model"]
    cols = payload.get("feature_columns", SPREAD_FEATURE_COLUMNS)
    prob_home_cover = []
    for idx, row in df.iterrows():
        X, _ = _select_features(pd.DataFrame([row]), cols)
        if X.empty:
            prob_home_cover.append(0.5)
            continue
        pred_margin = float(model.predict(X)[0])
        spread = float(row["closing_home_spread"])
        diff = pred_margin - spread
        k = 0.35
        p = 1.0 / (1.0 + np.exp(-k * diff))
        prob_home_cover.append(float(np.clip(p, 0.02, 0.98)))
    df["pred_prob_home_cover"] = prob_home_cover

    edges = np.linspace(0.4, 0.9, n_buckets + 1).tolist()
    edges[-1] = 1.01
    print("NCAAB spread model calibration (predicted P(home covers) vs actual cover rate)")
    print("=" * 60)
    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i + 1]
        mask = (df["pred_prob_home_cover"] >= low) & (df["pred_prob_home_cover"] < high)
        sub = df.loc[mask]
        if len(sub) == 0:
            continue
        actual = sub["home_covered"].mean()
        pred_avg = sub["pred_prob_home_cover"].mean()
        print(f"  [{low:.2f}, {high:.2f}): n={len(sub):4d}  pred_avg={pred_avg:.3f}  actual_cover_rate={actual:.3f}")
    print()
    print("CLV note: To evaluate closing-line value, only count a pick as correct when the bet was placed")
    print("at a price at least as good as closing. Add opening lines to the pipeline for a full CLV backtest.")


def main() -> int:
    argv = [a for a in sys.argv[1:] if a.startswith("--")]
    espn_db = ESPN_DB
    n_buckets = 10
    for a in argv:
        if a.startswith("--espn-db="):
            espn_db = Path(a.split("=", 1)[1])
        elif a.startswith("--buckets="):
            try:
                n_buckets = int(a.split("=", 1)[1])
            except ValueError:
                pass
    run_calibration(espn_db_path=espn_db, n_buckets=n_buckets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
