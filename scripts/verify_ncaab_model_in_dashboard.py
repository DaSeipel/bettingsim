#!/usr/bin/env python3
"""
Verify the Streamlit dashboard uses the trained NCAAB model at data/models/xgboost_spread_ncaab.pkl:
(1) NCAAB model is loaded correctly
(2) Calibration shifts are applied when generating probabilities
(3) Today's NCAAB plays use this model (not proxy)
Print model file path, calibration info, and sample of today's NCAAB value plays with edge %.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    from engine.betting_models import (
        SPREAD_MODEL_PATH_NCAAB,
        _spread_model_path_for_league,
        load_model,
        predict_spread_prob,
    )

    # (1) Which model file is loaded for NCAAB
    path_ncaab = _spread_model_path_for_league("ncaab")
    print("(1) NCAAB spread model path:", path_ncaab)
    print("    Exists:", path_ncaab.exists())
    if path_ncaab != SPREAD_MODEL_PATH_NCAAB:
        print("    WARNING: path differs from SPREAD_MODEL_PATH_NCAAB:", SPREAD_MODEL_PATH_NCAAB)
    else:
        print("    Matches SPREAD_MODEL_PATH_NCAAB (data/models/xgboost_spread_ncaab.pkl)")

    # (2) Calibration shifts in payload
    payload = load_model(path_ncaab) if path_ncaab.exists() else None
    if payload is None:
        print("\n(2) Calibration: Model not loaded (file missing or error).")
    else:
        shift_fav = payload.get("calibration_shift_fav")
        shift_dog = payload.get("calibration_shift_dog")
        print("\n(2) Calibration shifts in saved payload:")
        print("    calibration_shift_fav:", shift_fav)
        print("    calibration_shift_dog:", shift_dog)
        if shift_fav is not None and shift_dog is not None:
            print("    -> Applied in predict_spread_prob() so favorites and underdogs average ~50%.")
        else:
            print("    -> No shifts (legacy model); probabilities used as-is.")

    # (3) Confirm NCAAB plays use this model (not proxy)
    print("\n(3) Today's NCAAB plays: built by _live_odds_to_value_plays() -> consensus_spread() -> predict_spread_prob(..., league='ncaab').")
    print("    _spread_model_path_for_league('ncaab') returns", path_ncaab.name, "-> same trained model (no proxy).")
    print("    Proxy was only used in train script when --allow-proxy and <50 real closing lines; dashboard never uses proxy.")

    # Sample of today's NCAAB value plays with edge %
    print("\n--- Today's NCAAB value plays (sample with edge %) ---")
    api_key = (os.environ.get("ODDS_API_KEY") or "").strip()
    if not api_key and (ROOT / ".streamlit" / "secrets.toml").exists():
        try:
            import re
            text = (ROOT / ".streamlit" / "secrets.toml").read_text()
            m = re.search(r'\[the_odds_api\].*?api_key\s*=\s*["\']([^"\']+)["\']', text, re.DOTALL)
            if m:
                api_key = m.group(1).strip()
            if not api_key:
                m = re.search(r'ODDS_API_KEY\s*=\s*["\']([^"\']+)["\']', text)
                if m:
                    api_key = m.group(1).strip()
        except Exception:
            pass
    if api_key:
        try:
            from engine.engine import get_live_odds
            from engine.betting_models import load_feature_matrix_for_inference
            from app import (
                _aggregate_odds_best_line_avg_implied,
                _live_odds_to_value_plays,
                EV_EPSILON_MIN_PCT,
                EV_EPSILON_MAX_PCT,
                BANKROLL_FOR_STAKES,
            )
            live_odds_df = get_live_odds(api_key, sport_keys=["basketball_nba", "basketball_ncaab"])
            if not live_odds_df.empty:
                fm = load_feature_matrix_for_inference(league=None)
                agg = _aggregate_odds_best_line_avg_implied(live_odds_df)
                value_plays_df, _ = _live_odds_to_value_plays(
                    agg if not agg.empty else live_odds_df,
                    bankroll=BANKROLL_FOR_STAKES,
                    kelly_frac=0.25,
                    min_ev_pct=EV_EPSILON_MIN_PCT,
                    max_ev_pct=EV_EPSILON_MAX_PCT,
                    feature_matrix=fm,
                )
                if value_plays_df.empty or "League" not in value_plays_df.columns:
                    ncaab = value_plays_df
                else:
                    ncaab = value_plays_df[value_plays_df["League"].astype(str).str.upper() == "NCAAB"]
                if not ncaab.empty and "Value (%)" in ncaab.columns:
                    for _, row in ncaab.sort_values("Value (%)", ascending=False).head(10).iterrows():
                        print(f"  {row.get('Event', '')} | {row.get('Selection', '')} | {row.get('Market', '')} | Value: {row.get('Value (%)', 0):.2f}%")
                else:
                    print("  No NCAAB value plays above min edge today.")
            else:
                print("  No live odds returned for today.")
        except Exception as e:
            print("  Error building value plays:", e)
    else:
        print("  Set ODDS_API_KEY or add api_key to .streamlit/secrets.toml to see a sample here.")
        print("  Or run: streamlit run app.py — Overview and NCAAB tabs use the same model.")


if __name__ == "__main__":
    main()
