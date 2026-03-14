#!/usr/bin/env python3
"""
Backtest NCAAB spread strategies on 2025 season.

Replays 2025 season game-by-game: for each game with closing spread and features,
computes predictions (BARTHAG*50 and XGBoost), edges vs closing spread, and
simulates $100 flat bets at -110 on value plays. Uses team stats from
training_data (season-level stats merged per game; no rolling point-in-time stats).
Compares:
  (1) BARTHAG*50 model, 3-pt threshold
  (2) XGBoost model, 3-pt threshold
  (3) XGBoost model, calibrated threshold (Step 6; configurable CALIBRATED_THRESHOLD)
  (4) XGBoost_3pt_Kelly: same picks as (2), stake = 25% Kelly % of current bankroll ($10k start), 0.5–5% clamp
  (5) Totals_3pt: Over/Under from xgboost_totals_ncaab.pkl when |edge| > 3 pts (requires closing_total in historical_games)

Outputs: summary table (picks, win%, ROI, max drawdown, Sharpe-equivalent),
detailed results to data/backtest_results.csv.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import numpy as np
import pandas as pd

from engine.utils import game_season_from_date

DATA_DIR = APP_ROOT / "data"
NCAAB_DIR = DATA_DIR / "ncaab"
MODELS_DIR = DATA_DIR / "models"
HISTORICAL_GAMES_PATH = NCAAB_DIR / "historical_games.csv"
TRAINING_DATA_PATH = NCAAB_DIR / "training_data.csv"
MODEL_PATH = MODELS_DIR / "xgboost_spread_ncaab.pkl"
TOTALS_MODEL_PATH = MODELS_DIR / "xgboost_totals_ncaab.pkl"
OUTPUT_CSV = DATA_DIR / "backtest_results.csv"
TOTALS_EDGE_THRESHOLD = 5.0  # Match predict_games; totals need larger edge
TOTALS_FEATURE_COLUMNS = [
    "home_ADJ_T", "away_ADJ_T",
    "home_ADJOE", "home_ADJDE", "away_ADJOE", "away_ADJDE",
    "tempo_diff", "combined_tempo", "combined_offense", "combined_defense",
    "is_neutral", "is_conference_tourney", "is_ncaa_tourney",
]

# Strategy config
BARTHAG_SCALE = 50.0
NEUTRAL_ADJUSTMENT = 3.0
THRESHOLD_3PT = 3.0
# Calibrated threshold from Step 6 (tune to maximize ROI on holdout if available)
CALIBRATED_THRESHOLD = 2.5
STAKE_USD = 100.0
ODDS_AMERICAN = -110  # win 100/110 * stake
TEST_SEASON = 2025
# Kelly strategy: same -110 decimal 1.909, 25% fractional Kelly, stake % clamp [0.5, 5.0]
COVER_PROB_PARAMS_PATH = MODELS_DIR / "cover_prob_params.json"
DECIMAL_ODDS_110 = 1.909
KELLY_FRACTION_PCT = 0.25
STAKE_PCT_MIN = 0.5
STAKE_PCT_MAX = 5.0
STARTING_BANKROLL = 10_000.0


def _load_2025_games() -> pd.DataFrame:
    """Load 2025 season games with closing spread; merge training_data features. One row per game."""
    if not HISTORICAL_GAMES_PATH.exists():
        raise FileNotFoundError(f"Missing {HISTORICAL_GAMES_PATH}")
    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {TRAINING_DATA_PATH}. Run scripts/build_training_data.py first.")

    games = pd.read_csv(HISTORICAL_GAMES_PATH)
    games["date"] = pd.to_datetime(games["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    games = games.dropna(subset=["date", "home_team", "away_team"])
    games["season"] = games["date"].apply(lambda d: game_season_from_date(d))
    games = games[games["season"] == TEST_SEASON].copy()
    games = games.dropna(subset=["closing_spread"])
    games["closing_spread"] = pd.to_numeric(games["closing_spread"], errors="coerce")
    games = games.dropna(subset=["closing_spread"])
    if "margin" not in games.columns and "home_score" in games.columns and "away_score" in games.columns:
        games["margin"] = games["home_score"].astype(float) - games["away_score"].astype(float)
    games = games.dropna(subset=["margin"])

    train = pd.read_csv(TRAINING_DATA_PATH)
    train["date"] = pd.to_datetime(train["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    train = train[train["season"] == TEST_SEASON].copy()
    # Merge on date + home_team + away_team (training_data has actual_margin; we keep closing_spread from games)
    merge_cols = ["date", "home_team", "away_team"]
    for c in merge_cols:
        if c not in train.columns or c not in games.columns:
            return pd.DataFrame()
    # Keep games columns: date, home_team, away_team, margin, closing_spread; add feature columns from train
    train_sub = train.drop(columns=["actual_margin"], errors="ignore")
    merged = games.merge(
        train_sub,
        on=merge_cols,
        how="inner",
        suffixes=("", "_y"),
    )
    # Drop duplicate columns from merge
    merged = merged[[c for c in merged.columns if not c.endswith("_y")]]
    # actual_total for totals strategy
    if "home_score" in merged.columns and "away_score" in merged.columns:
        merged["actual_total"] = merged["home_score"].astype(float) + merged["away_score"].astype(float)
    # Merge closing total (O/U) from historical_games if available
    merged = _merge_closing_total(merged)
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def _merge_closing_total(df: pd.DataFrame) -> pd.DataFrame:
    """Merge closing_total from historical_games if column exists."""
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
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    merged = df.merge(
        games,
        on=["date", "home_team", "away_team"],
        how="left",
        suffixes=("", "_y"),
    )
    return merged[[c for c in merged.columns if not c.endswith("_y")]]


def _edge_pts(pred_margin: float, closing_spread: float) -> float:
    """Edge in points. closing_spread = home spread (negative = home favored). Same convention as predict_games."""
    return pred_margin + float(closing_spread)


def _barthag_pred(row: pd.Series) -> float | None:
    """Predicted margin (home - away) from BARTHAG*scale; subtract NEUTRAL_ADJUSTMENT if neutral."""
    h = row.get("home_BARTHAG")
    a = row.get("away_BARTHAG")
    if pd.isna(h) or pd.isna(a):
        return None
    try:
        pred = (float(h) - float(a)) * BARTHAG_SCALE
    except (TypeError, ValueError):
        return None
    if row.get("is_neutral") in (1, True, "1", "yes") or (isinstance(row.get("is_neutral"), (int, float)) and row.get("is_neutral") not in (0, 0.0)):
        pred -= NEUTRAL_ADJUSTMENT
    return pred


def _xgboost_pred(row: pd.Series, model, feature_columns: list[str], calibration_shift: float) -> float | None:
    """Predicted margin from XGBoost model. Returns None if features missing."""
    try:
        vals = [float(row.get(c, 0) if pd.notna(row.get(c)) else 0.0) for c in feature_columns]
    except (TypeError, ValueError, KeyError):
        return None
    X_arr = np.array(vals, dtype=np.float64).reshape(1, -1)
    raw = float(model.predict(X_arr)[0])
    return raw - calibration_shift


def _value_play(edge: float, threshold: float) -> bool:
    return abs(edge) > threshold


def _pick_side(edge: float, threshold: float) -> str | None:
    """Return 'Home' or 'Away' if value play, else None."""
    if not _value_play(edge, threshold):
        return None
    return "Home" if edge > threshold else "Away"


def _covered_home(actual_margin: float, closing_spread: float) -> bool:
    """True if home covered. closing_spread is home spread (e.g. -7 = home favored by 7); home covers if margin > 7."""
    return float(actual_margin) > -float(closing_spread)


def _covered_away(actual_margin: float, closing_spread: float) -> bool:
    """True if away covered (margin < -closing_spread)."""
    return float(actual_margin) < -float(closing_spread)


def _won(pick: str, actual_margin: float, closing_spread: float) -> bool:
    if pick == "Home":
        return _covered_home(actual_margin, closing_spread)
    return _covered_away(actual_margin, closing_spread)


def _load_cover_prob_params() -> dict | None:
    """Load calibrated P(cover) logistic params from cover_prob_params.json. None if missing."""
    if not COVER_PROB_PARAMS_PATH.exists():
        return None
    try:
        with open(COVER_PROB_PARAMS_PATH) as f:
            data = json.load(f)
        k, intercept = data.get("k"), data.get("intercept")
        if k is None or intercept is None:
            return None
        return {"k": float(k), "intercept": float(intercept)}
    except Exception:
        return None


def _p_cover_and_kelly_stake_pct(
    pred_margin: float,
    closing_spread: float,
    pick: str,
    cover_params: dict,
) -> tuple[float, float]:
    """P(our pick covers) and suggested stake % (0.5--5). edge = pred_margin - closing_spread; logistic gives P(home covers)."""
    edge = pred_margin - float(closing_spread)
    z = cover_params["k"] * edge + cover_params["intercept"]
    try:
        p_home = 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        p_home = 1.0 if z > 0 else 0.0
    p_home = max(0.0, min(1.0, p_home))
    p_cover = p_home if pick == "Home" else (1.0 - p_home)
    denom = DECIMAL_ODDS_110 - 1.0
    kelly_raw = (p_cover * DECIMAL_ODDS_110 - 1.0) / denom if denom > 0 else 0.0
    kelly_raw = max(0.0, kelly_raw)
    fractional = kelly_raw * KELLY_FRACTION_PCT
    stake_pct = max(STAKE_PCT_MIN, min(STAKE_PCT_MAX, fractional * 100.0))
    return p_cover, stake_pct


def _profit(won: bool) -> float:
    if won:
        return STAKE_USD * (100.0 / abs(ODDS_AMERICAN))
    return -STAKE_USD


def _profit_kelly(won: bool, stake: float) -> float:
    """Profit for a single bet of size stake at -110."""
    if won:
        return stake * (100.0 / 110.0)
    return -stake


def _load_totals_model() -> tuple[object, list[str], float] | None:
    """Load xgboost_totals_ncaab.pkl; return (model, feature_columns, bias_correction) or None."""
    if not TOTALS_MODEL_PATH.exists():
        return None
    try:
        import joblib
        payload = joblib.load(TOTALS_MODEL_PATH)
        model = payload.get("model")
        cols = payload.get("feature_columns")
        if model is None or not cols:
            return None
        bias = float(payload.get("bias_correction", 0.0))
        return (model, list(cols), bias)
    except Exception:
        return None


def _totals_features_from_row(row: pd.Series, feature_columns: list[str]) -> np.ndarray | None:
    """Build totals feature vector from backtest row. Add derived columns if missing."""
    def _f(val, default: float = 0.0) -> float:
        if pd.isna(val):
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default
    out = {}
    out["home_ADJ_T"] = _f(row.get("home_ADJ_T"))
    out["away_ADJ_T"] = _f(row.get("away_ADJ_T"))
    out["home_ADJOE"] = _f(row.get("home_ADJOE"))
    out["home_ADJDE"] = _f(row.get("home_ADJDE"))
    out["away_ADJOE"] = _f(row.get("away_ADJOE"))
    out["away_ADJDE"] = _f(row.get("away_ADJDE"))
    out["tempo_diff"] = _f(row.get("tempo_diff"), out["home_ADJ_T"] - out["away_ADJ_T"])
    out["combined_tempo"] = out["home_ADJ_T"] + out["away_ADJ_T"]
    out["combined_offense"] = out["home_ADJOE"] + out["away_ADJOE"]
    out["combined_defense"] = out["home_ADJDE"] + out["away_ADJDE"]
    out["is_neutral"] = 1.0 if _f(row.get("is_neutral")) else 0.0
    out["is_conference_tourney"] = 1.0 if _f(row.get("is_conference_tourney")) else 0.0
    out["is_ncaa_tourney"] = 1.0 if _f(row.get("is_ncaa_tourney")) else 0.0
    try:
        return np.array([[out[c] for c in feature_columns]], dtype=np.float64)
    except KeyError:
        return None


def _totals_pick_and_won(pred_total: float, closing_total: float, actual_total: float) -> tuple[str | None, bool]:
    """Return (pick, won). pick is 'Over' or 'Under' or None; won is True if bet would win."""
    edge = pred_total - closing_total
    if edge > TOTALS_EDGE_THRESHOLD:
        pick = "Over"
        won = actual_total > closing_total
    elif edge < -TOTALS_EDGE_THRESHOLD:
        pick = "Under"
        won = actual_total < closing_total
    else:
        return None, False
    return pick, won


def run_backtest() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Load 2025 games, run three strategies, return (games_df, summary_df, detail_rows)."""
    df = _load_2025_games()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    # Load XGBoost model
    import joblib
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Run scripts/train_spread_model.py first.")
    payload = joblib.load(MODEL_PATH)
    model = payload["model"]
    feature_columns = payload.get("feature_columns", [])
    calibration_shift = payload.get("margin_calibration_shift", 0.0)
    # Ensure we only use columns present in df
    feature_columns = [c for c in feature_columns if c in df.columns]
    if not feature_columns:
        raise ValueError("No XGBoost feature columns found in training data.")

    # Ensure numeric
    for c in feature_columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    detail_rows = []
    strategies = [
        ("BARTHAG_3pt", "barthag", THRESHOLD_3PT),
        ("XGBoost_3pt", "xgb", THRESHOLD_3PT),
        ("XGBoost_calibrated", "xgb", CALIBRATED_THRESHOLD),
    ]

    for strat_name, model_type, thresh in strategies:
        cum_pl = 0.0
        peak = 0.0
        returns_list = []

        for idx, row in df.iterrows():
            date = row["date"]
            home = row["home_team"]
            away = row["away_team"]
            spread = float(row["closing_spread"])
            actual_margin = float(row["margin"])

            if model_type == "barthag":
                pred = _barthag_pred(row)
            else:
                pred = _xgboost_pred(row, model, feature_columns, calibration_shift)

            if pred is None:
                continue

            edge = _edge_pts(pred, spread)
            pick = _pick_side(edge, thresh)
            if pick is None:
                continue

            won = _won(pick, actual_margin, spread)
            pl = _profit(won)
            cum_pl += pl
            peak = max(peak, cum_pl)
            drawdown = peak - cum_pl
            ret = pl / STAKE_USD
            returns_list.append(ret)

            detail_rows.append({
                "strategy": strat_name,
                "date": date,
                "home_team": home,
                "away_team": away,
                "closing_spread": spread,
                "actual_margin": actual_margin,
                "pred_margin": pred,
                "edge_pts": round(edge, 2),
                "pick": pick,
                "won": won,
                "profit": round(pl, 2),
                "cumulative_pl": round(cum_pl, 2),
                "drawdown": round(drawdown, 2),
            })

    # Fourth strategy: XGBoost_3pt_Kelly — same picks as XGBoost_3pt, stake = Kelly % of current bankroll
    cover_params = _load_cover_prob_params()
    if cover_params is not None:
        bankroll = STARTING_BANKROLL
        peak = bankroll

        for idx, row in df.iterrows():
            date = row["date"]
            home = row["home_team"]
            away = row["away_team"]
            spread = float(row["closing_spread"])
            actual_margin = float(row["margin"])

            pred = _xgboost_pred(row, model, feature_columns, calibration_shift)
            if pred is None:
                continue

            edge = _edge_pts(pred, spread)
            pick = _pick_side(edge, THRESHOLD_3PT)
            if pick is None:
                continue

            p_cover, stake_pct = _p_cover_and_kelly_stake_pct(pred, spread, pick, cover_params)
            stake = bankroll * (stake_pct / 100.0)
            won = _won(pick, actual_margin, spread)
            pl = _profit_kelly(won, stake)
            bankroll += pl
            peak = max(peak, bankroll)
            drawdown = peak - bankroll

            detail_rows.append({
                "strategy": "XGBoost_3pt_Kelly",
                "date": date,
                "home_team": home,
                "away_team": away,
                "closing_spread": spread,
                "actual_margin": actual_margin,
                "pred_margin": pred,
                "edge_pts": round(edge, 2),
                "pick": pick,
                "won": won,
                "profit": round(pl, 2),
                "cumulative_pl": round(bankroll - STARTING_BANKROLL, 2),
                "drawdown": round(drawdown, 2),
                "stake_used": round(stake, 2),
            })

    # Fifth strategy: Totals_3pt — Over/Under from totals model when |edge| > 3 (requires closing_total in data)
    totals_model_payload = _load_totals_model()
    if totals_model_payload is not None and "closing_total" in df.columns and "actual_total" in df.columns:
        totals_model, totals_feature_columns, totals_bias = totals_model_payload
        # Ensure feature columns exist in df (derived cols may be missing)
        for c in ("combined_tempo", "combined_offense", "combined_defense"):
            if c not in df.columns and "home_ADJ_T" in df.columns:
                if c == "combined_tempo":
                    df["combined_tempo"] = df["home_ADJ_T"].fillna(0) + df["away_ADJ_T"].fillna(0)
                elif c == "combined_offense":
                    df["combined_offense"] = df["home_ADJOE"].fillna(0) + df["away_ADJOE"].fillna(0)
                elif c == "combined_defense":
                    df["combined_defense"] = df["home_ADJDE"].fillna(0) + df["away_ADJDE"].fillna(0)
        cum_pl_totals = 0.0
        peak_totals = 0.0
        for idx, row in df.iterrows():
            date = row["date"]
            home = row["home_team"]
            away = row["away_team"]
            closing_total = row.get("closing_total")
            actual_total = row.get("actual_total")
            if pd.isna(closing_total) or pd.isna(actual_total):
                continue
            closing_total = float(closing_total)
            actual_total = float(actual_total)
            X = _totals_features_from_row(row, totals_feature_columns)
            if X is None:
                continue
            try:
                pred_total = float(totals_model.predict(X)[0]) - totals_bias
            except Exception:
                continue
            pick, won = _totals_pick_and_won(pred_total, closing_total, actual_total)
            if pick is None:
                continue
            pl = _profit(won)
            cum_pl_totals += pl
            peak_totals = max(peak_totals, cum_pl_totals)
            drawdown = peak_totals - cum_pl_totals
            detail_rows.append({
                "strategy": "Totals_3pt",
                "date": date,
                "home_team": home,
                "away_team": away,
                "closing_spread": np.nan,
                "actual_margin": np.nan,
                "pred_margin": pred_total,
                "edge_pts": round(pred_total - closing_total, 2),
                "pick": pick,
                "won": won,
                "profit": round(pl, 2),
                "cumulative_pl": round(cum_pl_totals, 2),
                "drawdown": round(drawdown, 2),
            })

    # Build summary per strategy
    all_strategies = list(strategies) + ([("XGBoost_3pt_Kelly", "xgb", THRESHOLD_3PT)] if cover_params is not None else [])
    if totals_model_payload is not None:
        all_strategies.append(("Totals_3pt", "totals", TOTALS_EDGE_THRESHOLD))
    summary_rows = []
    for strat_name, _, thresh in all_strategies:
        rows = [r for r in detail_rows if r["strategy"] == strat_name]
        n = len(rows)
        if n == 0:
            summary_rows.append({
                "strategy": strat_name,
                "total_picks": 0,
                "wins": 0,
                "win_pct": None,
                "total_staked": 0.0,
                "total_profit": 0.0,
                "roi_pct": None,
                "max_drawdown": 0.0,
                "max_drawdown_pct": None,
                "sharpe_equiv": None,
            })
            continue
        wins = sum(1 for r in rows if r["won"])
        if strat_name == "XGBoost_3pt_Kelly":
            total_staked = sum(r.get("stake_used", 0) for r in rows)
            returns = [r["profit"] / r["stake_used"] for r in rows if r.get("stake_used", 0) > 0]
        else:
            total_staked = n * STAKE_USD
            returns = [r["profit"] / STAKE_USD for r in rows]
        total_profit = sum(r["profit"] for r in rows)
        roi = (total_profit / total_staked * 100) if total_staked else 0.0
        if strat_name == "XGBoost_3pt_Kelly" and rows:
            # Max drawdown as % of peak bankroll (dollar drawdown can be huge with compounding)
            peaks = [STARTING_BANKROLL + r["cumulative_pl"] + r["drawdown"] for r in rows]
            max_dd = max(r["drawdown"] for r in rows)
            max_dd_pct = max(100.0 * r["drawdown"] / p for r, p in zip(rows, peaks) if p > 0) if peaks else 0.0
        else:
            max_dd = max(r["drawdown"] for r in rows) if rows else 0.0
            max_dd_pct = None
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret and std_ret > 1e-10:
            sharpe_equiv = (mean_ret / std_ret) * np.sqrt(len(returns))
        else:
            sharpe_equiv = 0.0 if mean_ret == 0 else (np.inf if mean_ret > 0 else -np.inf)

        summary_rows.append({
            "strategy": strat_name,
            "total_picks": n,
            "wins": wins,
            "win_pct": round(100.0 * wins / n, 1),
            "total_staked": total_staked,
            "total_profit": round(total_profit, 2),
            "roi_pct": round(roi, 1),
            "max_drawdown": round(max_dd, 2),
            "max_drawdown_pct": round(max_dd_pct, 2) if max_dd_pct is not None else None,
            "sharpe_equiv": round(sharpe_equiv, 3),
        })

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)
    return df, summary_df, detail_rows


def main() -> int:
    print("NCAAB spread backtest — 2025 season")
    print("Strategies: (1) BARTHAG*50 @ 3pt (2) XGBoost @ 3pt (3) XGBoost @ calibrated (4) XGBoost_3pt_Kelly (5) Totals_3pt")
    print()

    try:
        games_df, summary_df, detail_rows = run_backtest()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if games_df.empty:
        print("No 2025 games with closing spread and features. Check historical_games.csv and training_data.csv.")
        return 1

    print(f"Games in backtest: {len(games_df)} (2025 season, with closing spread + features)\n")
    print("Summary")
    print("=" * 100)
    print(f"{'Strategy':<22} | {'Picks':>6} | {'Win%':>6} | {'ROI%':>7} | {'MaxDD':>10} | {'MaxDD%':>7} | {'Sharpe':>8}")
    print("=" * 100)
    for _, r in summary_df.iterrows():
        wp = f"{r['win_pct']:.1f}" if r["win_pct"] is not None and not pd.isna(r["win_pct"]) else "—"
        roi = f"{r['roi_pct']:.1f}" if r["roi_pct"] is not None and not pd.isna(r["roi_pct"]) else "—"
        sh = f"{r['sharpe_equiv']:.3f}" if r["sharpe_equiv"] is not None and not pd.isna(r["sharpe_equiv"]) else "—"
        dd = r["max_drawdown"]
        dd_str = f"{dd:,.0f}" if dd < 1e9 else ">1e9"
        dd_pct_val = r.get("max_drawdown_pct")
        dd_pct = f"{dd_pct_val:.1f}%" if dd_pct_val is not None and not (isinstance(dd_pct_val, float) and pd.isna(dd_pct_val)) else "—"
        print(f"{r['strategy']:<22} | {r['total_picks']:>6} | {wp:>6} | {roi:>7} | {dd_str:>10} | {dd_pct:>7} | {sh:>8}")
    print("=" * 100)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if detail_rows:
        pd.DataFrame(detail_rows).to_csv(OUTPUT_CSV, index=False)
        print(f"\nDetailed results saved to {OUTPUT_CSV} ({len(detail_rows)} rows)")
    else:
        print("\nNo value-play rows to save.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
