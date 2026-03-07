"""
Three specialized XGBoost models: spreads, totals (over/under), and moneylines.
Each is trained on bet-type-specific features, evaluated independently, and used to
route predictions. Falls back to heuristics when models are not trained or features missing.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Feature column sets per bet type (only columns that may exist in games_with_team_stats or merged situational).
# Totals: pace and scoring efficiency matter most.
TOTALS_FEATURE_COLUMNS = [
    "home_pace", "away_pace",
    "home_offensive_rating", "away_offensive_rating",
    "home_defensive_rating", "away_defensive_rating",
    "home_true_shooting_pct", "away_true_shooting_pct",
    "home_off_eff_roll10", "away_off_eff_roll10",
    "home_def_eff_roll10", "away_def_eff_roll10",
    "home_pace_roll10", "away_pace_roll10",
    "home_ts_pct_roll10", "away_ts_pct_roll10",
]
# Spreads: defensive ratings, situational spots, and momentum (heavily weighted by sharp bettors).
# home_is_favorite included for SHAP bias audit; remove and retrain if it has disproportionate importance.
SPREAD_FEATURE_COLUMNS = [
    "home_defensive_rating", "away_defensive_rating",
    "home_offensive_rating", "away_offensive_rating",
    "home_days_rest", "away_days_rest",
    "home_is_b2b", "away_is_b2b",
    "home_travel_miles", "away_travel_miles",
    "home_win_pct_last30", "away_win_pct_last30",
    "home_def_eff_roll10", "away_def_eff_roll10",
    "home_off_eff_roll10", "away_off_eff_roll10",
    "home_streak", "away_streak",
    "home_ats_pct_last10", "away_ats_pct_last10",
    "home_over_pct_last10", "away_over_pct_last10",
    "home_pt_diff_trend_last5", "away_pt_diff_trend_last5",
    "home_ats_pct_home_season", "home_ats_pct_road_season",
    "away_ats_pct_home_season", "away_ats_pct_road_season",
    "line_move_direction", "line_move_magnitude", "sharp_money_indicator",
    "home_is_favorite",
]
# NCAAB spread: same as SPREAD_FEATURE_COLUMNS plus seed columns when available (March context).
NCAAB_SPREAD_FEATURE_COLUMNS = SPREAD_FEATURE_COLUMNS + ["home_seed", "away_seed"]

# NCAAB spread trained only on games with KenPom stats + closing line: KenPom + situational.
NCAAB_KENPOM_SPREAD_FEATURE_COLUMNS = [
    "home_ADJOE", "away_ADJOE", "home_ADJDE", "away_ADJDE",
    "home_BARTHAG", "away_BARTHAG",
    "home_EFG_O", "away_EFG_O", "home_EFG_D", "away_EFG_D",
    "home_TOR", "away_TOR", "home_ORB", "away_ORB",
    "home_ADJ_T", "away_ADJ_T", "home_SEED", "away_SEED",
    "home_days_rest", "away_days_rest", "home_is_b2b", "away_is_b2b",
]

# Moneylines: power-style and situational.
MONEYLINE_FEATURE_COLUMNS = [
    "home_offensive_rating", "away_offensive_rating",
    "home_defensive_rating", "away_defensive_rating",
    "home_days_rest", "away_days_rest",
    "home_is_b2b", "away_is_b2b",
    "home_win_pct_last30", "away_win_pct_last30",
    "home_off_eff_roll10", "away_off_eff_roll10",
    "home_def_eff_roll10", "away_def_eff_roll10",
    "line_move_direction", "line_move_magnitude", "sharp_money_indicator",
]

MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
SPREAD_MODEL_PATH = MODELS_DIR / "xgboost_spread.pkl"
TOTALS_MODEL_PATH = MODELS_DIR / "xgboost_totals.pkl"
MONEYLINE_MODEL_PATH = MODELS_DIR / "xgboost_moneyline.pkl"
SPREAD_MODEL_PATH_NCAAB = MODELS_DIR / "xgboost_spread_ncaab.pkl"
TOTALS_MODEL_PATH_NCAAB = MODELS_DIR / "xgboost_totals_ncaab.pkl"
MONEYLINE_MODEL_PATH_NCAAB = MODELS_DIR / "xgboost_moneyline_ncaab.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"


def _ensure_models_dir() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def _select_features(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Return df with only columns that exist; fill NaN with 0. Returns (X, used_cols)."""
    used = [c for c in columns if c in df.columns]
    if not used:
        return pd.DataFrame(), []
    X = df[used].copy()
    for c in used:
        if X[c].dtype in (np.float64, np.int64, np.float32, np.int32):
            X[c] = X[c].fillna(0.0)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    return X, used


def get_training_data(
    db_path: Optional[Path] = None,
    league: Optional[str] = None,
    merge_situational: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load completed games (with home_score, away_score) from games_with_team_stats,
    optionally merge situational features. Returns (df_with_features, df_spread_ready, df_ml_ready)
    where df has targets: margin (home - away), total_pts (home + away), home_win (0/1).
    """
    from .sportsref_stats import load_merged_games_from_sqlite
    from .situational_features import load_situational_features_from_sqlite

    path = db_path or (Path(__file__).resolve().parent.parent / "data" / "espn.db")
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df = load_merged_games_from_sqlite(league=league, db_path=path)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for col in ("home_score", "away_score"):
        if col not in df.columns:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df = df.copy()
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"])
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df["margin"] = df["home_score"] - df["away_score"]
    df["total_pts"] = df["home_score"] + df["away_score"]
    df["home_win"] = (df["margin"] > 0).astype(int)
    if merge_situational:
        try:
            sit = load_situational_features_from_sqlite(league=league, db_path=path)
            if not sit.empty and "league" in sit.columns and "game_id" in sit.columns:
                drop = [c for c in sit.columns if c in df.columns and c not in ("league", "game_id")]
                sit = sit.drop(columns=drop, errors="ignore")
                df = df.merge(sit, on=["league", "game_id"], how="left", suffixes=("", "_sit"))
                df = df.loc[:, ~df.columns.duplicated()]
        except Exception:
            pass
    # Favorite flag for spread model (proxy: home has higher offensive rating). Used for SHAP bias audit.
    if "home_offensive_rating" in df.columns and "away_offensive_rating" in df.columns:
        df["home_is_favorite"] = (
            (df["home_offensive_rating"].fillna(0) - df["away_offensive_rating"].fillna(0)) > 0
        ).astype(int)
    else:
        df["home_is_favorite"] = 0
    return df, df, df


def train_spread_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    db_path: Optional[Path] = None,
    spread_features: Optional[list[str]] = None,
    model_path: Optional[Path] = None,
) -> dict[str, float]:
    """Train XGBoost regressor to predict margin (home - away). Evaluate with MAE and R². Returns metrics.
    If spread_features is provided, use it instead of SPREAD_FEATURE_COLUMNS (e.g. to debias by removing home_is_favorite).
    If model_path is provided, save there; else SPREAD_MODEL_PATH."""
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score
    except ImportError:
        return {}
    cols = spread_features if spread_features is not None else SPREAD_FEATURE_COLUMNS
    X, used = _select_features(df, cols)
    if X.empty or "margin" not in df.columns:
        return {}
    y = df["margin"].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=random_state)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    _ensure_models_dir()
    out_path = model_path or SPREAD_MODEL_PATH
    with open(out_path, "wb") as f:
        pickle.dump({"model": model, "feature_columns": used}, f)
    return {"spread_mae": round(mae, 4), "spread_r2": round(r2, 4), "spread_n_features": len(used)}


def train_totals_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    db_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
) -> dict[str, float]:
    """Train XGBoost regressor to predict total_pts. Returns metrics. If model_path given, save there."""
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score
    except ImportError:
        return {}
    X, used = _select_features(df, TOTALS_FEATURE_COLUMNS)
    if X.empty or "total_pts" not in df.columns:
        return {}
    y = df["total_pts"].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=random_state)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    _ensure_models_dir()
    out_path = model_path or TOTALS_MODEL_PATH
    with open(out_path, "wb") as f:
        pickle.dump({"model": model, "feature_columns": used}, f)
    return {"totals_mae": round(mae, 4), "totals_r2": round(r2, 4), "totals_n_features": len(used)}


def train_moneyline_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    db_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
) -> dict[str, float]:
    """Train XGBoost classifier for home_win (0/1). Returns metrics (accuracy, log_loss). If model_path given, save there."""
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, log_loss
    except ImportError:
        return {}
    X, used = _select_features(df, MONEYLINE_FEATURE_COLUMNS)
    if X.empty or "home_win" not in df.columns:
        return {}
    y = df["home_win"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=random_state, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    pred_proba = model.predict_proba(X_test)[:, 1]
    pred_label = model.predict(X_test)
    acc = accuracy_score(y_test, pred_label)
    ll = log_loss(y_test, pred_proba)
    _ensure_models_dir()
    out_path = model_path or MONEYLINE_MODEL_PATH
    with open(out_path, "wb") as f:
        pickle.dump({"model": model, "feature_columns": used}, f)
    return {"moneyline_accuracy": round(acc, 4), "moneyline_log_loss": round(ll, 4), "moneyline_n_features": len(used)}


def _spread_shap_favorite_rank(payload: dict, df: pd.DataFrame) -> Optional[int]:
    """Return 1-based rank of home_is_favorite by mean |SHAP| (1 = most important). None if not computable."""
    try:
        import shap
    except ImportError:
        return None
    model = payload.get("model")
    used = payload.get("feature_columns") or []
    if model is None or "home_is_favorite" not in used:
        return None
    X, _ = _select_features(df, used)
    if X.empty or len(X) > 500:
        X = X.head(500)  # limit for speed
    if X.empty:
        return None
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[0] if len(sv) else sv
    mean_abs = np.abs(sv).mean(axis=0)
    idx = used.index("home_is_favorite")
    rank = int((mean_abs >= mean_abs[idx]).sum())
    return rank


def train_all_models(
    db_path: Optional[Path] = None,
    league: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    debias_spread_if_favorite_dominant: bool = True,
) -> dict[str, float]:
    """Load training data, train all three models, save metrics. Returns combined metrics dict.
    If debias_spread_if_favorite_dominant, after training the spread model we run SHAP; if home_is_favorite
    is in the top 3 by importance we retrain the spread model without it."""
    df, _, _ = get_training_data(db_path=db_path, league=league, merge_situational=True)
    if df.empty or len(df) < 50:
        return {}
    metrics = {}
    metrics.update(train_spread_model(df, test_size=test_size, random_state=random_state))
    if debias_spread_if_favorite_dominant:
        payload = load_model(SPREAD_MODEL_PATH)
        if payload:
            rank = _spread_shap_favorite_rank(payload, df)
            if rank is not None and rank <= 3:
                spread_without_fav = [c for c in SPREAD_FEATURE_COLUMNS if c != "home_is_favorite"]
                metrics.update(
                    train_spread_model(
                        df, test_size=test_size, random_state=random_state, spread_features=spread_without_fav
                    )
                )
    metrics.update(train_totals_model(df, test_size=test_size, random_state=random_state))
    metrics.update(train_moneyline_model(df, test_size=test_size, random_state=random_state))
    _ensure_models_dir()
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def train_ncaab_models(
    db_path: Optional[Path] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, float]:
    """Train spread, totals, and moneyline models on NCAAB only; save to *_ncaab.pkl paths."""
    df, _, _ = get_training_data(db_path=db_path, league="ncaab", merge_situational=True)
    if df.empty or len(df) < 50:
        return {}
    metrics = {}
    metrics.update(
        train_spread_model(
            df,
            test_size=test_size,
            random_state=random_state,
            model_path=SPREAD_MODEL_PATH_NCAAB,
            spread_features=NCAAB_SPREAD_FEATURE_COLUMNS,
        )
    )
    metrics.update(
        train_totals_model(df, test_size=test_size, random_state=random_state, model_path=TOTALS_MODEL_PATH_NCAAB)
    )
    metrics.update(
        train_moneyline_model(df, test_size=test_size, random_state=random_state, model_path=MONEYLINE_MODEL_PATH_NCAAB)
    )
    return metrics


def load_model(path: Path) -> Optional[dict]:
    """Load pickle with model and feature_columns. Returns None if missing or error."""
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _spread_model_path_for_league(league: Optional[str]) -> Path:
    """Return NCAAB spread model path when league is ncaab and file exists; else default."""
    if str(league or "").strip().lower() == "ncaab" and SPREAD_MODEL_PATH_NCAAB.exists():
        return SPREAD_MODEL_PATH_NCAAB
    return SPREAD_MODEL_PATH


# Map Live Odds (Odds API) team names to feature-matrix (ESPN/KenPom) names for NCAAB lookup.
# Keys: normalized (lower, stripped); values: name to use when matching feature_matrix.
NCAAB_ODDS_TO_FEATURE_TEAM: dict[str, str] = {
    "north carolina tar heels": "North Carolina",
    "unc": "North Carolina",
    "duke blue devils": "Duke",
    "kentucky wildcats": "Kentucky",
    "florida gators": "Florida",
    "louisville cardinals": "Louisville",
    "miami hurricanes": "Miami (FL)",
    "miami (fl) hurricanes": "Miami (FL)",
    "byu cougars": "BYU",
    "brigham young cougars": "BYU",
    "texas tech red raiders": "Texas Tech",
    "alabama crimson tide": "Alabama",
    "auburn tigers": "Auburn",
    "arizona wildcats": "Arizona",
    "colorado buffaloes": "Colorado",
    "wisconsin badgers": "Wisconsin",
    "purdue boilermakers": "Purdue",
    "saint louis billikens": "Saint Louis",
    "st louis billikens": "Saint Louis",
    "george mason patriots": "George Mason",
    "iowa state cyclones": "Iowa State",
    "arizona state sun devils": "Arizona State",
    "kansas jayhawks": "Kansas",
    "kansas state wildcats": "Kansas State",
    "vanderbilt commodores": "Vanderbilt",
    "tennessee volunteers": "Tennessee",
    "connecticut huskies": "UConn",
    "uconn huskies": "UConn",
    "marquette golden eagles": "Marquette",
}


def _normalize_team_for_lookup(name: str) -> str:
    """Normalize team name for lookup (strip, lower)."""
    if not name or not isinstance(name, str):
        return ""
    return name.strip().lower()


def _team_names_match(live_name: str, fm_name: str) -> bool:
    """True if live (Odds API) name matches feature-matrix name (exact, contains, or mapped)."""
    live = _normalize_team_for_lookup(live_name)
    fm = _normalize_team_for_lookup(fm_name)
    if not live or not fm:
        return False
    if live == fm:
        return True
    if live in fm or fm in live:
        return True
    mapped = NCAAB_ODDS_TO_FEATURE_TEAM.get(live)
    if mapped and _normalize_team_for_lookup(mapped) == fm:
        return True
    return False


def get_spread_predicted_margin(
    row: Optional[pd.Series],
    market_spread: float,
    league: Optional[str] = None,
    home_team: Optional[str] = None,
    away_team: Optional[str] = None,
) -> Optional[float]:
    """
    Return the model's predicted margin (home - away) for the spread, or None.
    Inference assumes model output is home - away. For NCAAB, legacy models saved without
    margin_home_minus_away=True are assumed to have been trained on away - home; we negate to get home - away.
    Only works for margin (regressor) models; returns None for classifier.
    Used for debug (e.g. line_error = pred_margin - market_spread).
    Optional home_team/away_team used for "Missing features for ..." logging when row is None or X.empty.
    """
    def _team_label() -> str:
        h = home_team if home_team is not None else (row.get("home_team_name") if row is not None else None)
        a = away_team if away_team is not None else (row.get("away_team_name") if row is not None else None)
        if h or a:
            return f"{h or '?'} / {a or '?'}"
        return "(unknown teams)"

    if row is None:
        logging.getLogger(__name__).warning("Missing features for %s", _team_label())
        return None
    path = _spread_model_path_for_league(league)
    payload = load_model(path)
    if payload is None or payload.get("is_underdog_cover_classifier", False):
        return None
    model = payload["model"]
    cols = payload.get("feature_columns", [])
    X, _ = _select_features(pd.DataFrame([row]), cols)
    if X.empty:
        logging.getLogger(__name__).warning("Missing features for %s", _team_label())
        return None
    raw = float(model.predict(X)[0])
    # Standard: model output = home - away. Legacy NCAAB models may have been trained as away - home; negate.
    if str(league or "").strip().lower() == "ncaab" and not payload.get("margin_home_minus_away", False):
        raw = -raw
    pred = raw - payload.get("margin_calibration_shift", 0.0)
    return pred


def _totals_model_path_for_league(league: Optional[str]) -> Path:
    if str(league or "").strip().lower() == "ncaab" and TOTALS_MODEL_PATH_NCAAB.exists():
        return TOTALS_MODEL_PATH_NCAAB
    return TOTALS_MODEL_PATH


def _moneyline_model_path_for_league(league: Optional[str]) -> Path:
    if str(league or "").strip().lower() == "ncaab" and MONEYLINE_MODEL_PATH_NCAAB.exists():
        return MONEYLINE_MODEL_PATH_NCAAB
    return MONEYLINE_MODEL_PATH


def _spread_row_with_line_features(row, market_spread: float):
    """Add line_value and public_fade_spot to row for NCAAB underdog-cover classifier. Returns a copy (Series or dict)."""
    if isinstance(row, pd.Series):
        r = row.copy()
    else:
        r = dict(row) if row is not None else {}
    try:
        h = float(r.get("home_BARTHAG") or 0)
        a = float(r.get("away_BARTHAG") or 0)
        barthag_diff_pts = (h - a) * 100.0
    except (TypeError, ValueError):
        barthag_diff_pts = 0.0
    if abs(barthag_diff_pts) >= 0.5:
        r["line_value"] = float(np.clip(market_spread / barthag_diff_pts, -10, 10))
    else:
        r["line_value"] = 0.0
    r["public_fade_spot"] = 1 if abs(barthag_diff_pts) > 8.0 else 0
    return r


def predict_spread_prob(
    row: pd.Series,
    market_spread: float,
    we_cover_favorite: bool,
    fallback_prob: Optional[float] = None,
    league: Optional[str] = None,
) -> float:
    """
    Predict P(cover) for the spread. market_spread is home spread in home-minus-away terms (e.g. +3.5 = home favored by 3.5).
    we_cover_favorite True = we're on the favorite (home/away favored).
    Assumes model output is home - away when margin regressor. Uses NCAAB model when league is ncaab.
    NCAAB model may be binary classifier or margin regressor; legacy regressors without margin_home_minus_away are negated.
    """
    path = _spread_model_path_for_league(league)
    payload = load_model(path)
    if payload is None:
        return fallback_prob if fallback_prob is not None else 0.5
    model = payload["model"]
    cols = payload.get("feature_columns", [])
    is_classifier = payload.get("is_underdog_cover_classifier", False)
    if is_classifier and row is not None:
        row = _spread_row_with_line_features(row, market_spread)
    X, _ = _select_features(pd.DataFrame([row]), cols)
    if X.empty:
        return fallback_prob if fallback_prob is not None else 0.5
    if is_classifier:
        prob_underdog_cover = float(model.predict_proba(X)[0, 1])
        home_underdog = market_spread > 0
        prob_home_cover = prob_underdog_cover if home_underdog else (1.0 - prob_underdog_cover)
        if we_cover_favorite:
            raw = (1.0 - prob_home_cover) if home_underdog else prob_home_cover
        else:
            raw = prob_home_cover if home_underdog else (1.0 - prob_home_cover)
        return float(np.clip(raw, 0.02, 0.98))
    raw_margin = float(model.predict(X)[0])
    if str(league or "").strip().lower() == "ncaab" and not payload.get("margin_home_minus_away", False):
        raw_margin = -raw_margin
    pred_margin = raw_margin - payload.get("margin_calibration_shift", 0.0)
    diff = pred_margin - market_spread
    k = 0.35
    prob_home_cover = 1.0 / (1.0 + np.exp(-k * diff))
    home_favored = market_spread <= 0
    if we_cover_favorite:
        raw = prob_home_cover if home_favored else (1.0 - prob_home_cover)
        is_favorite = True
    else:
        raw = (1.0 - prob_home_cover) if home_favored else prob_home_cover
        is_favorite = False
    shift_fav = payload.get("calibration_shift_fav")
    shift_dog = payload.get("calibration_shift_dog")
    if shift_fav is not None and shift_dog is not None:
        raw = raw + (float(shift_fav) if is_favorite else float(shift_dog))
    return float(np.clip(raw, 0.02, 0.98))


def predict_totals_prob(
    row: pd.Series,
    market_total: float,
    prefer_over: bool,
    fallback_prob: Optional[float] = None,
    league: Optional[str] = None,
) -> float:
    """Predict P(over) or P(under) from totals model. prefer_over True = Over. Uses NCAAB model when league is ncaab."""
    path = _totals_model_path_for_league(league)
    payload = load_model(path)
    if payload is None:
        return fallback_prob if fallback_prob is not None else 0.5
    model = payload["model"]
    cols = payload.get("feature_columns", [])
    X, _ = _select_features(pd.DataFrame([row]), cols)
    if X.empty:
        return fallback_prob if fallback_prob is not None else 0.5
    pred_total = float(model.predict(X)[0])
    diff = pred_total - market_total
    k = 0.12
    prob_over = 1.0 / (1.0 + np.exp(-k * diff))
    if prefer_over:
        return float(np.clip(prob_over, 0.02, 0.98))
    return float(np.clip(1.0 - prob_over, 0.02, 0.98))


def predict_moneyline_prob(
    row: pd.Series,
    selection_is_home: bool,
    fallback_prob: Optional[float] = None,
    league: Optional[str] = None,
) -> float:
    """Predict P(selection wins). selection_is_home True = home team. Uses NCAAB model when league is ncaab."""
    path = _moneyline_model_path_for_league(league)
    payload = load_model(path)
    if payload is None:
        return fallback_prob if fallback_prob is not None else 0.5
    model = payload["model"]
    cols = payload.get("feature_columns", [])
    X, _ = _select_features(pd.DataFrame([row]), cols)
    if X.empty:
        return fallback_prob if fallback_prob is not None else 0.5
    prob_home = float(model.predict_proba(X)[0, 1])
    if selection_is_home:
        return float(np.clip(prob_home, 0.02, 0.98))
    return float(np.clip(1.0 - prob_home, 0.02, 0.98))


# League default totals for consensus third vote (totals)
NBA_LEAGUE_AVG_TOTAL = 220.0
NCAAB_LEAGUE_AVG_TOTAL = 140.0


def _spread_direction_home_cover(
    row: Optional[pd.Series], market_spread: float, league: Optional[str] = None
) -> Optional[bool]:
    """True if model says home covers. None if model unavailable. Uses NCAAB model when league is ncaab."""
    if row is None:
        return None
    path = _spread_model_path_for_league(league)
    payload = load_model(path)
    if payload is None:
        return None
    model = payload["model"]
    cols = payload.get("feature_columns", [])
    is_classifier = payload.get("is_underdog_cover_classifier", False)
    if is_classifier:
        row = _spread_row_with_line_features(row, market_spread)
    X, _ = _select_features(pd.DataFrame([row]), cols)
    if X.empty:
        return None
    if is_classifier:
        prob_underdog_cover = float(model.predict_proba(X)[0, 1])
        home_underdog = market_spread > 0
        prob_home_cover = prob_underdog_cover if home_underdog else (1.0 - prob_underdog_cover)
        return prob_home_cover > 0.5
    raw_margin = float(model.predict(X)[0])
    if str(league or "").strip().lower() == "ncaab" and not payload.get("margin_home_minus_away", False):
        raw_margin = -raw_margin
    pred_margin = raw_margin - payload.get("margin_calibration_shift", 0.0)
    diff = pred_margin - market_spread
    prob_home_cover = 1.0 / (1.0 + np.exp(-0.35 * diff))
    return prob_home_cover > 0.5


def _moneyline_direction_home(row: Optional[pd.Series], league: Optional[str] = None) -> Optional[bool]:
    """True if model says home wins. None if model unavailable. Uses NCAAB model when league is ncaab."""
    if row is None:
        return None
    path = _moneyline_model_path_for_league(league)
    payload = load_model(path)
    if payload is None:
        return None
    model = payload["model"]
    cols = payload.get("feature_columns", [])
    X, _ = _select_features(pd.DataFrame([row]), cols)
    if X.empty:
        return None
    prob_home = float(model.predict_proba(X)[0, 1])
    return prob_home > 0.5


def _totals_direction_over(
    row: Optional[pd.Series], market_total: float, league: Optional[str] = None
) -> Optional[bool]:
    """True if model says over. None if model unavailable. Uses NCAAB model when league is ncaab."""
    if row is None:
        return None
    path = _totals_model_path_for_league(league)
    payload = load_model(path)
    if payload is None:
        return None
    model = payload["model"]
    cols = payload.get("feature_columns", [])
    X, _ = _select_features(pd.DataFrame([row]), cols)
    if X.empty:
        return None
    pred_total = float(model.predict(X)[0])
    return pred_total > market_total


def consensus_spread(
    feature_row: Optional[pd.Series],
    market_spread: float,
    we_cover_favorite: bool,
    in_house_spread: float,
    fallback_prob: float,
    league: Optional[str] = None,
) -> tuple[float, bool]:
    """
    Return (model_prob, passes_consensus). Only pass when >= 2 of 3 sub-models agree on direction.
    Sub-models: (1) XGBoost spread, (2) in-house spread, (3) XGBoost moneyline. Uses NCAAB models when league is ncaab.
    """
    prob = predict_spread_prob(feature_row, market_spread, we_cover_favorite, fallback_prob=fallback_prob, league=league)
    recommended_home_cover = (we_cover_favorite and market_spread <= 0) or (not we_cover_favorite and market_spread > 0)
    v1 = _spread_direction_home_cover(feature_row, market_spread, league=league)
    v2 = in_house_spread > market_spread  # in-house likes home to cover when our line is higher than market
    v3 = _moneyline_direction_home(feature_row, league=league)
    votes = [v for v in (v1, v2, v3) if v is not None]
    if len(votes) < 2:
        return (prob, True)  # not enough models, don't suppress
    agree = sum(1 for v in votes if v == recommended_home_cover)
    return (prob, agree >= 2)


def consensus_totals(
    feature_row: Optional[pd.Series],
    market_total: float,
    prefer_over: bool,
    in_house_total: float,
    league: str,
    fallback_prob: float,
) -> tuple[float, bool]:
    """
    Return (model_prob, passes_consensus). Only pass when >= 2 of 3 sub-models agree on over/under.
    Sub-models: (1) XGBoost totals, (2) in-house total, (3) league-avg lean (market low => lean over).
    """
    prob = predict_totals_prob(feature_row, market_total, prefer_over, fallback_prob=fallback_prob, league=league)
    v1 = _totals_direction_over(feature_row, market_total, league=league)
    v2 = in_house_total > market_total
    league_avg = NBA_LEAGUE_AVG_TOTAL if str(league).strip().upper() == "NBA" else NCAAB_LEAGUE_AVG_TOTAL
    v3 = market_total < league_avg  # low total => lean over
    votes = [v1, v2, v3]
    votes = [v for v in votes if v is not None]
    if len(votes) < 2:
        return (prob, True)
    agree = sum(1 for v in votes if v == prefer_over)
    return (prob, agree >= 2)


def consensus_moneyline(
    feature_row: Optional[pd.Series],
    selection_is_home: bool,
    home_rating: float,
    away_rating: float,
    fallback_prob: float,
    league: Optional[str] = None,
) -> tuple[float, bool]:
    """
    Return (model_prob, passes_consensus). Only pass when >= 2 of 3 sub-models agree on home vs away.
    Sub-models: (1) XGBoost moneyline, (2) in-house ratings, (3) XGBoost spread (home cover => home win lean). Uses NCAAB when league is ncaab.
    """
    prob = predict_moneyline_prob(feature_row, selection_is_home, fallback_prob=fallback_prob, league=league)
    in_house_spread = home_rating - away_rating
    recommended_home = selection_is_home
    v1 = _moneyline_direction_home(feature_row, league=league)
    v2 = in_house_spread > 0  # ratings favor home
    v3 = _spread_direction_home_cover(feature_row, in_house_spread, league=league)  # use in_house_spread as proxy market
    votes = [v for v in (v1, v2, v3) if v is not None]
    if len(votes) < 2:
        return (prob, True)
    agree = sum(1 for v in votes if v == recommended_home)
    return (prob, agree >= 2)


def get_feature_row_for_game(
    feature_matrix: pd.DataFrame,
    home_team: str,
    away_team: str,
    league: str,
    game_date: Optional[str] = None,
) -> Optional[pd.Series]:
    """Return a single row from feature_matrix matching (league, home_team, away_team, optional game_date)."""
    # #region agent log
    _log_path = "/Users/robertseipel/Desktop/bettingsim/bettingsim/.cursor/debug-e8fe0d.log"
    def _dbg(reason: str, extra: dict | None = None):
        try:
            import json, time
            data = {"home": home_team, "away": away_team, "league": league}
            if extra:
                data.update(extra)
            with open(_log_path, "a") as f:
                f.write(json.dumps({"sessionId":"e8fe0d","hypothesisId":"H1","location":"betting_models.py:get_feature_row_for_game","message":reason,"data":data,"timestamp":int(time.time()*1000)}) + "\n")
        except Exception:
            pass
    # #endregion
    if feature_matrix.empty:
        _dbg("return_none", {"reason":"empty_matrix"})
        return None
    for col in ("home_team_name", "away_team_name", "league"):
        if col not in feature_matrix.columns:
            _dbg("return_none", {"reason":"missing_col", "col":col})
            return None
    league_norm = str(league).strip().lower()
    home_strip = str(home_team).strip()
    away_strip = str(away_team).strip()

    mask = (
        (feature_matrix["league"].astype(str).str.strip().str.lower() == league_norm)
        & (feature_matrix["home_team_name"].astype(str).str.strip() == home_strip)
        & (feature_matrix["away_team_name"].astype(str).str.strip() == away_strip)
    )
    if game_date is not None and "game_date" in feature_matrix.columns:
        try:
            fd = pd.to_datetime(game_date).normalize()
            fm_dates_norm = pd.to_datetime(feature_matrix["game_date"], errors="coerce", utc=True)
            if hasattr(fm_dates_norm.dtype, "tz") and fm_dates_norm.dtype.tz is not None:
                fm_dates_norm = fm_dates_norm.dt.tz_localize(None)
            fm_dates_norm = fm_dates_norm.dt.normalize()
            mask = mask & (fm_dates_norm == fd)
        except Exception:
            pass
    subset = feature_matrix.loc[mask]
    if not subset.empty:
        return subset.iloc[0]

    # Flexible match: Live Odds names (e.g. "Kentucky Wildcats") vs feature-matrix names (e.g. "Kentucky")
    fm_league = feature_matrix["league"].astype(str).str.strip().str.lower() == league_norm
    candidates = feature_matrix.loc[fm_league]
    best = None
    best_ts = None
    for idx in candidates.index:
        row = candidates.loc[idx]
        r_h = str(row.get("home_team_name", "")).strip()
        r_a = str(row.get("away_team_name", "")).strip()
        if _team_names_match(home_team, r_h) and _team_names_match(away_team, r_a):
            try:
                ts = pd.to_datetime(row.get("game_date"), errors="coerce")
                if pd.notna(ts) and (best_ts is None or ts > best_ts):
                    best_ts = ts
                    best = row
            except Exception:
                if best is None:
                    best = row
    if best is not None:
        return best

    sample_home = feature_matrix["home_team_name"].astype(str).str.strip().iloc[:5].tolist() if len(feature_matrix) > 0 else []
    sample_away = feature_matrix["away_team_name"].astype(str).str.strip().iloc[:5].tolist() if len(feature_matrix) > 0 else []
    _dbg("return_none", {"reason":"no_match","sample_home":sample_home,"sample_away":sample_away,"league_sample":feature_matrix["league"].astype(str).iloc[:5].tolist() if "league" in feature_matrix.columns else []})
    return None


def _game_season_from_date(game_date: Optional[str]) -> Optional[int]:
    """Derive season (end year) from game_date. Oct+ -> next year; else current year."""
    if not game_date or not isinstance(game_date, str):
        return None
    s = str(game_date).strip()
    if len(s) < 4:
        return None
    try:
        year = int(s[:4])
        if len(s) >= 7:
            month = int(s[5:7])
            if month >= 10:
                return year + 1
        return year
    except (ValueError, TypeError):
        return None


# KenPom CSV often uses "Miami FL" / "Saint Louis" vs ESPN "Miami (FL)" / "Saint Louis Billikens"
NCAAB_FEATURE_TO_KENPOM_TEAM: dict[str, str] = {
    "Miami (FL)": "Miami FL",
    "Saint Louis": "Saint Louis",
}


def _resolve_team_for_kenpom_lookup(team_name: str, stats_teams: list[str]) -> Optional[str]:
    """Map live/odds team name to a TEAM value that appears in ncaab_team_season_stats."""
    if not team_name or not stats_teams:
        return None
    name = str(team_name).strip()
    norm = _normalize_team_for_lookup(name)
    mapped = NCAAB_ODDS_TO_FEATURE_TEAM.get(norm)
    if mapped:
        if mapped in stats_teams:
            return mapped
        kenpom = NCAAB_FEATURE_TO_KENPOM_TEAM.get(mapped)
        if kenpom and kenpom in stats_teams:
            return kenpom
    if name in stats_teams:
        return name
    norm_list = [t.strip().lower() for t in stats_teams]
    if norm in norm_list:
        return stats_teams[norm_list.index(norm)]
    for st in stats_teams:
        if norm == _normalize_team_for_lookup(st) or (norm in st.lower()) or (st.lower() in norm):
            return st
    return None


def build_ncaab_feature_row_from_team_stats(
    home_team: str,
    away_team: str,
    game_date: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Optional[pd.Series]:
    """
    Build a synthetic NCAAB feature row from ncaab_team_season_stats when no game row exists.
    Returns a Series with NCAAB_KENPOM_SPREAD_FEATURE_COLUMNS + home_team_name, away_team_name, league;
    days_rest/b2b default to 0. Returns None if table missing or either team not found.
    """
    import sqlite3
    path = db_path or (Path(__file__).resolve().parent.parent / "data" / "espn.db")
    if not path.exists():
        return None
    conn = sqlite3.connect(path)
    try:
        try:
            stats = pd.read_sql_query("SELECT * FROM ncaab_team_season_stats", conn)
        except Exception:
            return None
    finally:
        conn.close()
    if stats.empty or "TEAM" not in stats.columns or "season" not in stats.columns:
        return None
    season = _game_season_from_date(game_date)
    if season is None:
        season = _game_season_from_date(pd.Timestamp.now().strftime("%Y-%m-%d"))
    if season is None:
        season = 2025
    stats_teams = stats["TEAM"].astype(str).str.strip().dropna().unique().tolist()
    home_key = _resolve_team_for_kenpom_lookup(home_team, stats_teams)
    away_key = _resolve_team_for_kenpom_lookup(away_team, stats_teams)
    if not home_key or not away_key:
        return None
    season_stats = stats[stats["season"].astype(int) == season]
    if season_stats.empty and season == 2026:
        season_stats = stats[stats["season"].astype(int) == 2025]
    home_row = season_stats[season_stats["TEAM"].astype(str).str.strip() == home_key]
    away_row = season_stats[season_stats["TEAM"].astype(str).str.strip() == away_key]
    if home_row.empty or away_row.empty:
        return None
    home_row = home_row.iloc[0]
    away_row = away_row.iloc[0]
    # Base KenPom stat names that may exist in ncaab_team_season_stats
    base_stats = ["ADJOE", "ADJDE", "BARTHAG", "EFG_O", "EFG_D", "TOR", "TORD", "ORB", "DRB", "FTR", "FTRD", "ADJ_T", "SEED"]
    row = {"league": "ncaab", "home_team_name": home_team, "away_team_name": away_team}
    for c in base_stats:
        if c in home_row.index:
            row[f"home_{c}"] = float(home_row[c]) if pd.notna(home_row[c]) else 0.0
        if c in away_row.index:
            row[f"away_{c}"] = float(away_row[c]) if pd.notna(away_row[c]) else 0.0
    for col in NCAAB_KENPOM_SPREAD_FEATURE_COLUMNS:
        if col not in row:
            row[col] = 0.0
    row["home_days_rest"] = 0.0
    row["away_days_rest"] = 0.0
    row["home_is_b2b"] = 0
    row["away_is_b2b"] = 0
    return pd.Series(row)


def load_metrics() -> dict[str, float]:
    """Load last saved metrics from train_all_models."""
    if not METRICS_PATH.exists():
        return {}
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _shap_feature_to_sentence(
    feature_name: str,
    shap_val: float,
    raw_val: Any,
    home_team: str,
    away_team: str,
) -> Optional[str]:
    """Turn one (feature, SHAP value, raw value) into a short human-readable phrase with exact numbers. Returns None if no phrase."""
    try:
        val = float(raw_val) if raw_val is not None and pd.notna(raw_val) else 0.0
    except (TypeError, ValueError):
        val = 0.0
    name = (feature_name or "").strip().lower()

    # Rest / back-to-back — always use exact numbers
    if name == "home_days_rest" and val >= 0:
        if val >= 3:
            return f"{home_team} are on {int(val)}+ days rest"
        if val == 2:
            return f"{home_team} are on 2 days rest"
        if val == 1:
            return f"{home_team} are on 1 day rest"
        if val == 0:
            return f"{home_team} played last night"
        return None
    if name == "away_days_rest" and val >= 0:
        if val >= 3:
            return f"{away_team} are on {int(val)}+ days rest"
        if val == 2:
            return f"{away_team} are on 2 days rest"
        if val == 1:
            return f"{away_team} are on 1 day rest"
        if val == 0:
            return f"{away_team} played last night"
        return None
    if name == "home_is_b2b" and val >= 0.5:
        return f"{home_team} are on a back-to-back"
    if name == "away_is_b2b" and val >= 0.5:
        return f"{away_team} are on a back-to-back"

    # Travel — exact miles
    if name == "home_travel_miles" and val > 0:
        return f"{home_team} traveled {int(val)} miles"
    if name == "away_travel_miles" and val > 0:
        return f"{away_team} traveled {int(val)} miles"

    # Form / win% — exact percentage
    if name == "home_win_pct_last30" and 0 <= val <= 1:
        return f"{home_team} are {int(round(val * 100))}% over last 30 days"
    if name == "away_win_pct_last30" and 0 <= val <= 1:
        return f"{away_team} are {int(round(val * 100))}% over last 30 days"

    # Defense / offense — include actual rating when available
    if "defensive_rating" in name:
        if "home" in name and shap_val > 0:
            return f"{home_team} allow {int(val)} pts/100 (stronger defense)"
        if "away" in name and shap_val < 0:
            return f"{away_team} allow {int(val)} pts/100 (weaker defense)"
        if "home" in name and shap_val < 0:
            return f"{away_team} allow {int(val)} pts/100 (stronger defense)"
        if "away" in name and shap_val > 0:
            return f"{home_team} allow {int(val)} pts/100 (weaker defense)"
    if "def_eff" in name and "roll" in name:
        if "home" in name and shap_val > 0:
            return f"{home_team} rolling defense (last 10) favors this side"
        if "away" in name and shap_val < 0:
            return f"{away_team} rolling defense (last 10) favors this side"
    if "offensive_rating" in name:
        if "home" in name and shap_val > 0:
            return f"{home_team} score {int(val)} pts/100 (stronger offense)"
        if "away" in name and shap_val < 0:
            return f"{away_team} score {int(val)} pts/100 (weaker offense)"
    if "off_eff" in name and "roll" in name:
        if "home" in name and shap_val > 0:
            return f"{home_team} rolling offense (last 10) favors this side"
        if "away" in name and shap_val < 0:
            return f"{away_team} rolling offense (last 10) favors this side"

    # Pace / totals — exact pace when available
    if "pace" in name and "roll" not in name and val > 0:
        return f"Combined pace: {int(val)} poss/game"
    if "pace" in name and "roll" in name and val > 0:
        return f"Recent pace ({int(val)} poss) supports this total"
    if "ts_pct" in name or "true_shooting" in name:
        if val > 0:
            return f"Shooting efficiency ({val:.1f}% TS) supports this side"
        return "Shooting efficiency supports this side"

    # Line movement / sharp — exact direction/magnitude
    if name == "sharp_money_indicator" and val >= 0.5:
        return "Line moved against public (sharp money on this side)"
    if name == "line_move_magnitude" and val != 0:
        return f"Line moved {abs(val):.1f} pts in our favor"
    if name == "line_move_direction" and val != 0:
        return "Line movement favors this side"

    # Momentum — exact numbers
    if "streak" in name:
        if "home" in name and val > 0:
            return f"{home_team} on a {int(val)}-game win streak"
        if "home" in name and val < 0:
            return f"{home_team} on a {int(-val)}-game losing streak"
        if "away" in name and val > 0:
            return f"{away_team} on a {int(val)}-game win streak"
        if "away" in name and val < 0:
            return f"{away_team} on a {int(-val)}-game losing streak"
    if "ats_pct_last10" in name and isinstance(val, (int, float)) and 0 <= val <= 1:
        pct = int(round(val * 100))
        if "home" in name:
            return f"{home_team} covering {pct}% ATS last 10"
        return f"{away_team} covering {pct}% ATS last 10"
    if "ats_pct_home_season" in name and isinstance(val, (int, float)) and val > 0.5:
        return f"{home_team} {int(round(val * 100))}% ATS at home this season"
    if "ats_pct_road_season" in name and isinstance(val, (int, float)) and val > 0.5:
        return f"{away_team} {int(round(val * 100))}% ATS on the road this season"

    # NCAAB KenPom: ADJOE (adj offensive eff), ADJDE (adj defensive eff), SEED, BARTHAG
    if name == "home_adjoe" and val > 0:
        return f"{home_team} ADJOE {val:.1f} (adjusted offensive efficiency)"
    if name == "away_adjoe" and val > 0:
        return f"{away_team} ADJOE {val:.1f} (adjusted offensive efficiency)"
    if name == "home_adjde" and val > 0:
        return f"{home_team} ADJDE {val:.1f} (adjusted defensive efficiency)"
    if name == "away_adjde" and val > 0:
        return f"{away_team} ADJDE {val:.1f} (adjusted defensive efficiency)"
    if name == "home_barthag" and 0 <= val <= 1:
        return f"{home_team} BARTHAG {val:.2f} (win probability)"
    if name == "away_barthag" and 0 <= val <= 1:
        return f"{away_team} BARTHAG {val:.2f} (win probability)"
    if name == "home_seed" and val > 0:
        return f"{home_team} projected as No. {int(val)} seed"
    if name == "away_seed" and val > 0:
        return f"{away_team} projected as No. {int(val)} seed"
    if "efg_o" in name and val > 0:
        team = home_team if "home" in name else away_team
        return f"{team} effective FG% (off) {val:.1f}%"
    if "efg_d" in name and val > 0:
        team = home_team if "home" in name else away_team
        return f"{team} effective FG% (def) {val:.1f}%"
    if name == "home_adj_t" and val > 0:
        return f"{home_team} tempo (adj) {val:.1f}"
    if name == "away_adj_t" and val > 0:
        return f"{away_team} tempo (adj) {val:.1f}"

    return None


def _feature_value_to_sentence(
    feature_name: str,
    raw_val: Any,
    home_team: str,
    away_team: str,
) -> Optional[str]:
    """Build a phrase from raw feature value (no SHAP). For fallback when SHAP unavailable."""
    name = (feature_name or "").strip().lower()
    # Rest: only output when we have real data (not NaN/default)
    if name in ("home_days_rest", "away_days_rest") and (raw_val is None or pd.isna(raw_val)):
        return None
    try:
        val = float(raw_val) if raw_val is not None and pd.notna(raw_val) else 0.0
    except (TypeError, ValueError):
        val = 0.0
    if name == "home_days_rest" and val >= 0:
        if val >= 3:
            return f"{home_team} are on {int(val)}+ days rest"
        if val == 2:
            return f"{home_team} are on 2 days rest"
        if val == 1:
            return f"{home_team} are on 1 day rest"
        if val == 0:
            return f"{home_team} played last night"
    if name == "away_days_rest" and val >= 0:
        if val >= 3:
            return f"{away_team} are on {int(val)}+ days rest"
        if val == 2:
            return f"{away_team} are on 2 days rest"
        if val == 1:
            return f"{away_team} are on 1 day rest"
        if val == 0:
            return f"{away_team} played last night"
    if name == "home_is_b2b" and val >= 0.5:
        return f"{home_team} are on a back-to-back"
    if name == "away_is_b2b" and val >= 0.5:
        return f"{away_team} are on a back-to-back"
    if name == "home_travel_miles" and val > 0:
        return f"{home_team} traveled {int(val)} miles"
    if name == "away_travel_miles" and val > 0:
        return f"{away_team} traveled {int(val)} miles"
    if name == "home_win_pct_last30" and 0 <= val <= 1:
        return f"{home_team} are {int(round(val * 100))}% over last 30 days"
    if name == "away_win_pct_last30" and 0 <= val <= 1:
        return f"{away_team} are {int(round(val * 100))}% over last 30 days"
    if ("def_eff" in name or "def_eff_roll10" in name) and val > 0:
        if "home" in name:
            return f"{home_team} allow {int(val)} pts/100 over last 10"
        return f"{away_team} allow {int(val)} pts/100 over last 10"
    if ("off_eff" in name or "off_eff_roll10" in name) and val > 0:
        if "home" in name:
            return f"{home_team} score {int(val)} pts/100 over last 10"
        return f"{away_team} score {int(val)} pts/100 over last 10"
    if "defensive_rating" in name and val > 0:
        if "home" in name:
            return f"{home_team} allow {int(val)} pts/100"
        return f"{away_team} allow {int(val)} pts/100"
    if "offensive_rating" in name and val > 0:
        if "home" in name:
            return f"{home_team} score {int(val)} pts/100"
        return f"{away_team} score {int(val)} pts/100"
    if "pace" in name and val > 0:
        if "home" in name:
            return f"{home_team} play at {int(val)} poss/game"
        return f"{away_team} play at {int(val)} poss/game"
    if name == "sharp_money_indicator" and val >= 0.5:
        return "Line moved against public (sharp money on this side)"
    if name == "line_move_magnitude" and val != 0:
        return f"Line moved {abs(val):.1f} pts in our favor"
    if "ats_pct_last10" in name and isinstance(val, (int, float)) and 0 <= val <= 1:
        pct = int(round(val * 100))
        if "home" in name:
            return f"{home_team} covering {pct}% ATS last 10"
        return f"{away_team} covering {pct}% ATS last 10"
    if "streak" in name:
        if "home" in name and val != 0:
            return f"{home_team} on a {int(abs(val))}-game {'win' if val > 0 else 'losing'} streak"
        if "away" in name and val != 0:
            return f"{away_team} on a {int(abs(val))}-game {'win' if val > 0 else 'losing'} streak"
    # NCAAB KenPom: ADJOE, ADJDE, SEED, BARTHAG (for feature-based fallback when SHAP unavailable)
    if name == "home_adjoe" and val > 0:
        return f"{home_team} ADJOE {val:.1f} (adjusted offensive efficiency)"
    if name == "away_adjoe" and val > 0:
        return f"{away_team} ADJOE {val:.1f} (adjusted offensive efficiency)"
    if name == "home_adjde" and val > 0:
        return f"{home_team} ADJDE {val:.1f} (adjusted defensive efficiency)"
    if name == "away_adjde" and val > 0:
        return f"{away_team} ADJDE {val:.1f} (adjusted defensive efficiency)"
    if name == "home_barthag" and 0 <= val <= 1:
        return f"{home_team} BARTHAG {val:.2f} (win probability)"
    if name == "away_barthag" and 0 <= val <= 1:
        return f"{away_team} BARTHAG {val:.2f} (win probability)"
    if name == "home_seed" and val > 0:
        return f"{home_team} projected as No. {int(val)} seed"
    if name == "away_seed" and val > 0:
        return f"{away_team} projected as No. {int(val)} seed"
    if "efg_o" in name and val > 0:
        team = home_team if "home" in name else away_team
        return f"{team} effective FG% (off) {val:.1f}%"
    if "efg_d" in name and val > 0:
        team = home_team if "home" in name else away_team
        return f"{team} effective FG% (def) {val:.1f}%"
    return None


# Priority order for feature-based reasoning (when SHAP unavailable)
# Include NCAAB KenPom so fallback uses ADJOE/ADJDE/SEED when SHAP unavailable
_FEATURE_REASON_PRIORITY = [
    "home_days_rest", "away_days_rest", "home_is_b2b", "away_is_b2b",
    "home_travel_miles", "away_travel_miles",
    "home_win_pct_last30", "away_win_pct_last30",
    "home_defensive_rating", "away_defensive_rating",
    "home_offensive_rating", "away_offensive_rating",
    "home_def_eff_roll10", "away_def_eff_roll10",
    "home_pace", "away_pace",
    "sharp_money_indicator", "line_move_magnitude",
    "home_ats_pct_last10", "away_ats_pct_last10",
    "home_streak", "away_streak",
    "home_ADJOE", "away_ADJOE", "home_ADJDE", "away_ADJDE",
    "home_BARTHAG", "away_BARTHAG", "home_SEED", "away_SEED",
    "home_EFG_O", "away_EFG_O", "home_EFG_D", "away_EFG_D",
    "home_ADJ_T", "away_ADJ_T",
]


def _rest_differential_phrase(
    feature_row: pd.Series,
    home_team: str,
    away_team: str,
) -> Optional[str]:
    """
    Build one phrase for rest differential only when there's a meaningful edge.
    Returns None if both teams have same rest, or data is unknown.
    """
    home_days = feature_row.get("home_days_rest")
    away_days = feature_row.get("away_days_rest")
    home_b2b = float(feature_row.get("home_is_b2b") or 0) >= 0.5
    away_b2b = float(feature_row.get("away_is_b2b") or 0) >= 0.5
    home_known = home_days is not None and pd.notna(home_days)
    away_known = away_days is not None and pd.notna(away_days)
    if not home_known and not away_known:
        return None
    if home_b2b and away_b2b:
        return None  # Both on B2B = no edge
    def _rest_str(d: float) -> str:
        if d >= 3:
            return f"{int(d)}+ days rest"
        if d == 2:
            return "2 days rest"
        if d == 1:
            return "1 day rest"
        return "played last night"
    if home_b2b:
        away_val = float(away_days) if away_known else 0
        return f"{away_team} have a rest advantage ({_rest_str(away_val)} vs {home_team}'s back-to-back)"
    if away_b2b:
        home_val = float(home_days) if home_known else 0
        return f"{home_team} have a rest advantage ({_rest_str(home_val)} vs {away_team}'s back-to-back)"
    if not home_known or not away_known:
        return None
    h, a = float(home_days), float(away_days)
    if h == a:
        return None  # Same rest = no edge
    if h > a:
        return f"{home_team} have a rest advantage ({_rest_str(h)} vs {away_team}'s {_rest_str(a)})"
    return f"{away_team} have a rest advantage ({_rest_str(a)} vs {home_team}'s {_rest_str(h)})"


def get_feature_based_reasoning(
    feature_row: pd.Series,
    market: str,
    home_team: str,
    away_team: str,
    top_k: int = 2,
) -> Optional[str]:
    """
    Build 2-sentence reasoning from raw feature values when SHAP unavailable.
    Uses actual numbers only—no generic phrases. Only includes differential facts (no edge when both teams same).
    """
    # #region agent log
    _feat_log_path = "/Users/robertseipel/Desktop/bettingsim/bettingsim/.cursor/debug-e8fe0d.log"
    def _feat_dbg(reason: str, extra: dict | None = None):
        try:
            import json, time
            data = {"market": market, "home": home_team, "away": away_team}
            if extra:
                data.update(extra)
            with open(_feat_log_path, "a") as f:
                f.write(json.dumps({"sessionId":"e8fe0d","hypothesisId":"H3","location":"betting_models.py:get_feature_based_reasoning","message":reason,"data":data,"timestamp":int(time.time()*1000)}) + "\n")
        except Exception:
            pass
    # #endregion
    phrases: list[str] = []
    seen: set[str] = set()
    # Rest: one phrase only when there's a differential
    rest_phrase = _rest_differential_phrase(feature_row, home_team, away_team)
    if rest_phrase:
        phrases.append(rest_phrase)
    # Skip rest/B2B in main loop (already handled)
    skip_for_rest = {"home_days_rest", "away_days_rest", "home_is_b2b", "away_is_b2b"}
    for name in _FEATURE_REASON_PRIORITY:
        if name in skip_for_rest or name not in feature_row.index:
            continue
        if len(phrases) >= top_k:
            break
        phrase = _feature_value_to_sentence(name, feature_row.get(name), home_team, away_team)
        if phrase and phrase not in seen:
            seen.add(phrase)
            phrases.append(phrase)
    if not phrases:
        _feat_dbg("return_none", {"reason": "no_phrases"})
        return None
    return ". ".join(phrases)


def get_top_shap_reasoning(
    feature_row: pd.Series,
    market: str,
    home_team: str,
    away_team: str,
    top_k: int = 3,
    league: Optional[str] = None,
) -> Optional[str]:
    """
    Pull top-k SHAP feature values for this prediction and build a unique sentence.
    market: 'spreads' | 'totals' | 'h2h'. Uses NCAAB model when league is ncaab. Returns None if SHAP unavailable.
    """
    # #region agent log
    _shap_log_path = "/Users/robertseipel/Desktop/bettingsim/bettingsim/.cursor/debug-e8fe0d.log"
    def _shap_dbg(reason: str, extra: dict | None = None):
        try:
            import json, time
            data = {"market": market, "home": home_team, "away": away_team}
            if extra:
                data.update(extra)
            with open(_shap_log_path, "a") as f:
                f.write(json.dumps({"sessionId":"e8fe0d","hypothesisId":"H2","location":"betting_models.py:get_top_shap_reasoning","message":reason,"data":data,"timestamp":int(time.time()*1000)}) + "\n")
        except Exception:
            pass
    # #endregion
    try:
        import shap
    except ImportError:
        _shap_dbg("return_none", {"reason": "shap_import_error"})
        return None
    model_path: Optional[Path] = None
    cols: list[str] = []
    if market == "spreads":
        model_path, cols = _spread_model_path_for_league(league), SPREAD_FEATURE_COLUMNS
    elif market == "totals":
        model_path, cols = _totals_model_path_for_league(league), TOTALS_FEATURE_COLUMNS
    elif market == "h2h":
        model_path, cols = _moneyline_model_path_for_league(league), MONEYLINE_FEATURE_COLUMNS
    else:
        _shap_dbg("return_none", {"reason": "unknown_market"})
        return None
    payload = load_model(model_path)
    if payload is None:
        _shap_dbg("return_none", {"reason": "model_load_failed", "path": str(model_path)})
        return None
    model = payload["model"]
    used_cols = payload.get("feature_columns", cols)
    X, used = _select_features(pd.DataFrame([feature_row]), used_cols)
    if X.empty or len(used) == 0:
        _shap_dbg("return_none", {"reason": "X_empty_or_no_used", "used_len": len(used)})
        return None
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        sv = np.asarray(shap_values)
        if sv.ndim == 2:
            row_shap = sv[0]
        else:
            row_shap = sv
        if len(row_shap) != len(used):
            _shap_dbg("return_none", {"reason": "shap_len_mismatch"})
            return None
        indexed = [(used[i], float(row_shap[i]), X.iloc[0].get(used[i])) for i in range(len(used))]
        indexed.sort(key=lambda x: abs(x[1]), reverse=True)
        phrases: list[str] = []
        seen: set[str] = set()
        for feat_name, sh, raw in indexed[: top_k * 3]:
            if len(phrases) >= top_k:
                break
            phrase = _shap_feature_to_sentence(feat_name, sh, raw, home_team, away_team)
            if phrase and phrase not in seen:
                seen.add(phrase)
                phrases.append(phrase)
        if not phrases:
            _shap_dbg("return_none", {"reason": "no_phrases_from_shap"})
            return None
        return ". ".join(phrases)
    except Exception as e:
        _shap_dbg("return_none", {"reason": "exception", "err": str(e)})
        return None


def build_feature_row_for_upcoming_game(
    home_team: str,
    away_team: str,
    league: str,
    game_date: Optional[str] = None,
    b2b_teams: Optional[set[str]] = None,
    db_path: Optional[Path] = None,
) -> Optional[pd.Series]:
    """
    Build a synthetic feature row for an upcoming game when it's not in games_with_team_stats.
    Uses team_advanced_stats (season ratings), games table (days_rest), and b2b_teams.
    Returns a Series with columns needed for get_feature_based_reasoning.
    """
    from datetime import datetime
    from .sportsref_stats import load_team_advanced_stats_from_sqlite, load_games_with_season, _apply_name_mapping, TEAM_NAME_MAPPING

    path = db_path or (Path(__file__).resolve().parent.parent / "data" / "espn.db")
    if not path.exists():
        return None
    league_key = str(league).strip().lower()
    if league_key not in ("nba", "ncaab"):
        return None

    # Current season (e.g. 2024 for 2023-24)
    now = datetime.now()
    season = now.year if now.month >= 10 else now.year - 1

    # Team stats from team_advanced_stats (may not exist if pipeline not run)
    try:
        stats_df = load_team_advanced_stats_from_sqlite(league=league_key, db_path=path)
        if stats_df.empty or "team_name" not in stats_df.columns:
            stats_df = load_team_advanced_stats_from_sqlite(league=None, db_path=path)
        stats_df = stats_df[stats_df["season"] == season] if "season" in stats_df.columns else stats_df
    except Exception:
        stats_df = pd.DataFrame()

    def _get_team_stat(team: str, col: str) -> float:
        mapped = _apply_name_mapping(team, TEAM_NAME_MAPPING)
        for _, r in stats_df.iterrows():
            if str(r.get("team_name", "")).strip() == mapped.strip():
                v = r.get(col)
                if v is not None and not pd.isna(v):
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        pass
        return 0.0

    # Days rest and B2B from games table (may not exist)
    try:
        games_df = load_games_with_season(db_path=path, league=league_key)
        if games_df.empty:
            games_df = load_games_with_season(db_path=path, league=None)
    except Exception:
        games_df = pd.DataFrame()
    game_dt = None
    if game_date:
        try:
            game_dt = datetime.strptime(str(game_date)[:10], "%Y-%m-%d")
        except ValueError:
            pass
    if game_dt is None:
        game_dt = now
    games_before = games_df
    if "game_date" in games_df.columns:
        gd_ser = pd.to_datetime(games_df["game_date"], errors="coerce", utc=True)
        if hasattr(gd_ser.dtype, "tz") and gd_ser.dtype.tz is not None:
            gd_ser = gd_ser.dt.tz_localize(None)
        games_before = games_df[gd_ser < pd.Timestamp(game_dt)]

    # Days rest: only use when we have games data; otherwise leave as NaN (unknown)
    has_games_data = not games_before.empty and "game_date" in games_before.columns
    if has_games_data:
        from .situational_features import _days_rest
        home_days, home_b2b = _days_rest(home_team.strip(), game_dt, games_before)
        away_days, away_b2b = _days_rest(away_team.strip(), game_dt, games_before)
    else:
        home_days, away_days = np.nan, np.nan
        home_b2b, away_b2b = False, False
    b2b_set = b2b_teams or set()
    if home_team.strip() in b2b_set:
        home_b2b = True
        home_days = 0.0
    if away_team.strip() in b2b_set:
        away_b2b = True
        away_days = 0.0

    row = pd.Series(dtype=object)
    row["home_team_name"] = home_team.strip()
    row["away_team_name"] = away_team.strip()
    row["league"] = league_key
    row["home_days_rest"] = float(home_days) if pd.notna(home_days) else np.nan
    row["away_days_rest"] = float(away_days) if pd.notna(away_days) else np.nan
    row["home_is_b2b"] = 1.0 if home_b2b else 0.0
    row["away_is_b2b"] = 1.0 if away_b2b else 0.0
    row["home_defensive_rating"] = _get_team_stat(home_team, "defensive_rating")
    row["away_defensive_rating"] = _get_team_stat(away_team, "defensive_rating")
    row["home_offensive_rating"] = _get_team_stat(home_team, "offensive_rating")
    row["away_offensive_rating"] = _get_team_stat(away_team, "offensive_rating")
    row["home_pace"] = _get_team_stat(home_team, "pace")
    row["away_pace"] = _get_team_stat(away_team, "pace")
    return row


def _ncaab_inference_feature_columns() -> set[str]:
    """Union of feature_columns from all three NCAAB model pkl files, plus required keys for lookup."""
    required = {"league", "game_id", "game_date", "home_team_name", "away_team_name"}
    out = set(required)
    for model_path in (SPREAD_MODEL_PATH_NCAAB, TOTALS_MODEL_PATH_NCAAB, MONEYLINE_MODEL_PATH_NCAAB):
        payload = load_model(model_path)
        if payload and isinstance(payload.get("feature_columns"), list):
            out.update(payload["feature_columns"])
    if out == required:
        out.update(NCAAB_SPREAD_FEATURE_COLUMNS + TOTALS_FEATURE_COLUMNS + MONEYLINE_FEATURE_COLUMNS)
    return out


def load_feature_matrix_for_inference(
    db_path: Optional[Path] = None,
    league: Optional[str] = None,
    max_seasons: int = 2,
) -> pd.DataFrame:
    """Load games_with_team_stats and merge situational features for prediction-time lookups.
    Caps to last max_seasons (default 2). Only keeps columns used by NCAAB model feature_columns (plus keys)."""
    from .sportsref_stats import load_merged_games_from_sqlite
    from .situational_features import load_situational_features_from_sqlite

    path = db_path or (Path(__file__).resolve().parent.parent / "data" / "espn.db")
    fm = load_merged_games_from_sqlite(league=league, db_path=path, max_seasons_for_inference=max_seasons)
    if fm.empty:
        return fm
    try:
        sit = load_situational_features_from_sqlite(league=league, db_path=path)
        if not sit.empty and "league" in sit.columns and "game_id" in sit.columns:
            drop = [c for c in sit.columns if c in fm.columns and c not in ("league", "game_id")]
            sit = sit.drop(columns=drop, errors="ignore")
            fm = fm.merge(sit, on=["league", "game_id"], how="left", suffixes=("", "_sit"))
            fm = fm.loc[:, ~fm.columns.duplicated()]
    except Exception:
        pass
    keep = _ncaab_inference_feature_columns()
    cols_to_keep = [c for c in fm.columns if c in keep]
    if cols_to_keep:
        fm = fm[cols_to_keep].copy()
    return fm
