"""
Three specialized XGBoost models: spreads, totals (over/under), and moneylines.
Each is trained on bet-type-specific features, evaluated independently, and used to
route predictions. Falls back to heuristics when models are not trained or features missing.
"""

from __future__ import annotations

import json
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
# Spreads: defensive ratings and situational spots matter more.
SPREAD_FEATURE_COLUMNS = [
    "home_defensive_rating", "away_defensive_rating",
    "home_offensive_rating", "away_offensive_rating",
    "home_days_rest", "away_days_rest",
    "home_is_b2b", "away_is_b2b",
    "home_travel_miles", "away_travel_miles",
    "home_win_pct_last30", "away_win_pct_last30",
    "home_def_eff_roll10", "away_def_eff_roll10",
    "home_off_eff_roll10", "away_off_eff_roll10",
    "line_move_direction", "line_move_magnitude", "sharp_money_indicator",
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
    return df, df, df


def train_spread_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    db_path: Optional[Path] = None,
) -> dict[str, float]:
    """Train XGBoost regressor to predict margin (home - away). Evaluate with MAE and R². Returns metrics."""
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score
    except ImportError:
        return {}
    X, used = _select_features(df, SPREAD_FEATURE_COLUMNS)
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
    with open(SPREAD_MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "feature_columns": used}, f)
    return {"spread_mae": round(mae, 4), "spread_r2": round(r2, 4), "spread_n_features": len(used)}


def train_totals_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    db_path: Optional[Path] = None,
) -> dict[str, float]:
    """Train XGBoost regressor to predict total_pts. Returns metrics."""
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
    with open(TOTALS_MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "feature_columns": used}, f)
    return {"totals_mae": round(mae, 4), "totals_r2": round(r2, 4), "totals_n_features": len(used)}


def train_moneyline_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    db_path: Optional[Path] = None,
) -> dict[str, float]:
    """Train XGBoost classifier for home_win (0/1). Returns metrics (accuracy, log_loss)."""
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
    with open(MONEYLINE_MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "feature_columns": used}, f)
    return {"moneyline_accuracy": round(acc, 4), "moneyline_log_loss": round(ll, 4), "moneyline_n_features": len(used)}


def train_all_models(
    db_path: Optional[Path] = None,
    league: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, float]:
    """Load training data, train all three models, save metrics. Returns combined metrics dict."""
    df, _, _ = get_training_data(db_path=db_path, league=league, merge_situational=True)
    if df.empty or len(df) < 50:
        return {}
    metrics = {}
    metrics.update(train_spread_model(df, test_size=test_size, random_state=random_state))
    metrics.update(train_totals_model(df, test_size=test_size, random_state=random_state))
    metrics.update(train_moneyline_model(df, test_size=test_size, random_state=random_state))
    _ensure_models_dir()
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
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


def predict_spread_prob(
    row: pd.Series,
    market_spread: float,
    we_cover_favorite: bool,
    fallback_prob: Optional[float] = None,
) -> float:
    """
    Predict P(cover) for the spread. market_spread is home spread (e.g. -3.5).
    we_cover_favorite True = we're on the favorite (home/away favored).
    Uses loaded XGBoost spread model if available; else returns fallback_prob or 0.5.
    """
    payload = load_model(SPREAD_MODEL_PATH)
    if payload is None:
        return fallback_prob if fallback_prob is not None else 0.5
    model = payload["model"]
    cols = payload.get("feature_columns", [])
    X, _ = _select_features(pd.DataFrame([row]), cols)
    if X.empty:
        return fallback_prob if fallback_prob is not None else 0.5
    pred_margin = float(model.predict(X)[0])
    # P(home covers market_spread) = P(margin > market_spread). Approximate with sigmoid.
    diff = pred_margin - market_spread
    # Home covers if margin > market_spread. So prob = 1 / (1 + exp(-k*diff)), k~0.3
    k = 0.35
    prob_home_cover = 1.0 / (1.0 + np.exp(-k * diff))
    if we_cover_favorite:
        # Favorite: home favored when market_spread < 0, away when market_spread > 0. We're on favorite.
        if market_spread <= 0:
            return float(np.clip(prob_home_cover, 0.02, 0.98))
        return float(np.clip(1.0 - prob_home_cover, 0.02, 0.98))
    if market_spread <= 0:
        return float(np.clip(1.0 - prob_home_cover, 0.02, 0.98))
    return float(np.clip(prob_home_cover, 0.02, 0.98))


def predict_totals_prob(
    row: pd.Series,
    market_total: float,
    prefer_over: bool,
    fallback_prob: Optional[float] = None,
) -> float:
    """Predict P(over) or P(under) from totals model. prefer_over True = Over."""
    payload = load_model(TOTALS_MODEL_PATH)
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
) -> float:
    """Predict P(selection wins). selection_is_home True = home team."""
    payload = load_model(MONEYLINE_MODEL_PATH)
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


def get_feature_row_for_game(
    feature_matrix: pd.DataFrame,
    home_team: str,
    away_team: str,
    league: str,
    game_date: Optional[str] = None,
) -> Optional[pd.Series]:
    """Return a single row from feature_matrix matching (league, home_team, away_team, optional game_date)."""
    if feature_matrix.empty:
        return None
    for col in ("home_team_name", "away_team_name", "league"):
        if col not in feature_matrix.columns:
            return None
    mask = (
        (feature_matrix["league"].astype(str).str.strip().str.lower() == str(league).strip().lower())
        & (feature_matrix["home_team_name"].astype(str).str.strip() == str(home_team).strip())
        & (feature_matrix["away_team_name"].astype(str).str.strip() == str(away_team).strip())
    )
    if game_date is not None and "game_date" in feature_matrix.columns:
        try:
            fd = pd.to_datetime(game_date).normalize()
            mask = mask & (pd.to_datetime(feature_matrix["game_date"], errors="coerce").dt.normalize() == fd)
        except Exception:
            pass
    subset = feature_matrix.loc[mask]
    if subset.empty:
        return None
    return subset.iloc[0]


def load_metrics() -> dict[str, float]:
    """Load last saved metrics from train_all_models."""
    if not METRICS_PATH.exists():
        return {}
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def load_feature_matrix_for_inference(
    db_path: Optional[Path] = None,
    league: Optional[str] = None,
) -> pd.DataFrame:
    """Load games_with_team_stats and merge situational features for prediction-time lookups."""
    from .sportsref_stats import load_merged_games_from_sqlite
    from .situational_features import load_situational_features_from_sqlite

    path = db_path or (Path(__file__).resolve().parent.parent / "data" / "espn.db")
    fm = load_merged_games_from_sqlite(league=league, db_path=path)
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
    return fm
