#!/usr/bin/env python3
"""
Load the most recent NCAAB odds CSV from data/odds/, fuzzy-match team names to the team stats
database, and run spread + totals predictions. No fixed filename: always uses the latest file
in data/odds/ by modification time.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from datetime import date, datetime, timezone
import math

APP_ROOT = Path(__file__).resolve().parent.parent
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import pandas as pd

ODDS_DIR = APP_ROOT / "data" / "odds"
NCAAB_DIR = APP_ROOT / "data" / "ncaab"
MODELS_DIR = APP_ROOT / "data" / "models"
NCAAB_SPREAD_MODEL_PATH = MODELS_DIR / "xgboost_spread_ncaab.pkl"
NCAAB_TOTALS_MODEL_PATH = MODELS_DIR / "xgboost_totals_ncaab.pkl"
TOTALS_EDGE_THRESHOLD = 5.0  # Totals markets are tighter; need larger edge for meaningful value play
# When False, totals picks are not written to the cache (hidden on dashboard pending further model calibration).
SHOW_TOTALS_PICKS = False
CURRENT_FORM_PATH = NCAAB_DIR / "current_form_2026.csv"
TEAM_HCA_PATH = NCAAB_DIR / "team_hca_by_season.csv"
CONF_HCA_PATH = NCAAB_DIR / "conf_hca_by_season.csv"
TEAM_CONF_PATH = NCAAB_DIR / "team_conf_by_season.csv"
HISTORICAL_GAMES_PATH = NCAAB_DIR / "historical_games.csv"
REQUIRED_COLUMNS = ["Home_Team", "Away_Team", "Spread", "Over_Under"]
FUZZY_MIN_SCORE = 70
FUZZY_WARN_SCORE = 80  # Warn if best match is below this (suggest adding to mapping)
# KenPom stat columns used by xgboost_spread_ncaab.pkl (must match train_spread_model STAT_COLS + SEED)
KENPOM_STAT_COLUMNS = ["ADJOE", "ADJDE", "BARTHAG", "ADJ_T", "EFG_O", "EFG_D", "TOR", "ORB", "FTR", "THREE_PT_RATE", "SEED"]
# Form feature names (model expects these + home_hca, is_neutral, diffs)
FORM_FEATURE_COLS = [
    "home_last5_margin", "home_last10_margin", "home_last5_winpct",
    "away_last5_margin", "away_last10_margin", "away_last5_winpct",
    "last5_margin_diff", "last10_margin_diff", "last5_winpct_diff",
]

# Odds CSV / display name -> name used in team stats (KenPom / Torvik TEAM)
ODDS_TO_STATS_NAME: dict[str, str] = {
    "FGCU": "Florida Gulf Coast",
    "NC Central": "North Carolina Central",
    "Central Ark": "Central Arkansas",
    "Maryland-Eastern Shore": "Maryland Eastern Shore",
    "NW State": "Northwestern St.",
    "Nicholls": "Nicholls St.",
    "W. Carolina": "Western Carolina",
    "Queens Charlotte": "Queens",
    "Ga. Southern": "Georgia Southern",
    "Boston U": "Boston University",
    # Expanded mappings for Torvik naming
    "Kansas State": "Kansas St.",
    "Penn State": "Penn St.",
    "Oklahoma State": "Oklahoma St.",
    "Wright State": "Wright St.",
    "Detroit Mercy": "Detroit Mercy",
    "Missouri State": "Missouri St.",
    "Jacksonville State": "Jacksonville St.",
    "Portland State": "Portland St.",
    "Jackson State": "Jackson St.",
}

# Value play: only when |edge| > this (pts). Pick Home if edge > 3 (model thinks home does better than spread), Away if edge < -3, skip if |edge| <= 3.
VALUE_PLAY_THRESHOLD = 3.0
# Confidence: High = |edge| > 7, Medium = 3 < |edge| <= 7, Low = |edge| <= 3
EDGE_HIGH_CONFIDENCE = 7.0
# BARTHAG-based margin: pred_margin = (home_BARTHAG - away_BARTHAG) * this (research: 50-60 maps better to actual margins)
BARTHAG_TO_POINTS_SCALE = 50.0
# Neutral site: add ~3 pts to away team (reduce home advantage). Typical HCA ~3 pts.
NEUTRAL_SITE_ADJUSTMENT_PTS = 3.0
# Rebounding: ⭐ when recommended side has ROff advantage over opponent > this
ROFF_ADVANTAGE_THRESHOLD = 3.0
# Reliability: if away BARTHAG > home BARTHAG + this, flag as low reliability (temporarily not excluding)
BARTHAG_MISMATCH_THRESHOLD = 0.25
LOW_RELIABILITY_FLAG = "Low Reliability - Large KenPom Mismatch"
# Diversity: max share of value plays that can be Home picks; min floor so Home has at least this share
VALUE_PLAY_MAX_HOME_RATIO = 0.60
VALUE_PLAY_MIN_HOME_RATIO = 0.30
# Use 2026 in dates to match stats database
ODDS_YEAR = 2026

# Market blending: blend model pred_margin with market-implied margin for edge calculation
USE_MARKET_BLEND = True
MODEL_WEIGHT = 0.7  # weight for model; (1 - MODEL_WEIGHT) for market. blended = MODEL_WEIGHT * pred + (1 - MODEL_WEIGHT) * (-spread)

# Calibrated cover-probability params (from data/models/cover_prob_params.json)
COVER_PROB_PARAMS_PATH = MODELS_DIR / "cover_prob_params.json"
# Kelly stake: -110 juice -> decimal odds 1.909; use 25% fractional Kelly; stake % clamped [0.5, 5.0]
DECIMAL_ODDS_110 = 1.909
KELLY_FRACTION_PCT = 0.25
SUGGESTED_STAKE_PCT_MIN = 0.5
SUGGESTED_STAKE_PCT_MAX = 5.0


def _load_ncaab_model_feature_columns() -> list[str]:
    """Load NCAAB spread model and return its feature_columns. Raises if file missing or no feature_columns."""
    import joblib
    if not NCAAB_SPREAD_MODEL_PATH.exists():
        raise FileNotFoundError(f"NCAAB spread model not found: {NCAAB_SPREAD_MODEL_PATH}")
    payload = joblib.load(NCAAB_SPREAD_MODEL_PATH)
    cols = payload.get("feature_columns")
    if cols is None and hasattr(payload.get("model"), "get_booster"):
        try:
            cols = payload["model"].get_booster().feature_names
        except Exception:
            pass
    if not cols:
        raise ValueError("NCAAB model has no feature_columns or feature_names")
    return list(cols)


def _load_current_form() -> pd.DataFrame:
    """Load current_form_2026.csv; return empty DataFrame if missing (caller uses 0.0 for missing teams)."""
    if not CURRENT_FORM_PATH.exists():
        return pd.DataFrame(columns=["team", "last5_margin", "last10_margin", "last5_winpct"])
    try:
        return pd.read_csv(CURRENT_FORM_PATH)
    except Exception:
        return pd.DataFrame(columns=["team", "last5_margin", "last10_margin", "last5_winpct"])


def _load_totals_model() -> tuple[object, list[str], float] | None:
    """Load xgboost_totals_ncaab.pkl; return (model, feature_columns, bias_correction) or None."""
    if not NCAAB_TOTALS_MODEL_PATH.exists():
        return None
    try:
        import joblib
        payload = joblib.load(NCAAB_TOTALS_MODEL_PATH)
        model = payload.get("model")
        cols = payload.get("feature_columns")
        if model is None or not cols:
            return None
        bias = float(payload.get("bias_correction", 0.0))
        return (model, list(cols), bias)
    except Exception:
        return None


def _build_totals_feature_row(feature_row: pd.Series, feature_columns: list[str]) -> dict[str, float] | None:
    """Build feature dict for totals model from spread feature_row. Fills missing with 0."""
    def _f(val) -> float:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return 0.0
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0
    idx = getattr(feature_row, "index", None)
    get = (lambda k: feature_row[k] if idx is not None and k in idx else feature_row.get(k)) if hasattr(feature_row, "get") else (lambda k: 0.0)
    out = {}
    out["home_ADJ_T"] = _f(get("home_ADJ_T"))
    out["away_ADJ_T"] = _f(get("away_ADJ_T"))
    out["home_ADJOE"] = _f(get("home_ADJOE"))
    out["home_ADJDE"] = _f(get("home_ADJDE"))
    out["away_ADJOE"] = _f(get("away_ADJOE"))
    out["away_ADJDE"] = _f(get("away_ADJDE"))
    out["tempo_diff"] = _f(get("tempo_diff")) if "tempo_diff" in (feature_columns or []) else (out["home_ADJ_T"] - out["away_ADJ_T"])
    out["combined_tempo"] = out["home_ADJ_T"] + out["away_ADJ_T"]
    out["combined_offense"] = out["home_ADJOE"] + out["away_ADJOE"]
    out["combined_defense"] = out["home_ADJDE"] + out["away_ADJDE"]
    out["is_neutral"] = 1.0 if _f(get("is_neutral")) else 0.0
    out["is_conference_tourney"] = 1.0 if _f(get("is_conference_tourney")) else 0.0
    out["is_ncaa_tourney"] = 1.0 if _f(get("is_ncaa_tourney")) else 0.0
    return out


def _load_cover_prob_params() -> dict | None:
    """Load calibrated logistic parameters for P(cover) from cover_prob_params.json, or None if missing/invalid."""
    if not COVER_PROB_PARAMS_PATH.exists():
        return None
    try:
        with open(COVER_PROB_PARAMS_PATH) as f:
            data = json.load(f)
    except Exception:
        return None
    k = data.get("k")
    intercept = data.get("intercept")
    if k is None or intercept is None:
        return None
    try:
        return {"k": float(k), "intercept": float(intercept)}
    except (TypeError, ValueError):
        return None


def _load_hca_lookups() -> tuple[dict[tuple[int, str], float], dict[tuple[int, str], float], dict[tuple[int, str], str]]:
    """Load team_hca, conf_hca, team_conf. (season, team) -> hca; (season, conf) -> hca; (season, team) -> conf."""
    team_hca: dict[tuple[int, str], float] = {}
    conf_hca: dict[tuple[int, str], float] = {}
    team_conf: dict[tuple[int, str], str] = {}
    if TEAM_HCA_PATH.exists():
        try:
            df = pd.read_csv(TEAM_HCA_PATH)
            for _, r in df.iterrows():
                s, t = int(r.get("season", 0)), str(r.get("team", "")).strip()
                if t:
                    team_hca[(s, t)] = float(r.get("home_hca", 0) or 0)
        except Exception:
            pass
    if CONF_HCA_PATH.exists():
        try:
            df = pd.read_csv(CONF_HCA_PATH)
            for _, r in df.iterrows():
                s, c = int(r.get("season", 0)), str(r.get("conf", "")).strip()
                if c:
                    conf_hca[(s, c)] = float(r.get("home_hca", 0) or 0)
        except Exception:
            pass
    if TEAM_CONF_PATH.exists():
        try:
            df = pd.read_csv(TEAM_CONF_PATH)
            for _, r in df.iterrows():
                s, t, c = int(r.get("season", 0)), str(r.get("team", "")).strip(), str(r.get("conf", "")).strip()
                if t:
                    team_conf[(s, t)] = c
        except Exception:
            pass
    return team_hca, conf_hca, team_conf


def _compute_2026_hca_from_games() -> dict[tuple[int, str], float]:
    """Compute team HCA for 2026 from historical_games if we have 2026 games; else return empty."""
    from collections import defaultdict
    from engine.utils import game_season_from_date
    if not HISTORICAL_GAMES_PATH.exists():
        return {}
    games = pd.read_csv(HISTORICAL_GAMES_PATH)
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    games = games.dropna(subset=["date"])
    games["season"] = games["date"].apply(lambda d: game_season_from_date(d))
    g = games[games["season"] == 2026].copy()
    if g.empty or "margin" not in g.columns:
        return {}
    team_home_margins: dict[str, list[float]] = defaultdict(list)
    team_all_margins: dict[str, list[float]] = defaultdict(list)
    for _, row in g.iterrows():
        home = str(row.get("home_team", "")).strip()
        away = str(row.get("away_team", "")).strip()
        margin = float(row.get("margin", 0))
        neutral = row.get("neutral_site") in (1, True, "1", "yes") or (isinstance(row.get("neutral_site"), (int, float)) and row.get("neutral_site") not in (0, 0.0))
        if not home or not away:
            continue
        team_all_margins[home].append(margin)
        team_all_margins[away].append(-margin)
        if not neutral:
            team_home_margins[home].append(margin)
    out: dict[tuple[int, str], float] = {}
    for team, home_margins in team_home_margins.items():
        if not home_margins:
            continue
        all_m = team_all_margins.get(team, [])
        if not all_m:
            continue
        out[(2026, team)] = (sum(home_margins) / len(home_margins)) - (sum(all_m) / len(all_m))
    return out


def _get_home_hca(season: int, home_team: str, is_neutral: bool,
                  team_hca: dict, conf_hca: dict, team_conf: dict,
                  hca_2026_computed: dict | None) -> float:
    """Return home HCA for (season, home_team). For 2026, use computed if available, else 2025 team/conf fallback."""
    if is_neutral:
        return 0.0
    key = (season, str(home_team).strip())
    v = team_hca.get(key)
    if v is not None and (season != 2026 or v != 0.0):
        return float(v)
    if season == 2026 and hca_2026_computed:
        v = hca_2026_computed.get(key)
        if v is not None:
            return float(v)
    key_2025 = (2025, str(home_team).strip())
    v = team_hca.get(key_2025)
    if v is not None:
        return float(v)
    conf = team_conf.get(key) or team_conf.get(key_2025)
    if conf:
        v = conf_hca.get((season, conf)) or conf_hca.get((2025, conf))
        if v is not None:
            return float(v)
    return 0.0


def _ensure_ncaab_model_feature_alignment(model_feature_columns: list[str]) -> None:
    """Verify we can build a feature row with exactly the columns the model expects. Raise if mismatch."""
    # Columns we can populate: from KenPom (home_* / away_* for STAT_COLS), diffs, home_hca, is_neutral, form, tourney/rest
    stat_cols = [c for c in KENPOM_STAT_COLUMNS if c != "SEED"]
    our_cols = set()
    for c in stat_cols:
        our_cols.add(f"home_{c}")
        our_cols.add(f"away_{c}")
    our_cols.update(["BARTHAG_diff", "ADJOE_diff", "ADJDE_diff", "tempo_diff", "home_hca", "is_neutral"])
    our_cols.update(["is_conference_tourney", "is_ncaa_tourney", "seed_diff", "days_rest_home", "days_rest_away"])
    our_cols.update(FORM_FEATURE_COLS)
    model_set = set(model_feature_columns)
    missing = model_set - our_cols
    extra = our_cols - model_set
    if missing:
        err_lines = [
            "NCAAB model feature alignment failed: prediction feature row does not include all model features.",
            f"  Missing columns (model expects, we do not build): {sorted(missing)}",
        ]
        if extra:
            err_lines.append(f"  Extra columns we build but model does not use: {sorted(extra)}")
        raise Exception("\n".join(err_lines))
    if len(model_feature_columns) != len(model_set):
        raise Exception(
            "NCAAB model feature alignment failed: model feature_columns has duplicate names."
        )


def _enrich_feature_row_for_ncaab_model(
    feature_row: pd.Series,
    home_team: str,
    away_team: str,
    is_neutral: bool,
    season: int,
    form_df: pd.DataFrame,
    team_hca: dict,
    conf_hca: dict,
    team_conf: dict,
    hca_2026_computed: dict | None,
    game_date: str | None = None,
    last_game_dates: dict | None = None,
) -> None:
    """Add BARTHAG_diff, ADJOE_diff, ADJDE_diff, tempo_diff, home_hca, is_neutral, form, is_conference_tourney, is_ncaa_tourney, days_rest_home, days_rest_away to feature_row in place."""
    def _f(val: float | None) -> float:
        return float(val) if val is not None and not (isinstance(val, float) and pd.isna(val)) else 0.0
    h_b = _f(feature_row.get("home_BARTHAG"))
    a_b = _f(feature_row.get("away_BARTHAG"))
    feature_row["BARTHAG_diff"] = h_b - a_b
    feature_row["ADJOE_diff"] = _f(feature_row.get("home_ADJOE")) - _f(feature_row.get("away_ADJOE"))
    feature_row["ADJDE_diff"] = _f(feature_row.get("home_ADJDE")) - _f(feature_row.get("away_ADJDE"))
    feature_row["tempo_diff"] = _f(feature_row.get("home_ADJ_T")) - _f(feature_row.get("away_ADJ_T"))
    feature_row["home_hca"] = _get_home_hca(season, home_team, is_neutral, team_hca, conf_hca, team_conf, hca_2026_computed)
    feature_row["is_neutral"] = 1 if is_neutral else 0
    # Form from current_form_2026.csv (team name match; default 0.0)
    def form_for_team(team: str) -> tuple[float, float, float]:
        if form_df.empty or "team" not in form_df.columns:
            return 0.0, 0.0, 0.0
        row = form_df[form_df["team"].astype(str).str.strip() == str(team).strip()]
        if row.empty:
            return 0.0, 0.0, 0.0
        r = row.iloc[0]
        return _f(r.get("last5_margin")), _f(r.get("last10_margin")), _f(r.get("last5_winpct"))
    h5_m, h10_m, h5_w = form_for_team(home_team)
    a5_m, a10_m, a5_w = form_for_team(away_team)
    feature_row["home_last5_margin"] = h5_m
    feature_row["home_last10_margin"] = h10_m
    feature_row["home_last5_winpct"] = h5_w
    feature_row["away_last5_margin"] = a5_m
    feature_row["away_last10_margin"] = a10_m
    feature_row["away_last5_winpct"] = a5_w
    feature_row["last5_margin_diff"] = h5_m - a5_m
    feature_row["last10_margin_diff"] = h10_m - a10_m
    feature_row["last5_winpct_diff"] = h5_w - a5_w
    # Seed diff from seed columns when available: away_seed - home_seed (0 if missing)
    home_seed = feature_row.get("home_seed")
    away_seed = feature_row.get("away_seed")
    try:
        h_seed_val = float(home_seed) if home_seed not in (None, 0, "0") and not (isinstance(home_seed, float) and pd.isna(home_seed)) else None
    except (TypeError, ValueError):
        h_seed_val = None
    try:
        a_seed_val = float(away_seed) if away_seed not in (None, 0, "0") and not (isinstance(away_seed, float) and pd.isna(away_seed)) else None
    except (TypeError, ValueError):
        a_seed_val = None
    if h_seed_val is not None and a_seed_val is not None:
        feature_row["seed_diff"] = a_seed_val - h_seed_val
    else:
        feature_row["seed_diff"] = 0.0
    # Tournament flags from game_date (March 1–15 = conference tourney; March 16–April 10 = NCAA)
    if game_date:
        try:
            parts = game_date.split("-")
            if len(parts) >= 3:
                month, day = int(parts[1]), int(parts[2])
                feature_row["is_conference_tourney"] = 1.0 if (month == 3 and 1 <= day <= 15) else 0.0
                feature_row["is_ncaa_tourney"] = 1.0 if ((month == 3 and day >= 16) or (month == 4 and day <= 10)) else 0.0
            else:
                feature_row["is_conference_tourney"] = 0.0
                feature_row["is_ncaa_tourney"] = 0.0
        except (ValueError, IndexError):
            feature_row["is_conference_tourney"] = 0.0
            feature_row["is_ncaa_tourney"] = 0.0
    else:
        feature_row["is_conference_tourney"] = 0.0
        feature_row["is_ncaa_tourney"] = 0.0
    # Days rest (model expects days_rest_home, days_rest_away; 0–14 cap)
    def _days_rest(team: str) -> float:
        if not last_game_dates or not team or not game_date:
            return 0.0
        prev = last_game_dates.get(team.strip())
        if prev is None:
            return 0.0
        try:
            gd = datetime.strptime(game_date, "%Y-%m-%d").date()
            d = prev.date() if hasattr(prev, "date") else pd.Timestamp(prev).date()
            delta = (gd - d).days
            return float(min(14, max(0, delta)))
        except Exception:
            return 0.0
    feature_row["days_rest_home"] = _days_rest(home_team)
    feature_row["days_rest_away"] = _days_rest(away_team)


def _get_latest_odds_path() -> Path | None:
    """Return the most recent CSV in data/odds/ by modification time, or None."""
    if not ODDS_DIR.exists():
        return None
    csvs = list(ODDS_DIR.glob("*.csv"))
    if not csvs:
        return None
    return max(csvs, key=lambda p: p.stat().st_mtime)


def _game_date_from_odds_path(odds_path: Path) -> str:
    """Derive YYYY-MM-DD from odds filename (e.g. March_08_2026_Odds.csv -> 2026-03-08), else use today with ODDS_YEAR."""
    stem = odds_path.stem  # e.g. March_08_2026_Odds
    try:
        # Expect Month_DD_YYYY or Month_D_YYYY before _Odds
        base = stem.replace("_Odds", "").strip()
        parts = base.split("_")
        if len(parts) >= 3:
            month_str, day_str, year_str = parts[0], parts[1], parts[2]
            month_map = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
                         "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}
            mo = month_map.get(month_str)
            if mo is not None:
                day = int(day_str)
                yr = int(year_str)
                if 1 <= mo <= 12 and 1 <= day <= 31 and 2020 <= yr <= 2030:
                    return f"{yr}-{mo:02d}-{day:02d}"
    except (ValueError, IndexError, AttributeError):
        pass
    today_md = datetime.now().strftime("%m-%d")
    return f"{ODDS_YEAR}-{today_md}"


def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Allow case-insensitive headers; map to exact Home_Team, Away_Team, Spread, Over_Under, Time."""
    rename = {}
    for col in df.columns:
        n = col.strip().lower().replace(" ", "_")
        if n in ("home_team", "home"):
            rename[col] = "Home_Team"
        elif n in ("away_team", "away"):
            rename[col] = "Away_Team"
        elif n in ("spread", "line"):
            rename[col] = "Spread"
        elif n in ("over_under", "total", "ou"):
            rename[col] = "Over_Under"
        elif n in ("time", "start_time", "tip", "game_time", "start"):
            rename[col] = "Time"
        elif n in ("is_neutral", "neutral", "neutral_site"):
            rename[col] = "Is_Neutral"
    return df.rename(columns=rename)


def _get_time_column(df: pd.DataFrame) -> str | None:
    """Return 'Time' if present in df, else None."""
    return "Time" if "Time" in df.columns else None


def _is_neutral_from_row(row: pd.Series) -> bool:
    """True if Is_Neutral column is truthy (1, True, yes, y, etc.)."""
    if "Is_Neutral" not in row.index:
        return False
    val = row.get("Is_Neutral")
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "y"):
        return True
    try:
        return bool(float(val))
    except (TypeError, ValueError):
        return False


def _parse_time_value_to_iso_utc(raw: str, game_date: str) -> str | None:
    """Parse a time string (e.g. '7:00 PM', '19:00') with game_date YYYY-MM-DD; assume ET; return ISO UTC string."""
    raw = str(raw).strip() if raw is not None else ""
    if not raw or raw.lower() in ("nan", "nat", "—", "-"):
        return None
    try:
        yr, mo, day = (int(game_date[:4]), int(game_date[5:7]), int(game_date[8:10]))
        et = ZoneInfo("America/New_York")
        # Try common formats
        for fmt in ("%I:%M %p", "%I:%M%p", "%H:%M", "%I %p", "%I%p"):
            try:
                # Parse time only
                t = datetime.strptime(raw.strip(), fmt)
                dt_et = datetime(yr, mo, day, t.hour, t.minute, 0, tzinfo=et)
                dt_utc = dt_et.astimezone(timezone.utc)
                return dt_utc.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
            except ValueError:
                continue
        # 12:00 or 12 PM style
        m = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", raw.strip(), re.I)
        if m:
            h = int(m.group(1))
            minute = int(m.group(2) or 0)
            ap = (m.group(3) or "").lower()
            if ap == "pm" and h != 12:
                h += 12
            elif ap == "am" and h == 12:
                h = 0
            elif not ap and h <= 12:
                pass  # assume PM for afternoon
            dt_et = datetime(yr, mo, day, h % 24, minute, 0, tzinfo=et)
            dt_utc = dt_et.astimezone(timezone.utc)
            return dt_utc.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    except (ValueError, TypeError, IndexError):
        pass
    return None


def _game_start_from_row(row: pd.Series, game_date: str, time_col: str | None) -> str | None:
    """Get game start time ISO UTC from row: use Time column if present and parseable."""
    if not time_col or time_col not in row.index:
        return None
    return _parse_time_value_to_iso_utc(row.get(time_col), game_date)


def _espn_start_time_for_match(espn_times: dict[tuple[str, str], str], home: str, away: str) -> str | None:
    """Look up (home, away) in espn_times; try exact then (away, home); then fuzzy match keys."""
    home = (home or "").strip()
    away = (away or "").strip()
    if (home, away) in espn_times:
        return espn_times[(home, away)]
    if (away, home) in espn_times:
        return espn_times[(away, home)]
    # Fuzzy match against keys
    try:
        from thefuzz import fuzz
        from thefuzz import process as fuzz_process
    except ImportError:
        return None
    best_score = 0
    best_commence: str | None = None
    for (h, a), commence in espn_times.items():
        sh = fuzz.token_set_ratio(home, h) if home and h else 0
        sa = fuzz.token_set_ratio(away, a) if away and a else 0
        if sh >= 70 and sa >= 70 and (sh + sa) > best_score:
            best_score = sh + sa
            best_commence = commence
    return best_commence


def _load_kenpom_from_csv() -> pd.DataFrame:
    """Load KenPom stats from data/ncaab/ (same CSVs used by engine/betting_models.py via merge script).
    Prefer team_stats_2026.csv (current season); else team_stats_combined.csv, team_stats_2025.csv, etc. Returns DataFrame with
    season, TEAM, and KENPOM_STAT_COLUMNS; empty if no file found."""
    for name in ["team_stats_2026.csv", "team_stats_combined.csv", "team_stats_2025.csv", "team_stats_2024.csv", "team_stats_2023.csv"]:
        path = NCAAB_DIR / name
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty or "TEAM" not in df.columns:
            continue
        if "season" not in df.columns and name != "team_stats_combined.csv":
            # Per-season file: infer from filename (e.g. team_stats_2025 -> 2025)
            try:
                year = int(name.replace("team_stats_", "").replace(".csv", ""))
                df["season"] = year
            except ValueError:
                continue
        df["season"] = df["season"].astype(int)
        keep = ["season", "TEAM"] + [c for c in KENPOM_STAT_COLUMNS if c in df.columns]
        return df[[c for c in keep if c in df.columns]].copy()
    return pd.DataFrame()


def _get_candidate_teams(kenpom_df: pd.DataFrame | None = None) -> list[str]:
    """Collect team names from team_stats_history, ncaab_team_season_stats, and KenPom CSV."""
    candidates = []
    try:
        from engine.team_stats_history import load_team_stats_history
        hist = load_team_stats_history()
        if not hist.empty and "Team" in hist.columns:
            candidates.extend(hist["Team"].astype(str).str.strip().dropna().unique().tolist())
    except Exception:
        pass
    try:
        import sqlite3
        db = APP_ROOT / "data" / "espn.db"
        if db.exists():
            conn = sqlite3.connect(db)
            try:
                df = pd.read_sql_query("SELECT TEAM FROM ncaab_team_season_stats", conn)
                if not df.empty:
                    candidates.extend(df["TEAM"].astype(str).str.strip().dropna().unique().tolist())
            finally:
                conn.close()
    except Exception:
        pass
    if kenpom_df is not None and not kenpom_df.empty and "TEAM" in kenpom_df.columns:
        candidates.extend(kenpom_df["TEAM"].astype(str).str.strip().dropna().unique().tolist())
    return list(dict.fromkeys(candidates))


def _resolve_team_for_lookup(raw_name: str) -> str:
    """Apply static mapping from odds name to stats DB name."""
    name = str(raw_name).strip()
    return ODDS_TO_STATS_NAME.get(name, name)


def _fuzzy_match_team(name: str, candidates: list[str], raw_name_for_warning: str | None = None) -> tuple[str | None, int]:
    """Return (best_match, score) or (None, 0). Uses thefuzz (fuzzywuzzy-compatible)."""
    if not name or not candidates:
        return None, 0
    try:
        from thefuzz import fuzz
        from thefuzz import process as fuzz_process
    except ImportError:
        return None, 0
    name = str(name).strip()
    display_name = (raw_name_for_warning or name).strip()
    match = fuzz_process.extractOne(name, candidates, scorer=fuzz.ratio)
    if not match or match[1] < FUZZY_MIN_SCORE:
        match2 = fuzz_process.extractOne(name, candidates, scorer=fuzz.token_set_ratio)
        if match2 and match2[1] >= FUZZY_MIN_SCORE:
            if match2[1] < FUZZY_WARN_SCORE:
                print(f"MISSING DATA FOR: [{display_name}] (fuzzy score {match2[1]}, consider adding to ODDS_TO_STATS_NAME)", file=sys.stderr)
            return match2[0], match2[1]
        if match and match[1] >= FUZZY_MIN_SCORE:
            if match[1] < FUZZY_WARN_SCORE:
                print(f"MISSING DATA FOR: [{display_name}] (fuzzy score {match[1]}, consider adding to ODDS_TO_STATS_NAME)", file=sys.stderr)
            return match[0], match[1]
        if match:
            print(f"MISSING DATA FOR: [{display_name}] (best match: {match[0]} @ {match[1]}, below threshold)", file=sys.stderr)
        return (match[0], match[1]) if match and match[1] >= FUZZY_MIN_SCORE else (None, 0)
    if match[1] < FUZZY_WARN_SCORE:
        print(f"MISSING DATA FOR: [{display_name}] (fuzzy score {match[1]}, consider adding to ODDS_TO_STATS_NAME)", file=sys.stderr)
    return match[0], match[1]


def _build_feature_row_from_kenpom(
    kenpom_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    effective_season: int,
) -> pd.Series | None:
    """Build a feature row for NCAAB XGBoost spread model from KenPom DataFrame.
    home_team/away_team must match TEAM in kenpom_df (use fuzzy-matched names). Returns None if either team missing."""
    if kenpom_df.empty or "TEAM" not in kenpom_df.columns or "season" not in kenpom_df.columns:
        return None
    sub = kenpom_df[kenpom_df["season"] == effective_season]
    if sub.empty:
        # Fallback to latest season in dataframe
        sub = kenpom_df[kenpom_df["season"] == kenpom_df["season"].max()]
    if sub.empty:
        return None
    home_row = sub[sub["TEAM"].astype(str).str.strip() == str(home_team).strip()]
    away_row = sub[sub["TEAM"].astype(str).str.strip() == str(away_team).strip()]
    if home_row.empty or away_row.empty:
        return None
    home_row = home_row.iloc[0]
    away_row = away_row.iloc[0]
    _seed_default = 99.0

    def _float(val, default: float = 0.0) -> float:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    row = {"league": "ncaab", "home_team_name": home_team, "away_team_name": away_team}
    for col in KENPOM_STAT_COLUMNS:
        if col == "SEED":
            row["home_seed"] = _float(home_row.get(col), _seed_default) or _seed_default
            row["away_seed"] = _float(away_row.get(col), _seed_default) or _seed_default
        else:
            # THREE_PT_RATE may appear as 3PR or 3pr in CSV
            val_h = _float(home_row.get(col)) if col in home_row.index else None
            val_a = _float(away_row.get(col)) if col in away_row.index else None
            if val_h is None and col == "THREE_PT_RATE":
                val_h = _float(home_row.get("3PR") or home_row.get("3pr"))
            if val_a is None and col == "THREE_PT_RATE":
                val_a = _float(away_row.get("3PR") or away_row.get("3pr"))
            row[f"home_{col}"] = val_h if val_h is not None else 0.0
            row[f"away_{col}"] = val_a if val_a is not None else 0.0
    row["home_SEED"] = row.get("home_seed", _seed_default)
    row["away_SEED"] = row.get("away_seed", _seed_default)
    row["home_days_rest"] = 0.0
    row["away_days_rest"] = 0.0
    row["home_games_in_last_5_days"] = 0
    row["away_games_in_last_5_days"] = 0
    row["home_is_b2b"] = 0
    row["away_is_b2b"] = 0
    return pd.Series(row)


def main() -> None:
    path = _get_latest_odds_path()
    if path is None:
        print(f"No CSV found in {ODDS_DIR}. Add a file (e.g. March_07_2026_Odds.csv) and run again.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    df = _normalize_headers(df)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"Missing columns {missing}. Required: {REQUIRED_COLUMNS}. File: {path}", file=sys.stderr)
        sys.exit(1)
    # Raw odds CSV: Spread is from Home_Team perspective — negative = home favored, positive = away favored
    print("--- Raw odds CSV (Home_Team, Away_Team, Spread) — Spread = home perspective: neg = home favored ---")
    print(f"File: {path}")
    for _, row in df.iterrows():
        h = row.get("Home_Team", "")
        a = row.get("Away_Team", "")
        s = row.get("Spread", "")
        print(f"  {h} | {a} | Spread: {s}")
    print()
    # Critical: ensure prediction feature row matches NCAAB model feature_columns
    model_feature_columns = _load_ncaab_model_feature_columns()
    _ensure_ncaab_model_feature_alignment(model_feature_columns)
    # Try to load XGBoost spread model for predictions (otherwise fall back to BARTHAG)
    spread_model_available = False
    if NCAAB_SPREAD_MODEL_PATH.exists():
        try:
            import joblib as _jl
            _payload = _jl.load(NCAAB_SPREAD_MODEL_PATH)
            if _payload.get("model") is not None and _payload.get("feature_columns"):
                spread_model_available = True
        except Exception:
            pass
    if spread_model_available:
        print("Spread model: Using XGBoost spread model (data/models/xgboost_spread_ncaab.pkl)")
    else:
        print("Spread model: Falling back to BARTHAG formula (model not found or invalid: data/models/xgboost_spread_ncaab.pkl)")
    form_df = _load_current_form()
    cover_prob_params = _load_cover_prob_params()
    totals_model_payload = _load_totals_model()
    team_hca, conf_hca, team_conf = _load_hca_lookups()
    hca_2026_computed = _compute_2026_hca_from_games()
    from engine.utils import game_season_from_date, effective_kenpom_season
    kenpom_df = _load_kenpom_from_csv()
    candidates = _get_candidate_teams(kenpom_df)
    if not candidates:
        print("No team stats found (team_stats_history, ncaab_team_season_stats, or KenPom CSV). Fuzzy match will be limited.", file=sys.stderr)
    from engine.betting_models import (
        build_ncaab_feature_row_from_team_stats,
        consensus_spread,
        consensus_totals,
        get_spread_predicted_margin,
    )
    from engine.engine import get_team_power_ratings, get_nba_team_pace_stats, NBA_LEAGUE_AVG_PACE, NBA_LEAGUE_AVG_OFF_RATING
    from engine.espn_odds import get_ncaab_start_times_for_date
    NCAAB_DEFAULT_RATING = (NBA_LEAGUE_AVG_PACE * NBA_LEAGUE_AVG_OFF_RATING) / 100.0
    from strategies.strategies import implied_probability_no_vig, spread_cover_prob_from_margins
    # Use date from odds filename (e.g. March_08_2026_Odds.csv -> 2026-03-08) so saved plays match the slate
    game_date = _game_date_from_odds_path(path)
    game_date_obj = datetime.strptime(game_date, "%Y-%m-%d").date()
    espn_start_times = get_ncaab_start_times_for_date(game_date_obj)
    time_col = _get_time_column(df)
    season = game_season_from_date(game_date) or ODDS_YEAR
    effective_season = effective_kenpom_season(game_date, season) or season
    pace_stats = get_nba_team_pace_stats()
    power_ratings = get_team_power_ratings(pace_stats, NBA_LEAGUE_AVG_PACE, NBA_LEAGUE_AVG_OFF_RATING)
    default_rating = NCAAB_DEFAULT_RATING
    # Days rest from most recent completed game per team before game_date
    last_game_dates: dict[str, datetime] = {}
    if HISTORICAL_GAMES_PATH.exists():
        hg = pd.read_csv(HISTORICAL_GAMES_PATH)
        if "date" in hg.columns and "home_team" in hg.columns and "away_team" in hg.columns:
            hg["date"] = pd.to_datetime(hg["date"], errors="coerce")
            hg = hg.dropna(subset=["date"])
            hg = hg[hg["date"] < pd.to_datetime(game_date)].copy()
            if not hg.empty:
                hg = hg.sort_values("date")
                for _, r in hg.iterrows():
                    d = r["date"]
                    ht = str(r.get("home_team", "")).strip()
                    at = str(r.get("away_team", "")).strip()
                    if ht:
                        last_game_dates[ht] = d
                    if at:
                        last_game_dates[at] = d
    results = []
    used_kenpom = 0
    used_fallback = 0
    for _, row in df.iterrows():
        home_raw = str(row.get("Home_Team", "")).strip()
        away_raw = str(row.get("Away_Team", "")).strip()
        try:
            spread = float(row.get("Spread", 0))
        except (TypeError, ValueError):
            spread = 0.0
        if home_raw == "Michigan" and away_raw == "Wisconsin":
            print(f"DEBUG: Michigan vs Wisconsin — Spread read from CSV: {spread} (expected -12.5 if home favored)")
        try:
            total = float(row.get("Over_Under", 0))
        except (TypeError, ValueError):
            total = 0.0
        home_lookup = _resolve_team_for_lookup(home_raw)
        away_lookup = _resolve_team_for_lookup(away_raw)
        home_matched, home_score = _fuzzy_match_team(home_lookup, candidates, raw_name_for_warning=home_raw)
        away_matched, away_score = _fuzzy_match_team(away_lookup, candidates, raw_name_for_warning=away_raw)
        home = home_matched or home_lookup
        away = away_matched or away_lookup
        is_neutral = _is_neutral_from_row(row)
        # Prefer KenPom from data/ncaab/ to run XGBoost spread model; else fall back to espn.db / simple model
        feature_row = None
        used_kenpom_this_game = False
        if not kenpom_df.empty:
            feature_row = _build_feature_row_from_kenpom(kenpom_df, home, away, effective_season)
            if feature_row is not None:
                used_kenpom += 1
                used_kenpom_this_game = True
        if feature_row is None:
            feature_row = build_ncaab_feature_row_from_team_stats(home, away, game_date=game_date)
            if feature_row is None:
                used_fallback += 1
        if feature_row is not None:
            if "home_seed" in feature_row.index and "home_SEED" not in feature_row.index:
                feature_row["home_SEED"] = feature_row.get("home_seed", 0)
            if "away_seed" in feature_row.index and "away_SEED" not in feature_row.index:
                feature_row["away_SEED"] = feature_row.get("away_seed", 0)
            _enrich_feature_row_for_ncaab_model(
                feature_row, home, away, is_neutral, season,
                form_df, team_hca, conf_hca, team_conf, hca_2026_computed,
                game_date=game_date,
                last_game_dates=last_game_dates,
            )
        # BARTHAG: extract for margin (used for fallback and for Michigan vs Wisconsin debug)
        h_b, a_b = None, None
        if feature_row is not None:
            idx = getattr(feature_row, "index", None)
            if idx is not None and hasattr(idx, "__contains__"):
                h_b = feature_row["home_BARTHAG"] if "home_BARTHAG" in idx else feature_row.get("home_BARTHAG")
                a_b = feature_row["away_BARTHAG"] if "away_BARTHAG" in idx else feature_row.get("away_BARTHAG")
            else:
                h_b = feature_row.get("home_BARTHAG")
                a_b = feature_row.get("away_BARTHAG")
        market_spread = float(spread)
        barthag_margin = None
        if h_b is not None and pd.notna(h_b) and a_b is not None and pd.notna(a_b):
            try:
                barthag_margin = (float(h_b) - float(a_b)) * BARTHAG_TO_POINTS_SCALE
                if is_neutral:
                    barthag_margin -= NEUTRAL_SITE_ADJUSTMENT_PTS
            except (TypeError, ValueError):
                pass
        # XGB path: use model when available and feature row is valid; else fall back to BARTHAG
        pred_margin = None
        margin_source = "BARTHAG"
        fallback_reason = ""
        xgb_margin_raw = None
        if spread_model_available and feature_row is not None:
            xgb_margin_raw = get_spread_predicted_margin(
                feature_row, market_spread, "ncaab", home_team=home, away_team=away
            )
            if xgb_margin_raw is not None:
                pred_margin = float(xgb_margin_raw)  # model has is_neutral in features, no extra adj
                margin_source = "XGB"
            else:
                fallback_reason = "XGB returned None (missing features or model error)"
        else:
            if not spread_model_available:
                fallback_reason = "XGB model not loaded"
            elif feature_row is None:
                fallback_reason = "no feature row"
            else:
                fallback_reason = "model not available"
        if pred_margin is None and barthag_margin is not None:
            pred_margin = barthag_margin
            if not fallback_reason:
                fallback_reason = "BARTHAG fallback (no XGB)"
        # Per-game path and Michigan vs Wisconsin: print both margins when applicable
        if home_raw == "Michigan" and away_raw == "Wisconsin":
            print(f"  [Michigan vs Wisconsin] XGB pred margin: {xgb_margin_raw:+.2f}" if xgb_margin_raw is not None else "  [Michigan vs Wisconsin] XGB pred margin: N/A")
            print(f"  [Michigan vs Wisconsin] BARTHAG pred margin: {barthag_margin:+.2f}" if barthag_margin is not None else "  [Michigan vs Wisconsin] BARTHAG pred margin: N/A")
            print(f"  [Michigan vs Wisconsin] Used: {margin_source}  (pred_margin = {pred_margin:+.2f})")
        else:
            if margin_source == "BARTHAG" and fallback_reason:
                print(f"  [{away} @ {home}] BARTHAG — {fallback_reason}")
            elif margin_source == "XGB":
                print(f"  [{away} @ {home}] XGB")
        # Market blending: blend with market-implied margin (home perspective: margin = -spread)
        market_implied_margin = -market_spread
        blended_margin = None
        margin_for_edge = pred_margin
        if pred_margin is not None and USE_MARKET_BLEND:
            blended_margin = (MODEL_WEIGHT * pred_margin) + ((1 - MODEL_WEIGHT) * market_implied_margin)
            margin_for_edge = blended_margin
        elif pred_margin is not None:
            blended_margin = pred_margin
        # Edge: compare margin_for_edge to required margin. Home favored (spread < 0): edge = margin - abs(spread). Away favored: edge = margin + abs(spread).
        if margin_for_edge is not None:
            abs_spread = abs(market_spread)
            if market_spread < 0:
                edge_pts = margin_for_edge - abs_spread
            else:
                edge_pts = margin_for_edge + abs_spread
        else:
            edge_pts = None
        # Pick: Home if edge > 3, Away if edge < -3, Skip if |edge| <= 3
        if edge_pts is not None:
            if edge_pts > VALUE_PLAY_THRESHOLD:
                pick_spread = "Home"
            elif edge_pts < -VALUE_PLAY_THRESHOLD:
                pick_spread = "Away"
            else:
                pick_spread = "Skip"
        else:
            pick_spread = "Skip"
        is_value_play = edge_pts is not None and abs(edge_pts) > VALUE_PLAY_THRESHOLD
        we_cover_favorite = spread <= 0
        in_house_spread = 0.0
        if feature_row is not None and power_ratings:
            home_r = power_ratings.get(home, default_rating)
            away_r = power_ratings.get(away, default_rating)
            in_house_spread = (home_r - away_r) / 100.0
        spread_prob, spread_ok = consensus_spread(feature_row, spread, we_cover_favorite, in_house_spread, 0.5, league="ncaab") if feature_row is not None else (0.5, True)
        # Calibrated P(cover): logistic using raw model edge (pred_margin - market_spread), if params available
        calibrated_p_cover = None
        if cover_prob_params is not None and pred_margin is not None:
            edge_for_prob = pred_margin - market_spread
            k = cover_prob_params["k"]
            intercept = cover_prob_params["intercept"]
            z = k * edge_for_prob + intercept
            try:
                calibrated_p_cover = 1.0 / (1.0 + math.exp(-z))
            except OverflowError:
                calibrated_p_cover = 1.0 if z > 0 else 0.0
            spread_prob = float(max(0.0, min(1.0, calibrated_p_cover)))
        elif margin_for_edge is not None:
            we_pick_favorite = (pick_spread == "Home" and market_spread <= 0) or (pick_spread == "Away" and market_spread > 0)
            spread_prob = spread_cover_prob_from_margins(margin_for_edge, market_spread, we_pick_favorite)
        over_prob, tot_ok = consensus_totals(feature_row, total, True, total, "ncaab", 0.5) if feature_row is not None and total else (0.5, True)
        under_prob = 1.0 - over_prob if total else 0.5
        # Kelly stake from P(cover): full Kelly = (p * decimal_odds - 1) / (decimal_odds - 1); use 25% fractional, cap stake 0.5–5%
        p_cover_for_kelly = (calibrated_p_cover if pick_spread == "Home" else (1.0 - calibrated_p_cover)) if calibrated_p_cover is not None else spread_prob
        denom = DECIMAL_ODDS_110 - 1.0
        kelly_fraction_raw = (p_cover_for_kelly * DECIMAL_ODDS_110 - 1.0) / denom if denom and denom > 0 else 0.0
        kelly_fraction_raw = max(0.0, kelly_fraction_raw)
        kelly_fraction = kelly_fraction_raw * KELLY_FRACTION_PCT
        suggested_stake_pct = max(SUGGESTED_STAKE_PCT_MIN, min(SUGGESTED_STAKE_PCT_MAX, kelly_fraction * 100.0))
        # Confidence bands: use calibrated P(cover) when available; else fall back to edge-based bands
        if calibrated_p_cover is not None:
            if calibrated_p_cover >= 0.65:
                confidence = "High"
            elif calibrated_p_cover >= 0.57:
                confidence = "Medium"
            else:
                confidence = "Low"
        else:
            if edge_pts is not None:
                abs_edge = abs(edge_pts)
                if abs_edge > EDGE_HIGH_CONFIDENCE:
                    confidence = "High"
                elif abs_edge > VALUE_PLAY_THRESHOLD:
                    confidence = "Medium"
                else:
                    confidence = "Low"
            else:
                confidence = "Low"
        home_roff_raw = feature_row.get("home_ROff") if feature_row is not None else None
        away_roff_raw = feature_row.get("away_ROff") if feature_row is not None else None
        roff_available = (
            home_roff_raw is not None
            and away_roff_raw is not None
            and not pd.isna(home_roff_raw)
            and not pd.isna(away_roff_raw)
        )
        if roff_available:
            try:
                home_roff = float(home_roff_raw)
                away_roff = float(away_roff_raw)
            except (TypeError, ValueError):
                roff_available = False
                home_roff = away_roff = 0.0
        else:
            home_roff = away_roff = 0.0
        # Advantage = picked side's ROff minus opponent's (only star when picked side is actually higher)
        roff_advantage = (home_roff - away_roff) if pick_spread == "Home" else (away_roff - home_roff) if pick_spread == "Away" and roff_available else 0.0
        rebounding_star = roff_available and confidence == "High" and roff_advantage > ROFF_ADVANTAGE_THRESHOLD
        # Reliability: flag when away BARTHAG > home BARTHAG + 0.25 (exclude from value plays)
        reliability_flag = None
        if h_b is not None and a_b is not None:
            try:
                h_val = float(h_b) if pd.notna(h_b) else None
                a_val = float(a_b) if pd.notna(a_b) else None
                if h_val is not None and a_val is not None and (a_val - h_val) > BARTHAG_MISMATCH_THRESHOLD:
                    reliability_flag = LOW_RELIABILITY_FLAG
            except (TypeError, ValueError):
                pass
        start_time = _game_start_from_row(row, game_date, time_col) or _espn_start_time_for_match(espn_start_times, home, away)
        if start_time and not start_time.endswith("Z") and "+" not in start_time:
            try:
                dt = datetime.fromisoformat(start_time.replace("Z", ""))
                if dt.tzinfo is None:
                    et = ZoneInfo("America/New_York")
                    dt_et = dt.replace(tzinfo=et)
                    start_time = dt_et.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z"
            except (ValueError, TypeError):
                pass
        results.append({
            "Home": home,
            "Away": away,
            "Home_Raw": home_raw,
            "Away_Raw": away_raw,
            "Home_BARTHAG": float(h_b) if h_b is not None and pd.notna(h_b) else None,
            "Away_BARTHAG": float(a_b) if a_b is not None and pd.notna(a_b) else None,
            "Market_Spread": market_spread,
            "Over_Under": total,
            "Pred_Margin": pred_margin,
            "Margin_Source": margin_source,
            "Blended_Margin": blended_margin,
            "Edge": edge_pts,
            "Calibrated_P_Cover": calibrated_p_cover,
            "Confidence": confidence,
            "Is_Value_Play": is_value_play,
            "Rebounding_Star": rebounding_star,
            "ROff_Stats_Available": roff_available,
            "Used_KenPom": used_kenpom_this_game,
            "Spread_Prob": spread_prob,
            "Over_Prob": over_prob,
            "Pick_Spread": pick_spread,
            "Pick_Total": "Over" if over_prob > 0.5 else "Under",
            "Reliability_Flag": reliability_flag,
            "start_time": start_time,
            "Is_Neutral": is_neutral,
            "Kelly_Fraction": kelly_fraction,
            "Suggested_Stake_Pct": suggested_stake_pct,
            "_feature_row": feature_row,
        })
    # Totals model: predicted total and edge vs market; recommend Over/Under if |edge| > 3
    if totals_model_payload is not None and results:
        totals_model, totals_feature_columns, totals_bias_correction = totals_model_payload
        import numpy as np
        for r in results:
            total = r.get("Over_Under")
            fr = r.get("_feature_row")
            if fr is None or total is None or not total:
                r["Predicted_Total"] = None
                r["Totals_Edge"] = None
                r["Pick_Total_Value"] = None
                continue
            totals_feat = _build_totals_feature_row(fr, totals_feature_columns)
            if not totals_feat:
                r["Predicted_Total"] = None
                r["Totals_Edge"] = None
                r["Pick_Total_Value"] = None
                continue
            X = np.array([[totals_feat[c] for c in totals_feature_columns]], dtype=np.float64)
            try:
                pred_raw = float(totals_model.predict(X)[0])
                pred_total = pred_raw - totals_bias_correction
            except Exception:
                r["Predicted_Total"] = None
                r["Totals_Edge"] = None
                r["Pick_Total_Value"] = None
                continue
            totals_edge = pred_total - float(total)
            r["Predicted_Total"] = pred_total
            r["Totals_Edge"] = totals_edge
            if totals_edge > TOTALS_EDGE_THRESHOLD:
                r["Pick_Total_Value"] = "Over"
            elif totals_edge < -TOTALS_EDGE_THRESHOLD:
                r["Pick_Total_Value"] = "Under"
            else:
                r["Pick_Total_Value"] = None
        for r in results:
            r.pop("_feature_row", None)
    else:
        for r in results:
            r["Predicted_Total"] = None
            r["Totals_Edge"] = None
            r["Pick_Total_Value"] = None
    value_plays_raw = [r for r in results if r["Is_Value_Play"]]
    value_plays = value_plays_raw
    filtered_mismatch = 0
    n_after_reliability = len(value_plays)
    if value_plays:
        home_plays = sorted([r for r in value_plays if r["Pick_Spread"] == "Home"], key=lambda x: abs(x.get("Edge") or 0), reverse=True)
        away_plays = sorted([r for r in value_plays if r["Pick_Spread"] == "Away"], key=lambda x: abs(x.get("Edge") or 0), reverse=True)
        n_total = len(value_plays)
        max_home = max(1, int(VALUE_PLAY_MAX_HOME_RATIO * n_total))
        if len(home_plays) > max_home:
            value_plays = home_plays[:max_home] + away_plays
        else:
            value_plays = home_plays + away_plays
        value_plays = sorted(value_plays, key=lambda x: abs(x.get("Edge") or 0), reverse=True)
        n_home = sum(1 for r in value_plays if r["Pick_Spread"] == "Home")
        n_total_after = len(value_plays)
        min_home = max(1, (n_total_after * 30 + 99) // 100)
        if n_home < min_home and len(home_plays) > n_home:
            in_set = {(r["Home"], r["Away"]) for r in value_plays}
            need = min_home - n_home
            extra_home = [p for p in home_plays if (p["Home"], p["Away"]) not in in_set][:need]
            if extra_home:
                value_plays = value_plays + extra_home
                value_plays = sorted(value_plays, key=lambda x: abs(x.get("Edge") or 0), reverse=True)
            n_home = sum(1 for r in value_plays if r["Pick_Spread"] == "Home")
        n_away = len(value_plays) - n_home
        pct_home = (100.0 * n_home / len(value_plays)) if value_plays else 0
        print(f"Value plays after diversity (max {int(VALUE_PLAY_MAX_HOME_RATIO*100)}% Home, min {int(VALUE_PLAY_MIN_HOME_RATIO*100)}% Home): {n_home} Home / {n_away} Away  ({pct_home:.0f}% Home).")
        print(f"Final Home vs Away ratio: {n_home} / {n_away}  ({pct_home:.0f}% Home).")
    value_plays_set = {(r["Home"], r["Away"]) for r in value_plays} if value_plays else set()
    high_confidence = [r for r in results if r.get("Confidence") == "High" and not r.get("Reliability_Flag")]
    n_xgb = sum(1 for r in results if r.get("Margin_Source") == "XGB")
    n_barthag = sum(1 for r in results if r.get("Margin_Source") == "BARTHAG")
    print(f"Loaded: {path.name} ({len(results)} games)  |  Value plays (|edge| > {VALUE_PLAY_THRESHOLD} pts): {len(value_plays_raw)}  |  After diversity: {len(value_plays)}  |  High confidence: {len(high_confidence)}")
    print(f"Spread predictions: {n_xgb} XGB, {n_barthag} BARTHAG  |  KenPom stats: {used_kenpom}  |  Fallback (Stats N/A): {used_fallback}")
    print(f"Market blend: {'ON' if USE_MARKET_BLEND else 'OFF'}  (model weight={MODEL_WEIGHT}, edge uses {'blended' if USE_MARKET_BLEND else 'raw'} margin)\n")
    # Team name lookup: odds file name -> matched name used for stats (verify no wrong matches)
    print("Team name lookup (odds -> stats) — ALL games from odds file")
    print("-" * 95)
    print(f"{'Odds Home':<28} -> {'Matched Home':<28} | {'Odds Away':<28} -> {'Matched Away':<28}")
    print("-" * 95)
    for r in results:
        h_raw = r.get("Home_Raw", "")
        a_raw = r.get("Away_Raw", "")
        h_mat = r.get("Home", "")
        a_mat = r.get("Away", "")
        print(f"{h_raw:<28} -> {h_mat:<28} | {a_raw:<28} -> {a_mat:<28}")
    print("-" * 95)
    # BARTHAG by game: Home vs Away so we can see if home is genuinely better in most games
    print("\nBARTHAG by game (today's slate) — ALL games")
    print("-" * 80)
    print(f"{'Game':<45} | {'Home BARTHAG':>12} | {'Away BARTHAG':>12} | Who is better")
    print("-" * 80)
    for r in results:
        game = f"{r['Away']} @ {r['Home']}"
        h_b = r.get("Home_BARTHAG")
        a_b = r.get("Away_BARTHAG")
        h_str = f"{h_b:.3f}" if h_b is not None else "—"
        a_str = f"{a_b:.3f}" if a_b is not None else "—"
        if h_b is not None and a_b is not None:
            if h_b > a_b:
                who = "Home"
            elif a_b > h_b:
                who = "Away"
            else:
                who = "Tie"
        else:
            who = "N/A"
        print(f"{game:<45} | {h_str:>12} | {a_str:>12} | {who}")
    print("-" * 80)
    # Full diagnostic: Home_BARTHAG, Away_BARTHAG, market_spread, pred_margin (raw + blended), edge, pick
    print("\nBARTHAG + margin + edge + pick — ALL games from odds file (including Skip)")
    if USE_MARKET_BLEND:
        print(f"  (Market blend ON: weight={MODEL_WEIGHT}; edge uses blended_margin)")
    print("-" * 140)
    header = f"{'Game':<38} | {'H_BARTHAG':>9} | {'A_BARTHAG':>9} | {'Spread':>7} | {'Pred(raw)':>9} | {'Blended':>8} | {'Edge':>6} | Pick"
    if not USE_MARKET_BLEND:
        header = f"{'Game':<38} | {'H_BARTHAG':>9} | {'A_BARTHAG':>9} | {'Spread':>7} | {'Pred':>6} | {'Edge':>6} | Pick"
    print(header)
    print("-" * 140)
    for r in results:
        game = f"{r['Away']} @ {r['Home']}"
        h_b = r.get("Home_BARTHAG")
        a_b = r.get("Away_BARTHAG")
        h_str = f"{h_b:.3f}" if h_b is not None else "—"
        a_str = f"{a_b:.3f}" if a_b is not None else "—"
        spread = r.get("Market_Spread")
        spread_s = f"{spread:+.1f}" if spread is not None else "—"
        pred = r.get("Pred_Margin")
        pred_s = f"{pred:+.1f}" if pred is not None else "—"
        blended = r.get("Blended_Margin")
        blended_s = f"{blended:+.1f}" if blended is not None else "—"
        edge = r.get("Edge")
        edge_s = f"{edge:+.1f}" if edge is not None else "—"
        pick = r.get("Pick_Spread", "Skip")
        if USE_MARKET_BLEND:
            print(f"{game:<38} | {h_str:>9} | {a_str:>9} | {spread_s:>7} | {pred_s:>9} | {blended_s:>8} | {edge_s:>6} | {pick}")
        else:
            print(f"{game:<38} | {h_str:>9} | {a_str:>9} | {spread_s:>7} | {pred_s:>6} | {edge_s:>6} | {pick}")
    print("-" * 140)
    # Spread sign check: flag games where BARTHAG disagrees with spread sign (likely CSV sign error)
    # Convention: Spread = home perspective. Negative = home favored, positive = away favored.
    # Flag if: (home has higher BARTHAG but spread > 0) or (away has higher BARTHAG but spread < 0)
    spread_sign_flags = []
    for r in results:
        h_b = r.get("Home_BARTHAG")
        a_b = r.get("Away_BARTHAG")
        spread = r.get("Market_Spread")
        if h_b is None or a_b is None or spread is None:
            continue
        home_better = h_b > a_b
        away_better = a_b > h_b
        if home_better and spread > 0:
            spread_sign_flags.append({
                "game": f"{r['Away']} @ {r['Home']}",
                "home": r["Home"],
                "away": r["Away"],
                "spread": spread,
                "home_barthag": h_b,
                "away_barthag": a_b,
                "reason": "Home has higher BARTHAG but spread is positive (home listed as underdog) — likely sign error",
            })
        elif away_better and spread < 0:
            spread_sign_flags.append({
                "game": f"{r['Away']} @ {r['Home']}",
                "home": r["Home"],
                "away": r["Away"],
                "spread": spread,
                "home_barthag": h_b,
                "away_barthag": a_b,
                "reason": "Away has higher BARTHAG but spread is negative (home listed as favored) — likely sign error",
            })
    # Full diagnostic table: Home, Away, Spread, XGB_Margin, Edge, Pick_Direction (favorites vs underdogs)
    print("\n--- Full diagnostic: Home | Away | Spread | XGB_Margin | Edge | Pick_Direction ---")
    print(f"{'Home':<22} | {'Away':<22} | {'Spread':>7} | {'XGB_Margin':>10} | {'Edge':>6} | Pick")
    print("-" * 95)
    for r in results:
        home = str(r.get("Home", ""))[:22]
        away = str(r.get("Away", ""))[:22]
        spread = r.get("Market_Spread")
        spread_s = f"{spread:+.1f}" if spread is not None else "—"
        xgb_m = r.get("Pred_Margin")
        xgb_s = f"{xgb_m:+.1f}" if xgb_m is not None else "—"
        edge = r.get("Edge")
        edge_s = f"{edge:+.1f}" if edge is not None else "—"
        pick = r.get("Pick_Spread", "Skip")
        print(f"{home:<22} | {away:<22} | {spread_s:>7} | {xgb_s:>10} | {edge_s:>6} | {pick}")
    print("-" * 95)
    # Favorites: spread < 0 means home favored. Model projects favorite to cover when pred_margin matches sign and |pred| > |spread|
    n_home_fav = sum(1 for r in results if r.get("Market_Spread") is not None and r["Market_Spread"] < 0)
    n_fav_cover_pick = sum(1 for r in results if (r.get("Pick_Spread") == "Home" and r.get("Market_Spread") is not None and r["Market_Spread"] < 0) or (r.get("Pick_Spread") == "Away" and r.get("Market_Spread") is not None and r["Market_Spread"] > 0))
    print(f"  (Home favored in {n_home_fav} games. Picks on favorite (Home when spread<0, Away when spread>0): {n_fav_cover_pick})")
    print()

    if spread_sign_flags:
        print("\n*** SPREAD SIGN CONVENTION CHECK — Likely CSV sign errors (fix before rerun) ***")
        print("Convention: Spread column = Home perspective. Negative = home favored, positive = away favored.")
        print("-" * 95)
        for f in spread_sign_flags:
            print(f"  {f['game']}")
            print(f"    Spread (from CSV): {f['spread']:+.1f}   Home BARTHAG: {f['home_barthag']:.3f}   Away BARTHAG: {f['away_barthag']:.3f}")
            print(f"    → {f['reason']}")
        print("-" * 95)
        print("Summary — games to fix in CSV (flip spread sign):")
        for f in spread_sign_flags:
            suggested = -f["spread"] if f["spread"] != 0 else 0
            print(f"  {f['home']} vs {f['away']}: current Spread {f['spread']:+.1f} → should be {suggested:+.1f}")
        print()
    else:
        print("\nSpread sign check: no games flagged (BARTHAG and spread sign consistent).\n")
    # Debug: games where away has better BARTHAG — show pred_margin and pick
    _away_better_home_pick_games = [
        ("Colgate", "Lehigh"),
        ("Memphis", "Tulane"),
        ("San Francisco", "Oregon St."),
    ]
    print("\nAway better (BARTHAG) — pred_margin and pick:")
    print("-" * 80)
    for r in results:
        away_s = str(r.get("Away", "")).strip()
        home_s = str(r.get("Home", "")).strip()
        for away_key, home_key in _away_better_home_pick_games:
            if away_key in away_s and home_key in home_s:
                raw = r.get("Pred_Margin")
                edge = r.get("Edge")
                pick = r.get("Pick_Spread", "")
                raw_s = f"{raw:+.2f}" if raw is not None else "—"
                edge_s = f"{edge:+.2f}" if edge is not None else "—"
                print(f"  {away_s} @ {home_s}")
                print(f"    pred_margin (home): {raw_s}   Edge: {edge_s}   Pick_Spread: {pick}")
                break
    print("-" * 80)
    print()
    for r in results:
        pm = f"{r['Pred_Margin']:+.1f}" if r.get("Pred_Margin") is not None else "—"
        bm = r.get("Blended_Margin")
        bm_str = f"{bm:+.1f}" if bm is not None else "—"
        edge_str = f"{r['Edge']:+.1f}" if r["Edge"] is not None else "—"
        vp_tag = "  *** VALUE PLAY ***" if ((r["Home"], r["Away"]) in value_plays_set) else ""
        star = " ⭐" if r.get("Rebounding_Star") else ""
        stats_na = "  (Stats N/A)" if not r.get("Used_KenPom", False) else ""
        print(f"{r['Away']} @ {r['Home']}{vp_tag}{star}{stats_na}")
        reliability_note = f"  [{r.get('Reliability_Flag')}]" if r.get("Reliability_Flag") else ""
        pick_str = r["Pick_Spread"] if r["Pick_Spread"] != "Skip" else "Skip (no pick)"
        if USE_MARKET_BLEND and bm is not None:
            print(f"  Market (home): {r['Market_Spread']:+.1f}   Pred (raw): {pm}   Blended: {bm_str}   Edge: {edge_str} pts   Confidence: {r['Confidence']}   P(cover): {r['Spread_Prob']:.2f}  → {pick_str}{reliability_note}")
        else:
            print(f"  Market (home): {r['Market_Spread']:+.1f}   Pred (home): {pm}   Edge: {edge_str} pts   Confidence: {r['Confidence']}   P(cover): {r['Spread_Prob']:.2f}  → {pick_str}{reliability_note}")
        print(f"  Total: {r['Over_Under']:.1f}  P(Over): {r['Over_Prob']:.2f}  → {r['Pick_Total']}\n")
    if value_plays:
        header = "--- Value Plays Summary (|edge| > 3 pts) ---"
        print(header)
        for r in value_plays:
            star = " ⭐" if r.get("Rebounding_Star") else ""
            print(f"  [{r['Confidence']}] {r['Away']} @ {r['Home']}: Edge {r['Edge']:+.1f} pts  → {r['Pick_Spread']}{star}")

    print("\n" + "=" * 60)
    if high_confidence:
        high_sorted = sorted(high_confidence, key=lambda x: abs(x["Edge"] or 0), reverse=True)
        top5 = high_sorted[:5]
        print("HIGH CONFIDENCE ONLY (Edge > 7.0 pts) — Top 5 by Edge")
        print("=" * 60)
        for r in top5:
            star = " ⭐ Rebounding edge" if r.get("Rebounding_Star") else ""
            stats_na = " (Stats N/A)" if not r.get("Used_KenPom", False) else ""
            print(f"  {r['Away']} @ {r['Home']}{stats_na}")
            pred_s = f"{r['Pred_Margin']:+.1f}" if r.get("Pred_Margin") is not None else "—"
            blended_s = f"{r['Blended_Margin']:+.1f}" if r.get("Blended_Margin") is not None else "—"
            if USE_MARKET_BLEND and r.get("Blended_Margin") is not None:
                print(f"    Pick: {r['Pick_Spread']}   Edge: {r['Edge']:+.1f} pts   Market (home): {r['Market_Spread']:+.1f}   Pred (raw): {pred_s}   Blended: {blended_s}{star}")
            else:
                print(f"    Pick: {r['Pick_Spread']}   Edge: {r['Edge']:+.1f} pts   Market (home): {r['Market_Spread']:+.1f}   Pred (home): {pred_s}{star}")
        if len(high_sorted) > 5:
            print(f"  ... and {len(high_sorted) - 5} more high-confidence plays (see full list above)")
        print("=" * 60)

    # Save value plays to historical performance CSV (2026 date, with new columns)
    n_saved = _save_historical_performance(results, value_plays, path, game_date)
    n_home_final = sum(1 for r in value_plays if r.get("Pick_Spread") == "Home")
    n_away_final = len(value_plays) - n_home_final
    pct_home_final = (100.0 * n_home_final / len(value_plays)) if value_plays else 0
    print(f"\nSUCCESS: [{n_saved}] picks saved to historical_betting_performance.csv for {game_date}.")
    print(f"FINAL HOME/AWAY RATIO: {n_home_final} Home / {n_away_final} Away  ({pct_home_final:.0f}% Home).")

    # Totals value plays: |Totals_Edge| > 3
    totals_plays = sorted(
        [r for r in results if r.get("Pick_Total_Value") in ("Over", "Under")],
        key=lambda x: abs(x.get("Totals_Edge") or 0),
        reverse=True,
    )
    # Write value plays (spread + totals) to dashboard cache
    _write_value_plays_cache(value_plays, totals_plays, path, game_date)


def _write_value_plays_cache(
    value_plays: list[dict],
    totals_plays: list[dict],
    odds_path: Path,
    game_date: str,
) -> None:
    """Write value plays (spread and totals) to data/cache/value_plays_cache.json.
    Always overwrites the entire file with this run's plays only (never appends)."""
    cache_dir = APP_ROOT / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "value_plays_cache.json"

    value_plays_list = []
    for r in value_plays:
        home = str(r.get("Home", "")).strip()
        away = str(r.get("Away", "")).strip()
        pick = str(r.get("Pick_Spread", "Home")).strip()
        selection = home if pick == "Home" else away
        edge_pts = r.get("Edge")
        try:
            edge_pts_f = float(edge_pts) if edge_pts is not None else 0.0
        except (TypeError, ValueError):
            edge_pts_f = 0.0
        # Value % = edge magnitude for all picks (Away edges are negative; use abs so Away shows non-zero)
        value_pct = min(15.0, abs(edge_pts_f) / 3.0)
        market_spread = r.get("Market_Spread")
        try:
            spread_f = float(market_spread) if market_spread is not None else None
        except (TypeError, ValueError):
            spread_f = None
        # Spread from picked team's perspective: Home = market (home line), Away = -market (away line)
        point_picked = spread_f if pick == "Home" else (-spread_f if spread_f is not None else None)
        point_str = f"{point_picked:+.1f}" if point_picked is not None else "—"
        # Human-readable reasoning: use margin that drove the pick (blended when USE_MARKET_BLEND)
        pred_margin = r.get("Blended_Margin") if (r.get("Blended_Margin") is not None) else r.get("Pred_Margin")
        try:
            pred_f = float(pred_margin) if pred_margin is not None else None
        except (TypeError, ValueError):
            pred_f = None
        p_cover = r.get("Spread_Prob")
        p_cover_pct_str = f"{p_cover * 100:.1f}%" if isinstance(p_cover, (int, float)) else None
        if pred_f is not None and spread_f is not None:
            abs_spread = abs(spread_f)
            pick_home = pick == "Home"
            if pred_f >= 0:
                proj = f"Model projects {home} to win by ~{int(round(pred_f))} pts."
            else:
                proj = f"Model projects {away} to win by ~{int(round(abs(pred_f)))} pts."
            home_favored = spread_f < 0
            if pick_home and home_favored:
                take = f"take {selection} -{abs_spread:.1f}"
            elif pick_home and not home_favored:
                take = f"take {selection} +{abs_spread:.1f}"
            elif not pick_home and home_favored:
                take = f"take {selection} +{abs_spread:.1f}"
            else:
                take = f"take {selection} -{abs_spread:.1f}"
            reason = f"{proj} Market needs {abs_spread:.1f} — {take}."
            if p_cover_pct_str:
                reason += f" P(cover): {p_cover_pct_str}."
        else:
            edge_str = f"{edge_pts_f:+.1f}" if edge_pts_f is not None else "—"
            if point_str != "—":
                reason = f"Model favors {selection} ({point_str}) — Edge {edge_str} pts."
            else:
                reason = f"Model favors {selection}."
            if p_cover_pct_str:
                reason += f" P(cover): {p_cover_pct_str}."
        start_time_iso = r.get("start_time")
        if start_time_iso and str(start_time_iso).strip():
            try:
                s = str(start_time_iso).replace("Z", "+00:00")
                dt = datetime.fromisoformat(s)
                if dt.tzinfo:
                    dt_et = dt.astimezone(ZoneInfo("America/New_York"))
                    start_time_display = dt_et.strftime("%b %d, %I:%M %p ET")
                else:
                    start_time_display = dt.strftime("%b %d, %I:%M %p")
            except (ValueError, TypeError):
                start_time_display = "Today"
        else:
            start_time_display = "Today"
        suggested_pct = r.get("Suggested_Stake_Pct")
        try:
            suggested_pct = float(suggested_pct) if suggested_pct is not None else None
        except (TypeError, ValueError):
            suggested_pct = None
        value_plays_list.append({
            "League": "NCAAB",
            "Event": f"{away} @ {home}",
            "Selection": selection,
            "Market": "Spread",
            "Odds": -110,
            "Value (%)": round(value_pct, 2),
            "point": point_picked,
            "Point": point_picked,
            "Edge_pts": round(edge_pts_f, 1) if edge_pts_f is not None else None,
            "Spread_Prob": round(float(p_cover), 3) if isinstance(p_cover, (int, float)) else None,
            "Recommended Stake": None,
            "Suggested_Stake_Pct": round(suggested_pct, 2) if suggested_pct is not None else None,
            "Kelly_Fraction": round(r.get("Kelly_Fraction", 0), 6) if r.get("Kelly_Fraction") is not None else None,
            "Injury Alert": "—",
            "Start Time": start_time_display,
            "start_time": start_time_iso,
            "home_team": home,
            "away_team": away,
            "confidence_tier": str(r.get("Confidence", "Medium")).strip() or "Medium",
            "reason": reason,
            "reasoning_summary": reason,
        })

    totals_plays_list = []
    if SHOW_TOTALS_PICKS:
        for r in totals_plays:
            home = str(r.get("Home", "")).strip()
            away = str(r.get("Away", "")).strip()
            pick_tot = str(r.get("Pick_Total_Value", "")).strip()  # "Over" or "Under"
            if pick_tot not in ("Over", "Under"):
                continue
            edge_t = r.get("Totals_Edge")
            try:
                edge_t_f = float(edge_t) if edge_t is not None else 0.0
            except (TypeError, ValueError):
                edge_t_f = 0.0
            value_pct_t = min(15.0, abs(edge_t_f) / 3.0)
            over_under = r.get("Over_Under")
            try:
                ou_f = float(over_under) if over_under is not None else None
            except (TypeError, ValueError):
                ou_f = None
            pred_total = r.get("Predicted_Total")
            try:
                pred_f = float(pred_total) if pred_total is not None else None
            except (TypeError, ValueError):
                pred_f = None
            reason_t = f"Model projects total {pred_f:.1f} pts (line {ou_f:.1f}) — edge {edge_t_f:+.1f} pts. Take {pick_tot}."
            start_time_iso = r.get("start_time")
            if start_time_iso and str(start_time_iso).strip():
                try:
                    s = str(start_time_iso).replace("Z", "+00:00")
                    dt = datetime.fromisoformat(s)
                    if dt.tzinfo:
                        dt_et = dt.astimezone(ZoneInfo("America/New_York"))
                        start_time_display = dt_et.strftime("%b %d, %I:%M %p ET")
                    else:
                        start_time_display = dt.strftime("%b %d, %I:%M %p")
                except (ValueError, TypeError):
                    start_time_display = "Today"
            else:
                start_time_display = "Today"
            totals_plays_list.append({
                "League": "NCAAB Totals",
                "Event": f"{away} @ {home}",
                "Selection": pick_tot,
                "Market": "Total",
                "Odds": -110,
                "Value (%)": round(value_pct_t, 2),
                "point": ou_f,
                "Point": ou_f,
                "Over_Under": ou_f,
                "Predicted_Total": pred_f,
                "Totals_Edge": round(edge_t_f, 2),
                "Injury Alert": "—",
                "Start Time": start_time_display,
                "start_time": start_time_iso,
                "home_team": home,
                "away_team": away,
                "confidence_tier": "Medium",
                "reason": reason_t,
                "reasoning_summary": reason_t,
            })

    # POTD: top 2 spread by abs(Edge); top 2 totals by abs(Totals_Edge) when SHOW_TOTALS_PICKS
    potd_picks = {"NCAAB Pick 1": None, "NCAAB Pick 2": None, "NCAAB Totals Pick 1": None, "NCAAB Totals Pick 2": None}
    for i, vp in enumerate(value_plays_list[:2]):
        pick = vp.copy()
        pick["League"] = f"NCAAB Pick {i + 1}"
        potd_picks[f"NCAAB Pick {i + 1}"] = pick
    if SHOW_TOTALS_PICKS:
        for i, tp in enumerate(totals_plays_list[:2]):
            pick = tp.copy()
            pick["League"] = f"NCAAB Totals Pick {i + 1}"
            potd_picks[f"NCAAB Totals Pick {i + 1}"] = pick

    payload = {
        "value_plays": value_plays_list,
        "totals_plays": totals_plays_list,
        "potd_picks": potd_picks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "odds_source_meta": {"source": "manual_odds", "odds_file": odds_path.name},
        "value_plays_flagged_count": 0,
    }
    with open(cache_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {len(value_plays_list)} spread + {len(totals_plays_list)} totals plays to {cache_path.relative_to(APP_ROOT)} (cache overwritten with this run only).")


def _save_historical_performance(
    results: list[dict],
    value_plays: list[dict],
    odds_path: Path,
    game_date: str,
) -> int:
    """Append value plays to data/historical_betting_performance.csv with Confidence_Level, Edge_Points, Has_Rebound_Advantage. Returns count saved."""
    hist_path = APP_ROOT / "data" / "historical_betting_performance.csv"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "Date", "Odds_File", "Home", "Away", "Home_BARTHAG", "Away_BARTHAG", "Market_Spread", "Pred_Margin", "Pick_Spread",
        "Confidence_Level", "Edge_Points", "Has_Rebound_Advantage", "Spread_Prob", "Over_Under", "Pick_Total",
        "Kelly_Fraction", "Suggested_Stake_Pct",
    ]
    # Use 2026 in Odds_File display name to match stats year (e.g. March_07_2026_Odds.csv)
    try:
        _dt = datetime.strptime(game_date, "%Y-%m-%d")
        odds_file_display = f"{_dt.strftime('%B_%d')}_{ODDS_YEAR}_Odds.csv"
    except Exception:
        odds_file_display = odds_path.name
    rows = []
    for r in value_plays:
        rows.append({
            "Date": game_date,
            "Odds_File": odds_file_display,
            "Home": r["Home"],
            "Away": r["Away"],
            "Home_BARTHAG": r.get("Home_BARTHAG"),
            "Away_BARTHAG": r.get("Away_BARTHAG"),
            "Market_Spread": r["Market_Spread"],
            "Pred_Margin": r["Pred_Margin"],
            "Pick_Spread": r["Pick_Spread"],
            "Confidence_Level": r["Confidence"],
            "Edge_Points": r["Edge"],
            "Has_Rebound_Advantage": bool(r.get("Rebounding_Star", False)),
            "Spread_Prob": r["Spread_Prob"],
            "Over_Under": r["Over_Under"],
            "Pick_Total": r["Pick_Total"],
            "Kelly_Fraction": r.get("Kelly_Fraction"),
            "Suggested_Stake_Pct": r.get("Suggested_Stake_Pct"),
        })
    if not rows:
        return 0
    df_new = pd.DataFrame(rows)
    if hist_path.exists():
        existing = pd.read_csv(hist_path)
        for c in columns:
            if c not in existing.columns:
                existing[c] = None
        # Keep only one set of plays per date: drop any existing rows with today's date before appending
        if "Date" in existing.columns:
            existing["Date"] = existing["Date"].astype(str).str.strip()
            existing = existing[existing["Date"] != game_date]
        df_new = df_new[[c for c in columns if c in df_new.columns]]
        combined = pd.concat([existing, df_new], ignore_index=True)
    else:
        combined = df_new
    combined.to_csv(hist_path, index=False)
    return len(rows)


if __name__ == "__main__":
    main()
