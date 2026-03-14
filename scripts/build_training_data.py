#!/usr/bin/env python3
"""
Build NCAAB training dataset from historical games + Torvik team stats.

Loads historical_games.csv, merges team stats by season, computes home_hca and
recent form (last 5/10 margin, win%). Writes training_data.csv and HCA lookups.
"""
from __future__ import annotations

import io
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

from engine.utils import game_season_from_date

REQUEST_TIMEOUT = 30
SEASONS = (2021, 2022, 2023, 2024, 2025)
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ncaab"
HISTORICAL_GAMES_PATH = DATA_DIR / "historical_games.csv"
TRAINING_DATA_PATH = DATA_DIR / "training_data.csv"
TEAM_HCA_PATH = DATA_DIR / "team_hca_by_season.csv"

COLUMN_RENAME = {
    "team": "TEAM", "conf": "CONF", "adjoe": "ADJOE", "adjde": "ADJDE",
    "barthag": "BARTHAG", "adjt": "ADJ_T", "rank": "RANK", "record": "RECORD",
    "oe Rank": "OE_RANK", "de Rank": "DE_RANK", "rank.1": "BARTHAG_RANK",
    "proj. W": "Proj_W", "Proj. L": "Proj_L", "WAB": "WAB", "WAB Rk": "WAB_Rk", "Fun Rk": "Fun_Rk",
}
EXTRA_RENAME = {"efg_o": "EFG_O", "efg_d": "EFG_D", "tor": "TOR", "tord": "TORD", "orb": "ORB",
    "drb": "DRB", "ftr": "FTR", "ftrd": "FTRD", "3pr": "THREE_PT_RATE", "3prd": "3PRD"}
COLUMN_RENAME = {**COLUMN_RENAME, **EXTRA_RENAME}

STAT_COLS = [
    "ADJOE", "ADJDE", "BARTHAG", "ADJ_T", "EFG_O", "EFG_D",
    "TOR", "ORB", "FTR", "THREE_PT_RATE",
]
ALIAS_3PR = "3PR"

FORM_COLS = [
    "home_last5_margin", "home_last10_margin", "home_last5_winpct",
    "away_last5_margin", "away_last10_margin", "away_last5_winpct",
]
FORM_DIFF_COLS = ["last5_margin_diff", "last10_margin_diff", "last5_winpct_diff"]


def _sanitize_header(name: str) -> str:
    s = str(name).strip()
    s = re.sub(r"[%\s.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else name


def fetch_team_stats_year(year: int) -> pd.DataFrame:
    url = f"https://barttorvik.com/{year}_team_results.csv"
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch {url}: {e}") from e
    df = pd.read_csv(io.StringIO(r.text))
    if df.empty:
        return df
    rename = {}
    for c in df.columns:
        key = c.strip().lower().replace(" ", "_").replace(".", "")
        if c in COLUMN_RENAME:
            rename[c] = COLUMN_RENAME[c]
        elif key in {k.lower().replace(" ", "_"): k for k in EXTRA_RENAME}:
            for orig, canon in EXTRA_RENAME.items():
                if key == orig.lower().replace(" ", "_"):
                    rename[c] = canon
                    break
        else:
            rename[c] = _sanitize_header(c)
    df = df.rename(columns=rename)
    if "RECORD" in df.columns:
        g_list, w_list = [], []
        for rec in df["RECORD"].astype(str):
            try:
                parts = rec.split("-")
                if len(parts) >= 2:
                    w = int(parts[0].strip())
                    l = int(parts[1].strip())
                    w_list.append(w)
                    g_list.append(w + l)
                else:
                    w_list.append(0)
                    g_list.append(0)
            except (ValueError, AttributeError):
                w_list.append(0)
                g_list.append(0)
        df["G"] = g_list
        df["W"] = w_list
    else:
        df["G"] = 0
        df["W"] = 0
    df["season"] = year
    return df


def ensure_team_stats_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for year in SEASONS:
        path = DATA_DIR / f"team_stats_{year}.csv"
        if path.exists():
            print(f"  Using existing {path.name}")
            continue
        print(f"  Fetching {year} team results from Barttorvik...")
        df = fetch_team_stats_year(year)
        if not df.empty:
            df.to_csv(path, index=False)
            print(f"  Saved {path.name} ({len(df)} teams)")


def _normalize_team_name(s: str) -> str:
    if pd.isna(s):
        return ""
    return " ".join(str(s).strip().lower().split())


def load_team_stats_by_season() -> dict[int, pd.DataFrame]:
    out = {}
    for year in list(SEASONS) + [2026]:
        path = DATA_DIR / f"team_stats_{year}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "TEAM" not in df.columns:
            continue
        df["_team_norm"] = df["TEAM"].astype(str).map(_normalize_team_name)
        out[year] = df
    return out


def lookup_team_stats(stats_df: pd.DataFrame, team_name: str) -> pd.Series | None:
    if stats_df.empty or not team_name:
        return None
    norm = _normalize_team_name(team_name)
    if not norm:
        return None
    match = stats_df[stats_df["_team_norm"] == norm]
    return match.iloc[0] if not match.empty else None


def _is_neutral(g: pd.Series) -> bool:
    ns = g.get("neutral_site")
    if pd.isna(ns) or ns is None:
        return False
    return str(ns).strip().lower() in ("1", "true", "y", "yes")


def compute_team_hca(games: pd.DataFrame) -> tuple[dict, dict, dict]:
    team_home_margins = defaultdict(list)
    team_all_margins = defaultdict(list)
    for _, g in games.iterrows():
        season = int(g["season"])
        home = str(g.get("home_team", "")).strip()
        away = str(g.get("away_team", "")).strip()
        margin = float(g.get("margin", 0))
        neutral = _is_neutral(g)
        if not home or not away:
            continue
        team_all_margins[(season, home)].append(margin)
        team_all_margins[(season, away)].append(-margin)
        if not neutral:
            team_home_margins[(season, home)].append(margin)
    team_hca = {}
    for (season, team), home_margins in team_home_margins.items():
        if not home_margins:
            continue
        all_margins = team_all_margins.get((season, team), [])
        if not all_margins:
            continue
        avg_home = sum(home_margins) / len(home_margins)
        avg_all = sum(all_margins) / len(all_margins)
        team_hca[(season, team)] = avg_home - avg_all
    team_conf = {}
    stats_by_season = load_team_stats_by_season()
    for (season, team) in team_hca:
        if season not in stats_by_season:
            continue
        row = lookup_team_stats(stats_by_season[season], team)
        if row is not None and "CONF" in row.index:
            team_conf[(season, team)] = str(row["CONF"]).strip() or ""
    by_conf = defaultdict(list)
    for (season, team), hca in team_hca.items():
        conf = team_conf.get((season, team), "")
        if conf:
            by_conf[(season, conf)].append(hca)
    conf_hca = {k: sum(v) / len(v) for k, v in by_conf.items() if v}
    return team_hca, conf_hca, team_conf


def compute_form_features(games: pd.DataFrame) -> pd.DataFrame:
    """For each game, compute last 5 and last 10 avg margin and win% for home and away (prior to game date)."""
    g = games.sort_values("date").copy()
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    g = g.dropna(subset=["date"])
    team_hist: dict[str, list[tuple[pd.Timestamp, float, float]]] = defaultdict(list)

    rows = []
    for _, row in g.iterrows():
        dt = row["date"]
        home = str(row.get("home_team", "")).strip()
        away = str(row.get("away_team", "")).strip()
        margin = float(row.get("margin", 0))
        if not home or not away:
            rows.append({
                "date": row["date"], "home_team": home, "away_team": away,
                **{c: None for c in FORM_COLS + FORM_DIFF_COLS},
            })
            continue

        def form(hist: list, n: int) -> tuple[float | None, float | None]:
            if not hist:
                return None, None
            take = hist[-n:] if len(hist) >= n else hist
            margins = [x[1] for x in take]
            wins = [x[2] for x in take]
            avg_m = sum(margins) / len(margins) if margins else None
            winpct = sum(wins) / len(wins) if wins else None
            return avg_m, winpct

        h_hist = team_hist[home]
        a_hist = team_hist[away]
        h5_m, h5_w = form(h_hist, 5)
        h10_m, h10_w = form(h_hist, 10)
        a5_m, a5_w = form(a_hist, 5)
        a10_m, a10_w = form(a_hist, 10)

        rows.append({
            "date": row["date"],
            "home_team": home,
            "away_team": away,
            "home_last5_margin": h5_m,
            "home_last10_margin": h10_m,
            "home_last5_winpct": h5_w,
            "away_last5_margin": a5_m,
            "away_last10_margin": a10_m,
            "away_last5_winpct": a5_w,
        })
        team_hist[home].append((dt, margin, 1.0 if margin > 0 else 0.0))
        team_hist[away].append((dt, -margin, 1.0 if margin < 0 else 0.0))

    form_df = pd.DataFrame(rows)
    form_df["last5_margin_diff"] = form_df["home_last5_margin"] - form_df["away_last5_margin"]
    form_df["last10_margin_diff"] = form_df["home_last10_margin"] - form_df["away_last10_margin"]
    form_df["last5_winpct_diff"] = form_df["home_last5_winpct"] - form_df["away_last5_winpct"]
    return form_df


def get_stat_value(row: pd.Series | None, col: str, alias: str | None = None) -> float | None:
    if row is None:
        return None
    if col in row.index and pd.notna(row[col]):
        try:
            return float(row[col])
        except (TypeError, ValueError):
            pass
    if alias and alias in row.index and pd.notna(row[alias]):
        try:
            return float(row[alias])
        except (TypeError, ValueError):
            pass
    return None


def build_training_data() -> pd.DataFrame:
    if not HISTORICAL_GAMES_PATH.exists():
        raise FileNotFoundError(f"Historical games not found: {HISTORICAL_GAMES_PATH}")
    stats_by_season = load_team_stats_by_season()
    if not stats_by_season:
        raise FileNotFoundError("No team_stats_{year}.csv files found in data/ncaab/")

    games = pd.read_csv(HISTORICAL_GAMES_PATH)
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    games = games.dropna(subset=["date"])
    games["season"] = games["date"].dt.strftime("%Y-%m-%d").apply(game_season_from_date)
    games = games[games["season"].notna()].copy()
    games["season"] = games["season"].astype(int)

    # Sort once so we can compute days rest chronologically
    games = games.sort_values("date").reset_index(drop=True)

    form_df = compute_form_features(games.copy())
    form_df["date"] = pd.to_datetime(form_df["date"]).dt.strftime("%Y-%m-%d")

    team_hca, conf_hca, team_conf = compute_team_hca(games)

    # Track last game date per team for days-rest features
    last_game_date: dict[str, pd.Timestamp] = {}

    def get_home_hca(season: int, home_team: str, neutral: bool) -> float:
        if neutral:
            return 0.0
        key = (season, str(home_team).strip())
        v = team_hca.get(key)
        if v is not None:
            return float(v)
        conf = team_conf.get(key, "")
        if conf:
            v = conf_hca.get((season, conf))
            if v is not None:
                return float(v)
        return 0.0

    rows = []
    for _, g in games.iterrows():
        season = int(g["season"])
        if season not in stats_by_season:
            continue
        stats_df = stats_by_season[season]
        home_row = lookup_team_stats(stats_df, str(g.get("home_team", "")))
        away_row = lookup_team_stats(stats_df, str(g.get("away_team", "")))
        if home_row is None or away_row is None:
            continue
        g_date = g["date"]
        month = g_date.month
        day = g_date.day
        # Conference / NCAA tournament windows by calendar date
        is_conference_tourney = 1 if (month == 3 and 1 <= day <= 15) else 0
        is_ncaa_tourney = 1 if ((month == 3 and day >= 16) or (month == 4 and day <= 10)) else 0

        def _days_rest(team: str) -> int:
            name = str(team).strip()
            if not name:
                return 7
            prev = last_game_date.get(name)
            if prev is None:
                return 7
            delta = (g_date - prev).days
            if delta < 0:
                return 7
            return min(14, max(0, delta))

        home_team_name = str(g.get("home_team", "")).strip()
        away_team_name = str(g.get("away_team", "")).strip()
        days_rest_home = _days_rest(home_team_name)
        days_rest_away = _days_rest(away_team_name)

        # Update tracker for subsequent games
        if home_team_name:
            last_game_date[home_team_name] = g_date
        if away_team_name:
            last_game_date[away_team_name] = g_date
        feat = {}
        for col in STAT_COLS:
            alias = ALIAS_3PR if col == "THREE_PT_RATE" else None
            feat[f"home_{col}"] = get_stat_value(home_row, col, alias)
            feat[f"away_{col}"] = get_stat_value(away_row, col, alias)
        feat["BARTHAG_diff"] = (feat.get("home_BARTHAG") or 0) - (feat.get("away_BARTHAG") or 0)
        feat["ADJOE_diff"] = (feat.get("home_ADJOE") or 0) - (feat.get("away_ADJOE") or 0)
        feat["ADJDE_diff"] = (feat.get("home_ADJDE") or 0) - (feat.get("away_ADJDE") or 0)
        feat["tempo_diff"] = (feat.get("home_ADJ_T") or 0) - (feat.get("away_ADJ_T") or 0)
        neutral = _is_neutral(g)
        feat["is_neutral"] = 1 if neutral else 0
        feat["home_hca"] = get_home_hca(season, str(g.get("home_team", "")), neutral)
        # Tournament flags
        feat["is_conference_tourney"] = is_conference_tourney
        feat["is_ncaa_tourney"] = is_ncaa_tourney
        # Seed diff: away_seed - home_seed (positive => home higher/better seed). 0 if missing.
        home_seed = home_row.get("SEED") if "SEED" in home_row.index else None
        away_seed = away_row.get("SEED") if "SEED" in away_row.index else None
        try:
            h_seed_val = float(home_seed) if home_seed not in (None, 0, "0") and pd.notna(home_seed) else None
        except (TypeError, ValueError):
            h_seed_val = None
        try:
            a_seed_val = float(away_seed) if away_seed not in (None, 0, "0") and pd.notna(away_seed) else None
        except (TypeError, ValueError):
            a_seed_val = None
        if h_seed_val is not None and a_seed_val is not None:
            feat["seed_diff"] = a_seed_val - h_seed_val
        else:
            feat["seed_diff"] = 0.0
        feat["days_rest_home"] = float(days_rest_home)
        feat["days_rest_away"] = float(days_rest_away)

        feat["actual_margin"] = int(g["home_score"]) - int(g["away_score"])
        feat["actual_total"] = int(g["home_score"]) + int(g["away_score"])
        feat["date"] = g["date"].strftime("%Y-%m-%d") if hasattr(g["date"], "strftime") else str(g["date"])
        feat["home_team"] = home_team_name
        feat["away_team"] = away_team_name
        feat["season"] = season
        rows.append(feat)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Merge form features
    merge_keys = ["date", "home_team", "away_team"]
    form_cols = FORM_COLS + FORM_DIFF_COLS
    df = df.merge(form_df[merge_keys + form_cols], on=merge_keys, how="left")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    hca_rows = [{"season": s, "team": t, "home_hca": hca} for (s, t), hca in team_hca.items()]
    if hca_rows:
        pd.DataFrame(hca_rows).to_csv(TEAM_HCA_PATH, index=False)
    conf_rows = [{"season": s, "conf": c, "home_hca": hca} for (s, c), hca in conf_hca.items()]
    if conf_rows:
        pd.DataFrame(conf_rows).to_csv(DATA_DIR / "conf_hca_by_season.csv", index=False)
    conf_lookup_rows = []
    for season, stats_df in stats_by_season.items():
        if "TEAM" not in stats_df.columns or "CONF" not in stats_df.columns:
            continue
        for _, r in stats_df.iterrows():
            t = str(r.get("TEAM", "")).strip()
            c = str(r.get("CONF", "")).strip()
            if t:
                conf_lookup_rows.append({"season": season, "team": t, "conf": c})
    if conf_lookup_rows:
        pd.DataFrame(conf_lookup_rows).drop_duplicates(subset=["season", "team"], keep="first").to_csv(
            DATA_DIR / "team_conf_by_season.csv", index=False
        )

    id_cols = ["date", "season", "home_team", "away_team"]
    home_cols = [f"home_{c}" for c in STAT_COLS]
    away_cols = [f"away_{c}" for c in STAT_COLS]
    diff_cols = ["BARTHAG_diff", "ADJOE_diff", "ADJDE_diff", "tempo_diff"]
    extra_cols = ["home_hca", "is_neutral", "is_conference_tourney", "is_ncaa_tourney", "seed_diff", "days_rest_home", "days_rest_away"]
    order = id_cols + home_cols + away_cols + diff_cols + extra_cols
    order += FORM_COLS + FORM_DIFF_COLS
    order += ["actual_margin", "actual_total"]
    # Flag that current version uses full-season team stats (not point-in-time)
    df["uses_full_season_stats"] = True
    order = [c for c in order if c in df.columns]
    return df[order]


def main() -> None:
    print("Building NCAAB training data...")
    print("(1) Ensuring team stats for 2021-2025...")
    ensure_team_stats_files()
    print("(2) Loading historical games, form features, and matching team stats...")
    df = build_training_data()
    if df.empty:
        print("No training rows produced.")
        return
    TRAINING_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TRAINING_DATA_PATH, index=False)
    print(f"Saved {len(df)} rows to {TRAINING_DATA_PATH}")
    feats = [c for c in df.columns if c not in ("date", "season", "home_team", "away_team", "actual_margin", "actual_total")]
    print(f"  Features ({len(feats)}): {feats}")
    print(f"  Target: actual_margin (mean={df['actual_margin'].mean():.2f}), actual_total (mean={df['actual_total'].mean():.1f})")


if __name__ == "__main__":
    main()
