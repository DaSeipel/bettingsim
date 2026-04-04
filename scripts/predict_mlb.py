#!/usr/bin/env python3
"""
Generate data/cache/mlb_value_plays.json from data/odds/live_mlb_odds.json,
data/mlb/team_stats.csv (scripts/fetch_mlb_stats.py), and optionally
data/mlb/pitcher_stats.csv (scripts/fetch_mlb_pitchers.py).

Requires fetch_mlb_odds.py to have run so live_mlb_odds.json exists.
Best results: run fetch_mlb_pitchers.py after odds so starter FIP/K9/BB9/innings_pitched are available.
Low-IP starters shrink pitcher weight toward team stats (see PITCH_MATCHUP_WEIGHT / _pitcher_confidence).

card_date: prefers games_date_et from live odds (MLB calendar day in ET); else America/New_York today.
"""

from __future__ import annotations

import json
import math
import os
import sys
import warnings

# Set MLB_DEBUG_TEAM_QUALITY=1 to print per-team inputs inside _team_quality_for_model().
_MLB_DEBUG_TEAM_QUALITY = os.environ.get("MLB_DEBUG_TEAM_QUALITY", "").strip().lower() in (
    "1",
    "true",
    "yes",
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*urllib3 v2 only supports OpenSSL.*")
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ODDS_PATH = ROOT / "data" / "odds" / "live_mlb_odds.json"
OUT_PATH = ROOT / "data" / "cache" / "mlb_value_plays.json"
PITCHER_STATS_PATH = ROOT / "data" / "mlb" / "pitcher_stats.csv"
PARK_FACTORS_PATH = ROOT / "data" / "mlb" / "mlb_park_factors.json"

# Edge and probability filters (decimal edge = EV fraction, e.g. 0.03 = 3%).
MIN_EDGE_DECIMAL = 0.03  # minimum 3% edge to qualify as a play
MAX_EDGE_DECIMAL = 0.15  # skip plays above this; prints FLAGGED_HIGH_EDGE
MIN_MODEL_PROB = 0.42  # skip picks below this win probability; prints SKIP_LOW_PROB

# Pull model win prob toward two-way fair implied (after logistic + clamp, before edge).
PROB_SHRINK_TOWARD_MARKET = 0.25

# When a starter is missing from pitcher_stats.csv, use league-average rate stats and
# IP=50 → confidence 0.50 → ~30/70 pitcher/team split (not 0/100).
MISSING_PITCHER_STATS_ROW: dict[str, float] = {
    "fip": 4.25,
    "k9": 8.8,
    "bb9": 3.2,
    "innings_pitched": 50.0,
}

# Max pitcher share when both starters have 100+ IP (confidence 1.0). Below that,
# pitcher_weight = PITCH_MATCHUP_WEIGHT * min(home_conf, away_conf), team_weight = 1 - pitcher_weight.
PITCH_MATCHUP_WEIGHT = 0.6
# Softer than 0.58 so wide diff_team (+ larger RD cap) does not pin every mismatch at 0.30/0.70.
LOGISTIC_K = 0.36
HOME_WIN_PROB_MIN = 0.30
HOME_WIN_PROB_MAX = 0.70
DIFF_TEAM_CLAMP = 2.0
RUN_DIFFERENTIAL_CAP = 500.0

# Always emit _team_quality_for_model-style stderr lines for these slates (away, home).
_TEAM_QUALITY_MATCHUP_DEBUG: frozenset[tuple[str, str]] = frozenset(
    {
        ("Colorado Rockies", "Toronto Blue Jays"),
        ("Cleveland Guardians", "Los Angeles Dodgers"),
    }
)

def _load_park_factors() -> dict[str, float]:
    """
    Load mlb_park_factors.json → {team_name: runs_factor_float}.
    Returns runs / 100.0 so 1.0 = neutral; missing teams default to 1.0.
    """
    if not PARK_FACTORS_PATH.exists():
        print(
            f"MLB predict: park factors not found at {PARK_FACTORS_PATH} — using neutral 1.0 for all teams.",
            file=sys.stderr,
            flush=True,
        )
        return {}
    try:
        with open(PARK_FACTORS_PATH, encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception as exc:
        print(f"MLB predict: could not read park factors: {exc} — using neutral 1.0.", file=sys.stderr)
        return {}
    out: dict[str, float] = {}
    for team, data in raw.items():
        if team.startswith("_"):
            continue
        try:
            out[team] = float(data["runs"]) / 100.0
        except (KeyError, TypeError, ValueError):
            out[team] = 1.0
    return out


PARK_FACTORS: dict[str, float] = _load_park_factors()


def _park_mult(home_team_raw: str) -> float:
    """
    Return the runs park factor multiplier for the home team's park.
    Falls back to any SCHEDULE_NAME_TO_STATS_NAME alias, then to 1.0.
    """
    v = PARK_FACTORS.get(home_team_raw)
    if v is not None:
        return v
    full = SCHEDULE_NAME_TO_STATS_NAME.get(home_team_raw)
    if full:
        v = PARK_FACTORS.get(full)
        if v is not None:
            return v
    return 1.0


# Schedule/API sometimes uses a short nickname; CSV uses full MLB Stats API team_name.
SCHEDULE_NAME_TO_STATS_NAME: dict[str, str] = {
    "Athletics": "Oakland Athletics",
    "Angels": "Los Angeles Angels",
    "Dodgers": "Los Angeles Dodgers",
    "Cubs": "Chicago Cubs",
    "White Sox": "Chicago White Sox",
    "Mets": "New York Mets",
    "Yankees": "New York Yankees",
    "Red Sox": "Boston Red Sox",
    "Blue Jays": "Toronto Blue Jays",
    "Guardians": "Cleveland Guardians",
    "Rays": "Tampa Bay Rays",
}


def _card_date_iso(blob: dict) -> str:
    gde = str(blob.get("games_date_et") or "").strip()
    if gde:
        return gde
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def _american_to_float(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v > 0 and v < 100:
        return None
    if v < 0 and v > -100:
        return None
    return v


def _find_team_row(stats_df, raw_name: str):
    """Return one stats row for this team name, or None."""
    from engine.mlb_engine import normalize_mlb_team_name_for_join

    if stats_df is None or stats_df.empty or "team_name" not in stats_df.columns:
        return None
    norm = normalize_mlb_team_name_for_join(raw_name)
    if not norm:
        return None
    full = SCHEDULE_NAME_TO_STATS_NAME.get(norm, norm)
    m = stats_df[stats_df["team_name"].astype(str).str.strip() == full]
    if len(m) == 1:
        return m.iloc[0]
    m = stats_df[stats_df["team_name"].astype(str).str.strip() == norm]
    if len(m) == 1:
        return m.iloc[0]
    suf = stats_df[stats_df["team_name"].astype(str).str.endswith(" " + norm, na=False)]
    if len(suf) == 1:
        return suf.iloc[0]
    return None


def _team_strength(row) -> float:
    """Higher = better team (rough heuristic from era / woba)."""
    try:
        era = float(row.get("era"))
    except (TypeError, ValueError):
        era = float("nan")
    try:
        woba = float(row.get("woba"))
    except (TypeError, ValueError):
        woba = float("nan")
    if era != era:  # NaN
        era = 4.2
    if woba != woba:
        woba = 0.305
    era = min(max(era, 2.8), 6.5)
    woba = min(max(woba, 0.26), 0.38)
    return (4.2 - era) * 0.28 + (woba - 0.305) * 42.0


def _team_strength_with_bullpen(row) -> float:
    """Team strength including bullpen ERA (lower bullpen ERA = better)."""
    base = _team_strength(row)
    try:
        bpen = float(row.get("bullpen_era"))
    except (TypeError, ValueError):
        bpen = float("nan")
    if bpen != bpen:
        bpen = 4.2
    bpen = min(max(bpen, 2.5), 6.0)
    pen_adj = (4.2 - bpen) * 0.12
    return base + pen_adj


def _team_quality_components(row):
    """
    Same math as _team_quality_for_model; returns values used for debug lines.
    (final_quality, win_pct_used, run_diff_used, core_strength, mult, team_name)
    """
    core = _team_strength_with_bullpen(row)
    try:
        wp = float(row.get("win_pct"))
    except (TypeError, ValueError):
        wp = float("nan")
    if wp != wp:
        wp = 0.5
    wp = min(max(wp, 0.265), 0.70)
    # Center at wp=0.5 → 1.0×; spread vs old (0.70+0.60·wp) is doubled (~0.72–1.24 at clamp).
    mult = 1.0 + 1.2 * (wp - 0.5)

    try:
        rd = float(row.get("run_differential"))
    except (TypeError, ValueError):
        rd = float("nan")
    if rd != rd:
        rd = 0.0
    rd = max(-RUN_DIFFERENTIAL_CAP, min(RUN_DIFFERENTIAL_CAP, rd))
    rd_bump = rd / 750.0

    final_quality = (core + rd_bump) * mult
    tname = row.get("team_name") if row is not None else None
    return final_quality, wp, rd, core, mult, tname


def _team_quality_for_model(row) -> float:
    """
    Team signal for win prob: core run prevention/offense + bullpen, then run-differential
    bump, scaled by win percentage (bad teams damped, good teams boosted).
    """
    final_quality, wp, rd, core, mult, tname = _team_quality_components(row)
    if _MLB_DEBUG_TEAM_QUALITY:
        print(
            "[MLB debug _team_quality_for_model] "
            f"team_name={tname!r} win_pct={wp:.6f} run_differential={rd:.2f} "
            f"core_strength={core:.6f} mult={mult:.6f} final_quality={final_quality:.6f}",
            file=sys.stderr,
            flush=True,
        )
    return final_quality


def _find_pitcher_row(pitcher_df, raw_name: str | None):
    """Match probable starter string to pitcher_stats.csv row (odds_name column)."""
    if pitcher_df is None or pitcher_df.empty or not raw_name:
        return None
    if "odds_name" not in pitcher_df.columns:
        return None
    key = " ".join(str(raw_name).split()).strip()
    if not key:
        return None
    m = pitcher_df[pitcher_df["odds_name"].astype(str).str.strip() == key]
    if len(m) == 1:
        return m.iloc[0]
    return None


def _starter_innings(row) -> float:
    """Innings pitched from pitcher_stats row; 0 if unknown (low confidence)."""
    if row is None:
        return 0.0
    try:
        v = float(row.get("innings_pitched"))
    except (TypeError, ValueError):
        return 0.0
    if v != v:
        return 0.0
    return max(0.0, v)


def _pitcher_confidence(innings_pitched: float) -> float:
    """min(1.0, IP/100); 180 IP → 1.0, 50 IP → 0.5."""
    return min(1.0, max(0.0, innings_pitched) / 100.0)


def _starter_fip_k_bb(row) -> tuple[float, float, float]:
    """Defaults = league-average-ish when stats missing."""
    dflt_fip, dflt_k9, dflt_bb9 = 4.25, 8.8, 3.2
    if row is None:
        return (dflt_fip, dflt_k9, dflt_bb9)

    def _g(key: str, d: float) -> float:
        try:
            v = float(row.get(key))
            if v != v:
                return d
            return v
        except (TypeError, ValueError):
            return d

    fip = _g("fip", dflt_fip)
    k9 = _g("k9", dflt_k9)
    bb9 = _g("bb9", dflt_bb9)

    if _starter_innings(row) < 30:
        fip = 0.5 * fip + 0.5 * dflt_fip

    fip = min(max(fip, 2.5), 6.0)
    k9 = min(max(k9, 5.0), 13.0)
    bb9 = min(max(bb9, 1.0), 6.5)
    return (fip, k9, bb9)


def _pitcher_matchup_diff_home(home_pitcher_row, away_pitcher_row) -> float:
    """
    > 0 when home rotation matchup favors the home starter (lower FIP / more Ks / fewer walks).
    """
    h_fip, h_k9, h_bb9 = _starter_fip_k_bb(home_pitcher_row)
    a_fip, a_k9, a_bb9 = _starter_fip_k_bb(away_pitcher_row)
    return (
        0.48 * (a_fip - h_fip)
        + 0.28 * ((h_k9 - a_k9) / 2.2)
        + 0.24 * (a_bb9 - h_bb9)
    )


def _home_win_prob(
    home_team_row,
    away_team_row,
    home_pitcher_row,
    away_pitcher_row,
    home_team_name: str = "",
) -> tuple[float, float]:
    """
    Returns (p_home, park_mult).
    Park factor is applied to the home team quality signal only (not pitcher side).
    """
    diff_pitch = _pitcher_matchup_diff_home(home_pitcher_row, away_pitcher_row)
    rh = _team_quality_for_model(home_team_row)
    ra = _team_quality_for_model(away_team_row)

    pm = _park_mult(home_team_name)
    rh *= 1.0 + 0.3 * (pm - 1.0)

    diff_team = (rh - ra) + 0.02

    diff_pitch = max(-1.15, min(1.15, diff_pitch))
    diff_team = max(-DIFF_TEAM_CLAMP, min(DIFF_TEAM_CLAMP, diff_team))

    h_conf = _pitcher_confidence(_starter_innings(home_pitcher_row))
    a_conf = _pitcher_confidence(_starter_innings(away_pitcher_row))
    game_pitcher_conf = min(h_conf, a_conf)
    pitcher_w = PITCH_MATCHUP_WEIGHT * game_pitcher_conf
    team_w = 1.0 - pitcher_w
    combined = pitcher_w * diff_pitch + team_w * diff_team
    p = 1.0 / (1.0 + math.exp(-combined * LOGISTIC_K))
    return min(HOME_WIN_PROB_MAX, max(HOME_WIN_PROB_MIN, p)), pm


def main() -> int:
    import pandas as pd

    from engine.mlb_engine import (
        DEFAULT_MLB_TEAM_STATS_CSV,
        implied_probability_fair_two_sided,
        load_live_mlb_odds,
        value_summary_moneyline,
    )

    if not ODDS_PATH.exists():
        print("Missing data/odds/live_mlb_odds.json — run scripts/fetch_mlb_odds.py first.", file=sys.stderr)
        return 1

    blob = load_live_mlb_odds(ODDS_PATH)
    card_date = _card_date_iso(blob)

    stats_path = DEFAULT_MLB_TEAM_STATS_CSV
    if not stats_path.exists():
        print(
            f"MLB predict: no team stats at {stats_path} — matched 0 games (run scripts/fetch_mlb_stats.py).",
            file=sys.stderr,
        )
        stats_df = pd.DataFrame()
    else:
        try:
            stats_df = pd.read_csv(stats_path)
            _need = (
                "team_name",
                "win_pct",
                "run_differential",
                "era",
                "woba",
                "bullpen_era",
            )
            _have = set(stats_df.columns.astype(str))
            _miss = [c for c in _need if c not in _have]
            print(
                "MLB predict: team_stats loaded with pandas.read_csv — "
                f"path={stats_path} rows={len(stats_df)} "
                f"columns_match={not _miss} missing={_miss if _miss else 'none'}",
                file=sys.stderr,
                flush=True,
            )
        except Exception as e:
            print(f"MLB predict: failed to read team stats: {e} — matched 0 games.", file=sys.stderr)
            stats_df = pd.DataFrame()

    if PITCHER_STATS_PATH.exists():
        try:
            pitcher_df = pd.read_csv(PITCHER_STATS_PATH)
        except Exception as e:
            print(f"MLB predict: could not read pitcher_stats.csv: {e}", file=sys.stderr)
            pitcher_df = pd.DataFrame()
    else:
        pitcher_df = pd.DataFrame()

    games = blob.get("games") or []
    matched = 0
    plays: list[dict] = []

    for g in games:
        if not isinstance(g, dict):
            continue
        home = str(g.get("home_team") or "").strip()
        away = str(g.get("away_team") or "").strip()
        ml = g.get("moneyline") or {}
        home_odds_am = _american_to_float(ml.get("home_odds"))
        away_odds_am = _american_to_float(ml.get("away_odds"))

        hr = _find_team_row(stats_df, home)
        ar = _find_team_row(stats_df, away)
        if hr is not None and ar is not None:
            matched += 1
            if (away, home) in _TEAM_QUALITY_MATCHUP_DEBUG:
                print(
                    f"[MLB matchup debug team_quality] {away} @ {home}",
                    file=sys.stderr,
                    flush=True,
                )
                for r in (ar, hr):
                    fq, wp, rd_u, core, mult, tname = _team_quality_components(r)
                    print(
                        "[MLB debug _team_quality_for_model] "
                        f"team_name={tname!r} win_pct={wp:.6f} run_differential={rd_u:.2f} "
                        f"core_strength={core:.6f} mult={mult:.6f} final_quality={fq:.6f}",
                        file=sys.stderr,
                        flush=True,
                    )

        skip_parts: list[str] = []
        if hr is None:
            skip_parts.append(f"no_team_stats_row_home({home!r})")
        if ar is None:
            skip_parts.append(f"no_team_stats_row_away({away!r})")
        if home_odds_am is None:
            skip_parts.append("missing_or_invalid_home_moneyline")
        if away_odds_am is None:
            skip_parts.append("missing_or_invalid_away_moneyline")
        if skip_parts:
            print(
                f"{away} @ {home} | SKIP (before model/edges): {'; '.join(skip_parts)}",
                flush=True,
            )
            continue

        hp = _find_pitcher_row(pitcher_df, g.get("home_pitcher"))
        ap = _find_pitcher_row(pitcher_df, g.get("away_pitcher"))
        if hp is None:
            hp = MISSING_PITCHER_STATS_ROW
        if ap is None:
            ap = MISSING_PITCHER_STATS_ROW

        p_home_model, game_park_mult = _home_win_prob(hr, ar, hp, ap, home_team_name=home)

        fair_home, fair_away = implied_probability_fair_two_sided(
            float(home_odds_am), float(away_odds_am)
        )
        p_home = float(
            p_home_model
            - PROB_SHRINK_TOWARD_MARKET * (float(p_home_model) - float(fair_home))
        )

        # edge_h: evaluate HOME side using home_odds_am (first arg = that side's American price).
        # edge_away: evaluate AWAY side using away_odds_am. Uses shrunk p_home toward market fair.
        summ_home = value_summary_moneyline(
            float(p_home), float(home_odds_am), float(away_odds_am)
        )
        summ_away = value_summary_moneyline(
            float(1.0 - p_home), float(away_odds_am), float(home_odds_am)
        )
        edge_h = float(summ_home["edge"])
        edge_a = float(summ_away["edge"])

        if edge_h >= edge_a:
            pick_side = "home"
        else:
            pick_side = "away"

        if pick_side == "home":
            pick = home
            odds_used = float(home_odds_am)
            model_p = float(p_home)
            summ = value_summary_moneyline(model_p, odds_used, float(away_odds_am))
        else:
            pick = away
            odds_used = float(away_odds_am)
            model_p = float(1.0 - p_home)
            summ = value_summary_moneyline(model_p, odds_used, float(home_odds_am))

        edge = float(summ["edge"])
        # One line per game before edge / MIN_EDGE filtering (stdout so it shows even when stderr is quiet).
        print(
            f"{away} @ {home} | ho={home_odds_am} ao={away_odds_am} | "
            f"p_home={p_home:.3f} raw_model={float(p_home_model):.3f} fair_home={fair_home:.3f} | "
            f"park={game_park_mult:.3f} | "
            f"edge_h={edge_h:.4f} edge_a={edge_a:.4f} | pick={pick} at {odds_used}",
            flush=True,
        )

        if model_p < MIN_MODEL_PROB:
            print(
                f"SKIP_LOW_PROB: {pick} | model_prob={model_p:.3f}",
                flush=True,
            )
            continue

        if edge > MAX_EDGE_DECIMAL:
            print(
                f"FLAGGED_HIGH_EDGE: {pick} | edge={edge:.3f} exceeds MAX",
                flush=True,
            )
            continue

        if edge < MIN_EDGE_DECIMAL:
            if edge_h < 0 and edge_a < 0:
                print(
                    f"{away} @ {home} | NO_PLAY: both edges negative "
                    f"(edge_h={edge_h:.4f} edge_a={edge_a:.4f}); best_pick_edge={edge:.4f} < MIN {MIN_EDGE_DECIMAL}",
                    flush=True,
                )
            else:
                print(
                    f"{away} @ {home} | NO_PLAY: best_pick_edge={edge:.4f} < MIN {MIN_EDGE_DECIMAL} "
                    f"(edge_h={edge_h:.4f} edge_a={edge_a:.4f})",
                    flush=True,
                )
            continue

        plays.append(
            {
                "card_date": card_date,
                "event_id": str(g.get("event_id") or ""),
                "commence_time": str(g.get("commence_time") or ""),
                "home_team": home,
                "away_team": away,
                "home_pitcher": g.get("home_pitcher"),
                "away_pitcher": g.get("away_pitcher"),
                "market": "moneyline",
                "selection": pick,
                "odds_american": odds_used,
                "model_prob": float(model_p),
                "edge": float(edge),
                "park_mult": float(game_park_mult),
            }
        )

    print(
        f"MLB predict: matched {matched} game(s) between odds slate and team_stats (both teams found).",
        flush=True,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"card_date": card_date, "plays": plays}
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {OUT_PATH} ({len(plays)} play(s)), card_date={card_date}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
