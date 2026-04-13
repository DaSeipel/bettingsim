#!/usr/bin/env python3
"""
Generate data/cache/mlb_value_plays.json from data/odds/live_mlb_odds.json,
data/mlb/team_stats.csv (scripts/fetch_mlb_stats.py), and optionally
data/mlb/pitcher_stats.csv (scripts/fetch_mlb_pitchers.py), and optionally
data/mlb/recent_form.csv (scripts/fetch_mlb_recent_form.py — run once per day;
includes recent_ra_avg for short-term runs allowed and recent_era from last 5 boxscores).

Requires fetch_mlb_odds.py to have run so live_mlb_odds.json exists.
Optional: scripts/fetch_mlb_weather.py after odds (writes data/cache/mlb_weather.json) for
temperature nudge on moneyline team quality and wind nudge on predicted totals.

Best results: run fetch_mlb_pitchers.py after odds so starter FIP/xFIP blend, K9/BB9, and innings_pitched are available.
Low-IP starters shrink pitcher weight toward team stats (see PITCH_MATCHUP_WEIGHT / _pitcher_confidence;
max ~45% pitcher / 55% team when both starters have 50+ IP (full confidence; see _pitcher_confidence).

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
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategies.strategies import american_to_decimal

ODDS_PATH = ROOT / "data" / "odds" / "live_mlb_odds.json"
WEATHER_PATH = ROOT / "data" / "cache" / "mlb_weather.json"
OUT_PATH = ROOT / "data" / "cache" / "mlb_value_plays.json"
PITCHER_STATS_PATH = ROOT / "data" / "mlb" / "pitcher_stats.csv"
RECENT_FORM_PATH = ROOT / "data" / "mlb" / "recent_form.csv"
PARK_FACTORS_PATH = ROOT / "data" / "mlb" / "mlb_park_factors.json"

# Recent win% streak adjustment: added to diff_team as (home_bonus - away_bonus).
FORM_WIN_PCT_COEFF = 0.08

# Recent runs-allowed / game (from recent_form.csv, up to 7 games in window vs ~4.5 league avg).
RECENT_PITCH_RA_COEFF = 0.04
RECENT_PITCH_RA_TARGET = 4.50

# Edge and probability filters (decimal edge = EV fraction, e.g. 0.05 = 5%).
MIN_EDGE_DECIMAL = 0.03  # minimum 3% edge to qualify as a play
MAX_EDGE_DECIMAL = 0.13  # skip plays above this; prints FLAGGED_HIGH_EDGE
MIN_MODEL_PROB = 0.42  # skip picks below this win probability; prints SKIP_LOW_PROB

# Moneyline only: hard skip when the chosen side is this heavy or worse (more negative).
MAX_FAVORITE_ODDS = -160

# Moneyline only: gradual edge shrink for moderate chalk (-160 exclusive through -150 inclusive).
JUICE_PENALTY_THRESHOLD = -150
JUICE_PENALTY_RATE = 0.003

# Pull model win prob toward two-way fair implied (after logistic + clamp, before edge).
PROB_SHRINK_TOWARD_MARKET = 0.15

# When a starter is missing from pitcher_stats.csv, use league-average rate stats and
# IP=50 → confidence 1.0 with early-season /50 scale → pitcher_weight ≈ 0.45 (not 0/100).
MISSING_PITCHER_STATS_ROW: dict[str, float] = {
    "fip": 4.25,
    "xfip": 4.25,
    "k9": 8.8,
    "bb9": 3.2,
    "innings_pitched": 50.0,
}

# Max pitcher share when both starters have 50+ IP (confidence 1.0). Below that,
# pitcher_weight = PITCH_MATCHUP_WEIGHT * min(home_conf, away_conf), team_weight = 1 - pitcher_weight.
PITCH_MATCHUP_WEIGHT = 0.45
# Softer than 0.58 so wide diff_team (+ larger RD cap) does not pin every mismatch at 0.30/0.70.
LOGISTIC_K = 0.45
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


def _juice_penalized_edge(edge: float, odds_am: float) -> float:
    """
    For odds in (-160, -150] (more negative than -150 but not past MAX_FAVORITE_ODDS), scale edge down:
    edge *= (1 - rate * (|odds| - 150)). -150 unchanged; -159 gets a small haircut.
    Odds past -160 are handled by a hard skip in main(); odds weaker than -150 are unchanged.
    """
    if odds_am > JUICE_PENALTY_THRESHOLD:
        return edge
    if odds_am <= MAX_FAVORITE_ODDS:
        return edge
    factor = 1.0 - JUICE_PENALTY_RATE * (abs(float(odds_am)) - 150.0)
    return edge * max(0.0, factor)


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


def _load_recent_form_by_team_id() -> dict[int, dict]:
    """recent_form.csv keyed by team_id; empty if missing or unreadable."""
    if not RECENT_FORM_PATH.exists():
        return {}
    try:
        import pandas as pd

        df = pd.read_csv(RECENT_FORM_PATH)
    except Exception as exc:
        print(f"MLB predict: could not read recent_form.csv: {exc} — no form adjustment.", file=sys.stderr)
        return {}
    if df.empty or "team_id" not in df.columns:
        return {}
    out: dict[int, dict] = {}
    for _, row in df.iterrows():
        try:
            tid = int(row["team_id"])
        except (TypeError, ValueError):
            continue
        out[tid] = row.to_dict()
    return out


def _form_bonus_from_row(stats_row, form_by_team_id: dict[int, dict]) -> tuple[float, float | None]:
    """
    form_bonus = FORM_WIN_PCT_COEFF * (recent_win_pct - 0.5). Returns (bonus, recent_win_pct or None).
    """
    if not form_by_team_id or stats_row is None:
        return 0.0, None
    try:
        tid = int(stats_row.get("team_id"))
    except (TypeError, ValueError):
        return 0.0, None
    rec = form_by_team_id.get(tid)
    if not rec:
        return 0.0, None
    try:
        rwp = float(rec.get("recent_win_pct", 0.5))
    except (TypeError, ValueError):
        return 0.0, None
    if rwp != rwp:  # NaN
        return 0.0, None
    bonus = FORM_WIN_PCT_COEFF * (rwp - 0.5)
    return bonus, rwp


def _recent_pitch_adj_from_row(stats_row, form_by_team_id: dict[int, dict]) -> tuple[float, float | None]:
    """
    recent_pitch_adj = RECENT_PITCH_RA_COEFF * (RECENT_PITCH_RA_TARGET - recent_ra_avg).
    recent_ra_avg is mean runs allowed per game over the sample in recent_form (up to 7 games).
    Returns (adj, recent_ra_avg or None).
    """
    if not form_by_team_id or stats_row is None:
        return 0.0, None
    try:
        tid = int(stats_row.get("team_id"))
    except (TypeError, ValueError):
        return 0.0, None
    rec = form_by_team_id.get(tid)
    if not rec:
        return 0.0, None
    try:
        ra_avg = float(rec.get("recent_ra_avg"))
    except (TypeError, ValueError):
        return 0.0, None
    if ra_avg != ra_avg:  # NaN
        return 0.0, None
    adj = RECENT_PITCH_RA_COEFF * (RECENT_PITCH_RA_TARGET - ra_avg)
    return adj, ra_avg


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
    (final_quality, win_pct_used, run_diff_used, core_strength, mult, team_name, rs_bonus)
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

    try:
        rspg = float(row.get("runs_scored_per_game"))
    except (TypeError, ValueError):
        rspg = float("nan")
    if rspg != rspg:
        rspg = 4.5
    rs_bonus = (rspg - 4.5) * 0.15

    final_quality = (core + rd_bump + rs_bonus) * mult
    tname = row.get("team_name") if row is not None else None
    return final_quality, wp, rd, core, mult, tname, rs_bonus


def _team_quality_for_model(row) -> float:
    """
    Team signal for win prob: core run prevention/offense + bullpen, run-differential bump,
    runs-per-game bonus (vs ~league-average 4.5), scaled by win percentage.
    """
    final_quality, wp, rd, core, mult, tname, rs_bonus = _team_quality_components(row)
    if _MLB_DEBUG_TEAM_QUALITY:
        print(
            "[MLB debug _team_quality_for_model] "
            f"team_name={tname!r} win_pct={wp:.6f} run_differential={rd:.2f} "
            f"core_strength={core:.6f} rs_bonus={rs_bonus:.6f} mult={mult:.6f} final_quality={final_quality:.6f}",
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
    """
    min(1.0, IP/50); 50+ IP → 1.0, 25 IP → 0.5.
    Denominator 50 is for April / early season; mid-May onward consider raising to 75 or 100
    when starters have accumulated more innings (change the literal in the return below).
    """
    return min(1.0, max(0.0, innings_pitched) / 50.0)


def _starter_pitcher_rates(row) -> dict[str, float]:
    """
    FIP/xFIP blend for pitcher run-prevention signal (used for matchup + totals).
    blended = 0.5 * fip + 0.5 * xfip after <30 IP regression of both toward 4.25.
    """
    dflt_fip, dflt_k9, dflt_bb9 = 4.25, 8.8, 3.2
    if row is None:
        return {
            "blended": dflt_fip,
            "k9": dflt_k9,
            "bb9": dflt_bb9,
            "fip": dflt_fip,
            "xfip": dflt_fip,
            "ip": 0.0,
        }

    def _g(key: str, d: float) -> float:
        try:
            v = float(row.get(key))
            if v != v:
                return d
            return v
        except (TypeError, ValueError):
            return d

    fip_v = _g("fip", dflt_fip)
    try:
        xv_raw = row.get("xfip")
        if xv_raw is None or (isinstance(xv_raw, float) and xv_raw != xv_raw):
            xfip_v = fip_v
        else:
            xfip_v = float(xv_raw)
    except (TypeError, ValueError):
        xfip_v = fip_v

    k9 = _g("k9", dflt_k9)
    bb9 = _g("bb9", dflt_bb9)
    ip = _starter_innings(row)

    if ip < 30:
        fip_v = 0.5 * fip_v + 0.5 * dflt_fip
        xfip_v = 0.5 * xfip_v + 0.5 * dflt_fip

    blended = 0.5 * fip_v + 0.5 * xfip_v
    blended = min(max(blended, 2.5), 6.0)
    k9 = min(max(k9, 5.0), 13.0)
    bb9 = min(max(bb9, 1.0), 6.5)
    return {
        "blended": blended,
        "k9": k9,
        "bb9": bb9,
        "fip": fip_v,
        "xfip": xfip_v,
        "ip": ip,
    }


def _starter_fip_k_bb(row) -> tuple[float, float, float]:
    """Returns (blended_fip, k9, bb9) for matchup and totals."""
    r = _starter_pitcher_rates(row)
    return (r["blended"], r["k9"], r["bb9"])


def _pitcher_blend_debug_line(display_name: str, row) -> None:
    """Stdout: FIP, xFIP (after IP regression), blended, IP."""
    nm = (display_name or "").strip() or "?"
    r = _starter_pitcher_rates(row)
    print(
        f"[PITCHER] {nm}: FIP={r['fip']:.2f}, xFIP={r['xfip']:.2f}, blended={r['blended']:.2f}, IP={r['ip']:.1f}",
        flush=True,
    )


def _pitcher_matchup_diff_home(home_pitcher_row, away_pitcher_row) -> float:
    """
    > 0 when home rotation matchup favors the home starter (lower blended FIP/xFIP / more Ks / fewer walks).
    """
    h_fip, h_k9, h_bb9 = _starter_fip_k_bb(home_pitcher_row)
    a_fip, a_k9, a_bb9 = _starter_fip_k_bb(away_pitcher_row)
    return (
        0.48 * (a_fip - h_fip)
        + 0.28 * ((h_k9 - a_k9) / 2.2)
        + 0.24 * (a_bb9 - h_bb9)
    )


TOTALS_SCALING_CONSTANT = 2.15
TOTALS_LEAGUE_AVG_FIP = 4.25
TOTALS_PITCHER_ADJ_WEIGHT = 0.15
# Early-season scoring suppression (cold weather, pitch limits, etc.); applied in April only.
APRIL_TOTALS_ADJUSTMENT = -0.3
TOTALS_DIFF_THRESHOLD = 0.5
TOTALS_LOGISTIC_SLOPE = 0.8


def _team_era(row) -> float:
    """Blended team ERA from team_stats.csv; defaults to 4.20 if missing."""
    try:
        v = float(row.get("era"))
        if v != v:
            return 4.20
        return min(max(v, 2.5), 7.0)
    except (TypeError, ValueError):
        return 4.20


def _load_mlb_weather_by_matchup() -> dict[str, dict]:
    """Return matchup key 'Away @ Home' -> weather dict from mlb_weather.json; empty if missing."""
    if not WEATHER_PATH.exists():
        return {}
    try:
        with open(WEATHER_PATH, encoding="utf-8") as fh:
            raw = json.load(fh)
        w = raw.get("weather")
        if isinstance(w, dict):
            return w
    except Exception:
        pass
    return {}


def _matchup_weather_key(away: str, home: str) -> str:
    return f"{away.strip()} @ {home.strip()}"


def _temp_adj_from_weather(wx: Optional[dict]) -> float:
    """temp_adj = 0.02 * ((temp_f - 72) / 30); cold suppresses offense signal for home acclimation."""
    if not wx:
        return 0.0
    tf = wx.get("temp_f")
    if tf is None:
        return 0.0
    try:
        t = float(tf)
    except (TypeError, ValueError):
        return 0.0
    return 0.02 * ((t - 72.0) / 30.0)


def _totals_wind_bonus_mph_deg(wind_speed_mph: float, wind_direction_deg: float) -> float:
    """
    Totals-only nudge when wind_speed_mph > 8.
    Out (carry): direction in [135, 225] deg. In: [315, 360] U [0, 45]. Else crosswind -> 0.
    """
    if wind_speed_mph <= 8.0:
        return 0.0
    try:
        d = float(wind_direction_deg) % 360.0
    except (TypeError, ValueError):
        return 0.0
    if 135.0 <= d <= 225.0:
        return 0.4
    if d >= 315.0 or d <= 45.0:
        return -0.4
    return 0.0


def _weather_debug_print(away: str, home: str, wx: Optional[dict]) -> None:
    if wx and wx.get("temp_f") is not None:
        try:
            tf = float(wx["temp_f"])
            ws = float(wx.get("wind_speed_mph") or 0.0)
            wd = int(wx.get("wind_direction_deg") or 0)
            pp = int(wx.get("precip_prob") or 0)
        except (TypeError, ValueError):
            print(f"[WEATHER] {away} @ {home}: (invalid weather fields)", flush=True)
            return
        print(
            f"[WEATHER] {away} @ {home}: {tf:.1f}°F, wind {ws:.1f}mph dir={wd}, precip={pp}%",
            flush=True,
        )
    else:
        print(f"[WEATHER] {away} @ {home}: (no data)", flush=True)


def _predict_total(
    home_team_row,
    away_team_row,
    home_pitcher_row,
    away_pitcher_row,
    park_mult: float,
    wind_bonus: float = 0.0,
) -> float:
    """
    Predicted total runs for a game.
    Base: (home_era + away_era) * 0.5 * park_mult * TOTALS_SCALING_CONSTANT
    Adj: pitcher blended FIP/xFIP nudge (avg_blended - 4.25) * TOTALS_PITCHER_ADJ_WEIGHT.
    In April (month <= 4): APRIL_TOTALS_ADJUSTMENT applied to predicted total.
    """
    home_era = _team_era(home_team_row)
    away_era = _team_era(away_team_row)
    predicted = (home_era + away_era) * 0.5 * park_mult * TOTALS_SCALING_CONSTANT

    h_fip, _, _ = _starter_fip_k_bb(home_pitcher_row)
    a_fip, _, _ = _starter_fip_k_bb(away_pitcher_row)
    pitcher_adj = ((h_fip + a_fip) / 2.0 - TOTALS_LEAGUE_AVG_FIP) * TOTALS_PITCHER_ADJ_WEIGHT
    predicted += pitcher_adj

    if date.today().month <= 4:
        predicted += APRIL_TOTALS_ADJUSTMENT

    predicted += float(wind_bonus)
    return predicted


def _home_win_prob(
    home_team_row,
    away_team_row,
    home_pitcher_row,
    away_pitcher_row,
    home_team_name: str = "",
    form_diff: float = 0.0,
    recent_pitch_diff: float = 0.0,
    weather_temp_adj: float = 0.0,
) -> tuple[float, float]:
    """
    Returns (p_home, park_mult).
    Park factor is applied to the home team quality signal only (not pitcher side).
    form_diff = home form_bonus minus away form_bonus (recent win% streak).
    recent_pitch_diff = home minus away recent_pitch_adj (short-term runs allowed vs league avg).
    weather_temp_adj: add to home quality, subtract from away (home acclimation); from mlb_weather.json.
    All are folded into diff_team before logistic.
    """
    diff_pitch = _pitcher_matchup_diff_home(home_pitcher_row, away_pitcher_row)
    rh = _team_quality_for_model(home_team_row)
    ra = _team_quality_for_model(away_team_row)

    pm = _park_mult(home_team_name)
    rh *= 1.0 + 0.3 * (pm - 1.0)
    ta = float(weather_temp_adj)
    rh += ta
    ra -= ta

    diff_team = (rh - ra) + 0.02 + float(form_diff) + float(recent_pitch_diff)

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

    form_by_team_id = _load_recent_form_by_team_id()
    if form_by_team_id:
        print(
            f"MLB predict: recent_form.csv loaded — {len(form_by_team_id)} team row(s).",
            file=sys.stderr,
            flush=True,
        )

    games = blob.get("games") or []
    weather_by_matchup = _load_mlb_weather_by_matchup()
    if weather_by_matchup:
        print(
            f"MLB predict: loaded weather for {len(weather_by_matchup)} matchup(s) from {WEATHER_PATH}.",
            file=sys.stderr,
            flush=True,
        )

    matched = 0
    plays: list[dict] = []

    for g in games:
        if not isinstance(g, dict):
            continue
        home = str(g.get("home_team") or "").strip()
        away = str(g.get("away_team") or "").strip()
        wx = weather_by_matchup.get(_matchup_weather_key(away, home))
        _weather_debug_print(away, home, wx)

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
                    fq, wp, rd_u, core, mult, tname, rs_b = _team_quality_components(r)
                    print(
                        "[MLB debug _team_quality_for_model] "
                        f"team_name={tname!r} win_pct={wp:.6f} run_differential={rd_u:.2f} "
                        f"core_strength={core:.6f} rs_bonus={rs_b:.6f} mult={mult:.6f} final_quality={fq:.6f}",
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

        _pitcher_blend_debug_line(str(g.get("away_pitcher") or "away"), ap)
        _pitcher_blend_debug_line(str(g.get("home_pitcher") or "home"), hp)

        bonus_h, rwp_h = _form_bonus_from_row(hr, form_by_team_id)
        bonus_a, rwp_a = _form_bonus_from_row(ar, form_by_team_id)
        pitch_h, raa_h = _recent_pitch_adj_from_row(hr, form_by_team_id)
        pitch_a, raa_a = _recent_pitch_adj_from_row(ar, form_by_team_id)
        for tlabel, b, rwp in ((home, bonus_h, rwp_h), (away, bonus_a, rwp_a)):
            if b != 0.0 and rwp is not None:
                print(
                    f"[FORM] {tlabel}: recent_win_pct={rwp:.3f} form_bonus={b:+.3f}",
                    flush=True,
                )
        for tlabel, adj, raa in ((home, pitch_h, raa_h), (away, pitch_a, raa_a)):
            if raa is not None:
                print(
                    f"[RECENT_PITCH] {tlabel}: recent_ra_avg={raa:.3f} pitch_adj={adj:+.3f}",
                    flush=True,
                )
        form_diff = bonus_h - bonus_a
        recent_pitch_diff = pitch_h - pitch_a
        temp_adj = _temp_adj_from_weather(wx)

        p_home_model, game_park_mult = _home_win_prob(
            hr,
            ar,
            hp,
            ap,
            home_team_name=home,
            form_diff=form_diff,
            recent_pitch_diff=recent_pitch_diff,
            weather_temp_adj=temp_adj,
        )

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
        if odds_used <= MAX_FAVORITE_ODDS:
            print(
                f"SKIP_HEAVY_CHALK: {pick} at {odds_used:.0f} exceeds MAX_FAVORITE_ODDS={MAX_FAVORITE_ODDS}",
                flush=True,
            )
            continue
        edge = _juice_penalized_edge(edge, odds_used)
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

    # ——— Totals (over/under) pass ———
    totals_considered = 0
    for g in games:
        if not isinstance(g, dict):
            continue
        home = str(g.get("home_team") or "").strip()
        away = str(g.get("away_team") or "").strip()
        wx_t = weather_by_matchup.get(_matchup_weather_key(away, home))

        total_data = g.get("total") or {}
        market_line = total_data.get("line")
        over_odds_am = _american_to_float(total_data.get("over_odds"))
        under_odds_am = _american_to_float(total_data.get("under_odds"))

        hr = _find_team_row(stats_df, home)
        ar = _find_team_row(stats_df, away)

        skip_t: list[str] = []
        if hr is None:
            skip_t.append(f"no_team_stats_home({home!r})")
        if ar is None:
            skip_t.append(f"no_team_stats_away({away!r})")
        if market_line is None:
            skip_t.append("missing_total_line")
        if over_odds_am is None:
            skip_t.append("missing_over_odds")
        if under_odds_am is None:
            skip_t.append("missing_under_odds")
        if skip_t:
            print(f"{away} @ {home} | TOTALS SKIP: {'; '.join(skip_t)}", flush=True)
            continue

        market_line = float(market_line)

        hp = _find_pitcher_row(pitcher_df, g.get("home_pitcher"))
        ap = _find_pitcher_row(pitcher_df, g.get("away_pitcher"))
        if hp is None:
            hp = MISSING_PITCHER_STATS_ROW
        if ap is None:
            ap = MISSING_PITCHER_STATS_ROW

        pm = _park_mult(home)
        wind_bonus = 0.0
        if wx_t:
            try:
                wspd = float(wx_t.get("wind_speed_mph") or 0.0)
                wdir = float(wx_t.get("wind_direction_deg") or 0.0)
                wind_bonus = _totals_wind_bonus_mph_deg(wspd, wdir)
            except (TypeError, ValueError):
                wind_bonus = 0.0
        pred_total = _predict_total(hr, ar, hp, ap, pm, wind_bonus=wind_bonus)
        total_diff = pred_total - market_line

        over_prob = 1.0 / (1.0 + math.exp(-total_diff * TOTALS_LOGISTIC_SLOPE))

        over_dec = american_to_decimal(float(over_odds_am))
        under_dec = american_to_decimal(float(under_odds_am))
        edge_over = over_prob * over_dec - 1.0
        edge_under = (1.0 - over_prob) * under_dec - 1.0

        if total_diff > TOTALS_DIFF_THRESHOLD:
            t_side = "over"
            t_selection = f"Over {market_line}"
            t_odds_am = float(over_odds_am)
            t_model_p = over_prob
            t_edge = edge_over
        elif total_diff < -TOTALS_DIFF_THRESHOLD:
            t_side = "under"
            t_selection = f"Under {market_line}"
            t_odds_am = float(under_odds_am)
            t_model_p = 1.0 - over_prob
            t_edge = edge_under
        else:
            print(
                f"{away} @ {home} | TOTALS NO_PLAY: pred={pred_total:.2f} line={market_line} diff={total_diff:+.2f} (within ±{TOTALS_DIFF_THRESHOLD})",
                flush=True,
            )
            continue

        totals_considered += 1
        print(
            f"{away} @ {home} | TOTALS: pred={pred_total:.2f} line={market_line} diff={total_diff:+.2f} "
            f"side={t_side} prob={t_model_p:.3f} edge={t_edge:.4f} odds={t_odds_am}",
            flush=True,
        )

        if t_model_p < MIN_MODEL_PROB:
            print(f"TOTALS SKIP_LOW_PROB: {t_selection} | model_prob={t_model_p:.3f}", flush=True)
            continue
        if t_edge > MAX_EDGE_DECIMAL:
            print(f"TOTALS FLAGGED_HIGH_EDGE: {t_selection} | edge={t_edge:.3f} exceeds MAX", flush=True)
            continue
        if t_edge < MIN_EDGE_DECIMAL:
            print(
                f"{away} @ {home} | TOTALS NO_PLAY: edge={t_edge:.4f} < MIN {MIN_EDGE_DECIMAL}",
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
                "market": "total",
                "selection": t_selection,
                "odds_american": t_odds_am,
                "model_prob": float(t_model_p),
                "edge": float(t_edge),
                "park_mult": float(pm),
                "predicted_total": round(float(pred_total), 2),
                "total_line": float(market_line),
            }
        )

    ml_plays = [p for p in plays if p.get("market") == "moneyline"]
    tot_plays = [p for p in plays if p.get("market") == "total"]
    print(
        f"MLB predict: matched {matched} game(s) between odds slate and team_stats (both teams found). "
        f"Moneyline plays: {len(ml_plays)}, Totals plays: {len(tot_plays)}.",
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
