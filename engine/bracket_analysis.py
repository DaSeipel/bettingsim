"""
2026 NCAA Tournament Bracket Analysis: sleepers, Monte Carlo Final 4, Walters plays.
Uses XGBoost spread model plus Tournament Success Multipliers (Veteran, Closer FT, 3P variance).
"""

from __future__ import annotations

import csv
import io
import random
import sqlite3
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .betting_models import (
    build_ncaab_feature_row_from_team_stats,
    get_spread_predicted_margin,
)

# March factor thresholds
VETERAN_EXP_MIN = 2.2
NON_VETERAN_EXP_MAX = 1.8
VETERAN_MARGIN_BOOST = 1.5
CLOSER_GAME_MARGIN_THRESHOLD = 4.0
ELITE_FT_PCT_MIN = 77.0
CLOSER_FT_WIN_PROB_BOOST = 0.03
TOP_3P_PCT_PERCENTILE = 90
MARGIN_SIGMA_BASE = 4.0
MARGIN_SIGMA_ELITE_3P = 6.0
GLITCH_F4_PCT_MIN = 5.0
GLITCH_TOP_N_CATEGORIES = 50
GLITCH_MIN_CATEGORIES = 2

# Tier calibration: when |ModelRank_A - ModelRank_B| > this, apply strength weight to margin
TIER_RANK_DIFF_THRESHOLD = 100
TIER_ANCHOR_RANK_DIFF_FOR_22 = 50   # when 1-4 vs 13-16 and rank_diff > this, use 22-pt floor (else 10)
TIER_MULTIPLIER_FAVORS_BETTER = 1.8  # expand margin when it already favors the better-ranked team
TIER_MULTIPLIER_FAVORS_WORSE = 0.6   # shrink margin when it favors the worse-ranked team
# Anchor: 1-4 seed vs 13-16 seed — predicted margin floor for the favorite (unless underdog is glitch)
ANCHOR_MIN_ELITE_VS_LOW = 10.0       # minimum margin when rank_diff <= TIER_ANCHOR_RANK_DIFF_FOR_22
ANCHOR_MIN_ELITE_VS_LOW_BIG_TIER = 22.0  # when rank_diff > TIER_ANCHOR_RANK_DIFF_FOR_22 (strong tier gap)
# Walters: only show plays with delta in this range; delta > 15 flagged as data error
WALTERS_DELTA_MIN = 2.0
WALTERS_DELTA_MAX = 7.0
WALTERS_DATA_ERROR_THRESHOLD = 15.0

# Alias map: common bracket/rankings names -> canonical name used in ncaab_team_season_stats (and DB).
# Enables "Saint Mary's" / "UConn" etc. to pull the correct stats.
BRACKET_TEAM_ALIAS_MAP: dict[str, str] = {
    "Saint Mary's": "St. Mary's",
    "Saint Marys": "St. Mary's",
    "St Mary's": "St. Mary's",
    "UConn": "Connecticut",
    "UNC": "North Carolina",
    "USC": "Southern California",
    "BYU": "Brigham Young",
    "Ole Miss": "Mississippi",
    "VCU": "Virginia Commonwealth",
    "SMU": "Southern Methodist",
    "UNLV": "Nevada Las Vegas",
    "UCLA": "California Los Angeles",
    "LSU": "Louisiana State",
    "Texas A&M": "Texas A&M",
    "Florida St.": "Florida State",
    "Florida St": "Florida State",
    "FSU": "Florida State",
    "NC State": "North Carolina State",
    "N.C. State": "North Carolina State",
    "Ohio St.": "Ohio State",
    "Ohio St": "Ohio State",
    "Michigan St.": "Michigan State",
    "Michigan St": "Michigan State",
    "Penn St.": "Penn State",
    "Penn St": "Penn State",
    "Iowa St.": "Iowa State",
    "Iowa St": "Iowa State",
    "Oklahoma St.": "Oklahoma State",
    "Oklahoma St": "Oklahoma State",
    "Kansas St.": "Kansas State",
    "Kansas St": "Kansas State",
    "San Diego St.": "San Diego State",
    "San Diego St": "San Diego State",
    "Wichita St.": "Wichita State",
    "Wichita St": "Wichita State",
    "Boise St.": "Boise State",
    "Boise St": "Boise State",
    "Colorado St.": "Colorado State",
    "Colorado St": "Colorado State",
    "Utah St.": "Utah State",
    "Utah St": "Utah State",
    "New Mexico St.": "New Mexico State",
    "New Mexico St": "New Mexico State",
    "Murray St.": "Murray State",
    "Murray St": "Murray State",
    # Bracket cross-reference (bracket_2026.csv vs ncaab_team_season_stats)
    "CA Baptist": "Cal Baptist",
    "Hawai'i": "Hawaii",
    "N Dakota St.": "North Dakota St.",
    "Long Island": "LIU",
}


def resolve_team_name(team: str, alias_map: Optional[dict[str, str]] = None) -> str:
    """Return canonical name for DB/stats lookup. Uses BRACKET_TEAM_ALIAS_MAP if alias_map is None."""
    t = (team or "").strip()
    if not t:
        return t
    m = alias_map if alias_map is not None else BRACKET_TEAM_ALIAS_MAP
    return m.get(t, t)


# --- Parsing ---

def parse_bracket_csv(text: str) -> list[dict]:
    """
    Parse bracket CSV: first-round matchups (32 games).
    Expected columns: TeamA, SeedA, TeamB, SeedB [, MarketSpread]
    or: Region, SeedA, TeamA, SeedB, TeamB [, MarketSpread]
    Returns list of {team_a, seed_a, team_b, seed_b, market_spread}.
    """
    rows = []
    for line in csv.reader(io.StringIO(text.strip())):
        if not line or all(not str(c).strip() for c in line):
            continue
        parts = [str(x).strip() for x in line]
        # Skip header row
        if parts and (parts[0].lower().startswith("team") or parts[0].lower() == "region" or parts[0].lower() == "region_name"):
            continue
        if len(parts) >= 4:
            team_a = team_b = None
            seed_a = seed_b = None
            market_spread = None
            # Format: Region, SeedA, TeamA, SeedB, TeamB [, MarketSpread] (when first col is region name)
            if len(parts) >= 5 and parts[0] in ("East", "West", "South", "Midwest"):
                try:
                    seed_a = int(float(parts[1]))
                    seed_b = int(float(parts[3]))
                    team_a = parts[2]
                    team_b = parts[4]
                except (ValueError, IndexError, TypeError):
                    pass
            # Format: TeamA, SeedA, TeamB, SeedB [, MarketSpread]
            if (team_a is None or team_b is None) and len(parts) >= 4:
                try:
                    seed_a = int(float(parts[1]))
                    seed_b = int(float(parts[3]))
                    team_a = parts[0]
                    team_b = parts[2]
                except (ValueError, IndexError, TypeError):
                    pass
            # Format: Region, SeedA, TeamA, SeedB, TeamB (fallback when first format didn't parse)
            if (team_a is None or team_b is None) and len(parts) >= 5:
                try:
                    seed_a = int(float(parts[1]))
                    seed_b = int(float(parts[3]))
                    team_a = parts[2]
                    team_b = parts[4]
                except (ValueError, IndexError, TypeError):
                    pass
            if team_a and team_b is not None and seed_a is not None and seed_b is not None:
                if len(parts) >= 6:
                    try:
                        market_spread = float(parts[5])
                    except (ValueError, TypeError):
                        pass
                elif len(parts) >= 5:
                    try:
                        market_spread = float(parts[4])
                    except (ValueError, TypeError):
                        pass
                rows.append({
                    "team_a": team_a,
                    "seed_a": seed_a,
                    "team_b": team_b,
                    "seed_b": seed_b,
                    "market_spread": market_spread,
                })
    return rows


def parse_power_rankings_csv(text: str) -> list[dict]:
    """
    Parse power rankings CSV: Team, Seed, ModelRank (or OfficialSeed, ModelRank).
    ModelRank 1 = best team. Returns list of {team, seed, model_rank}.
    """
    rows = []
    for line in csv.reader(io.StringIO(text.strip())):
        if not line or all(not str(c).strip() for c in line):
            continue
        parts = [str(x).strip() for x in line]
        if len(parts) >= 3:
            if str(parts[0]).lower() in ("team", "school", "name") and str(parts[1]).lower() in ("seed", "official_seed") and "rank" in str(parts[2]).lower():
                continue  # skip header
            try:
                team = parts[0]
                seed = int(float(parts[1]))
                model_rank = int(float(parts[2]))
                rows.append({"team": team, "seed": seed, "model_rank": model_rank})
            except (ValueError, TypeError, IndexError):
                pass
    return rows


# --- Value Delta (Seed Gap) ---

def official_rank_from_seed(seed: int) -> float:
    """Convert seed (1-16) to approximate overall committee rank (1-64). Midpoint of the four teams with that seed."""
    return 4.0 * (seed - 1) + 2.0


def compute_value_deltas(
    power_rankings: list[dict],
    bracket_matchups: list[dict],
    alias_map: Optional[dict[str, str]] = None,
) -> list[dict]:
    """
    Value Delta = (Official Seed Rank) - (Model Projected Rank). Higher positive = undervalued sleeper.
    Returns list of {team, seed, model_rank, value_delta} for all teams in bracket + rankings.
    Uses resolve_team_name so bracket aliases (e.g. UConn) match rankings (e.g. Connecticut).
    """
    rank_by_team = {r["team"].strip(): r for r in power_rankings}
    teams_seen = set()
    deltas = []
    for m in bracket_matchups:
        for key in ("team_a", "team_b"):
            team = m[key].strip()
            seed = m["seed_a"] if key == "team_a" else m["seed_b"]
            if team in teams_seen:
                continue
            teams_seen.add(team)
            r = rank_by_team.get(team) or rank_by_team.get(resolve_team_name(team, alias_map))
            if r is None:
                # Try fuzzy: match first word or strip common suffixes
                for rteam, data in rank_by_team.items():
                    if team in rteam or rteam in team:
                        r = data
                        break
            if r is None:
                continue
            model_rank = r.get("model_rank")
            if model_rank is None:
                continue
            official = official_rank_from_seed(seed)
            value_delta = official - float(model_rank)
            deltas.append({
                "team": team,
                "seed": seed,
                "model_rank": int(model_rank),
                "value_delta": round(value_delta, 1),
            })
    return sorted(deltas, key=lambda x: -x["value_delta"])


# --- March factor stats: load from ncaab_team_season_stats ---

def load_march_stats(db_path: Path, season: Optional[int] = None) -> tuple[dict[str, dict], dict[str, int], dict[str, int], dict[str, int]]:
    """
    Load free_throw_pct, three_point_pct, roster_experience_years for all teams in ncaab_team_season_stats.
    Uses columns if present; falls back to FTR (as FT proxy), 3P_O for 3P.
    Returns (team_stats, rank_ft, rank_3p, rank_exp) where rank_* are 1-based (1 = best).
    """
    if not db_path.exists():
        return {}, {}, {}, {}
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM ncaab_team_season_stats", conn)
    except Exception:
        return {}, {}, {}, {}
    finally:
        conn.close()
    if df.empty or "TEAM" not in df.columns:
        return {}, {}, {}, {}
    if "season" in df.columns and season is not None:
        df = df[df["season"].astype(int) == int(season)]
    if df.empty and "season" in df.columns:
        conn2 = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM ncaab_team_season_stats", conn2)
        conn2.close()
        if not df.empty:
            df = df[df["season"].astype(int) == df["season"].astype(int).max()]
    team_stats = {}
    for _, r in df.iterrows():
        team = str(r.get("TEAM", "")).strip()
        if not team:
            continue
        ft = None
        if "free_throw_pct" in r.index and pd.notna(r.get("free_throw_pct")):
            try:
                ft = float(r["free_throw_pct"])
            except (TypeError, ValueError):
                pass
        if ft is None and "FTR" in r.index and pd.notna(r.get("FTR")):
            try:
                ft = float(r["FTR"]) * 100.0
            except (TypeError, ValueError):
                pass
        three_p = None
        if "three_point_pct" in r.index and pd.notna(r.get("three_point_pct")):
            try:
                three_p = float(r["three_point_pct"])
            except (TypeError, ValueError):
                pass
        if three_p is None and "3P_O" in r.index and pd.notna(r.get("3P_O")):
            try:
                v = float(r["3P_O"])
                three_p = v * 100.0 if v <= 1.0 else v
            except (TypeError, ValueError):
                pass
        exp = None
        if "roster_experience_years" in r.index and pd.notna(r.get("roster_experience_years")):
            try:
                exp = float(r["roster_experience_years"])
            except (TypeError, ValueError):
                pass
        team_stats[team] = {
            "free_throw_pct": ft if ft is not None else 0.0,
            "three_point_pct": three_p if three_p is not None else 0.0,
            "roster_experience_years": exp if exp is not None else 0.0,
        }
    # National rankings (1 = best). Higher FT% / 3P% / experience = better rank.
    teams = list(team_stats.keys())
    by_ft = sorted(teams, key=lambda t: -(team_stats[t]["free_throw_pct"]))
    by_3p = sorted(teams, key=lambda t: -(team_stats[t]["three_point_pct"]))
    by_exp = sorted(teams, key=lambda t: -(team_stats[t]["roster_experience_years"]))
    rank_ft = {t: i + 1 for i, t in enumerate(by_ft)}
    rank_3p = {t: i + 1 for i, t in enumerate(by_3p)}
    rank_exp = {t: i + 1 for i, t in enumerate(by_exp)}
    return team_stats, rank_ft, rank_3p, rank_exp


def _march_tags_for_team(
    team: str,
    team_stats: dict,
    rank_ft: dict,
    rank_3p: dict,
) -> list[str]:
    """Return list of tags: Elite FT (top 40 & >77%), Elite 3P (top 10%), Veteran (exp > 2.2)."""
    tags = []
    s = team_stats.get(team, {})
    ft = s.get("free_throw_pct") or 0
    if ft > ELITE_FT_PCT_MIN and rank_ft.get(team, 999) <= 40:
        tags.append("Elite FT")
    n_teams = len(rank_3p)
    if n_teams and rank_3p.get(team, 999) <= max(1, int(n_teams * (TOP_3P_PCT_PERCENTILE / 100.0))):
        tags.append("Elite 3P")
    if (s.get("roster_experience_years") or 0) > VETERAN_EXP_MIN:
        tags.append("Veteran")
    return tags


# --- Win probability from XGBoost margin (neutral site) ---

MARGIN_TO_WIN_PROB_K = 0.35


def margin_to_win_prob(pred_margin: float, k: float = MARGIN_TO_WIN_PROB_K) -> float:
    """P(home/team_a wins) from predicted margin (home - away). Clipped to (0.02, 0.98)."""
    p = 1.0 / (1.0 + np.exp(-k * pred_margin))
    return float(max(0.02, min(0.98, p)))


def get_win_prob_for_matchup(
    team_a: str,
    team_b: str,
    db_path: Optional[Path] = None,
    game_date: Optional[str] = None,
) -> Optional[float]:
    """
    Return P(team_a wins) in a neutral-site game using XGBoost predicted margin (team_a as home).
    Returns None if feature row cannot be built. No March factors applied.
    """
    row = build_ncaab_feature_row_from_team_stats(team_a, team_b, game_date=game_date, db_path=db_path)
    if row is None:
        return None
    margin = get_spread_predicted_margin(row, 0.0, league="ncaab", home_team=team_a, away_team=team_b)
    if margin is None:
        return None
    return margin_to_win_prob(float(margin))


def _potential_glitch_teams(
    r1_matchups: list[dict],
    rank_ft: Optional[dict],
    rank_3p: Optional[dict],
    rank_exp: Optional[dict],
    alias_map: Optional[dict[str, str]] = None,
) -> set[str]:
    """Canonical names of teams that are seed 7+ and Top 50 in at least 2 of FT%, 3P%, Experience (for anchor exception)."""
    team_to_seed = {}
    for m in r1_matchups:
        team_to_seed[resolve_team_name(m["team_a"], alias_map)] = m["seed_a"]
        team_to_seed[resolve_team_name(m["team_b"], alias_map)] = m["seed_b"]
    out = set()
    for team_c, seed in team_to_seed.items():
        if seed is None or seed < 7:
            continue
        n = 0
        if rank_ft and rank_ft.get(team_c, 999) <= GLITCH_TOP_N_CATEGORIES:
            n += 1
        if rank_3p and rank_3p.get(team_c, 999) <= GLITCH_TOP_N_CATEGORIES:
            n += 1
        if rank_exp and rank_exp.get(team_c, 999) <= GLITCH_TOP_N_CATEGORIES:
            n += 1
        if n >= GLITCH_MIN_CATEGORIES:
            out.add(team_c)
    return out


def _apply_tier_and_anchor(
    raw_margin: float,
    ta_c: str,
    tb_c: str,
    model_rank_by_canonical: dict[str, int],
    team_to_seed: dict[str, int],
    potential_glitch: set[str],
) -> float:
    """
    Apply tier-adjustment and anchor to raw model margin (team_a - team_b).
    When |ModelRank_A - ModelRank_B| > 100, scale margin; when 1-4 seed vs 13-16 seed, enforce minimum margin for favorite unless underdog is glitch.
    """
    margin = float(raw_margin)
    rank_a = model_rank_by_canonical.get(ta_c, 999)
    rank_b = model_rank_by_canonical.get(tb_c, 999)
    seed_a = team_to_seed.get(ta_c)
    seed_b = team_to_seed.get(tb_c)
    rank_diff = abs(rank_a - rank_b)

    # Tier adjustment: when rank diff > 100, scale margin
    if rank_diff > TIER_RANK_DIFF_THRESHOLD:
        better_favored = (margin > 0 and rank_a < rank_b) or (margin < 0 and rank_a > rank_b)
        if better_favored:
            margin *= TIER_MULTIPLIER_FAVORS_BETTER
        else:
            margin *= TIER_MULTIPLIER_FAVORS_WORSE

    # Anchor: 1-4 vs 13-16 — margin cannot be less than floor for the favorite unless underdog is glitch
    def _low_seed(seed): return seed is not None and 13 <= seed <= 16
    def _elite_seed(seed): return seed is not None and 1 <= seed <= 4
    floor_small = ANCHOR_MIN_ELITE_VS_LOW
    floor_big = ANCHOR_MIN_ELITE_VS_LOW_BIG_TIER if rank_diff > TIER_ANCHOR_RANK_DIFF_FOR_22 else floor_small

    if _elite_seed(seed_a) and _low_seed(seed_b) and tb_c not in potential_glitch:
        margin = max(margin, floor_big)
    elif _elite_seed(seed_b) and _low_seed(seed_a) and ta_c not in potential_glitch:
        margin = min(margin, -floor_big)

    return margin


def _is_elite_3p(team: str, team_stats: dict, rank_3p: dict) -> bool:
    """Top 10% of three_point_pct (by rank)."""
    n = len(rank_3p)
    if n == 0:
        return False
    return rank_3p.get(team, 999) <= max(1, int(n * (TOP_3P_PCT_PERCENTILE / 100.0)))


def _build_get_winner_with_march_factors(
    db_path: Path,
    game_date: Optional[str],
    team_stats: dict,
    rank_ft: dict,
    rank_3p: dict,
    margin_cache: dict,
    alias_map: Optional[dict[str, str]] = None,
    model_rank_by_canonical: Optional[dict[str, int]] = None,
    team_to_seed: Optional[dict[str, int]] = None,
    potential_glitch: Optional[set] = None,
) -> Callable[[str, str, random.Random], str]:
    """
    Return a function winner(ta, tb, rng) that returns the winning team using cached margin + March factors.
    Caches pred_margin by canonical (resolved) names. Applies tier adjustment and anchor when model_rank_by_canonical/team_to_seed provided.
    """
    model_rank_by_canonical = model_rank_by_canonical or {}
    team_to_seed = team_to_seed or {}
    potential_glitch = potential_glitch or set()

    def _get_margin(ta: str, tb: str) -> Optional[float]:
        ta_c = resolve_team_name(ta, alias_map)
        tb_c = resolve_team_name(tb, alias_map)
        key = (ta_c, tb_c)
        if key not in margin_cache:
            row = build_ncaab_feature_row_from_team_stats(ta_c, tb_c, game_date=game_date, db_path=db_path)
            if row is None:
                margin_cache[key] = None
                return None
            m = get_spread_predicted_margin(row, 0.0, league="ncaab", home_team=ta_c, away_team=tb_c)
            if m is not None and (model_rank_by_canonical or team_to_seed):
                m = _apply_tier_and_anchor(m, ta_c, tb_c, model_rank_by_canonical, team_to_seed, potential_glitch)
            margin_cache[key] = m
        return margin_cache[key]

    def winner(ta: str, tb: str, rng: random.Random) -> str:
        pred = _get_margin(ta, tb)
        if pred is None:
            return ta if rng.random() < 0.5 else tb
        margin = float(pred)
        ta_c = resolve_team_name(ta, alias_map)
        tb_c = resolve_team_name(tb, alias_map)
        # Veteran Edge (lookup by canonical name)
        exp_a = (team_stats.get(ta_c) or {}).get("roster_experience_years") or 0
        exp_b = (team_stats.get(tb_c) or {}).get("roster_experience_years") or 0
        if exp_a > VETERAN_EXP_MIN and exp_b < NON_VETERAN_EXP_MAX:
            margin += VETERAN_MARGIN_BOOST
        elif exp_b > VETERAN_EXP_MIN and exp_a < NON_VETERAN_EXP_MAX:
            margin -= VETERAN_MARGIN_BOOST
        # 3P Variance
        if _is_elite_3p(ta_c, team_stats, rank_3p) or _is_elite_3p(tb_c, team_stats, rank_3p):
            sigma = MARGIN_SIGMA_ELITE_3P
            margin_sim = margin + rng.gauss(0, sigma)
            return ta if margin_sim > 0 else tb
        # Else: win prob + Closer Factor
        p = margin_to_win_prob(margin)
        if abs(margin) <= CLOSER_GAME_MARGIN_THRESHOLD:
            ft_a = (team_stats.get(ta_c) or {}).get("free_throw_pct") or 0
            ft_b = (team_stats.get(tb_c) or {}).get("free_throw_pct") or 0
            rank_a = rank_ft.get(ta_c, 999)
            rank_b = rank_ft.get(tb_c, 999)
            if ft_a > ELITE_FT_PCT_MIN and rank_a <= 40 and (ft_b <= ELITE_FT_PCT_MIN or rank_b > 40):
                p = min(0.98, p + CLOSER_FT_WIN_PROB_BOOST)
            elif ft_b > ELITE_FT_PCT_MIN and rank_b <= 40 and (ft_a <= ELITE_FT_PCT_MIN or rank_a > 40):
                p = max(0.02, p - CLOSER_FT_WIN_PROB_BOOST)
        return ta if rng.random() < p else tb

    return winner


def get_model_spread_for_matchup(
    team_a: str,
    team_b: str,
    db_path: Optional[Path] = None,
    game_date: Optional[str] = None,
    alias_map: Optional[dict[str, str]] = None,
) -> Optional[float]:
    """Model's predicted spread in team_a perspective (team_a - team_b). Positive = team_a favored."""
    ta_c = resolve_team_name(team_a, alias_map)
    tb_c = resolve_team_name(team_b, alias_map)
    row = build_ncaab_feature_row_from_team_stats(ta_c, tb_c, game_date=game_date, db_path=db_path)
    if row is None:
        return None
    return get_spread_predicted_margin(row, 0.0, league="ncaab", home_team=ta_c, away_team=tb_c)


# --- Monte Carlo bracket simulation ---

def _run_one_bracket(
    r1_matchups: list[dict],
    get_winner: Callable[[str, str, random.Random], str],
    rng: random.Random,
) -> list[str]:
    """
    Run one bracket: 32 R1 -> 16 R2 -> 8 R3 -> 4 R4 (Elite 8). Return the 4 Final Four teams (winners of R4).
    get_winner(ta, tb, rng) returns the winning team.
    """
    # Round 1: 32 games -> 32 winners
    r1_winners = []
    for m in r1_matchups:
        w = get_winner(m["team_a"], m["team_b"], rng)
        r1_winners.append(w)

    # Round 2: 16 games (pairs of R1 winners)
    r2_winners = []
    for i in range(0, 32, 2):
        w = get_winner(r1_winners[i], r1_winners[i + 1], rng)
        r2_winners.append(w)

    # Round 3: 8 games
    r3_winners = []
    for i in range(0, 16, 2):
        w = get_winner(r2_winners[i], r2_winners[i + 1], rng)
        r3_winners.append(w)

    # Round 4 (Elite 8): 4 games -> Final Four
    final_four = []
    for i in range(0, 8, 2):
        w = get_winner(r3_winners[i], r3_winners[i + 1], rng)
        final_four.append(w)

    return final_four


def run_monte_carlo_bracket(
    r1_matchups: list[dict],
    get_winner: Callable[[str, str, random.Random], str],
    n_sims: int = 10_000,
    seed: Optional[int] = None,
) -> dict:
    """
    Run n_sims bracket simulations. Return {
        "final4_counts": {team: count},
        "final4_pct": {team: pct},
        "r1_matchups": r1_matchups (with seeds for glitch lookup),
    }.
    """
    rng = random.Random(seed)
    final4_counts = {}
    for _ in range(n_sims):
        f4 = _run_one_bracket(r1_matchups, get_winner, rng)
        for team in f4:
            final4_counts[team] = final4_counts.get(team, 0) + 1
    final4_pct = {t: (c / n_sims) * 100.0 for t, c in final4_counts.items()}
    return {
        "final4_counts": final4_counts,
        "final4_pct": final4_pct,
        "n_sims": n_sims,
        "r1_matchups": r1_matchups,
    }


# --- Glitch teams: seed 7+ and Final 4 >5% AND Top 50 in at least 2 of (FT%, 3P%, Experience) ---

def get_glitch_teams(
    result: dict,
    r1_matchups: list[dict],
    team_stats: Optional[dict] = None,
    rank_ft: Optional[dict] = None,
    rank_3p: Optional[dict] = None,
    rank_exp: Optional[dict] = None,
    alias_map: Optional[dict[str, str]] = None,
) -> list[dict]:
    """Teams seeded #7 or lower that reach Final 4 in >5% of sims AND rank Top 50 in at least 2 of FT%, 3P%, Experience."""
    final4_pct = result["final4_pct"]
    team_to_seed = {}
    for m in r1_matchups:
        team_to_seed[m["team_a"]] = m["seed_a"]
        team_to_seed[m["team_b"]] = m["seed_b"]
    glitch = []
    for team, pct in final4_pct.items():
        seed = team_to_seed.get(team)
        if seed is None or seed < 7 or pct <= GLITCH_F4_PCT_MIN:
            continue
        team_c = resolve_team_name(team, alias_map)
        top50_count = 0
        if rank_ft and rank_ft.get(team_c, 999) <= GLITCH_TOP_N_CATEGORIES:
            top50_count += 1
        if rank_3p and rank_3p.get(team_c, 999) <= GLITCH_TOP_N_CATEGORIES:
            top50_count += 1
        if rank_exp and rank_exp.get(team_c, 999) <= GLITCH_TOP_N_CATEGORIES:
            top50_count += 1
        if top50_count >= GLITCH_MIN_CATEGORIES:
            glitch.append({"team": team, "seed": seed, "final4_pct": round(pct, 1)})
    return sorted(glitch, key=lambda x: -x["final4_pct"])


# --- Walters Play: first-round games where model spread vs market is in 2-7 pt range; delta > 15 = data error ---

def get_walters_plays(
    r1_matchups: list[dict],
    get_model_spread: Callable[[str, str], Optional[float]],
    top_n: int = 3,
) -> tuple[list[dict], list[dict]]:
    """
    First-round matchups where model spread differs from market spread.
    Returns (plays, data_errors):
    - plays: diff_pts in [WALTERS_DELTA_MIN, WALTERS_DELTA_MAX] (2-7 pts), up to top_n, sorted by diff desc.
    - data_errors: diff_pts > WALTERS_DATA_ERROR_THRESHOLD (15), flagged and excluded from plays.
    Spread in team_a perspective: model_spread = team_a - team_b.
    """
    plays = []
    data_errors = []
    for m in r1_matchups:
        market = m.get("market_spread")
        if market is None:
            continue
        model = get_model_spread(m["team_a"], m["team_b"])
        if model is None:
            continue
        diff = abs(float(model) - float(market))
        rec = {
            "team_a": m["team_a"],
            "team_b": m["team_b"],
            "seed_a": m["seed_a"],
            "seed_b": m["seed_b"],
            "market_spread": round(float(market), 1),
            "model_spread": round(float(model), 1),
            "diff_pts": round(diff, 1),
        }
        if diff > WALTERS_DATA_ERROR_THRESHOLD:
            data_errors.append({**rec, "data_error": True})
        elif WALTERS_DELTA_MIN <= diff <= WALTERS_DELTA_MAX:
            plays.append(rec)
    plays.sort(key=lambda x: -x["diff_pts"])
    return plays[:top_n], data_errors


# --- Full analysis ---

def run_bracket_analysis(
    bracket_csv: str,
    power_rankings_csv: str,
    n_sims: int = 10_000,
    db_path: Optional[Path] = None,
    game_date: Optional[str] = None,
) -> dict:
    """
    Run full analysis. Returns {
        "value_sleepers": list of top 5 {team, seed, model_rank, value_delta},
        "final4_probabilities": list of {team, final4_pct} sorted by pct desc,
        "glitch_teams": list of {team, seed, final4_pct},
        "walters_plays": list of up to 3 {team_a, team_b, market_spread, model_spread, diff_pts},
        "errors": list of str,
    }.
    """
    errors = []
    bracket = parse_bracket_csv(bracket_csv)
    rankings = parse_power_rankings_csv(power_rankings_csv)

    if len(bracket) < 32:
        errors.append(f"Bracket has {len(bracket)} first-round matchups; expected 32.")
    if not rankings:
        errors.append("No power rankings parsed.")

    db = db_path or (Path(__file__).resolve().parent.parent / "data" / "espn.db")
    team_stats, rank_ft, rank_3p, rank_exp = load_march_stats(db)
    margin_cache = {}

    # Tier calibration: model rank and seed by canonical name; potential glitch set for anchor exception
    model_rank_by_canonical = {resolve_team_name(r["team"]): r["model_rank"] for r in rankings}
    team_to_seed = {}
    for m in bracket:
        team_to_seed[resolve_team_name(m["team_a"])] = m["seed_a"]
        team_to_seed[resolve_team_name(m["team_b"])] = m["seed_b"]
    potential_glitch = _potential_glitch_teams(bracket, rank_ft, rank_3p, rank_exp, BRACKET_TEAM_ALIAS_MAP)

    value_sleepers = []
    if bracket and rankings:
        deltas = compute_value_deltas(rankings, bracket, alias_map=BRACKET_TEAM_ALIAS_MAP)
        value_sleepers = [dict(d) for d in deltas[:5]]
        for s in value_sleepers:
            tags = _march_tags_for_team(resolve_team_name(s["team"]), team_stats, rank_ft, rank_3p)
            s["march_factors"] = ", ".join(tags) if tags else "—"

    get_winner = _build_get_winner_with_march_factors(
        db, game_date, team_stats, rank_ft, rank_3p, margin_cache, BRACKET_TEAM_ALIAS_MAP,
        model_rank_by_canonical=model_rank_by_canonical,
        team_to_seed=team_to_seed,
        potential_glitch=potential_glitch,
    )

    def get_model_spread(ta: str, tb: str) -> Optional[float]:
        raw = get_model_spread_for_matchup(ta, tb, db_path=db, game_date=game_date, alias_map=BRACKET_TEAM_ALIAS_MAP)
        if raw is None:
            return None
        ta_c = resolve_team_name(ta)
        tb_c = resolve_team_name(tb)
        return _apply_tier_and_anchor(raw, ta_c, tb_c, model_rank_by_canonical, team_to_seed, potential_glitch)

    result_mc = {}
    final4_probabilities = []
    glitch_teams = []
    if len(bracket) >= 32:
        result_mc = run_monte_carlo_bracket(bracket, get_winner, n_sims=n_sims)
        final4_probabilities = [
            {"team": t, "final4_pct": round(p, 1)}
            for t, p in sorted(result_mc["final4_pct"].items(), key=lambda x: -x[1])
        ]
        glitch_teams = get_glitch_teams(
            result_mc, bracket,
            team_stats=team_stats, rank_ft=rank_ft, rank_3p=rank_3p, rank_exp=rank_exp,
            alias_map=BRACKET_TEAM_ALIAS_MAP,
        )

    walters_plays, walters_data_errors = get_walters_plays(bracket, get_model_spread, top_n=3)

    return {
        "value_sleepers": value_sleepers,
        "final4_probabilities": final4_probabilities,
        "glitch_teams": glitch_teams,
        "walters_plays": walters_plays,
        "walters_data_errors": walters_data_errors,
        "errors": errors,
        "n_bracket_games": len(bracket),
        "n_rankings": len(rankings),
        "n_sims": n_sims,
    }
