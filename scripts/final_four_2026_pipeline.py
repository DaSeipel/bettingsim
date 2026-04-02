#!/usr/bin/env python3
"""
Backfill Elite 8 results, sync SQLite, run Final Four odds (neutral), print reports:
picks table, cumulative tournament vs model, bracket sim comparison, Walters breakdown.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "data" / "historical_betting_performance.csv"
DB = ROOT / "data" / "espn.db"
BRACKET_JSON = ROOT / "data" / "ncaab" / "bracket_filled_2026.json"
ODDS_F4 = ROOT / "data" / "odds" / "April_05_2026_Odds.csv"
GAME_DATE_F4 = "2026-04-05"

ELITE8_ACTUAL: dict[tuple[str, str], float] = {
    ("Illinois", "Iowa"): 12.0,
    ("Arizona", "Purdue"): 15.0,
    ("Michigan", "Tennessee"): 33.0,
    ("DUKE", "CONN"): -1.0,
}


def _norm(s: str) -> str:
    return str(s or "").strip()


def ats_result(pick: str, actual_margin: float, market_spread: float) -> str:
    pick = str(pick).strip()
    if pick not in ("Home", "Away"):
        return ""
    try:
        m = float(actual_margin)
        s = float(market_spread)
    except (TypeError, ValueError):
        return ""
    home_covers = m > (-s)
    if pick == "Home":
        return "Win" if home_covers else "Loss"
    return "Win" if not home_covers else "Loss"


def apply_elite8_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mask = out["Date"].astype(str).str.strip() == "2026-03-28"
    for idx in out.index[mask]:
        h, a = _norm(out.at[idx, "Home"]), _norm(out.at[idx, "Away"])
        key = (h, a)
        if key not in ELITE8_ACTUAL:
            continue
        am = ELITE8_ACTUAL[key]
        out.at[idx, "Actual_Margin"] = am
        pick = str(out.at[idx, "Pick_Spread"]).strip()
        ms = out.at[idx, "Market_Spread"]
        out.at[idx, "ATS_Result"] = (
            ats_result(pick, am, float(ms)) if pick in ("Home", "Away") and pd.notna(ms) else ""
        )
    return out


def surprise_vs_model(row: pd.Series, team: str) -> float | None:
    team = _norm(team)
    h, a = _norm(row.get("Home")), _norm(row.get("Away"))
    try:
        pred = float(row["Pred_Margin"])
        act = float(row["Actual_Margin"])
    except (TypeError, ValueError, KeyError):
        return None
    if pd.isna(row.get("Actual_Margin")):
        return None
    if team == h:
        return act - pred
    if team == a:
        return pred - act
    return None


def team_margin_expected(row: pd.Series, team: str) -> float | None:
    h, a = _norm(row.get("Home")), _norm(row.get("Away"))
    try:
        pred = float(row["Pred_Margin"])
    except (TypeError, ValueError, KeyError):
        return None
    t = _norm(team)
    if t == h:
        return pred
    if t == a:
        return -pred
    return None


def team_margin_actual(row: pd.Series, team: str) -> float | None:
    h, a = _norm(row.get("Home")), _norm(row.get("Away"))
    try:
        act = float(row["Actual_Margin"])
    except (TypeError, ValueError, KeyError):
        return None
    if pd.isna(row.get("Actual_Margin")):
        return None
    t = _norm(team)
    if t == h:
        return act
    if t == a:
        return -act
    return None


def sync_sqlite_elite8() -> None:
    """Set result W/L and payout for 2026-03-28 spread plays from CSV rows."""
    if not DB.exists() or not HIST.exists():
        return
    df = pd.read_csv(HIST)
    sub = df[df["Date"].astype(str).str.strip() == "2026-03-28"]
    conn = sqlite3.connect(DB)
    try:
        for _, row in sub.iterrows():
            pick = str(row.get("Pick_Spread", "")).strip()
            if pick not in ("Home", "Away"):
                continue
            h, a = _norm(row["Home"]), _norm(row["Away"])
            ms = float(row["Market_Spread"])
            if pick == "Home":
                side = h
                line = ms
            else:
                side = a
                line = abs(ms)
            ats = str(row.get("ATS_Result", "")).strip()
            res = {"Win": "W", "Loss": "L", "Push": "P"}.get(ats)
            if res is None:
                continue
            stake = 1.0
            try:
                sp = row.get("Suggested_Stake_Pct")
                if sp is not None and not pd.isna(sp):
                    stake = float(sp)
            except (TypeError, ValueError):
                pass
            if res == "W":
                payout = round(stake * 0.909, 2)
            elif res == "L":
                payout = round(-stake, 2)
            else:
                payout = 0.0
            conn.execute(
                """
                UPDATE play_history
                SET result = ?, actual_payout = ?
                WHERE date_generated = '2026-03-28'
                  AND home_team = ? AND away_team = ?
                  AND recommended_side = ? AND ABS(spread_or_total - ?) < 0.01
                """,
                (res, payout, h, a, side, line),
            )
        conn.commit()
    finally:
        conn.close()


def write_final_four_odds() -> None:
    ODDS_F4.parent.mkdir(parents=True, exist_ok=True)
    ODDS_F4.write_text(
        "Home_Team,Away_Team,Spread,Over_Under,Time,Is_Neutral\n"
        "Illinois,CONN,-1.5,145.5,6:09 PM ET,True\n"
        "Michigan,Arizona,-1.5,147.5,8:49 PM ET,True\n",
        encoding="utf-8",
    )


def run_predict_save_all() -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "predict_games.py"), "--save-all-games"],
        cwd=str(ROOT),
        env=env,
    )
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def trim_historical_f4_min_edge(min_edge: float = 3.0) -> None:
    """Keep only 2026-04-05 rows with |Edge_Points| >= min_edge (per tracking request)."""
    df = pd.read_csv(HIST)
    d = df["Date"].astype(str).str.strip()
    mask_date = d == GAME_DATE_F4
    if not mask_date.any():
        return
    def keep_row(i: int) -> bool:
        if not mask_date.iloc[i]:
            return True
        try:
            e = abs(float(df.at[i, "Edge_Points"]))
        except (TypeError, ValueError):
            return False
        return e >= min_edge

    keep_idx = [i for i in df.index if keep_row(i)]
    df.iloc[keep_idx].to_csv(HIST, index=False)


def print_walters_pair(home: str, away: str, label: str) -> None:
    sys.path.insert(0, str(ROOT))
    from engine.bracket_analysis import (
        BRACKET_TEAM_ALIAS_MAP,
        _potential_glitch_teams,
        get_matchup_march_factor_breakdown,
        load_march_stats,
        parse_bracket_csv,
        parse_power_rankings_csv,
        resolve_team_name,
    )

    bracket_path = ROOT / "data" / "ncaab" / "bracket_2026.csv"
    rankings_path = ROOT / "data" / "ncaab" / "power_rankings_2026.csv"
    db_path = ROOT / "data" / "espn.db"
    if not all(p.exists() for p in (bracket_path, rankings_path, db_path)):
        print(f"  [{label}] Missing bracket/rankings/db for Walters breakdown.\n")
        return

    bracket = parse_bracket_csv(bracket_path.read_text())
    rankings = parse_power_rankings_csv(rankings_path.read_text())
    team_stats, rank_ft, rank_3p, rank_exp = load_march_stats(db_path)
    model_rank_by_canonical = {
        resolve_team_name(r["team"], BRACKET_TEAM_ALIAS_MAP): r["model_rank"] for r in rankings
    }
    team_to_seed: dict[str, int] = {}
    for m in bracket:
        team_to_seed[resolve_team_name(m["team_a"], BRACKET_TEAM_ALIAS_MAP)] = m["seed_a"]
        team_to_seed[resolve_team_name(m["team_b"], BRACKET_TEAM_ALIAS_MAP)] = m["seed_b"]
    potential_glitch = _potential_glitch_teams(
        bracket, rank_ft, rank_3p, rank_exp, BRACKET_TEAM_ALIAS_MAP
    )

    # team_a = home perspective in CSV (first team in user phrasing "Illinois vs UConn")
    team_a, team_b = home, away
    b = get_matchup_march_factor_breakdown(
        team_a,
        team_b,
        db_path,
        GAME_DATE_F4,
        team_stats,
        rank_ft,
        rank_3p,
        model_rank_by_canonical,
        team_to_seed,
        potential_glitch,
        BRACKET_TEAM_ALIAS_MAP,
    )
    print("=" * 72)
    print(f"  WALTERS FACTOR BREAKDOWN — {label}: {away} @ {home}")
    print(f"  (Margins from {team_a} minus {team_b}; positive = {team_a} favored.)")
    print("=" * 72)
    if b.get("raw_margin") is None:
        print("  (No model margin — feature row unavailable)\n")
        return
    print(f"  Raw model margin:           {b['raw_margin']:+.2f} pts")
    print(f"  Adjusted (tier/anchor):     {b['adjusted_margin']:+.2f} pts")
    print(f"  Veteran Edge (pts):         {b['veteran_edge_pts']}")
    print(f"  Closer Factor:              {b['closer_factor']}")
    print(
        f"  3P Variance: elite {team_a}={b['elite_3p_team_a']}, {team_b}={b['elite_3p_team_b']}; "
        f"sigma pts = {b['three_p_sigma_pts']}"
    )
    ra, rb, rd = b["power_rank_a"], b["power_rank_b"], b["power_rank_delta"]
    print(f"  Power ranks: {team_a}={ra}, {team_b}={rb}, delta (a−b) = {rd}")
    print()


def main() -> int:
    os.chdir(ROOT)
    print("Step 1: Elite 8 → historical CSV (Actual_Margin, ATS_Result)...")
    df = pd.read_csv(HIST)
    df = apply_elite8_outcomes(df)
    df.to_csv(HIST, index=False)

    print("Step 2: SQLite play_history results for 2026-03-28...")
    sync_sqlite_elite8()

    print("Step 3: Write Final Four odds →", ODDS_F4.relative_to(ROOT))
    write_final_four_odds()

    print("Step 4: Run predict_games.py --save-all-games...", flush=True)
    run_predict_save_all()

    df_all = pd.read_csv(HIST)
    f4 = df_all[df_all["Date"].astype(str).str.strip() == GAME_DATE_F4].copy()

    print("Step 5: Trim 2026-04-05 historical rows to |edge| >= 3 only...", flush=True)
    trim_historical_f4_min_edge(3.0)

    df_all = pd.read_csv(HIST)
    tourney = df_all[
        (df_all["Date"].astype(str).str.strip() >= "2026-03-20")
        & (df_all["Actual_Margin"].notna())
    ].copy()

    teams_f4 = ["Illinois", "CONN", "Michigan", "Arizona"]

    # --- Picks table (no skips): lean from edge sign ---
    print("\n" + "=" * 100)
    print("FINAL FOUR — SPREAD PICKS (sorted by |edge|)")
    print("=" * 100)
    rows = []
    for _, r in f4.iterrows():
        game = f"{r['Away']} @ {r['Home']}"
        pm, ms, edge = r.get("Pred_Margin"), r.get("Market_Spread"), r.get("Edge_Points")
        conf = r.get("Confidence_Level", "")
        try:
            edge_f = float(edge) if pd.notna(edge) else 0.0
        except (TypeError, ValueError):
            edge_f = 0.0
        pick0 = str(r.get("Pick_Spread", "Skip")).strip()
        if pick0 != "Skip":
            pick_disp = pick0
            side_name = r["Home"] if pick0 == "Home" else r["Away"]
        else:
            lean = "Home" if edge_f > 0 else "Away"
            pick_disp = f"{lean} (lean)"
            side_name = r["Home"] if lean == "Home" else r["Away"]
        rows.append(
            {
                "Game": game,
                "Model_Spread": f"{float(pm):+.1f}" if pd.notna(pm) else "—",
                "Market_Spread": f"{float(ms):+.1f}" if pd.notna(ms) else "—",
                "Edge": f"{edge_f:+.1f}",
                "Pick": f"{side_name} ({pick_disp})",
                "Confidence": conf,
                "_ae": abs(edge_f),
            }
        )
    rows.sort(key=lambda x: -x["_ae"])
    print(f"  {'Game':<28} {'Model':>8} {'Mkt':>8} {'Edge':>8} {'Pick':<28} {'Conf':<8}")
    print("  " + "-" * 96)
    for x in rows:
        print(
            f"  {x['Game']:<28} {x['Model_Spread']:>8} {x['Market_Spread']:>8} {x['Edge']:>8} "
            f"  {x['Pick']:<28} {x['Confidence']}"
        )

    # --- Cumulative tournament performance ---
    print("\n" + "=" * 100)
    print("FINAL FOUR TEAMS — CUMULATIVE TOURNAMENT vs MODEL (2026-03-20+, graded rows)")
    print("  Actual Σ = sum of game margins from team perspective; Pred Σ = sum of model margins (team POV).")
    print("=" * 100)
    for team in teams_f4:
        exp_l, act_l, surpr = [], [], []
        for _, row in tourney.iterrows():
            h, a = _norm(row["Home"]), _norm(row["Away"])
            if team not in (h, a):
                continue
            e = team_margin_expected(row, team)
            u = team_margin_actual(row, team)
            s = surprise_vs_model(row, team)
            if e is None or u is None or s is None:
                continue
            exp_l.append(e)
            act_l.append(u)
            surpr.append(s)
        if not exp_l:
            print(f"  {team}: no graded games found.")
            continue
        sum_e, sum_a, cum_surp = sum(exp_l), sum(act_l), sum(surpr)
        tag = (
            "running hot vs model"
            if cum_surp > 8
            else ("running cold vs model" if cum_surp < -8 else "near model expectations")
        )
        print(
            f"  {team}: games={len(exp_l)}  |  Σ actual margin {sum_a:+.1f}  vs  Σ model {sum_e:+.1f} "
            f"(cumulative surprise {cum_surp:+.1f}) → {tag}"
        )
        for _, row in tourney.iterrows():
            h, a = _norm(row["Home"]), _norm(row["Away"])
            if team not in (h, a):
                continue
            s = surprise_vs_model(row, team)
            if s is None:
                continue
            print(
                f"      {row['Date']} {a} @ {h}: actual {team_margin_actual(row, team):+.1f} vs "
                f"model {team_margin_expected(row, team):+.1f} (surprise {s:+.1f})"
            )

    # --- Bracket sim ---
    print("\n" + "=" * 100)
    print("BRACKET SIM (data/ncaab/bracket_filled_2026.json) vs MODEL PICK")
    print("=" * 100)
    if BRACKET_JSON.exists():
        blob = json.loads(BRACKET_JSON.read_text())
        r5 = [g for g in blob.get("games", []) if g.get("round") == 5]
        r5.sort(key=lambda g: g.get("game_idx", 0))
        print("  Original National Semifinals from bracket sim:")
        for g in r5:
            print(
                f"    {g['team_a']} vs {g['team_b']}  → bracket pick: {g['pick']} "
                f"({g['win_pct_a']:.1f}% / {g['win_pct_b']:.1f}%)"
            )
        semi2 = r5[1] if len(r5) > 1 else {}
        semi1 = r5[0] if r5 else {}
        print("\n  Comparison to this slate:")
        print(
            f"    Michigan vs Arizona: bracket sim picked **{semi2.get('pick', '?')}** for slot "
            f"({semi2.get('team_a')} vs {semi2.get('team_b')})."
        )
        for x in rows:
            if "Michigan" in x["Game"] and "Arizona" in x["Game"]:
                brp = str(semi2.get("pick", ""))
                match = "aligns with" if brp and brp in x["Pick"] else "differs from"
                print(
                    f"      Daily model pick: {x['Pick']} — {match} bracket winner **{brp}**."
                )
        print(
            f"    Illinois vs CONN: bracket sim first semi was **{semi1.get('team_a')} vs {semi1.get('team_b')}** "
            f"(pick {semi1.get('pick')}); actual matchup is Illinois vs UConn — no direct bracket mapping."
        )
        for x in rows:
            if "CONN" in x["Game"] or "Illinois" in x["Game"]:
                if "@" in x["Game"]:
                    print(f"      Daily spread lean: {x['Pick']} (edge {x['Edge']} pts).")

    print_walters_pair("Illinois", "CONN", "Semifinal 1")
    print_walters_pair("Michigan", "Arizona", "Semifinal 2")

    kept = df_all[df_all["Date"].astype(str).str.strip() == GAME_DATE_F4]
    print("=" * 100)
    print(
        f"Done. {GAME_DATE_F4}: {len(kept)} row(s) in historical CSV with |Edge_Points| >= 3 "
        f"(of {len(f4)} games modeled)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
