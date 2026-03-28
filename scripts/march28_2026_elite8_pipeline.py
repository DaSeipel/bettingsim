#!/usr/bin/env python3
"""
Backfill Sweet 16 results, save full March 27 slate to historical CSV, run Elite 8 (March 28)
with --save-all-games, print Elite 8 report (picks table, tournament vs model, misprice flags).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

APP_ROOT = Path(__file__).resolve().parent.parent
HIST = APP_ROOT / "data" / "historical_betting_performance.csv"
ODDS_DIR = APP_ROOT / "data" / "odds"
MARCH27 = ODDS_DIR / "March_27_2026_Odds.csv"
MARCH28 = ODDS_DIR / "March_28_2026_Odds.csv"

# Sweet 16 final margins (home score - away score), keys = (Home, Away) as stored after predict (matched names)
SWEET16_ACTUAL = {
    ("Purdue", "Texas"): 2.0,
    ("Nebraska", "Iowa"): -6.0,
    ("Arizona", "Arkansas"): 21.0,
    ("Houston", "Illinois"): -10.0,
    ("DUKE", "St. John's"): 5.0,
    ("Michigan", "Alabama"): 13.0,
    ("CONN", "Michigan St."): 4.0,
    ("Iowa St.", "Tennessee"): -14.0,
}


def _norm_team(s: str) -> str:
    return str(s or "").strip()


def ats_result(pick: str, actual_margin: float, market_spread: float) -> str:
    if pick == "Skip" or pick is None or str(pick).strip() == "":
        return ""
    try:
        m = float(actual_margin)
        s = float(market_spread)
    except (TypeError, ValueError):
        return ""
    home_covers = m > (-s)
    if pick == "Home":
        return "Win" if home_covers else "Loss"
    if pick == "Away":
        return "Win" if not home_covers else "Loss"
    return ""


def apply_sweet16_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Date" not in out.columns:
        return out
    mask = out["Date"].astype(str).str.strip() == "2026-03-27"
    for idx in out.index[mask]:
        h = _norm_team(out.at[idx, "Home"])
        a = _norm_team(out.at[idx, "Away"])
        key = (h, a)
        if key not in SWEET16_ACTUAL:
            for alt in SWEET16_ACTUAL:
                if alt[0].lower() == h.lower() and alt[1].lower() == a.lower():
                    key = alt
                    break
        if key not in SWEET16_ACTUAL:
            continue
        am = SWEET16_ACTUAL[key]
        out.at[idx, "Actual_Margin"] = am
        pick = str(out.at[idx, "Pick_Spread"]).strip()
        ms = out.at[idx, "Market_Spread"]
        out.at[idx, "ATS_Result"] = ats_result(pick, am, float(ms)) if pd.notna(ms) else ""
    return out


def run_predict_save_all() -> None:
    env = {**os.environ, "PYTHONPATH": str(APP_ROOT)}
    r = subprocess.run(
        [sys.executable, str(APP_ROOT / "scripts" / "predict_games.py"), "--save-all-games"],
        cwd=str(APP_ROOT),
        env=env,
    )
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def touch(path: Path) -> None:
    path.touch(exist_ok=True)


def surprise_vs_model(row: pd.Series, team: str) -> float | None:
    """Points team outperformed model-expected margin (positive = hot)."""
    team = _norm_team(team)
    h = _norm_team(row.get("Home"))
    a = _norm_team(row.get("Away"))
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
    if team.lower() in h.lower() or h.lower() in team.lower():
        return act - pred
    if team.lower() in a.lower() or a.lower() in team.lower():
        return pred - act
    return None


def print_elite8_report(elite8_df: pd.DataFrame, hist: pd.DataFrame) -> None:
    elite_teams = {
        "Illinois", "Iowa", "Arizona", "Purdue", "Michigan", "Tennessee", "DUKE", "Duke", "UConn", "CONN",
    }
    # Normalize elite list for matching
    def is_elite_side(name: str) -> bool:
        n = name.strip()
        if n in ("DUKE", "Duke"):
            return True
        if n in ("CONN", "UConn"):
            return True
        return n in ("Illinois", "Iowa", "Arizona", "Purdue", "Michigan", "Tennessee")

    hhist = hist.copy()
    hhist["Date"] = hhist["Date"].astype(str).str.strip()
    tourney = hhist[
        (hhist["Date"] >= "2026-03-18")
        & (hhist["Date"] <= "2026-03-28")
        & hhist["Actual_Margin"].notna()
    ].copy()

    print("\n" + "=" * 100)
    print("ELITE 8 — SPREAD PICKS (every game has a side; sorted by |edge|)")
    print("  Model_Spread = model projected margin (home − away). Edge = vs closing spread.")
    print("=" * 100)
    rows = []
    for _, r in elite8_df.iterrows():
        game = f"{r['Away']} @ {r['Home']}"
        pm = r.get("Pred_Margin")
        ms = r.get("Market_Spread")
        edge = r.get("Edge_Points")
        if pd.isna(edge) and "Edge" in r:
            edge = r.get("Edge")
        pick = str(r.get("Pick_Spread", "Skip")).strip()
        conf = r.get("Confidence_Level", r.get("Confidence", ""))
        try:
            edge_f = float(edge) if pd.notna(edge) else 0.0
        except (TypeError, ValueError):
            edge_f = 0.0
        # No skips: lean Home if edge > 0 else Away when model would not bet threshold
        if pick == "Skip":
            every_pick = "Home" if edge_f > 0 else "Away"
            pick_display = f"{every_pick} (lean)"
        else:
            every_pick = pick
            pick_display = pick
        rows.append(
            {
                "Game": game,
                "Model_Spread": f"{float(pm):+.1f}" if pd.notna(pm) else "—",
                "Market_Spread": f"{float(ms):+.1f}" if pd.notna(ms) else "—",
                "Edge": f"{edge_f:+.1f}" if pd.notna(edge) else "—",
                "Pick": pick_display,
                "Confidence": conf,
                "_abs_edge": abs(edge_f),
            }
        )
    rows.sort(key=lambda x: -x["_abs_edge"])
    hdr = f"  {'Game':<32}  {'Model':>8}  {'Mkt':>8}  {'Edge':>8}  {'Pick':<14}  Conf"
    print(hdr)
    print("  " + "-" * 95)
    for x in rows:
        print(
            f"  {x['Game']:<32}  {x['Model_Spread']:>8}  {x['Market_Spread']:>8}  "
            f"{x['Edge']:>8}  {x['Pick']:<14}  {x['Confidence']}"
        )

    print("\n" + "=" * 100)
    print("ELITE 8 TEAMS — TOURNAMENT vs MODEL (completed games with Actual_Margin in CSV)")
    print("  Surprise = actual vs model-expected margin for that team (positive = running hot).")
    print("=" * 100)

    teams_order = ["Illinois", "Iowa", "Arizona", "Purdue", "Michigan", "Tennessee", "DUKE", "CONN"]
    for t in teams_order:
        surprises = []
        for _, row in tourney.iterrows():
            h = _norm_team(row["Home"])
            a = _norm_team(row["Away"])
            if not (is_elite_side(t) or t in h or t in a or h in t or t in ("UConn",) and "Conn" in h):
                pass
            # match team to row
            matched = False
            if t in ("DUKE", "Duke") and "DUKE" in h or "Duke" == t and h == "DUKE":
                matched = True
            elif t == "CONN" and h == "CONN":
                matched = True
            elif t == "Illinois" and (h == "Illinois" or a == "Illinois"):
                matched = True
            elif t == "Iowa" and (h == "Iowa" or a == "Iowa"):
                matched = True
            elif t == "Arizona" and (h == "Arizona" or a == "Arizona"):
                matched = True
            elif t == "Purdue" and (h == "Purdue" or a == "Purdue"):
                matched = True
            elif t == "Michigan" and (h == "Michigan" or a == "Michigan"):
                matched = True
            elif t == "Tennessee" and (h == "Tennessee" or a == "Tennessee"):
                matched = True
            elif t == "CONN" and ("CONN" in h or "UConn" in h or a == "Michigan St."):
                matched = h == "CONN" or "Conn" in str(row["Home"])
            if not matched:
                if t == "Illinois" and ("Illinois" == h or "Illinois" == a):
                    matched = True
                elif t == "Iowa" and ("Iowa" == h or "Iowa" == a):
                    matched = True
                elif t == "Arizona" and ("Arizona" == h or "Arizona" == a):
                    matched = True
                elif t == "Purdue" and ("Purdue" == h or "Purdue" == a):
                    matched = True
                elif t == "Michigan" and ("Michigan" == h or "Michigan" == a):
                    matched = True
                elif t == "Tennessee" and ("Tennessee" == h or "Tennessee" == a):
                    matched = True
                elif t in ("DUKE",) and h == "DUKE":
                    matched = True
                elif t == "CONN" and h == "CONN":
                    matched = True
            if not matched:
                continue
            label = "DUKE" if t in ("DUKE", "Duke") else ("CONN" if t == "CONN" else t)
            sup = surprise_vs_model(row, label)
            if sup is not None:
                surprises.append((str(row["Date"]), f"{row['Away']} @ {row['Home']}", sup))
        if not surprises:
            print(f"  {t}: no graded games found in CSV")
            continue
        avg = sum(s[2] for s in surprises) / len(surprises)
        tag = "hot vs model" if avg > 2 else ("cold vs model" if avg < -2 else "near model")
        print(f"  {t}: n={len(surprises)}  avg surprise {avg:+.1f} pts  → {tag}")
        for d, g, s in surprises:
            print(f"      {d}  {g}: {s:+.1f}")

    print("\n" + "=" * 100)
    print("MISPRICE FLAGS (cumulative tournament surprise vs current Elite 8 market)")
    print("=" * 100)
    avg_by: dict[str, float] = {}
    for t in teams_order:
        sups = []
        for _, row in tourney.iterrows():
            h, a = _norm_team(row["Home"]), _norm_team(row["Away"])
            label = None
            if t in ("DUKE",) and h == "DUKE":
                label = "DUKE"
            elif t == "CONN" and h == "CONN":
                label = "CONN"
            elif t == "Illinois" and (t == h or t == a):
                label = "Illinois"
            elif t == "Iowa" and (t == h or t == a):
                label = "Iowa"
            elif t == "Arizona" and (t == h or t == a):
                label = "Arizona"
            elif t == "Purdue" and (t == h or t == a):
                label = "Purdue"
            elif t == "Michigan" and (t == h or t == a):
                label = "Michigan"
            elif t == "Tennessee" and (t == h or t == a):
                label = "Tennessee"
            if label is None:
                continue
            sup = surprise_vs_model(row, label)
            if sup is not None:
                sups.append(sup)
        if sups:
            avg_by[t] = sum(sups) / len(sups)
    for _, r in elite8_df.iterrows():
        h, a = _norm_team(r["Home"]), _norm_team(r["Away"])
        ms = float(r["Market_Spread"]) if pd.notna(r.get("Market_Spread")) else None
        edge = float(r["Edge_Points"]) if pd.notna(r.get("Edge_Points")) else None
        if edge is None and pd.notna(r.get("Edge")):
            edge = float(r["Edge"])
        notes = []
        for side, name in [("home", h), ("away", a)]:
            key = "DUKE" if name == "DUKE" else ("CONN" if name == "CONN" else name)
            for t in ("Illinois", "Iowa", "Arizona", "Purdue", "Michigan", "Tennessee", "DUKE", "CONN"):
                if key != t and not (t == "DUKE" and name == "DUKE"):
                    continue
                avg = avg_by.get(t)
                if avg is None:
                    continue
                if avg > 3.5:
                    notes.append(f"{name} avg +{avg:.1f} vs model (market may be short)")
                elif avg < -3.5:
                    notes.append(f"{name} avg {avg:.1f} vs model (market may be high)")
        if notes:
            print(f"  {a} @ {h}  (mkt {ms:+.1f}):  " + "  |  ".join(notes))
        elif ms is not None and edge is not None and abs(edge) >= 5:
            print(f"  {a} @ {h}: large model edge ({edge:+.1f}) — review vs injury/rest narrative.")
    print("=" * 100)
    print("\nHistorical CSV: rows with |Edge_Points| >= 3 for 2026-03-28 are saved for tracking (full slate in --save-all-games).")
    print()


def main() -> None:
    os.chdir(APP_ROOT)
    sys.path.insert(0, str(APP_ROOT))

    print("Step 1: Re-save full Sweet 16 slate (March 27) to historical CSV...")
    touch(MARCH27)
    run_predict_save_all()

    print("Step 2: Apply Sweet 16 final scores and ATS...")
    df = pd.read_csv(HIST)
    df = apply_sweet16_outcomes(df)
    df.to_csv(HIST, index=False)
    print(f"  Updated {HIST.relative_to(APP_ROOT)}")

    print("Step 3: Run Elite 8 model (March 28 odds)...")
    touch(MARCH28)
    run_predict_save_all()

    print("Step 4: Elite 8 report...")
    df_all = pd.read_csv(HIST)
    elite8 = df_all[df_all["Date"].astype(str).str.strip() == "2026-03-28"].copy()
    if elite8.empty:
        print("  No 2026-03-28 rows found in historical CSV.")
        return
    print_elite8_report(elite8, df_all)


if __name__ == "__main__":
    main()
