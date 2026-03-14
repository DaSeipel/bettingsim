#!/usr/bin/env python3
"""
Run the full daily pipeline: fetch stats, update games/form, optionally retrain models,
check for today's odds, run predictions, and print a summary.

Steps:
  1. fetch_barttorvik_2026.py  — latest team stats (continue on failure)
  2. fetch_historical_games.py — update game results (continue on failure)
  3. compute_current_form.py   — update form stats for 2026 (continue on failure)
  [if --retrain] build_training_data.py, train_spread_model.py, train_totals_model.py
  4. Check data/odds/ for today's file: Month_DD_YYYY_Odds.csv (e.g. March_10_2026_Odds.csv)
  5. predict_games.py — generate predictions
  6. Print summary: value plays count, top 2 POTD, warnings

Flags:
  --retrain       Run build_training_data, train_spread_model, train_totals_model between steps 3 and 4.
  --date YYYY-MM-DD  Override date for odds-file check (e.g. for backtesting a specific day).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
ODDS_DIR = APP_ROOT / "data" / "odds"
CACHE_PATH = APP_ROOT / "data" / "cache" / "value_plays_cache.json"

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _odds_filename_for_date(d: date) -> str:
    """Return Month_DD_YYYY_Odds.csv for the given date (e.g. March_10_2026_Odds.csv)."""
    month = MONTH_NAMES[d.month - 1]
    return f"{month}_{d.day:02d}_{d.year}_Odds.csv"


def _run_step(name: str, script: str, cwd: Path, continue_on_fail: bool = True) -> bool:
    """Run a script via subprocess. Return True if success, False otherwise. Print timing and errors."""
    script_path = APP_ROOT / "scripts" / script
    if not script_path.exists():
        print(f"  [SKIP] {name}: script not found ({script})")
        return False
    start = time.perf_counter()
    env = {**os.environ, "PYTHONPATH": str(cwd)}
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        elapsed = time.perf_counter() - start
        if result.returncode != 0:
            print(f"  [FAIL] {name} ({elapsed:.1f}s)")
            if result.stderr:
                print(result.stderr.rstrip())
            if result.stdout and result.returncode != 0:
                for line in result.stdout.strip().splitlines()[-10:]:
                    print(line)
            if not continue_on_fail:
                sys.exit(1)
            return False
        print(f"  [OK]   {name} ({elapsed:.1f}s)")
        return True
    except subprocess.TimeoutExpired:
        print(f"  [FAIL] {name} (timeout)")
        if not continue_on_fail:
            sys.exit(1)
        return False
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        if not continue_on_fail:
            sys.exit(1)
        return False


def _load_cache() -> dict | None:
    """Load value_plays_cache.json if it exists."""
    if not CACHE_PATH.exists():
        return None
    try:
        with open(CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full daily pipeline")
    parser.add_argument("--retrain", action="store_true", help="Run build_training_data, train_spread_model, train_totals_model")
    parser.add_argument("--date", type=str, metavar="YYYY-MM-DD", help="Override date for odds check (e.g. for backtesting)")
    args = parser.parse_args()

    if args.date:
        try:
            run_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print(f"Invalid --date: {args.date}. Use YYYY-MM-DD.", file=sys.stderr)
            return 1
    else:
        run_date = date.today()

    odds_filename = _odds_filename_for_date(run_date)
    odds_path = ODDS_DIR / odds_filename

    print("=" * 60)
    print(f"Daily pipeline — date: {run_date}  odds file: {odds_filename}")
    print("=" * 60)

    # 1. Fetch Barttorvik 2026
    _run_step("fetch_barttorvik_2026", "fetch_barttorvik_2026.py", APP_ROOT, continue_on_fail=True)

    # 2. Fetch historical games
    _run_step("fetch_historical_games", "fetch_historical_games.py", APP_ROOT, continue_on_fail=True)

    # 3. Compute current form
    _run_step("compute_current_form", "compute_current_form.py", APP_ROOT, continue_on_fail=True)

    # Optional retrain
    if args.retrain:
        print("\n--- Retrain (build_training_data + train_spread_model + train_totals_model) ---")
        _run_step("build_training_data", "build_training_data.py", APP_ROOT, continue_on_fail=True)
        _run_step("train_spread_model", "train_spread_model.py", APP_ROOT, continue_on_fail=True)
        _run_step("train_totals_model", "train_totals_model.py", APP_ROOT, continue_on_fail=True)

    # 4. Check odds file for today
    if not ODDS_DIR.exists():
        print("\nNo odds file for today — skipping predictions")
        print(f"  (directory data/odds/ missing; expected {odds_filename})")
        return 0
    if not odds_path.exists():
        print("\nNo odds file for today — skipping predictions")
        print(f"  (expected {odds_path.relative_to(APP_ROOT)})")
        return 0

    # 5. Run predictions (uses most recent CSV in data/odds/ by mtime; ensure desired file is latest if backtesting)
    print("\n--- Predictions ---")
    ok = _run_step("predict_games", "predict_games.py", APP_ROOT, continue_on_fail=False)
    if not ok:
        return 1

    # 6. Summary
    cache = _load_cache()
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    warnings = []
    n_spread = 0
    n_totals = 0
    if cache:
        value_plays = cache.get("value_plays") or []
        totals_plays = cache.get("totals_plays") or []
        n_spread = len(value_plays)
        n_totals = len(totals_plays)
        print(f"  Value plays (spread): {n_spread}")
        print(f"  Value plays (totals): {n_totals}")

        potd = cache.get("potd_picks") or {}
        for key in ("NCAAB Pick 1", "NCAAB Pick 2"):
            p = potd.get(key)
            if p:
                event = p.get("Event", "—")
                selection = p.get("Selection", "—")
                point = p.get("point", p.get("Point"))
                pt_str = f" {point:+.1f}" if point is not None else ""
                print(f"  {key}: {selection}{pt_str} — {event}")
            else:
                print(f"  {key}: —")
        for key in ("NCAAB Totals Pick 1", "NCAAB Totals Pick 2"):
            p = potd.get(key)
            if p:
                event = p.get("Event", "—")
                selection = p.get("Selection", "—")
                line = p.get("point", p.get("Over_Under"))
                ln_str = f" {line}" if line is not None else ""
                print(f"  {key}: {selection}{ln_str} — {event}")
    else:
        warnings.append("Could not load value_plays_cache.json")

    if n_spread == 0 and n_totals == 0 and cache:
        warnings.append("No value plays found for today")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
