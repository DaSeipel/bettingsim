#!/usr/bin/env python3
"""Run the daily MLB pipeline end-to-end with fail-fast behavior."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "data" / "cache" / "mlb_value_plays.json"

STEPS: list[tuple[str, list[str]]] = [
    ("Fetching odds", ["scripts/fetch_mlb_odds.py", "--force"]),
    ("Fetching weather", ["scripts/fetch_mlb_weather.py"]),
    ("Fetching pitchers", ["scripts/fetch_mlb_pitchers.py"]),
    ("Fetching team stats", ["scripts/fetch_mlb_stats.py", "--season-blend"]),
    ("Fetching recent form", ["scripts/fetch_mlb_recent_form.py"]),
    ("Predicting MLB picks", ["scripts/predict_mlb.py"]),
]


def _print_table(rows: list[dict]) -> None:
    columns = ["Pick", "Odds", "Model Prob", "Edge %", "Type"]
    data = []

    for play in rows:
        selection = str(play.get("selection", ""))
        odds = play.get("odds_american")
        model_prob = play.get("model_prob")
        edge = play.get("edge")
        market = str(play.get("market", "")).lower()

        play_type = "ML" if market == "moneyline" else "Total"
        odds_text = f"{int(odds):+d}" if isinstance(odds, (int, float)) else "-"
        prob_text = f"{float(model_prob) * 100:.1f}%" if isinstance(model_prob, (int, float)) else "-"
        edge_text = f"{float(edge) * 100:.2f}%" if isinstance(edge, (int, float)) else "-"

        data.append([selection, odds_text, prob_text, edge_text, play_type])

    widths = []
    for idx, col in enumerate(columns):
        max_cell = max((len(str(row[idx])) for row in data), default=0)
        widths.append(max(len(col), max_cell))

    def fmt_row(items: list[str]) -> str:
        return " | ".join(str(items[i]).ljust(widths[i]) for i in range(len(items)))

    divider = "-+-".join("-" * width for width in widths)

    print()
    print("Today's MLB Picks")
    print(fmt_row(columns))
    print(divider)
    for row in data:
        print(fmt_row(row))


def _print_summary() -> None:
    if not CACHE_PATH.exists():
        print(f"\nCould not find cache file: {CACHE_PATH}")
        return

    try:
        payload = json.loads(CACHE_PATH.read_text())
    except Exception as exc:
        print(f"\nFailed to read picks cache ({CACHE_PATH}): {exc}")
        return

    plays = payload.get("plays", [])
    if not isinstance(plays, list) or not plays:
        print("\nNo picks found in data/cache/mlb_value_plays.json")
        return

    _print_table(plays)


def main() -> int:
    total = len(STEPS)
    for idx, (title, args) in enumerate(STEPS, start=1):
        print(f"\n=== Step {idx}/{total}: {title} ===")
        cmd = [sys.executable, *args]

        try:
            subprocess.run(cmd, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"\nERROR: Step {idx}/{total} failed with exit code {exc.returncode}")
            print(f"Command: {' '.join(cmd)}")
            return exc.returncode or 1
        except Exception as exc:
            print(f"\nERROR: Step {idx}/{total} failed before execution completed: {exc}")
            print(f"Command: {' '.join(cmd)}")
            return 1

    _print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
