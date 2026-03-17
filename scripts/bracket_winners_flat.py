#!/usr/bin/env python3
"""
Print the filled bracket as a flat list of winners only, by round, in bracket order.
For pasting into ESPN bracket challenge.
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
ROUND_LABELS = {
    1: "Round of 64",
    2: "Round of 32",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final Four",
    6: "Championship",
}


def main() -> int:
    path = ROOT / "data" / "ncaab" / "bracket_filled_2026.json"
    if not path.exists():
        print("Run scripts/fill_bracket_2026.py first.")
        return 1
    with open(path) as f:
        data = json.load(f)
    games = data.get("games", [])
    last_r = None
    for g in games:
        r = g.get("round")
        if r != last_r:
            if last_r is not None:
                print()
            print(ROUND_LABELS.get(r, f"Round {r}"))
            last_r = r
        print(g.get("pick", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
