#!/usr/bin/env python3
"""
Identify the "danger game" for each Final Four team from the filled bracket.

For each of the specified teams (Michigan, Michigan St., Arizona, Florida),
walk their actual path in the filled bracket (from first game through their
elimination or the championship), compute the win probability for that team
in each game, and flag the game with the lowest win probability as the
danger game.

Print the full path with win probabilities and highlight the danger game.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys
from typing import Dict, List


ROOT = Path(__file__).resolve().parent.parent

ROUND_NAMES = {
    1: "Round of 64",
    2: "Round of 32",
    3: "Sweet 16",
    4: "Elite 8",
    5: "National Semifinals",
    6: "Championship",
}

FINAL_FOUR_TEAMS = ["Michigan", "Michigan St.", "Arizona", "Florida"]


def _team_win_pct_in_game(team: str, game: Dict) -> float:
    if team == game.get("team_a"):
        return float(game.get("win_pct_a", 0.0))
    if team == game.get("team_b"):
        return float(game.get("win_pct_b", 0.0))
    return 0.0


def _opponent_in_game(team: str, game: Dict) -> str:
    if team == game.get("team_a"):
        return str(game.get("team_b"))
    if team == game.get("team_b"):
        return str(game.get("team_a"))
    return "—"


def _path_for_team(team: str, games: List[Dict]) -> List[Dict]:
    """
    Return the ordered list of games that this team actually plays in the
    filled bracket: start with its first appearance, then follow rounds until
    it is eliminated (or wins the title).
    """
    path: List[Dict] = []

    # Games are already listed in chronological bracket order in JSON.
    for g in games:
        if team != g.get("team_a") and team != g.get("team_b"):
            continue
        path.append(g)
        # Stop once the team loses a game (pick != team) or after championship.
        if g.get("pick") != team:
            break
    return path


def main() -> int:
    json_path = ROOT / "data" / "ncaab" / "bracket_filled_2026.json"
    if not json_path.exists():
        print(f"Missing {json_path}. Run scripts/fill_bracket_2026.py first.")
        return 1

    with open(json_path) as f:
        data = json.load(f)
    games = data.get("games", [])

    print("=" * 80)
    print("  Final Four danger games (lowest win-probability game on each path)")
    print("=" * 80)

    for team in FINAL_FOUR_TEAMS:
        path = _path_for_team(team, games)
        if not path:
            print(f"\n{team}: no games found in bracket.")
            continue

        # Identify danger game = minimum win probability along this path.
        win_pcts = [_team_win_pct_in_game(team, g) for g in path]
        min_idx = min(range(len(path)), key=lambda i: win_pcts[i])

        print(f"\n{team} path:")
        print(f"  {'Round':<22} {'Opponent':<24} {'Win%':>7}  Note")
        print("  " + "-" * 62)
        for i, g in enumerate(path):
            r = g.get("round")
            round_name = ROUND_NAMES.get(r, f"Round {r}")
            opp = _opponent_in_game(team, g)
            wp = win_pcts[i]
            note = "DANGER GAME" if i == min_idx else ""
            print(f"  {round_name:<22} {opp:<24} {wp:6.1f}%  {note}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

