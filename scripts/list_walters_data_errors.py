#!/usr/bin/env python3
"""
List all first-round matchups excluded as data errors (delta > 15 pts).
For each: market spread, model spread, and whether each team exists in power_rankings_2026.csv.
Flags any team missing from power rankings.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.bracket_analysis import (
    parse_bracket_csv,
    parse_power_rankings_csv,
    run_bracket_analysis,
    resolve_team_name,
    BRACKET_TEAM_ALIAS_MAP,
)


def main() -> int:
    bracket_path = ROOT / "data" / "ncaab" / "bracket_2026.csv"
    rankings_path = ROOT / "data" / "ncaab" / "power_rankings_2026.csv"

    if not bracket_path.exists():
        print(f"Bracket not found: {bracket_path}")
        return 1
    if not rankings_path.exists():
        print(f"Power rankings not found: {rankings_path}")
        return 1

    bracket_content = bracket_path.read_text()
    power_content = rankings_path.read_text()

    # Build set of team names in power rankings (canonical / as-is for matching)
    rankings = parse_power_rankings_csv(power_content)
    power_teams_raw = {str(r.get("team", "")).strip() for r in rankings if r.get("team")}
    # Also lowercase for case-insensitive match
    power_teams_lower = {t.lower() for t in power_teams_raw}

    def in_power_rankings(team: str) -> bool:
        canonical = resolve_team_name(team, BRACKET_TEAM_ALIAS_MAP)
        if not canonical:
            return False
        return canonical in power_teams_raw or canonical.lower() in power_teams_lower

    out = run_bracket_analysis(
        bracket_content,
        power_content,
        n_sims=100,  # minimal for speed; we only need walters_data_errors
    )

    errors = out.get("walters_data_errors", [])
    if not errors:
        print("No matchups excluded as data errors (delta > 15 pts).")
        return 0

    print(f"First-round matchups excluded as data errors (delta > 15 pts): {len(errors)}\n")
    print(f"{'Matchup (TeamA vs TeamB)':<55} {'Mkt':>6} {'Model':>6} {'Delta':>6}  TeamA in PR?  TeamB in PR?  Flags")
    print("-" * 115)

    for e in errors:
        team_a = e.get("team_a", "")
        team_b = e.get("team_b", "")
        market = e.get("market_spread")
        model = e.get("model_spread")
        diff = e.get("diff_pts")
        in_a = "Yes" if in_power_rankings(team_a) else "MISSING"
        in_b = "Yes" if in_power_rankings(team_b) else "MISSING"
        flags = []
        if in_a == "MISSING":
            flags.append("TeamA missing from power_rankings_2026.csv")
        if in_b == "MISSING":
            flags.append("TeamB missing from power_rankings_2026.csv")
        flag_str = "; ".join(flags) if flags else "—"
        matchup = f"{team_a} vs {team_b}"
        if len(matchup) > 54:
            matchup = matchup[:51] + "..."
        print(f"{matchup:<55} {market:>6.1f} {model:>6.1f} {diff:>6.1f}  {in_a:^11}  {in_b:^11}  {flag_str}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
