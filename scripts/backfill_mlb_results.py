#!/usr/bin/env python3
"""Backfill unresolved MLB moneyline results in play_history from MLB Stats API.

Default mode is dry-run preview. Use --apply to write updates.
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "espn.db"
SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
REQUEST_TIMEOUT = 20

FINAL_CODED_STATES = {"F", "O", "R"}  # Final, Over, game completed (retrosheet-style)
MONEYLINE_PLACEHOLDER = -999.0

TEAM_ALIASES = {
    "athletics": {"athletics", "oakland athletics", "a's"},
    "diamondbacks": {"diamondbacks", "arizona diamondbacks", "d-backs", "dbacks"},
}


@dataclass
class PlannedUpdate:
    play_id: int
    date_generated: str
    recommended_side: str
    result: str
    reason: str


@dataclass
class ReviewItem:
    play_id: int
    date_generated: str
    recommended_side: str
    spread_or_total: float | None
    reason: str


def _canon_team(name: str) -> str:
    n = " ".join(str(name or "").strip().lower().replace(".", "").split())
    if not n:
        return ""
    for canonical, variants in TEAM_ALIASES.items():
        if n in variants:
            return canonical
    parts = n.split()
    return parts[-1] if parts else n


def _load_unresolved_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        """
        SELECT play_id, date_generated, recommended_side, market_odds_at_time, my_edge_pct,
               spread_or_total, home_team, away_team
        FROM play_history
        WHERE sport='MLB' AND result IS NULL
        ORDER BY date_generated ASC, recommended_side ASC, play_id ASC
        """
    )
    return cur.fetchall()


def _fetch_schedule_for_date(date_str: str) -> list[dict[str, Any]]:
    r = requests.get(
        SCHEDULE_URL,
        params={"sportId": 1, "date": date_str},
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    payload = r.json()
    dates = payload.get("dates", [])
    if not dates:
        return []
    return dates[0].get("games", []) or []


def _is_final(game: dict[str, Any]) -> bool:
    status = game.get("status") or {}
    coded = str(status.get("codedGameState") or "").strip().upper()
    abstract = str(status.get("abstractGameState") or "").strip().lower()
    return coded in FINAL_CODED_STATES or abstract == "final"


def _team_in_game(game: dict[str, Any], team_name: str) -> bool:
    teams = game.get("teams") or {}
    home_name = ((teams.get("home") or {}).get("team") or {}).get("name") or ""
    away_name = ((teams.get("away") or {}).get("team") or {}).get("name") or ""
    needle = _canon_team(team_name)
    return needle in {_canon_team(home_name), _canon_team(away_name)}


def _match_game(row: sqlite3.Row, games: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None]:
    side = str(row["recommended_side"] or "").strip()
    home = str(row["home_team"] or "").strip()
    away = str(row["away_team"] or "").strip()
    side_c = _canon_team(side)
    home_c = _canon_team(home)
    away_c = _canon_team(away)

    by_side = [g for g in games if _team_in_game(g, side)]
    if not by_side:
        return None, "no_game_for_team_on_date"

    # Prefer exact matchup if home/away are populated in DB.
    by_matchup = []
    if home_c and away_c:
        for g in by_side:
            teams = g.get("teams") or {}
            g_home = _canon_team(((teams.get("home") or {}).get("team") or {}).get("name") or "")
            g_away = _canon_team(((teams.get("away") or {}).get("team") or {}).get("name") or "")
            if g_home == home_c and g_away == away_c:
                by_matchup.append(g)
    candidates = by_matchup if by_matchup else by_side

    finals = [g for g in candidates if _is_final(g)]
    if not finals:
        return None, "game_not_final"

    if len(finals) == 1:
        return finals[0], None

    # Doubleheader handling: if exactly two final candidates, default to game 2 (later start time).
    if len(finals) == 2:
        finals_sorted = sorted(finals, key=lambda g: str(g.get("gameDate") or ""))
        return finals_sorted[1], "doubleheader_defaulted_to_game2"

    return None, "ambiguous_multiple_final_games"


def _result_from_game_for_side(game: dict[str, Any], side: str) -> str | None:
    teams = game.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    home_name = ((home.get("team") or {}).get("name") or "").strip()
    away_name = ((away.get("team") or {}).get("name") or "").strip()
    try:
        home_score = int(home.get("score"))
        away_score = int(away.get("score"))
    except (TypeError, ValueError):
        return None

    side_c = _canon_team(side)
    home_c = _canon_team(home_name)
    away_c = _canon_team(away_name)
    if side_c not in {home_c, away_c}:
        return None
    picked_score = home_score if side_c == home_c else away_score
    opp_score = away_score if side_c == home_c else home_score
    if picked_score > opp_score:
        return "W"
    if picked_score < opp_score:
        return "L"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill unresolved MLB moneyline results from MLB Stats API")
    parser.add_argument("--apply", action="store_true", help="Apply updates to database (default is dry-run)")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    rows = _load_unresolved_rows(conn)
    print(f"Unresolved MLB rows: {len(rows)}")

    schedule_cache: dict[str, list[dict[str, Any]]] = {}
    planned: list[PlannedUpdate] = []
    skipped_totals: list[ReviewItem] = []
    manual_review: list[ReviewItem] = []

    for row in rows:
        spread_or_total = row["spread_or_total"]
        if spread_or_total is None or float(spread_or_total) != MONEYLINE_PLACEHOLDER:
            skipped_totals.append(
                ReviewItem(
                    play_id=int(row["play_id"]),
                    date_generated=str(row["date_generated"]),
                    recommended_side=str(row["recommended_side"] or ""),
                    spread_or_total=spread_or_total,
                    reason="totals_row_skipped",
                )
            )
            continue

        d = str(row["date_generated"])
        if d not in schedule_cache:
            try:
                schedule_cache[d] = _fetch_schedule_for_date(d)
            except Exception as exc:
                manual_review.append(
                    ReviewItem(
                        play_id=int(row["play_id"]),
                        date_generated=d,
                        recommended_side=str(row["recommended_side"] or ""),
                        spread_or_total=spread_or_total,
                        reason=f"schedule_fetch_error:{exc}",
                    )
                )
                continue

        game, note = _match_game(row, schedule_cache[d])
        if game is None:
            manual_review.append(
                ReviewItem(
                    play_id=int(row["play_id"]),
                    date_generated=d,
                    recommended_side=str(row["recommended_side"] or ""),
                    spread_or_total=spread_or_total,
                    reason=note or "no_match",
                )
            )
            continue

        result = _result_from_game_for_side(game, str(row["recommended_side"] or ""))
        if result is None:
            manual_review.append(
                ReviewItem(
                    play_id=int(row["play_id"]),
                    date_generated=d,
                    recommended_side=str(row["recommended_side"] or ""),
                    spread_or_total=spread_or_total,
                    reason="could_not_compute_wl_or_tie",
                )
            )
            continue

        planned.append(
            PlannedUpdate(
                play_id=int(row["play_id"]),
                date_generated=d,
                recommended_side=str(row["recommended_side"] or ""),
                result=result,
                reason=note or "matched",
            )
        )

    print("\n=== DRY-RUN PREVIEW: PLANNED UPDATES ===")
    if not planned:
        print("(none)")
    else:
        for p in planned:
            print(f"play_id={p.play_id} | {p.date_generated} | {p.recommended_side} -> {p.result} | {p.reason}")
    print(f"Planned update count: {len(planned)}")

    print("\n=== SKIPPED TOTALS (MANUAL REVIEW) ===")
    if not skipped_totals:
        print("(none)")
    else:
        for s in skipped_totals:
            print(f"play_id={s.play_id} | {s.date_generated} | {s.recommended_side} | line={s.spread_or_total} | {s.reason}")
    print(f"Skipped totals count: {len(skipped_totals)}")

    print("\n=== COULD NOT MATCH / MANUAL REVIEW ===")
    if not manual_review:
        print("(none)")
    else:
        for m in manual_review:
            print(f"play_id={m.play_id} | {m.date_generated} | {m.recommended_side} | {m.reason}")
    print(f"Manual review count: {len(manual_review)}")

    if not args.apply:
        remaining_null = conn.execute("SELECT COUNT(*) FROM play_history WHERE sport='MLB' AND result IS NULL").fetchone()[0]
        print("\nDry run only. Re-run with --apply to write updates.")
        print(f"Current MLB NULL result count: {remaining_null}")
        conn.close()
        return 0

    updated = 0
    for p in planned:
        cur = conn.execute(
            """
            UPDATE play_history
            SET result = ?, actual_payout = NULL
            WHERE play_id = ?
              AND sport = 'MLB'
              AND result IS NULL
              AND spread_or_total = -999.0
            """,
            (p.result, p.play_id),
        )
        updated += cur.rowcount
    conn.commit()

    wl = {r[0]: r[1] for r in conn.execute(
        "SELECT result, COUNT(*) FROM play_history WHERE sport='MLB' AND result IN ('W','L') GROUP BY result"
    )}
    remaining_null = conn.execute("SELECT COUNT(*) FROM play_history WHERE sport='MLB' AND result IS NULL").fetchone()[0]

    print("\n=== APPLY SUMMARY ===")
    print(f"Rows updated automatically: {updated}")
    print(f"W count: {wl.get('W', 0)}")
    print(f"L count: {wl.get('L', 0)}")
    print(f"Rows skipped (totals): {len(skipped_totals)}")
    print(f"Rows not matched / manual review: {len(manual_review)}")
    print(f"Final MLB NULL result count: {remaining_null}")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
