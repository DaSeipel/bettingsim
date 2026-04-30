#!/usr/bin/env python3
"""Bootstrap pitcher_stats.csv with starters from Apr 1, 2026 through today.

Default is dry-run. Use --apply to write cumulative upserts into pitcher_stats.csv.
"""

from __future__ import annotations

import argparse
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from fetch_mlb_pitchers import (
    OUTPUT_PATH,
    PITCHER_BLEND_2026_WEIGHT,
    REQUEST_SLEEP_S,
    REQUEST_TIMEOUT,
    blend_pitcher_rows,
    fetch_pitcher_season_row,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = "https://statsapi.mlb.com/api/v1"
START_DATE = date(2026, 4, 1)


def _schedule_for_date(d: date) -> list[dict[str, Any]]:
    r = requests.get(
        f"{BASE}/schedule",
        params={"sportId": 1, "date": d.isoformat(), "hydrate": "probablePitcher,decisions"},
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    payload = r.json()
    dates = payload.get("dates") or []
    if not dates:
        return []
    return dates[0].get("games") or []


def _is_final(game: dict[str, Any]) -> bool:
    status = game.get("status") or {}
    coded = str(status.get("codedGameState") or "").strip().upper()
    abstract = str(status.get("abstractGameState") or "").strip().lower()
    return coded in {"F", "O", "R"} or abstract == "final"


def _add_pitcher(out: dict[int, str], pitcher: dict | None) -> None:
    if not isinstance(pitcher, dict):
        return
    pid = pitcher.get("id")
    name = str(pitcher.get("fullName") or "").strip()
    if not pid or not name:
        return
    try:
        out[int(pid)] = name
    except (TypeError, ValueError):
        return


def _pitchers_from_game(game: dict[str, Any]) -> dict[int, str]:
    out: dict[int, str] = {}
    teams = game.get("teams") or {}

    # schedule hydrate=probablePitcher provides game-side pitchers; for historical games this
    # is the best available starter field in this endpoint. `decisions` are W/L/save pitchers,
    # not necessarily starters, so do not add them to the starter bootstrap set.
    for side in ("home", "away"):
        team_obj = teams.get(side) or {}
        _add_pitcher(out, team_obj.get("probablePitcher"))

    return out


def _fetch_pitcher_row(player_id: int, name: str) -> dict | None:
    time.sleep(REQUEST_SLEEP_S)
    row_2025 = fetch_pitcher_season_row(player_id, 2025, name, name)
    time.sleep(REQUEST_SLEEP_S)
    row_2026 = fetch_pitcher_season_row(player_id, 2026, name, name)
    return blend_pitcher_rows(row_2025, row_2026, w26=PITCHER_BLEND_2026_WEIGHT)


def _merge_rows(rows: list[dict]) -> tuple[pd.DataFrame, int, int]:
    existing_df = pd.DataFrame()
    if OUTPUT_PATH.exists():
        existing_df = pd.read_csv(OUTPUT_PATH)
    fetched_df = pd.DataFrame(rows)

    output_cols = [
        "odds_name",
        "player_id",
        "full_name",
        "season",
        "era",
        "fip",
        "xfip",
        "whip",
        "k9",
        "bb9",
        "innings_pitched",
    ]

    if existing_df.empty:
        merged_df = fetched_df.copy()
        added = len(merged_df)
        updated = 0
    else:
        existing_df = existing_df.copy()
        fetched_df = fetched_df.copy()
        existing_df["player_id"] = pd.to_numeric(existing_df["player_id"], errors="coerce")
        existing_df = existing_df[existing_df["player_id"].notna()].copy()
        existing_df["player_id"] = existing_df["player_id"].astype(int)
        fetched_df["player_id"] = pd.to_numeric(fetched_df["player_id"], errors="coerce")
        fetched_df = fetched_df[fetched_df["player_id"].notna()].copy()
        fetched_df["player_id"] = fetched_df["player_id"].astype(int)

        existing_ids = set(existing_df["player_id"].tolist())
        fetched_ids = set(fetched_df["player_id"].tolist())
        added = len(fetched_ids - existing_ids)
        updated = len(existing_ids & fetched_ids)

        merged = {int(r["player_id"]): r.to_dict() for _, r in existing_df.iterrows()}
        for _, r in fetched_df.iterrows():
            merged[int(r["player_id"])] = r.to_dict()
        merged_df = pd.DataFrame(list(merged.values()))

    for c in output_cols:
        if c not in merged_df.columns:
            merged_df[c] = None
    return merged_df[output_cols], added, updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap pitcher_stats.csv from historical MLB starters")
    parser.add_argument("--apply", action="store_true", help="Write merged rows to pitcher_stats.csv")
    args = parser.parse_args()

    today = datetime.now().date()
    pitchers: dict[int, str] = {}
    dates_processed = 0
    d = START_DATE
    while d <= today:
        time.sleep(REQUEST_SLEEP_S)
        games = _schedule_for_date(d)
        dates_processed += 1
        for game in games:
            pitchers.update(_pitchers_from_game(game))
        print(
            f"Scanned {d.isoformat()}: games={len(games)} cumulative_pitchers={len(pitchers)}",
            flush=True,
        )
        d += timedelta(days=1)

    rows: list[dict] = []
    pitcher_items = sorted(pitchers.items(), key=lambda kv: kv[1].lower())
    print(f"Fetching season rows for {len(pitcher_items)} unique pitchers...", flush=True)
    for idx, (pid, name) in enumerate(pitcher_items, start=1):
        print(f"[{idx}/{len(pitcher_items)}] {name} ({pid})", flush=True)
        try:
            row = _fetch_pitcher_row(pid, name)
        except requests.RequestException as exc:
            print(f"Stats fetch failed for {name} ({pid}): {exc}")
            row = None
        if row:
            rows.append(row)
        else:
            print(f"No season pitching row for {name} ({pid})")

    merged_df, added, updated = _merge_rows(rows)

    print(f"Dates processed: {dates_processed}")
    print(f"Unique pitchers found: {len(pitchers)}")
    print(f"Pitchers with fetched season rows: {len(rows)}")
    print(f"Pitchers added to CSV: {added}")
    print(f"Pitchers updated: {updated}")
    print(f"Final CSV row count after merge: {len(merged_df)}")

    if args.apply:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(OUTPUT_PATH, index=False)
        print(f"Applied merge to {OUTPUT_PATH}")
    else:
        print("Dry run only. Re-run with --apply to write pitcher_stats.csv.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
