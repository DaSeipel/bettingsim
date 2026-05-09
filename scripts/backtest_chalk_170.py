#!/usr/bin/env python3
"""
Read-only backtest: compare MAX_FAVORITE_ODDS = -160 vs -170 for marginal chalk plays.

Scans for historical MLB odds snapshots. If none exist (only rolling live_mlb_odds.json),
prints an honest summary and exits — no fabricated backtest.

When dated MLB odds JSON archives are present under data/ (see discover_historical_mlb_odds()),
replays scripts/predict_mlb.py moneyline logic per snapshot using MAX_FAVORITE_ODDS=-170,
filters to marginal favorite chalk (-170 < odds <= -160), fetches results from MLB Stats API.

Usage (from repo root):
  python3 scripts/backtest_chalk_170.py

Constraints: read-only (no DB writes). Does not modify predict_mlb.py or other pipelines.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*urllib3 v2 only supports OpenSSL.*")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

# Reuse MLB schedule / result resolution from backfill (read-only).
import backfill_mlb_results as bf

# Prediction helpers — same module predict_mlb.py uses (do not edit predict_mlb.py).
from engine.mlb_engine import (
    DEFAULT_MLB_TEAM_STATS_CSV,
    implied_probability_fair_two_sided,
    load_live_mlb_odds,
    value_summary_moneyline,
)

import predict_mlb as pm

STAKE_USD = 10.0
# Marginal zone: blocked by -160 chalk rule but passes -170 chalk rule.
MARGINAL_MIN_EXCLUSIVE = -170.0
MARGINAL_MAX_INCLUSIVE = -160.0


def _print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _is_mlb_odds_blob(data: Any) -> bool:
    return isinstance(data, dict) and data.get("sport_key") == "baseball_mlb" and "games" in data


def discover_historical_mlb_odds() -> dict[str, Any]:
    """
    Survey data/odds, data/cache, and SQLite for anything usable as a dated MLB odds archive.
    Returns a dict with lists and notes for stdout.
    """
    odds_dir = ROOT / "data" / "odds"
    cache_dir = ROOT / "data" / "cache"
    db_path = ROOT / "data" / "espn.db"

    odds_dir_files = sorted(odds_dir.iterdir()) if odds_dir.is_dir() else []
    json_in_odds = [p for p in odds_dir_files if p.suffix.lower() == ".json"]
    csv_in_odds = [p for p in odds_dir_files if p.suffix.lower() == ".csv"]

    cache_json: list[Path] = []
    if cache_dir.is_dir():
        cache_json = sorted(cache_dir.glob("*.json"))

    sqlite_odds_notes: list[str] = []
    sqlite_table_names: list[str] = []
    mlb_play_history_n = 0
    clv_mlb_n = 0

    if db_path.is_file():
        import sqlite3

        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            cur0 = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            sqlite_table_names = [r[0] for r in cur0.fetchall()]
            cur = conn.execute(
                "SELECT COUNT(*) FROM play_history WHERE sport=?",
                ("MLB",),
            )
            row = cur.fetchone()
            mlb_play_history_n = int(row[0]) if row else 0
            try:
                cur2 = conn.execute(
                    "SELECT COUNT(*) FROM clv_tracker WHERE league=? OR league LIKE ?",
                    ("MLB", "%MLB%"),
                )
                r2 = cur2.fetchone()
                clv_mlb_n = int(r2[0]) if r2 else 0
            except sqlite3.Error:
                clv_mlb_n = 0
        finally:
            conn.close()
        sqlite_odds_notes.append(
            f"play_history: {mlb_play_history_n} MLB row(s) with market_odds_at_time on saved picks only "
            "(cannot reconstruct full daily odds slates for predict_mlb.py replay)."
        )
        sqlite_odds_notes.append(
            f"clv_tracker: {clv_mlb_n} row(s) matching MLB (no full-slate pre-game archive expected here)."
        )

    archived_mlb_json: list[Path] = []
    seen: set[str] = set()

    def _consider_json_path(p: Path) -> None:
        if not p.is_file():
            return
        key = str(p.resolve())
        if key in seen:
            return
        if p.name == "live_mlb_odds.json":
            return
        try:
            with open(p, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return
        if _is_mlb_odds_blob(raw):
            seen.add(key)
            archived_mlb_json.append(p)

    for p in json_in_odds:
        _consider_json_path(p)

    # Full-tree scan: dated MLB snapshots could live under data/.../mlb_archive/ etc.
    data_root = ROOT / "data"
    if data_root.is_dir():
        for p in data_root.rglob("*.json"):
            _consider_json_path(p)

    archived_mlb_json.sort(key=lambda x: str(x))

    live_path = odds_dir / "live_mlb_odds.json"
    live_ok = live_path.is_file()
    rolling_note = (
        f"data/odds/live_mlb_odds.json: {'present (single rolling snapshot; overwritten when fetch runs)' if live_ok else 'missing'}."
    )

    return {
        "odds_dir_json": json_in_odds,
        "odds_dir_csv_count": len(csv_in_odds),
        "archived_mlb_json": archived_mlb_json,
        "cache_json": cache_json,
        "sqlite_odds_notes": sqlite_odds_notes,
        "sqlite_table_names": sqlite_table_names,
        "rolling_note": rolling_note,
    }


def print_discovery_report(survey: dict[str, Any]) -> None:
    _print_section("1) Historical odds discovery (read-only)")
    print(f"Repo root: {ROOT}")
    print()
    print("data/odds — JSON files:")
    for p in survey["odds_dir_json"]:
        tag = ""
        if p.name == "live_mlb_odds.json":
            tag = " [rolling MLB slate]"
        print(f"  - {p.name}{tag}")
    print(f"data/odds — CSV count: {survey['odds_dir_csv_count']} (NCAAB-style archived lines; not MLB JSON slates.)")
    print()
    print("Archived MLB odds JSON (sport_key=baseball_mlb, excluding live_mlb_odds.json):")
    if survey["archived_mlb_json"]:
        for p in survey["archived_mlb_json"]:
            print(f"  - {p.relative_to(ROOT)}")
    else:
        print("  (none)")
    print()
    print("data/cache — JSON files (no dated MLB pre-game odds archive expected):")
    for p in survey["cache_json"]:
        print(f"  - {p.name}")
    print()
    print("SQLite (data/espn.db):")
    if survey.get("sqlite_table_names"):
        print(f"  Tables: {', '.join(survey['sqlite_table_names'])}")
    print("  Odds-related: no dedicated historical MLB odds snapshot table; see below.")
    for line in survey["sqlite_odds_notes"]:
        print(f"  {line}")
    print()
    print(survey["rolling_note"])


def is_marginal_chalk_favorite(odds_used: float) -> bool:
    """True if the pick is favorite chalk in (-170, -160] per American odds (more negative = heavier favorite)."""
    return MARGINAL_MIN_EXCLUSIVE < odds_used <= MARGINAL_MAX_INCLUSIVE


def flat_stake_pnl_usd(american_odds: float, won: bool, stake: float = STAKE_USD) -> float:
    if not won:
        return -stake
    o = float(american_odds)
    if o > 0:
        return stake * (o / 100.0)
    return stake * (100.0 / abs(o))


def _load_stats_and_supporting() -> tuple[pd.DataFrame, pd.DataFrame, dict[int, dict], dict[str, dict]]:
    stats_path = DEFAULT_MLB_TEAM_STATS_CSV
    if not stats_path.exists():
        stats_df = pd.DataFrame()
    else:
        try:
            stats_df = pd.read_csv(stats_path)
        except Exception:
            stats_df = pd.DataFrame()

    if pm.PITCHER_STATS_PATH.exists():
        try:
            pitcher_df = pd.read_csv(pm.PITCHER_STATS_PATH)
        except Exception:
            pitcher_df = pd.DataFrame()
    else:
        pitcher_df = pd.DataFrame()

    form_by_team_id = pm._load_recent_form_by_team_id()
    weather_by_matchup = pm._load_mlb_weather_by_matchup()
    return stats_df, pitcher_df, form_by_team_id, weather_by_matchup


def marginal_plays_from_blob(
    blob: dict[str, Any],
    stats_df: pd.DataFrame,
    pitcher_df: pd.DataFrame,
    form_by_team_id: dict[int, dict],
    weather_by_matchup: dict[str, dict],
) -> list[dict[str, Any]]:
    """
    Replay predict_mlb moneyline path with MAX_FAVORITE_ODDS=-170.
    Return plays that pass all filters AND lie in marginal chalk zone.
    """
    saved_max = pm.MAX_FAVORITE_ODDS
    try:
        pm.MAX_FAVORITE_ODDS = -170
        out: list[dict[str, Any]] = []
        games = blob.get("games") or []

        for g in games:
            if not isinstance(g, dict):
                continue
            home = str(g.get("home_team") or "").strip()
            away = str(g.get("away_team") or "").strip()
            wx = weather_by_matchup.get(pm._matchup_weather_key(away, home))

            ml = g.get("moneyline") or {}
            home_odds_am = pm._american_to_float(ml.get("home_odds"))
            away_odds_am = pm._american_to_float(ml.get("away_odds"))

            hr = pm._find_team_row(stats_df, home)
            ar = pm._find_team_row(stats_df, away)
            if hr is None or ar is None or home_odds_am is None or away_odds_am is None:
                continue

            hp = pm._find_pitcher_row(pitcher_df, g.get("home_pitcher"))
            ap = pm._find_pitcher_row(pitcher_df, g.get("away_pitcher"))
            if hp is None:
                hp = pm.MISSING_PITCHER_STATS_ROW
            if ap is None:
                ap = pm.MISSING_PITCHER_STATS_ROW

            bonus_h, _ = pm._form_bonus_from_row(hr, form_by_team_id)
            bonus_a, _ = pm._form_bonus_from_row(ar, form_by_team_id)
            pitch_h, _ = pm._recent_pitch_adj_from_row(hr, form_by_team_id)
            pitch_a, _ = pm._recent_pitch_adj_from_row(ar, form_by_team_id)
            form_diff = bonus_h - bonus_a
            recent_pitch_diff = pitch_h - pitch_a
            temp_adj = pm._temp_adj_from_weather(wx)

            p_home_model, game_park_mult = pm._home_win_prob(
                hr,
                ar,
                hp,
                ap,
                home_team_name=home,
                form_diff=form_diff,
                recent_pitch_diff=recent_pitch_diff,
                weather_temp_adj=temp_adj,
            )

            fair_home, _ = implied_probability_fair_two_sided(float(home_odds_am), float(away_odds_am))
            p_home = float(
                p_home_model
                - pm.PROB_SHRINK_TOWARD_MARKET * (float(p_home_model) - float(fair_home))
            )

            summ_home = value_summary_moneyline(float(p_home), float(home_odds_am), float(away_odds_am))
            summ_away = value_summary_moneyline(float(1.0 - p_home), float(away_odds_am), float(home_odds_am))
            edge_h = float(summ_home["edge"])
            edge_a = float(summ_away["edge"])

            if edge_h >= edge_a:
                pick_side = "home"
            else:
                pick_side = "away"

            if pick_side == "home":
                pick = home
                odds_used = float(home_odds_am)
                model_p = float(p_home)
                summ = value_summary_moneyline(model_p, odds_used, float(away_odds_am))
            else:
                pick = away
                odds_used = float(away_odds_am)
                model_p = float(1.0 - p_home)
                summ = value_summary_moneyline(model_p, odds_used, float(home_odds_am))

            edge = float(summ["edge"])
            if odds_used <= pm.MAX_FAVORITE_ODDS:
                continue
            edge = pm._juice_penalized_edge(edge, odds_used)

            if model_p < pm.MIN_MODEL_PROB:
                continue
            if edge > pm.MAX_EDGE_DECIMAL:
                continue
            if edge < pm.MIN_EDGE_DECIMAL:
                continue

            if not is_marginal_chalk_favorite(odds_used):
                continue

            out.append(
                {
                    "card_date": pm._card_date_iso(blob),
                    "event_id": str(g.get("event_id") or ""),
                    "away_team": away,
                    "home_team": home,
                    "pick": pick,
                    "odds_american": odds_used,
                    "edge": edge,
                    "model_prob": model_p,
                    "park_mult": float(game_park_mult),
                }
            )

        return out
    finally:
        pm.MAX_FAVORITE_ODDS = saved_max


def resolve_pick_result(date_iso: str, away: str, home: str, pick: str) -> tuple[str | None, str]:
    try:
        games = bf._fetch_schedule_for_date(date_iso)
    except Exception as exc:
        return None, f"schedule_error:{exc}"

    gmatch, reason = _select_game_for_matchup(games, away, home)
    if gmatch is None:
        return None, reason or "no_game"

    res = bf._result_from_game_for_side(gmatch, pick)
    if res is None:
        return None, "no_result_for_side"
    return res, "ok"


def _select_game_for_matchup(
    games: list[dict[str, Any]], away: str, home: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Pick a single final game for (away @ home), mirroring backfill doubleheader handling."""
    home_c = bf._canon_team(home)
    away_c = bf._canon_team(away)
    candidates: list[dict[str, Any]] = []
    for g in games:
        teams = g.get("teams") or {}
        g_home = bf._canon_team(((teams.get("home") or {}).get("team") or {}).get("name") or "")
        g_away = bf._canon_team(((teams.get("away") or {}).get("team") or {}).get("name") or "")
        if g_home == home_c and g_away == away_c:
            candidates.append(g)

    if not candidates:
        return None, "no_matchup"

    finals = [g for g in candidates if bf._is_final(g)]
    if not finals:
        return None, "game_not_final"

    if len(finals) == 1:
        return finals[0], None

    if len(finals) == 2:
        finals_sorted = sorted(finals, key=lambda gg: str(gg.get("gameDate") or ""))
        return finals_sorted[1], "doubleheader_defaulted_to_game2"

    return None, "ambiguous_multiple_final_games"


def run_backtest_on_archives(archives: list[Path]) -> int:
    _print_section("2) Backtest (historical snapshots found)")
    print(
        "Caveat: team_stats.csv / pitcher_stats / recent_form / weather reflect the repo as of "
        "this run, not necessarily the state on each archived card_date. Edge/prob are replayed "
        "with current files — interpret as illustrative if archives span many dates."
    )
    print()

    stats_df, pitcher_df, form_by_team_id, weather_by_matchup = _load_stats_and_supporting()

    rows_for_table: list[dict[str, Any]] = []

    for path in archives:
        blob = load_live_mlb_odds(path)
        if not blob.get("games"):
            print(f"Skip (no games): {path}")
            continue

        plays = marginal_plays_from_blob(
            blob,
            stats_df,
            pitcher_df,
            form_by_team_id,
            weather_by_matchup,
        )
        for pl in plays:
            d = str(pl["card_date"]).strip()
            res, note = resolve_pick_result(d, pl["away_team"], pl["home_team"], pl["pick"])
            won = res == "W"
            lost = res == "L"
            if res is None:
                wl = "?"
                pnl = 0.0
            else:
                wl = res
                pnl = flat_stake_pnl_usd(pl["odds_american"], won)

            rows_for_table.append(
                {
                    "date": d,
                    "matchup": f"{pl['away_team']} @ {pl['home_team']}",
                    "pick": pl["pick"],
                    "odds": pl["odds_american"],
                    "edge_pct": pl["edge"] * 100.0,
                    "result": wl,
                    "pnl": pnl,
                    "source_file": str(path.relative_to(ROOT)),
                    "resolve_note": note,
                }
            )

    _print_section("3) Qualifying marginal plays (MAX=-170 replay; odds in (-170, -160])")
    if not rows_for_table:
        print("No qualifying games in the provided archives with current stats/inputs.")
        return 0

    col_w = [12, 42, 22, 8, 10, 8, 12]
    hdr = f"{'Date':<12} {'Matchup':<42} {'Pick':<22} {'Odds':>8} {'Edge %':>9} {'Res':>4} {'P/L $':>10}"
    print(hdr)
    print("-" * len(hdr))

    total_pnl = 0.0
    wins = losses = 0
    unknown = 0

    for r in sorted(rows_for_table, key=lambda x: (x["date"], x["matchup"])):
        print(
            f"{r['date']:<12} {r['matchup']:<42} {r['pick']:<22} {r['odds']:>8.0f} "
            f"{r['edge_pct']:>8.1f}% {r['result']:>4} {r['pnl']:>10.2f}"
        )
        if r["result"] == "W":
            wins += 1
            total_pnl += r["pnl"]
        elif r["result"] == "L":
            losses += 1
            total_pnl += r["pnl"]
        else:
            unknown += 1

    n_decided = wins + losses
    stake_total = STAKE_USD * n_decided
    roi = (total_pnl / stake_total * 100.0) if stake_total > 0 else 0.0
    win_pct = (100.0 * wins / n_decided) if n_decided > 0 else 0.0

    _print_section("4) Summary")
    print(f"Total qualifying picks (rows): {len(rows_for_table)}")
    print(f"Resolved W-L: {wins}-{losses}" + (f" ({unknown} unresolved)" if unknown else ""))
    if n_decided:
        print(f"Win % (on resolved): {win_pct:.1f}%")
    print(f"Total wagered (flat ${STAKE_USD:.0f} on resolved only): ${stake_total:.2f}")
    print(f"Total P/L (resolved): ${total_pnl:.2f}")
    print(f"ROI (resolved): {roi:.2f}%")
    print()
    if n_decided < 10:
        print(
            "Recommendation: Sample is very small — treat ROI as anecdotal even if positive or negative."
        )
    elif total_pnl > 0 and roi > 0:
        print(
            "Recommendation: In this (limited) replay, loosening chalk to -170 would have been profitable "
            "on marginal plays; confirm with more archived slates and same-day stats if you add them."
        )
    else:
        print(
            "Recommendation: In this replay, loosening to -170 does not show clear profitability on "
            "marginal plays; keep -160 unless you gather better historical inputs."
        )

    return 0


def main() -> int:
    survey = discover_historical_mlb_odds()
    print_discovery_report(survey)

    archives = survey["archived_mlb_json"]

    if not archives:
        _print_section("Result: insufficient historical MLB odds for this backtest")
        print(
            "Only data/odds/live_mlb_odds.json (or missing file) serves as the MLB odds source in this repo. "
            "That file is overwritten when scripts/fetch_mlb_odds.py runs, so past slates are not preserved "
            "here."
        )
        print()
        print(
            "No separate dated MLB JSON snapshots were found under data/odds/ (excluding live_mlb_odds.json). "
            "SQLite does not store full pre-game odds matrices per date — only outcomes/picks on saved rows."
        )
        print()
        print(
            "Honest conclusion: not enough historical MLB odds in this workspace to estimate whether "
            "MAX_FAVORITE_ODDS = -170 would have beaten -160 over time. Add archived copies of "
            "live_mlb_odds.json per card (e.g. data/odds/mlb_archive/2026-05-01.json) and re-run this script."
        )
        return 0

    return run_backtest_on_archives(archives)


if __name__ == "__main__":
    raise SystemExit(main())
