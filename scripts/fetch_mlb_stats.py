#!/usr/bin/env python3
"""
Fetch MLB team-level stats from the public MLB Stats API and save to data/mlb/team_stats.csv.

Columns: team_id, team_name, season, wins, losses, win_pct, run_differential,
era, whip, woba, runs_scored_per_game (team runs / games played from hitting + standings),
bullpen_era
- wins / losses / win_pct / run_differential: regular-season standings via
  /standings?leagueId=103,104&standingsTypes=regularSeason (joined by team_id).
- era / whip: team season pitching (all pitchers).
- woba: FanGraphs-style linear weights from team hitting counting stats (API does not expose wOBA directly).
- bullpen_era: ERA aggregated over roster pitchers classified as relievers (gamesStarted < 0.5 * gamesPitched),
  using per-player season splits for the given team only (handles mid-season trades).

Usage:
  python3 scripts/fetch_mlb_stats.py
  python3 scripts/fetch_mlb_stats.py --season 2025
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

REQUEST_TIMEOUT = 30
REQUEST_SLEEP_S = 0.12
BASE = "https://statsapi.mlb.com/api/v1"
APP_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = APP_ROOT / "data" / "mlb" / "team_stats.csv"
OUTPUT_2025_PATH = APP_ROOT / "data" / "mlb" / "team_stats_2025.csv"
OUTPUT_2026_PATH = APP_ROOT / "data" / "mlb" / "team_stats_2026.csv"

# When blending 2025 + 2026 stats, how much weight goes to current (2026) season.
BLEND_2026_WEIGHT = 0.50  # Increase toward 0.75-0.80 by mid-June as 2026 samples grow
BLEND_NUMERIC_COLS = (
    "era",
    "whip",
    "woba",
    "bullpen_era",
    "win_pct",
    "run_differential",
    "runs_scored_per_game",
)


def _get_json(url: str) -> dict:
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def parse_innings_to_outs(ip) -> int:
    """MLB API returns innings as '1442.1' (thirds of an inning after the dot)."""
    if ip is None:
        return 0
    if isinstance(ip, (int, float)):
        whole = int(ip)
        frac = ip - whole
        thirds = int(round(frac * 10)) % 10
        if thirds > 2:
            thirds = thirds % 3
        return whole * 3 + thirds
    s = str(ip).strip()
    if not s:
        return 0
    if "." in s:
        whole, frac = s.split(".", 1)
        w = int(whole) if whole.isdigit() else 0
        f = int(frac[0]) if frac and frac[0].isdigit() else 0
        if f > 2:
            f = f % 3
        return w * 3 + f
    return int(s) * 3 if s.isdigit() else 0


def woba_from_hitting_stat(st: dict) -> float:
    """wOBA from team hitting counting stats (2024-style weights, close enough for cross-team rank)."""
    pa = float(st.get("plateAppearances") or 0)
    if pa <= 0:
        return float("nan")
    walks = float(st.get("baseOnBalls") or 0)
    ibb = float(st.get("intentionalWalks") or 0)
    ubb = max(0.0, walks - ibb)
    hbp = float(st.get("hitByPitch") or 0)
    hits = float(st.get("hits") or 0)
    doubles = float(st.get("doubles") or 0)
    triples = float(st.get("triples") or 0)
    hr = float(st.get("homeRuns") or 0)
    singles = hits - doubles - triples - hr
    if singles < 0:
        singles = 0.0
    num = (
        0.69 * ubb
        + 0.72 * hbp
        + 0.89 * singles
        + 1.27 * doubles
        + 1.62 * triples
        + 2.10 * hr
        + 0.77 * ibb
    )
    return num / pa


def team_pitching_row(team_id: int, season: int) -> dict:
    url = f"{BASE}/teams/{team_id}/stats?stats=season&group=pitching&season={season}"
    data = _get_json(url)
    splits = (data.get("stats") or [{}])[0].get("splits") or []
    if not splits:
        return {"era": float("nan"), "whip": float("nan")}
    st = splits[0].get("stat") or {}
    era_s = st.get("era")
    whip_s = st.get("whip")
    try:
        era = float(era_s) if era_s is not None and str(era_s).strip() != "" else float("nan")
    except (TypeError, ValueError):
        era = float("nan")
    try:
        whip = float(whip_s) if whip_s is not None and str(whip_s).strip() != "" else float("nan")
    except (TypeError, ValueError):
        whip = float("nan")
    return {"era": era, "whip": whip}


def team_hitting_woba_and_runs(team_id: int, season: int) -> tuple[float, float]:
    """Team season wOBA and runs scored from one hitting stats request (stat.runs from MLB API)."""
    url = f"{BASE}/teams/{team_id}/stats?stats=season&group=hitting&season={season}"
    data = _get_json(url)
    splits = (data.get("stats") or [{}])[0].get("splits") or []
    if not splits:
        return float("nan"), float("nan")
    st = splits[0].get("stat") or {}
    woba = woba_from_hitting_stat(st)
    rs_raw = st.get("runs") if st.get("runs") is not None else st.get("runsScored")
    try:
        runs = float(rs_raw) if rs_raw is not None and str(rs_raw).strip() != "" else float("nan")
    except (TypeError, ValueError):
        runs = float("nan")
    return woba, runs


def is_reliever_split(stat: dict) -> bool:
    gs = int(stat.get("gamesStarted") or 0)
    gp = int(stat.get("gamesPitched") or stat.get("gamesPlayed") or 0)
    if gp <= 0:
        return False
    # Starters: meaningful GS share or high GS count; else count toward bullpen aggregate.
    starter_threshold = max(8, int(0.5 * gp))
    return gs < starter_threshold


def bullpen_era_for_team(team_id: int, season: int) -> float:
    roster_url = f"{BASE}/teams/{team_id}/roster/active"
    roster = _get_json(roster_url)
    entries = roster.get("roster") or []
    pitcher_ids: list[int] = []
    for e in entries:
        pos = (e.get("position") or {}).get("abbreviation")
        if pos != "P":
            continue
        pid = (e.get("person") or {}).get("id")
        if pid is not None:
            pitcher_ids.append(int(pid))

    total_outs = 0
    total_er = 0
    for pid in pitcher_ids:
        time.sleep(REQUEST_SLEEP_S)
        url = f"{BASE}/people/{pid}/stats?stats=season&group=pitching&season={season}"
        try:
            data = _get_json(url)
        except requests.RequestException:
            continue
        for block in data.get("stats") or []:
            for sp in block.get("splits") or []:
                tm = sp.get("team") or {}
                if int(tm.get("id") or 0) != team_id:
                    continue
                st = sp.get("stat") or {}
                if not is_reliever_split(st):
                    continue
                outs = int(st.get("outs") or 0)
                if outs <= 0:
                    outs = parse_innings_to_outs(st.get("inningsPitched"))
                er = int(st.get("earnedRuns") or 0)
                total_outs += outs
                total_er += er

    if total_outs <= 0:
        return float("nan")
    return (total_er * 27.0) / float(total_outs)  # ERA = 9 * ER / IP, IP = outs/3


def standings_map_by_team_id(season: int) -> dict[int, dict[str, int | float | None]]:
    """Regular-season wins/losses and run differential keyed by MLB team id."""
    time.sleep(REQUEST_SLEEP_S)
    url = f"{BASE}/standings?leagueId=103,104&season={season}&standingsTypes=regularSeason"
    data = _get_json(url)
    out: dict[int, dict[str, int | float | None]] = {}
    for rec in data.get("records") or []:
        for tr in rec.get("teamRecords") or []:
            tm = tr.get("team") or {}
            tid = int(tm.get("id") or 0)
            if not tid:
                continue
            wins = int(tr.get("wins") or 0)
            losses = int(tr.get("losses") or 0)
            gp = wins + losses
            rd_raw = tr.get("runDifferential")
            try:
                rd = int(rd_raw) if rd_raw is not None and str(rd_raw) != "" else None
            except (TypeError, ValueError):
                rd = None
            out[tid] = {
                "wins": wins,
                "losses": losses,
                "run_differential": rd,
            }
    return out


def fetch_mlb_team_stats(season: int) -> pd.DataFrame:
    time.sleep(REQUEST_SLEEP_S)
    teams_data = _get_json(f"{BASE}/teams?sportId=1&season={season}")
    teams = [t for t in teams_data.get("teams", []) if t.get("sport", {}).get("id") == 1]
    wl_by_tid = standings_map_by_team_id(season)
    rows: list[dict] = []
    for t in sorted(teams, key=lambda x: x.get("name", "")):
        tid = int(t["id"])
        name = str(t.get("name", ""))
        time.sleep(REQUEST_SLEEP_S)
        pitch = team_pitching_row(tid, season)
        time.sleep(REQUEST_SLEEP_S)
        woba, runs_scored = team_hitting_woba_and_runs(tid, season)
        time.sleep(REQUEST_SLEEP_S)
        bp_era = bullpen_era_for_team(tid, season)
        wl = wl_by_tid.get(tid) or {}
        w = int(wl.get("wins") or 0)
        l_ = int(wl.get("losses") or 0)
        gp = w + l_
        wp_f = float(w / gp) if gp > 0 else float("nan")
        rd = wl.get("run_differential")
        if gp > 0 and runs_scored == runs_scored:
            runs_pg = round(runs_scored / float(gp), 5)
        else:
            runs_pg = float("nan")
        rows.append(
            {
                "team_id": tid,
                "team_name": name,
                "season": season,
                "wins": w,
                "losses": l_,
                "win_pct": round(wp_f, 5) if wp_f == wp_f else float("nan"),
                "run_differential": rd if rd is not None else float("nan"),
                "era": pitch["era"],
                "whip": pitch["whip"],
                "woba": round(woba, 4) if woba == woba else float("nan"),
                "runs_scored_per_game": runs_pg,
                "bullpen_era": round(bp_era, 2) if bp_era == bp_era else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def fetch_mlb_team_stats_light(season: int) -> pd.DataFrame:
    """Like fetch_mlb_team_stats but skips the expensive bullpen_era roster lookup.
    Uses team-level ERA as bullpen_era placeholder (good enough for early-season 2026)."""
    time.sleep(REQUEST_SLEEP_S)
    teams_data = _get_json(f"{BASE}/teams?sportId=1&season={season}")
    teams = [t for t in teams_data.get("teams", []) if t.get("sport", {}).get("id") == 1]
    wl_by_tid = standings_map_by_team_id(season)
    rows: list[dict] = []
    for t in sorted(teams, key=lambda x: x.get("name", "")):
        tid = int(t["id"])
        name = str(t.get("name", ""))
        time.sleep(REQUEST_SLEEP_S)
        pitch = team_pitching_row(tid, season)
        time.sleep(REQUEST_SLEEP_S)
        woba, runs_scored = team_hitting_woba_and_runs(tid, season)
        wl = wl_by_tid.get(tid) or {}
        w = int(wl.get("wins") or 0)
        l_ = int(wl.get("losses") or 0)
        gp = w + l_
        wp_f = float(w / gp) if gp > 0 else float("nan")
        rd = wl.get("run_differential")
        if gp > 0 and runs_scored == runs_scored:
            runs_pg = round(runs_scored / float(gp), 5)
        else:
            runs_pg = float("nan")
        rows.append(
            {
                "team_id": tid,
                "team_name": name,
                "season": season,
                "wins": w,
                "losses": l_,
                "win_pct": round(wp_f, 5) if wp_f == wp_f else float("nan"),
                "run_differential": rd if rd is not None else float("nan"),
                "era": pitch["era"],
                "whip": pitch["whip"],
                "woba": round(woba, 4) if woba == woba else float("nan"),
                "runs_scored_per_game": runs_pg,
                "bullpen_era": pitch["era"],  # team ERA as proxy
            }
        )
    return pd.DataFrame(rows)


def blend_team_stats(df_2025: pd.DataFrame, df_2026: pd.DataFrame, w26: float = BLEND_2026_WEIGHT) -> pd.DataFrame:
    """Blend 2025 and 2026 per-team numeric stats. If a team has 0 games in 2026, use 100% 2025."""
    w25 = 1.0 - w26
    merged = df_2025.merge(df_2026, on="team_id", suffixes=("_25", "_26"), how="left")
    rows: list[dict] = []
    for _, r in merged.iterrows():
        gp_26 = int(r.get("wins_26") or 0) + int(r.get("losses_26") or 0)
        has_2026 = gp_26 > 0

        row: dict = {
            "team_id": int(r["team_id"]),
            "team_name": r.get("team_name_25") or r.get("team_name_26") or "",
            "season": "blend",
            "wins": int(r.get("wins_26") or r.get("wins_25") or 0),
            "losses": int(r.get("losses_26") or r.get("losses_25") or 0),
        }
        for col in BLEND_NUMERIC_COLS:
            v25 = r.get(f"{col}_25")
            v26 = r.get(f"{col}_26")
            v25_ok = v25 is not None and v25 == v25  # not NaN
            v26_ok = v26 is not None and v26 == v26 and has_2026
            if v25_ok and v26_ok:
                row[col] = round(w25 * float(v25) + w26 * float(v26), 5)
            elif v25_ok:
                row[col] = float(v25)
            elif v26_ok:
                row[col] = float(v26)
            else:
                row[col] = float("nan")
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch MLB team stats into data/mlb/team_stats.csv")
    p.add_argument("--season", type=int, default=None, help="Season year (default: latest completed MLB season heuristic)")
    p.add_argument(
        "--season-blend",
        action="store_true",
        help="Fetch both 2025 and 2026, blend into team_stats.csv (saves per-year CSVs too).",
    )
    args = p.parse_args()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.season_blend:
        print("Fetching 2025 team stats (full, with bullpen roster lookup)...")
        df_25 = fetch_mlb_team_stats(2025)
        df_25.to_csv(OUTPUT_2025_PATH, index=False)
        print(f"Saved {len(df_25)} teams to {OUTPUT_2025_PATH}")

        print("Fetching 2026 team stats (light, no bullpen roster lookup)...")
        df_26 = fetch_mlb_team_stats_light(2026)
        df_26.to_csv(OUTPUT_2026_PATH, index=False)
        print(f"Saved {len(df_26)} teams to {OUTPUT_2026_PATH}")

        blended = blend_team_stats(df_25, df_26, w26=BLEND_2026_WEIGHT)
        blended.to_csv(OUTPUT_PATH, index=False)
        print(
            f"Blended {len(blended)} teams (weight: {BLEND_2026_WEIGHT:.0%} 2026 / "
            f"{1 - BLEND_2026_WEIGHT:.0%} 2025) → {OUTPUT_PATH}"
        )
        return

    season = args.season
    if season is None:
        from datetime import date

        today = date.today()
        if today.month >= 11:
            season = today.year
        elif today.month < 4:
            season = today.year - 1
        else:
            season = today.year

    print(f"Fetching MLB team stats for season={season} (this may take several minutes due to roster lookups)...")
    df = fetch_mlb_team_stats(season)
    if df.empty:
        print("No teams returned.")
        return
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} teams to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
