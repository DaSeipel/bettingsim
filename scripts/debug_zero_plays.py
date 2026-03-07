#!/usr/bin/env python3
"""
Debug why the dashboard shows 0 plays:
(1) Run odds fetch and print game counts for basketball_nba and basketball_ncaab.
(2) and (3) Raw edge and threshold: set DEBUG_VALUE_PLAYS=1 and run the app (see below).
(4) Check NCAAB model file exists and loads correctly.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _get_api_key() -> str:
    key = (os.environ.get("ODDS_API_KEY") or "").strip()
    if key:
        return key
    secrets = ROOT / ".streamlit" / "secrets.toml"
    if secrets.exists():
        try:
            text = secrets.read_text()
            m = re.search(r'\[the_odds_api\].*?api_key\s*=\s*["\']([^"\']+)["\']', text, re.DOTALL)
            if m:
                return m.group(1).strip()
            m = re.search(r'ODDS_API_KEY\s*=\s*["\']([^"\']+)["\']', text)
            if m:
                return m.group(1).strip()
        except Exception:
            pass
    return ""


def main() -> None:
    from engine.engine import get_live_odds, BASKETBALL_NBA, BASKETBALL_NCAAB

    api_key = _get_api_key()
    if not api_key:
        print("No ODDS_API_KEY found. Set env ODDS_API_KEY or add to .streamlit/secrets.toml")
        print("Continuing with (4) NCAAB model check only.\n")

    # (1) Odds fetch and game counts
    print("=" * 60)
    print("(1) ODDS FETCH — games from The Odds API (today, US Eastern)")
    print("=" * 60)
    if api_key:
        try:
            df = get_live_odds(
                api_key=api_key,
                sport_keys=[BASKETBALL_NBA, BASKETBALL_NCAAB],
                commence_on_date=None,
                display_timezone="America/New_York",
            )
            if df.empty:
                print("  No rows returned. Check API key, quota, and that games are scheduled today (ET).")
            else:
                for sport_key in [BASKETBALL_NBA, BASKETBALL_NCAAB]:
                    sub = df[df["sport_key"] == sport_key]
                    events = sub.drop_duplicates(subset=["event_id"]) if "event_id" in sub.columns else sub
                    n_events = len(events)
                    n_rows = len(sub)
                    label = "basketball_nba (NBA)" if sport_key == BASKETBALL_NBA else "basketball_ncaab (NCAAB)"
                    print("  {}: {} games ({} outcome rows)".format(label, n_events, n_rows))
                print("  Total outcome rows (all markets): {}".format(len(df)))
        except Exception as e:
            print("  Error fetching odds: {}".format(e))
    else:
        print("  Skipped (no API key).")

    # (4) NCAAB model
    print()
    print("=" * 60)
    print("(4) NCAAB MODEL — file exists and load")
    print("=" * 60)
    from engine.betting_models import (
        SPREAD_MODEL_PATH_NCAAB,
        TOTALS_MODEL_PATH_NCAAB,
        MONEYLINE_MODEL_PATH_NCAAB,
        load_model,
        _spread_model_path_for_league,
    )

    for name, path in [
        ("Spread", SPREAD_MODEL_PATH_NCAAB),
        ("Totals", TOTALS_MODEL_PATH_NCAAB),
        ("Moneyline", MONEYLINE_MODEL_PATH_NCAAB),
    ]:
        exists = path.exists()
        payload = load_model(path) if exists else None
        ok = payload is not None and payload.get("model") is not None
        print("  {}: {}  exists={}  load_ok={}".format(name, path.name, exists, ok))
    path_used = _spread_model_path_for_league("ncaab")
    print("  _spread_model_path_for_league('ncaab') -> {} (exists={})".format(path_used.name, path_used.exists()))

    print()
    print("=" * 60)
    print("(2) & (3) — Raw edge and threshold pass/fail")
    print("=" * 60)
    print("  Run the app with debug env and check the terminal for:")
    print("  - [DEBUG_VALUE_PLAYS] Thresholds: min_ev_pct, max_ev_pct, MIN_BOOKMAKERS_VALUE_PLAY")
    print("  - [DEBUG] RAW_EDGE: league | event | market | selection | model_prob | implied_prob | raw_edge_pct")
    print("  - [DEBUG_VALUE_PLAYS] Summary: aggregated_rows | skip_* | below_min_ev | above_max_ev | PASSED")
    print()
    print("  Command:")
    print("    DEBUG_VALUE_PLAYS=1 python -m streamlit run app.py --server.headless true")
    print("  Then open the app and look at the terminal where Streamlit is running.")
    print()


if __name__ == "__main__":
    main()
