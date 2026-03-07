#!/usr/bin/env python3
"""Run the value-play pipeline with DEBUG_VALUE_PLAYS=1 and write debug log. Mocks streamlit so app can be imported."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ["DEBUG_VALUE_PLAYS"] = "1"

# Mock streamlit so app.py can be loaded without a running server
from unittest.mock import MagicMock

def _ctx():
    m = MagicMock()
    m.__enter__ = MagicMock(return_value=m)
    m.__exit__ = MagicMock(return_value=False)
    return m

st = MagicMock()
st.cache_data = lambda **kw: (lambda f: f)
# Load secrets from .streamlit/secrets.toml so pipeline gets API key
try:
    import re
    _sec = (ROOT / ".streamlit" / "secrets.toml").read_text()
    _m = re.search(r'api_key\s*=\s*["\']([^"\']+)["\']', _sec)
    st.secrets = {"the_odds_api": type("", (), {"api_key": _m.group(1) if _m else ""})(), "ODDS_API_KEY": _m.group(1) if _m else ""} if _m else {}
except Exception:
    st.secrets = {}
st.sidebar = MagicMock()
st.session_state = {}
st.tabs = lambda titles: [_ctx() for _ in (titles if isinstance(titles, list) else [])]
st.columns = lambda n: [_ctx() for _ in range(n if isinstance(n, int) else sum(n))]
st.set_page_config = lambda **kw: None
st.title = st.header = st.subheader = st.caption = st.markdown = st.write = lambda x=None, **kw: None
st.slider = lambda *a, **kw: kw.get("value", 2.0)
st.checkbox = lambda *a, **kw: kw.get("value", False)
st.text_input = lambda *a, **kw: None
st.button = lambda *a, **kw: False
st.dataframe = st.data_editor = lambda x=None, **kw: None
st.selectbox = lambda *a, **kw: (kw.get("options") or [None])[0]
st.radio = lambda *a, **kw: (kw.get("options") or [None])[0]
st.warning = st.info = st.error = st.success = lambda x=None, **kw: None
st.empty = lambda: MagicMock()
st.spinner = _ctx
st.expander = _ctx
st.container = _ctx
st.column_config = MagicMock()
sys.modules["streamlit"] = st

def main():
    from engine.engine import get_live_odds
    from engine.betting_models import load_feature_matrix_for_inference
    from engine.engine import get_nba_team_pace_stats, get_nba_teams_back_to_back
    from datetime import date
    from zoneinfo import ZoneInfo

    # Get API key same way as app
    api_key = os.environ.get("ODDS_API_KEY", "").strip()
    if not api_key and (ROOT / ".streamlit" / "secrets.toml").exists():
        import re
        text = (ROOT / ".streamlit" / "secrets.toml").read_text()
        m = re.search(r'api_key\s*=\s*["\']([^"\']+)["\']', text)
        if m:
            api_key = m.group(1).strip()
    if not api_key:
        print("No API key. Set ODDS_API_KEY or use .streamlit/secrets.toml")
        return

    today_et = date.today().isoformat()
    live_odds_df = get_live_odds(api_key=api_key, sport_keys=["basketball_nba", "basketball_ncaab"], commence_on_date=None)
    if live_odds_df.empty:
        print("No odds returned.")
        return
    aggregated = __import__("app", fromlist=["_aggregate_odds_best_line_avg_implied"])._aggregate_odds_best_line_avg_implied(live_odds_df)
    if aggregated.empty:
        aggregated = live_odds_df
    feature_matrix = load_feature_matrix_for_inference(league=None)
    b2b = set(get_nba_teams_back_to_back(api_key, date.today()))
    pace_stats = get_nba_team_pace_stats()
    from app import _live_odds_to_value_plays, BANKROLL_FOR_STAKES, EV_EPSILON_MIN_PCT, EV_EPSILON_MAX_PCT
    value_plays_df, n_flagged = _live_odds_to_value_plays(
        aggregated,
        bankroll=BANKROLL_FOR_STAKES,
        min_ev_pct=EV_EPSILON_MIN_PCT,
        max_ev_pct=EV_EPSILON_MAX_PCT,
        pace_stats=pace_stats,
        b2b_teams=b2b,
        as_of_date=date.today(),
        feature_matrix=feature_matrix,
        debug=True,
    )
    print("Passed plays:", len(value_plays_df))
    log_path = ROOT / "data" / "logs" / "debug_value_plays.log"
    if log_path.exists():
        print("Log written to", log_path)

if __name__ == "__main__":
    main()
