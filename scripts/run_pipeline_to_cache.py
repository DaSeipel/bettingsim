#!/usr/bin/env python3
"""
Run the value-plays pipeline and write results to data/cache/value_plays_cache.json.
Call this from the dashboard Refresh button (subprocess) or from cron; do not run the pipeline inside Streamlit.
"""
import os
import re
import sys
from pathlib import Path

# Run from project root so engine and strategies are importable
APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
os.chdir(APP_ROOT)

from engine.value_plays_pipeline import run_pipeline_to_cache


def _get_api_key() -> str:
    key = (os.environ.get("ODDS_API_KEY") or "").strip()
    if not key:
        secrets_path = APP_ROOT / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            try:
                text = secrets_path.read_text()
                m = re.search(r'api_key\s*=\s*["\']([^"\']+)["\']', text)
                if m:
                    key = m.group(1).strip()
            except Exception:
                pass
    return key


def main() -> None:
    api_key = _get_api_key()
    if not api_key:
        print("No Odds API key. Set ODDS_API_KEY or add to .streamlit/secrets.toml [the_odds_api] api_key = \"...\"", file=sys.stderr)
        sys.exit(1)
    cache_path = APP_ROOT / "data" / "cache" / "value_plays_cache.json"
    run_pipeline_to_cache(api_key, cache_path, app_root=APP_ROOT)
    print("Wrote", cache_path)


if __name__ == "__main__":
    main()
