"""CFBD HTTP client with call counting and polite rate limiting."""

from __future__ import annotations

import json
import time
from typing import Any

import requests

from engine.cfb_config import CFBD_BASE_URL, REQUEST_SLEEP_S, REQUEST_TIMEOUT_S, get_cfbd_api_key


class CFBDClient:
    def __init__(self, api_key: str | None = None, *, sleep_s: float = REQUEST_SLEEP_S) -> None:
        self.api_key = (api_key or get_cfbd_api_key()).strip()
        if not self.api_key:
            raise RuntimeError(
                "CFBD API key missing. Set CFBD_API_KEY or add [cfbd] api_key to .streamlit/secrets.toml"
            )
        self.sleep_s = sleep_s
        self.call_count = 0
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"})

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{CFBD_BASE_URL}{path}"
        if self.sleep_s:
            time.sleep(self.sleep_s)
        response = self.session.get(url, params=params or {}, timeout=REQUEST_TIMEOUT_S)
        self.call_count += 1
        remaining = response.headers.get("X-CallLimit-Remaining")
        if response.status_code != 200:
            raise RuntimeError(
                f"CFBD GET {path} failed ({response.status_code}): {response.text[:500]}"
            )
        if remaining is not None:
            print(f"[cfbd] {path} | calls={self.call_count} remaining={remaining}", flush=True)
        return response.json()

    @staticmethod
    def print_teams_probe(raw: Any) -> None:
        print("=== CFBD /teams?year=2025 probe ===")
        print("top-level type:", type(raw).__name__)
        if isinstance(raw, list):
            print("row count:", len(raw))
            if raw:
                first = raw[0]
                print("first-row keys:", sorted(first.keys()) if isinstance(first, dict) else "n/a")
                print("first-row sample:")
                print(json.dumps(first, indent=2)[:2000])
        else:
            print(json.dumps(raw, indent=2)[:2000])
