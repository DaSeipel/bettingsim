"""
Supabase sync for MLB picks. Credentials from Streamlit secrets:

  [supabase]
  url = "https://xxxx.supabase.co"
  key = "service_role_or_anon_key"

Table `mlb_picks` (create in Supabase SQL), example:

  create table mlb_picks (
    id text primary key,
    card_date date not null,
    home_team text not null,
    away_team text not null,
    home_pitcher text,
    away_pitcher text,
    market text not null,
    selection text not null,
    odds_american double precision,
    model_prob double precision,
    edge_pct double precision,
    total_line double precision,
    updated_at timestamptz default now()
  );

Use get_supabase_client() for a single shared client (lazy init from st.secrets).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

import pandas as pd

TABLE_MLB_PICKS = "mlb_picks"

# Keys sent to PostgREST for mlb_picks (matches normalize_mlb_pick_row output + updated_at).
# Your listed fields are all included; also: id (PK), market, model_prob, total_line, updated_at.
MLB_PICK_UPSERT_KEYS: tuple[str, ...] = (
    "id",
    "card_date",
    "home_team",
    "away_team",
    "home_pitcher",
    "away_pitcher",
    "market",
    "selection",
    "odds_american",
    "model_prob",
    "edge_pct",
    "total_line",
    "updated_at",
)

_supabase_client: Any | None = None


def _read_supabase_url_and_key() -> tuple[str, str]:
    """Load Supabase URL and key from st.secrets ([supabase] url/key or SUPABASE_URL / SUPABASE_KEY)."""
    try:
        import streamlit as st
    except ImportError as e:
        raise RuntimeError("streamlit is required to read secrets; pass client= explicitly") from e

    url: str | None = None
    key: str | None = None

    # Preferred: [supabase] section in .streamlit/secrets.toml
    if hasattr(st, "secrets") and st.secrets:
        try:
            sb = st.secrets["supabase"]
            url = str(sb["url"]).strip()
            key = str(sb["key"]).strip()
        except Exception:
            try:
                sb = st.secrets.get("supabase") if hasattr(st.secrets, "get") else None
                if sb is not None:
                    url = str(sb.get("url") or sb["url"]).strip()
                    key = str(sb.get("key") or sb["key"]).strip()
            except Exception:
                pass
        if not url:
            try:
                url = str(st.secrets.get("SUPABASE_URL", "") or "").strip() or None
            except Exception:
                url = None
        if not key:
            try:
                key = str(st.secrets.get("SUPABASE_KEY", "") or "").strip() or None
            except Exception:
                key = None

    if not url or not key:
        raise ValueError(
            "Missing Supabase config. Add [supabase] with url and key to .streamlit/secrets.toml "
            "(or SUPABASE_URL / SUPABASE_KEY)."
        )
    return url, key


def get_supabase_client() -> Any:
    """
    Single shared Supabase client (lazy init). Safe to call on every button click;
    the connection is reused after the first successful create_client.
    """
    global _supabase_client
    if _supabase_client is None:
        from supabase import create_client

        url, key = _read_supabase_url_and_key()
        _supabase_client = create_client(url, key)
    return _supabase_client


def reset_supabase_client() -> None:
    """Clear cached client (e.g. after secret rotation in tests)."""
    global _supabase_client
    _supabase_client = None


def _scalar(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    if hasattr(v, "item"):  # numpy scalar
        try:
            return v.item()
        except Exception:
            pass
    return v


def normalize_mlb_pick_row(p: Mapping[str, Any]) -> dict[str, Any]:
    """
    Build one row dict matching mlb_picks columns. Accepts cache keys including legacy `edge`
    (decimal EV); maps to `edge_pct` as percentage points, e.g. 0.1122 -> 11.22.
    If `edge_pct` is already set (e.g. 11.22), it is stored as-is.
    """
    d = {k: _scalar(v) for k, v in dict(p).items()}

    edge_pct: float | None = None
    if d.get("edge_pct") is not None:
        try:
            edge_pct = float(d["edge_pct"])
        except (TypeError, ValueError):
            edge_pct = None
    elif d.get("edge") is not None:
        try:
            edge_pct = float(d["edge"]) * 100.0
        except (TypeError, ValueError):
            edge_pct = None

    def _f(name: str) -> float | None:
        v = d.get(name)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _s(name: str) -> str | None:
        v = d.get(name)
        if v is None:
            return None
        t = str(v).strip()
        return t if t else None

    cd = d.get("card_date")
    card_date = ""
    if cd is not None:
        s = str(cd).strip()
        card_date = s[:10] if len(s) >= 10 else s

    out: dict[str, Any] = {
        "id": str(d.get("id") or "").strip(),
        "card_date": card_date,
        "home_team": str(d.get("home_team") or "").strip(),
        "away_team": str(d.get("away_team") or "").strip(),
        "home_pitcher": _s("home_pitcher"),
        "away_pitcher": _s("away_pitcher"),
        "market": str(d.get("market") or "").strip(),
        "selection": str(d.get("selection") or "").strip(),
        "odds_american": _f("odds_american"),
        "model_prob": _f("model_prob"),
        "edge_pct": edge_pct,
        "total_line": _f("total_line"),
    }
    return out


def mlb_rows_from_dataframe(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert the current MLB value-plays DataFrame to normalized rows for upsert."""
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        rows.append(normalize_mlb_pick_row(r.to_dict()))
    return rows


def upsert_mlb_pick(
    pick_data: Mapping[str, Any],
    client: Any | None = None,
    table: str = TABLE_MLB_PICKS,
) -> dict[str, Any]:
    """
    Upsert one MLB pick row. Keys should match mlb_picks (use edge_pct, or legacy edge as decimal EV).
    """
    if client is None:
        client = get_supabase_client()

    row = normalize_mlb_pick_row(pick_data)
    row["updated_at"] = datetime.now(timezone.utc).isoformat()

    res = client.table(table).upsert(row, on_conflict="id").execute()
    return getattr(res, "data", res)


def upsert_mlb_card(picks: list[Mapping[str, Any]], client: Any | None = None, table: str = TABLE_MLB_PICKS) -> Any:
    """
    Upsert a full daily card in one request (each row must include primary key `id`).

    Each row is normalized to match `mlb_picks` columns, including exactly:
    card_date, home_team, away_team, home_pitcher, away_pitcher, selection,
    odds_american, edge_pct (plus id, market, model_prob, total_line, updated_at).
    Legacy `edge` (decimal EV) in the payload is converted to edge_pct (% points).
    """
    if not picks:
        return []
    if client is None:
        client = get_supabase_client()
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for p in picks:
        row = normalize_mlb_pick_row(p)
        row["updated_at"] = now
        rows.append(row)
    res = client.table(table).upsert(rows, on_conflict="id").execute()
    return res


def upsert_mlb_card_from_dataframe(df: pd.DataFrame, client: Any | None = None, table: str = TABLE_MLB_PICKS) -> Any:
    """Upsert from the on-screen MLB value-plays DataFrame (same rows the user sees)."""
    return upsert_mlb_card(mlb_rows_from_dataframe(df), client=client, table=table)
