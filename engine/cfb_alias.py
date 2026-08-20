"""Exact + prefix alias matching for CFBD vs Odds API names. No API calls."""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from typing import Any


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("'", "").replace("’", "").replace(".", "")
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prefix_candidates(cfbd_name: str, odds_names: list[str], *, normalized: bool) -> list[str]:
    hits: list[str] = []
    if normalized:
        prefix = normalize_name(cfbd_name) + " "
        for odds in odds_names:
            if normalize_name(odds).startswith(prefix):
                hits.append(odds)
        return hits
    prefix = cfbd_name + " "
    return [odds for odds in odds_names if odds.startswith(prefix)]


def match_aliases(
    cfbd_names: list[str],
    odds_names: list[str],
) -> dict[str, Any]:
    """
    Match order: exact, Odds starts with CFBD+' ', then same after normalize.
    Ambiguity is evaluated against the full Odds list (not remaining after other assigns).
    """
    cfbd_names = sorted({n.strip() for n in cfbd_names if n and n.strip()})
    odds_names = sorted({n.strip() for n in odds_names if n and n.strip()})
    odds_set = set(odds_names)

    assigned: dict[str, tuple[str, str]] = {}
    ambiguous: list[dict[str, Any]] = []
    used_odds: set[str] = set()
    counts = {"exact": 0, "prefix": 0, "normalized_prefix": 0}

    for cfbd in cfbd_names:
        if cfbd in odds_set:
            assigned[cfbd] = (cfbd, "exact")
            used_odds.add(cfbd)
            counts["exact"] += 1

    for cfbd in cfbd_names:
        if cfbd in assigned:
            continue
        hits = prefix_candidates(cfbd, odds_names, normalized=False)
        if len(hits) > 1:
            ambiguous.append({"cfbd_name": cfbd, "rule": "prefix", "candidates": hits})
            continue
        if len(hits) == 1:
            odds = hits[0]
            if odds in used_odds:
                continue
            assigned[cfbd] = (odds, "prefix")
            used_odds.add(odds)
            counts["prefix"] += 1

    for cfbd in cfbd_names:
        if cfbd in assigned or any(item["cfbd_name"] == cfbd for item in ambiguous):
            continue
        hits = prefix_candidates(cfbd, odds_names, normalized=True)
        if len(hits) > 1:
            ambiguous.append({"cfbd_name": cfbd, "rule": "normalized_prefix", "candidates": hits})
            continue
        if len(hits) == 1:
            odds = hits[0]
            if odds in used_odds:
                continue
            assigned[cfbd] = (odds, "normalized_prefix")
            used_odds.add(odds)
            counts["normalized_prefix"] += 1

    unmatched_cfbd = sorted(n for n in cfbd_names if n not in assigned and n not in {a["cfbd_name"] for a in ambiguous})
    unmatched_odds = sorted(odds_set - used_odds)
    return {
        "assigned": assigned,
        "counts": counts,
        "ambiguous": ambiguous,
        "unmatched_cfbd": unmatched_cfbd,
        "unmatched_odds": unmatched_odds,
    }


def persist_aliases(
    conn: sqlite3.Connection,
    *,
    match_result: dict[str, Any],
) -> None:
    from engine.cfb_schema import ensure_cfb_schema

    ensure_cfb_schema(conn)
    assigned: dict[str, tuple[str, str]] = match_result["assigned"]
    ambiguous_names = {item["cfbd_name"] for item in match_result["ambiguous"]}
    rows = conn.execute(
        "SELECT canonical_name, cfbd_name FROM cfb_team_alias"
    ).fetchall()
    for canonical, cfbd_name in rows:
        odds_name, method = (None, None)
        if canonical in assigned:
            odds_name, method = assigned[canonical]
        elif cfbd_name in assigned:
            odds_name, method = assigned[cfbd_name]
        elif canonical in ambiguous_names or cfbd_name in ambiguous_names:
            method = "ambiguous"
        conn.execute(
            """
            UPDATE cfb_team_alias
            SET odds_api_name = ?, match_method = ?
            WHERE canonical_name = ?
            """,
            (odds_name, method, canonical),
        )
    conn.commit()
