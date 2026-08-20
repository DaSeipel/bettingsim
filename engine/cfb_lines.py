"""CFB closing-line normalization — spread_home and opener flags."""

from __future__ import annotations

import sqlite3

# DraftKings 2023 opening spreads were systematically unreliable in Phase 0 audit.
_DK_2023_OPENER_SUSPECT = (
    "DraftKings 2023 opener spreads flagged in Phase 0 provider audit; exclude from opener analysis."
)


def _home_favored_from_moneyline(home_ml: float | None, away_ml: float | None) -> bool | None:
    if home_ml is None or away_ml is None:
        return None
    if home_ml < 0 and away_ml < 0:
        return home_ml < away_ml
    if home_ml < 0:
        return True
    if away_ml < 0:
        return False
    return home_ml < away_ml


def align_betting_spread_to_home(
    spread: float | None,
    home_moneyline: float | None,
    away_moneyline: float | None,
) -> float | None:
    """Provider spread aligned to cfb_games home team (betting convention: negative = home favored)."""
    if spread is None:
        return None
    home_fav_ml = _home_favored_from_moneyline(home_moneyline, away_moneyline)
    if home_fav_ml is None:
        return spread
    home_fav_spread = spread < 0
    if home_fav_ml != home_fav_spread:
        return -spread
    return spread


def betting_spread_to_line_margin(betting_spread: float | None) -> float | None:
    """Convert home betting spread to margin scale (positive = home favored)."""
    if betting_spread is None:
        return None
    return -betting_spread


def populate_spread_home(conn: sqlite3.Connection) -> int:
    """Compute spread_home on margin scale (positive = home favored) for all cfb_lines rows."""
    rows = conn.execute(
        """
        SELECT l.game_id, l.provider, l.spread, l.home_moneyline, l.away_moneyline
        FROM cfb_lines l
        WHERE l.spread IS NOT NULL
        """
    ).fetchall()
    updated = 0
    for game_id, provider, spread, home_ml, away_ml in rows:
        betting = align_betting_spread_to_home(spread, home_ml, away_ml)
        spread_home = betting_spread_to_line_margin(betting)
        conn.execute(
            """
            UPDATE cfb_lines
            SET spread_home = ?
            WHERE game_id = ? AND provider = ?
            """,
            (spread_home, game_id, provider),
        )
        updated += 1
    conn.commit()
    return updated


def populate_opener_suspect(conn: sqlite3.Connection) -> int:
    """Mark suspect opener rows. See _DK_2023_OPENER_SUSPECT."""
    conn.execute("UPDATE cfb_lines SET opener_suspect = 0")
    cur = conn.execute(
        """
        UPDATE cfb_lines
        SET opener_suspect = 1
        WHERE provider = 'DraftKings'
          AND game_id IN (SELECT game_id FROM cfb_games WHERE season = 2023)
        """
    )
    conn.commit()
    return cur.rowcount
