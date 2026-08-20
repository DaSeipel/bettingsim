"""CFB Phase 0 SQLite schema for data/espn.db."""

from __future__ import annotations

import sqlite3

CFB_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS cfb_games (
    game_id INTEGER PRIMARY KEY,
    season INTEGER NOT NULL,
    week INTEGER,
    season_type TEXT,
    start_date TEXT,
    home_team TEXT,
    away_team TEXT,
    home_points INTEGER,
    away_points INTEGER,
    venue_id INTEGER,
    neutral_site INTEGER,
    conference_game INTEGER,
    home_conference TEXT,
    away_conference TEXT,
    home_division TEXT,
    away_division TEXT
);

CREATE TABLE IF NOT EXISTS cfb_game_stats (
    game_id INTEGER NOT NULL,
    team TEXT NOT NULL,
    is_home INTEGER NOT NULL,
    points INTEGER,
    stats_json TEXT,
    PRIMARY KEY (game_id, team)
);

CREATE TABLE IF NOT EXISTS cfb_lines (
    game_id INTEGER NOT NULL,
    provider TEXT NOT NULL,
    spread REAL,
    spread_open REAL,
    over_under REAL,
    home_moneyline REAL,
    away_moneyline REAL,
    captured_at TEXT NOT NULL,
    is_backtest_reference INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (game_id, provider)
);

CREATE TABLE IF NOT EXISTS cfb_team_stats_adv (
    season INTEGER NOT NULL,
    team TEXT NOT NULL,
    conference TEXT,
    offense_json TEXT,
    defense_json TEXT,
    stats_json TEXT,
    PRIMARY KEY (season, team)
);

CREATE TABLE IF NOT EXISTS cfb_ppa (
    season INTEGER NOT NULL,
    team TEXT NOT NULL,
    conference TEXT,
    off_ppa_overall REAL,
    off_ppa_pass REAL,
    off_ppa_rush REAL,
    def_ppa_overall REAL,
    def_ppa_pass REAL,
    def_ppa_rush REAL,
    ppa_json TEXT,
    PRIMARY KEY (season, team)
);

CREATE TABLE IF NOT EXISTS cfb_ratings_sp (
    season INTEGER NOT NULL,
    team TEXT NOT NULL,
    conference TEXT,
    rating REAL,
    offense REAL,
    defense REAL,
    special REAL,
    rating_scope TEXT NOT NULL,
    sp_json TEXT,
    PRIMARY KEY (season, team, rating_scope)
);

CREATE TABLE IF NOT EXISTS cfb_returning (
    season INTEGER NOT NULL,
    team TEXT NOT NULL,
    conference TEXT,
    total_ppa REAL,
    total_passing_ppa REAL,
    total_receiving_ppa REAL,
    total_rushing_ppa REAL,
    percent_ppa REAL,
    percent_passing_ppa REAL,
    percent_receiving_ppa REAL,
    percent_rushing_ppa REAL,
    usage REAL,
    passing_usage REAL,
    receiving_usage REAL,
    rushing_usage REAL,
    returning_json TEXT,
    PRIMARY KEY (season, team)
);

CREATE TABLE IF NOT EXISTS cfb_venues (
    venue_id INTEGER PRIMARY KEY,
    name TEXT,
    city TEXT,
    state TEXT,
    elevation REAL,
    latitude REAL,
    longitude REAL,
    capacity INTEGER,
    dome INTEGER,
    venue_json TEXT
);

CREATE TABLE IF NOT EXISTS cfb_team_alias (
    canonical_name TEXT PRIMARY KEY,
    cfbd_name TEXT,
    odds_api_name TEXT,
    espn_name TEXT,
    conference_2026 TEXT,
    match_method TEXT
);

CREATE TABLE IF NOT EXISTS cfb_season_flags (
    season INTEGER PRIMARY KEY,
    is_anomalous INTEGER NOT NULL DEFAULT 0,
    exclude_from_hfa INTEGER NOT NULL DEFAULT 0,
    exclude_from_training INTEGER NOT NULL DEFAULT 0,
    note TEXT
);

CREATE INDEX IF NOT EXISTS idx_cfb_games_season_week ON cfb_games(season, week);
CREATE INDEX IF NOT EXISTS idx_cfb_games_season_team_home ON cfb_games(season, home_team);
CREATE INDEX IF NOT EXISTS idx_cfb_games_season_team_away ON cfb_games(season, away_team);
CREATE INDEX IF NOT EXISTS idx_cfb_game_stats_season_lookup ON cfb_game_stats(game_id);
CREATE INDEX IF NOT EXISTS idx_cfb_lines_game_id ON cfb_lines(game_id);
CREATE INDEX IF NOT EXISTS idx_cfb_team_stats_adv_season_team ON cfb_team_stats_adv(season, team);
CREATE INDEX IF NOT EXISTS idx_cfb_ppa_season_team ON cfb_ppa(season, team);
CREATE INDEX IF NOT EXISTS idx_cfb_ratings_sp_season_team ON cfb_ratings_sp(season, team);
CREATE INDEX IF NOT EXISTS idx_cfb_returning_season_team ON cfb_returning(season, team);
CREATE UNIQUE INDEX IF NOT EXISTS idx_cfb_team_alias_odds_api_name
    ON cfb_team_alias(odds_api_name) WHERE odds_api_name IS NOT NULL;
"""


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def ensure_cfb_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(CFB_TABLES_SQL)
    _ensure_column(conn, "cfb_team_alias", "match_method", "match_method TEXT")
    _ensure_column(conn, "cfb_lines", "is_backtest_reference", "is_backtest_reference INTEGER NOT NULL DEFAULT 0")
    conn.commit()
