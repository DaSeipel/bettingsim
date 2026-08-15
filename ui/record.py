"""Overall and CFB record tab bodies."""

from __future__ import annotations

from datetime import date
from typing import Any, Callable

import pandas as pd
import streamlit as st

from ui.config import season_bounds

# Match the MLB Record tab tracked universe (display/query filter only — do not alter espn.db).
MLB_PIPELINE_START_DATE = date(2026, 5, 3)
MLB_RECORD_MIN_EDGE_PCT = 5.0
MLB_MONEYLINE_SENTINEL = -999.0


def _clean_results(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history.copy()
    frame = history.copy()
    frame["result_clean"] = frame["result"].apply(
        lambda value: str(value).strip().upper() if value is not None and not pd.isna(value) else None
    )
    return frame


def _dedupe_play_history_natural_key(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep last row per play_history UNIQUE key (same game + bet may be re-archived)."""
    if frame.empty:
        return frame
    cols = [
        "date_generated",
        "sport",
        "home_team",
        "away_team",
        "bet_type",
        "recommended_side",
        "spread_or_total",
    ]
    if not all(column in frame.columns for column in cols):
        return frame
    return frame.drop_duplicates(subset=cols, keep="last")


def _filter_mlb_tracked_universe(frame: pd.DataFrame) -> pd.DataFrame:
    """Same filter as the MLB Record tab: ML, edge ≥ 5%, date ≥ 2026-05-03, dedupe keep-last."""
    if frame.empty:
        return frame
    out = _dedupe_play_history_natural_key(frame)
    if "spread_or_total" not in out.columns or "my_edge_pct" not in out.columns:
        return out.iloc[0:0].copy()
    sot = pd.to_numeric(out["spread_or_total"], errors="coerce")
    ml_ok = (sot - MLB_MONEYLINE_SENTINEL).abs() <= 0.01
    edge_ok = pd.to_numeric(out["my_edge_pct"], errors="coerce") >= MLB_RECORD_MIN_EDGE_PCT
    dates = pd.to_datetime(out["date_generated"], errors="coerce").dt.date
    date_ok = dates >= MLB_PIPELINE_START_DATE
    return out[ml_ok & edge_ok & date_ok].copy()


def _fixed_pl(
    frame: pd.DataFrame,
    profit_for_result: Callable[[Any, str, float], float],
    *,
    results: tuple[str, ...] = ("W", "L", "P"),
) -> tuple[pd.DataFrame, float, float]:
    resolved = frame[frame["result_clean"].isin(results)].copy()
    if resolved.empty:
        return resolved, 0.0, 0.0
    resolved["pnl"] = resolved.apply(
        lambda row: profit_for_result(
            row.get("market_odds_at_time"),
            row.get("result_clean"),
            10.0,
        ),
        axis=1,
    )
    total = float(resolved["pnl"].sum())
    wagered = float(len(resolved) * 10)
    return resolved, total, (total / wagered * 100 if wagered else 0.0)


def _sport_mask(history: pd.DataFrame, sport: str) -> pd.Series:
    values = history["sport"].astype(str).str.strip().str.upper()
    return values.str.startswith("NCAAB") if sport == "NCAAB" else values.eq(sport)


def calculate_record_summary(
    history: pd.DataFrame,
    *,
    profit_for_result: Callable[[Any, str, float], float],
    on_date=None,
) -> dict:
    """Calculate the shared current-window combined and per-sport record."""
    on_date = on_date or pd.Timestamp.now(tz="America/New_York").date()
    empty = {
        "frame": pd.DataFrame(),
        "wins": 0,
        "losses": 0,
        "pushes": 0,
        "pl": 0.0,
        "roi": 0.0,
        "sports": {},
        "start_date": None,
        "end_date": on_date,
    }
    required = {"sport", "date_generated", "result"}
    if history.empty or not required.issubset(history.columns):
        return empty

    cleaned = _clean_results(history)
    generated_dates = pd.to_datetime(
        cleaned["date_generated"], errors="coerce"
    ).dt.date
    sport_summaries = {}
    resolved_frames = []
    included_starts = []
    for sport in ("CFB", "MLB", "NCAAB"):
        start, end = season_bounds(sport, on_date)
        window = cleaned[
            _sport_mask(cleaned, sport) & generated_dates.between(start, min(end, on_date))
        ].copy()
        if window.empty:
            continue
        # MLB Overall row must match the MLB Record tab tracked universe (not all season rows).
        resolved_results: tuple[str, ...] = ("W", "L", "P")
        sport_start = start
        if sport == "MLB":
            window = _filter_mlb_tracked_universe(window)
            resolved_results = ("W", "L")  # MLB tab headline excludes pushes
            sport_start = max(start, MLB_PIPELINE_START_DATE)
        if window.empty:
            continue
        resolved, total, roi = _fixed_pl(
            window, profit_for_result, results=resolved_results
        )
        resolved_frames.append(resolved)
        included_starts.append(sport_start)
        sport_summaries[sport] = {
            "frame": resolved,
            "plays": len(window),
            "wins": int((resolved["result_clean"] == "W").sum()) if not resolved.empty else 0,
            "losses": int((resolved["result_clean"] == "L").sum()) if not resolved.empty else 0,
            "pushes": int((resolved["result_clean"] == "P").sum()) if not resolved.empty else 0,
            "pl": total,
            "roi": roi,
            "start_date": sport_start,
            "end_date": min(end, on_date),
        }

    resolved = (
        pd.concat(resolved_frames, ignore_index=True)
        if resolved_frames
        else pd.DataFrame()
    )
    total = float(resolved["pnl"].sum()) if not resolved.empty else 0.0
    wagered = float(len(resolved) * 10)
    return {
        "frame": resolved,
        "wins": int((resolved["result_clean"] == "W").sum()) if not resolved.empty else 0,
        "losses": int((resolved["result_clean"] == "L").sum()) if not resolved.empty else 0,
        "pushes": int((resolved["result_clean"] == "P").sum()) if not resolved.empty else 0,
        "pl": total,
        "roi": total / wagered * 100 if wagered else 0.0,
        "sports": sport_summaries,
        "start_date": min(included_starts) if included_starts else None,
        "end_date": on_date,
    }


def render_overall_record(
    history: pd.DataFrame,
    *,
    profit_for_result: Callable[[Any, str, float], float],
) -> None:
    st.subheader("Overall Record")
    summary = calculate_record_summary(
        history, profit_for_result=profit_for_result
    )
    resolved = summary["frame"]
    columns = st.columns(4)
    columns[0].metric(
        "Record",
        f"{summary['wins']}-{summary['losses']}-{summary['pushes']}",
    )
    columns[1].metric("P/L", f"${summary['pl']:+.2f}")
    columns[2].metric("ROI", f"{summary['roi']:+.1f}%")
    columns[3].metric("Resolved plays", len(resolved))
    if resolved.empty:
        st.caption("No resolved plays yet.")
        return
    by_sport = []
    for sport, sport_summary in summary["sports"].items():
        by_sport.append(
            {
                "Sport": sport,
                "W-L": f"{sport_summary['wins']}-{sport_summary['losses']}",
                "P/L": f"${sport_summary['pl']:+.2f}",
                "ROI": f"{sport_summary['roi']:+.1f}%",
            }
        )
    st.dataframe(pd.DataFrame(by_sport), use_container_width=True, hide_index=True)


def render_cfb_record(
    history: pd.DataFrame,
    *,
    profit_for_result: Callable[[Any, str, float], float],
) -> None:
    st.subheader("CFB Record")
    summary = calculate_record_summary(
        history, profit_for_result=profit_for_result
    )
    cfb = summary["sports"].get("CFB")
    if not cfb:
        st.caption("No CFB record history yet.")
        return
    resolved = cfb["frame"]
    start, end = cfb["start_date"], cfb["end_date"]
    st.caption(f"Season window: {start:%b %d, %Y} – {end:%b %d, %Y}")
    columns = st.columns(3)
    columns[0].metric("Record", f"{cfb['wins']}-{cfb['losses']}-{cfb['pushes']}")
    columns[1].metric("P/L", f"${cfb['pl']:+.2f}")
    columns[2].metric("ROI", f"{cfb['roi']:+.1f}%")
    if resolved.empty:
        st.caption("No resolved CFB plays yet.")
        return
    display_columns = [
        column
        for column in (
            "date_generated",
            "away_team",
            "home_team",
            "recommended_side",
            "spread_or_total",
            "my_edge_pct",
            "result_clean",
        )
        if column in resolved
    ]
    st.dataframe(
        resolved.sort_values("date_generated", ascending=False)[display_columns],
        use_container_width=True,
        hide_index=True,
    )
