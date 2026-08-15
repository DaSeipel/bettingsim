"""Live plays tab bodies for existing MLB and NCAAB functionality."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st

from ui.config import season_note, sport_is_in_season


def render_ncaab_plays_tab(
    *,
    value_plays_df: pd.DataFrame,
    already_started: int,
    odds_api_key: str,
    march_madness_mode: bool,
    format_american: Callable,
    market_labels: dict,
) -> None:
    st.subheader("NCAAB")
    if not sport_is_in_season("NCAAB"):
        st.caption(season_note("NCAAB"))
        return
    st.caption("Today's NCAAB value plays ranked by edge. Picks powered by XGBoost multi-feature model.")
    ncaab_value = (
        value_plays_df[value_plays_df["League"] == "NCAAB"]
        if not value_plays_df.empty and "League" in value_plays_df.columns
        else pd.DataFrame()
    )
    if already_started:
        st.caption(f"**{already_started}** play(s) already started — showing **{len(ncaab_value)}** remaining.")
    if (odds_api_key or "").strip() and st.button(
        "Refresh",
        key="ncaab_refresh",
        help="Run the value-plays pipeline and reload cache",
    ):
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline_to_cache.py"
        env = os.environ.copy()
        env["ODDS_API_KEY"] = odds_api_key.strip()
        try:
            subprocess.run(
                [sys.executable, str(script)],
                cwd=str(script.parent.parent),
                env=env,
                timeout=300,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            st.error("Pipeline timed out after 5 minutes.")
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
        st.rerun()

    if ncaab_value.empty:
        st.caption("No qualifying NCAAB plays today.")
        return
    best = (
        ncaab_value.sort_values("Value (%)", ascending=False)
        .groupby("Event")
        .head(1)
        .reset_index(drop=True)
    )
    if march_madness_mode:
        best = best[best["Value (%)"] >= 5.0]
    display = best.copy()
    if "Odds" in display:
        display["Odds"] = display["Odds"].apply(format_american)
    if "Market" in display:
        display["Market"] = display["Market"].map(
            lambda value: market_labels.get(value, value) if pd.notna(value) else value
        )
    if "Recommended Stake" in display:
        display = display.drop(columns=["Recommended Stake"])
    drop = [
        "League",
        "Injury Alert",
        "home_team",
        "away_team",
        "March context",
        "Tournament Context",
    ]
    display = display.drop(columns=[column for column in drop if column in display])
    st.dataframe(
        display.rename(columns={"Event": "Game", "confidence_tier": "Confidence"}),
        use_container_width=True,
        hide_index=True,
    )


def render_mlb_plays_tab(
    *,
    mlb_df: pd.DataFrame,
    mlb_cache_name: str,
    format_american: Callable,
    kelly_fraction: Callable,
    kelly_frac: float,
    bankroll: float,
    park_lookup: dict,
    park_home_display: Callable,
    confidence_from_edge: Callable,
    edge_row_style: Callable,
    render_top_card: Callable,
    dataframe_for_history: Callable,
    archive_value_plays: Callable,
    clear_history_cache: Callable,
) -> None:
    st.subheader("MLB")
    if not sport_is_in_season("MLB"):
        st.caption(season_note("MLB"))
        return
    st.caption(
        f"Today's moneyline value plays from `{mlb_cache_name}`. "
        "Edge = (model probability × decimal odds) − 1."
    )
    if mlb_df.empty:
        st.caption("No qualifying MLB plays today.")
        return

    display = mlb_df.copy()
    if "market" in display:
        markets = display["market"].astype(str).str.strip().str.lower()
        display = display[markets.isin(("moneyline", "h2h"))].copy()
    if display.empty:
        st.caption("No qualifying MLB moneyline plays today.")
        return
    display["Odds"] = display.get("odds_american", pd.Series(index=display.index)).apply(format_american)
    if "edge" in display:
        display["Edge %"] = pd.to_numeric(display["edge"], errors="coerce") * 100
    elif "edge_pct" in display:
        raw = pd.to_numeric(display["edge_pct"], errors="coerce")
        display["Edge %"] = raw.where(raw.abs() >= 1, raw * 100)
    else:
        display["Edge %"] = None
    display["Model prob"] = pd.to_numeric(
        display["model_prob"]
        if "model_prob" in display
        else pd.Series(None, index=display.index),
        errors="coerce",
    )

    stakes = []
    for _, row in display.iterrows():
        try:
            fraction = kelly_fraction(
                float(row.get("odds_american", 0)),
                float(row.get("model_prob", 0)),
                fraction=kelly_frac,
            )
            stakes.append(round(bankroll * fraction, 2))
        except (TypeError, ValueError):
            stakes.append(None)
    display["Rec. stake ($)"] = stakes
    display["Confidence"] = display["Edge %"].apply(confidence_from_edge)
    away_values = (
        display["away_team"]
        if "away_team" in display
        else pd.Series("", index=display.index, dtype=object)
    )
    home_values = (
        display["home_team"]
        if "home_team" in display
        else pd.Series("", index=display.index, dtype=object)
    )
    display["Away"] = away_values.astype(str).str.strip()
    display["Home"] = home_values.apply(
        lambda team: park_home_display(str(team).strip(), park_lookup)
    )
    display["SP (away)"] = (
        display["away_pitcher"] if "away_pitcher" in display else "TBD"
    )
    display["SP (home)"] = (
        display["home_pitcher"] if "home_pitcher" in display else "TBD"
    )
    display["Pick"] = display["selection"] if "selection" in display else "—"

    top_row = None
    edge_values = pd.to_numeric(
        display["edge"] if "edge" in display else pd.Series(None, index=display.index),
        errors="coerce",
    )
    if edge_values.notna().any():
        top_row = display.loc[edge_values.idxmax()]
    st.markdown(render_top_card(top_row, park_lookup), unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Moneyline plays</div>', unsafe_allow_html=True)

    columns = [
        name
        for name in (
            "Away",
            "Home",
            "SP (away)",
            "SP (home)",
            "Pick",
            "Odds",
            "Model prob",
            "Edge %",
            "Confidence",
            "Rec. stake ($)",
        )
        if name in display
    ]
    table = display[columns].copy()
    styled = (
        table.style.apply(edge_row_style, axis=1)
        .format(
            {
                "Model prob": "{:.3f}",
                "Edge %": "{:.2f}",
                "Rec. stake ($)": "${:.2f}",
            },
            na_rep="—",
        )
        .hide(axis="index")
    )
    st.dataframe(styled, use_container_width=True)

    if st.button("Save Picks", key="mlb_save_play_history"):
        try:
            archive_df = dataframe_for_history(mlb_df)
            if archive_df.empty:
                st.warning("Nothing to save — no valid rows on today's card.")
            else:
                count = archive_value_plays(archive_df, as_of_date=date.today())
                clear_history_cache()
                st.success(f"Saved **{count}** MLB pick(s) to play history.")
        except Exception as exc:
            st.error(f"Could not save picks — {exc}")
