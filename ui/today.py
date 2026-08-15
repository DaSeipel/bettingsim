"""TODAY dashboard: status, freshness, unified cards, and compact performance."""

from __future__ import annotations

import html
from datetime import date, datetime, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.cfb import cfb_pick
from ui.config import CFB_SHADOW_MODE, active_sports, pipeline_status
from ui.record import calculate_record_summary


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if number == number else None
    except (TypeError, ValueError):
        return None


def _signed_currency(value: float) -> str:
    sign = "+" if value > 0 else ("-" if value < 0 else "")
    return f"{sign}${abs(value):,.2f}"


def _timestamp(raw: Any) -> datetime | None:
    if raw is None or not str(raw).strip():
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).strip().replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
    except (TypeError, ValueError):
        return None


def _timestamp_label(raw: Any, now: datetime) -> tuple[str, float | None]:
    parsed = _timestamp(raw)
    if parsed is None:
        return "Unavailable", None
    age_minutes = max(0.0, (now - parsed.astimezone(timezone.utc)).total_seconds() / 60)
    local = parsed.astimezone(ZoneInfo("America/New_York"))
    return f"{local.strftime('%b %d, %I:%M %p ET')} · {age_minutes:.0f} min ago", age_minutes


def _today_mlb(plays: list[dict]) -> list[dict]:
    today = date.today().isoformat()
    output = []
    for play in plays:
        if not isinstance(play, dict):
            continue
        card_date = str(play.get("card_date") or today)
        market = str(play.get("market") or "").lower()
        if card_date == today and market in {"moneyline", "h2h", ""}:
            output.append(play)
    return output


def _today_ncaab(frame: pd.DataFrame) -> list[dict]:
    if frame.empty or "League" not in frame:
        return []
    rows = frame[frame["League"].astype(str).str.upper().eq("NCAAB")].copy()
    if rows.empty:
        return []
    if "Value (%)" in rows:
        rows = rows.sort_values("Value (%)", ascending=False)
    if "Event" in rows:
        rows = rows.groupby("Event").head(1)
    return rows.to_dict("records")


def _normalize_cards(
    *,
    mlb_plays: list[dict],
    ncaab_df: pd.DataFrame,
    cfb_plays: list[dict],
    active: dict[str, bool],
) -> tuple[list[dict], int]:
    cards: list[dict] = []
    if active["MLB"]:
        for play in _today_mlb(mlb_plays):
            edge = _number(play.get("edge"))
            if edge is not None and abs(edge) <= 1:
                edge *= 100
            cards.append(
                {
                    "sport": "MLB",
                    "matchup": f"{play.get('away_team', 'Away')} @ {play.get('home_team', 'Home')}",
                    "pick": str(play.get("selection") or play.get("pick_team") or "—"),
                    "edge": edge or 0.0,
                    "edge_label": f"{edge:.1f}% edge" if edge is not None else "Edge unavailable",
                    "probability": _number(play.get("model_prob")),
                    "time": play.get("commence_time"),
                    "odds": play.get("odds_american"),
                    "qualifies_at_fresh": play.get("qualifies_at_fresh"),
                }
            )
    if active["NCAAB"]:
        for play in _today_ncaab(ncaab_df):
            edge = _number(play.get("Value (%)"))
            point = _number(play.get("point"))
            selection = str(play.get("Selection") or "—")
            pick = f"{selection} {point:+.1f}" if point is not None else selection
            cards.append(
                {
                    "sport": "NCAAB",
                    "matchup": str(play.get("Event") or "Matchup unavailable"),
                    "pick": pick,
                    "edge": edge or 0.0,
                    "edge_label": f"{edge:.1f}% edge" if edge is not None else "Edge unavailable",
                    "probability": _number(play.get("model_prob")),
                    "time": play.get("commence_time") or play.get("Start Time"),
                    "odds": play.get("Odds"),
                    "qualifies_at_fresh": play.get("qualifies_at_fresh"),
                }
            )
    shadow_count = len(cfb_plays) if active["CFB"] and CFB_SHADOW_MODE else 0
    if active["CFB"] and not CFB_SHADOW_MODE:
        for play in cfb_plays:
            team, line = cfb_pick(play)
            edge = _number(play.get("edge_points"))
            cards.append(
                {
                    "sport": "CFB",
                    "matchup": str(play.get("matchup") or "Matchup unavailable"),
                    "pick": f"{team} {line:+.1f}" if line is not None else team,
                    "edge": edge or 0.0,
                    "edge_label": f"{edge:.1f} pts edge" if edge is not None else "Edge unavailable",
                    "probability": _number(play.get("cover_probability")),
                    "time": play.get("kickoff_utc"),
                    "odds": None,
                    "qualifies_at_fresh": play.get("qualifies_at_fresh"),
                }
            )
    return sorted(cards, key=lambda card: card["edge"], reverse=True), shadow_count


def _local_game_time(raw: Any) -> str:
    parsed = _timestamp(raw)
    if parsed is None:
        return "Time TBD"
    return parsed.astimezone(ZoneInfo("America/New_York")).strftime("%b %d · %I:%M %p ET")


def _card_html(card: dict, format_american: Callable) -> str:
    probability = card.get("probability")
    if probability is not None and probability <= 1:
        probability *= 100
    probability_label = "—" if probability is None else f"{probability:.1f}%"
    odds = card.get("odds")
    odds_label = format_american(odds) if odds is not None else "—"
    sport = html.escape(card["sport"])
    return (
        f'<div class="sport-card sport-card--{sport}">'
        f'<div class="sport-card__eyebrow">{sport} · {_local_game_time(card.get("time"))}</div>'
        f'<div class="sport-card__matchup">{html.escape(card["matchup"])}</div>'
        f'<div class="sport-card__pick">{html.escape(card["pick"])}</div>'
        f'<div class="sport-card__edge">{html.escape(card["edge_label"])}</div>'
        f'<div class="sport-card__meta">Model probability {probability_label} · Odds {odds_label}</div>'
        f'</div>'
    )


def _pipeline_notice(sport: str) -> str:
    return (
        f'<div class="pipeline-note-muted">No {html.escape(sport)} results for today '
        "— see freshness above.</div>"
    )


def render_today(
    *,
    value_plays_df: pd.DataFrame,
    mlb_plays: list[dict],
    mlb_meta: dict,
    ncaab_meta: dict,
    cfb_plays: list[dict],
    cfb_meta: dict,
    history: pd.DataFrame,
    format_american: Callable,
    profit_for_result: Callable[[Any, str, float], float],
    unresolved_stale: int = 0,
    pipeline_error: Any = None,
) -> None:
    now = datetime.now(timezone.utc)
    active = active_sports()
    caches = {"CFB": cfb_meta, "MLB": mlb_meta, "NCAAB": ncaab_meta}
    statuses = {
        sport: pipeline_status(sport, cache=caches[sport], now=now)
        for sport in active
    }
    renderable = {
        sport: is_active and statuses[sport] == "fresh"
        for sport, is_active in active.items()
    }
    cards, shadow_count = _normalize_cards(
        mlb_plays=mlb_plays,
        ncaab_df=value_plays_df,
        cfb_plays=cfb_plays,
        active=renderable,
    )
    performance = calculate_record_summary(
        history,
        profit_for_result=profit_for_result,
        on_date=now.astimezone(ZoneInfo("America/New_York")).date(),
    )

    st.title("Bobby Bottle's Betting Model")
    st.markdown(
        f'<div class="today-date">{date.today().strftime("%A, %B %d, %Y")}</div>',
        unsafe_allow_html=True,
    )

    sport_counts = {
        sport: sum(card["sport"] == sport for card in cards)
        for sport, is_fresh in renderable.items()
        if is_fresh
    }
    count_caption = " · ".join(f"{sport} {count}" for sport, count in sport_counts.items())
    any_fresh = any(renderable.values())
    best = cards[0] if cards else None
    metrics = st.columns(4)
    metrics[0].metric("Plays today", len(cards) if any_fresh else "—")
    with metrics[0]:
        st.caption(count_caption or "No fresh in-season pipeline")
    metrics[1].metric(
        "Best edge today",
        best["edge_label"] if best else "—",
    )
    with metrics[1]:
        st.caption(best["sport"] if best else "—")
    metrics[2].metric("Record", f"{performance['wins']}-{performance['losses']}")
    start_date = performance.get("start_date")
    with metrics[2]:
        if start_date:
            st.caption(
                f"All sports · {start_date:%b %-d, %Y} to date "
                f"({performance['end_date']:%b %-d, %Y})"
            )
        else:
            st.caption("All sports · no current-window plays")
    metrics[3].metric(
        "P/L",
        _signed_currency(performance["pl"]),
        f"{performance['roi']:+.1f}% ROI",
        delta_color="normal",
    )

    breakdown = " · ".join(
        f"{sport} {summary['wins']}-{summary['losses']} "
        f"({_signed_currency(summary['pl'])})"
        for sport, summary in performance["sports"].items()
    )
    if breakdown:
        st.markdown(
            f'<div class="record-breakdown">{html.escape(breakdown)}</div>',
            unsafe_allow_html=True,
        )

    odds_candidates = [
        mlb_meta.get("odds_captured_at"),
        ncaab_meta.get("odds_captured_at"),
        cfb_meta.get("odds_captured_at"),
    ]
    parsed_candidates = [(raw, _timestamp(raw)) for raw in odds_candidates]
    parsed_candidates = [(raw, parsed) for raw, parsed in parsed_candidates if parsed is not None]
    latest_odds = max(parsed_candidates, key=lambda pair: pair[1])[0] if parsed_candidates else None
    odds_label, odds_age = _timestamp_label(latest_odds, now)
    pitcher_label, pitcher_age = _timestamp_label(mlb_meta.get("pitchers_captured_at"), now)
    mismatches = sum(card.get("qualifies_at_fresh") is False for card in cards)
    active_statuses = [
        statuses[sport] for sport, is_active in active.items() if is_active
    ]
    stale = (
        odds_age is None
        or odds_age > 360
        or (active["MLB"] and (pitcher_age is None or pitcher_age > 360))
        or mismatches > 0
        or any(status != "fresh" for status in active_statuses)
    )
    if "missing" in active_statuses:
        freshness_class = "freshness-strip freshness-strip--critical"
    elif stale:
        freshness_class = "freshness-strip freshness-strip--warning"
    else:
        freshness_class = "freshness-strip"
    pitcher_text = pitcher_label if active["MLB"] else "MLB out of season"
    st.markdown(
        f'<div class="{freshness_class}">'
        f'<div><strong>Last odds fetch</strong><span>{html.escape(odds_label)}</span></div>'
        f'<div><strong>Last pitcher fetch</strong><span>{html.escape(pitcher_text)}</span></div>'
        f'<div><strong>Fresh-line mismatches</strong><span>{mismatches}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if unresolved_stale:
        st.warning(f"{unresolved_stale} play(s) still need results marked.")
    if pipeline_error:
        st.error(f"Value plays pipeline failed: {pipeline_error[0] if isinstance(pipeline_error, tuple) else pipeline_error}")

    st.markdown('<div class="section-heading">Today’s plays</div>', unsafe_allow_html=True)
    for sport, is_active in active.items():
        if is_active and statuses[sport] != "fresh":
            st.markdown(
                _pipeline_notice(sport),
                unsafe_allow_html=True,
            )
    if shadow_count:
        st.markdown(
            f'<div class="shadow-count">CFB: {shadow_count} shadow plays logged (not bet)</div>',
            unsafe_allow_html=True,
        )
    if not cards and any_fresh:
        st.markdown('<div class="empty-line">No qualifying plays today.</div>', unsafe_allow_html=True)
    elif cards:
        for start in range(0, len(cards), 3):
            columns = st.columns(3)
            for index, card in enumerate(cards[start : start + 3]):
                with columns[index]:
                    st.markdown(_card_html(card, format_american), unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Performance snapshot</div>', unsafe_allow_html=True)
    resolved = performance["frame"].copy()
    last_ten = resolved.sort_values("date_generated").tail(10) if not resolved.empty else resolved
    last_wins = int((last_ten["result_clean"] == "W").sum()) if not last_ten.empty else 0
    last_losses = int((last_ten["result_clean"] == "L").sum()) if not last_ten.empty else 0
    st.caption(f"Last 10: {last_wins}-{last_losses}")
    if resolved.empty:
        st.caption("Performance will appear after results are recorded.")
        return
    resolved["date"] = pd.to_datetime(resolved["date_generated"], errors="coerce")
    resolved = resolved[resolved["date"].notna()].sort_values("date")
    resolved["cumulative"] = resolved["pnl"].cumsum()
    figure = go.Figure(
        go.Scatter(
            x=resolved["date"],
            y=resolved["cumulative"],
            mode="lines",
            line={"color": "#66bb6a", "width": 2},
        )
    )
    figure.add_hline(
        y=0,
        line_width=1,
        line_dash="dash",
        line_color="rgba(255,255,255,.42)",
        annotation_text="$0",
        annotation_position="bottom right",
        annotation_font_color="rgba(255,255,255,.7)",
    )
    peak_index = resolved["cumulative"].idxmax()
    peak = resolved.loc[peak_index]
    figure.add_annotation(
        x=peak["date"],
        y=peak["cumulative"],
        text=f"Peak {_signed_currency(float(peak['cumulative']))}",
        showarrow=True,
        arrowhead=2,
        ax=45,
        ay=35,
        arrowcolor="#81c784",
        font={"color": "#a5d6a7", "size": 10},
        bgcolor="rgba(20,30,24,.82)",
        bordercolor="rgba(129,199,132,.45)",
    )
    figure.update_layout(
        height=220,
        margin={"t": 8, "b": 25, "l": 30, "r": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.03)",
        showlegend=False,
        font={"color": "rgba(255,255,255,.75)", "size": 11},
        xaxis={"gridcolor": "rgba(255,255,255,.06)", "title": None},
        yaxis={"gridcolor": "rgba(255,255,255,.06)", "title": "P/L ($)"},
    )
    st.plotly_chart(figure, use_container_width=True)
