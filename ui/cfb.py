"""College-football cache reader and spread-card presentation."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import streamlit as st

from ui.config import CFB_SHADOW_MODE, SPORT_CONFIG, season_note, sport_is_in_season


CFB_CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "cache" / "cfb_value_plays.json"
CFB_EXPECTED_FIELDS = (
    "matchup",
    "kickoff_utc",
    "venue",
    "neutral_site",
    "market_spread",
    "market_book",
    "projected_margin",
    "edge_points",
    "cover_probability",
    "odds_captured_at",
    "model_version",
)


def load_cfb_value_plays(path: Path = CFB_CACHE_PATH) -> tuple[list[dict], dict]:
    """Read the future CFB cache defensively; missing/malformed files are empty."""
    diagnostics: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "missing_fields": list(CFB_EXPECTED_FIELDS),
        "error": None,
    }
    if not path.exists():
        return [], diagnostics
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        diagnostics["error"] = str(exc)
        return [], diagnostics

    if isinstance(raw, list):
        candidates = raw
    elif isinstance(raw, dict):
        candidates = raw.get("plays") or raw.get("value_plays") or []
    else:
        candidates = []
    plays = [dict(play) for play in candidates if isinstance(play, dict)]
    present = {key for play in plays for key in play}
    diagnostics["missing_fields"] = [key for key in CFB_EXPECTED_FIELDS if key not in present]
    return plays, diagnostics


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if number == number else None
    except (TypeError, ValueError):
        return None


def _format_local_timestamp(raw: Any) -> str:
    if raw is None or not str(raw).strip():
        return "Kickoff TBD"
    try:
        parsed = datetime.fromisoformat(str(raw).strip().replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        local = parsed.astimezone(ZoneInfo("America/New_York"))
        return local.strftime("%a, %b %d · %I:%M %p ET")
    except (TypeError, ValueError):
        return "Kickoff TBD"


def _matchup_teams(matchup: Any) -> tuple[str, str]:
    text = str(matchup or "").strip()
    for token in (" @ ", " at ", " vs. ", " vs "):
        if token in text:
            away, home = text.split(token, 1)
            return away.strip(), home.strip()
    return text or "Away", "Home"


def cfb_pick(play: dict) -> tuple[str, float | None]:
    """Resolve a display side from explicit cache fields, then model-vs-market."""
    away, home = _matchup_teams(play.get("matchup"))
    explicit = str(
        play.get("pick_side")
        or play.get("selection")
        or play.get("recommended_pick")
        or ""
    ).strip()
    market = _number(play.get("market_spread"))
    projection = _number(play.get("projected_margin"))

    if explicit:
        if explicit.lower() in {"home", "h"}:
            return home, market
        if explicit.lower() in {"away", "a"}:
            return away, -market if market is not None else None
        is_home = explicit.lower() == home.lower()
        return explicit, market if is_home else (-market if market is not None else None)

    home_cover_edge = None
    if market is not None and projection is not None:
        home_cover_edge = projection + market
    if home_cover_edge is None or home_cover_edge >= 0:
        return home, market
    return away, -market if market is not None else None


def _signed(value: float | None) -> str:
    return "—" if value is None else f"{value:+.1f}"


def render_cfb_card(play: dict, *, muted: bool = CFB_SHADOW_MODE) -> str:
    matchup = html.escape(str(play.get("matchup") or "Matchup TBD"))
    pick_team, pick_line = cfb_pick(play)
    projected = _number(play.get("projected_margin"))
    market = _number(play.get("market_spread"))
    edge = _number(play.get("edge_points"))
    cover = _number(play.get("cover_probability"))
    if cover is not None and cover <= 1:
        cover *= 100
    cover_text = "—" if cover is None else f"{cover:.1f}%"
    neutral = '<span class="cfb-neutral">Neutral site</span>' if bool(play.get("neutral_site")) else ""
    venue = html.escape(str(play.get("venue") or "Venue TBD"))
    book = html.escape(str(play.get("market_book") or "Book unavailable"))
    model = html.escape(str(play.get("model_version") or "Model version unavailable"))
    muted_class = " cfb-card--shadow" if muted else ""
    return (
        f'<div class="sport-card cfb-card{muted_class}">'
        f'<div class="sport-card__eyebrow">CFB · {_format_local_timestamp(play.get("kickoff_utc"))}</div>'
        f'<div class="sport-card__matchup">{matchup}</div>'
        f'<div class="sport-card__pick">{html.escape(pick_team)} {_signed(pick_line)}</div>'
        f'<div class="cfb-comparison">'
        f'<span><small>Projected margin</small><strong>{_signed(projected)}</strong></span>'
        f'<span><small>Market spread</small><strong>{_signed(market)}</strong></span>'
        f'</div>'
        f'<div class="sport-card__edge">{("—" if edge is None else f"{edge:.1f}")} pts edge</div>'
        f'<div class="sport-card__meta">Cover probability <strong>{cover_text}</strong> · break-even 52.4%</div>'
        f'<div class="sport-card__meta">{venue} · {book} · {model} {neutral}</div>'
        f'</div>'
    )


def render_cfb_plays_tab(plays: list[dict]) -> None:
    st.subheader("College Football")
    if CFB_SHADOW_MODE:
        st.markdown(
            '<div class="cfb-shadow-banner"><strong>SHADOW MODE — Weeks 0-5.</strong> '
            "Logging only, no bets. Model has not cleared live calibration. "
            "Review scheduled after Week 5.</div>",
            unsafe_allow_html=True,
        )
    if not sport_is_in_season("CFB"):
        st.caption(season_note("CFB"))
        return
    if not plays:
        st.caption("No CFB shadow plays logged today." if CFB_SHADOW_MODE else "No CFB plays today.")
        return
    for row_start in range(0, len(plays), 3):
        columns = st.columns(3)
        for index, play in enumerate(plays[row_start : row_start + 3]):
            with columns[index]:
                st.markdown(render_cfb_card(play), unsafe_allow_html=True)


def inject_cfb_styles() -> None:
    accent = SPORT_CONFIG["CFB"]["accent"]
    st.markdown(
        f"""
        <style>
        .cfb-shadow-banner {{
          border: 1px solid rgba(142,124,195,.6); border-left: 5px solid {accent};
          background: rgba(142,124,195,.14); padding: .75rem 1rem; border-radius: 8px;
          margin: .25rem 0 1rem; color: rgba(255,255,255,.9);
        }}
        .cfb-card {{ border-left-color: {accent} !important; }}
        .cfb-card--shadow {{ filter: saturate(.45); opacity: .84; }}
        .cfb-comparison {{ display:grid;grid-template-columns:1fr 1fr;gap:.6rem;margin:.7rem 0; }}
        .cfb-comparison span {{ background:rgba(255,255,255,.045);padding:.5rem;border-radius:6px; }}
        .cfb-comparison small {{ display:block;color:rgba(255,255,255,.55); }}
        .cfb-comparison strong {{ display:block;margin-top:.15rem; }}
        .cfb-neutral {{ color:#d1c4e9;font-weight:700; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
