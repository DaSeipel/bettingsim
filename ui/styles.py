"""Shared dashboard spacing, cards, freshness, and nested-navigation styles."""

import streamlit as st

from ui.config import SPORT_CONFIG


def inject_dashboard_styles() -> None:
    st.markdown(
        f"""
        <style>
        .block-container {{
          padding-top: 3.5rem;
        }}
        h1 {{ margin-bottom: .15rem !important; }}
        h2, h3, h4, h5 {{ font-weight: 750 !important; }}
        [data-testid="stVerticalBlock"] {{ gap: .7rem; }}
        .today-date {{
          color: rgba(255,255,255,.58); font-size: .9rem; margin: -.2rem 0 .85rem;
        }}
        .freshness-strip {{
          display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.65rem;
          padding:.72rem .85rem;border-radius:8px;margin:.4rem 0 1rem;
          background:rgba(120,144,156,.13);border:1px solid rgba(144,164,174,.28);
          font-size:.82rem;
        }}
        .freshness-strip--warning {{
          background:rgba(255,152,0,.085);border-color:#ffb300;
          box-shadow:0 0 0 1px rgba(255,179,0,.12);color:#ffd166;
        }}
        .freshness-strip.freshness-strip--warning strong,
        .freshness-strip.freshness-strip--warning span {{
          color:#ffd166;
        }}
        .freshness-strip--critical {{
          background:rgba(244,67,54,.075);border-color:#ff5252;
          box-shadow:0 0 0 1px rgba(255,82,82,.12);color:#ff8a80;
        }}
        .freshness-strip.freshness-strip--critical strong,
        .freshness-strip.freshness-strip--critical span {{
          color:#ff8a80;
        }}
        .freshness-strip strong {{ display:block;color:#fafafa;margin-bottom:.12rem; }}
        .freshness-strip span {{ color:rgba(255,255,255,.7); }}
        .section-heading {{
          font-size:1.12rem;font-weight:750;margin:1.05rem 0 .65rem;color:#fafafa;
        }}
        .empty-line {{ color:rgba(255,255,255,.52);text-align:center;padding:1.2rem 0; }}
        .pipeline-note-muted {{
          color:rgba(255,255,255,.52);font-size:.84rem;margin:.35rem 0 .65rem;
        }}
        .record-breakdown {{
          color:rgba(255,255,255,.68);font-size:.82rem;margin:-.15rem 0 .55rem;
        }}
        .shadow-count {{ color:#b39ddb;font-size:.84rem;margin:-.2rem 0 .65rem; }}
        .sport-card {{
          border-radius:10px;padding:1rem 1.05rem;margin-bottom:.8rem;
          border-left:5px solid;background:rgba(255,255,255,.055);
          min-height:15rem;box-sizing:border-box;
        }}
        .sport-card--MLB {{ border-left-color:{SPORT_CONFIG["MLB"]["accent"]}; }}
        .sport-card--NCAAB {{ border-left-color:{SPORT_CONFIG["NCAAB"]["accent"]}; }}
        .sport-card__eyebrow {{
          font-size:.72rem;text-transform:uppercase;letter-spacing:.055em;
          color:rgba(255,255,255,.55);margin-bottom:.45rem;
        }}
        .sport-card__matchup {{ font-size:1rem;font-weight:700;line-height:1.28; }}
        .sport-card__pick {{ font-size:1.18rem;font-weight:800;margin:.72rem 0 .35rem; }}
        .sport-card__edge {{ color:#66bb6a;font-size:1.55rem;font-weight:850;margin:.55rem 0; }}
        .sport-card__meta {{ color:rgba(255,255,255,.64);font-size:.78rem;line-height:1.45; }}
        @media (max-width: 700px) {{
          .freshness-strip {{ grid-template-columns:1fr; }}
          .sport-card {{ min-height:0; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
