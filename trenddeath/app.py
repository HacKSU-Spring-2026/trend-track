import sys
import os
from datetime import date, datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Make sure trenddeath submodules are importable when running from repo root
sys.path.insert(0, os.path.dirname(__file__))

from data.fetch import fetch_trending_now
from data.mongo import get_recent_searches
from model.trend_phase import TrendPhase
from charts.lifecycle_chart import build_lifecycle_chart
from charts.velocity_chart import build_velocity_chart
from utils.cache import get_or_fetch

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrendDeath",
    page_icon="💀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Global styles ──────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Base */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0d1117;
        color: #e6edf3;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background-color: #161b22; }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px 20px;
    }
    [data-testid="metric-container"] label {
        color: #8b949e !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e6edf3 !important;
        font-size: 1.6rem !important;
        font-weight: 700;
    }

    /* Search input */
    [data-testid="stTextInput"] input {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        color: #e6edf3;
        font-size: 1rem;
        padding: 10px 14px;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: #58a6ff;
        box-shadow: 0 0 0 3px rgba(88,166,255,0.15);
    }

    /* Buttons */
    [data-testid="stButton"] button {
        background: #1f6feb;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: background 0.2s;
    }
    [data-testid="stButton"] button:hover {
        background: #388bfd;
    }

    /* Selectbox */
    [data-testid="stSelectbox"] > div > div {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        color: #e6edf3;
    }

    /* Divider */
    hr { border-color: #21262d; margin: 1.5rem 0; }

    /* Recent search rows */
    .recent-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 6px;
        font-size: 0.88rem;
    }
    .recent-topic { color: #58a6ff; font-weight: 600; }
    .recent-meta  { color: #8b949e; font-size: 0.8rem; }

    /* Phase badge */
    .phase-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }

    /* Section header */
    .section-label {
        color: #8b949e;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    /* Error / warning banners */
    [data-testid="stAlert"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="padding: 2rem 0 1rem 0;">
        <h1 style="font-size:2.2rem; font-weight:800; color:#e6edf3; margin:0; letter-spacing:-0.02em;">
            💀 TrendDeath
        </h1>
        <p style="color:#8b949e; font-size:1rem; margin:6px 0 0 0; letter-spacing:0.01em;">
            Every trend has an expiry date. Find out when yours dies.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<hr>", unsafe_allow_html=True)

# ─── Search area ────────────────────────────────────────────────────────────
col_input, col_btn, col_trending = st.columns([4, 1, 3])

with col_input:
    st.markdown('<p class="section-label">Search any topic</p>', unsafe_allow_html=True)
    search_query = st.text_input(
        label="search",
        placeholder='e.g. "ChatGPT", "Wordle", "NFT"',
        label_visibility="collapsed",
        key="search_input",
    )

with col_btn:
    st.markdown('<p class="section-label">&nbsp;</p>', unsafe_allow_html=True)
    analyze_clicked = st.button("Analyze", use_container_width=True)

with col_trending:
    st.markdown('<p class="section-label">Or pick a trending topic</p>', unsafe_allow_html=True)
    trending_topics = fetch_trending_now()
    if trending_topics:
        selected_trending = st.selectbox(
            label="trending",
            options=["— select —"] + trending_topics,
            label_visibility="collapsed",
            key="trending_select",
        )
    else:
        selected_trending = "— select —"
        st.caption("Could not load trending topics")

# Resolve the active keyword
keyword = ""
if analyze_clicked and search_query.strip():
    keyword = search_query.strip()
elif selected_trending and selected_trending != "— select —":
    keyword = selected_trending

# ─── Analysis ───────────────────────────────────────────────────────────────
if keyword:
    st.markdown("<hr>", unsafe_allow_html=True)

    with st.spinner(f"Fetching data and running forecast for **{keyword}**…"):
        try:
            result = get_or_fetch(keyword)
            error = None
        except ValueError as e:
            result = None
            error = str(e)
        except Exception as e:
            result = None
            error = f"Unexpected error: {e}"

    if error:
        st.error(f"**Could not analyze '{keyword}'** — {error}")
        cached = None
        from data.mongo import get_cached_result
        cached = get_cached_result(keyword)
        if cached:
            st.info("Showing last cached result instead.")
            result = cached

    if result:
        phase_str   = result.get("trend_phase", "Unknown")
        current     = result.get("current_score", 0)
        peak        = result.get("peak_score", 0)
        peak_date   = result.get("peak_date", "—")
        death_str   = result.get("predicted_death")
        days_left   = result.get("days_remaining")
        conf_upper  = result.get("confidence_upper", 0.0)
        conf_lower  = result.get("confidence_lower", 0.0)
        forecast_df = result.get("forecast_df")

        # Reconstruct forecast_df from raw_data if cache hit (no forecast_df stored)
        if forecast_df is None and result.get("raw_data"):
            raw_df = pd.DataFrame(result["raw_data"])
            raw_df["date"] = pd.to_datetime(raw_df["date"])
            raw_df = raw_df.set_index("date")
            raw_df = raw_df.rename(columns={"interest": "interest"})
            with st.spinner("Re-running forecast from cached data…"):
                from model.prophet_model import fit_and_forecast
                forecast_df = fit_and_forecast(raw_df, periods=90)

        # Phase object for colour/emoji
        try:
            phase = TrendPhase(phase_str)
        except ValueError:
            phase = TrendPhase.DECLINING

        death_display = (
            datetime.strptime(death_str, "%Y-%m-%d").strftime("%b %d, %Y")
            if death_str else "Beyond forecast"
        )
        days_display = (
            f"{days_left} days" if days_left is not None else "N/A"
        )
        if days_left == 0:
            days_display = "Already Dead"

        # ── Stat cards ──────────────────────────────────────────────────────
        st.markdown(
            f'<p class="section-label">Analysis — <span style="color:#58a6ff">{keyword}</span></p>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Current Score", f"{current} / 100")
        c2.metric("All-time Peak", f"{peak} / 100", delta=peak_date, delta_color="off")
        c3.metric(
            "Trend Phase",
            f"{phase.emoji()} {phase.value}",
        )
        c4.metric("Predicted Death", death_display)
        c5.metric("Days Until Death", days_display)

        # Confidence interval note under the cards
        if death_str:
            st.markdown(
                f'<p style="color:#8b949e; font-size:0.8rem; margin-top:4px;">'
                f'80% confidence interval at death: '
                f'<span style="color:#a78bfa">{conf_lower:.1f}</span> – '
                f'<span style="color:#a78bfa">{conf_upper:.1f}</span>'
                f'</p>',
                unsafe_allow_html=True,
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Charts ──────────────────────────────────────────────────────────
        if forecast_df is not None:
            death_date_obj = (
                datetime.strptime(death_str, "%Y-%m-%d").date() if death_str else None
            )

            chart_col, vel_col = st.columns([3, 2])

            with chart_col:
                fig_life = build_lifecycle_chart(forecast_df, keyword, phase, death_date_obj)
                st.plotly_chart(fig_life, use_container_width=True, config={"displayModeBar": False})

            with vel_col:
                fig_vel = build_velocity_chart(forecast_df, keyword)
                st.plotly_chart(fig_vel, use_container_width=True, config={"displayModeBar": False})
        else:
            st.warning("Forecast chart unavailable — no model output found.")

# ─── Recent searches ────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<p class="section-label">Recent searches</p>', unsafe_allow_html=True)

recent = get_recent_searches(limit=10)

if not recent:
    st.markdown(
        '<p style="color:#8b949e; font-size:0.88rem;">No searches yet — analyze a topic above to get started.</p>',
        unsafe_allow_html=True,
    )
else:
    cols = st.columns(2)
    for i, doc in enumerate(recent):
        topic       = doc.get("topic", "—")
        phase_val   = doc.get("trend_phase", "—")
        score       = doc.get("current_score", "—")
        death       = doc.get("predicted_death", None)
        searched_at = doc.get("searched_at", None)

        try:
            phase_obj = TrendPhase(phase_val)
            phase_color = phase_obj.color()
            phase_emoji = phase_obj.emoji()
        except Exception:
            phase_color = "#8b949e"
            phase_emoji = ""

        death_label = (
            datetime.strptime(death, "%Y-%m-%d").strftime("%b %d, %Y")
            if death else "Unknown"
        )
        date_label = (
            searched_at.strftime("%b %d, %H:%M") if searched_at else ""
        )

        with cols[i % 2]:
            st.markdown(
                f"""
                <div class="recent-row">
                    <div>
                        <span class="recent-topic">{topic}</span>
                        <span style="margin-left:10px; background:{phase_color}22;
                              color:{phase_color}; padding:2px 8px; border-radius:12px;
                              font-size:0.75rem; font-weight:600;">
                            {phase_emoji} {phase_val}
                        </span>
                    </div>
                    <div class="recent-meta">
                        Score: <b style="color:#e6edf3">{score}</b> &nbsp;·&nbsp;
                        Dies: <b style="color:#e6edf3">{death_label}</b> &nbsp;·&nbsp;
                        {date_label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
