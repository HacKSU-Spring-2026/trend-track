import sys
import os
from datetime import date, datetime

import pandas as pd
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Make sure trenddeath submodules are importable when running from repo root
sys.path.insert(0, os.path.dirname(__file__))

# ─── Gemini setup ───────────────────────────────────────────────────────────
_groq_key = os.getenv("GROQ_API_KEY")
_groq_client = Groq(api_key=_groq_key) if _groq_key else None


def generate_trend_report(
    keyword: str,
    phase: str,
    current: int,
    peak: int,
    peak_date: str,
    death_display: str,
    days_left,
    momentum_pct: float,
    avg_score: float,
    volatility: float,
    weeks_above_50: int,
) -> str:
    if _groq_client is None:
        return "⚠️ GROQ_API_KEY not set — AI report unavailable."

    prompt = f"""You are a trend analyst. Write a concise, insightful report (2–3 short paragraphs) about the trend lifecycle of "{keyword}" based on the data below. Be direct and analytical — explain what is happening, why it might be happening, and what to expect. Do not use bullet points or headers. Write in plain prose.

Data:
- Current interest score: {current}/100
- All-time peak: {peak}/100 on {peak_date}
- Current phase: {phase}
- Predicted death date: {death_display}
- Days until death: {days_left if days_left is not None else "beyond forecast window"}
- 3-month momentum: {momentum_pct:+.1f}%
- 5-year average score: {avg_score:.1f}
- Volatility (std dev): {volatility:.1f}
- Weeks with score ≥ 50: {weeks_above_50}

Write the report now:"""

    response = _groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content

from data.fetch import fetch_trending_now
from data.mongo import get_recent_searches
from model.trend_phase import TrendPhase
from charts.lifecycle_chart import build_lifecycle_chart
from charts.velocity_chart import build_velocity_chart
from utils.cache import get_or_fetch

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrendDeath",
    # page_icon="💀",
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
            TrendDeath
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

# Populate the text input from the dropdown selection via session state
if "trending_select" in st.session_state:
    chosen = st.session_state["trending_select"]
    if chosen and chosen != "— select —":
        st.session_state["search_input"] = chosen

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
    st.markdown('<p class="section-label">Or pick a suggested topic</p>', unsafe_allow_html=True)
    trending_topics = fetch_trending_now()
    if trending_topics:
        st.selectbox(
            label="trending",
            options=["— select —"] + trending_topics,
            label_visibility="collapsed",
            key="trending_select",
        )
    else:
        st.caption("Could not load trending topics")

# Resolve the active keyword
keyword = ""
if analyze_clicked and search_query.strip():
    keyword = search_query.strip()

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
                forecast_df = fit_and_forecast(raw_df, periods=365)

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
                st.plotly_chart(fig_life, width="stretch", config={"displayModeBar": False})

            with vel_col:
                fig_vel = build_velocity_chart(forecast_df, keyword)
                st.plotly_chart(fig_vel, width="stretch", config={"displayModeBar": False})
        else:
            st.warning("Forecast chart unavailable — no model output found.")

        # ── Extended metrics + data tables ──────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">Extended metrics</p>', unsafe_allow_html=True)

        raw_data = result.get("raw_data", [])
        if raw_data:
            hist_df = pd.DataFrame(raw_data)
            hist_df["date"] = pd.to_datetime(hist_df["date"])
            hist_df = hist_df.sort_values("date")

            scores = hist_df["interest"]
            avg_score      = scores.mean()
            volatility     = scores.std()
            median_score   = scores.median()
            score_3m_ago   = scores.iloc[-13] if len(scores) >= 13 else scores.iloc[0]
            momentum_pct   = ((current - float(score_3m_ago)) / float(score_3m_ago) * 100) if score_3m_ago else 0.0
            pct_of_peak    = (current / peak * 100) if peak else 0.0
            weeks_above_50 = int((scores >= 50).sum())

            m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
            m1.metric("Avg Score (5y)",   f"{avg_score:.1f}")
            m2.metric("Median Score",     f"{median_score:.1f}")
            m3.metric("Volatility (σ)",   f"{volatility:.1f}")
            m4.metric("3-month Momentum", f"{momentum_pct:+.1f}%")
            m5.metric("% of Peak",        f"{pct_of_peak:.1f}%")
            m6.metric("Weeks ≥ 50",       str(weeks_above_50))
            m7.metric("Data Points",      str(len(scores)))

        st.markdown("<hr>", unsafe_allow_html=True)

        with st.expander("📊 Raw data & forecast tables", expanded=False):
            tab_hist, tab_forecast = st.tabs(["Historical interest", "Forecast"])

            with tab_hist:
                if raw_data:
                    display_hist = hist_df.copy()
                    display_hist["date"] = display_hist["date"].dt.strftime("%Y-%m-%d")
                    display_hist.columns = ["Date", "Interest Score"]
                    display_hist = display_hist.sort_values("Date", ascending=False).reset_index(drop=True)

                    st.dataframe(
                        display_hist,
                        use_container_width=True,
                        height=320,
                        hide_index=True,
                    )
                    csv_hist = display_hist.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇ Download historical CSV",
                        data=csv_hist,
                        file_name=f"{keyword}_historical.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No historical data available.")

            with tab_forecast:
                if forecast_df is not None:
                    today_ts = pd.Timestamp(date.today())
                    fcast_display = forecast_df[forecast_df["ds"] >= today_ts][
                        ["ds", "yhat", "yhat_lower", "yhat_upper"]
                    ].copy()
                    fcast_display["ds"] = fcast_display["ds"].dt.strftime("%Y-%m-%d")
                    fcast_display = fcast_display.rename(columns={
                        "ds": "Date",
                        "yhat": "Forecast",
                        "yhat_lower": "Lower (80%)",
                        "yhat_upper": "Upper (80%)",
                    })
                    fcast_display[["Forecast", "Lower (80%)", "Upper (80%)"]] = (
                        fcast_display[["Forecast", "Lower (80%)", "Upper (80%)"]].round(1)
                    )
                    fcast_display = fcast_display.reset_index(drop=True)

                    st.dataframe(
                        fcast_display,
                        use_container_width=True,
                        height=320,
                        hide_index=True,
                    )
                    csv_fcast = fcast_display.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇ Download forecast CSV",
                        data=csv_fcast,
                        file_name=f"{keyword}_forecast.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No forecast data available.")

        # ── AI Report ───────────────────────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">AI analyst report</p>', unsafe_allow_html=True)

        _avg_score     = avg_score     if raw_data else 0.0
        _volatility    = volatility    if raw_data else 0.0
        _momentum_pct  = momentum_pct  if raw_data else 0.0
        _weeks_above50 = weeks_above_50 if raw_data else 0

        report_key = f"report_{keyword}"

        if report_key not in st.session_state:
            if st.button("Generate AI report", key="gen_report_btn"):
                with st.spinner("Generating report…"):
                    st.session_state[report_key] = generate_trend_report(
                        keyword        = keyword,
                        phase          = phase_str,
                        current        = current,
                        peak           = peak,
                        peak_date      = peak_date,
                        death_display  = death_display,
                        days_left      = days_left,
                        momentum_pct   = _momentum_pct,
                        avg_score      = _avg_score,
                        volatility     = _volatility,
                        weeks_above_50 = _weeks_above50,
                    )
                st.rerun()
        else:
            report_text = st.session_state[report_key]
            st.markdown(
                f"""
                <div style="background:#161b22; border:1px solid #30363d; border-radius:10px;
                            padding:20px 24px; line-height:1.7; color:#e6edf3; font-size:0.95rem;">
                    {report_text.replace(chr(10), '<br>')}
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Regenerate", key="regen_report_btn"):
                del st.session_state[report_key]
                st.rerun()

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
