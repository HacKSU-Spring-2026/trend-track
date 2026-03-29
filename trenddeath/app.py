import sys
import os
import concurrent.futures
from datetime import date, datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Make sure trenddeath submodules are importable when running from repo root
sys.path.insert(0, os.path.dirname(__file__))

# ─── Groq setup ─────────────────────────────────────────────────────────────
_groq_key = os.getenv("GROQ_API_KEY")
_groq_client = Groq(api_key=_groq_key) if _groq_key else None


def _groq(prompt: str) -> str:
    if _groq_client is None:
        return "⚠️ GROQ_API_KEY not set — AI report unavailable."
    response = _groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content


def generate_trend_report(
    keyword: str, phase: str, current: int, peak: int, peak_date: str,
    death_display: str, days_left, momentum_pct: float, avg_score: float,
    volatility: float, weeks_above_50: int,
) -> str:
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
    return _groq(prompt)


def generate_comparison_report(kw_a: str, data_a: dict, kw_b: str, data_b: dict) -> str:
    def fmt(d: dict) -> str:
        return (
            f"  - Current score: {d['current']}/100\n"
            f"  - All-time peak: {d['peak']}/100 on {d['peak_date']}\n"
            f"  - Phase: {d['phase']}\n"
            f"  - Predicted death: {d['death_display']}\n"
            f"  - 3-month momentum: {d['momentum_pct']:+.1f}%\n"
            f"  - 5-year average: {d['avg_score']:.1f}\n"
            f"  - Volatility (σ): {d['volatility']:.1f}\n"
            f"  - Weeks ≥ 50: {d['weeks_above_50']}"
        )
    prompt = f"""You are a trend analyst. Compare the trend lifecycles of "{kw_a}" and "{kw_b}" in 2–3 short paragraphs. Be direct — explain which is stronger, which is declining faster, which has a longer runway, and what it means for someone tracking these trends. Do not use bullet points or headers. Write in plain prose.

{kw_a}:
{fmt(data_a)}

{kw_b}:
{fmt(data_b)}

Write the comparison now:"""
    return _groq(prompt)


from data.fetch import fetch_trending_now
from data.mongo import get_recent_searches, save_ai_report, get_cached_result, save_comparison, get_recent_comparisons, save_comparison_report, get_comparison_report
from model.trend_phase import TrendPhase
from charts.lifecycle_chart import build_lifecycle_chart
from charts.velocity_chart import build_velocity_chart
from utils.cache import get_or_fetch

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trendlife",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Global styles ──────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: #f7f6f2 !important;
        color: #1a1a1a !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e5e7eb; }
    #MainMenu, footer { visibility: hidden; }

    /* Metrics */
    [data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    [data-testid="metric-container"] label {
        color: #6b7280 !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-weight: 600 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #1a1a1a !important;
        font-size: 1.6rem !important;
        font-weight: 700;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: #6b7280 !important;
    }

    /* Inputs */
    [data-testid="stTextInput"] input {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        color: #1a1a1a;
        font-size: 1rem;
        padding: 10px 14px;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: #e63946;
        box-shadow: 0 0 0 3px rgba(230,57,70,0.1);
    }
    [data-testid="stTextInput"] input::placeholder { color: #9ca3af; }

    /* Buttons */
    [data-testid="stButton"] button {
        background: #ffffff;
        color: #e63946;
        border: 1px solid #e63946;
        border-radius: 999px;
        padding: 10px 24px;
        font-weight: 500;
        font-size: 0.9rem;
        transition: background 0.2s, transform 0.15s;
    }
    [data-testid="stButton"] button:hover {
        background: #fdecea;
        transform: translateY(-1px);
    }
    [data-testid="stButton"] button:disabled {
        background: #f3f4f6 !important;
        color: #9ca3af !important;
        border-color: #e5e7eb !important;
        transform: none;
    }

    /* Selectbox */
    [data-testid="stSelectbox"] > div > div {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        color: #1a1a1a;
    }

    /* Radio */
    [data-testid="stRadio"] label { color: #1a1a1a !important; }
    [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p { color: #1a1a1a !important; }

    /* Divider */
    hr { border-color: #e5e7eb; margin: 1.5rem 0; }

    /* Recent row */
    .recent-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 6px;
        font-size: 0.88rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .recent-topic { color: #e63946; font-weight: 600; }
    .recent-meta  { color: #6b7280; font-size: 0.8rem; }

    /* Section label */
    .section-label {
        color: #e63946;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    /* Compare cards */
    .compare-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 20px 24px;
        height: 100%;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .compare-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 14px;
    }
    .compare-row {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid #f3f4f6;
        font-size: 0.88rem;
    }
    .compare-row:last-child { border-bottom: none; }
    .compare-label { color: #6b7280; }
    .compare-value { color: #1a1a1a; font-weight: 600; }
    .winner { color: #16a34a !important; }

    /* Alerts / info boxes */
    [data-testid="stAlert"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        color: #1a1a1a;
    }

    /* Spinner */
    [data-testid="stSpinner"] { color: #e63946 !important; }

    /* Expander */
    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
    }
    [data-testid="stExpander"] summary {
        background: #ffffff !important;
        color: #1a1a1a !important;
    }
    [data-testid="stExpander"] summary *,
    .st-emotion-cache-11ofl8m {
        color: #1a1a1a !important;
    }

    /* Tabs */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        background: transparent;
        border-bottom: 1px solid #e5e7eb;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        background: transparent;
        color: #6b7280 !important;
        font-weight: 500;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        color: #1a1a1a !important;
        border-bottom: 2px solid #e63946;
    }
    [data-testid="stTabs"] [data-baseweb="tab-panel"] {
        background: #ffffff;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        background: #ffffff;
    }
    [data-testid="stDataFrame"] iframe {
        background: #ffffff;
        color-scheme: light;
    }
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background: #ffffff;
    }

    /* Download button — keep consistent */
    [data-testid="stDownloadButton"] button {
        background: #ffffff !important;
        color: #1a1a1a !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 999px !important;
        font-weight: 500 !important;
    }
    [data-testid="stDownloadButton"] button:hover {
        background: #f3f4f6 !important;
        transform: translateY(-1px);
    }

    /* General text visibility */
    [data-testid="stMarkdownContainer"] p { color: #1a1a1a; }

    /* Keep button text red */
    [data-testid="stButton"] button,
    [data-testid="stButton"] button span,
    [data-testid="stButton"] button p {
        color: #e63946 !important;
    }
    [data-testid="stButton"] button:disabled,
    [data-testid="stButton"] button:disabled span {
        color: #9ca3af !important;
    }

    /* Tab text */
    [data-testid="stTabs"] button p,
    [data-testid="stTabs"] button span {
        color: inherit !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="padding: 2rem 0 1rem 0;">
        <h1 style="font-size:2.2rem; font-weight:800; color:#1a1a1a; margin:0; letter-spacing:-0.03em;">
            Trend<span style="color:#e63946">Life</span>
        </h1>
        <p style="color:#6b7280; font-size:1rem; margin:6px 0 0 0; letter-spacing:0.01em;">
            Every trend has a lifecycle. Find out where yours stands.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<hr>", unsafe_allow_html=True)

# ─── Mode toggle ────────────────────────────────────────────────────────────
mode = st.radio(
    "mode",
    ["Single topic", "Compare two topics"],
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown("<br>", unsafe_allow_html=True)

trending_topics = fetch_trending_now()

# ─── Search area ────────────────────────────────────────────────────────────
if mode == "Single topic":
    if "trending_select" in st.session_state:
        chosen = st.session_state["trending_select"]
        if chosen and chosen != "— select —":
            st.session_state["search_input"] = chosen

    col_input, col_btn, col_trending = st.columns([4, 1, 3])
    with col_input:
        st.markdown('<p class="section-label">Search any topic</p>', unsafe_allow_html=True)
        search_query = st.text_input(
            label="search", placeholder='e.g. "ChatGPT", "Wordle", "NFT"',
            label_visibility="collapsed", key="search_input",
        )
    with col_btn:
        st.markdown('<p class="section-label">&nbsp;</p>', unsafe_allow_html=True)
        analyze_clicked = st.button("Analyze", use_container_width=True)
    with col_trending:
        st.markdown('<p class="section-label">Or pick a suggested topic</p>', unsafe_allow_html=True)
        if trending_topics:
            st.selectbox(
                label="trending", options=["— select —"] + trending_topics,
                label_visibility="collapsed", key="trending_select",
            )
        else:
            st.caption("Could not load trending topics")

    if analyze_clicked and search_query.strip():
        st.session_state["active_keyword"] = search_query.strip()
        st.session_state.pop("compare_kw_a", None)
        st.session_state.pop("compare_kw_b", None)

else:  # Compare mode
    col_a, col_b, col_btn = st.columns([4, 4, 1])
    with col_a:
        st.markdown('<p class="section-label">Topic A</p>', unsafe_allow_html=True)
        query_a = st.text_input(
            label="topic_a", placeholder='e.g. "ChatGPT"',
            label_visibility="collapsed", key="compare_input_a",
        )
    with col_b:
        st.markdown('<p class="section-label">Topic B</p>', unsafe_allow_html=True)
        query_b = st.text_input(
            label="topic_b", placeholder='e.g. "Claude AI"',
            label_visibility="collapsed", key="compare_input_b",
        )
    with col_btn:
        st.markdown('<p class="section-label">&nbsp;</p>', unsafe_allow_html=True)
        compare_clicked = st.button("Compare", use_container_width=True)

    if compare_clicked and query_a.strip() and query_b.strip():
        st.session_state["compare_kw_a"] = query_a.strip()
        st.session_state["compare_kw_b"] = query_b.strip()
        st.session_state.pop("active_keyword", None)

# ─── helpers ────────────────────────────────────────────────────────────────

def _extract_metrics(result: dict, forecast_df) -> dict:
    """Pull all display-ready metrics out of a result dict."""
    raw_data = result.get("raw_data", [])
    current  = result.get("current_score", 0)
    peak     = result.get("peak_score", 0)

    avg_score = volatility = median_score = momentum_pct = pct_of_peak = 0.0
    weeks_above_50 = 0
    hist_df = None

    if raw_data:
        hist_df   = pd.DataFrame(raw_data)
        hist_df["date"] = pd.to_datetime(hist_df["date"])
        hist_df   = hist_df.sort_values("date")
        scores    = hist_df["interest"]
        avg_score      = float(scores.mean())
        volatility     = float(scores.std())
        median_score   = float(scores.median())
        score_3m_ago   = float(scores.iloc[-13]) if len(scores) >= 13 else float(scores.iloc[0])
        momentum_pct   = ((current - score_3m_ago) / score_3m_ago * 100) if score_3m_ago else 0.0
        pct_of_peak    = (current / peak * 100) if peak else 0.0
        weeks_above_50 = int((scores >= 50).sum())

    death_str     = result.get("predicted_death")
    death_display = (
        datetime.strptime(death_str, "%Y-%m-%d").strftime("%b %d, %Y")
        if death_str else "Beyond forecast"
    )
    days_left   = result.get("days_remaining")
    days_display = f"{days_left} days" if days_left is not None else "N/A"
    if days_left == 0:
        days_display = "Already Dead"

    phase_str = result.get("trend_phase", "Unknown")
    try:
        phase = TrendPhase(phase_str)
    except ValueError:
        phase = TrendPhase.DECLINING

    return dict(
        current=current, peak=peak,
        peak_date=result.get("peak_date", "—"),
        death_str=death_str, death_display=death_display,
        days_left=days_left, days_display=days_display,
        conf_upper=result.get("confidence_upper", 0.0),
        conf_lower=result.get("confidence_lower", 0.0),
        phase_str=phase_str, phase=phase,
        avg_score=avg_score, volatility=volatility,
        median_score=median_score, momentum_pct=momentum_pct,
        pct_of_peak=pct_of_peak, weeks_above_50=weeks_above_50,
        raw_data=raw_data, hist_df=hist_df,
        forecast_df=forecast_df,
    )


def _render_stat_cards(m: dict, keyword: str) -> None:
    st.markdown(
        f'<p class="section-label">Analysis — <span style="color:#e63946">{keyword}</span></p>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current Score",   f"{m['current']} / 100")
    c2.metric("All-time Peak",   f"{m['peak']} / 100", delta=m['peak_date'], delta_color="off")
    c3.metric("Trend Phase",     f"{m['phase'].emoji()} {m['phase'].value}")
    c4.metric("Predicted Death", m['death_display'])
    c5.metric("Days Until Death", m['days_display'])

    if m['death_str']:
        st.markdown(
            f'<p style="color:#6b7280; font-size:0.8rem; margin-top:4px;">'
            f'80% confidence interval at death: '
            f'<span style="color:#e63946">{m["conf_lower"]:.1f}</span> – '
            f'<span style="color:#e63946">{m["conf_upper"]:.1f}</span>'
            f'</p>',
            unsafe_allow_html=True,
        )


def _render_extended_metrics(m: dict) -> None:
    st.markdown('<p class="section-label">Extended metrics</p>', unsafe_allow_html=True)
    if m['raw_data']:
        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
        m1.metric("Avg Score (5y)",   f"{m['avg_score']:.1f}")
        m2.metric("Median Score",     f"{m['median_score']:.1f}")
        m3.metric("Volatility (σ)",   f"{m['volatility']:.1f}")
        m4.metric("3-month Momentum", f"{m['momentum_pct']:+.1f}%")
        m5.metric("% of Peak",        f"{m['pct_of_peak']:.1f}%")
        m6.metric("Weeks ≥ 50",       str(m['weeks_above_50']))
        m7.metric("Data Points",      str(len(m['hist_df'])))


def _render_data_tables(m: dict, keyword: str) -> None:
    with st.expander("📊 Raw data & forecast tables", expanded=False):
        tab_hist, tab_forecast = st.tabs(["Historical interest", "Forecast"])
        with tab_hist:
            if m['raw_data']:
                dh = m['hist_df'].copy()
                dh["date"] = dh["date"].dt.strftime("%Y-%m-%d")
                dh.columns = ["Date", "Interest Score"]
                dh = dh.sort_values("Date", ascending=False).reset_index(drop=True)
                st.dataframe(dh, use_container_width=True, height=320, hide_index=True)
                st.download_button(
                    "⬇ Download historical CSV",
                    dh.to_csv(index=False).encode("utf-8"),
                    file_name=f"{keyword}_historical.csv", mime="text/csv",
                )
            else:
                st.info("No historical data available.")
        with tab_forecast:
            fdf = m['forecast_df']
            if fdf is not None:
                today_ts = pd.Timestamp(date.today())
                fd = fdf[fdf["ds"] >= today_ts][["ds","yhat","yhat_lower","yhat_upper"]].copy()
                fd["ds"] = fd["ds"].dt.strftime("%Y-%m-%d")
                fd = fd.rename(columns={"ds":"Date","yhat":"Forecast","yhat_lower":"Lower (80%)","yhat_upper":"Upper (80%)"})
                fd[["Forecast","Lower (80%)","Upper (80%)"]] = fd[["Forecast","Lower (80%)","Upper (80%)"]].round(1)
                st.dataframe(fd.reset_index(drop=True), use_container_width=True, height=320, hide_index=True)
                st.download_button(
                    "⬇ Download forecast CSV",
                    fd.to_csv(index=False).encode("utf-8"),
                    file_name=f"{keyword}_forecast.csv", mime="text/csv",
                )
            else:
                st.info("No forecast data available.")


def _render_ai_report_single(keyword: str, m: dict) -> None:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">AI analyst report</p>', unsafe_allow_html=True)

    report_key = f"report_{keyword}"
    report_exists = bool(report_key in st.session_state and st.session_state[report_key])

    col_btn_report, _ = st.columns([1, 4])
    with col_btn_report:
        if st.button("Generate AI report", key="gen_report_btn", disabled=report_exists):
            st.session_state[report_key] = None

    if report_key in st.session_state:
        if st.session_state[report_key] is None:
            with st.spinner("Generating report…"):
                st.session_state[report_key] = generate_trend_report(
                    keyword=keyword, phase=m['phase_str'],
                    current=m['current'], peak=m['peak'], peak_date=m['peak_date'],
                    death_display=m['death_display'], days_left=m['days_left'],
                    momentum_pct=m['momentum_pct'], avg_score=m['avg_score'],
                    volatility=m['volatility'], weeks_above_50=m['weeks_above_50'],
                )
                save_ai_report(keyword, st.session_state[report_key])
            st.rerun()

        if st.session_state[report_key]:
            st.markdown(
                f"""<div style="background:#ffffff; border:1px solid #e5e7eb; border-radius:10px;
                            padding:20px 24px; line-height:1.7; color:#1a1a1a; font-size:0.95rem;">
                    {st.session_state[report_key].replace(chr(10), '<br>')}
                </div>""",
                unsafe_allow_html=True,
            )


def _fetch_result(keyword: str):
    """Fetch + forecast for a keyword, returning (result, forecast_df, error)."""
    try:
        result = get_or_fetch(keyword)
        forecast_df = result.get("forecast_df")
        if forecast_df is None and result.get("raw_data"):
            raw_df = pd.DataFrame(result["raw_data"])
            raw_df["date"] = pd.to_datetime(raw_df["date"])
            raw_df = raw_df.set_index("date")
            from model.prophet_model import fit_and_forecast
            forecast_df = fit_and_forecast(raw_df, periods=365)
        return result, forecast_df, None
    except Exception as e:
        return None, None, str(e)


# ─── Single topic analysis ───────────────────────────────────────────────────
keyword = st.session_state.get("active_keyword", "")

if keyword and mode == "Single topic":
    st.markdown("<hr>", unsafe_allow_html=True)

    with st.spinner(f"Fetching data and running forecast for **{keyword}**…"):
        result, forecast_df, error = _fetch_result(keyword)

    if error:
        st.error(f"**Could not analyze '{keyword}'** — {error}")
        cached = get_cached_result(keyword)
        if cached:
            st.info("Showing last cached result instead.")
            result = cached
            forecast_df = None
            if result.get("raw_data"):
                raw_df = pd.DataFrame(result["raw_data"])
                raw_df["date"] = pd.to_datetime(raw_df["date"])
                raw_df = raw_df.set_index("date")
                with st.spinner("Re-running forecast from cached data…"):
                    from model.prophet_model import fit_and_forecast
                    forecast_df = fit_and_forecast(raw_df, periods=365)

    if result:
        # Pre-load stored AI report
        report_key = f"report_{keyword}"
        if report_key not in st.session_state and result.get("ai_report"):
            st.session_state[report_key] = result["ai_report"]

        m = _extract_metrics(result, forecast_df)

        _render_stat_cards(m, keyword)
        st.markdown("<hr>", unsafe_allow_html=True)

        if forecast_df is not None:
            death_date_obj = (
                datetime.strptime(m['death_str'], "%Y-%m-%d").date() if m['death_str'] else None
            )
            chart_col, vel_col = st.columns([3, 2])
            with chart_col:
                fig_life = build_lifecycle_chart(forecast_df, keyword, m['phase'], death_date_obj)
                st.plotly_chart(fig_life, use_container_width=True, config={"displayModeBar": False})
            with vel_col:
                fig_vel = build_velocity_chart(forecast_df, keyword)
                st.plotly_chart(fig_vel, use_container_width=True, config={"displayModeBar": False})
        else:
            st.warning("Forecast chart unavailable — no model output found.")

        st.markdown("<hr>", unsafe_allow_html=True)
        _render_extended_metrics(m)
        st.markdown("<hr>", unsafe_allow_html=True)
        _render_data_tables(m, keyword)
        _render_ai_report_single(keyword, m)


# ─── Compare two topics ──────────────────────────────────────────────────────
kw_a = st.session_state.get("compare_kw_a", "")
kw_b = st.session_state.get("compare_kw_b", "")

if kw_a and kw_b and mode == "Compare two topics":
    st.markdown("<hr>", unsafe_allow_html=True)

    with st.spinner(f"Fetching data for **{kw_a}** and **{kw_b}**…"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(_fetch_result, kw_a)
            fut_b = pool.submit(_fetch_result, kw_b)
            res_a, fdf_a, err_a = fut_a.result()
            res_b, fdf_b, err_b = fut_b.result()

    if err_a:
        st.error(f"Could not fetch '{kw_a}': {err_a}")
    if err_b:
        st.error(f"Could not fetch '{kw_b}': {err_b}")

    if res_a and res_b:
        _saved_cmp_key = f"_cmp_saved_{kw_a}_{kw_b}"
        if _saved_cmp_key not in st.session_state:
            save_comparison(kw_a, kw_b)
            st.session_state[_saved_cmp_key] = True
        ma = _extract_metrics(res_a, fdf_a)
        mb = _extract_metrics(res_b, fdf_b)

        # ── Overlaid lifecycle chart ─────────────────────────────────────────
        st.markdown('<p class="section-label">Interest over time — comparison</p>', unsafe_allow_html=True)

        _BG, _GRID = "#ffffff", "#f3f4f6"
        COLOR_A, COLOR_B = "#e63946", "#1a1a1a"
        FCAST_A, FCAST_B = "#f87171", "#6b7280"
        BAND_A,  BAND_B  = "rgba(230,57,70,0.1)", "rgba(26,26,26,0.08)"

        fig_overlay = go.Figure()

        for (fdf, kw, col_hist, col_fcast, col_band) in [
            (fdf_a, kw_a, COLOR_A, FCAST_A, BAND_A),
            (fdf_b, kw_b, COLOR_B, FCAST_B, BAND_B),
        ]:
            if fdf is None:
                continue
            hist    = fdf[fdf["y"].notna()]
            today_ts = pd.Timestamp(date.today())
            future  = fdf[fdf["ds"] >= today_ts]

            fig_overlay.add_trace(go.Scatter(
                x=hist["ds"], y=hist["y"], mode="lines",
                line=dict(color=col_hist, width=2.5), name=f"{kw} (historical)",
            ))
            fig_overlay.add_trace(go.Scatter(
                x=future["ds"], y=future["yhat"], mode="lines",
                line=dict(color=col_fcast, width=2, dash="dash"), name=f"{kw} (forecast)",
            ))
            fig_overlay.add_trace(go.Scatter(
                x=pd.concat([future["ds"], future["ds"].iloc[::-1]]),
                y=pd.concat([future["yhat_upper"], future["yhat_lower"].iloc[::-1]]),
                fill="toself",
                fillcolor=col_band,
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip", showlegend=False,
            ))

        fig_overlay.add_hline(y=10, line_dash="dot", line_color="#ef4444", line_width=1.5,
                               annotation_text="Death threshold", annotation_font_color="#ef4444")
        fig_overlay.update_layout(
            paper_bgcolor=_BG, plot_bgcolor=_BG, font=dict(color="#1a1a1a"),
            xaxis=dict(showgrid=True, gridcolor=_GRID, zeroline=False, title="Date"),
            yaxis=dict(showgrid=True, gridcolor=_GRID, zeroline=False,
                       title="Interest Score (0–100)", range=[0, 105]),
            legend=dict(
                bgcolor="#ffffff",
                bordercolor="#e5e7eb",
                borderwidth=1,
                font=dict(size=13, color="#1a1a1a"),
                itemsizing="constant",
                tracegroupgap=6,
            ),
            hovermode="x unified", margin=dict(l=50, r=30, t=40, b=50),
        )
        st.plotly_chart(fig_overlay, use_container_width=True, config={"displayModeBar": False})

        # ── Side-by-side velocity charts ─────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">Trend velocity</p>', unsafe_allow_html=True)
        vel_col_a, vel_col_b = st.columns(2)
        with vel_col_a:
            if fdf_a is not None:
                st.plotly_chart(build_velocity_chart(fdf_a, kw_a),
                                use_container_width=True, config={"displayModeBar": False})
        with vel_col_b:
            if fdf_b is not None:
                st.plotly_chart(build_velocity_chart(fdf_b, kw_b),
                                use_container_width=True, config={"displayModeBar": False})

        # ── Side-by-side stat cards ──────────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">Head-to-head metrics</p>', unsafe_allow_html=True)

        def _winner(val_a, val_b, higher_is_better=True):
            if higher_is_better:
                return "winner" if val_a > val_b else "", "winner" if val_b > val_a else ""
            else:
                return "winner" if val_a < val_b else "", "winner" if val_b < val_a else ""

        rows = [
            ("Current Score",    f"{ma['current']}/100",             f"{mb['current']}/100",         True,  ma['current'],        mb['current']),
            ("All-time Peak",    f"{ma['peak']}/100",                f"{mb['peak']}/100",             True,  ma['peak'],           mb['peak']),
            ("Phase",            f"{ma['phase'].emoji()} {ma['phase'].value}", f"{mb['phase'].emoji()} {mb['phase'].value}", None, 0, 0),
            ("Predicted Death",  ma['death_display'],                mb['death_display'],             None,  0, 0),
            ("Days Until Death", ma['days_display'],                 mb['days_display'],              False, ma['days_left'] or 9999, mb['days_left'] or 9999),
            ("3-month Momentum", f"{ma['momentum_pct']:+.1f}%",     f"{mb['momentum_pct']:+.1f}%",   True,  ma['momentum_pct'],   mb['momentum_pct']),
            ("Avg Score (5y)",   f"{ma['avg_score']:.1f}",          f"{mb['avg_score']:.1f}",         True,  ma['avg_score'],      mb['avg_score']),
            ("Volatility (σ)",   f"{ma['volatility']:.1f}",         f"{mb['volatility']:.1f}",        False, ma['volatility'],     mb['volatility']),
            ("% of Peak",        f"{ma['pct_of_peak']:.1f}%",       f"{mb['pct_of_peak']:.1f}%",      True,  ma['pct_of_peak'],    mb['pct_of_peak']),
            ("Weeks ≥ 50",       str(ma['weeks_above_50']),          str(mb['weeks_above_50']),        True,  ma['weeks_above_50'], mb['weeks_above_50']),
        ]

        card_a_html = f'<div class="compare-card"><div class="compare-title" style="color:{COLOR_A}">{kw_a}</div>'
        card_b_html = f'<div class="compare-card"><div class="compare-title" style="color:{COLOR_B}">{kw_b}</div>'

        for label, val_a, val_b, higher_is_better, num_a, num_b in rows:
            if higher_is_better is not None:
                cls_a, cls_b = _winner(num_a, num_b, higher_is_better)
            else:
                cls_a = cls_b = ""
            card_a_html += f'<div class="compare-row"><span class="compare-label">{label}</span><span class="compare-value {cls_a}">{val_a}</span></div>'
            card_b_html += f'<div class="compare-row"><span class="compare-label">{label}</span><span class="compare-value {cls_b}">{val_b}</span></div>'

        card_a_html += "</div>"
        card_b_html += "</div>"

        col_card_a, col_card_b = st.columns(2)
        with col_card_a:
            st.markdown(card_a_html, unsafe_allow_html=True)
        with col_card_b:
            st.markdown(card_b_html, unsafe_allow_html=True)

        # ── AI comparison report ─────────────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">AI comparison report</p>', unsafe_allow_html=True)

        cmp_key = f"compare_report_{kw_a}_vs_{kw_b}"
        if cmp_key not in st.session_state:
            stored = get_comparison_report(kw_a, kw_b)
            if stored:
                st.session_state[cmp_key] = stored
        cmp_report_exists = bool(cmp_key in st.session_state and st.session_state[cmp_key])

        col_cmp_btn, _ = st.columns([1, 4])
        with col_cmp_btn:
            if st.button("Generate comparison report", key="gen_cmp_btn", disabled=cmp_report_exists):
                st.session_state[cmp_key] = None

        if cmp_key in st.session_state:
            if st.session_state[cmp_key] is None:
                with st.spinner("Generating comparison report…"):
                    st.session_state[cmp_key] = generate_comparison_report(
                        kw_a, dict(
                            current=ma['current'], peak=ma['peak'], peak_date=ma['peak_date'],
                            phase=ma['phase_str'], death_display=ma['death_display'],
                            momentum_pct=ma['momentum_pct'], avg_score=ma['avg_score'],
                            volatility=ma['volatility'], weeks_above_50=ma['weeks_above_50'],
                        ),
                        kw_b, dict(
                            current=mb['current'], peak=mb['peak'], peak_date=mb['peak_date'],
                            phase=mb['phase_str'], death_display=mb['death_display'],
                            momentum_pct=mb['momentum_pct'], avg_score=mb['avg_score'],
                            volatility=mb['volatility'], weeks_above_50=mb['weeks_above_50'],
                        ),
                    )
                    save_comparison_report(kw_a, kw_b, st.session_state[cmp_key])
                st.rerun()

            if st.session_state[cmp_key]:
                st.markdown(
                    f"""<div style="background:#ffffff; border:1px solid #e5e7eb; border-radius:10px;
                                padding:20px 24px; line-height:1.7; color:#1a1a1a; font-size:0.95rem;">
                        {st.session_state[cmp_key].replace(chr(10), '<br>')}
                    </div>""",
                    unsafe_allow_html=True,
                )

# ─── Recent searches / comparisons ─────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)

if mode == "Compare two topics":
    st.markdown('<p class="section-label">Recent comparisons</p>', unsafe_allow_html=True)
    recent_cmp = get_recent_comparisons(limit=10)
    if not recent_cmp:
        st.markdown(
            '<p style="color:#6b7280; font-size:0.88rem;">No comparisons yet — compare two topics above to get started.</p>',
            unsafe_allow_html=True,
        )
    else:
        cols = st.columns(2)
        for i, doc in enumerate(recent_cmp):
            a           = doc.get("kw_a", "—")
            b           = doc.get("kw_b", "—")
            compared_at = doc.get("compared_at", None)
            date_label  = compared_at.strftime("%b %d, %H:%M") if compared_at else ""
            with cols[i % 2]:
                st.markdown(
                    f"""
                    <div class="recent-row">
                        <div>
                            <span class="recent-topic">{a}</span>
                            <span style="color:#6b7280; margin:0 6px;">vs</span>
                            <span class="recent-topic">{b}</span>
                        </div>
                        <div class="recent-meta">{date_label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"Compare again", key=f"cmp_again_{i}", use_container_width=True):
                    st.session_state["compare_kw_a"] = a
                    st.session_state["compare_kw_b"] = b
                    st.rerun()
else:
    st.markdown('<p class="section-label">Recent searches</p>', unsafe_allow_html=True)
    recent = get_recent_searches(limit=10)
    if not recent:
        st.markdown(
            '<p style="color:#6b7280; font-size:0.88rem;">No searches yet — analyze a topic above to get started.</p>',
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
                phase_obj   = TrendPhase(phase_val)
                phase_color = phase_obj.color()
                phase_emoji = phase_obj.emoji()
            except Exception:
                phase_color = "#8b949e"
                phase_emoji = ""

            death_label = (
                datetime.strptime(death, "%Y-%m-%d").strftime("%b %d, %Y") if death else "Unknown"
            )
            date_label = searched_at.strftime("%b %d, %H:%M") if searched_at else ""

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
                            Score: <b style="color:#1a1a1a">{score}</b> &nbsp;·&nbsp;
                            Dies: <b style="color:#1a1a1a">{death_label}</b> &nbsp;·&nbsp;
                            {date_label}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"View {topic}", key=f"recent_{i}", use_container_width=True):
                    st.session_state["active_keyword"] = topic
                    st.session_state.pop("compare_kw_a", None)
                    st.session_state.pop("compare_kw_b", None)
                    st.rerun()
