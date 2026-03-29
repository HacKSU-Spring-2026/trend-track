from datetime import date
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from model.trend_phase import TrendPhase
from utils.logger import get_logger

logger = get_logger(__name__)

_BG = "#0e1117"
_PAPER = "#0e1117"
_GRID = "#1f2937"
_HISTORICAL = "#60a5fa"      # blue
_FORECAST = "#a78bfa"        # purple
_INTERVAL = "rgba(167,139,250,0.15)"
_THRESHOLD = "#ef4444"       # red
_DEATH_LINE = "rgba(239,68,68,0.8)"


def build_lifecycle_chart(
    forecast_df: pd.DataFrame,
    keyword: str,
    phase: TrendPhase,
    death_date: Optional[date],
) -> go.Figure:
    """
    Build the main lifecycle chart:
      - Solid blue line   : historical interest (y column)
      - Dashed purple line: Prophet forecast (yhat)
      - Shaded band       : yhat_lower → yhat_upper confidence interval
      - Red dashed line   : death threshold at y=10
      - Red vertical line : predicted death date (if any)
    """
    fig = go.Figure()

    historical = forecast_df[forecast_df["y"].notna()].copy()
    future = forecast_df[forecast_df["y"].isna()].copy()
    today_ts = pd.Timestamp(date.today())
    forecast_from_today = forecast_df[forecast_df["ds"] >= today_ts].copy()

    # --- Confidence interval band (full forecast) ---
    fig.add_trace(
        go.Scatter(
            x=pd.concat([forecast_from_today["ds"], forecast_from_today["ds"].iloc[::-1]]),
            y=pd.concat([forecast_from_today["yhat_upper"], forecast_from_today["yhat_lower"].iloc[::-1]]),
            fill="toself",
            fillcolor=_INTERVAL,
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="80% Confidence",
        )
    )

    # --- Forecast line (future only) ---
    fig.add_trace(
        go.Scatter(
            x=forecast_from_today["ds"],
            y=forecast_from_today["yhat"],
            mode="lines",
            line=dict(color=_FORECAST, width=2, dash="dash"),
            name="Forecast",
        )
    )

    # --- Historical interest ---
    fig.add_trace(
        go.Scatter(
            x=historical["ds"],
            y=historical["y"],
            mode="lines",
            line=dict(color=_HISTORICAL, width=2.5),
            name="Interest Score",
        )
    )

    # --- Death threshold line ---
    fig.add_hline(
        y=10,
        line_dash="dot",
        line_color=_THRESHOLD,
        line_width=1.5,
        annotation_text="Death threshold (10)",
        annotation_position="top right",
        annotation_font_color=_THRESHOLD,
    )

    # --- Death date vertical marker ---
    if death_date is not None:
        death_ts_ms = pd.Timestamp(death_date).value / 1e6  # plotly expects milliseconds
        fig.add_vline(
            x=death_ts_ms,
            line_dash="dash",
            line_color=_DEATH_LINE,
            line_width=2,
            annotation_text=f"💀 {death_date.strftime('%b %d, %Y')}",
            annotation_position="top left",
            annotation_font_color=_THRESHOLD,
        )

    # --- Phase badge annotation ---
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.99,
        text=f"{phase.emoji()} {phase.value}",
        showarrow=False,
        font=dict(size=14, color=phase.color()),
        bgcolor="rgba(0,0,0,0.4)",
        bordercolor=phase.color(),
        borderwidth=1,
        borderpad=6,
        align="left",
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{keyword}</b> — Interest Over Time & Forecast",
            font=dict(size=16, color="#f9fafb"),
        ),
        paper_bgcolor=_PAPER,
        plot_bgcolor=_BG,
        font=dict(color="#d1d5db"),
        xaxis=dict(
            showgrid=True,
            gridcolor=_GRID,
            zeroline=False,
            title="Date",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=_GRID,
            zeroline=False,
            title="Interest Score (0–100)",
            range=[0, 105],
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=_GRID,
            borderwidth=1,
            font=dict(color="#d1d5db"),
        ),
        hovermode="x unified",
        margin=dict(l=50, r=30, t=60, b=50),
    )

    return fig
