import pandas as pd
import plotly.graph_objects as go

from utils.logger import get_logger

logger = get_logger(__name__)

_BG = "#0e1117"
_PAPER = "#0e1117"
_GRID = "#1f2937"
_POS = "rgba(34,197,94,0.75)"     # green
_NEG = "rgba(239,68,68,0.75)"     # red
_MA_LINE = "#fbbf24"              # amber rolling average


def build_velocity_chart(forecast_df: pd.DataFrame, keyword: str) -> go.Figure:
    """
    Build the trend velocity (momentum) chart.

    Shows week-over-week change in yhat as a colour-coded bar chart
    with a 4-week rolling average overlay.
    """
    df = forecast_df[["ds", "yhat"]].copy().sort_values("ds").reset_index(drop=True)

    # Week-over-week percentage change (7-day shift)
    df["velocity"] = df["yhat"].diff(7)

    # 4-week rolling average of velocity
    df["rolling_avg"] = df["velocity"].rolling(window=28, min_periods=7).mean()

    # Colour per bar
    colors = [_POS if v >= 0 else _NEG for v in df["velocity"]]

    fig = go.Figure()

    # --- Velocity bars ---
    fig.add_trace(
        go.Bar(
            x=df["ds"],
            y=df["velocity"],
            marker_color=colors,
            name="Weekly Velocity",
            hovertemplate="%{x|%b %d, %Y}<br>Velocity: %{y:.2f}<extra></extra>",
        )
    )

    # --- Rolling average line ---
    fig.add_trace(
        go.Scatter(
            x=df["ds"],
            y=df["rolling_avg"],
            mode="lines",
            line=dict(color=_MA_LINE, width=2),
            name="4-week Avg",
        )
    )

    # --- Zero line ---
    fig.add_hline(y=0, line_color="#6b7280", line_width=1)

    fig.update_layout(
        title=dict(
            text=f"<b>{keyword}</b> — Trend Velocity (Week-over-Week Change)",
            font=dict(size=14, color="#f9fafb"),
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
            title="Change in Interest Score",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=_GRID,
            borderwidth=1,
            font=dict(color="#d1d5db"),
        ),
        bargap=0.1,
        hovermode="x unified",
        margin=dict(l=50, r=30, t=60, b=50),
    )

    return fig
