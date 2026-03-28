from enum import Enum
from datetime import date

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

DEATH_THRESHOLD = 10          # score below this → Dead
PEAK_RATIO = 0.90             # within 90% of all-time peak → Peaking
DECLINING_RATIO = 0.50        # below 50% of all-time peak → Declining
THREE_MONTH_DAYS = 90


class TrendPhase(str, Enum):
    RISING = "Rising"
    PEAKING = "Peaking"
    DECLINING = "Declining"
    DEAD = "Dead"

    def color(self) -> str:
        return {
            TrendPhase.RISING: "#22c55e",
            TrendPhase.PEAKING: "#f59e0b",
            TrendPhase.DECLINING: "#f97316",
            TrendPhase.DEAD: "#ef4444",
        }[self]

    def emoji(self) -> str:
        return {
            TrendPhase.RISING: "📈",
            TrendPhase.PEAKING: "🔝",
            TrendPhase.DECLINING: "📉",
            TrendPhase.DEAD: "💀",
        }[self]


def classify_phase(forecast_df: pd.DataFrame, today: date) -> TrendPhase:
    """
    Classify the current trend phase using historical yhat values.

    Rules (evaluated in order):
      Dead      → current_score < 10
      Peaking   → current_score >= all_time_peak * 0.90
      Declining → current_score < all_time_peak * 0.50
      Rising    → current_score > 3-month average AND current_score < all_time_peak * 0.90
      (fallback)→ Declining
    """
    today_ts = pd.Timestamp(today)

    # Historical rows only (where y is not NaN)
    historical = forecast_df[forecast_df["y"].notna()].copy()

    if historical.empty:
        logger.warning("No historical data to classify phase — defaulting to Declining")
        return TrendPhase.DECLINING

    # Current score = latest historical y value
    current_score = float(historical.sort_values("ds").iloc[-1]["y"])
    all_time_peak = float(historical["y"].max())

    # 3-month average
    three_months_ago = today_ts - pd.Timedelta(days=THREE_MONTH_DAYS)
    recent = historical[historical["ds"] >= three_months_ago]
    three_month_avg = float(recent["y"].mean()) if not recent.empty else current_score

    logger.info(
        f"Phase classification — current: {current_score:.1f}, "
        f"peak: {all_time_peak:.1f}, 3m_avg: {three_month_avg:.1f}"
    )

    if current_score < DEATH_THRESHOLD:
        return TrendPhase.DEAD
    if current_score >= all_time_peak * PEAK_RATIO:
        return TrendPhase.PEAKING
    if current_score < all_time_peak * DECLINING_RATIO:
        return TrendPhase.DECLINING
    if current_score > three_month_avg:
        return TrendPhase.RISING
    return TrendPhase.DECLINING
