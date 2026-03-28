from datetime import date
from typing import Optional

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

DEATH_THRESHOLD = 10.0  # interest score below this = effectively dead


def find_death_date(forecast_df: pd.DataFrame, threshold: float = DEATH_THRESHOLD) -> Optional[date]:
    """
    Scan future forecast rows and return the first date where yhat drops below threshold.

    Only looks at rows beyond today (y is NaN = future rows).
    Returns None if the trend stays above the threshold for the entire forecast window.
    """
    today = pd.Timestamp(date.today())
    future = forecast_df[forecast_df["ds"] > today].copy()

    if future.empty:
        logger.warning("No future rows in forecast — cannot detect death date")
        return None

    below = future[future["yhat"] < threshold]
    if below.empty:
        logger.info(f"Trend stays above {threshold} for the full forecast window — no death detected")
        return None

    death_ts = below.iloc[0]["ds"]
    death_date = death_ts.date() if hasattr(death_ts, "date") else date.fromisoformat(str(death_ts)[:10])
    logger.info(f"Death date detected: {death_date} (yhat={below.iloc[0]['yhat']:.2f})")
    return death_date


def days_until_death(death_date: Optional[date], today: Optional[date] = None) -> Optional[int]:
    """Return the number of days from today until the death date, or None if no death detected."""
    if death_date is None:
        return None
    today = today or date.today()
    delta = (death_date - today).days
    return max(delta, 0)


def get_confidence_at_death(forecast_df: pd.DataFrame, death_date: Optional[date]) -> tuple[float, float]:
    """
    Return the (yhat_upper, yhat_lower) confidence interval at the predicted death date.
    Falls back to (0.0, 0.0) if death_date is None or not found in the forecast.
    """
    if death_date is None:
        return (0.0, 0.0)

    death_ts = pd.Timestamp(death_date)
    row = forecast_df[forecast_df["ds"] == death_ts]
    if row.empty:
        return (0.0, 0.0)

    upper = float(row.iloc[0]["yhat_upper"])
    lower = float(row.iloc[0]["yhat_lower"])
    return (upper, lower)
