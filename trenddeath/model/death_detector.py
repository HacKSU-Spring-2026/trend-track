from datetime import date
from typing import Optional

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

DEATH_THRESHOLD = 10.0  # interest score below this = effectively dead
DEATH_CONSECUTIVE_WEEKS = 4  # must stay below threshold for this many weeks in a row


def find_death_date(
    forecast_df: pd.DataFrame,
    threshold: float = DEATH_THRESHOLD,
    use_confidence_bound: bool = True,
) -> Optional[date]:
    """
    Scan future forecast rows and return the first date of a sustained drop below threshold.

    A trend is considered dead only when the chosen signal stays below the threshold
    for DEATH_CONSECUTIVE_WEEKS consecutive weeks.

    When use_confidence_bound=True (default), uses yhat_upper instead of yhat.
    This means we only call a trend dead when even the optimistic upper bound of
    the 80% confidence interval is below the threshold — significantly reducing
    false death predictions for volatile or seasonal trends.

    Only looks at rows beyond today (y is NaN = future rows).
    Returns None if no sustained drop is found in the forecast window.
    """
    today = pd.Timestamp(date.today())
    future = forecast_df[forecast_df["ds"] > today].copy().sort_values("ds").reset_index(drop=True)

    if future.empty:
        logger.warning("No future rows in forecast — cannot detect death date")
        return None

    # Use upper confidence bound for a more conservative (fewer false positives) death call
    signal_col = "yhat_upper" if use_confidence_bound else "yhat"
    logger.info(f"Death detection using '{signal_col}' against threshold={threshold}")

    below_mask = future[signal_col] < threshold
    consecutive = 0
    for i, is_below in enumerate(below_mask):
        if is_below:
            consecutive += 1
            if consecutive >= DEATH_CONSECUTIVE_WEEKS:
                death_idx = i - DEATH_CONSECUTIVE_WEEKS + 1
                death_ts = future.iloc[death_idx]["ds"]
                death_date = death_ts.date() if hasattr(death_ts, "date") else date.fromisoformat(str(death_ts)[:10])
                logger.info(f"Death date detected: {death_date} (sustained {DEATH_CONSECUTIVE_WEEKS}w below {threshold})")
                return death_date
        else:
            consecutive = 0

    logger.info(f"Trend stays above {threshold} for the full forecast window — no death detected")
    return None


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
