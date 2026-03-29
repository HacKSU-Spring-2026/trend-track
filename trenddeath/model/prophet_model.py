import logging
import warnings

import numpy as np
import pandas as pd

from utils.logger import get_logger

# Silence Prophet / Stan output before importing
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*Importing plotly failed.*")
warnings.filterwarnings("ignore", category=FutureWarning)

from prophet import Prophet  # noqa: E402

logger = get_logger(__name__)


def _choose_seasonality_mode(y_series: pd.Series) -> str:
    """
    Pick 'multiplicative' vs 'additive' seasonality based on whether
    seasonal swings scale with the trend level.

    Heuristic: split the series into two halves by time. If the ratio of
    standard deviations between the higher-mean half and the lower-mean half
    is large, seasonal amplitude scales with level → multiplicative.
    """
    half = len(y_series) // 2
    first, second = y_series.iloc[:half], y_series.iloc[half:]

    std_first = first.std()
    std_second = second.std()

    if std_first == 0 or std_second == 0:
        return "additive"

    # Ratio of the higher-variance half to the lower-variance half
    ratio = max(std_first, std_second) / min(std_first, std_second)

    # If one half has >1.8x the variance of the other, seasonality scales with level
    mode = "multiplicative" if ratio > 1.8 else "additive"
    logger.info(f"Seasonality mode: {mode} (variance ratio={ratio:.2f})")
    return mode


def _auto_changepoint_scale(y_series: pd.Series) -> float:
    """
    Quick heuristic fallback for changepoint_prior_scale based on
    coefficient of variation. Used when the dataset is too small for CV.
    """
    mean = y_series.mean()
    if mean == 0:
        return 0.05
    cv = y_series.std() / mean
    if cv > 1.2:
        return 0.3
    elif cv > 0.8:
        return 0.15
    elif cv > 0.4:
        return 0.08
    else:
        return 0.03



def fit_and_forecast(df: pd.DataFrame, periods: int = 365) -> pd.DataFrame:
    """
    Fit a Prophet model on historical interest data and produce a forecast.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex named 'date' and an 'interest' column (0–100).
        Expected to be weekly frequency (after resampling in fetch.py).
    periods : int
        Number of future weeks to forecast (default: 365 days ≈ 52 weeks).

    Returns
    -------
    pd.DataFrame
        Columns: ds, y (historical, NaN for future), yhat, yhat_lower, yhat_upper
    """
    forecast_weeks = max(periods // 7, 156)
    logger.info(f"Fitting Prophet model on {len(df)} rows, forecasting {forecast_weeks} weeks ahead")

    prophet_df = df.reset_index().rename(columns={"date": "ds", "interest": "y"})
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

    # Auto-select seasonality mode based on variance structure
    seas_mode = _choose_seasonality_mode(prophet_df["y"])

    # Heuristic changepoint scale based on coefficient of variation
    cp_scale = _auto_changepoint_scale(prophet_df["y"])
    logger.info(f"Using changepoint_prior_scale={cp_scale} (heuristic)")

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.80,
        changepoint_prior_scale=cp_scale,
        seasonality_mode=seas_mode,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(prophet_df, iter=300)

    future = model.make_future_dataframe(periods=forecast_weeks, freq="W")
    forecast = model.predict(future)

    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[col] = forecast[col].clip(0, 100)

    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].merge(
        prophet_df[["ds", "y"]], on="ds", how="left"
    )

    logger.info(f"Forecast complete. Last predicted date: {result['ds'].max().date()}")
    return result
