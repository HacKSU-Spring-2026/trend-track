import logging
import warnings

import pandas as pd

from utils.logger import get_logger

# Silence Prophet / Stan output before importing
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*Importing plotly failed.*")
warnings.filterwarnings("ignore", category=FutureWarning)

from prophet import Prophet  # noqa: E402

logger = get_logger(__name__)


def _auto_changepoint_scale(y_series: pd.Series) -> float:
    """
    Auto-tune changepoint_prior_scale based on coefficient of variation.

    High volatility (spiky trends like memes) → higher scale = more flexible fit.
    Low volatility (slow evergreen trends) → lower scale = smoother fit.
    """
    import numpy as np
    mean = y_series.mean()
    if mean == 0:
        return 0.05
    cv = y_series.std() / mean
    if cv > 1.2:
        return 0.3    # very spiky
    elif cv > 0.8:
        return 0.15   # moderately volatile
    elif cv > 0.4:
        return 0.08   # mild variation
    else:
        return 0.03   # stable / evergreen


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
    logger.info(f"Fitting Prophet model on {len(df)} rows, forecasting {periods} days ahead")

    # Prophet requires columns named 'ds' and 'y'
    prophet_df = df.reset_index().rename(columns={"date": "ds", "interest": "y"})
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

    # Auto-tune flexibility based on how volatile the trend is
    cp_scale = _auto_changepoint_scale(prophet_df["y"])
    logger.info(f"Auto-tuned changepoint_prior_scale={cp_scale} (cv={prophet_df['y'].std() / max(prophet_df['y'].mean(), 1):.2f})")

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,   # weekly data — disable weekly seasonality to avoid overfitting
        daily_seasonality=False,
        interval_width=0.80,
        changepoint_prior_scale=cp_scale,
    )

    # Suppress Stan output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(prophet_df, iter=500)  # more iterations for better convergence on 5y data

    # Forecast in weekly steps to match input frequency
    future = model.make_future_dataframe(periods=periods // 7, freq="W")
    forecast = model.predict(future)

    # Clip predictions to [0, 100]
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[col] = forecast[col].clip(0, 100)

    # Merge original 'y' values back in
    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].merge(
        prophet_df[["ds", "y"]], on="ds", how="left"
    )

    logger.info(f"Forecast complete. Last predicted date: {result['ds'].max().date()}")
    return result
