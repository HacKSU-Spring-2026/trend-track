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


def fit_and_forecast(df: pd.DataFrame, periods: int = 90) -> pd.DataFrame:
    """
    Fit a Prophet model on historical interest data and produce a forecast.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex named 'date' and an 'interest' column (0–100).
    periods : int
        Number of future days to forecast (default: 90).

    Returns
    -------
    pd.DataFrame
        Columns: ds, y (historical, NaN for future), yhat, yhat_lower, yhat_upper
    """
    logger.info(f"Fitting Prophet model on {len(df)} rows, forecasting {periods} days ahead")

    # Prophet requires columns named 'ds' and 'y'
    prophet_df = df.reset_index().rename(columns={"date": "ds", "interest": "y"})
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.80,
        changepoint_prior_scale=0.05,
    )

    # Suppress Stan output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(prophet_df, iter=300)

    future = model.make_future_dataframe(periods=periods, freq="D")
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
