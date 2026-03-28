from datetime import date
from typing import Optional

import pandas as pd

from data import fetch, mongo, snowflake_client, s3_client
from model import prophet_model, trend_phase, death_detector
from utils.logger import get_logger

logger = get_logger(__name__)


def get_or_fetch(keyword: str, force_refresh: bool = False) -> dict:
    """
    Main orchestrator. Returns a result dict for the given keyword.

    1. Check MongoDB cache — return immediately if fresh and force_refresh is False.
    2. Otherwise run the full pipeline:
       fetch → prophet → classify phase → detect death
    3. Persist results to MongoDB, S3, and Snowflake.
    4. Return the result dict.

    Result dict keys:
      topic            str
      raw_data         list[dict]  — [{date, interest}, ...]
      peak_score       int
      peak_date        str
      current_score    int
      trend_phase      str
      predicted_death  str | None  — ISO date
      days_remaining   int | None
      confidence_upper float
      confidence_lower float
      forecast_df      pd.DataFrame  — full prophet output (not persisted to Mongo)
    """
    # --- Cache check ---
    if not force_refresh:
        cached = mongo.get_cached_result(keyword)
        if cached and not mongo.is_stale(cached):
            logger.info(f"Returning fresh cache for '{keyword}'")
            # forecast_df is not stored in Mongo; flag its absence
            cached["forecast_df"] = None
            return cached

    # --- Fetch from Google Trends ---
    logger.info(f"Running full pipeline for '{keyword}'")
    df = fetch.fetch_interest_over_time(keyword)

    # --- Fit Prophet & forecast 90 days ---
    forecast_df = prophet_model.fit_and_forecast(df, periods=90)

    # --- Classify phase ---
    phase = trend_phase.classify_phase(forecast_df, date.today())

    # --- Detect death date ---
    death_date = death_detector.find_death_date(forecast_df)
    days_left = death_detector.days_until_death(death_date)
    conf_upper, conf_lower = death_detector.get_confidence_at_death(forecast_df, death_date)

    # --- Build raw_data list ---
    raw_data = [
        {"date": idx.strftime("%Y-%m-%d"), "interest": int(row["interest"])}
        for idx, row in df.iterrows()
    ]

    peak_idx = df["interest"].idxmax()
    current_score = int(df["interest"].iloc[-1])

    result = {
        "topic": keyword,
        "raw_data": raw_data,
        "peak_score": int(df["interest"].max()),
        "peak_date": peak_idx.strftime("%Y-%m-%d"),
        "current_score": current_score,
        "trend_phase": phase.value,
        "predicted_death": death_date.isoformat() if death_date else None,
        "days_remaining": days_left,
        "confidence_upper": round(conf_upper, 2),
        "confidence_lower": round(conf_lower, 2),
        "forecast_df": forecast_df,  # in-memory only
    }

    # --- Persist (non-blocking failures) ---
    _persist(keyword, df, forecast_df, result, phase, death_date, days_left, conf_upper, conf_lower)

    return result


def _persist(
    keyword: str,
    raw_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    result: dict,
    phase: trend_phase.TrendPhase,
    death_date: Optional[date],
    days_left: Optional[int],
    conf_upper: float,
    conf_lower: float,
) -> None:
    """Fire-and-forget persistence to all three stores. Errors are logged, not raised."""

    # MongoDB
    mongo_payload = {k: v for k, v in result.items() if k != "forecast_df"}
    mongo.save_result(keyword, mongo_payload)

    # S3 — raw data
    s3_client.upload_dataframe(
        raw_df,
        s3_client.build_s3_key(keyword, "raw"),
    )

    # S3 — forecast
    s3_client.upload_dataframe(
        forecast_df,
        s3_client.build_s3_key(keyword, "predictions"),
    )

    # Snowflake
    snowflake_client.write_trend_scores(keyword, raw_df)
    snowflake_client.write_prediction(
        topic=keyword,
        predicted_death=death_date.isoformat() if death_date else None,
        days_remaining=days_left,
        trend_phase=phase.value,
        confidence_upper=conf_upper,
        confidence_lower=conf_lower,
    )
