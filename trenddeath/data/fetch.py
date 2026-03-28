import time
import pandas as pd
from pytrends.request import TrendReq
from utils.logger import get_logger

logger = get_logger(__name__)

_pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25), retries=2, backoff_factor=0.5)


def fetch_interest_over_time(keyword: str, timeframe: str = "today 12-m") -> pd.DataFrame:
    """
    Fetch daily interest-over-time data from Google Trends for a single keyword.

    Returns a DataFrame with a DatetimeIndex and a single 'interest' column (0–100).
    Raises ValueError if pytrends returns no data for the keyword.
    """
    logger.info(f"Fetching Google Trends data for '{keyword}' ({timeframe})")

    _pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo="", gprop="")
    df = _pytrends.interest_over_time()

    if df is None or df.empty:
        raise ValueError(f"No Google Trends data found for '{keyword}'. The topic may be too obscure or misspelled.")

    # Drop the isPartial flag column
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])

    # Rename keyword column to generic 'interest'
    df = df.rename(columns={keyword: "interest"})
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    logger.info(f"Fetched {len(df)} rows for '{keyword}' (range: {df.index.min().date()} → {df.index.max().date()})")
    return df


def fetch_trending_now(geo: str = "US") -> list[str]:
    """
    Return the current top 20 trending search topics for the given region.
    Falls back to an empty list if the API call fails.
    """
    try:
        trending = _pytrends.trending_searches(pn=geo.lower())
        topics = trending[0].tolist()
        logger.info(f"Fetched {len(topics)} trending topics for geo={geo}")
        return topics[:20]
    except Exception as exc:
        logger.warning(f"Could not fetch trending topics: {exc}")
        return []
