import time
import pandas as pd
from pytrends.request import TrendReq
from utils.logger import get_logger

logger = get_logger(__name__)

_pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25), retries=2, backoff_factor=0.5)


def fetch_interest_over_time(keyword: str, timeframe: str = "today 5-y") -> pd.DataFrame:
    """
    Fetch weekly interest-over-time data from Google Trends for a single keyword.

    Uses 5 years of history (weekly granularity) for better Prophet accuracy.
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

    # Resample to weekly means to reduce noise (pytrends may return daily or weekly
    # depending on the timeframe — normalise to weekly regardless)
    df = df.resample("W").mean().round(1)

    logger.info(f"Fetched {len(df)} weekly rows for '{keyword}' (range: {df.index.min().date()} → {df.index.max().date()})")
    return df


_FALLBACK_TOPICS = [
    "ChatGPT", "Bitcoin", "Tesla", "TikTok", "NFT",
    "Wordle", "Metaverse", "AI", "Netflix", "Dogecoin",
    "Instagram", "Elon Musk", "OpenAI", "Climate Change", "Crypto",
    "YouTube Shorts", "Apple Vision Pro", "Threads", "Midjourney", "Sora",
]


def fetch_trending_now(geo: str = "US") -> list[str]:
    """
    Return trending search topics. Tries pytrends realtime_trending_searches first,
    then today_searches, then falls back to a curated static list.
    """
    # Attempt 1: realtime trending (newer endpoint)
    try:
        df = _pytrends.realtime_trending_searches(pn=geo)
        topics = df["title"].dropna().tolist()
        if topics:
            logger.info(f"Fetched {len(topics)} realtime trending topics")
            return topics[:20]
    except Exception as exc:
        logger.warning(f"realtime_trending_searches failed: {exc}")

    # Attempt 2: today_searches
    try:
        df = _pytrends.today_searches(pn=geo)
        topics = df.tolist()
        if topics:
            logger.info(f"Fetched {len(topics)} today_searches topics")
            return topics[:20]
    except Exception as exc:
        logger.warning(f"today_searches failed: {exc}")

    # Fallback: static curated list
    logger.warning("Using fallback topic list — live trending unavailable")
    return _FALLBACK_TOPICS
