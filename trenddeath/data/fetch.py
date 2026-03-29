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
    # AI & Tech
    "ChatGPT", "OpenAI", "Gemini AI", "Claude AI", "Midjourney", "Sora",
    "AI", "Generative AI", "LLM", "Prompt Engineering",
    "Apple Vision Pro", "Meta Quest", "Rabbit R1", "Humane AI Pin",
    # Social & Apps
    "TikTok", "Threads", "BeReal", "Bluesky", "Mastodon",
    "Instagram", "YouTube Shorts", "Twitch", "Discord", "Clubhouse",
    # Crypto & Finance
    "Bitcoin", "Ethereum", "Dogecoin", "NFT", "Crypto",
    "GameStop", "meme stocks", "SPAC", "Web3", "DeFi",
    # Pop culture & viral
    "Wordle", "Squid Game", "Wednesday Addams", "Barbie movie",
    "Taylor Swift", "Elon Musk", "Andrew Tate", "MrBeast",
    # Tech products & companies
    "Tesla", "Netflix", "Metaverse", "Neuralink", "SpaceX",
    "Vision Pro", "Notion", "Figma", "Perplexity AI",
    # Trends & movements
    "Climate Change", "quiet quitting", "remote work", "hustle culture",
    "plant-based meat", "electric vehicles", "Buy Now Pay Later",
]


def fetch_trending_now(geo: str = "US") -> list[str]:
    """Return the curated static topic list for the suggestions dropdown."""
    return _FALLBACK_TOPICS
