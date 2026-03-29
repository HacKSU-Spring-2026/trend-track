import os
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd
from pymongo import MongoClient, UpdateOne
from pymongo.errors import ConnectionFailure, OperationFailure
from dotenv import load_dotenv

from utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

_client: Optional[MongoClient] = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection

    uri = os.getenv("MONGO_URI", "")
    db_name = os.getenv("MONGO_DB_NAME", "trendlife")

    if not uri:
        logger.warning("MONGO_URI not set — MongoDB persistence disabled.")
        return None

    try:
        _client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        _client.admin.command("ping")
        db = _client[db_name]
        _collection = db["trend_searches"]
        _collection.create_index("topic", unique=True)
        logger.info(f"Connected to MongoDB database '{db_name}'")
    except (ConnectionFailure, OperationFailure) as exc:
        logger.error(f"MongoDB connection failed: {exc}")
        _collection = None

    return _collection


def get_cached_result(topic: str) -> Optional[dict]:
    """Return the cached document for a topic, or None if not found."""
    col = _get_collection()
    if col is None:
        return None
    doc = col.find_one({"topic": topic.strip().lower()}, {"_id": 0})
    if doc:
        logger.info(f"Cache hit for '{topic}'")
    return doc


def save_result(topic: str, payload: dict) -> None:
    """
    Upsert the result payload for a topic into MongoDB.

    payload keys expected:
      raw_data       → list[dict] with {date, interest}
      peak_score     → int
      peak_date      → str (ISO date)
      current_score  → int
      trend_phase    → str
      predicted_death→ str | None (ISO date)
      days_remaining → int | None
      forecast_json  → str (JSON-serialised forecast DataFrame)
    """
    col = _get_collection()
    if col is None:
        return

    doc = {
        "topic": topic,
        "searched_at": datetime.now(tz=timezone.utc),
        **payload,
    }

    try:
        col.update_one({"topic": topic}, {"$set": doc}, upsert=True)
        logger.info(f"Saved result for '{topic}' to MongoDB")
    except Exception as exc:
        logger.error(f"Failed to save to MongoDB: {exc}")


def is_stale(document: dict, max_age_hours: int = 24) -> bool:
    """Return True if the cached document is older than max_age_hours."""
    searched_at = document.get("searched_at")
    if searched_at is None:
        return True
    if searched_at.tzinfo is None:
        searched_at = searched_at.replace(tzinfo=timezone.utc)
    age = datetime.now(tz=timezone.utc) - searched_at
    return age > timedelta(hours=max_age_hours)


def get_recent_searches(limit: int = 10) -> list[dict]:
    """Return the last `limit` searches, sorted by most recent."""
    col = _get_collection()
    if col is None:
        return []
    try:
        docs = list(
            col.find({}, {"_id": 0, "topic": 1, "searched_at": 1, "trend_phase": 1, "predicted_death": 1, "current_score": 1})
            .sort("searched_at", -1)
            .limit(limit)
        )
        return docs
    except Exception as exc:
        logger.error(f"Failed to fetch recent searches: {exc}")
        return []


def save_comparison(kw_a: str, kw_b: str) -> None:
    """Upsert a comparison pair, updating the timestamp on re-compare."""
    col = _get_collection()
    if col is None:
        return
    try:
        db = col.database
        # Treat (A vs B) and (B vs A) as the same pair by sorting
        pair = sorted([kw_a, kw_b])
        db["comparisons"].update_one(
            {"pair": pair},
            {"$set": {"kw_a": pair[0], "kw_b": pair[1], "compared_at": datetime.now(tz=timezone.utc), "pair": pair}},
            upsert=True,
        )
        logger.info(f"Saved comparison '{kw_a}' vs '{kw_b}'")
    except Exception as exc:
        logger.error(f"Failed to save comparison: {exc}")


def get_recent_comparisons(limit: int = 10) -> list[dict]:
    """Return the last `limit` comparisons, sorted by most recent."""
    col = _get_collection()
    if col is None:
        return []
    try:
        db = col.database
        docs = list(
            db["comparisons"].find({}, {"_id": 0})
            .sort("compared_at", -1)
            .limit(limit)
        )
        return docs
    except Exception as exc:
        logger.error(f"Failed to fetch recent comparisons: {exc}")
        return []


def save_ai_report(topic: str, report: str) -> None:
    """Persist the AI-generated report text for a topic."""
    col = _get_collection()
    if col is None:
        return
    try:
        col.update_one({"topic": topic}, {"$set": {"ai_report": report}})
        logger.info(f"Saved AI report for '{topic}'")
    except Exception as exc:
        logger.error(f"Failed to save AI report: {exc}")


def save_comparison_report(kw_a: str, kw_b: str, report: str) -> None:
    """Persist the AI comparison report for a sorted pair."""
    col = _get_collection()
    if col is None:
        return
    try:
        db = col.database
        pair = sorted([kw_a, kw_b])
        db["comparisons"].update_one(
            {"pair": pair},
            {"$set": {"ai_report": report}},
        )
        logger.info(f"Saved comparison report for '{kw_a}' vs '{kw_b}'")
    except Exception as exc:
        logger.error(f"Failed to save comparison report: {exc}")


def get_comparison_report(kw_a: str, kw_b: str) -> Optional[str]:
    """Return the stored AI comparison report for a pair, or None."""
    col = _get_collection()
    if col is None:
        return None
    try:
        db = col.database
        pair = sorted([kw_a, kw_b])
        doc = db["comparisons"].find_one({"pair": pair}, {"ai_report": 1, "_id": 0})
        return doc.get("ai_report") if doc else None
    except Exception as exc:
        logger.error(f"Failed to fetch comparison report: {exc}")
        return None
