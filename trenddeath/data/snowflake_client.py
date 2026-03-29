import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

_conn = None


def _get_connection():
    global _conn
    if _conn is not None:
        try:
            _conn.cursor().execute("SELECT 1")
            return _conn
        except Exception:
            _conn = None

    account = os.getenv("SNOWFLAKE_ACCOUNT", "")
    if not account:
        logger.warning("SNOWFLAKE_ACCOUNT not set — Snowflake persistence disabled.")
        return None

    try:
        import snowflake.connector

        _conn = snowflake.connector.connect(
            account=account,
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            database=os.getenv("SNOWFLAKE_DATABASE", "TRENDDEATH"),
            schema=os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        )
        _ensure_tables(_conn)
        logger.info("Connected to Snowflake")
    except Exception as exc:
        logger.error(f"Snowflake connection failed: {exc}")
        _conn = None

    return _conn


def _ensure_tables(conn) -> None:
    ddl = [
        """
        CREATE TABLE IF NOT EXISTS trend_scores (
            topic         VARCHAR,
            date          DATE,
            interest      NUMBER,
            inserted_at   TIMESTAMP_TZ DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (topic, date)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS trend_predictions (
            topic              VARCHAR,
            predicted_death    DATE,
            days_remaining     NUMBER,
            trend_phase        VARCHAR,
            confidence_upper   FLOAT,
            confidence_lower   FLOAT,
            created_at         TIMESTAMP_TZ DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (topic, created_at)
        )
        """,
    ]
    cur = conn.cursor()
    for stmt in ddl:
        cur.execute(stmt)
    cur.close()


def write_trend_scores(topic: str, df: pd.DataFrame) -> None:
    """
    Write raw interest-over-time rows into trend_scores.
    df must have a DatetimeIndex ('date') and an 'interest' column.
    """
    conn = _get_connection()
    if conn is None:
        return

    rows = [(topic, idx.date(), int(row["interest"])) for idx, row in df.iterrows()]
    if not rows:
        return

    try:
        cur = conn.cursor()
        cur.executemany(
            """
            MERGE INTO trend_scores AS t
            USING (SELECT %s AS topic, %s::DATE AS date, %s AS interest) AS s
            ON t.topic = s.topic AND t.date = s.date
            WHEN NOT MATCHED THEN INSERT (topic, date, interest) VALUES (s.topic, s.date, s.interest)
            """,
            rows,
        )
        cur.close()
        logger.info(f"Wrote {len(rows)} rows to Snowflake trend_scores for '{topic}'")
    except Exception as exc:
        logger.error(f"Failed to write trend scores to Snowflake: {exc}")


def write_prediction(
    topic: str,
    predicted_death: Optional[str],
    days_remaining: Optional[int],
    trend_phase: str,
    confidence_upper: float,
    confidence_lower: float,
) -> None:
    """Insert a single prediction row into trend_predictions."""
    conn = _get_connection()
    if conn is None:
        return

    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO trend_predictions
                (topic, predicted_death, days_remaining, trend_phase, confidence_upper, confidence_lower)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (topic, predicted_death, days_remaining, trend_phase, confidence_upper, confidence_lower),
        )
        cur.close()
        logger.info(f"Wrote prediction to Snowflake for '{topic}'")
    except Exception as exc:
        logger.error(f"Failed to write prediction to Snowflake: {exc}")
