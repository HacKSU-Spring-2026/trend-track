import io
import os
from datetime import date
from typing import Optional

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

from utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

_s3 = None
_bucket: str = ""


def _get_client():
    global _s3, _bucket
    if _s3 is not None:
        return _s3

    key_id = os.getenv("AWS_ACCESS_KEY_ID", "")
    if not key_id:
        logger.warning("AWS_ACCESS_KEY_ID not set — S3 persistence disabled.")
        return None

    try:
        _s3 = boto3.client(
            "s3",
            aws_access_key_id=key_id,
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
        _bucket = os.getenv("AWS_BUCKET_NAME", "trenddeath-snapshots")
        logger.info(f"S3 client ready (bucket: {_bucket})")
    except Exception as exc:
        logger.error(f"Failed to create S3 client: {exc}")
        _s3 = None

    return _s3


def build_s3_key(topic: str, data_type: str, run_date: Optional[date] = None) -> str:
    """
    Build a consistent S3 object key.

    data_type: 'raw' | 'predictions'
    Example: raw/bitcoin/2024-03-15/data.parquet
    """
    run_date = run_date or date.today()
    safe_topic = topic.lower().replace(" ", "_")
    return f"{data_type}/{safe_topic}/{run_date.isoformat()}/data.parquet"


def upload_dataframe(df: pd.DataFrame, s3_key: str) -> bool:
    """Serialize df to Parquet in memory and upload to S3. Returns True on success."""
    client = _get_client()
    if client is None:
        return False

    try:
        buf = io.BytesIO()
        table = pa.Table.from_pandas(df, preserve_index=True)
        pq.write_table(table, buf)
        buf.seek(0)
        client.put_object(Bucket=_bucket, Key=s3_key, Body=buf.read())
        logger.info(f"Uploaded {s3_key} to s3://{_bucket}")
        return True
    except Exception as exc:
        logger.error(f"S3 upload failed for {s3_key}: {exc}")
        return False


def download_dataframe(s3_key: str) -> Optional[pd.DataFrame]:
    """Download a Parquet file from S3 and deserialize to DataFrame. Returns None on failure."""
    client = _get_client()
    if client is None:
        return None

    try:
        obj = client.get_object(Bucket=_bucket, Key=s3_key)
        buf = io.BytesIO(obj["Body"].read())
        table = pq.read_table(buf)
        df = table.to_pandas()
        logger.info(f"Downloaded {s3_key} from s3://{_bucket}")
        return df
    except Exception as exc:
        logger.error(f"S3 download failed for {s3_key}: {exc}")
        return None
