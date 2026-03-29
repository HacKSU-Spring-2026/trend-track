import sys
import logging
from loguru import logger as _loguru_logger

# Remove default loguru handler
_loguru_logger.remove()

# Add a clean stderr handler
_loguru_logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# Silence noisy third-party loggers
for noisy in ("cmdstanpy", "prophet", "botocore", "boto3", "urllib3", "pytrends"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str):
    """Return a loguru logger bound to the given module name."""
    return _loguru_logger.bind(name=name)
