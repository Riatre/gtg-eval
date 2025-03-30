from loguru import logger
import os
import sys


def setup_logging(level="INFO"):
    """Configure loguru logger with the specified level.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logger.remove()
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Add a JSON file handler for structured logging
    logger.add(
        "logs/gtg_eval.jsonl",
        serialize=True,  # Output as JSON
        rotation="10 MB",
        level=level,
    )

    logger_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level> | "
        "{extra}"
    )
    logger.add(sys.stderr, format=logger_format, level=level)
    return logger
