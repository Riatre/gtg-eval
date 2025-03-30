from loguru import logger
import os


def setup_logging(level="INFO"):
    """Configure loguru logger with the specified level.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Add a JSON file handler for structured logging
    logger.add(
        "logs/gtg_eval.jsonl",
        serialize=True,  # Output as JSON
        rotation="10 MB",
        level=level,
    )

    return logger
