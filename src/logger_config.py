"""Structured logging setup, shared by every module."""

import logging
import os

from config import DEFAULT_OUTPUT_DIR, LOG_FILE_NAME


def setup_logger(name: str, output_dir: str = DEFAULT_OUTPUT_DIR) -> logging.Logger:
    """Create (or fetch) a logger that writes to both console and a run.log file.

    Args:
        name: logger name, typically __name__ of the calling module.
        output_dir: directory where run.log will be written.

    Returns:
        A configured logging.Logger instance.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(name)

    if logger.handlers:
        # Already configured (e.g. re-imported); avoid duplicate handlers.
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(os.path.join(output_dir, LOG_FILE_NAME))
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
