"""Logger configuration."""

from __future__ import annotations

import sys

from loguru import logger

from src.utils.config import PROJECT_ROOT

def setup_logger(log_level: str = "INFO", log_file: bool = True) -> None:
    """Configure the global logger."""
    # Remove the default handler
    logger.remove()

    # Console output (coloured)
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File output
    if log_file:
        log_path = PROJECT_ROOT / "experiments" / "logs" / "skillforge_{time:YYYY-MM-DD}.log"
        logger.add(
            str(log_path),
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="30 days",
            encoding="utf-8",
        )

    logger.info(f"Logger initialised: level={log_level}")
