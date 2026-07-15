"""Per-run logging: every run writes to logs/<config_id>.log AND stdout."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_run_logger(
    config_id: str, logs_dir: str | Path, level: int = logging.INFO
) -> logging.Logger:
    """Return a logger for this run, writing to logs/<config_id>.log and stdout.

    Idempotent: calling this again for the same config_id in the same process
    returns the same logger without adding duplicate handlers.
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"lcmunet.run.{config_id}")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(_FORMAT)

    file_handler = logging.FileHandler(logs_dir / f"{config_id}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
