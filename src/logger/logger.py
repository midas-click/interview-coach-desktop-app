"""Structured logging setup.

Console output is human-readable. File output is JSON lines for
machine parsing / log aggregation.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.settings import Settings


class _JsonFormatter(logging.Formatter):
    """Format log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            entry["exc"] = str(record.exc_info[1])
        return json.dumps(entry)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def setup_logging(settings: Settings) -> None:
    """Initialise the logging system.

    - Console handler: human-readable, colour-free (safe for all terminals)
    - File handler: JSON lines for structured log analysis
    """
    root = logging.getLogger()
    root.setLevel(settings.log_level)
    root.handlers.clear()

    # -- console --
    console_fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(console_fmt)
    root.addHandler(console)

    # -- file --
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setFormatter(_JsonFormatter())
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name."""
    return logging.getLogger(name)
