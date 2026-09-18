"""Structured logging setup (Architecture §7 / §9).

Two concerns are handled here:

1. App-wide logging: a rotating file (JSON lines, machine-parseable) plus a
   readable console stream, configured once at startup via
   `configure_logging()`.
2. Per-job logging: `job_log_file(job_id)` attaches a dedicated handler so
   every comparison job's pipeline-stage messages land in their own log
   file under `logs/jobs/`, satisfying "one log per comparison job."

`StageTimer` is the instrumentation hook later phases use to record each
pipeline stage's wall-clock duration and peak memory delta
(`tracemalloc`, stdlib — avoids adding a `psutil` dependency for this
alone).
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import tracemalloc
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Iterator

from app import paths

LOG_DIR = paths.log_dir()
JOB_LOG_DIR = LOG_DIR / "jobs"

_STANDARD_LOG_RECORD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "asctime",  # set on the shared record by any handler whose format string uses %(asctime)s
}


class JsonLineFormatter(logging.Formatter):
    """One JSON object per log line, including any `extra=` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_LOG_RECORD_ATTRS
        }
        if extras:
            payload.update(extras)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger. Call once, at application startup."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    JOB_LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s", "%H:%M:%S")
    )
    root.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "app.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(JsonLineFormatter())
    root.addHandler(file_handler)

    if not tracemalloc.is_tracing():
        tracemalloc.start()


@contextmanager
def job_log_file(job_id: str, level: int = logging.INFO) -> Iterator[Path]:
    """Attach a per-job file handler to the root logger for the duration of the `with` block."""
    JOB_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = JOB_LOG_DIR / f"{job_id}.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(JsonLineFormatter())
    handler.setLevel(level)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield path
    finally:
        root.removeHandler(handler)
        handler.close()


class StageTimer:
    """Context manager logging a pipeline stage's duration and peak memory delta.

    Usage:
        with StageTimer(logger, "alignment"):
            ...do the work...
    """

    def __init__(self, logger: logging.Logger, stage: str):
        self.logger = logger
        self.stage = stage

    def __enter__(self) -> "StageTimer":
        self._start = perf_counter()
        self._mem_start = tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else 0
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        duration_ms = (perf_counter() - self._start) * 1000
        mem_end = tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else 0
        self.logger.info(
            "stage '%s' %s",
            self.stage,
            "failed" if exc_type else "completed",
            extra={
                "stage": self.stage,
                "duration_ms": round(duration_ms, 2),
                "memory_delta_bytes": mem_end - self._mem_start,
                "ok": exc_type is None,
            },
        )
