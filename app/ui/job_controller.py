"""Runs comparisons in a worker process without blocking the UI (Architecture §5).

The comparison is OpenCV / PyMuPDF work that takes seconds and, on a malformed
file, can crash the native code underneath. So it never runs in the UI process:

* **A separate process.** If it dies, only it does; the UI reports a clean error
  and the next job gets a fresh worker.
* **Cancellation kills the worker** and starts another. OpenCV calls cannot be
  interrupted politely, so the process is terminated rather than asked to stop.
* **Recycled after N jobs.** Native libraries hold memory Python's collector never
  sees; a fresh process every N jobs bounds that (NFR-2).
* **One job at a time** (backpressure): the UI disables Compare while one runs.
* **Started early.** Importing OpenCV, scikit-image and PyMuPDF takes a couple of
  seconds, so the worker is started when the app opens and again right after each
  cancel or recycle, and a comparison does not wait for it.

The result is a signal, so the UI thread only ever handles finished results.
"""

from __future__ import annotations

import logging
import multiprocessing
import threading
from typing import Callable
from uuid import uuid4

from PySide6.QtCore import QObject, QTimer, Signal

from app.config import RuntimeConfig
from app.engine.worker_process import ComparisonRequest, ErrorKind, worker_main

logger = logging.getLogger(__name__)

_JOIN_SECONDS = 3.0  # how long to wait for a worker to exit before killing it outright


class JobController(QObject):
    progressed = Signal(str)  # what the running job is doing
    completed = Signal(object)  # ComparisonOutcome
    failed = Signal(str, str)  # (ErrorKind, message for a person)
    cancelled = Signal()
    busyChanged = Signal(bool)

    # Messages arrive on a reader thread and are handed to the UI thread through this.
    _incoming = Signal(int, object)

    def __init__(
        self,
        runtime: RuntimeConfig | None = None,
        target: Callable = worker_main,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._runtime = runtime or RuntimeConfig()
        self._target = target
        self._context = multiprocessing.get_context("spawn")

        self._process = None
        self._connection = None
        self._generation = 0  # bumps whenever a worker is dropped, so its late messages are ignored
        self._worker_ready = False
        self._jobs_run = 0  # by the current worker

        self._job_id: str | None = None
        self._queued: tuple[str, ComparisonRequest] | None = None  # waiting for the worker to be ready

        self._timeout = QTimer(self)
        self._timeout.setSingleShot(True)
        self._timeout.timeout.connect(self._on_timeout)
        self._incoming.connect(self._on_incoming)

    # -- state ---------------------------------------------------------------
    @property
    def is_busy(self) -> bool:
        return self._job_id is not None

    @property
    def worker_pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    @property
    def jobs_run_by_worker(self) -> int:
        return self._jobs_run

    # -- running jobs ----------------------------------------------------------
    def start(self, request: ComparisonRequest) -> str:
        """Begin a comparison; results arrive as signals. Returns the job's id."""
        if self.is_busy:
            raise RuntimeError("a comparison is already running")
        self._job_id = uuid4().hex[:8]
        self.busyChanged.emit(True)
        self._timeout.start(self._runtime.job_timeout_s * 1000)

        self._ensure_worker()
        if self._worker_ready:
            self._send_job(self._job_id, request)
        else:
            self._queued = (self._job_id, request)
            self.progressed.emit("Starting the comparison engine…")
        return self._job_id

    def cancel(self) -> None:
        """Stop the running comparison. The worker is killed and replaced."""
        if not self.is_busy:
            return
        logger.info("comparison cancelled", extra={"job_id": self._job_id})
        self._drop_worker(kill=True)
        self._end_job()
        self.cancelled.emit()
        self.warm_up()

    def warm_up(self) -> None:
        """Start a worker now if there is none, so the next job does not wait for imports."""
        self._ensure_worker()

    def shutdown(self) -> None:
        """Stop the worker for good (the application is closing)."""
        busy = self.is_busy  # a worker mid-job will not hear "stop", so do not wait for it
        self._timeout.stop()
        self._job_id = None
        self._queued = None
        self._drop_worker(kill=busy)

    # -- the worker process -----------------------------------------------------
    def _ensure_worker(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._drop_worker(kill=True)  # clear a dead one's leftovers
        parent_end, child_end = self._context.Pipe()
        process = self._context.Process(target=self._target, args=(child_end,), daemon=True)
        process.start()
        child_end.close()  # the parent must not hold the child's end, or a dead worker never reads as EOF

        self._generation += 1
        self._process, self._connection = process, parent_end
        self._worker_ready = False
        self._jobs_run = 0
        threading.Thread(
            target=self._read, args=(parent_end, self._generation), name="worker-reader", daemon=True
        ).start()
        logger.info("comparison worker started", extra={"pid": process.pid})

    def _read(self, connection, generation: int) -> None:
        """Reader thread: forward every message from one worker to the UI thread."""
        try:
            while True:
                self._incoming.emit(generation, connection.recv())
            # (unreachable: recv() raises when the worker's end closes)
        except (EOFError, OSError):
            try:
                self._incoming.emit(generation, ("dead",))  # the worker exited or was killed
            except RuntimeError:
                pass  # the controller was deleted first
        except RuntimeError:
            pass  # the controller was deleted while a message was in flight

    def _drop_worker(self, kill: bool) -> None:
        """Forget the current worker, ending its process."""
        self._generation += 1  # anything it still says is stale
        process, connection = self._process, self._connection
        self._process = self._connection = None
        self._worker_ready = False
        if process is None:
            return
        try:
            if not kill and process.is_alive() and connection is not None:
                try:
                    connection.send(("stop",))
                except (OSError, ValueError):
                    pass
                process.join(_JOIN_SECONDS)
            if process.is_alive():
                process.terminate()
                process.join(_JOIN_SECONDS)
            if process.is_alive():
                process.kill()
                process.join(_JOIN_SECONDS)
        finally:
            if connection is not None:
                connection.close()

    def _send_job(self, job_id: str, request: ComparisonRequest) -> None:
        try:
            self._connection.send(("job", job_id, request))
        except (OSError, ValueError):
            self._on_worker_lost()

    # -- messages ---------------------------------------------------------------
    def _on_incoming(self, generation: int, message: tuple) -> None:
        if generation != self._generation:
            return  # from a worker that has since been dropped
        kind = message[0]

        if kind == "ready":
            self._worker_ready = True
            if self._queued is not None:
                job_id, request = self._queued
                self._queued = None
                self._send_job(job_id, request)
        elif kind == "progress" and message[1] == self._job_id:
            self.progressed.emit(message[2])
        elif kind == "result" and message[1] == self._job_id:
            self._jobs_run += 1
            self._end_job()
            self.completed.emit(message[2])
            self._recycle_if_due()
        elif kind == "error" and message[1] == self._job_id:
            self._jobs_run += 1
            self._end_job()
            self.failed.emit(message[2], message[3])
            self._recycle_if_due()
        elif kind == "dead":
            self._on_worker_lost()

    def _on_worker_lost(self) -> None:
        """The worker died without being told to: a native-library fault, usually a bad file."""
        was_running = self.is_busy
        logger.error("comparison worker died", extra={"job_id": self._job_id})
        self._drop_worker(kill=True)
        if was_running:
            self._end_job()
            self.failed.emit(
                ErrorKind.CRASH,
                "The comparison engine stopped unexpectedly, which usually means one of the files is "
                "damaged in a way the application could not read. The application itself is fine; "
                "check the files (re-export them if you can) and try again.",
            )
            self.warm_up()

    def _on_timeout(self) -> None:
        if not self.is_busy:
            return
        logger.error("comparison timed out", extra={"job_id": self._job_id})
        self._drop_worker(kill=True)
        self._end_job()
        self.failed.emit(
            ErrorKind.TIMEOUT,
            f"The comparison was stopped after {self._runtime.job_timeout_s // 60} minutes without finishing. "
            "The files may be too large or complex; try a smaller page size or resolution.",
        )
        self.warm_up()

    def _end_job(self) -> None:
        self._timeout.stop()
        self._job_id = None
        self._queued = None
        self.busyChanged.emit(False)

    def _recycle_if_due(self) -> None:
        """A fresh worker every N jobs, so native-library memory cannot creep (NFR-2)."""
        if self._jobs_run >= self._runtime.worker_recycle_after_jobs:
            logger.info("recycling comparison worker", extra={"jobs_run": self._jobs_run})
            self._drop_worker(kill=False)
            self.warm_up()
