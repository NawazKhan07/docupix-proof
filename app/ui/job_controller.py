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

The result is a signal, so the UI thread only ever handles finished results. And the UI thread never writes to the
worker either: a pipe takes only so much before the writer waits for the reader, and a worker deep in a long stage reads
nothing until the stage ends -- a request written from the UI thread then held the window ("Not Responding") for as long
as the stage took. Everything for the worker is handed to a writer thread of its own (`_Writer`), which does the waiting.
"""

from __future__ import annotations

import logging
import multiprocessing
import queue
import threading
from typing import Callable
from uuid import uuid4

from PySide6.QtCore import QObject, QTimer, Signal

from app.config import RuntimeConfig
from app.engine.worker_process import ComparisonRequest, ErrorKind, worker_main

logger = logging.getLogger(__name__)

_JOIN_SECONDS = 3.0  # how long to wait for a worker to exit before killing it outright


class _Writer:
    """Writes messages to one worker, in order, on a thread of its own (see the module): `put` never waits. A send that
    fails (the worker gone) calls `lost`, once; the reader notices the same and the controller handles it.

    A preparation already overtaken by a newer one waiting behind it is not sent at all -- the worker would only have
    dropped it. (One followed by a job is sent: the job may be the same comparison, and take over the preparation.)"""

    def __init__(self, connection, lost: Callable[[], None]) -> None:
        self._connection = connection
        self._lost = lost
        self._queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, name="worker-writer", daemon=True).start()

    def put(self, message: tuple) -> None:
        self._queue.put(message)

    def close(self) -> None:
        self._queue.put(None)

    def _newer_preparation(self) -> bool:
        with self._queue.mutex:
            return any(waiting is not None and waiting[0] == "prepare" for waiting in self._queue.queue)

    def _run(self) -> None:
        while True:
            message = self._queue.get()
            if message is None:
                return
            if message[0] == "prepare" and self._newer_preparation():
                continue  # overtaken already
            try:
                self._connection.send(message)
            except Exception:  # noqa: BLE001 -- the pipe broke or was closed: the worker is gone
                self._lost()
                return


class JobController(QObject):
    progressed = Signal(str)  # what the running job is doing
    completed = Signal(object)  # ComparisonOutcome
    # The rest of a result sent first without it (`ComparisonOutcome.ocr_pending`): (job id, the whole outcome), or
    # (job id, None) where it could not be made -- the result stands as it was.
    updated = Signal(str, object)
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
        self._writer: _Writer | None = None
        self._generation = 0  # bumps whenever a worker is dropped, so its late messages are ignored
        self._worker_ready = False
        self._jobs_run = 0  # by the current worker

        self._job_id: str | None = None
        self.last_completed_job: str | None = None  # (whose result was last handed on: an update says which it is for)
        self._finishing: str | None = None  # a job whose result was sent ahead of its rest, still to come
        self._queued: tuple[str, ComparisonRequest] | None = None  # waiting for the worker to be ready
        self._queued_prepare: ComparisonRequest | None = None  # likewise, a preparation (see `prepare`)

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

    def prepare(self, request: ComparisonRequest) -> None:
        """Have the worker run `request` ahead of Compare, quietly, keeping the work for when Compare is pressed (see
        `app.engine.prepared`). Nothing comes back; a comparison already running is never disturbed, and the worker
        drops a preparation the moment anything newer reaches it."""
        if self.is_busy:
            return
        self._ensure_worker()
        if self._worker_ready:
            self._send_prepare(request)
        else:
            self._queued_prepare = request  # the latest only: an older one is stale

    def cancel(self) -> None:
        """Stop the running comparison. The worker is killed and replaced."""
        if not self.is_busy:
            return
        logger.info("comparison cancelled", extra={"job_id": self._job_id})
        self._drop_worker(kill=True)
        self._rest_lost()
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
        generation = self._generation
        self._writer = _Writer(parent_end, lambda: self._lost_from_writer(generation))
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
        process, connection, writer = self._process, self._connection, self._writer
        self._process = self._connection = self._writer = None
        self._worker_ready = False
        if writer is not None:
            if not kill:
                writer.put(("stop",))  # (after anything still on its way: the worker is idle when it is let go politely)
            writer.close()
        if process is None:
            return
        try:
            if not kill and process.is_alive():
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

    def _send_prepare(self, request: ComparisonRequest) -> None:
        self._writer.put(("prepare", request))  # (a lost worker is noticed by its reader)

    def _send_job(self, job_id: str, request: ComparisonRequest) -> None:
        self._writer.put(("job", job_id, request))

    def _lost_from_writer(self, generation: int) -> None:
        """(Writer thread) a message could not be written: the worker is gone -- told the UI thread as the reader would."""
        try:
            self._incoming.emit(generation, ("dead",))
        except RuntimeError:
            pass  # the controller was deleted first

    # -- messages ---------------------------------------------------------------
    def _on_incoming(self, generation: int, message: tuple) -> None:
        if generation != self._generation:
            return  # from a worker that has since been dropped
        kind = message[0]

        if kind == "ready":
            self._worker_ready = True
            prepare, self._queued_prepare = self._queued_prepare, None
            if self._queued is not None:
                job_id, request = self._queued
                self._queued = None
                self._send_job(job_id, request)
            elif prepare is not None:
                self._send_prepare(prepare)
        elif kind == "progress" and message[1] == self._job_id:
            self.progressed.emit(message[2])
        elif kind == "result" and message[1] == self._job_id:
            self._jobs_run += 1
            self.last_completed_job = message[1]
            self._finishing = message[1] if getattr(message[2], "ocr_pending", False) else None
            self._end_job()
            self.completed.emit(message[2])
            if self._finishing is None:
                self._recycle_if_due()  # (not while the rest is being made: it would be lost)
        elif kind in ("update", "update_failed") and message[1] == self._finishing:
            self._finishing = None
            self.updated.emit(message[1], message[2] if kind == "update" else None)
            self._recycle_if_due()
        elif kind == "error" and message[1] == self._job_id:
            self._jobs_run += 1
            self._end_job()
            self.failed.emit(message[2], message[3])
            self._recycle_if_due()
        elif kind == "dead":
            self._on_worker_lost()

    def _rest_lost(self) -> None:
        """The worker was let go while it made the rest of a result sent ahead (`updated`): that rest will not come."""
        if self._finishing is not None:
            finishing, self._finishing = self._finishing, None
            self.updated.emit(finishing, None)

    def _on_worker_lost(self) -> None:
        """The worker died without being told to: a native-library fault, usually a bad file."""
        was_running = self.is_busy
        logger.error("comparison worker died", extra={"job_id": self._job_id})
        self._drop_worker(kill=True)
        self._rest_lost()
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
