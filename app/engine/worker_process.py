"""The comparison, run inside a worker process (Architecture §5).

A malformed file can crash the native code underneath OpenCV and PyMuPDF. Run
in the UI process, that would take the whole application down mid-review, so the
comparison runs in a separate process instead: if it dies, only it does, the UI
reports a clean error, and a fresh worker replaces it.

This module is what the worker process runs, so it must not import Qt (a worker
that pulled in the GUI toolkit would start slower and use more memory for
nothing). The Qt side -- starting, watching, recycling and cancelling the worker
-- is `app.ui.job_controller`.

Protocol over the pipe (each message is a tuple):

  parent -> worker   ("job", job_id, ComparisonRequest)   run one comparison
                     ("stop",)                             finish up and exit
  worker -> parent   ("ready",)                            imports done, waiting for a job
                     ("progress", job_id, text)            what it is doing now
                     ("result", job_id, ComparisonOutcome)
                     ("error", job_id, kind, message)      kind: see ErrorKind
"""

from __future__ import annotations

import gc
import logging
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Callable

from app import offline_guard
from app.config import AppConfig
from app.engine.alignment.aligner import AlignmentError
from app.engine.file_loader import ArtworkLoadError, load_artwork
from app.engine.ocr_locator import configure_ocr
from app.engine.worker import ComparisonOutcome, run_comparison
from app.logging_config import job_log_file
from app.models.isolation import IsolationGeometry
from app.models.job import ComparisonMode

logger = logging.getLogger(__name__)


class ErrorKind:
    """Why a job failed, so the UI can word it for a person (never a traceback)."""

    INPUT = "input"  # a file could not be loaded, or is too large: the user can fix it
    ALIGNMENT = "alignment"  # the two files could not be registered
    MEMORY = "memory"  # the machine ran out of memory
    CRASH = "crash"  # the worker process died (a native-library fault, usually a corrupt file)
    TIMEOUT = "timeout"  # the job ran past its time limit and was stopped
    INTERNAL = "internal"  # a bug; details are in the job log


@dataclass(frozen=True)
class ComparisonRequest:
    """Everything a worker needs -- small, so it is cheap to send.

    Files are passed as paths and rendered inside the worker rather than sent as
    pixels: a 310 x 420 mm carton is 55 MB per file, and rendering there keeps
    that off the pipe.
    """

    original_path: Path
    suspect_path: Path
    mode: ComparisonMode
    config: AppConfig
    original_geometry: IsolationGeometry | None = None
    suspect_geometry: IsolationGeometry | None = None


def execute_request(
    request: ComparisonRequest, progress: Callable[[str], None] | None = None
) -> ComparisonOutcome:
    """Load, render and compare the two files. Runs in the worker; also usable directly."""
    say = progress or (lambda _text: None)
    say("Loading the files…")
    limits = request.config.runtime
    with load_artwork(request.original_path, limits) as original, load_artwork(
        request.suspect_path, limits
    ) as suspect:
        say("Rendering the artwork…")
        original_image, suspect_image = original.render(), suspect.render()
        return run_comparison(
            original_image,
            suspect_image,
            request.mode,
            request.config,
            original_geometry=request.original_geometry,
            suspect_geometry=request.suspect_geometry,
            progress=say,
        )


def _classify(exc: BaseException) -> tuple[str, str]:
    """(ErrorKind, message a person can act on) for an exception raised by a job."""
    if isinstance(exc, ArtworkLoadError):
        return ErrorKind.INPUT, str(exc)
    if isinstance(exc, AlignmentError):
        return ErrorKind.ALIGNMENT, str(exc)
    if isinstance(exc, MemoryError):
        return ErrorKind.MEMORY, (
            "The computer ran out of memory comparing these files. Close other applications, "
            "or export the artwork at a smaller page size or resolution."
        )
    return ErrorKind.INTERNAL, (
        "The comparison failed unexpectedly. The details are in the job log; "
        f"the error was: {type(exc).__name__}: {exc}"
    )


def worker_main(connection: Connection) -> None:
    """The worker process's loop: take a job, run it, send back the result."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    offline_guard.install_if_requested()  # a spawned process starts clean; it must opt in like its parent
    configure_ocr()
    connection.send(("ready",))
    while True:
        try:
            message = connection.recv()
        except (EOFError, OSError):
            return  # the app went away; nothing left to serve
        if message[0] == "stop":
            return
        if message[0] != "job":
            continue

        _kind, job_id, request = message
        try:
            with job_log_file(job_id):
                outcome = execute_request(
                    request, progress=lambda text, job=job_id: connection.send(("progress", job, text))
                )
            connection.send(("result", job_id, outcome))
        except Exception as exc:  # a job's failure is reported, and the worker carries on
            error_kind, text = _classify(exc)
            if error_kind == ErrorKind.INTERNAL:
                logger.error("job %s failed\n%s", job_id, traceback.format_exc())
            connection.send(("error", job_id, error_kind, text))
        finally:
            # Large arrays from the finished job would otherwise wait for the next
            # collection while this process sits idle holding hundreds of MB.
            outcome = request = None
            gc.collect()
