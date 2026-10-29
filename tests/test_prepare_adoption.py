"""Compare pressed while the same comparison is being prepared: the preparation carries on as the job, instead of the
job starting over (`worker_process._prepare`)."""

from __future__ import annotations

import threading
from multiprocessing import Pipe
from pathlib import Path

from PIL import Image

from app.config import AppConfig
from app.engine import worker_process
from app.engine.worker_process import ComparisonRequest
from app.models.job import ComparisonMode


def _request(tmp_path: Path, name: str = "b.png") -> ComparisonRequest:
    import series_scenes as scenes

    picture = scenes.artwork(None)  # printing enough to align by
    other = scenes.artwork("text")
    Image.fromarray(picture).save(tmp_path / "a.png")
    Image.fromarray(other).save(tmp_path / name)
    return ComparisonRequest(tmp_path / "a.png", tmp_path / name, ComparisonMode.AUTOMATIC, AppConfig())


def _drain(connection) -> list:
    found = []
    while connection.poll(0.2):
        found.append(connection.recv())
    return found


def _listening(connection) -> tuple[threading.Thread, list]:
    """Read everything sent down `connection` as it comes, the way the window does (a result is large: its sending waits
    until someone reads it)."""
    found: list = []

    def listen() -> None:
        while True:
            try:
                message = connection.recv()
            except (EOFError, OSError):
                return
            found.append(message)
            # (a result sent ahead of its rest -- Automatic's words -- is followed by an update to it)
            if message[0] in ("error", "update", "update_failed") or (
                message[0] == "result" and not getattr(message[2], "ocr_pending", False)
            ):
                return

    thread = threading.Thread(target=listen, daemon=True)
    thread.start()
    return thread, found


def test_compare_for_the_comparison_being_prepared_gets_the_prepared_result(tmp_path):
    request = _request(tmp_path)
    app_end, worker_end = Pipe()
    app_end.send(("job", "job-1", request))  # Compare, pressed while this very comparison is being prepared
    listener, messages = _listening(app_end)
    waiting: list = []
    worker_process._prepare(worker_end, request, waiting)
    listener.join(timeout=60)
    results = [m for m in messages if m[0] == "result"]
    assert len(results) == 1 and results[0][1] == "job-1"
    if results[0][2].ocr_pending:  # (Automatic: the pixel findings first, the words after, as an update to job-1)
        assert [m[1] for m in messages if m[0] in ("update", "update_failed")] == ["job-1"]
    assert waiting == []  # nothing left to run again


def test_something_else_arriving_still_interrupts_a_preparation(tmp_path):
    request = _request(tmp_path)
    other = _request(tmp_path, "c.png")
    app_end, worker_end = Pipe()
    app_end.send(("job", "job-2", other))  # a different comparison
    waiting: list = []
    worker_process._prepare(worker_end, request, waiting)
    assert [m[0] for m in waiting] == ["job"] and waiting[0][1] == "job-2"  # handed back to be run next
    assert not [m for m in _drain(app_end) if m[0] == "result"]
