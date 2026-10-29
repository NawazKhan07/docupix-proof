"""Work done before Compare is pressed, kept for when it is (`app.engine.prepared`, the worker's "prepare")."""

from __future__ import annotations

import os
import time
from pathlib import Path

import fake_workers
import numpy as np
import pytest

from app.config import AppConfig, RuntimeConfig
from app.engine import prepared
from app.engine.worker_process import ComparisonRequest, _prepare, execute_request, request_key
from app.models.job import ComparisonMode
from tests.helpers import make_carton_pdf


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    # (what is kept depends on the memory free at the time; these tests are about what happens when there is room)
    monkeypatch.setattr(prepared, "_room_for_pictures", lambda: True)
    prepared.forget()
    yield
    prepared.forget()


def test_pictures_are_not_kept_when_memory_is_short(monkeypatch):
    monkeypatch.setattr(prepared, "_room_for_pictures", lambda: False)
    calls = []
    for _ in range(2):
        prepared.remembered("stage", "key", lambda: calls.append(1) or np.zeros((4, 4), np.uint8))
    prepared.keep_outcome(b"key", object())
    assert calls == [1, 1] and prepared.outcome_for(b"key") is None


class _Pipe:
    """Just enough of a worker's connection for `_prepare`: whether something newer is waiting."""

    def __init__(self, waiting: bool = False):
        self.waiting = waiting

    def poll(self) -> bool:
        return self.waiting


def _request(tmp_path: Path, recolour: bool = True) -> ComparisonRequest:
    original = make_carton_pdf(tmp_path / "original.pdf")
    suspect = make_carton_pdf(tmp_path / "suspect.pdf", recolour=recolour)
    return ComparisonRequest(original, suspect, ComparisonMode.AUTOMATIC, AppConfig())


# ---- the kept stages ------------------------------------------------------------------------------------------------


def test_a_stage_is_computed_once_and_every_caller_gets_its_own_copy():
    calls = []

    def compute():
        calls.append(1)
        return np.zeros((4, 4), np.uint8)

    first = prepared.remembered("stage", "key", compute)
    first[0, 0] = 255  # what one comparison does to its copy...
    second = prepared.remembered("stage", "key", compute)

    assert calls == [1]
    assert second[0, 0] == 0  # ...never reaches another's


def test_a_different_key_is_computed_again():
    calls = []
    prepared.remembered("stage", "a", lambda: calls.append("a") or 1)
    prepared.remembered("stage", "b", lambda: calls.append("b") or 2)
    assert calls == ["a", "b"]


def test_a_picture_digest_tells_pictures_apart_by_every_pixel():
    a = np.zeros((10, 10), np.uint8)
    b = a.copy()
    b[9, 9] = 1
    assert prepared.digest(a) == prepared.digest(a.copy())
    assert prepared.digest(a) != prepared.digest(b)
    assert prepared.digest(a) != prepared.digest(a.reshape(5, 20))


# ---- preparing a comparison ahead of Compare --------------------------------------------------------------------------


def test_a_prepared_comparison_is_what_compare_gets_at_once(tmp_path):
    request = _request(tmp_path)

    _prepare(_Pipe(), request)
    started = time.perf_counter()
    outcome = execute_request(request, reuse=True)

    assert outcome is prepared.outcome_for(request_key(request))
    assert time.perf_counter() - started < 0.5
    assert outcome.clusters  # the recoloured block, found while preparing


def test_the_prepared_outcome_is_the_same_as_comparing_from_scratch(tmp_path):
    request = _request(tmp_path)
    _prepare(_Pipe(), request)
    ready = execute_request(request, reuse=True)
    prepared.forget()
    fresh = execute_request(request)
    assert [(c.detail, c.bbox_in_original) for c in ready.clusters] == [(c.detail, c.bbox_in_original) for c in fresh.clusters]


def test_a_preparation_gives_way_to_anything_newer(tmp_path):
    request = _request(tmp_path)
    _prepare(_Pipe(waiting=True), request)
    assert prepared.outcome_for(request_key(request)) is None


def test_another_mode_reuses_the_prepared_stages(tmp_path):
    request = _request(tmp_path)
    _prepare(_Pipe(), request)
    reused = []
    original = prepared.remembered

    def spy(stage, key, compute):
        before = []
        result = original(stage, key, lambda: before.append(stage) or compute())
        if not before:
            reused.append(stage)
        return result

    prepared.remembered = spy
    try:
        execute_request(ComparisonRequest(request.original_path, request.suspect_path, ComparisonMode.BARCODE, AppConfig()), reuse=True)
    finally:
        prepared.remembered = original
    assert {"render", "align"} <= set(reused)


def test_a_file_changed_on_disk_is_not_answered_from_what_was_prepared(tmp_path):
    request = _request(tmp_path, recolour=False)
    _prepare(_Pipe(), request)
    key_before = request_key(request)

    make_carton_pdf(request.suspect_path, recolour=True)  # the suspect saved again, changed
    stat = os.stat(request.suspect_path)
    os.utime(request.suspect_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    assert request_key(request) != key_before
    assert execute_request(request, reuse=True).clusters  # compared again: the change is found


# ---- the app asks for it -----------------------------------------------------------------------------------------------


def test_the_controller_sends_a_preparation_ahead_of_the_job(qapp, tmp_path):
    from app.ui.job_controller import JobController

    controller = JobController(RuntimeConfig(), target=fake_workers.prepare_recording_main)
    results = []
    controller.completed.connect(results.append)
    request = _request(tmp_path)
    try:
        controller.prepare(request)  # the worker is still starting: it is sent once it is ready
        controller.start(request)
        deadline = time.monotonic() + 20
        while not results:
            assert time.monotonic() < deadline, "timed out"
            qapp.processEvents()
            time.sleep(0.01)
    finally:
        controller.shutdown()
    # (the job was queued as well, and a queued job goes first: a preparation never delays it)
    assert results[0]["request"] == request


def test_the_controller_prepares_when_the_worker_is_idle(qapp, tmp_path):
    from app.ui.job_controller import JobController

    controller = JobController(RuntimeConfig(), target=fake_workers.prepare_recording_main)
    results = []
    controller.completed.connect(results.append)
    request = _request(tmp_path)
    try:
        controller.warm_up()
        deadline = time.monotonic() + 20
        while not controller._worker_ready:
            assert time.monotonic() < deadline, "timed out"
            qapp.processEvents()
            time.sleep(0.01)
        controller.prepare(request)
        controller.start(request)
        while not results:
            assert time.monotonic() < deadline, "timed out"
            qapp.processEvents()
            time.sleep(0.01)
    finally:
        controller.shutdown()
    assert results[0]["prepared"] == [request]


def test_the_window_never_waits_for_a_busy_worker_to_read(qapp, tmp_path):
    """A request bigger than the pipe holds, sent while the worker is deep in a long stage (reading nothing), used to
    hold the window until the stage ended ("Not Responding"): now it is written on a thread of the controller's own."""
    from dataclasses import replace

    from app.ui.job_controller import JobController

    controller = JobController(RuntimeConfig(), target=fake_workers.deaf_while_busy_main)
    results = []
    controller.completed.connect(results.append)
    big = replace(_request(tmp_path), original_page_areas=tuple((page, "x" * 4000) for page in range(40)))
    try:
        controller.warm_up()
        deadline = time.monotonic() + 20
        while not controller._worker_ready:
            assert time.monotonic() < deadline, "timed out"
            qapp.processEvents()
            time.sleep(0.01)
        start = time.monotonic()
        controller.prepare(big)  # taken; the worker then reads nothing for 4 s
        time.sleep(0.3)
        controller.prepare(replace(big, original_page=0))  # overtaken before it is sent
        controller.prepare(replace(big, original_page=1))
        controller.start(big)
        assert time.monotonic() - start < 1.0
        while not results:
            assert time.monotonic() < deadline, "timed out"
            qapp.processEvents()
            time.sleep(0.01)
    finally:
        controller.shutdown()
    assert 2 <= results[0]["prepared"] <= 3  # the first, and the newest (the one in between at most)
