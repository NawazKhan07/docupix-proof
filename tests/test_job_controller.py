"""The comparison runs in a worker process: isolation, cancellation, recycling (Architecture §5, NFR-2)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication

import fake_workers
from app.config import AppConfig, RuntimeConfig
from app.engine.worker_process import ComparisonRequest, ErrorKind
from app.models.job import ComparisonMode
from app.ui.job_controller import JobController


def _request(name: str = "x") -> ComparisonRequest:
    return ComparisonRequest(Path(f"{name}_a.png"), Path(f"{name}_b.png"), ComparisonMode.AUTOMATIC, AppConfig())


def _wait(qapp, condition, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            pytest.fail("timed out waiting")
        qapp.processEvents()
        time.sleep(0.01)


class Recorder:
    def __init__(self, controller: JobController):
        self.results, self.errors, self.progress, self.cancelled, self.busy = [], [], [], 0, []
        controller.completed.connect(self.results.append)
        controller.failed.connect(lambda kind, message: self.errors.append((kind, message)))
        controller.progressed.connect(self.progress.append)
        controller.cancelled.connect(lambda: setattr(self, "cancelled", self.cancelled + 1))
        controller.busyChanged.connect(self.busy.append)


@pytest.fixture
def make_controller(qapp):
    made = []

    def make(target, **runtime):
        controller = JobController(RuntimeConfig(**runtime), target=target)
        made.append(controller)
        return controller, Recorder(controller)

    yield make
    for controller in made:
        controller.shutdown()


# ---- a normal job -----------------------------------------------------------------------


def test_a_job_runs_in_another_process_and_its_result_comes_back(qapp, make_controller):
    controller, seen = make_controller(fake_workers.echo_main)

    controller.start(_request())
    _wait(qapp, lambda: seen.results)

    assert seen.results[0]["pid"] != os.getpid()
    assert seen.results[0]["request"].mode is ComparisonMode.AUTOMATIC
    assert "working" in seen.progress
    assert seen.busy == [True, False]
    assert not controller.is_busy


def test_only_one_job_runs_at_a_time(qapp, make_controller):
    controller, _seen = make_controller(fake_workers.slow_main)
    controller.start(_request())
    with pytest.raises(RuntimeError):
        controller.start(_request())


def test_a_job_started_before_the_worker_is_ready_waits_for_it(qapp, make_controller):
    controller, seen = make_controller(fake_workers.never_ready_main)
    controller.start(_request())
    assert any("Starting" in text for text in seen.progress)
    _wait(qapp, lambda: seen.results)
    assert seen.results == ["ok"]


def test_warming_up_starts_a_worker_before_any_job(qapp, make_controller):
    controller, seen = make_controller(fake_workers.echo_main)
    assert controller.worker_pid is None
    controller.warm_up()
    assert controller.worker_pid is not None
    _wait(qapp, lambda: controller._worker_ready)
    controller.start(_request())
    _wait(qapp, lambda: seen.results)


# ---- a failing file must not take the app down ---------------------------------------------


def test_a_worker_that_dies_is_reported_as_a_clean_error_and_replaced(qapp, make_controller):
    controller, seen = make_controller(fake_workers.crash_main)
    controller.start(_request())
    first_pid = controller.worker_pid

    _wait(qapp, lambda: seen.errors)

    kind, message = seen.errors[0]
    assert kind == ErrorKind.CRASH and "damaged" in message
    assert not controller.is_busy
    assert QApplication.instance() is not None  # and this process is, evidently, still here
    _wait(qapp, lambda: controller.worker_pid not in (None, first_pid))  # a fresh worker is on its way


def test_a_reported_failure_reaches_the_ui_with_its_kind(qapp, make_controller):
    controller, seen = make_controller(fake_workers.failing_main)
    controller.start(_request())
    _wait(qapp, lambda: seen.errors)
    assert seen.errors == [(ErrorKind.INPUT, "the file is too large")]
    assert not controller.is_busy


def test_the_next_job_runs_after_a_crash(qapp, make_controller):
    controller, seen = make_controller(fake_workers.crash_main)
    controller.start(_request())
    _wait(qapp, lambda: seen.errors)
    controller.start(_request())  # accepted: no stuck "busy" state
    _wait(qapp, lambda: len(seen.errors) == 2)


# ---- cancellation ---------------------------------------------------------------------------


def test_cancelling_kills_the_worker_and_starts_a_fresh_one(qapp, make_controller):
    controller, seen = make_controller(fake_workers.slow_main)
    controller.start(_request())
    _wait(qapp, lambda: "thinking" in seen.progress)
    busy_worker = controller.worker_pid

    controller.cancel()

    assert seen.cancelled == 1 and not controller.is_busy
    assert controller.worker_pid not in (None, busy_worker)  # replaced, not reused
    assert seen.results == [] and seen.errors == []  # nothing from the killed job ever arrives


def test_a_cancelled_jobs_late_output_is_ignored(qapp, make_controller):
    controller, seen = make_controller(fake_workers.echo_main)
    controller.warm_up()
    _wait(qapp, lambda: controller._worker_ready)
    controller.start(_request())
    controller.cancel()  # before the answer is read
    for _ in range(50):
        qapp.processEvents()
        time.sleep(0.01)
    assert seen.results == [] and seen.cancelled == 1


def test_cancel_with_nothing_running_does_nothing(qapp, make_controller):
    controller, seen = make_controller(fake_workers.echo_main)
    controller.cancel()
    assert seen.cancelled == 0


def test_a_job_can_be_run_after_cancelling_one(qapp, make_controller):
    controller, seen = make_controller(fake_workers.echo_main)
    controller.warm_up()
    _wait(qapp, lambda: controller._worker_ready)
    controller.start(_request("first"))
    controller.cancel()
    controller.start(_request("second"))
    _wait(qapp, lambda: seen.results)
    assert seen.results[-1]["request"].original_path.name == "second_a.png"


# ---- time limit --------------------------------------------------------------------------------


def test_a_job_past_its_time_limit_is_stopped_and_reported(qapp, make_controller):
    controller, seen = make_controller(fake_workers.slow_main, job_timeout_s=0)
    controller.start(_request())
    _wait(qapp, lambda: seen.errors)
    assert seen.errors[0][0] == ErrorKind.TIMEOUT
    assert not controller.is_busy


# ---- recycling ------------------------------------------------------------------------------------


def test_the_worker_is_replaced_after_the_configured_number_of_jobs(qapp, make_controller):
    controller, seen = make_controller(fake_workers.echo_main, worker_recycle_after_jobs=2)

    pids = []
    for index in range(4):
        controller.start(_request(str(index)))
        _wait(qapp, lambda: len(seen.results) == index + 1)
        pids.append(seen.results[-1]["pid"])
        # once the recycle has happened, the replacement is ready before the next job
        _wait(qapp, lambda: controller.worker_pid is not None)

    assert pids[0] == pids[1]  # two jobs on the first worker
    assert pids[2] == pids[3] != pids[0]  # then a fresh one for the next two


def test_shutting_down_stops_the_worker(qapp, make_controller):
    controller, _seen = make_controller(fake_workers.echo_main)
    controller.warm_up()
    process = controller._process
    controller.shutdown()
    assert not process.is_alive()
    assert controller.worker_pid is None


def test_shutting_down_mid_job_does_not_wait_for_the_job(qapp, make_controller):
    controller, seen = make_controller(fake_workers.slow_main)
    controller.start(_request())
    _wait(qapp, lambda: "thinking" in seen.progress)
    started = time.monotonic()
    controller.shutdown()
    assert time.monotonic() - started < 2.5


# ---- the real worker, end to end -----------------------------------------------------------------


def _pair(tmp_path: Path) -> tuple[Path, Path]:
    image = np.full((400, 600, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (40, 30), (560, 370), (0, 0, 0), 3)
    cv2.putText(image, "PARACETAMOL 500mg", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
    cv2.rectangle(image, (70, 200), (240, 290), (200, 30, 40), -1)
    cv2.circle(image, (430, 240), 50, (30, 90, 200), -1)
    changed = image.copy()
    cv2.rectangle(changed, (70, 200), (240, 290), (40, 160, 60), -1)  # red -> green
    a, b = tmp_path / "original.png", tmp_path / "suspect.png"
    Image.fromarray(image).save(a)
    Image.fromarray(changed).save(b)
    return a, b


def test_the_real_worker_compares_two_files_from_their_paths(qapp, tmp_path):
    controller = JobController(AppConfig().runtime)
    seen = Recorder(controller)
    try:
        original, suspect = _pair(tmp_path)
        controller.start(ComparisonRequest(original, suspect, ComparisonMode.AUTOMATIC, AppConfig()))
        _wait(qapp, lambda: seen.results or seen.errors, timeout=90)

        assert not seen.errors, seen.errors
        outcome = seen.results[0]
        assert outcome.clusters  # the recoloured block was found
        assert outcome.aligned_original.shape == outcome.aligned_suspect.shape
        assert outcome.alignment.residual_error_px >= 0  # the residual travels with the result (NFR-3)
        assert any("Aligning" in text for text in seen.progress)
    finally:
        controller.shutdown()


def test_the_real_worker_reports_a_file_over_a_limit_someone_set_as_an_input_error(qapp, tmp_path):
    controller = JobController(RuntimeConfig(max_file_size_mb=-1))
    seen = Recorder(controller)
    try:
        original, suspect = _pair(tmp_path)
        controller.start(
            ComparisonRequest(original, suspect, ComparisonMode.AUTOMATIC, AppConfig(runtime=RuntimeConfig(max_file_size_mb=-1)))
        )
        _wait(qapp, lambda: seen.results or seen.errors, timeout=90)

        assert seen.errors and seen.errors[0][0] == ErrorKind.INPUT
        assert "max_file_size_mb" in seen.errors[0][1]
    finally:
        controller.shutdown()


def test_the_real_worker_reports_a_missing_file_as_an_input_error_and_carries_on(qapp, tmp_path):
    controller = JobController(AppConfig().runtime)
    seen = Recorder(controller)
    try:
        original, suspect = _pair(tmp_path)
        controller.start(ComparisonRequest(tmp_path / "gone.png", suspect, ComparisonMode.AUTOMATIC, AppConfig()))
        _wait(qapp, lambda: seen.errors, timeout=90)
        assert seen.errors[0][0] == ErrorKind.INPUT

        # the same worker is still serving
        first_worker = controller.worker_pid
        controller.start(ComparisonRequest(original, suspect, ComparisonMode.AUTOMATIC, AppConfig()))
        _wait(qapp, lambda: seen.results, timeout=90)
        assert controller.worker_pid == first_worker
    finally:
        controller.shutdown()


def test_a_result_sent_ahead_is_followed_by_its_rest_for_that_job(qapp, tmp_path):
    """Automatic's pixel findings are sent as soon as they are made, its words after: the controller hands on the result,
    then the update -- for that job -- and does not recycle the worker in between (that would lose the rest)."""
    import time

    from app.config import RuntimeConfig
    from app.ui.job_controller import JobController

    controller = JobController(RuntimeConfig(worker_recycle_after_jobs=1), target=fake_workers.ahead_main)
    results, updates = [], []
    controller.completed.connect(results.append)
    controller.updated.connect(lambda job, outcome: updates.append((job, outcome)))
    try:
        job = controller.start(_request(tmp_path))
        deadline = time.monotonic() + 20
        while not updates:
            assert time.monotonic() < deadline, "timed out"
            qapp.processEvents()
            time.sleep(0.01)
    finally:
        controller.shutdown()
    assert results[0].label == "pixels" and controller.last_completed_job == job
    assert updates == [(job, updates[0][1])] and updates[0][1].label == "whole"
