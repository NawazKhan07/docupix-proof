"""The main window while a comparison runs in the worker: responsive, cancellable, honest about failures."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import fake_workers
import numpy as np
import pytest
from PIL import Image

from app.config import RuntimeConfig
from app.ui.job_controller import JobController
from app.ui.main_window import MainWindow


def _pair(tmp_path: Path) -> tuple[Path, Path]:
    image = np.full((400, 600, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (40, 30), (560, 370), (0, 0, 0), 3)
    cv2.putText(image, "PARACETAMOL 500mg", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    Image.fromarray(image).save(a)
    Image.fromarray(image).save(b)
    return a, b


def _wait(qapp, condition, timeout=20.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting"
        qapp.processEvents()
        time.sleep(0.01)


@pytest.fixture
def busy_window(qapp, tmp_path):
    """A window whose worker never finishes, mid-comparison."""
    controller = JobController(RuntimeConfig(), target=fake_workers.slow_main)
    window = MainWindow(jobs=controller)
    window.show()
    a, b = _pair(tmp_path)
    window.upload_panel.original_slot.load_path(a)
    window.upload_panel.suspect_slot.load_path(b)
    window._on_compare()
    _wait(qapp, lambda: "thinking" in window.status_label.text() or window._job_stage == "thinking")
    yield window, controller
    controller.shutdown()
    window.close()


def test_while_comparing_the_inputs_are_locked_and_cancel_is_offered(busy_window):
    window, _controller = busy_window
    assert window.cancel_button.isVisible()
    assert not window.compare_button.isEnabled()
    assert not window.upload_panel.isEnabled()  # a file cannot be swapped under a running job
    assert not window.mode_selector.isEnabled()


def test_the_ui_keeps_responding_while_the_worker_is_busy(busy_window, qapp):
    """The whole point of the worker process: the event loop is never blocked."""
    window, _controller = busy_window
    started = time.monotonic()
    for _ in range(20):
        qapp.processEvents()
    assert time.monotonic() - started < 1.0
    assert window.status_label.text()  # and it is saying something


def test_the_status_shows_what_it_is_doing_and_for_how_long(busy_window):
    window, _controller = busy_window
    assert "thinking" in window.status_label.text()
    assert " s)" in window.status_label.text()


def test_cancelling_frees_the_window_for_another_go(busy_window, qapp):
    window, controller = busy_window

    window.cancel_button.click()

    assert window.status_label.text() == "Comparison cancelled."
    assert not window.cancel_button.isVisible()
    assert window.compare_button.isEnabled() and window.upload_panel.isEnabled() and window.mode_selector.isEnabled()
    assert not controller.is_busy
    assert window.stack.currentWidget() is not window.results_view


def test_a_second_click_on_compare_while_running_is_ignored(busy_window):
    window, controller = busy_window
    first = controller._job_id
    window._on_compare()
    assert controller._job_id == first


def test_a_file_over_a_limit_someone_set_says_so_in_the_status_strip(qapp, tmp_path, shared_jobs, run_comparison_in_window):
    a, b = _pair(tmp_path)
    window = MainWindow(jobs=shared_jobs)
    try:
        window.upload_panel.original_slot.load_path(a)
        window.upload_panel.suspect_slot.load_path(b)
        # the limit is applied by the worker when it loads the file, from the request's config
        window.config = type(window.config)(runtime=RuntimeConfig(max_file_size_mb=-1))
        run_comparison_in_window(window)

        assert window.status_label.property("status") == "error"
        assert "max_file_size_mb" in window.status_label.text()
        assert window.stack.currentWidget() is not window.results_view
        assert window.compare_button.isEnabled()
    finally:
        window.close()


def test_a_crashing_worker_leaves_a_window_that_still_works(qapp, tmp_path):
    controller = JobController(RuntimeConfig(), target=fake_workers.crash_main)
    window = MainWindow(jobs=controller)
    try:
        a, b = _pair(tmp_path)
        window.upload_panel.original_slot.load_path(a)
        window.upload_panel.suspect_slot.load_path(b)
        window._on_compare()
        _wait(qapp, lambda: window.status_label.property("status") == "error")

        assert "damaged" in window.status_label.text()
        assert window.compare_button.isEnabled()  # ready to try again
    finally:
        controller.shutdown()
        window.close()


def test_closing_the_window_stops_the_worker_it_started(qapp):
    window = MainWindow()  # owns its controller
    window.jobs.warm_up()
    process = window.jobs._process
    window.close()
    assert not process.is_alive()


def test_closing_the_window_leaves_a_shared_worker_alone(qapp, shared_jobs):
    shared_jobs.warm_up()
    window = MainWindow(jobs=shared_jobs)
    window.close()
    assert shared_jobs._process is not None and shared_jobs._process.is_alive()


def test_opening_the_window_warms_the_worker_up_unless_told_not_to(qapp, monkeypatch):
    monkeypatch.delenv("ARTWORK_COMPARE_NO_WARMUP", raising=False)
    window = MainWindow(jobs=None)
    window._owns_jobs = True
    warmed = []
    window.jobs.warm_up = lambda: warmed.append(1)
    window.show()
    qapp.processEvents()
    qapp.processEvents()
    window.close()
    assert warmed == [1]


def test_loading_both_files_prepares_the_comparison_before_compare_is_pressed(qapp, tmp_path):
    """Rendering, lifting the artwork and aligning start as soon as the files are in (see `app.engine.prepared`):
    once the setup has stayed put a moment, the worker is asked to prepare exactly what Compare would run."""
    controller = JobController(RuntimeConfig(), target=fake_workers.prepare_recording_main)
    window = MainWindow(jobs=controller)
    results = []
    controller.completed.connect(results.append)
    try:
        controller.warm_up()
        _wait(qapp, lambda: controller._worker_ready)
        a, b = _pair(tmp_path)
        window.upload_panel.original_slot.load_path(a)
        window.upload_panel.suspect_slot.load_path(b)
        _wait(qapp, lambda: not window._prepare_timer.isActive())  # the pause after the last change
        expected = window._setup_request()
        window._on_compare()
        _wait(qapp, lambda: bool(results))

        assert expected in results[0]["prepared"]
        assert results[0]["request"] == expected
    finally:
        controller.shutdown()
        window.close()
