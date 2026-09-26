"""Scan vs Digital compares text; its graphics are an option the reviewer turns on -- from the results page, so there is
no going back to tick a box and start again. The setup page also keeps a way back to the last results, and its mode panel
never changes width when a mode is chosen."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, Signal

from app.config import DetectionConfig
from app.engine.detectors import text_diff
from app.engine.worker import run_comparison
from app.config import AppConfig
from app.models.diff import BoundingBox, DiffCandidate, DiffSubtype, DiffType
from app.models.job import ComparisonMode
from app.ui.main_window import MainWindow
from app.ui.mode_selector import ModeSelector


class _RecordingJobs(QObject):
    """Stands in for the job controller: keeps the requests it was asked to run."""

    progressed = Signal(str)
    completed = Signal(object)
    failed = Signal(str, str)
    cancelled = Signal()
    busyChanged = Signal(bool)
    is_busy = False

    def __init__(self):
        super().__init__()
        self.requests = []

    def start(self, request):
        self.requests.append(request)
        return len(self.requests)

    def cancel(self):
        pass

    def warm_up(self):
        pass


def test_the_option_starts_off_because_a_scans_graphics_give_false_differences():
    assert DetectionConfig().scan_compare_graphics is False


def test_the_text_option_starts_on_because_it_is_what_scan_vs_digital_compares():
    assert DetectionConfig().scan_compare_text is True


def _pair(tmp_path: Path) -> tuple[Path, Path]:
    image = np.full((400, 600, 3), 255, dtype=np.uint8)
    cv2.putText(image, "PARACETAMOL 500mg", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    Image.fromarray(image).save(a)
    Image.fromarray(image).save(b)
    return a, b


def _loaded_window(qapp, tmp_path, mode):
    jobs = _RecordingJobs()
    window = MainWindow(jobs=jobs)
    a, b = _pair(tmp_path)
    window.upload_panel.original_slot.load_path(a)
    window.upload_panel.suspect_slot.load_path(b)
    window.mode_selector.set_mode(mode)
    return window, jobs


def _outcome(mode):
    """A real outcome of a comparison (a sheet with enough on it to register)."""
    page = np.full((900, 1400, 3), 235, np.uint8)
    for row in range(12):
        cv2.putText(page, f"Store below 25 C in a dry place, protect from light {row}", (60, 90 + row * 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.rectangle(page, (900, 100), (1300, 400), (30, 60, 200), -1)
    cv2.circle(page, (1100, 650), 120, (200, 120, 30), -1)
    return run_comparison(page, page.copy(), mode, AppConfig())


def _outcome_with_findings(monkeypatch):
    """A Scan vs Digital outcome with a real text finding AND a real graphics finding, so the results page's
    two checkboxes have something to filter (OCR is faked: this machine need not have Tesseract installed)."""
    fake_text = DiffCandidate(BoundingBox(50, 50, 150, 90), DiffType.TEXT, DiffSubtype.CHANGED, 0.9, "text changed")
    monkeypatch.setattr(text_diff, "detect_scan", lambda *a, **k: text_diff.TextScanResult([fake_text]))
    original = np.full((900, 1400, 3), 235, np.uint8)
    for row in range(12):
        cv2.putText(original, f"Store below 25 C in a dry place, protect from light {row}", (60, 90 + row * 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.rectangle(original, (900, 100), (1300, 400), (30, 60, 200), -1)
    changed = original.copy()
    cv2.rectangle(changed, (900, 100), (1300, 400), (226, 238, 246), -1)
    cv2.rectangle(changed, (950, 450), (1350, 750), (200, 60, 50), -1)
    return run_comparison(original, changed, ComparisonMode.SCAN_VS_DIGITAL, AppConfig())


# ---- the mode panel keeps its width ------------------------------------------------------------------------------------------------------


def test_the_mode_panel_is_the_same_width_whichever_mode_is_chosen(qapp):
    selector = ModeSelector()
    selector.show()
    widths = set()
    for mode in ComparisonMode:
        if mode in selector._cards:
            selector.set_mode(mode)
            qapp.processEvents()
            widths.add(selector.width())
    assert len(widths) == 1


# ---- the graphics option lives on the results page --------------------------------------------------------------------------------------


def test_the_graphics_toggle_is_offered_on_the_results_of_scan_vs_digital_only(qapp, tmp_path):
    window, _jobs = _loaded_window(qapp, tmp_path, ComparisonMode.SCAN_VS_DIGITAL)
    window.show()
    results = window.results_view
    window.stack.setCurrentWidget(results)
    results.show_outcome(_outcome(ComparisonMode.SCAN_VS_DIGITAL), "a vs b")
    assert results.graphics_check.isVisible() and not results.graphics_check.isChecked()

    results.show_outcome(_outcome(ComparisonMode.AUTOMATIC), "a vs b")
    assert not results.graphics_check.isVisible()
    window.close()


def test_ticking_graphics_on_shows_the_already_found_graphics_findings_instantly(qapp, tmp_path, monkeypatch):
    """Graphics are compared either way; the checkbox only decides what is shown (`app.models.diff.visible_clusters`),
    so this never asks for another comparison."""
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.SCAN_VS_DIGITAL)
    window._on_compare()
    window._on_completed(_outcome_with_findings(monkeypatch))
    assert not any(c.type is DiffType.GRAPHICS for c in window.results_view._outcome.clusters)

    window.results_view.graphics_check.setChecked(True)

    assert len(jobs.requests) == 1  # nothing recompared
    assert any(c.type is DiffType.GRAPHICS for c in window.results_view._outcome.clusters)
    assert window.results_view.graphics_check.isEnabled()  # never locked: nothing is running
    window.close()


def test_unticking_graphics_off_again_hides_them_instantly(qapp, tmp_path, monkeypatch):
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.SCAN_VS_DIGITAL)
    window._on_compare()
    window._on_completed(_outcome_with_findings(monkeypatch))
    window.results_view.graphics_check.setChecked(True)

    window.results_view.graphics_check.setChecked(False)

    assert len(jobs.requests) == 1  # still nothing recompared
    assert not any(c.type is DiffType.GRAPHICS for c in window.results_view._outcome.clusters)
    window.close()


def test_a_failure_elsewhere_does_not_touch_the_graphics_or_text_checkboxes(qapp, tmp_path, monkeypatch):
    """Nothing these two checkboxes do can fail (there is no comparison to fail): a re-comparison started some
    other way (here, editing a file's area) must not revert or lock either of them."""
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.SCAN_VS_DIGITAL)
    window._on_compare()
    window._on_completed(_outcome_with_findings(monkeypatch))
    window.results_view.graphics_check.setChecked(True)
    monkeypatch.setattr(type(window.upload_panel.original_slot), "edit_artwork_area", lambda self: True)
    window.results_view.editFilesRequested.emit("original")

    window._on_failed("internal", "It broke.")

    assert window.results_view.graphics_check.isChecked()  # untouched
    assert window.results_view.graphics_check.isEnabled()
    assert "It broke." in window.results_view.busy_label.text()
    window.close()


def test_the_choice_does_not_leak_into_other_modes(qapp, tmp_path):
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.AUTOMATIC)
    window._on_compare()
    (request,) = jobs.requests
    assert request.mode is ComparisonMode.AUTOMATIC
    assert request.config.detection.scan_compare_graphics is False
    window.close()


# ---- the text (OCR) option lives on the results page too, for a pure visual comparison ----------------------------------------------------


def test_the_text_toggle_is_offered_on_the_results_of_scan_vs_digital_only(qapp, tmp_path):
    window, _jobs = _loaded_window(qapp, tmp_path, ComparisonMode.SCAN_VS_DIGITAL)
    window.show()
    results = window.results_view
    window.stack.setCurrentWidget(results)
    results.show_outcome(_outcome(ComparisonMode.SCAN_VS_DIGITAL), "a vs b")
    assert results.text_check.isVisible() and results.text_check.isChecked()

    results.show_outcome(_outcome(ComparisonMode.AUTOMATIC), "a vs b")
    assert not results.text_check.isVisible()
    window.close()


def test_unticking_text_off_hides_the_text_findings_instantly(qapp, tmp_path, monkeypatch):
    """Text is read either way; the checkbox only decides what is shown, so this never asks for another
    comparison (`app.models.diff.visible_clusters`)."""
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.SCAN_VS_DIGITAL)
    window._on_compare()
    window._on_completed(_outcome_with_findings(monkeypatch))
    assert any(c.type is DiffType.TEXT for c in window.results_view._outcome.clusters)

    window.results_view.text_check.setChecked(False)

    assert len(jobs.requests) == 1  # nothing recompared
    assert not any(c.type is DiffType.TEXT for c in window.results_view._outcome.clusters)
    assert window.results_view.text_check.isEnabled()  # never locked: nothing is running
    window.close()


def test_the_graphics_and_text_checkboxes_do_not_affect_each_other(qapp, tmp_path, monkeypatch):
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.SCAN_VS_DIGITAL)
    window._on_compare()
    window._on_completed(_outcome_with_findings(monkeypatch))

    window.results_view.graphics_check.setChecked(True)
    window.results_view.text_check.setChecked(False)

    assert len(jobs.requests) == 1  # nothing ever recompared
    clusters = window.results_view._outcome.clusters
    assert any(c.type is DiffType.GRAPHICS for c in clusters)  # the earlier toggle stuck
    assert not any(c.type is DiffType.TEXT for c in clusters)
    window.close()


# ---- a way back to the last results ---------------------------------------------------------------------------------------------------------


def test_new_comparison_leaves_a_way_back_to_the_last_results(qapp, tmp_path):
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.AUTOMATIC)
    window.show()
    assert not window.results_button.isVisible()  # nothing to go back to yet

    window._on_compare()
    window._on_completed(_outcome(ComparisonMode.AUTOMATIC))
    assert window.stack.currentWidget() is window.results_view

    window.results_view.back_button.click()  # "New comparison", pressed by accident
    assert window.stack.currentWidget() is window._setup_page
    assert window.results_button.isVisible()

    window.results_button.click()
    assert window.stack.currentWidget() is window.results_view
    window.close()
