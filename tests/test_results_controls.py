"""Two results-page controls that act on the same files without going back to the setup page: a quick switch to
Barcode or Braille mode, and re-editing either file's artwork area (Build Prompt Phase 5 follow-up)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, Signal

from app.config import AppConfig
from app.engine.worker import run_comparison
from app.models.isolation import IsolationGeometry, NormalizedRect
from app.models.job import ComparisonMode
from app.ui.main_window import MainWindow


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
    page = np.full((900, 1400, 3), 235, np.uint8)
    for row in range(12):
        cv2.putText(page, f"Store below 25 C in a dry place, protect from light {row}", (60, 90 + row * 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.rectangle(page, (900, 100), (1300, 400), (30, 60, 200), -1)
    cv2.circle(page, (1100, 650), 120, (200, 120, 30), -1)
    return run_comparison(page, page.copy(), mode, AppConfig())


# ---- switching to Barcode / Braille from the results page ----------------------------------------------------


def test_the_switch_buttons_are_offered_for_every_mode_but_leaflet(qapp, tmp_path):
    window, _jobs = _loaded_window(qapp, tmp_path, ComparisonMode.AUTOMATIC)
    window.show()
    results = window.results_view
    window.stack.setCurrentWidget(results)
    results.show_outcome(_outcome(ComparisonMode.AUTOMATIC), "a vs b")
    assert results.barcode_button.isVisible() and results.braille_button.isVisible()
    window.close()


def test_the_switch_buttons_are_not_offered_from_scan_vs_digital_or_series(qapp, tmp_path):
    """Barcode/Braille are Automatic's own quick-checks (and each other's), not a general-purpose mode
    switcher: a mode with its own specific comparison must not offer them."""
    window, _jobs = _loaded_window(qapp, tmp_path, ComparisonMode.SCAN_VS_DIGITAL)
    window.show()
    results = window.results_view
    window.stack.setCurrentWidget(results)
    results.show_outcome(_outcome(ComparisonMode.SCAN_VS_DIGITAL), "a vs b")
    assert not results.barcode_button.isVisible() and not results.braille_button.isVisible()

    results.show_outcome(_outcome(ComparisonMode.SERIES), "a vs b")
    assert not results.barcode_button.isVisible() and not results.braille_button.isVisible()
    window.close()


def test_the_button_for_the_mode_already_shown_is_disabled(qapp, tmp_path):
    window, _jobs = _loaded_window(qapp, tmp_path, ComparisonMode.BARCODE)
    results = window.results_view
    results.show_outcome(_outcome(ComparisonMode.BARCODE), "a vs b")
    assert not results.barcode_button.isEnabled()
    assert results.braille_button.isEnabled()
    window.close()


def test_clicking_barcode_compares_the_same_files_again_restricted_to_barcode(qapp, tmp_path):
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.AUTOMATIC)
    window._on_compare()
    (first,) = jobs.requests
    window._on_completed(_outcome(ComparisonMode.AUTOMATIC))

    window.results_view.barcode_button.click()

    assert len(jobs.requests) == 2
    again = jobs.requests[1]
    assert again.mode is ComparisonMode.BARCODE
    assert (again.original_path, again.suspect_path) == (first.original_path, first.suspect_path)
    assert not window.results_view.barcode_button.isEnabled()  # locked while it runs
    window.close()


def test_the_results_subtitle_reflects_the_switched_to_mode(qapp, tmp_path):
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.AUTOMATIC)
    window._on_compare()
    window._on_completed(_outcome(ComparisonMode.AUTOMATIC))

    window.results_view.barcode_button.click()
    window._on_completed(_outcome(ComparisonMode.BARCODE))

    assert "Barcode" in window.results_view.subtitle_label.text()
    window.close()


# (Scan vs Digital's own graphics/text checkboxes now just filter findings already found -- no re-comparison, so
# nothing to revert if one fails; see test_scan_options.py. A results-page toggle that DOES still start a fresh
# comparison and reverts only its own check on failure is covered for Leaflet's two strategies in
# test_leaflet_options.py::test_a_leaflet_toggle_that_does_not_finish_puts_only_its_own_check_back.)


# ---- a way back to the mode this comparison actually started in -----------------------------------------------


def test_the_origin_button_is_hidden_when_already_showing_the_origin_mode(qapp, tmp_path):
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.AUTOMATIC)
    window._on_compare()
    window._on_completed(_outcome(ComparisonMode.AUTOMATIC))
    assert not window.results_view.origin_mode_button.isVisible()
    window.close()


def test_the_origin_button_appears_after_switching_to_barcode_and_names_the_origin_mode(qapp, tmp_path):
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.AUTOMATIC)
    window.show()
    window.stack.setCurrentWidget(window.results_view)
    window._on_compare()
    window._on_completed(_outcome(ComparisonMode.AUTOMATIC))

    window.results_view.barcode_button.click()
    window._on_completed(_outcome(ComparisonMode.BARCODE))

    assert window.results_view.origin_mode_button.isVisible()
    assert window.results_view.origin_mode_button.text() == "Automatic"
    window.close()


def test_clicking_the_origin_button_compares_the_same_files_again_in_the_origin_mode(qapp, tmp_path):
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.AUTOMATIC)
    window._on_compare()
    (first,) = jobs.requests
    window._on_completed(_outcome(ComparisonMode.AUTOMATIC))
    window.results_view.barcode_button.click()
    window._on_completed(_outcome(ComparisonMode.BARCODE))

    window.results_view.origin_mode_button.click()

    assert len(jobs.requests) == 3
    again = jobs.requests[2]
    assert again.mode is ComparisonMode.AUTOMATIC
    assert (again.original_path, again.suspect_path) == (first.original_path, first.suspect_path)
    assert not window.results_view.origin_mode_button.isEnabled()  # locked while it runs
    window.close()


def test_the_origin_button_hides_again_once_back_in_the_origin_mode(qapp, tmp_path):
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.AUTOMATIC)
    window._on_compare()
    window._on_completed(_outcome(ComparisonMode.AUTOMATIC))
    window.results_view.barcode_button.click()
    window._on_completed(_outcome(ComparisonMode.BARCODE))
    window.results_view.origin_mode_button.click()

    window._on_completed(_outcome(ComparisonMode.AUTOMATIC))

    assert not window.results_view.origin_mode_button.isVisible()
    window.close()


def test_the_origin_button_does_not_duplicate_a_barcode_or_braille_origin(qapp, tmp_path):
    """Starting the comparison in Barcode mode already has its own dedicated (disabled) button."""
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.BARCODE)
    window._on_compare()
    window._on_completed(_outcome(ComparisonMode.BARCODE))
    window.results_view.braille_button.click()
    window._on_completed(_outcome(ComparisonMode.BRAILLE))

    assert not window.results_view.origin_mode_button.isVisible()
    window.close()


def test_a_switch_between_barcode_and_braille_does_not_move_the_origin_mode(qapp, tmp_path):
    """Only a fresh Compare from the setup page redefines what 'the origin mode' means."""
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.AUTOMATIC)
    window._on_compare()
    window._on_completed(_outcome(ComparisonMode.AUTOMATIC))

    window.results_view.barcode_button.click()
    window._on_completed(_outcome(ComparisonMode.BARCODE))
    window.results_view.braille_button.click()
    window._on_completed(_outcome(ComparisonMode.BRAILLE))

    assert window.results_view.origin_mode_button.text() == "Automatic"
    window.close()


# ---- editing a file's artwork area from the results page -----------------------------------------------------


def test_editing_a_file_and_accepting_compares_again_with_the_new_geometry(qapp, tmp_path, monkeypatch):
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.AUTOMATIC)
    window._on_compare()
    (first,) = jobs.requests
    window._on_completed(_outcome(ComparisonMode.AUTOMATIC))

    new_geometry = IsolationGeometry(boundary=NormalizedRect(0.1, 0.1, 0.9, 0.9))

    def _accept(self):
        self.geometry = new_geometry
        return True

    monkeypatch.setattr(type(window.upload_panel.original_slot), "edit_artwork_area", _accept)

    window.results_view.editFilesRequested.emit("original")

    assert len(jobs.requests) == 2
    again = jobs.requests[1]
    assert again.original_geometry is new_geometry
    assert (again.original_path, again.suspect_path) == (first.original_path, first.suspect_path)
    window.close()


def test_cancelling_the_edit_dialog_does_not_start_another_comparison(qapp, tmp_path, monkeypatch):
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.AUTOMATIC)
    window._on_compare()
    window._on_completed(_outcome(ComparisonMode.AUTOMATIC))

    monkeypatch.setattr(type(window.upload_panel.suspect_slot), "edit_artwork_area", lambda self: False)

    window.results_view.editFilesRequested.emit("suspect")

    assert len(jobs.requests) == 1  # nothing new started
    window.close()


def test_editing_files_is_locked_while_a_comparison_is_running(qapp, tmp_path):
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.AUTOMATIC)
    window._on_compare()
    window._on_completed(_outcome(ComparisonMode.AUTOMATIC))

    window.results_view.barcode_button.click()  # starts a job, still "running" (jobs.is_busy is False by default...)

    assert not window.results_view.edit_files_button.isEnabled()
    window.close()
