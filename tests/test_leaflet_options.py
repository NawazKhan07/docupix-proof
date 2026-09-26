"""Leaflet's two independent strategies (text, visual) live on the results page, the same pattern as Scan vs
Digital's own graphics/text toggles (`tests/test_scan_options.py`), with a lock so at least one stays on."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz
from PIL import Image
from PySide6.QtCore import QObject, Signal

from app.config import AppConfig, LeafletConfig
from app.engine.worker import run_comparison
from app.models.job import ComparisonMode
from app.ui.main_window import MainWindow


class _RecordingJobs(QObject):
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


def test_text_is_on_and_visual_is_off_by_default():
    assert LeafletConfig().compare_text is True
    assert LeafletConfig().compare_visual is False


def _pair(tmp_path: Path) -> tuple[Path, Path]:
    """Two three-page leaflets (a leaflet's pages are what the page choice is about)."""
    paths = []
    for name in ("a.pdf", "b.pdf"):
        doc = fitz.open()
        for number in range(1, 4):
            page = doc.new_page(width=300, height=200)
            page.insert_text((30, 60), f"PARACETAMOL 500mg page {number}", fontsize=12)
        doc.save(tmp_path / name)
        doc.close()
        paths.append(tmp_path / name)
    return paths[0], paths[1]


def _loaded_window(qapp, tmp_path, mode):
    jobs = _RecordingJobs()
    window = MainWindow(jobs=jobs)
    a, b = _pair(tmp_path)
    window.upload_panel.original_slot.load_path(a)
    window.upload_panel.suspect_slot.load_path(b)
    window.mode_selector.set_mode(mode)
    return window, jobs


def _outcome(mode):
    """A real outcome of a comparison. `run_comparison` called directly like this (rather than through
    `compare_leaflet`) always uses the mode's own unchanged, structural options (both compared) -- fixed up
    here to LeafletConfig's own actual default (text on, visual off) for a Leaflet outcome, since that default
    is applied by `compare_leaflet`'s dispatch, not by `run_comparison` itself."""
    page = np.full((900, 1400, 3), 235, np.uint8)
    for row in range(12):
        cv2.putText(page, f"Store below 25 C in a dry place, protect from light {row}", (60, 90 + row * 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.rectangle(page, (900, 100), (1300, 400), (30, 60, 200), -1)
    cv2.circle(page, (1100, 650), 120, (200, 120, 30), -1)
    outcome = run_comparison(page, page.copy(), mode, AppConfig())
    if mode is ComparisonMode.LEAFLET:
        outcome = replace(outcome, text_compared=True, graphics_compared=False)
    return outcome


def test_the_leaflet_toggles_are_offered_on_leaflet_results_only(qapp, tmp_path):
    window, _jobs = _loaded_window(qapp, tmp_path, ComparisonMode.LEAFLET)
    window.show()
    results = window.results_view
    window.stack.setCurrentWidget(results)
    results.show_outcome(_outcome(ComparisonMode.LEAFLET), "a vs b")
    assert results.leaflet_text_check.isVisible() and results.leaflet_text_check.isChecked()
    assert results.leaflet_visual_check.isVisible() and not results.leaflet_visual_check.isChecked()

    results.show_outcome(_outcome(ComparisonMode.AUTOMATIC), "a vs b")
    assert not results.leaflet_text_check.isVisible()
    assert not results.leaflet_visual_check.isVisible()
    window.close()


def test_leaflet_hides_the_edit_files_and_mode_switch_controls(qapp, tmp_path):
    window, _jobs = _loaded_window(qapp, tmp_path, ComparisonMode.LEAFLET)
    results = window.results_view
    results.show_outcome(_outcome(ComparisonMode.LEAFLET), "a vs b")
    assert not results.edit_files_button.isVisible()
    assert not results.barcode_button.isVisible()
    assert not results.braille_button.isVisible()
    window.close()


def test_the_only_ticked_strategy_cannot_be_unticked(qapp, tmp_path):
    window, _jobs = _loaded_window(qapp, tmp_path, ComparisonMode.LEAFLET)
    results = window.results_view
    results.show_outcome(_outcome(ComparisonMode.LEAFLET), "a vs b")

    # only text is on: it is locked (nothing to fall back to), visual is free to tick
    assert results.leaflet_text_check.isChecked() and not results.leaflet_text_check.isEnabled()
    assert results.leaflet_visual_check.isEnabled()

    results.leaflet_visual_check.setChecked(True)
    # both on now: either could be unticked, since the other would still cover it
    assert results.leaflet_text_check.isEnabled()
    assert results.leaflet_visual_check.isEnabled()

    results.leaflet_text_check.setChecked(False)
    # only visual left: now IT is the one locked
    assert not results.leaflet_visual_check.isEnabled()
    window.close()


def test_ticking_visual_compares_the_same_files_again(qapp, tmp_path):
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.LEAFLET)
    window._on_compare()
    (first,) = jobs.requests
    assert first.config.leaflet.compare_visual is False
    window._on_completed(_outcome(ComparisonMode.LEAFLET))

    window.results_view.leaflet_visual_check.setChecked(True)

    assert len(jobs.requests) == 2
    again = jobs.requests[1]
    assert again.mode is ComparisonMode.LEAFLET
    assert (again.original_path, again.suspect_path) == (first.original_path, first.suspect_path)
    assert again.config.leaflet.compare_visual is True
    assert again.config.leaflet.compare_text is True  # untouched
    window.close()


def test_unticking_text_after_visual_is_on_compares_again_without_it(qapp, tmp_path):
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.LEAFLET)
    window._on_compare()
    window._on_completed(_outcome(ComparisonMode.LEAFLET))
    window.results_view.leaflet_visual_check.setChecked(True)
    window._on_completed(_outcome(ComparisonMode.LEAFLET))

    window.results_view.leaflet_text_check.setChecked(False)

    again = jobs.requests[-1]
    assert again.config.leaflet.compare_text is False
    assert again.config.leaflet.compare_visual is True  # the earlier toggle stuck
    window.close()


def test_a_leaflet_toggle_that_does_not_finish_puts_only_its_own_check_back(qapp, tmp_path):
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.LEAFLET)
    window._on_compare()
    window._on_completed(_outcome(ComparisonMode.LEAFLET))
    window.results_view.leaflet_visual_check.setChecked(True)

    window._on_failed("internal", "It broke.")

    assert not window.results_view.leaflet_visual_check.isChecked()
    assert window.results_view.leaflet_text_check.isChecked()  # untouched
    window.close()


# ---- the page picker ----------------------------------------------------------------------------------------


def _choose(picker, pages: set[int]) -> None:
    """Tick exactly `pages` in the picker's list, then close it (which is when the choice counts)."""
    for number, action in enumerate(picker._pages, start=1):
        action.setChecked(number in pages)
    picker._refresh()
    picker._menu.aboutToHide.emit()


def test_the_page_picker_is_offered_on_leaflet_results_only(qapp, tmp_path):
    window, _jobs = _loaded_window(qapp, tmp_path, ComparisonMode.LEAFLET)
    window.show()
    results = window.results_view
    window.stack.setCurrentWidget(results)
    original = window.upload_panel.original
    results.show_outcome(_outcome(ComparisonMode.LEAFLET), "a vs b", original=original)
    assert results.leaflet_pages_picker.isVisible() and results.leaflet_pages_picker.page_count == 3
    assert results.leaflet_pages_picker.text() == "Pages: All"

    results.show_outcome(_outcome(ComparisonMode.AUTOMATIC), "a vs b", original=original)
    assert not results.leaflet_pages_picker.isVisible()
    window.close()


def test_choosing_pages_compares_again_with_the_selection(qapp, tmp_path):
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.LEAFLET)
    window._on_compare()
    (first,) = jobs.requests
    assert first.config.leaflet.compare_pages == ""
    window._on_completed(_outcome(ComparisonMode.LEAFLET))

    _choose(window.results_view.leaflet_pages_picker, {1, 3})

    assert len(jobs.requests) == 2
    again = jobs.requests[1]
    assert (again.original_path, again.suspect_path) == (first.original_path, first.suspect_path)
    assert again.config.leaflet.compare_pages == "1,3"
    assert window.results_view.leaflet_pages_picker.text() == "Pages: 1, 3"
    assert not window.results_view.leaflet_pages_picker.isEnabled()  # locked while it runs
    window.close()


def test_closing_the_list_without_a_change_does_not_start_another_comparison(qapp, tmp_path):
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.LEAFLET)
    window._on_compare()
    window._on_completed(_outcome(ComparisonMode.LEAFLET))

    _choose(window.results_view.leaflet_pages_picker, {1, 2, 3})  # every page: the same as before

    assert len(jobs.requests) == 1
    window.close()


def test_a_page_choice_that_does_not_finish_puts_the_picker_back(qapp, tmp_path):
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.LEAFLET)
    window._on_compare()
    window._on_completed(_outcome(ComparisonMode.LEAFLET))

    _choose(window.results_view.leaflet_pages_picker, {2})
    window._on_failed("internal", "It broke.")

    assert window.results_view.leaflet_pages_picker.spec() == ""
    assert window.results_view.leaflet_pages_picker.isEnabled()
    window.close()


def test_pages_can_be_chosen_on_the_setup_page_before_comparing(qapp, tmp_path):
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.LEAFLET)
    window.show()
    picker = window.upload_panel.original_slot.pages_picker
    assert picker.isVisible() and not window.upload_panel.suspect_slot.pages_picker.isVisible()
    _choose(picker, {2, 3})

    window._on_compare()
    assert jobs.requests[-1].config.leaflet.compare_pages == "2-3"

    window.mode_selector.set_mode(ComparisonMode.AUTOMATIC)
    assert not picker.isVisible()  # a single-page mode: no page choice
    window.close()


def test_page_specs_join_runs_and_read_back():
    from app.ui.page_picker import page_spec, parse_spec

    assert page_spec({1, 2, 3, 5, 7, 8}, 9) == "1-3,5,7-8"
    assert page_spec({1, 2, 3}, 3) == ""  # every page
    assert parse_spec("1-3,5", 6) == {1, 2, 3, 5}
    assert parse_spec("", 4) == {1, 2, 3, 4}
