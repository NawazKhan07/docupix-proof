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
        self.prepared = []

    def start(self, request):
        self.requests.append(request)
        return len(self.requests)

    def prepare(self, request):
        self.prepared.append(request)

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


def test_the_origin_mode_appears_in_tools_dropdown_and_separate_button_is_removed(qapp, tmp_path):
    """Switching to Barcode offers Automatic in Tools dropdown, while separate button is removed."""
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.AUTOMATIC)
    window.show()
    window.stack.setCurrentWidget(window.results_view)
    window._on_compare()
    window._on_completed(_outcome(ComparisonMode.AUTOMATIC))

    window.results_view.barcode_button.click()
    window._on_completed(_outcome(ComparisonMode.BARCODE))

    # Separate origin button is removed from toolbar
    assert not window.results_view.origin_mode_button.isVisible()
    # Tools menu has "Automatic"
    assert window.results_view.automatic_action.text() == "Automatic"

    # Selecting "Automatic" resets Tools button text to "Tools"
    window.results_view.automatic_action.trigger()
    assert window.results_view.tools_button.text() == "Tools"

    # Clicking Recompare triggers comparison in Automatic mode
    window.results_view.recompare_button.click()
    assert len(jobs.requests) == 3
    assert jobs.requests[2].mode is ComparisonMode.AUTOMATIC
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

    assert window.results_view.automatic_action.text() == "Automatic"
    assert not window.results_view.origin_mode_button.isVisible()
    window.close()


# ---- editing a file's artwork area from the results page -----------------------------------------------------


def test_editing_a_file_and_accepting_compares_again_with_the_new_geometry(qapp, tmp_path, monkeypatch):
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.AUTOMATIC)
    window.show()
    window.stack.setCurrentWidget(window.results_view)
    window._on_compare()
    (first,) = jobs.requests
    window._on_completed(_outcome(ComparisonMode.AUTOMATIC))

    new_geometry = IsolationGeometry(boundary=NormalizedRect(0.1, 0.1, 0.9, 0.9))

    def _accept(self):
        self.geometry = new_geometry
        return True

    monkeypatch.setattr(type(window.upload_panel.original_slot), "edit_artwork_area", _accept)

    window.results_view.editFilesRequested.emit("original")

    # Editing starts an align-only request first to display aligned files in viewer:
    assert len(jobs.requests) == 2
    again = jobs.requests[1]
    assert again.align_only is True
    assert again.original_geometry is new_geometry
    assert (again.original_path, again.suspect_path) == (first.original_path, first.suspect_path)

    # When alignment finishes, viewer displays the aligned files and waits for user to click Recompare:
    window._on_completed(_outcome(ComparisonMode.AUTOMATIC))
    assert window.results_view.recompare_button.isVisible()
    assert len(jobs.prepared) == 1
    assert jobs.prepared[0].align_only is False
    assert jobs.prepared[0].original_geometry is new_geometry
    assert jobs.prepared[0].mode is ComparisonMode.AUTOMATIC

    # User clicks Recompare to run the full comparison:
    window.results_view.recompare_button.click()
    assert len(jobs.requests) == 3
    full = jobs.requests[2]
    assert full.align_only is False
    assert full.original_geometry is new_geometry
    assert full.mode is ComparisonMode.AUTOMATIC
    window.close()


def test_editing_files_in_scan_vs_digital_mode_aligns_first_and_prepares_in_background(qapp, tmp_path, monkeypatch):
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.SCAN_VS_DIGITAL)
    window.show()
    window.stack.setCurrentWidget(window.results_view)
    window._on_compare()
    (first,) = jobs.requests
    window._on_completed(_outcome(ComparisonMode.SCAN_VS_DIGITAL))

    new_geometry = IsolationGeometry(boundary=NormalizedRect(0.15, 0.15, 0.85, 0.85))
    monkeypatch.setattr(type(window.upload_panel.original_slot), "edit_artwork_area", lambda self: setattr(self, "geometry", new_geometry) or True)

    window.results_view.editFilesRequested.emit("original")

    # Editing starts an align-only request first to display aligned files in viewer:
    assert len(jobs.requests) == 2
    again = jobs.requests[1]
    assert again.align_only is True
    assert again.mode is ComparisonMode.SCAN_VS_DIGITAL
    assert again.original_geometry is new_geometry

    # When alignment finishes, viewer displays the aligned files and prepares in the background:
    window._on_completed(_outcome(ComparisonMode.SCAN_VS_DIGITAL))
    assert window.results_view.recompare_button.isVisible()
    assert len(jobs.prepared) == 1
    assert jobs.prepared[-1].align_only is False
    assert jobs.prepared[-1].mode is ComparisonMode.SCAN_VS_DIGITAL
    assert jobs.prepared[-1].original_geometry is new_geometry

    # User clicks Recompare to run full comparison:
    window.results_view.recompare_button.click()
    assert len(jobs.requests) == 3
    full = jobs.requests[2]
    assert full.align_only is False
    assert full.mode is ComparisonMode.SCAN_VS_DIGITAL
    assert full.original_geometry is new_geometry
    window.close()


def test_editing_files_in_series_mode_aligns_first_and_prepares_in_background(qapp, tmp_path, monkeypatch):
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.SERIES)
    window.show()
    window.stack.setCurrentWidget(window.results_view)
    window._on_compare()
    (first,) = jobs.requests
    window._on_completed(_outcome(ComparisonMode.SERIES))

    new_geometry = IsolationGeometry(boundary=NormalizedRect(0.2, 0.2, 0.8, 0.8))
    monkeypatch.setattr(type(window.upload_panel.original_slot), "edit_artwork_area", lambda self: setattr(self, "geometry", new_geometry) or True)

    window.results_view.editFilesRequested.emit("original")

    # Align-only request started first:
    assert len(jobs.requests) == 2
    again = jobs.requests[1]
    assert again.align_only is True
    assert again.mode is ComparisonMode.SERIES
    assert again.original_geometry is new_geometry

    # When alignment finishes, viewer displays aligned files and prepares in background:
    window._on_completed(_outcome(ComparisonMode.SERIES))
    assert window.results_view.recompare_button.isVisible()
    assert len(jobs.prepared) == 1
    assert jobs.prepared[-1].align_only is False
    assert jobs.prepared[-1].mode is ComparisonMode.SERIES
    assert jobs.prepared[-1].original_geometry is new_geometry

    # User clicks Recompare:
    window.results_view.recompare_button.click()
    assert len(jobs.requests) == 3
    full = jobs.requests[2]
    assert full.align_only is False
    assert full.mode is ComparisonMode.SERIES
    assert full.original_geometry is new_geometry
    window.close()


def test_editing_files_in_leaflet_mode_aligns_first_and_prepares_in_background(qapp, tmp_path, monkeypatch):
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.LEAFLET)
    window.show()
    window.stack.setCurrentWidget(window.results_view)
    window._on_compare()
    (first,) = jobs.requests
    window._on_completed(_outcome(ComparisonMode.LEAFLET))

    page_geometry = IsolationGeometry(boundary=NormalizedRect(0.05, 0.05, 0.95, 0.95))

    def _accept_leaflet(self):
        self.page_areas = {0: page_geometry}
        return True

    monkeypatch.setattr(type(window.upload_panel.original_slot), "edit_artwork_area", _accept_leaflet)

    window.results_view.editFilesRequested.emit("original")

    # Align-only request started first:
    assert len(jobs.requests) == 2
    again = jobs.requests[1]
    assert again.align_only is True
    assert again.mode is ComparisonMode.LEAFLET
    assert again.original_page_areas == ((0, page_geometry),)

    # When alignment finishes, viewer displays aligned files and prepares in background:
    window._on_completed(_outcome(ComparisonMode.LEAFLET))
    assert window.results_view.recompare_button.isVisible()
    assert len(jobs.prepared) == 1
    assert jobs.prepared[-1].align_only is False
    assert jobs.prepared[-1].mode is ComparisonMode.LEAFLET
    assert jobs.prepared[-1].original_page_areas == ((0, page_geometry),)

    # User clicks Recompare:
    window.results_view.recompare_button.click()
    assert len(jobs.requests) == 3
    full = jobs.requests[2]
    assert full.align_only is False
    assert full.mode is ComparisonMode.LEAFLET
    assert full.original_page_areas == ((0, page_geometry),)
    window.close()


def test_editing_files_in_barcode_and_braille_modes_aligns_first_and_prepares_in_background(qapp, tmp_path, monkeypatch):
    for mode in (ComparisonMode.BARCODE, ComparisonMode.BRAILLE):
        window, jobs = _loaded_window(qapp, tmp_path, mode)
        window.show()
        window.stack.setCurrentWidget(window.results_view)
        window._on_compare()
        window._on_completed(_outcome(mode))

        new_geometry = IsolationGeometry(boundary=NormalizedRect(0.1, 0.1, 0.9, 0.9))
        monkeypatch.setattr(type(window.upload_panel.original_slot), "edit_artwork_area", lambda self: setattr(self, "geometry", new_geometry) or True)

        window.results_view.editFilesRequested.emit("original")

        assert len(jobs.requests) == 2
        assert jobs.requests[1].align_only is True
        assert jobs.requests[1].mode is mode

        window._on_completed(_outcome(mode))
        assert window.results_view.recompare_button.isVisible()
        assert len(jobs.prepared) == 1
        assert jobs.prepared[-1].align_only is False
        assert jobs.prepared[-1].mode is mode

        window.results_view.recompare_button.click()
        assert len(jobs.requests) == 3
        assert jobs.requests[2].align_only is False
        assert jobs.requests[2].mode is mode
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


# ---- tools dropdown and recompare button -------------------------------------------------------------


def test_tools_and_recompare_buttons_appear_from_the_start(qapp, tmp_path):
    """In Automatic mode, Tools dropdown and Recompare button appear from the start."""
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.AUTOMATIC)
    window.show()
    window.stack.setCurrentWidget(window.results_view)
    window._on_compare()
    (first,) = jobs.requests
    window._on_completed(_outcome(ComparisonMode.AUTOMATIC))

    results = window.results_view
    assert results.tools_button.isVisible()
    assert results.tools_button.text() == "Tools"
    assert results.recompare_button.isVisible()
    assert results.recompare_button.text() == "Recompare"
    assert results.barcode_action.text() == "Barcode"
    assert results.braille_action.text() == "Braille"

    # Clicking Recompare directly re-compares in current mode (Automatic):
    results.recompare_button.click()
    assert len(jobs.requests) == 2
    again = jobs.requests[1]
    assert again.mode is ComparisonMode.AUTOMATIC
    assert again.align_only is False
    window.close()


def test_selecting_barcode_from_tools_updates_text_and_shows_recompare(qapp, tmp_path):
    """Selecting Barcode changes the Tools button label to Barcode and Recompare runs Barcode mode."""
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.AUTOMATIC)
    window.show()
    window.stack.setCurrentWidget(window.results_view)
    window._on_compare()
    (first,) = jobs.requests
    window._on_completed(_outcome(ComparisonMode.AUTOMATIC))

    results = window.results_view
    results.barcode_action.trigger()

    assert results.tools_button.text() == "Barcode"
    assert results.recompare_button.isVisible()
    assert results.recompare_button.text() == "Recompare"

    # Clicking Recompare triggers the Barcode comparison
    results.recompare_button.click()

    assert len(jobs.requests) == 2
    again = jobs.requests[1]
    assert again.mode is ComparisonMode.BARCODE
    assert (again.original_path, again.suspect_path) == (first.original_path, first.suspect_path)
    window.close()


def test_selecting_braille_from_tools_updates_text_and_shows_recompare(qapp, tmp_path):
    """Selecting Braille changes the Tools button label to Braille and Recompare runs Braille mode."""
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.AUTOMATIC)
    window.show()
    window.stack.setCurrentWidget(window.results_view)
    window._on_compare()
    (first,) = jobs.requests
    window._on_completed(_outcome(ComparisonMode.AUTOMATIC))

    results = window.results_view
    results.braille_action.trigger()

    assert results.tools_button.text() == "Braille"
    assert results.recompare_button.isVisible()

    results.recompare_button.click()

    assert len(jobs.requests) == 2
    again = jobs.requests[1]
    assert again.mode is ComparisonMode.BRAILLE
    assert (again.original_path, again.suspect_path) == (first.original_path, first.suspect_path)
    window.close()


def test_selecting_automatic_deselects_tool_mode_and_resets_tools_button(qapp, tmp_path):
    """Selecting Automatic from Tools dropdown deselects Barcode and resets the button label to Tools."""
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.AUTOMATIC)
    window.show()
    window.stack.setCurrentWidget(window.results_view)
    window._on_compare()
    window._on_completed(_outcome(ComparisonMode.AUTOMATIC))

    results = window.results_view
    results.barcode_action.trigger()
    assert results.tools_button.text() == "Barcode"

    # Selecting Automatic deselects tool and resets label to "Tools"
    results.automatic_action.trigger()
    assert results.tools_button.text() == "Tools"

    # Recompare compares in Automatic mode
    results.recompare_button.click()
    assert len(jobs.requests) == 2
    assert jobs.requests[1].mode is ComparisonMode.AUTOMATIC
    window.close()


def test_tools_button_hidden_in_modes_without_tools(qapp, tmp_path):
    """Modes like Scan vs Digital, Series, or Leaflet do not show the Tools button."""
    window, _jobs = _loaded_window(qapp, tmp_path, ComparisonMode.SCAN_VS_DIGITAL)
    window.show()
    results = window.results_view
    window.stack.setCurrentWidget(results)
    results.show_outcome(_outcome(ComparisonMode.SCAN_VS_DIGITAL), "a vs b")
    assert not results.tools_button.isVisible()
    assert results.recompare_button.isVisible()

    results.show_outcome(_outcome(ComparisonMode.SERIES), "a vs b")
    assert not results.tools_button.isVisible()
    assert results.recompare_button.isVisible()

    results.show_outcome(_outcome(ComparisonMode.LEAFLET), "a vs b")
    assert not results.tools_button.isVisible()
    assert results.recompare_button.isVisible()
    window.close()
