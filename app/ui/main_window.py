"""Main application window."""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from time import monotonic

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.config import load_config
from app.engine.worker import ComparisonOutcome
from app.engine.worker_process import ComparisonRequest, ErrorKind
from app.models.job import MODE_DISPLAY_NAMES, ComparisonMode
from app.ui.job_controller import JobController
from app.ui.mode_selector import ModeSelector
from app.ui.results_view import ResultsView
from app.ui.theme import set_variant
from app.ui.upload_panel import UploadPanel

logger = logging.getLogger(__name__)

NO_WARMUP_ENV = "ARTWORK_COMPARE_NO_WARMUP"  # set by the test suite: do not start a worker at open
_PROGRESS_TICK_MS = 500
_SETUP_MAX_WIDTH_PX = 1500  # the setup page's content is never wider than this, however wide the window


class MainWindow(QMainWindow):
    def __init__(self, jobs: JobController | None = None):
        """`jobs` lets a caller supply the job controller (tests share one worker); the
        window shuts down only the controller it created itself."""
        super().__init__()
        self.setWindowTitle("Artwork Comparison")
        self.resize(1320, 840)
        self.config = load_config()
        self._owns_jobs = jobs is None
        self.jobs = jobs or JobController(self.config.runtime, parent=self)
        self._job_context: tuple[object, object, ComparisonMode] | None = None
        self._last_request: ComparisonRequest | None = None  # what the results on show came from
        self._job_from_results = False  # the running job was started from the results page (a toggle there)
        self._reverting_check = None  # which results-page checkbox to flip back if that job does not finish
        self._reverting_field = None  # the pages field's previous text to restore instead, if that started it
        self._pending_request: ComparisonRequest | None = None
        self._job_started = 0.0
        self._job_stage = ""

        # Two pages: setting up a comparison, and reviewing its results.
        self.stack = QStackedWidget(self)
        self.setCentralWidget(self.stack)
        central = QWidget(self.stack)
        self.stack.addWidget(central)
        self.results_view = ResultsView(self.stack)
        self.stack.addWidget(self.results_view)
        self.results_view.backRequested.connect(self._show_setup)
        self.results_view.modeSwitchRequested.connect(self._on_mode_switch_requested)
        self.results_view.editFilesRequested.connect(self._on_edit_files_requested)
        self.results_view.leafletTextToggled.connect(self._on_leaflet_text_toggled)
        self.results_view.leafletVisualToggled.connect(self._on_leaflet_visual_toggled)
        self.results_view.leafletPagesChanged.connect(self._on_leaflet_pages_changed)
        self._setup_page = central

        # On a wide monitor the page keeps a readable width, centred, rather than stretch edge to edge: its
        # content sits in a column with a maximum width, and the spare width either side of it is empty.
        centring = QHBoxLayout(central)
        centring.setContentsMargins(0, 0, 0, 0)
        column = QWidget(central)
        column.setMaximumWidth(_SETUP_MAX_WIDTH_PX)
        centring.addStretch(1)
        centring.addWidget(column, stretch=1000)
        centring.addStretch(1)
        layout = QVBoxLayout(column)
        layout.setContentsMargins(24, 20, 24, 24)
        layout.setSpacing(16)

        heading = QLabel("Artwork Comparison")
        heading.setProperty("role", "heading")
        subheading = QLabel("Compare pharmaceutical packaging artwork files, offline.")
        subheading.setProperty("role", "subheading")
        subheading.setWordWrap(True)
        layout.addWidget(heading)
        layout.addWidget(subheading)

        top_row = QHBoxLayout()
        top_row.setSpacing(16)
        self.upload_panel = UploadPanel(column, config=self.config)
        self.mode_selector = ModeSelector(column)
        top_row.addWidget(self.upload_panel, stretch=1)
        top_row.addWidget(self.mode_selector, stretch=1)
        layout.addLayout(top_row, stretch=1)  # the panels take the spare height: Compare stays at the bottom

        self.status_label = QLabel("Load an original and a suspect file to compare.")
        self.status_label.setProperty("status", "idle")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        self.compare_button = QPushButton("Compare")
        self.compare_button.setProperty("variant", "primary")
        self.compare_button.setEnabled(False)
        self.compare_button.setMinimumHeight(40)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setMinimumHeight(40)
        self.cancel_button.setVisible(False)
        # After "New comparison" the last results are still there: a button takes the reviewer back to them.
        self.results_button = QPushButton("Back to the last results")
        self.results_button.setMinimumHeight(40)
        self.results_button.setVisible(False)
        self.results_button.clicked.connect(lambda: self.stack.setCurrentWidget(self.results_view))
        buttons.addWidget(self.compare_button, stretch=1)
        buttons.addWidget(self.results_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)

        self.mode_selector.modeChanged.connect(self.upload_panel.set_mode)
        self.upload_panel.filesChanged.connect(self._update_compare_enabled)
        # A replaced file is closed, so the results page must not keep drawing from it.
        self.upload_panel.filesChanged.connect(self.results_view.release_sources)
        self.compare_button.clicked.connect(self._on_compare)
        self.cancel_button.clicked.connect(self.jobs.cancel)

        self.jobs.progressed.connect(self._on_progress)
        self.jobs.completed.connect(self._on_completed)
        self.jobs.failed.connect(self._on_failed)
        self.jobs.cancelled.connect(self._on_cancelled)

        # The elapsed time on the status line, so a long comparison never looks frozen.
        self._ticker = QTimer(self)
        self._ticker.setInterval(_PROGRESS_TICK_MS)
        self._ticker.timeout.connect(self._show_progress)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._owns_jobs and not os.environ.get(NO_WARMUP_ENV):
            QTimer.singleShot(0, self.jobs.warm_up)  # imports take a while; do them while the user picks files

    def _update_compare_enabled(self) -> None:
        self.compare_button.setEnabled(self.upload_panel.both_loaded)

    def _set_status(self, text: str, state: str) -> None:
        self.status_label.setText(text)
        set_variant(self.status_label, "status", state)

    def _show_setup(self) -> None:
        self.results_button.setVisible(self.results_view.has_results)
        self.stack.setCurrentWidget(self._setup_page)

    def _on_compare(self) -> None:
        """Start the comparison in the worker process; the UI stays responsive meanwhile."""
        original, suspect = self.upload_panel.original, self.upload_panel.suspect
        assert original is not None and suspect is not None
        if self.jobs.is_busy:
            return
        config = self.config
        if self.mode_selector.mode is ComparisonMode.LEAFLET:
            pages = self.upload_panel.original_slot.pages_to_compare
            config = replace(config, leaflet=replace(config.leaflet, compare_pages=pages))
        request = ComparisonRequest(
            original_path=original.path,
            suspect_path=suspect.path,
            mode=self.mode_selector.mode,
            config=config,
            original_geometry=self.upload_panel.original_slot.geometry,
            suspect_geometry=self.upload_panel.suspect_slot.geometry,
        )
        # A fresh compare from the setup page is what "the mode this comparison started in" means -- a
        # results-page toggle or Barcode/Braille quick switch must never move this, or there would be no way
        # back to it once switched away.
        self.results_view.set_origin_mode(request.mode)
        self._start(request, (original, suspect, request.mode), from_results=False)

    def _on_leaflet_text_toggled(self, checked: bool) -> None:
        """Leaflet's text strategy switched on or off on the results page: the same files, compared again."""
        request, context = self._last_request, self._job_context
        if request is None or context is None or self.jobs.is_busy:
            return
        leaflet = replace(request.config.leaflet, compare_text=checked)
        self._reverting_check = self.results_view.leaflet_text_check
        self._reverting_field = None
        self.results_view.set_busy("Comparing again…")
        self._start(replace(request, config=replace(request.config, leaflet=leaflet)), context, from_results=True)

    def _on_leaflet_visual_toggled(self, checked: bool) -> None:
        """Leaflet's visual strategy switched on or off on the results page: the same files, compared again."""
        request, context = self._last_request, self._job_context
        if request is None or context is None or self.jobs.is_busy:
            return
        leaflet = replace(request.config.leaflet, compare_visual=checked)
        self._reverting_check = self.results_view.leaflet_visual_check
        self._reverting_field = None
        self.results_view.set_busy("Comparing again…")
        self._start(replace(request, config=replace(request.config, leaflet=leaflet)), context, from_results=True)

    def _on_leaflet_pages_changed(self, spec: str) -> None:
        """Leaflet's page selection edited on the results page: the same files, compared again."""
        request, context = self._last_request, self._job_context
        if request is None or context is None or self.jobs.is_busy:
            return
        leaflet = replace(request.config.leaflet, compare_pages=spec)
        self._reverting_check = None
        self._reverting_field = request.config.leaflet.compare_pages
        self.results_view.set_busy("Comparing again…")
        self._start(replace(request, config=replace(request.config, leaflet=leaflet)), context, from_results=True)

    def _on_mode_switch_requested(self, mode: ComparisonMode) -> None:
        """Barcode/Braille quick switch from the results page: the same files, a different mode, shown instantly."""
        request, context = self._last_request, self._job_context
        if request is None or context is None or self.jobs.is_busy:
            return
        original, suspect, _old_mode = context
        self._reverting_check = None
        self._reverting_field = None
        self.results_view.set_busy("Comparing again…")
        self._start(replace(request, mode=mode), (original, suspect, mode), from_results=True)

    def _on_edit_files_requested(self, which: str) -> None:
        """Edit files from the results page: re-crop or re-lift, then compare again if anything changed."""
        request, context = self._last_request, self._job_context
        if request is None or context is None or self.jobs.is_busy:
            return
        slot = self.upload_panel.original_slot if which == "original" else self.upload_panel.suspect_slot
        if slot.loaded is None or not slot.edit_artwork_area():
            return  # nothing loaded there, or the dialog was cancelled
        updated = replace(
            request,
            original_geometry=self.upload_panel.original_slot.geometry,
            suspect_geometry=self.upload_panel.suspect_slot.geometry,
        )
        self._reverting_check = None
        self._reverting_field = None
        self.results_view.set_busy("Comparing again…")
        self._start(updated, context, from_results=True)

    def _start(self, request: ComparisonRequest, context, from_results: bool) -> None:
        self._job_context = context
        self._job_from_results = from_results
        self._pending_request = request
        job_id = self.jobs.start(request)
        logger.info(
            "comparison job started",
            extra={
                "job_id": job_id,
                "mode": request.mode.value,
                "original": str(request.original_path),
                "suspect": str(request.suspect_path),
            },
        )
        self._set_running(True)
        self._job_started, self._job_stage = monotonic(), "Starting…"
        self._show_progress()
        self._ticker.start()

    def _set_running(self, running: bool) -> None:
        """While a comparison runs the inputs are locked, so a file cannot be swapped under it."""
        self.compare_button.setEnabled(not running and self.upload_panel.both_loaded)
        self.cancel_button.setVisible(running)
        self.upload_panel.setEnabled(not running)
        self.mode_selector.setEnabled(not running)
        if not running:
            self._ticker.stop()

    def _on_progress(self, stage: str) -> None:
        self._job_stage = stage
        self._show_progress()

    def _show_progress(self) -> None:
        elapsed = monotonic() - self._job_started
        text = f"{self._job_stage}  ({elapsed:.0f} s)"
        self._set_status(text, "idle")
        if self._job_from_results:
            self.results_view.set_busy(text)

    def _on_completed(self, outcome: ComparisonOutcome) -> None:
        self._set_running(False)
        original, suspect, mode = self._job_context
        self._last_request = self._pending_request
        self._set_status(*_describe(outcome))
        self.results_view.show_outcome(
            outcome,
            f"{original.path.name}  vs  {suspect.path.name}  \u00b7  {MODE_DISPLAY_NAMES[mode]}",
            original=original,
            suspect=suspect,
        )
        self.stack.setCurrentWidget(self.results_view)

    def _on_failed(self, kind: str, message: str) -> None:
        self._set_running(False)
        if self._job_from_results:
            self._back_on_results(f"Could not compare again: {message}")
        # Stay on the setup page: the files and settings are still there to fix and retry.
        self._set_status(message, "error")
        if kind in (ErrorKind.CRASH, ErrorKind.INTERNAL):
            logger.error("comparison did not complete", extra={"kind": kind})

    def _on_cancelled(self) -> None:
        self._set_running(False)
        if self._job_from_results:
            self._back_on_results("")
        self._set_status("Comparison cancelled.", "idle")

    def _back_on_results(self, message: str) -> None:
        """A comparison started from the results page did not finish: the results stay as they were, and the
        control that started it goes back to what they were compared with."""
        check = self._reverting_check
        if check is not None:
            check.blockSignals(True)
            check.setChecked(not check.isChecked())
            check.blockSignals(False)
        if self._reverting_field is not None:
            self.results_view.revert_leaflet_pages_field(self._reverting_field)
            self._reverting_field = None
        self.results_view.set_busy(message, locked=False)

    def closeEvent(self, event) -> None:
        if self._owns_jobs:
            self.jobs.shutdown()
        self.upload_panel.original_slot.clear()
        self.upload_panel.suspect_slot.clear()
        super().closeEvent(event)


def _describe(outcome) -> tuple[str, str]:
    """Summarise a comparison for the status strip: (text, severity state).

    Anything that was *not* checked (OCR unavailable, no decodable symbols)
    is stated alongside the result, so "0 differences" is never mistaken for
    "everything was verified".
    """
    alignment = outcome.alignment
    parts = [
        f"{len(outcome.clusters)} finding(s)" if outcome.clusters else "No differences found"
    ]

    for comparison in outcome.symbol_comparisons:
        parts.append(f"{comparison.symbol_type}: {comparison.status}")
    if outcome.braille is not None:
        braille = outcome.braille
        parts.append(
            f"Braille {braille.matched} matched / {braille.missing} missing / "
            f"{braille.added} added / {braille.moved} moved"
        )

    parts.append(
        f"alignment {alignment.residual_error_px:.2f} px"
        + (" (LOW CONFIDENCE)" if alignment.low_confidence else "")
    )
    parts.extend(outcome.warnings)

    if alignment.low_confidence or outcome.warnings:
        state = "warn"
    elif outcome.clusters:
        state = "warn"
    else:
        state = "ok"
    return " · ".join(parts), state
