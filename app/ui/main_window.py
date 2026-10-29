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
from app.models.languages import AUTO
from app.ui.automatic_settings import AutomaticSettings
from app.ui.job_controller import JobController
from app.ui.mode_selector import ModeSelector
from app.ui.results_view import ResultsView
from app.ui.theme import set_variant
from app.ui.upload_panel import UploadPanel

logger = logging.getLogger(__name__)

NO_WARMUP_ENV = "ARTWORK_COMPARE_NO_WARMUP"  # set by the test suite: do not start a worker at open
_PROGRESS_TICK_MS = 500
_PREPARE_DELAY_MS = 600  # the comparison is prepared once the setup has stayed put this long
_STALL_CHECK_MS = 250  # the window checks it is answering this often...
_STALL_REPORT_S = 1.5  # ...and logs it whenever it was held up longer than this (what held it, to find and fix)
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
        self._results_job: str | None = None  # the job whose result is on show (an update to it is for it)
        self._job_from_results = False  # the running job was started from the results page (a toggle there)
        self._swap_note = ""  # Scan vs Digital: the files were given the other way round and swapped (`_files_put_right`)
        self._reverting_check = None  # which results-page checkbox to flip back if that job does not finish
        self._reverting_pages = False  # whether a results-page page selector goes back if that job does not finish
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
        self.results_view.leafletPagesChanged.connect(self._on_leaflet_pages_changed)
        self.results_view.pageChangeRequested.connect(self._on_page_change_requested)
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

        # (Every mode reads text in the languages found printed on the files: there is nothing to choose.)
        # Automatic mode's own settings (Food label), shown only while it is the mode chosen.
        self.automatic_settings = AutomaticSettings(column)
        layout.addWidget(self.automatic_settings)

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
        self.mode_selector.modeChanged.connect(lambda mode: self.automatic_settings.setVisible(mode is ComparisonMode.AUTOMATIC))
        self.automatic_settings.changed.connect(self._prepare_soon)
        self.upload_panel.filesChanged.connect(self._update_compare_enabled)
        # A replaced file is closed, so the results page must not keep drawing from it.
        self.upload_panel.filesChanged.connect(self.results_view.release_sources)
        self.compare_button.clicked.connect(self._on_compare)
        self.cancel_button.clicked.connect(self.jobs.cancel)

        # The comparison is prepared while the reviewer is still choosing (see `_prepare`): once the files, the mode,
        # an outline or the pages to compare have stayed put for a moment.
        self._prepare_timer = QTimer(self)
        self._prepare_timer.setSingleShot(True)
        self._prepare_timer.setInterval(_PREPARE_DELAY_MS)
        self._prepare_timer.timeout.connect(self._prepare)
        self.upload_panel.filesChanged.connect(self._prepare_soon)
        self.mode_selector.modeChanged.connect(lambda _mode: self._prepare_soon())
        self.upload_panel.original_slot.pages_picker.pagesChanged.connect(lambda _spec: self._prepare_soon())

        self.jobs.progressed.connect(self._on_progress)
        self.jobs.completed.connect(self._on_completed)
        if hasattr(self.jobs, "updated"):
            self.jobs.updated.connect(self._on_updated)
        self.jobs.failed.connect(self._on_failed)
        self.jobs.cancelled.connect(self._on_cancelled)

        # The elapsed time on the status line, so a long comparison never looks frozen.
        self._ticker = QTimer(self)
        self._ticker.setInterval(_PROGRESS_TICK_MS)
        self._ticker.timeout.connect(self._show_progress)

        # A window that stops answering is a fault: each time it is held up, how long is logged, to be found and fixed.
        self._last_beat = monotonic()
        self._heartbeat = QTimer(self)
        self._heartbeat.setInterval(_STALL_CHECK_MS)
        self._heartbeat.timeout.connect(self._beat)
        self._heartbeat.start()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._owns_jobs and not os.environ.get(NO_WARMUP_ENV):
            QTimer.singleShot(0, self.jobs.warm_up)  # imports take a while; do them while the user picks files

    def _beat(self) -> None:
        now = monotonic()
        held = now - self._last_beat - _STALL_CHECK_MS / 1000
        self._last_beat = now
        if held > _STALL_REPORT_S:
            logger.warning("window held up", extra={"seconds": round(held, 1), "page": type(self.stack.currentWidget()).__name__})

    def _update_compare_enabled(self) -> None:
        self.compare_button.setEnabled(self.upload_panel.both_loaded)

    def _set_status(self, text: str, state: str) -> None:
        self.status_label.setText(text)
        set_variant(self.status_label, "status", state)

    def _show_setup(self) -> None:
        self.results_button.setVisible(self.results_view.has_results)
        self.stack.setCurrentWidget(self._setup_page)

    def _setup_request(self) -> ComparisonRequest | None:
        """The comparison the setup page would start right now (None until both files are loaded)."""
        original, suspect = self.upload_panel.original, self.upload_panel.suspect
        if original is None or suspect is None:
            return None
        config = self.config
        language = AUTO  # found on the files, while they are compared (`app.engine.language_detect`)
        if self.mode_selector.mode is ComparisonMode.LEAFLET:
            pages = self.upload_panel.original_slot.pages_to_compare
            config = replace(config, leaflet=replace(config.leaflet, compare_pages=pages, language=language))
        else:
            config = replace(config, detection=replace(config.detection, ocr_language=language))
        if self.mode_selector.mode is ComparisonMode.AUTOMATIC:
            config = replace(config, detection=replace(config.detection, food_label=self.automatic_settings.food_label))
        return ComparisonRequest(
            original_path=original.path,
            suspect_path=suspect.path,
            mode=self.mode_selector.mode,
            config=config,
            original_geometry=self.upload_panel.original_slot.geometry,
            suspect_geometry=self.upload_panel.suspect_slot.geometry,
            original_page=self.upload_panel.original_slot.page_choice,
            suspect_page=self.upload_panel.suspect_slot.page_choice,
            **self._page_areas(),
        )

    def _page_areas(self) -> dict:
        """Leaflet mode: what was set on each page of each file, as the request carries it."""
        if self.mode_selector.mode is not ComparisonMode.LEAFLET:
            return {}
        return {
            "original_page_areas": tuple(sorted(self.upload_panel.original_slot.page_areas.items())),
            "suspect_page_areas": tuple(sorted(self.upload_panel.suspect_slot.page_areas.items())),
        }

    def _prepare_soon(self) -> None:
        """Something the comparison depends on changed: prepare it again once the reviewer pauses."""
        self._prepare_timer.start()

    def _prepare(self) -> None:
        """Have the worker start on the comparison Compare would run now -- rendering, lifting the artwork, dividing
        it into panels, aligning -- so that pressing Compare finds the work done (see `app.engine.prepared`)."""
        request = self._setup_request()
        prepare = getattr(self.jobs, "prepare", None)
        if request is None or prepare is None or self.jobs.is_busy:
            return
        prepare(request)

    def _on_compare(self) -> None:
        """Start the comparison in the worker process; the UI stays responsive meanwhile."""
        original, suspect = self.upload_panel.original, self.upload_panel.suspect
        assert original is not None and suspect is not None
        if self.jobs.is_busy:
            return
        self._swap_note = self._files_put_right(original, suspect)
        original, suspect = self.upload_panel.original, self.upload_panel.suspect
        self._prepare_timer.stop()
        request = self._setup_request()
        # A fresh compare from the setup page is what "the mode this comparison started in" means -- a
        # results-page toggle or Barcode/Braille quick switch must never move this, or there would be no way
        # back to it once switched away.
        self.results_view.set_origin_mode(request.mode)
        self._start(request, (original, suspect, request.mode), from_results=False)

    def _files_put_right(self, original, suspect) -> str:
        """Scan vs Digital compares a digital file (the Original) with a scan of it (the Suspect): the digital file's own
        words are its text, a scanner's streaks are evened out of the scan, the Braille and the barcodes are found on
        the digital file. A scan given as the Original and a digital file as the Suspect are swapped here, on the setup
        page as well, so everything after is the right way round -- and said so. ("" if nothing was swapped; whichever
        way two scans, or two digital files, are given is left as it is: there is no telling.)"""
        if self.mode_selector.mode is not ComparisonMode.SCAN_VS_DIGITAL:
            return ""
        if not original.looks_scanned() or suspect.looks_scanned():
            return ""
        self.upload_panel.swap()
        return (
            f"The files were the other way round: '{original.path.name}' is the scan, so it was compared as the Suspect, "
            f"and the digital file '{suspect.path.name}' as the Original."
        )

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

    def _on_page_change_requested(self, which: str, page: int) -> None:
        """Another page of one file picked on the results page: the same comparison, by that page. An area drawn on the
        page compared before means nothing on this one, so it is dropped."""
        request, context = self._last_request, self._job_context
        if request is None or context is None or self.jobs.is_busy:
            return
        slot = self.upload_panel.original_slot if which == "original" else self.upload_panel.suspect_slot
        slot.set_page_choice(page)
        if which == "original":
            updated = replace(request, original_page=page, original_geometry=None)
        else:
            updated = replace(request, suspect_page=page, suspect_geometry=None)
        if slot.geometry is not None:
            slot.geometry = None
            slot.cutout = None
        self._reverting_check = None
        self._reverting_field = None
        self._reverting_pages = True
        self.results_view.set_busy("Comparing again…")
        self._start(updated, context, from_results=True)

    def _on_mode_switch_requested(self, mode: ComparisonMode) -> None:
        """Barcode/Braille quick switch or Recompare from the results page: compare again in mode."""
        request, context = self._last_request, self._job_context
        if request is None or context is None or self.jobs.is_busy:
            return
        original, suspect, _old_mode = context
        self._reverting_check = None
        self._reverting_field = None
        self.upload_panel.set_mode(mode)
        self.results_view.set_busy("Comparing again…")
        self._start(replace(request, mode=mode, align_only=False), (original, suspect, mode), from_results=True)

    def _on_edit_files_requested(self, which: str) -> None:
        """Edit files from the results page: re-crop or re-lift, display aligned files in the viewer, and wait for Recompare."""
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
            original_page=self.upload_panel.original_slot.page_choice,  # (another page may have been picked to edit)
            suspect_page=self.upload_panel.suspect_slot.page_choice,
            **(
                {
                    "original_page_areas": tuple(sorted(self.upload_panel.original_slot.page_areas.items())),
                    "suspect_page_areas": tuple(sorted(self.upload_panel.suspect_slot.page_areas.items())),
                }
                if request.mode is ComparisonMode.LEAFLET
                else {}
            ),
            align_only=True,
        )
        self._reverting_check = None
        self._reverting_field = None
        self.results_view.set_busy("Aligning edited files…")
        self._start(updated, (context[0], context[1], request.mode), from_results=True)

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

    def _on_updated(self, job_id: str, outcome: ComparisonOutcome | None) -> None:
        """The rest of the result on show (Automatic's OCR findings, read after the pixel ones were shown)."""
        if job_id == self._results_job:
            self.results_view.add_rest(outcome)

    def _on_completed(self, outcome: ComparisonOutcome) -> None:
        self._results_job = getattr(self.jobs, "last_completed_job", None)
        self._set_running(False)
        original, suspect, mode = self._job_context
        is_align_only = getattr(self._pending_request, "align_only", False)
        self._last_request = replace(self._pending_request, align_only=False)
        self._reverting_pages = False
        if outcome.mode is not ComparisonMode.LEAFLET:
            # the pages the comparison used (a best match found in the worker): the viewer's zoomed-in re-render and the
            # outline editor show those pages too
            for artwork, page in ((original, outcome.original_page), (suspect, outcome.suspect_page)):
                if artwork is not None and 0 <= page < artwork.page_count:
                    artwork.select_page(page)
        if is_align_only:
            residual = f" ({outcome.alignment.residual_error_px:.2f} px residual)" if outcome.alignment is not None else ""
            self._set_status(
                f"Files updated and aligned{residual} · Click Recompare to compare",
                "idle",
            )
        else:
            self._set_status(*_describe(outcome))
        if self._swap_note and not self._job_from_results:
            outcome = replace(outcome, warnings=[self._swap_note, *outcome.warnings])
        self.results_view.show_outcome(
            outcome,
            f"{original.path.name}  vs  {suspect.path.name}  \u00b7  {MODE_DISPLAY_NAMES[mode]}"
            + ("  \u00b7  Food label" if self._last_request is not None and self._last_request.config.detection.food_label else ""),
            original=original,
            suspect=suspect,
        )
        if is_align_only:
            self.results_view.set_busy("Files aligned. Click Recompare to run comparison.", locked=False)
            prepare = getattr(self.jobs, "prepare", None)
            if prepare is not None and self._last_request is not None:
                prepare(self._last_request)
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
        if self._reverting_pages:
            self._reverting_pages = False
            self.results_view.revert_page_selectors()
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

    if outcome.symbol_comparisons:
        codes = outcome.symbol_comparisons
        failed = sum(1 for comparison in codes if not comparison.passed)
        parts.append(f"{len(codes)} code(s) read: {len(codes) - failed} match" + (f", {failed} differ" if failed else ""))
    if outcome.braille is not None:
        braille = outcome.braille
        if braille.cells_compared:
            cells, differing = braille.cells_compared, braille.cells_differing
            parts.append(f"Braille {cells} cell(s): {cells - differing} match" + (f", {differing} differ" if differing else ""))
        else:
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
