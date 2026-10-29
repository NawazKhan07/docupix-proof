"""The results page: comparison viewer beside the findings list (Build Prompt Phase 5).

Wires the two components to each other so they never disagree: choosing a
finding jumps the viewer to it, clicking a marker selects its finding, and
filtering or ignoring in the list changes which markers the viewer draws.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.engine.file_loader import LoadedArtwork
from app.engine.worker import ComparisonOutcome
from app.models.categories import CATEGORY_ORDER
from app.models.diff import SOURCE_OCR, BoundingBox, renumbered, visible_by_source, visible_clusters
from app.models.job import MODE_DISPLAY_NAMES, ComparisonMode
from app.ui.braille_panel import BrailleResultsPanel
from app.ui.comparison_viewport import ComparisonViewport
from app.ui.diff_list_panel import DiffListPanel
from app.ui.flow_layout import flow_widget
from app.ui.page_picker import PagePicker, PageSelector
from app.ui.symbol_panel import SymbolResultsPanel

logger = logging.getLogger(__name__)

# controls that use the arrow keys themselves: they keep them
_ARROW_CONTROLS = (QComboBox, QAbstractSpinBox, QLineEdit)

PANEL_MIN_WIDTH = 290


class _ToolActionShim(QPushButton):
    """Compatibility shim for tests expecting barcode_button and braille_button directly."""

    def __init__(self, view: ResultsView, mode: ComparisonMode, label: str):
        super().__init__(label, view)
        self._view = view
        self._mode = mode
        self._label = label
        self.clicked.connect(self._handle_click)
        super().setVisible(False)

    def isVisible(self) -> bool:
        return self._view.tools_button.isVisible()

    def isEnabled(self) -> bool:
        return self._view.tools_button.isEnabled() and (self._view._current_mode is not self._mode)

    def setVisible(self, visible: bool) -> None:
        pass

    def setEnabled(self, enabled: bool) -> None:
        pass

    def _handle_click(self) -> None:
        self._view._on_tool_selected(self._mode, self._label)
        self._view._on_recompare_clicked()


class ResultsView(QWidget):
    backRequested = Signal()
    editFilesRequested = Signal(str)  # "original" or "suspect": edit its artwork area, then compare again
    modeSwitchRequested = Signal(object)  # ComparisonMode: the same files, a different mode, shown instantly
    leafletPagesChanged = Signal(str)  # Leaflet: which of the original's pages to compare ("" = every page)
    pageChangeRequested = Signal(str, int)  # ("original" | "suspect", 0-based page): compare that page of that file instead

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_mode: ComparisonMode | None = None
        self._origin_mode: ComparisonMode | None = None  # the mode this comparison was actually started in
        self._outcome: ComparisonOutcome | None = None  # the last shown outcome, refiltered instantly on toggle
        self._busy_locked = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 24)
        layout.setSpacing(14)

        header = QHBoxLayout()
        self.back_button = QPushButton("←  New comparison")
        self.back_button.clicked.connect(self.backRequested)
        heading = QLabel("Comparison results")
        heading.setProperty("role", "heading")
        self.subtitle_label = QLabel("")
        self.subtitle_label.setProperty("role", "subheading")
        self.subtitle_label.setWordWrap(True)  # two long file names must not widen the window
        titles = QVBoxLayout()
        titles.setSpacing(2)
        titles.addWidget(heading)
        # The file names only; what the comparison noted (the resolution used, the area found, the languages read) is
        # one click away, not a paragraph over the results.
        names = QHBoxLayout()
        names.setSpacing(10)
        names.addWidget(self.subtitle_label, stretch=1)
        self.details_button = QPushButton("Details")
        self.details_button.setCheckable(True)
        self.details_button.setFlat(True)
        self.details_button.setVisible(False)
        self.details_button.setToolTip("What the comparison noted about these files")
        names.addWidget(self.details_button, alignment=Qt.AlignmentFlag.AlignTop)
        titles.addLayout(names)
        self.note_label = QLabel("")
        self.note_label.setProperty("secondary", "true")
        self.note_label.setWordWrap(True)
        self.note_label.setVisible(False)
        self.details_button.toggled.connect(self.note_label.setVisible)
        titles.addWidget(self.note_label)
        header.addWidget(self.back_button, alignment=Qt.AlignmentFlag.AlignTop)
        header.addSpacing(8)
        header.addLayout(titles, stretch=3)
        # Scan vs Digital finds both graphics and text changes either way; these two just filter which are shown,
        # so ticking either is instant -- nothing is compared again (see `_refilter_scan_clusters`).
        self.graphics_check = QCheckBox("Also compare graphics")
        self.graphics_check.setToolTip(
            "Scan vs Digital compares text, numbers and barcode bars. Tick this to compare logos and other artwork too "
            "(a scan can show false differences there)."
        )
        self.graphics_check.setVisible(False)
        self.graphics_check.toggled.connect(self._on_graphics_toggled)
        self.text_check = QCheckBox("Compare text (OCR)")
        self.text_check.setToolTip(
            "Untick to skip reading and comparing text: a pure visual comparison (graphics, if that is also ticked, "
            "and barcode bars as marks) -- useful for a scan too soft or skewed for OCR to be worth it."
        )
        self.text_check.setVisible(False)
        self.text_check.toggled.connect(self._on_text_toggled)
        # Leaflet and Automatic: both strategies are always run -- reading and comparing the words (OCR), and comparing
        # the pictures pixel by pixel -- and these two show or hide each one's findings at once, nothing compared again.
        # (Automatic opens on its pixel findings; its OCR ones are there for the ticking.)
        self.leaflet_text_check = QCheckBox("OCR findings")
        self.leaflet_text_check.setToolTip(
            "Show the changes found by reading and comparing the words, word for word."
        )
        self.leaflet_text_check.setVisible(False)
        self.leaflet_text_check.toggled.connect(self._on_leaflet_sources_toggled)
        self.leaflet_visual_check = QCheckBox("Pixel findings")
        self.leaflet_visual_check.setToolTip(
            "Show the changes found by comparing the pictures pixel by pixel, marking exactly the areas that differ."
        )
        self.leaflet_visual_check.setVisible(False)
        self.leaflet_visual_check.toggled.connect(self._on_leaflet_sources_toggled)
        # Which of the leaflet's pages to compare: a list to tick, next to the other file controls.
        self.leaflet_pages_picker = PagePicker()
        self.leaflet_pages_picker.setVisible(False)
        self.leaflet_pages_picker.pagesChanged.connect(self._on_leaflet_pages_chosen)
        # Every other mode: which page of each file of several was compared -- pick another to compare that one.
        self.original_page_selector = PageSelector("Original page", with_best_match=False)
        self.suspect_page_selector = PageSelector("Suspect page", with_best_match=False)
        for which, selector in (("original", self.original_page_selector), ("suspect", self.suspect_page_selector)):
            selector.setVisible(False)
            selector.pageChosen.connect(lambda page, which=which: self._on_page_chosen(which, page))
        # A quick way to switch to Barcode or Braille mode on these same files without going back to the setup
        # page: a Tools dropdown offering Barcode and Braille, and a Recompare button appearing when picked.
        self.tools_button = QPushButton("Tools")
        self.tools_button.setProperty("variant", "segment")
        self.tools_button.setToolTip("Select a specialized comparison mode (Barcode, Braille).")
        self.tools_button.setVisible(False)
        self.tools_menu = QMenu(self.tools_button)
        self.automatic_action = self.tools_menu.addAction("Automatic")
        self.automatic_action.triggered.connect(
            lambda: self._on_tool_selected(self._origin_mode or ComparisonMode.AUTOMATIC, "Tools")
        )
        self.barcode_action = self.tools_menu.addAction("Barcode")
        self.barcode_action.triggered.connect(lambda: self._on_tool_selected(ComparisonMode.BARCODE, "Barcode"))
        self.braille_action = self.tools_menu.addAction("Braille")
        self.braille_action.triggered.connect(lambda: self._on_tool_selected(ComparisonMode.BRAILLE, "Braille"))
        self.tools_button.setMenu(self.tools_menu)

        self.recompare_button = QPushButton("Recompare")
        self.recompare_button.setProperty("variant", "primary")
        self.recompare_button.setToolTip("Compare these same files again in the current or selected mode.")
        self.recompare_button.setVisible(True)
        self.recompare_button.clicked.connect(self._on_recompare_clicked)
        self._selected_tool_mode: ComparisonMode | None = None

        self.barcode_button = _ToolActionShim(self, ComparisonMode.BARCODE, "Barcode")
        self.braille_button = _ToolActionShim(self, ComparisonMode.BRAILLE, "Braille")
        # A way back from Barcode/Braille to whatever mode this comparison actually started in -- switching to
        # one of those two must not strand the reviewer with no way back except returning to the setup page
        # and comparing again from scratch.
        self.origin_mode_button = QPushButton("")
        self.origin_mode_button.setProperty("variant", "segment")
        self.origin_mode_button.setToolTip("Compare these same files again, back in the mode this comparison started in.")
        self.origin_mode_button.setVisible(False)
        self.origin_mode_button.clicked.connect(lambda: self.modeSwitchRequested.emit(self._origin_mode))
        # Re-crop or re-lift either file's artwork area without going back to the setup page.
        self.edit_files_button = QPushButton("Edit files…")
        edit_menu = QMenu(self.edit_files_button)
        edit_menu.addAction("Edit Original…", lambda: self.editFilesRequested.emit("original"))
        edit_menu.addAction("Edit Suspect…", lambda: self.editFilesRequested.emit("suspect"))
        self.edit_files_button.setMenu(edit_menu)
        self.busy_label = QLabel("")
        self.busy_label.setProperty("status", "idle")
        self.busy_label.setVisible(False)
        # The controls wrap onto another line when the window is narrow, rather than force it wider than the screen.
        controls_widget, controls = flow_widget(self, spacing=8, align_right=True)
        for widget in (
            self.busy_label,
            self.text_check,
            self.graphics_check,
            self.leaflet_text_check,
            self.leaflet_visual_check,
            self.origin_mode_button,
            self.tools_button,
            self.recompare_button,
            self.edit_files_button,
            self.leaflet_pages_picker,
            self.original_page_selector,
            self.suspect_page_selector,
        ):
            controls.addWidget(widget)
        header.addWidget(controls_widget, stretch=2, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addLayout(header)

        self.viewport = ComparisonViewport(self)
        self.panel = DiffListPanel(self)
        # These two belong to the tab widget below, only when a mode calls for them; created
        # without a parent so that, until then, they are not stray widgets drawn over this page.
        self.symbol_panel = SymbolResultsPanel()  # Barcode mode (FR-10.2)
        self.braille_panel = BrailleResultsPanel()  # Braille mode (FR-11.1)

        # The findings list is always there; a mode that decodes something also
        # gets the panel for what it decoded, beside it.
        self.side = QTabWidget(self)
        self.side.setDocumentMode(True)
        self.side.setMinimumWidth(PANEL_MIN_WIDTH)
        self.side.addTab(self.panel, "Findings")

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self.viewport)
        splitter.addWidget(self.side)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([900, PANEL_MIN_WIDTH + 20])
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, stretch=1)

        self.panel.visibilityChanged.connect(self._sync_markers)
        self.panel.clusterSelected.connect(self._on_finding_selected)
        self.symbol_panel.symbolSelected.connect(self.viewport.highlight_box)
        self.braille_panel.regionSelected.connect(self.viewport.highlight_box)
        self.viewport.markerClicked.connect(self._on_marker_clicked)
        self.viewport.backgroundClicked.connect(self.deselect)
        self.panel.backgroundClicked.connect(self.deselect)
        self.viewport.mergeRegionSelected.connect(self._on_merge_region)
        self.viewport.undoMergeRequested.connect(self._on_undo_merge)
        self.viewport.eraseRegionSelected.connect(self._on_erase_region)
        self.viewport.undoEraseRequested.connect(self._on_undo_erase)
        # each area erased, latest last: (what the pictures had there, the findings before, every finding's number)
        self._erasures: list[tuple] = []

    def set_origin_mode(self, mode: ComparisonMode | None) -> None:
        """The mode this comparison actually started in (a fresh "Compare" from the setup page) -- not touched
        by a results-page toggle or a Barcode/Braille quick switch, so there is always a way back to it."""
        self._origin_mode = mode

    def show_outcome(
        self,
        outcome: ComparisonOutcome,
        subtitle: str,
        original: LoadedArtwork | None = None,
        suspect: LoadedArtwork | None = None,
    ) -> None:
        """Show a result. Passing the loaded files lets the viewer re-render PDFs
        from vector data when zoomed in (FR-9.1)."""
        self._outcome = outcome
        self.subtitle_label.setText(subtitle)
        self.note_label.setText("  ".join(outcome.notes))
        self.details_button.setVisible(bool(outcome.notes))
        self.note_label.setVisible(bool(outcome.notes) and self.details_button.isChecked())
        self.has_results = True
        scan_mode = outcome.mode is ComparisonMode.SCAN_VS_DIGITAL
        self.graphics_check.blockSignals(True)
        self.graphics_check.setChecked(bool(outcome.graphics_compared))
        self.graphics_check.blockSignals(False)
        self.graphics_check.setVisible(scan_mode)
        self.text_check.blockSignals(True)
        self.text_check.setChecked(bool(outcome.text_compared))
        self.text_check.blockSignals(False)
        self.text_check.setVisible(scan_mode)
        leaflet_mode = outcome.mode is ComparisonMode.LEAFLET
        both_ways = outcome.mode in (ComparisonMode.LEAFLET, ComparisonMode.AUTOMATIC)  # OCR and pixel findings, toggled
        self.leaflet_text_check.blockSignals(True)
        self.leaflet_text_check.setChecked(bool(outcome.text_compared))
        self.leaflet_text_check.blockSignals(False)
        self.leaflet_text_check.setVisible(both_ways)
        self._set_ocr_check_ready(not outcome.ocr_pending, outcome.mode)
        self.leaflet_visual_check.blockSignals(True)
        self.leaflet_visual_check.setChecked(bool(outcome.graphics_compared))
        self.leaflet_visual_check.blockSignals(False)
        self.leaflet_visual_check.setVisible(both_ways)
        # how each finding was found, as kinds to pick in the findings list: where both ways are run
        self.panel.set_source_kinds(both_ways or outcome.mode is ComparisonMode.SCAN_VS_DIGITAL)
        # a leaflet is compared for its words, colours and graphics only: it has no barcode or Braille to list
        self.panel.set_categories(("text", "colour", "graphics") if leaflet_mode else CATEGORY_ORDER)
        if leaflet_mode:
            self.leaflet_pages_picker.set_page_count(original.page_count if original is not None else 0)
            self.leaflet_pages_picker.set_spec(outcome.compared_pages)
        self.leaflet_pages_picker.setVisible(leaflet_mode and self.leaflet_pages_picker.page_count > 1)
        for selector, artwork, page in (
            (self.original_page_selector, original, outcome.original_page),
            (self.suspect_page_selector, suspect, outcome.suspect_page),
        ):
            count = artwork.page_count if artwork is not None else 0
            selector.set_page_count(count)
            selector.set_choice(page)
            selector.setVisible(not leaflet_mode and count > 1)
        # Barcode/Braille are Automatic's own quick-checks (and each other's): offered from Automatic itself or
        # from whichever of the two is currently shown, never from a mode with its own specific comparison
        # (Scan vs Digital, Series) or Leaflet (which pairs and compares its own pages, on its own screen).
        switchable = outcome.mode in (ComparisonMode.AUTOMATIC, ComparisonMode.BARCODE, ComparisonMode.BRAILLE)
        self.tools_button.setVisible(switchable)
        self.recompare_button.setVisible(True)
        self._selected_tool_mode = None
        if outcome.mode is ComparisonMode.AUTOMATIC:
            self.tools_button.setText("Tools")
        elif outcome.mode is ComparisonMode.BARCODE:
            self.tools_button.setText("Barcode")
        elif outcome.mode is ComparisonMode.BRAILLE:
            self.tools_button.setText("Braille")
        self.barcode_button.setVisible(switchable)
        self.braille_button.setVisible(switchable)
        origin_label = MODE_DISPLAY_NAMES.get(self._origin_mode, "Automatic") if self._origin_mode else "Automatic"
        self.automatic_action.setText(origin_label)
        self.origin_mode_button.setVisible(False)
        self._current_mode = outcome.mode
        self.set_busy("")
        ink_pair = (
            (outcome.ink_original, outcome.ink_suspect)
            if outcome.ink_original is not None and outcome.ink_suspect is not None
            else None
        )
        # Flicker/Overlay/Grayscale blend the two panes pixel-for-pixel: only sound when they are the same picture, one
        # warped onto the other (every mode but Series, whose suspect pane is a whole sheet beside one small artwork).
        same_size = outcome.aligned_original.shape[:2] == outcome.aligned_suspect.shape[:2]
        self.viewport.set_composite_modes_enabled(same_size)
        self.viewport.set_images(outcome.aligned_original, outcome.aligned_suspect, ink_pair)
        if leaflet_mode:
            # A leaflet is shown as its pages stacked (each pair as aligned): a zoomed-in re-render of one PDF page
            # would not line up with that picture, so it is resampled like an image.
            self.viewport.set_sources(None, None, None)
        else:
            self.viewport.set_sources(
                original, suspect, outcome.alignment.transform, outcome.block_alignments, (outcome.original_dpi, outcome.suspect_dpi)
            )
        self.viewport.set_compared_area(outcome.compared_outline)
        self.viewport.set_block_outlines(outcome.block_outlines or None)
        self.viewport.clear_highlight()
        self.panel.set_result(outcome.clusters, outcome.alignment, outcome.warnings)
        self.viewport.set_merge_mode(False)
        self.viewport.set_can_undo_merge(False)
        self.viewport.set_hide_highlights(False)
        self._erasures = []
        self.viewport.set_erase_available(leaflet_mode)
        self.viewport.show_merge_notice("")
        self._show_mode_panels(outcome)
        self._sync_markers()

    def _show_mode_panels(self, outcome: ComparisonOutcome) -> None:
        """Add the Symbols / Braille tab when the mode produced something to show in it."""
        for panel in (self.symbol_panel, self.braille_panel):
            index = self.side.indexOf(panel)
            if index >= 0:
                self.side.removeTab(index)
            panel.hide()  # removing a tab does not hide its page

        chosen: QWidget = self.panel
        if outcome.mode is ComparisonMode.BARCODE or outcome.symbol_comparisons:
            self.symbol_panel.set_comparisons(outcome.symbol_comparisons)
            self.side.addTab(self.symbol_panel, "Symbols")
            chosen = self.symbol_panel
        if outcome.mode is ComparisonMode.BRAILLE or outcome.braille is not None:
            self.braille_panel.set_comparison(outcome.braille)
            self.braille_panel.set_differences([c for c in outcome.clusters if c.category == "braille"])
            self.side.addTab(self.braille_panel, "Braille")
            chosen = self.braille_panel

        self.side.tabBar().setVisible(self.side.count() > 1)  # a lone tab is just noise
        self.side.setCurrentWidget(chosen)

    has_results = False

    def set_busy(self, text: str, locked: bool = True) -> None:
        """Say what a re-comparison is doing, or how it ended (empty for nothing to say); the controls that would
        start another one are locked while it runs."""
        self.busy_label.setText(text)
        self.busy_label.setVisible(bool(text))
        self._busy_locked = bool(text and locked)
        idle = not self._busy_locked
        self.graphics_check.setEnabled(idle)
        self.text_check.setEnabled(idle)
        self.tools_button.setEnabled(idle)
        self.recompare_button.setEnabled(idle)
        self.barcode_button.setEnabled(idle and self._current_mode is not ComparisonMode.BARCODE)
        self.braille_button.setEnabled(idle and self._current_mode is not ComparisonMode.BRAILLE)
        self.origin_mode_button.setEnabled(idle)
        self.edit_files_button.setEnabled(idle)
        self.leaflet_pages_picker.setEnabled(idle)
        self.original_page_selector.setEnabled(idle)
        self.suspect_page_selector.setEnabled(idle)

    def select_tool(self, mode: ComparisonMode) -> None:
        """Select a tool mode ('Barcode', 'Braille', or 'Automatic') from the Tools dropdown."""
        if mode in (ComparisonMode.BARCODE, ComparisonMode.BRAILLE):
            label = "Barcode" if mode is ComparisonMode.BARCODE else "Braille"
        else:
            label = "Tools"
        self._on_tool_selected(mode, label)

    def _on_tool_selected(self, mode: ComparisonMode, label: str) -> None:
        if mode == self._current_mode and label == "Tools":
            self._selected_tool_mode = None
        else:
            self._selected_tool_mode = mode
        self.tools_button.setText(label)
        self.recompare_button.setVisible(True)

    def _on_recompare_clicked(self) -> None:
        target_mode = self._selected_tool_mode if self._selected_tool_mode is not None else self._current_mode
        if target_mode is not None:
            self.modeSwitchRequested.emit(target_mode)

    def _on_graphics_toggled(self, checked: bool) -> None:
        self._refilter_scan_clusters(graphics_wanted=checked, text_wanted=self.text_check.isChecked())

    def _on_text_toggled(self, checked: bool) -> None:
        self._refilter_scan_clusters(graphics_wanted=self.graphics_check.isChecked(), text_wanted=checked)

    def _refilter_scan_clusters(self, *, graphics_wanted: bool, text_wanted: bool) -> None:
        """Scan vs Digital finds both graphics and text changes either way (`app.engine.worker.run_comparison`):
        these two checkboxes only filter which of `outcome.all_clusters` are shown, so this is instant --
        nothing is compared again, unlike every other results-page control."""
        if self._outcome is None:
            return
        self._outcome = replace(
            self._outcome,
            clusters=renumbered(
                visible_clusters(self._outcome.all_clusters, text_wanted=text_wanted, graphics_wanted=graphics_wanted),
                self._outcome.all_clusters,
            ),
            graphics_compared=graphics_wanted,
            text_compared=text_wanted,
        )
        self.panel.set_result(self._outcome.clusters, self._outcome.alignment, self._outcome.warnings)
        self._sync_markers()

    def _set_ocr_check_ready(self, ready: bool, mode: ComparisonMode | None) -> None:
        """The OCR findings' checkbox, usable once they are there (Automatic sends its pixel findings first)."""
        self.leaflet_text_check.setEnabled(ready)
        if not ready:
            self.leaflet_text_check.setToolTip("Reading the words… the OCR findings can be shown in a moment.")
        elif mode is ComparisonMode.AUTOMATIC:
            self.leaflet_text_check.setToolTip("Show the words spelt differently in the suspect (read and compared word for word).")
        else:
            self.leaflet_text_check.setToolTip("Show the changes found by reading and comparing the words, word for word.")

    def add_rest(self, whole: ComparisonOutcome | None) -> None:
        """The rest of the result on show, sent after it (`ComparisonOutcome.ocr_pending`): its OCR findings added to
        what is shown -- the findings already there kept as they are (and whatever was done with them: ignored, merged)
        -- or, where they could not be made (`None`), the result left as it was."""
        if self._outcome is None or not self._outcome.ocr_pending:
            return
        if whole is None:
            self._outcome = replace(self._outcome, ocr_pending=False)
            self._set_ocr_check_ready(True, self._outcome.mode)
            self.leaflet_text_check.setToolTip("The words could not be read, so there are no OCR findings.")
            self.leaflet_text_check.setEnabled(False)
            return
        read = [cluster for cluster in (whole.all_clusters or whole.clusters) if cluster.sources == {SOURCE_OCR}]
        everything = list(self._outcome.all_clusters or self._outcome.clusters) + read
        ocr, pixels = self.leaflet_text_check.isChecked(), self.leaflet_visual_check.isChecked()
        self._outcome = replace(
            self._outcome,
            all_clusters=everything,
            clusters=renumbered(visible_by_source(everything, ocr_wanted=ocr, pixels_wanted=pixels), everything),
            candidates=self._outcome.candidates + [m for c in read for m in c.members],
            warnings=self._outcome.warnings + [w for w in whole.warnings if w not in self._outcome.warnings],
            ocr_pending=False,
        )
        self._set_ocr_check_ready(True, self._outcome.mode)
        self.panel.set_result(self._outcome.clusters, self._outcome.alignment, self._outcome.warnings)
        self._sync_markers()

    def _on_leaflet_sources_toggled(self, _checked: bool) -> None:
        """Leaflet's or Automatic's OCR / pixel findings shown or hidden: a filter over everything both strategies found
        (`outcome.all_clusters`), so it is instant -- nothing is compared again."""
        if self._outcome is None:
            return
        ocr, pixels = self.leaflet_text_check.isChecked(), self.leaflet_visual_check.isChecked()
        self._outcome = replace(
            self._outcome,
            clusters=renumbered(
                visible_by_source(self._outcome.all_clusters or self._outcome.clusters, ocr_wanted=ocr, pixels_wanted=pixels),
                self._outcome.all_clusters or [],
            ),
            text_compared=ocr,
            graphics_compared=pixels,
        )
        self.panel.set_result(self._outcome.clusters, self._outcome.alignment, self._outcome.warnings)
        self._sync_markers()

    def revert_leaflet_pages_field(self, text: str) -> None:
        """A page choice whose comparison did not finish: the picker goes back to what the results on show used."""
        self.leaflet_pages_picker.set_spec(text)

    def _on_leaflet_pages_chosen(self, spec: str) -> None:
        if self._busy_locked:
            return
        self.leafletPagesChanged.emit(spec)

    def _on_page_chosen(self, which: str, page: int | None) -> None:
        if self._busy_locked or page is None:
            return
        self.pageChangeRequested.emit(which, page)

    def revert_page_selectors(self) -> None:
        """A page choice whose comparison did not finish: the selectors go back to the pages the results on show used."""
        if self._outcome is not None:
            self.original_page_selector.set_choice(self._outcome.original_page)
            self.suspect_page_selector.set_choice(self._outcome.suspect_page)
    def release_sources(self) -> None:
        """Stop holding the loaded files (they are closed when replaced)."""
        self.viewport.release_sources()

    def _sync_markers(self) -> None:
        self.viewport.set_marker_clusters(self.panel.marker_clusters(), by_source=self.panel.colour_by_source)

    def _on_finding_selected(self, cluster_id: int | None) -> None:
        if cluster_id is None:  # the already-selected card was clicked again: deselect, without moving the view
            self.viewport.set_selected(None)
            return
        cluster = self.panel.cluster(cluster_id)
        if cluster is not None:
            self.viewport.focus_cluster(cluster)

    def _on_marker_clicked(self, cluster_id: int) -> None:
        if cluster_id == self.panel._selected_id:  # the selected finding clicked again: deselect it
            self.deselect()
            return
        self.viewport.set_selected(cluster_id)
        self.panel.select(cluster_id)

    def deselect(self) -> None:
        """No finding selected (the reviewer clicked the picture where there is none, or anywhere else on the page)."""
        if self.panel._selected_id is None:
            return
        self.panel.select(None)
        self.viewport.set_selected(None)
        self.viewport.clear_highlight()

    def mousePressEvent(self, event) -> None:
        # a click on the page that nothing else took (its heading, the space between the panels): deselect
        self.deselect()
        super().mousePressEvent(event)

    # -- stepping through the findings with the arrow keys --------------------------------------------------------------
    def step_finding(self, step: int) -> bool:
        """With a finding selected, select the one `step` places below (+1) or above (-1) it in the findings list -- the
        list's own order, whatever is filtered out -- and take the view to it. False if there is none that way (or none
        selected): nothing changes."""
        shown = self.panel.visible_clusters()
        ids = [cluster.id for cluster in shown]
        current = self.panel._selected_id
        if current not in ids:
            return False
        index = ids.index(current) + step
        if not 0 <= index < len(ids):
            return False
        self.panel.select(ids[index])
        self._on_finding_selected(ids[index])
        return True

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QApplication.instance().installEventFilter(self)

    def hideEvent(self, event) -> None:
        QApplication.instance().removeEventFilter(self)
        super().hideEvent(event)

    def eventFilter(self, watched, event) -> bool:
        """Up / Down, while a finding is selected on this page -- or a code, a Braille line or difference in the list on
        show beside the viewer -- step to the one above / below it, whichever part of the page has the keyboard (the
        viewer or the list, which would otherwise scroll), except a control that uses the arrows itself (a drop-down, a
        number box) or another window (a dialog) in front."""
        if event.type() != QEvent.Type.KeyPress or event.key() not in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            return super().eventFilter(watched, event)  # (every event of the application passes here: the rest is for arrows)
        side = self.side.currentWidget()
        stepping_side = side in (self.symbol_panel, self.braille_panel) and side.has_selection
        if (
            not (event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier)
            and (self.panel._selected_id is not None or stepping_side)
            and self.isVisible()
            and QApplication.activeModalWidget() is None
            and isinstance(watched, QWidget)
            and watched.window() is self.window()
            and not isinstance(QApplication.focusWidget(), _ARROW_CONTROLS)
        ):
            delta = 1 if event.key() == Qt.Key.Key_Down else -1
            if stepping_side:
                side.step(delta)  # (the Symbols or Braille list on show: its codes, its lines and differences)
            else:
                self.step_finding(delta)
            return True  # handled (at the first or last finding, the arrow simply does nothing)
        return super().eventFilter(watched, event)

    # -- merging findings ------------------------------------------------------------
    def _on_merge_region(self, region) -> None:
        """The reviewer drew a rectangle: the findings in it (mostly inside it) become one."""
        inside = [
            cluster
            for cluster in self.panel.marker_clusters()
            if _mostly_inside(cluster.bbox_in_original, region)
        ]
        if len(inside) < 2:
            self.viewport.show_merge_notice(
                "That rectangle holds "
                + ("one finding" if inside else "no findings")
                + ": drag round two or more to merge them."
            )
            return
        merged = self.panel.merge([cluster.id for cluster in inside])
        self.viewport.set_can_undo_merge(self.panel.can_undo_merge)
        self._sync_markers()
        if merged is not None:
            self.viewport.set_selected(merged.id)
            self.viewport.show_merge_notice(f"Merged {len(inside)} findings into #{merged.id}.")

    def _on_erase_region(self, region: BoundingBox) -> None:
        """The reviewer erased an area (Leaflet): it is cleared from both pictures and every finding in it is removed,
        at once -- nothing is compared again."""
        if self._outcome is None:
            return
        shown = self.panel.clusters
        known = shown + [c for c in (self._outcome.all_clusters or self._outcome.clusters) if not any(c is k for k in shown)]
        patches = self.viewport.erase_region(region)
        if patches is None:
            self.viewport.show_merge_notice("That rectangle is off the pages: nothing was erased.")
            return
        doomed = {id(c) for c in known if _erased_by(c.bbox_in_original, region)}
        self._erasures.append(
            (patches, shown, list(self._outcome.clusters), list(self._outcome.all_clusters), [(c, c.id) for c in known])
        )

        def kept(clusters: list) -> list:
            return [c for c in clusters if id(c) not in doomed]

        self._outcome = replace(self._outcome, clusters=kept(self._outcome.clusters), all_clusters=kept(self._outcome.all_clusters))
        remaining = renumbered(kept(shown), self._outcome.all_clusters)
        self.panel.replace_clusters(remaining)
        self.viewport.set_selected(None)
        self._sync_markers()
        # Erasing is the reviewer's decision about what is not part of the leaflet: leave a trace.
        logger.info(
            "area erased",
            extra={"area": [round(v) for v in (region.x0, region.y0, region.x1, region.y1)], "findings_removed": len(doomed)},
        )
        self.viewport.set_can_undo_erase(True)
        removed = len(doomed)
        self.viewport.show_merge_notice(
            f"Erased the area and {removed} finding{'s' if removed != 1 else ''} in it." if removed else "Erased the area (no finding was in it)."
        )

    def _on_undo_erase(self) -> None:
        if not self._erasures or self._outcome is None:
            return
        patches, shown, clusters, all_clusters, numbers = self._erasures.pop()
        self.viewport.restore_region(patches)
        for cluster, number in numbers:
            cluster.id = number
        self._outcome = replace(self._outcome, clusters=clusters, all_clusters=all_clusters)
        self.panel.replace_clusters(shown)
        self.viewport.set_selected(None)
        self._sync_markers()
        logger.info("area erase undone")
        self.viewport.set_can_undo_erase(bool(self._erasures))
        self.viewport.show_merge_notice("Put back the last area erased.")

    def _on_undo_merge(self) -> None:
        if self.panel.undo_merge():
            self.viewport.set_can_undo_merge(self.panel.can_undo_merge)
            self.viewport.set_selected(self.panel._selected_id)
            self._sync_markers()
            self.viewport.show_merge_notice("Split the last merge.")


_MERGE_MIN_INSIDE = 0.5  # a finding is in the rectangle if at least this much of it is


def _mostly_inside(box, region) -> bool:
    """Whether at least half of `box` (a finding) lies inside `region` (the rectangle drawn)."""
    if box.area <= 0:
        return region.x0 <= box.center[0] <= region.x1 and region.y0 <= box.center[1] <= region.y1
    return box.intersection_area(region) >= _MERGE_MIN_INSIDE * box.area


_ERASED_SHARE = 0.5  # a finding is erased with an area when its centre is in it, or at least this share of it is


def _erased_by(box: BoundingBox, region: BoundingBox) -> bool:
    cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
    if region.x0 <= cx <= region.x1 and region.y0 <= cy <= region.y1:
        return True
    width = min(box.x1, region.x1) - max(box.x0, region.x0)
    height = min(box.y1, region.y1) - max(box.y0, region.y0)
    area = max(0.0, box.x1 - box.x0) * max(0.0, box.y1 - box.y0)
    return width > 0 and height > 0 and area > 0 and width * height >= _ERASED_SHARE * area
