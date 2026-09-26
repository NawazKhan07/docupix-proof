"""The results page: comparison viewer beside the findings list (Build Prompt Phase 5).

Wires the two components to each other so they never disagree: choosing a
finding jumps the viewer to it, clicking a marker selects its finding, and
filtering or ignoring in the list changes which markers the viewer draws.
"""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.engine.file_loader import LoadedArtwork
from app.engine.worker import ComparisonOutcome
from app.models.diff import visible_clusters
from app.models.job import MODE_DISPLAY_NAMES, ComparisonMode
from app.ui.braille_panel import BrailleResultsPanel
from app.ui.comparison_viewport import ComparisonViewport
from app.ui.diff_list_panel import DiffListPanel
from app.ui.flow_layout import flow_widget
from app.ui.page_picker import PagePicker
from app.ui.symbol_panel import SymbolResultsPanel

PANEL_MIN_WIDTH = 290


class ResultsView(QWidget):
    backRequested = Signal()
    editFilesRequested = Signal(str)  # "original" or "suspect": edit its artwork area, then compare again
    modeSwitchRequested = Signal(object)  # ComparisonMode: the same files, a different mode, shown instantly
    leafletTextToggled = Signal(bool)  # Leaflet: read and compare the words on each page (or not)
    leafletVisualToggled = Signal(bool)  # Leaflet: pure pixel-by-pixel comparison (or not)
    leafletPagesChanged = Signal(str)  # Leaflet: which of the original's pages to compare ("" = every page)

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
        titles.addWidget(self.subtitle_label)
        self.note_label = QLabel("")
        self.note_label.setProperty("secondary", "true")
        self.note_label.setWordWrap(True)
        self.note_label.setVisible(False)
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
        # Leaflet's own two independent strategies: which words differ, and/or exactly which pixels do. At
        # least one must stay on, so each disables itself while it is the only one left ticked.
        self.leaflet_text_check = QCheckBox("Compare text")
        self.leaflet_text_check.setToolTip(
            "Read and compare the words on each page. On by default -- a leaflet's pages are what they say."
        )
        self.leaflet_text_check.setVisible(False)
        self.leaflet_text_check.toggled.connect(self._on_leaflet_text_toggled)
        self.leaflet_visual_check = QCheckBox("Compare visually (pixel by pixel)")
        self.leaflet_visual_check.setToolTip(
            "A pure pixel-by-pixel comparison marking the exact areas that differ, whatever they say."
        )
        self.leaflet_visual_check.setVisible(False)
        self.leaflet_visual_check.toggled.connect(self._on_leaflet_visual_toggled)
        # Which of the leaflet's pages to compare: a list to tick, next to the other file controls.
        self.leaflet_pages_picker = PagePicker()
        self.leaflet_pages_picker.setVisible(False)
        self.leaflet_pages_picker.pagesChanged.connect(self._on_leaflet_pages_chosen)
        # A quick way to switch to Barcode or Braille mode on these same files without going back to the setup
        # page: the one currently shown is disabled (nothing to switch to), the other re-compares instantly.
        self.barcode_button = QPushButton("Barcode")
        self.barcode_button.setProperty("variant", "segment")
        self.barcode_button.setToolTip("Compare these same files again, restricted to barcode/Pharmacode/QR symbols.")
        self.barcode_button.setVisible(False)
        self.barcode_button.clicked.connect(lambda: self.modeSwitchRequested.emit(ComparisonMode.BARCODE))
        self.braille_button = QPushButton("Braille")
        self.braille_button.setProperty("variant", "segment")
        self.braille_button.setToolTip("Compare these same files again, restricted to the Braille dot matrix.")
        self.braille_button.setVisible(False)
        self.braille_button.clicked.connect(lambda: self.modeSwitchRequested.emit(ComparisonMode.BRAILLE))
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
            self.barcode_button,
            self.braille_button,
            self.edit_files_button,
            self.leaflet_pages_picker,
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
        self.viewport.mergeRegionSelected.connect(self._on_merge_region)
        self.viewport.undoMergeRequested.connect(self._on_undo_merge)

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
        self.note_label.setVisible(bool(outcome.notes))
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
        self.leaflet_text_check.blockSignals(True)
        self.leaflet_text_check.setChecked(bool(outcome.text_compared))
        self.leaflet_text_check.blockSignals(False)
        self.leaflet_text_check.setVisible(leaflet_mode)
        self.leaflet_visual_check.blockSignals(True)
        self.leaflet_visual_check.setChecked(bool(outcome.graphics_compared))
        self.leaflet_visual_check.blockSignals(False)
        self.leaflet_visual_check.setVisible(leaflet_mode)
        self._refresh_leaflet_strategy_locks()
        if leaflet_mode:
            self.leaflet_pages_picker.set_page_count(original.page_count if original is not None else 0)
            self.leaflet_pages_picker.set_spec(outcome.compared_pages)
        self.leaflet_pages_picker.setVisible(leaflet_mode and self.leaflet_pages_picker.page_count > 1)
        # Barcode/Braille are Automatic's own quick-checks (and each other's): offered from Automatic itself or
        # from whichever of the two is currently shown, never from a mode with its own specific comparison
        # (Scan vs Digital, Series) or Leaflet (which pairs and compares its own pages, on its own screen).
        switchable = outcome.mode in (ComparisonMode.AUTOMATIC, ComparisonMode.BARCODE, ComparisonMode.BRAILLE)
        self.barcode_button.setVisible(switchable)
        self.braille_button.setVisible(switchable)
        # Barcode/Braille already have their own dedicated buttons; the origin button is only for getting back
        # to whatever OTHER mode (Automatic, Scan vs Digital, Series) this comparison actually started in.
        origin_has_own_button = self._origin_mode in (ComparisonMode.BARCODE, ComparisonMode.BRAILLE, ComparisonMode.LEAFLET, None)
        show_origin = switchable and not origin_has_own_button and self._origin_mode is not outcome.mode
        if show_origin:
            self.origin_mode_button.setText(MODE_DISPLAY_NAMES[self._origin_mode])
        self.origin_mode_button.setVisible(show_origin)
        # A leaflet's pages have no crop/lift area of their own to re-edit (every page is compared in full).
        self.edit_files_button.setVisible(not leaflet_mode)
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
            self.viewport.set_sources(original, suspect, outcome.alignment.transform, outcome.block_alignments)
        self.viewport.set_compared_area(outcome.compared_outline)
        self.viewport.set_block_outlines(outcome.block_outlines or None)
        self.viewport.clear_highlight()
        self.panel.set_result(outcome.clusters, outcome.alignment, outcome.warnings)
        self.viewport.set_merge_mode(False)
        self.viewport.set_can_undo_merge(False)
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
        self.barcode_button.setEnabled(idle and self._current_mode is not ComparisonMode.BARCODE)
        self.braille_button.setEnabled(idle and self._current_mode is not ComparisonMode.BRAILLE)
        self.origin_mode_button.setEnabled(idle)
        self.edit_files_button.setEnabled(idle)
        self.leaflet_pages_picker.setEnabled(idle)
        self._refresh_leaflet_strategy_locks()

    def _refresh_leaflet_strategy_locks(self) -> None:
        """At least one of Leaflet's two strategies must stay on: whichever is the only one ticked disables
        itself, so un-ticking it is not even possible, rather than silently doing nothing or leaving neither on."""
        idle = not self._busy_locked
        text_only = self.leaflet_text_check.isChecked() and not self.leaflet_visual_check.isChecked()
        visual_only = self.leaflet_visual_check.isChecked() and not self.leaflet_text_check.isChecked()
        self.leaflet_text_check.setEnabled(idle and not text_only)
        self.leaflet_visual_check.setEnabled(idle and not visual_only)

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
            clusters=visible_clusters(self._outcome.all_clusters, text_wanted=text_wanted, graphics_wanted=graphics_wanted),
            graphics_compared=graphics_wanted,
            text_compared=text_wanted,
        )
        self.panel.set_result(self._outcome.clusters, self._outcome.alignment, self._outcome.warnings)
        self._sync_markers()

    def _on_leaflet_text_toggled(self, checked: bool) -> None:
        self._refresh_leaflet_strategy_locks()
        self.leafletTextToggled.emit(checked)

    def _on_leaflet_visual_toggled(self, checked: bool) -> None:
        self._refresh_leaflet_strategy_locks()
        self.leafletVisualToggled.emit(checked)

    def revert_leaflet_pages_field(self, text: str) -> None:
        """A page choice whose comparison did not finish: the picker goes back to what the results on show used."""
        self.leaflet_pages_picker.set_spec(text)

    def _on_leaflet_pages_chosen(self, spec: str) -> None:
        if self._busy_locked:
            return
        self.leafletPagesChanged.emit(spec)
    def release_sources(self) -> None:
        """Stop holding the loaded files (they are closed when replaced)."""
        self.viewport.release_sources()

    def _sync_markers(self) -> None:
        self.viewport.set_marker_clusters(self.panel.marker_clusters())

    def _on_finding_selected(self, cluster_id: int | None) -> None:
        if cluster_id is None:  # the already-selected card was clicked again: deselect, without moving the view
            self.viewport.set_selected(None)
            return
        cluster = self.panel.cluster(cluster_id)
        if cluster is not None:
            self.viewport.focus_cluster(cluster)

    def _on_marker_clicked(self, cluster_id: int) -> None:
        self.viewport.set_selected(cluster_id)
        self.panel.select(cluster_id)

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
