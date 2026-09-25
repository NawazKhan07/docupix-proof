"""The results page: comparison viewer beside the findings list (Build Prompt Phase 5).

Wires the two components to each other so they never disagree: choosing a
finding jumps the viewer to it, clicking a marker selects its finding, and
filtering or ignoring in the list changes which markers the viewer draws.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QSplitter, QTabWidget, QVBoxLayout, QWidget

from app.engine.file_loader import LoadedArtwork
from app.engine.worker import ComparisonOutcome
from app.models.job import ComparisonMode
from app.ui.braille_panel import BrailleResultsPanel
from app.ui.comparison_viewport import ComparisonViewport
from app.ui.diff_list_panel import DiffListPanel
from app.ui.symbol_panel import SymbolResultsPanel

PANEL_MIN_WIDTH = 340


class ResultsView(QWidget):
    backRequested = Signal()
    graphicsToggled = Signal(bool)  # Scan vs Digital: compare the graphics too (or not), on these same files

    def __init__(self, parent=None):
        super().__init__(parent)
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
        header.addLayout(titles, stretch=1)
        # Scan vs Digital compares text; its graphics are an option, switched here so the reviewer need not go back,
        # tick a box and start again. A change re-compares the same files.
        self.graphics_check = QCheckBox("Also compare graphics")
        self.graphics_check.setToolTip(
            "Scan vs Digital compares text, numbers and barcode bars. Tick this to compare logos and other artwork too "
            "(a scan can show false differences there)."
        )
        self.graphics_check.setVisible(False)
        self.graphics_check.toggled.connect(self._on_graphics_toggled)
        self.busy_label = QLabel("")
        self.busy_label.setProperty("status", "idle")
        self.busy_label.setVisible(False)
        header.addWidget(self.busy_label, alignment=Qt.AlignmentFlag.AlignTop)
        header.addWidget(self.graphics_check, alignment=Qt.AlignmentFlag.AlignTop)
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
        self.symbol_panel.symbolSelected.connect(self.viewport.focus_box)
        self.viewport.markerClicked.connect(self._on_marker_clicked)
        self.viewport.mergeRegionSelected.connect(self._on_merge_region)
        self.viewport.undoMergeRequested.connect(self._on_undo_merge)

    def show_outcome(
        self,
        outcome: ComparisonOutcome,
        subtitle: str,
        original: LoadedArtwork | None = None,
        suspect: LoadedArtwork | None = None,
    ) -> None:
        """Show a result. Passing the loaded files lets the viewer re-render PDFs
        from vector data when zoomed in (FR-9.1)."""
        self.subtitle_label.setText(subtitle)
        self.note_label.setText("  ".join(outcome.notes))
        self.note_label.setVisible(bool(outcome.notes))
        self.has_results = True
        scan_mode = outcome.mode is ComparisonMode.SCAN_VS_DIGITAL
        self.graphics_check.blockSignals(True)
        self.graphics_check.setChecked(bool(outcome.graphics_compared))
        self.graphics_check.blockSignals(False)
        self.graphics_check.setVisible(scan_mode)
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
        self.viewport.set_sources(original, suspect, outcome.alignment.transform)
        self.viewport.set_compared_area(outcome.compared_outline)
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
            self.side.addTab(self.braille_panel, "Braille")
            chosen = self.braille_panel

        self.side.tabBar().setVisible(self.side.count() > 1)  # a lone tab is just noise
        self.side.setCurrentWidget(chosen)

    has_results = False

    def set_busy(self, text: str, locked: bool = True) -> None:
        """Say what a re-comparison is doing, or how it ended (empty for nothing to say); the toggle is locked while
        it runs."""
        self.busy_label.setText(text)
        self.busy_label.setVisible(bool(text))
        self.graphics_check.setEnabled(not (text and locked))

    def _on_graphics_toggled(self, checked: bool) -> None:
        self.graphicsToggled.emit(checked)

    def release_sources(self) -> None:
        """Stop holding the loaded files (they are closed when replaced)."""
        self.viewport.release_sources()

    def _sync_markers(self) -> None:
        self.viewport.set_marker_clusters(self.panel.marker_clusters())

    def _on_finding_selected(self, cluster_id: int) -> None:
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
