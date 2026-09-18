"""The results page: comparison viewer beside the findings list (Build Prompt Phase 5).

Wires the two components to each other so they never disagree: choosing a
finding jumps the viewer to it, clicking a marker selects its finding, and
filtering or ignoring in the list changes which markers the viewer draws.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSplitter, QTabWidget, QVBoxLayout, QWidget

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
        header.addWidget(self.back_button, alignment=Qt.AlignmentFlag.AlignTop)
        header.addSpacing(8)
        header.addLayout(titles, stretch=1)
        layout.addLayout(header)

        self.viewport = ComparisonViewport(self)
        self.panel = DiffListPanel(self)
        self.symbol_panel = SymbolResultsPanel(self)  # Barcode mode (FR-10.2)
        self.braille_panel = BrailleResultsPanel(self)  # Braille mode (FR-11.1)

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
        self.viewport.set_images(outcome.aligned_original, outcome.aligned_suspect)
        self.viewport.set_sources(original, suspect, outcome.alignment.transform)
        self.panel.set_result(outcome.clusters, outcome.alignment, outcome.warnings)
        self._show_mode_panels(outcome)
        self._sync_markers()

    def _show_mode_panels(self, outcome: ComparisonOutcome) -> None:
        """Add the Symbols / Braille tab when the mode produced something to show in it."""
        for panel in (self.symbol_panel, self.braille_panel):
            index = self.side.indexOf(panel)
            if index >= 0:
                self.side.removeTab(index)

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
