"""Main application window (Build Prompt Phase 1)."""

from __future__ import annotations

import logging

from PySide6.QtWidgets import QHBoxLayout, QMainWindow, QPushButton, QVBoxLayout, QWidget

from app.ui.mode_selector import ModeSelector
from app.ui.upload_panel import UploadPanel

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Artwork Comparison")
        self.resize(900, 600)

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        top_row = QHBoxLayout()
        self.upload_panel = UploadPanel(central)
        self.mode_selector = ModeSelector(central)
        top_row.addWidget(self.upload_panel, stretch=1)
        top_row.addWidget(self.mode_selector)
        layout.addLayout(top_row)

        self.compare_button = QPushButton("Compare")
        self.compare_button.setEnabled(False)
        layout.addWidget(self.compare_button)

        self.upload_panel.filesChanged.connect(self._update_compare_enabled)
        self.compare_button.clicked.connect(self._on_compare)

    def _update_compare_enabled(self) -> None:
        self.compare_button.setEnabled(self.upload_panel.both_loaded)

    def _on_compare(self) -> None:
        # The comparison pipeline (isolation -> alignment -> detectors ->
        # aggregation) lands in Phases 2-5; Phase 1 only wires this button up.
        logger.info(
            "Compare requested (pipeline not yet implemented)",
            extra={"mode": self.mode_selector.mode.value},
        )

    def closeEvent(self, event) -> None:
        self.upload_panel.original_slot.clear()
        self.upload_panel.suspect_slot.clear()
        super().closeEvent(event)
