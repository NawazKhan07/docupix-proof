"""Main application window."""

from __future__ import annotations

import logging
from uuid import uuid4

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

from app.config import load_config
from app.engine.alignment.aligner import AlignmentError
from app.engine.worker import run_comparison
from app.logging_config import job_log_file
from app.ui.mode_selector import ModeSelector
from app.ui.theme import set_variant
from app.ui.upload_panel import UploadPanel

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Artwork Comparison")
        self.resize(1000, 700)
        self.config = load_config()

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(24, 20, 24, 24)
        layout.setSpacing(16)

        heading = QLabel("Artwork Comparison")
        heading.setProperty("role", "heading")
        subheading = QLabel("Compare pharmaceutical packaging artwork files, offline.")
        subheading.setProperty("role", "subheading")
        layout.addWidget(heading)
        layout.addWidget(subheading)

        top_row = QHBoxLayout()
        top_row.setSpacing(16)
        self.upload_panel = UploadPanel(central)
        self.mode_selector = ModeSelector(central)
        top_row.addWidget(self.upload_panel, stretch=1)
        top_row.addWidget(self.mode_selector)
        layout.addLayout(top_row, stretch=1)

        self.status_label = QLabel("Load an original and a suspect file to compare.")
        self.status_label.setProperty("status", "idle")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.compare_button = QPushButton("Compare")
        self.compare_button.setProperty("variant", "primary")
        self.compare_button.setEnabled(False)
        self.compare_button.setMinimumHeight(40)
        layout.addWidget(self.compare_button)

        self.upload_panel.filesChanged.connect(self._update_compare_enabled)
        self.compare_button.clicked.connect(self._on_compare)

    def _update_compare_enabled(self) -> None:
        self.compare_button.setEnabled(self.upload_panel.both_loaded)

    def _set_status(self, text: str, state: str) -> None:
        self.status_label.setText(text)
        set_variant(self.status_label, "status", state)

    def _on_compare(self) -> None:
        """Run the pipeline as far as it currently goes: isolate, align, detect.

        Clustering, severity and the diff list panel join this in Phase 5.
        The work runs on the UI thread for now; Phase 7 moves it into a
        worker process.
        """
        original, suspect = self.upload_panel.original, self.upload_panel.suspect
        assert original is not None and suspect is not None

        job_id = uuid4().hex[:8]
        self.compare_button.setEnabled(False)
        self._set_status("Comparing…", "idle")
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QGuiApplication.processEvents()  # paint the busy state before blocking
        try:
            with job_log_file(job_id):
                logger.info(
                    "comparison job started",
                    extra={
                        "job_id": job_id,
                        "mode": self.mode_selector.mode.value,
                        "original": str(original.path),
                        "suspect": str(suspect.path),
                    },
                )
                outcome = run_comparison(
                    original.render(),
                    suspect.render(),
                    self.mode_selector.mode,
                    self.config,
                    original_geometry=self.upload_panel.original_slot.geometry,
                    suspect_geometry=self.upload_panel.suspect_slot.geometry,
                )
        except AlignmentError as exc:
            self._set_status(str(exc), "error")
            return
        finally:
            QGuiApplication.restoreOverrideCursor()
            self.compare_button.setEnabled(self.upload_panel.both_loaded)

        self._set_status(*_describe(outcome))

    def closeEvent(self, event) -> None:
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
        f"{len(outcome.candidates)} difference(s) found"
        if outcome.candidates
        else "No differences found"
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
    elif outcome.candidates:
        state = "warn"
    else:
        state = "ok"
    return " · ".join(parts), state
