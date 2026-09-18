"""Original/Suspect file upload (SRS FR-1.1/FR-1.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.engine.file_loader import ArtworkLoadError, LoadedArtwork, SUPPORTED_EXTENSIONS, load_artwork

_FILE_DIALOG_FILTER = "Artwork files (*" + " *".join(sorted(SUPPORTED_EXTENSIONS)) + ")"


class UploadSlot(QWidget):
    """One file slot (Original or Suspect): browse button + status + loaded state."""

    changed = Signal()

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.loaded: Optional[LoadedArtwork] = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.browse_button = QPushButton(f"Browse {title}…")
        self.status_label = QLabel("No file selected")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.browse_button)
        layout.addWidget(self.status_label, stretch=1)

        self.browse_button.clicked.connect(self._on_browse)

    def _on_browse(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, f"Select {self.title} file", "", _FILE_DIALOG_FILTER
        )
        if path_str:
            self.load_path(Path(path_str))

    def load_path(self, path: Path) -> None:
        try:
            artwork = load_artwork(path)
        except ArtworkLoadError as exc:
            QMessageBox.warning(self, f"Could not load {self.title} file", str(exc))
            return

        if self.loaded is not None:
            self.loaded.close()
        self.loaded = artwork
        pages = f", {artwork.page_count} page(s)" if artwork.kind == "pdf" else ""
        self.status_label.setText(f"{path.name} ({artwork.kind}{pages})")
        self.changed.emit()

    def clear(self) -> None:
        if self.loaded is not None:
            self.loaded.close()
        self.loaded = None
        self.status_label.setText("No file selected")
        self.changed.emit()


class UploadPanel(QGroupBox):
    """Holds the Original and Suspect upload slots side by side."""

    filesChanged = Signal()

    def __init__(self, parent=None):
        super().__init__("Upload", parent)
        layout = QVBoxLayout(self)

        self.original_slot = UploadSlot("Original", self)
        self.suspect_slot = UploadSlot("Suspect", self)
        layout.addWidget(self.original_slot)
        layout.addWidget(self.suspect_slot)

        self.original_slot.changed.connect(self.filesChanged)
        self.suspect_slot.changed.connect(self.filesChanged)

    @property
    def original(self) -> Optional[LoadedArtwork]:
        return self.original_slot.loaded

    @property
    def suspect(self) -> Optional[LoadedArtwork]:
        return self.suspect_slot.loaded

    @property
    def both_loaded(self) -> bool:
        return self.original is not None and self.suspect is not None
