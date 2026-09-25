"""Original/Suspect file upload (SRS FR-1.1/FR-1.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.config import AppConfig, RuntimeConfig
from app.engine.file_loader import ArtworkLoadError, LoadedArtwork, SUPPORTED_EXTENSIONS, load_artwork
from app.engine.isolation.cutout import Cutout
from app.models.isolation import IsolationGeometry
from app.models.job import ComparisonMode
from app.ui.isolation_dialog import IsolationDialog, SubjectReference
from app.ui.theme import set_variant

_FILE_DIALOG_FILTER = "Artwork files (*" + " *".join(sorted(SUPPORTED_EXTENSIONS)) + ")"


class UploadSlot(QWidget):
    """One file slot (Original or Suspect): title + status pill, buttons, detail line."""

    changed = Signal()

    def __init__(
        self, title: str, parent=None, limits: RuntimeConfig | None = None, config: AppConfig | None = None
    ):
        super().__init__(parent)
        self.title = title
        self.config = config or AppConfig()
        self.limits = limits or self.config.runtime
        self.loaded: Optional[LoadedArtwork] = None
        self.geometry: Optional[IsolationGeometry] = None
        # The artwork lifted from this file (if any), which the other file can lay over itself.
        self.cutout: Optional[Cutout] = None
        self.partner: Optional[UploadSlot] = None  # the other slot; set by the panel
        self.mode = ComparisonMode.AUTOMATIC  # only affects the wording of the status line (e.g. "every page compared")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        header_row = QHBoxLayout()
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: 700; color: #1f2430; font-size: 13px;")
        self.status_pill = QLabel("Not loaded")
        self.status_pill.setProperty("pill", "empty")
        self.status_pill.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        header_row.addWidget(title_label, alignment=Qt.AlignmentFlag.AlignVCenter)
        header_row.addStretch(1)
        header_row.addWidget(self.status_pill, alignment=Qt.AlignmentFlag.AlignVCenter)
        outer.addLayout(header_row)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        self.browse_button = QPushButton(f"Browse {title}…")
        self.isolate_button = QPushButton("Edit Artwork Area…")
        self.isolate_button.setEnabled(False)
        button_row.addWidget(self.browse_button)
        button_row.addWidget(self.isolate_button)
        button_row.addStretch(1)
        outer.addLayout(button_row)

        self.detail_label = QLabel("No file selected")
        self.detail_label.setProperty("secondary", "true")
        self.detail_label.setWordWrap(True)
        outer.addWidget(self.detail_label)

        # Series mode never isolates this slot's own file (the suspect is used as the whole sheet, and every repeat
        # of the original's artwork is found on it automatically): said here rather than left for the reviewer to
        # discover by the isolate button quietly doing nothing.
        self.mode_hint_label = QLabel("")
        self.mode_hint_label.setProperty("secondary", "true")
        self.mode_hint_label.setWordWrap(True)
        self.mode_hint_label.setVisible(False)
        outer.addWidget(self.mode_hint_label)

        self.browse_button.clicked.connect(self._on_browse)
        self.isolate_button.clicked.connect(self._on_edit_artwork_area)

    def set_mode_hint(self, text: str) -> None:
        self.mode_hint_label.setText(text)
        self.mode_hint_label.setVisible(bool(text))

    def set_mode(self, mode: ComparisonMode) -> None:
        """Only the status line's wording changes (e.g. Leaflet mode compares every page, not just the first)."""
        self.mode = mode
        if self.loaded is not None:
            self._refresh_status()

    def _on_browse(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, f"Select {self.title} file", "", _FILE_DIALOG_FILTER
        )
        if path_str:
            self.load_path(Path(path_str))

    def load_path(self, path: Path) -> None:
        try:
            artwork = load_artwork(path, self.limits)
        except ArtworkLoadError as exc:
            QMessageBox.warning(self, f"Could not load {self.title} file", str(exc))
            return

        if self.loaded is not None:
            self.loaded.close()
        self.loaded = artwork
        self.geometry = None
        self.cutout = None
        self.isolate_button.setEnabled(True)
        set_variant(self.status_pill, "pill", "loaded")
        self.status_pill.setText("Loaded")
        self._refresh_status()
        self.changed.emit()

    def _on_edit_artwork_area(self) -> None:
        assert self.loaded is not None
        reference = None
        if self.partner is not None and self.partner.cutout is not None and self.partner.loaded is not None:
            partner = self.partner.loaded
            reference = SubjectReference(
                self.partner.cutout, partner.path.name, source_dpi=partner.working_dpi() if partner.kind == "pdf" else None
            )
        partner_title = self.partner.title if self.partner is not None else None
        dialog = IsolationDialog(
            self.loaded, self.title, self, reference=reference, config=self.config, partner_title=partner_title
        )
        if dialog.exec() == IsolationDialog.DialogCode.Accepted:
            self.geometry = dialog.result_geometry()
            lifted = dialog.result_cutout()
            if lifted is not None:  # keep an earlier lift when this visit did not make a new one
                self.cutout = lifted
            self._refresh_status()
            if self.partner is not None and self.partner.loaded is not None:
                self.partner._refresh_status()  # its note may now offer, or stop offering, the artwork lifted here
            self.changed.emit()

    def _refresh_status(self) -> None:
        assert self.loaded is not None
        pages = ""
        if self.loaded.kind == "pdf":
            pages = f", {self.loaded.page_count} page(s)"
            if self.loaded.page_count > 1:
                pages += " \u2014 every page is compared" if self.mode is ComparisonMode.LEAFLET else " \u2014 only page 1 is compared"
        geometry_note = ""
        if self.geometry is not None and self.geometry.boundary is not None:
            origin = self.geometry.boundary_origin
            if origin == "lifted":
                geometry_note = " · artwork lifted"
                if self._can_offer_cutout():
                    geometry_note += f" (place it on the {self.partner.title} from its Edit Artwork Area)"
            elif origin == "placed":
                geometry_note = " · trimmed to the placed outline"
            else:
                kind = "polygon" if hasattr(self.geometry.boundary, "points") else "crop"
                geometry_note = f" · {kind} area set"
        if self.geometry is not None and self.geometry.erase_regions:
            count = len(self.geometry.erase_regions)
            geometry_note += f" · {count} area{'s' if count != 1 else ''} erased"
        self.detail_label.setText(
            f"{self.loaded.path.name} ({self.loaded.kind}{pages}){geometry_note}"
        )

    def _can_offer_cutout(self) -> bool:
        """Whether the artwork lifted here is waiting to be placed on a partner that has no area of its own yet."""
        partner = self.partner
        return (
            self.cutout is not None
            and partner is not None
            and partner.loaded is not None
            and (partner.geometry is None or partner.geometry.boundary is None)
        )

    def clear(self) -> None:
        if self.loaded is not None:
            self.loaded.close()
        self.loaded = None
        self.geometry = None
        self.cutout = None
        self.isolate_button.setEnabled(False)
        set_variant(self.status_pill, "pill", "empty")
        self.status_pill.setText("Not loaded")
        self.detail_label.setText("No file selected")
        self.changed.emit()


class UploadPanel(QGroupBox):
    """Holds the Original and Suspect upload slots, stacked with a divider."""

    filesChanged = Signal()

    def __init__(self, parent=None, limits: RuntimeConfig | None = None, config: AppConfig | None = None):
        super().__init__("Upload", parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        self.original_slot = UploadSlot("Original", self, limits, config)
        self.suspect_slot = UploadSlot("Suspect", self, limits, config)
        self.original_slot.partner, self.suspect_slot.partner = self.suspect_slot, self.original_slot

        divider = QFrame(self)
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #e3e5ea;")

        layout.addWidget(self.original_slot)
        layout.addWidget(divider)
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

    def set_mode(self, mode: ComparisonMode) -> None:
        """Series mode uses the suspect as a whole sheet (every repeat of the original's artwork on it is found
        automatically), so its own crop/lift area is beside the point -- said here, not left silently unused. Both
        slots also pick up the mode for their own status line (Leaflet mode compares every page, not just the first)."""
        self.original_slot.set_mode(mode)
        self.suspect_slot.set_mode(mode)
        self.suspect_slot.set_mode_hint(
            "Series mode: used as the whole sheet. Crop or lift the artwork in the Original instead; every repeat "
            "of it here is found automatically."
            if mode is ComparisonMode.SERIES
            else ""
        )
