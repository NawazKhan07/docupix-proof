"""Comparison mode selector (SRS §2.3). Mode is a parameter object handed to
the job, never a UI-level code fork (Architecture §3.1)."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFrame,
    QGroupBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
)

from app.models.job import MODE_DISPLAY_NAMES, ComparisonMode
from app.ui.theme import set_variant

_MODE_DESCRIPTIONS = {
    ComparisonMode.AUTOMATIC: "Auto-detects the artwork area and compares every difference category.",
    ComparisonMode.SCAN_VS_DIGITAL: "Ignores colour: both files become black ink on white, and their text and numbers are read and compared, with barcode bars as marks.",
    ComparisonMode.BARCODE: "Restricts comparison to detected barcode/pharmacode/QR symbols, decoded and compared.",
    ComparisonMode.BRAILLE: "Restricts comparison to the coloured dot-matrix representing Braille content.",
}

_MODE_ORDER = [
    ComparisonMode.AUTOMATIC,
    ComparisonMode.SCAN_VS_DIGITAL,
    ComparisonMode.BARCODE,
    ComparisonMode.BRAILLE,
]


class _ModeCard(QFrame):
    """One selectable mode row: a radio button, title, and short description.

    Clicking anywhere in the card selects it, not just the radio dot.
    """

    def __init__(self, mode: ComparisonMode, group: QButtonGroup, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.setProperty("role", "card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self.radio = QRadioButton(MODE_DISPLAY_NAMES[mode])
        group.addButton(self.radio)

        description = QLabel(_MODE_DESCRIPTIONS[mode])
        description.setProperty("secondary", "true")
        description.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(3)
        layout.addWidget(self.radio)
        layout.addWidget(description)

        self.radio.toggled.connect(self._refresh_selected_style)
        self._refresh_selected_style()

    def _refresh_selected_style(self) -> None:
        set_variant(self, "selected", "true" if self.radio.isChecked() else "false")

    def mousePressEvent(self, event) -> None:
        self.radio.setChecked(True)
        super().mousePressEvent(event)


class ModeSelector(QGroupBox):
    """Selectable cards for the four comparison modes. Automatic is default."""

    modeChanged = Signal(ComparisonMode)

    def __init__(self, parent=None):
        super().__init__("Comparison Mode", parent)
        self._button_group = QButtonGroup(self)
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self._cards: dict[ComparisonMode, _ModeCard] = {}
        for mode in _MODE_ORDER:
            card = _ModeCard(mode, self._button_group, self)
            layout.addWidget(card)
            self._cards[mode] = card
            card.radio.toggled.connect(self._on_toggled(mode))

        # A scan changes every colour and blurs every edge, so comparing its graphics gives false differences round the
        # real ones: text is what this mode compares, and graphics are an option.
        self.scan_graphics = QCheckBox("Also compare graphics (a scan can show false differences)")
        self.scan_graphics.setToolTip(
            "Scan vs Digital compares text, numbers and barcode bars. Tick this to compare logos and other artwork too."
        )
        self.scan_graphics.setVisible(False)
        layout.addWidget(self.scan_graphics)
        self.modeChanged.connect(lambda mode: self.scan_graphics.setVisible(mode is ComparisonMode.SCAN_VS_DIGITAL))

        self._cards[ComparisonMode.AUTOMATIC].radio.setChecked(True)

    def _on_toggled(self, mode: ComparisonMode):
        def handler(checked: bool) -> None:
            if checked:
                self.modeChanged.emit(mode)

        return handler

    @property
    def mode(self) -> ComparisonMode:
        for mode, card in self._cards.items():
            if card.radio.isChecked():
                return mode
        return ComparisonMode.AUTOMATIC  # unreachable: the group always has one checked

    def set_mode(self, mode: ComparisonMode) -> None:
        self._cards[mode].radio.setChecked(True)

    @property
    def compare_scan_graphics(self) -> bool:
        """Whether a Scan vs Digital comparison should also compare graphics."""
        return self.scan_graphics.isChecked()
