"""Comparison mode selector (SRS §2.3). Mode is a parameter object handed to
the job, never a UI-level code fork (Architecture §3.1)."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QButtonGroup, QGroupBox, QRadioButton, QVBoxLayout

from app.models.job import MODE_DISPLAY_NAMES, ComparisonMode

_MODE_ORDER = [
    ComparisonMode.AUTOMATIC,
    ComparisonMode.SCAN_VS_DIGITAL,
    ComparisonMode.BARCODE,
    ComparisonMode.BRAILLE,
]


class ModeSelector(QGroupBox):
    """Radio-button group for the four comparison modes. Automatic is default."""

    modeChanged = Signal(ComparisonMode)

    def __init__(self, parent=None):
        super().__init__("Comparison Mode", parent)
        self._button_group = QButtonGroup(self)
        layout = QVBoxLayout(self)

        self._buttons: dict[ComparisonMode, QRadioButton] = {}
        for mode in _MODE_ORDER:
            button = QRadioButton(MODE_DISPLAY_NAMES[mode])
            self._button_group.addButton(button)
            layout.addWidget(button)
            self._buttons[mode] = button
            button.toggled.connect(self._on_toggled(mode))

        self._buttons[ComparisonMode.AUTOMATIC].setChecked(True)

    def _on_toggled(self, mode: ComparisonMode):
        def handler(checked: bool) -> None:
            if checked:
                self.modeChanged.emit(mode)

        return handler

    @property
    def mode(self) -> ComparisonMode:
        for mode, button in self._buttons.items():
            if button.isChecked():
                return mode
        return ComparisonMode.AUTOMATIC  # unreachable: the group always has one checked

    def set_mode(self, mode: ComparisonMode) -> None:
        self._buttons[mode].setChecked(True)
