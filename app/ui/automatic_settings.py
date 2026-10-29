"""Automatic mode's settings on the setup page: shown only while Automatic is the mode chosen.

One for now: FOOD LABEL -- compare the files as a consumer label (a snack pack, a drinks wrapper, a bag of fries) by
the things printed on it, not pixel by pixel (`app.engine.food_label`). Off unless ticked.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMenu, QToolButton, QWidget


class AutomaticSettings(QWidget):
    """A settings button (with its menu) and a line saying what is set."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.button = QToolButton(self)
        self.button.setText("⚙  Automatic settings")
        self.button.setToolTip("How Automatic mode compares these files")
        self.button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.button)
        self.food_label_action = QAction("Food label", menu)
        self.food_label_action.setCheckable(True)
        self.food_label_action.setToolTip(
            "Compare as a food or consumer label (snack packs, drinks, food bags): by the things printed -- a logo, a "
            "sticker, a strip, a block of type, a photograph -- one finding each, instead of pixel by pixel."
        )
        menu.addAction(self.food_label_action)
        menu.setToolTipsVisible(True)
        self.button.setMenu(menu)
        self.label = QLabel("")
        self.label.setProperty("secondary", "true")
        layout.addWidget(self.button)
        layout.addWidget(self.label, stretch=1)
        self.food_label_action.toggled.connect(self._on_toggled)
        self._show()

    @property
    def food_label(self) -> bool:
        return self.food_label_action.isChecked()

    def set_food_label(self, on: bool) -> None:
        self.food_label_action.setChecked(on)

    def _on_toggled(self, _checked: bool) -> None:
        self._show()
        self.changed.emit()

    def _show(self) -> None:
        self.label.setText("Food label comparison is on" if self.food_label else "")
