"""Barcode / Pharmacode / QR results (SRS FR-10.2).

"Results shall be grouped by symbol type; each decoded component shall be
matched original-to-suspect and marked with a pass (green tick) or fail (red
cross) indicator." One card per symbol, under a heading for its group, with the
value read from each file side by side so a reviewer can see *what* differs.
Choosing a card shows where the symbol is on the artwork.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from app.models.diff import BoundingBox
from app.models.result import SymbolComparison
from app.ui.theme import set_variant

GROUP_ORDER = ("Barcode", "Pharmacode", "QR code")
TICK, CROSS = "✓", "✗"
_PASS_COLOUR, _FAIL_COLOUR = "#15803d", "#b91c1c"
_MISSING_TEXT = "not found"


def _badge(passed: bool) -> QLabel:
    """The green tick / red cross of FR-10.2."""
    badge = QLabel(TICK if passed else CROSS)
    badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
    badge.setFixedSize(26, 26)
    colour, background = (_PASS_COLOUR, "#dcfce7") if passed else (_FAIL_COLOUR, "#fee2e2")
    badge.setStyleSheet(
        f"background: {background}; color: {colour}; border-radius: 13px; font-size: 15px; font-weight: 800;"
    )
    badge.setAccessibleName("pass" if passed else "fail")
    return badge


class SymbolCard(QFrame):
    """One symbol: pass/fail badge, its type, and the value read from each file."""

    selectRequested = Signal(object)  # BoundingBox | None

    def __init__(self, comparison: SymbolComparison, parent=None):
        super().__init__(parent)
        self.comparison = comparison
        self.setProperty("role", "card")
        self.setProperty("selected", "false")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self.badge = _badge(comparison.passed)

        title = QLabel(comparison.symbol_type)
        title.setStyleSheet("font-weight: 700;")
        status = QLabel(comparison.status.capitalize())
        status.setStyleSheet(f"color: {_PASS_COLOUR if comparison.passed else _FAIL_COLOUR}; font-weight: 600;")
        self.title_label, self.status_label = title, status

        heading = QHBoxLayout()
        heading.addWidget(title)
        heading.addStretch(1)
        heading.addWidget(status)

        self.original_label = self._value_row("Original", comparison.original_value, mismatch=False)
        self.suspect_label = self._value_row("Suspect", comparison.suspect_value, mismatch=not comparison.passed)

        text = QVBoxLayout()
        text.setSpacing(3)
        text.addLayout(heading)
        text.addWidget(self.original_label)
        text.addWidget(self.suspect_label)

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(10)
        row.addWidget(self.badge, alignment=Qt.AlignmentFlag.AlignTop)
        row.addLayout(text, stretch=1)

    @staticmethod
    def _value_row(caption: str, value: str | None, mismatch: bool) -> QLabel:
        label = QLabel()
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        shown = value if value is not None else f"<i>{_MISSING_TEXT}</i>"
        colour = _FAIL_COLOUR if mismatch else "#1f2430"
        label.setText(
            f'<span style="color:#6b7280">{caption}&nbsp;&nbsp;</span>'
            f'<span style="color:{colour}; font-family:Consolas,monospace">{shown}</span>'
        )
        return label

    def set_selected(self, selected: bool) -> None:
        set_variant(self, "selected", "true" if selected else "false")

    def mousePressEvent(self, event) -> None:
        self.selectRequested.emit(self.comparison.bbox)
        super().mousePressEvent(event)


class SymbolResultsPanel(QWidget):
    """Decoded symbols grouped by type, each marked pass or fail (FR-10.2)."""

    symbolSelected = Signal(object)  # BoundingBox

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cards: list[SymbolCard] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header = QHBoxLayout()
        self.title_label = QLabel("Symbols")
        self.title_label.setProperty("role", "heading")
        self.title_label.setStyleSheet("font-size: 17px;")
        self.summary_pill = QLabel("")
        self.summary_pill.setProperty("pill", "empty")
        header.addWidget(self.title_label)
        header.addStretch(1)
        header.addWidget(self.summary_pill)
        layout.addLayout(header)

        self.empty_label = QLabel(
            "No barcode, Pharmacode or QR code could be read in either file, so nothing was compared. "
            "If the artwork has one, it may be too small or low-contrast to decode."
        )
        self.empty_label.setWordWrap(True)
        self.empty_label.setProperty("status", "warn")
        layout.addWidget(self.empty_label)

        self._body = QVBoxLayout()
        self._body.setSpacing(8)
        self._body.setContentsMargins(0, 0, 4, 0)
        container = QWidget()
        container.setLayout(self._body)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setWidget(container)
        layout.addWidget(self._scroll, stretch=1)

        self.set_comparisons([])

    # -- data -------------------------------------------------------------------
    def set_comparisons(self, comparisons: list[SymbolComparison]) -> None:
        self._clear()
        self.empty_label.setVisible(not comparisons)

        failed = sum(1 for comparison in comparisons if not comparison.passed)
        if not comparisons:
            self.title_label.setText("Symbols (0)")
            self.summary_pill.setText("nothing read")
            set_variant(self.summary_pill, "pill", "warn")
        else:
            self.title_label.setText(f"Symbols ({len(comparisons)})")
            self.summary_pill.setText(f"{TICK} all match" if not failed else f"{CROSS} {failed} of {len(comparisons)} differ")
            set_variant(self.summary_pill, "pill", "loaded" if not failed else "warn")

        for group in self._ordered_groups(comparisons):
            members = [c for c in comparisons if c.group == group]
            self._body.addLayout(self._group_header(group, members))
            for comparison in members:
                card = SymbolCard(comparison)
                card.selectRequested.connect(lambda box, c=card: self._on_selected(c, box))
                self._cards.append(card)
                self._body.addWidget(card)
        self._body.addStretch(1)

    @staticmethod
    def _ordered_groups(comparisons: list[SymbolComparison]) -> list[str]:
        present = {comparison.group for comparison in comparisons}
        return [group for group in GROUP_ORDER if group in present] + sorted(present - set(GROUP_ORDER))

    def _group_header(self, group: str, members: list[SymbolComparison]) -> QHBoxLayout:
        failed = sum(1 for member in members if not member.passed)
        row = QHBoxLayout()
        name = QLabel(f"{group} ({len(members)})")
        name.setStyleSheet("font-weight: 700; font-size: 14px;")
        pill = QLabel(f"{TICK} all match" if not failed else f"{CROSS} {failed} failed")
        pill.setProperty("pill", "loaded" if not failed else "warn")
        row.addWidget(name)
        row.addStretch(1)
        row.addWidget(pill)
        return row

    def _clear(self) -> None:
        self._cards.clear()
        while self._body.count():
            item = self._body.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()
                item.widget().setParent(None)
                item.widget().deleteLater()
            elif item.layout() is not None:
                self._clear_layout(item.layout())

    @staticmethod
    def _clear_layout(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()
                item.widget().setParent(None)
                item.widget().deleteLater()

    # -- interaction --------------------------------------------------------------
    def _on_selected(self, chosen: SymbolCard, box: BoundingBox | None) -> None:
        for card in self._cards:
            card.set_selected(card is chosen)
        if box is not None:
            self.symbolSelected.emit(box)

    def cards(self) -> list[SymbolCard]:
        return list(self._cards)

    def groups(self) -> dict[str, list[SymbolComparison]]:
        """The comparisons shown, by group heading, in display order."""
        result: dict[str, list[SymbolComparison]] = {}
        for card in self._cards:
            result.setdefault(card.comparison.group, []).append(card.comparison)
        return result
