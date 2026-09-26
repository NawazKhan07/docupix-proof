"""Braille dot-pattern results (SRS FR-11.1).

Braille on artwork is marked as coloured dots; the comparison matches each dot
in the original to the nearest one in the suspect and reports what is missing,
added or moved. This panel gives the verdict and the counts behind it, and each dot difference: clicking one (or the verdict, for the
whole Braille) shows it in the viewer.

The colour that marks a dot is an open item in the SRS (§7), so the panel says
plainly when no dots were found at all -- the likeliest cause is that the
artwork uses a different colour from the one assumed, not that it has no Braille.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from app.models.diff import BoundingBox, DiffCluster
from app.models.result import BrailleComparison
from app.ui.symbol_panel import CROSS, TICK, _badge
from app.ui.theme import set_variant

_COUNTS = (
    ("original_dot_count", "Dots in original", "#1f2430"),
    ("suspect_dot_count", "Dots in suspect", "#1f2430"),
    ("matched", "Matched", "#15803d"),
    ("missing", "Missing from suspect", "#b91c1c"),
    ("added", "Added in suspect", "#b91c1c"),
    ("moved", "Moved", "#b45309"),
)


class _ClickableCard(QFrame):
    """A card that says when it is clicked, with the region it stands for."""

    clicked = Signal(object)  # BoundingBox

    def __init__(self, box: BoundingBox | None = None, parent=None):
        super().__init__(parent)
        self.box = box
        self.setProperty("role", "card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event) -> None:
        if self.box is not None:
            self.clicked.emit(self.box)
        super().mousePressEvent(event)


class BrailleResultsPanel(QWidget):
    """The verdict on the dot patterns, with the counts behind it (FR-11.1), and each dot difference."""

    regionSelected = Signal(object)  # BoundingBox: show it in the viewer

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        header = QHBoxLayout()
        self.title_label = QLabel("Braille")
        self.title_label.setProperty("role", "heading")
        self.title_label.setStyleSheet("font-size: 17px;")
        self.summary_pill = QLabel("")
        self.summary_pill.setProperty("pill", "empty")
        header.addWidget(self.title_label)
        header.addStretch(1)
        header.addWidget(self.summary_pill)
        layout.addLayout(header)

        self.verdict_card = _ClickableCard()
        self.verdict_card.setToolTip("Show the Braille in the viewer.")
        self.verdict_card.clicked.connect(self.regionSelected)
        self._verdict_row = QHBoxLayout(self.verdict_card)
        self._verdict_row.setContentsMargins(12, 10, 12, 10)
        self._verdict_row.setSpacing(12)
        self.verdict_badge: QLabel | None = None
        self.verdict_label = QLabel("")
        self.verdict_label.setWordWrap(True)
        self.verdict_label.setStyleSheet("font-weight: 700; font-size: 14px;")
        self._verdict_row.addWidget(self.verdict_label, stretch=1)
        layout.addWidget(self.verdict_card)

        self.counts_card = QFrame()
        self.counts_card.setProperty("role", "card")
        grid = QGridLayout(self.counts_card)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(6)
        self._count_labels: dict[str, QLabel] = {}
        for row, (key, caption, colour) in enumerate(_COUNTS):
            name = QLabel(caption)
            name.setProperty("secondary", "true")
            value = QLabel("0")
            value.setAlignment(Qt.AlignmentFlag.AlignRight)
            value.setStyleSheet(f"font-weight: 700; color: {colour};")
            grid.addWidget(name, row, 0)
            grid.addWidget(value, row, 1)
            self._count_labels[key] = value
        layout.addWidget(self.counts_card)

        self.note_label = QLabel("")
        self.note_label.setWordWrap(True)
        self.note_label.setProperty("status", "warn")
        self.note_label.setVisible(False)
        layout.addWidget(self.note_label)

        # Each dot difference, clickable (a list that can be long: it scrolls, like the findings list).
        self.differences_title = QLabel("Differences")
        self.differences_title.setStyleSheet("font-weight: 700;")
        self.differences_title.setVisible(False)
        layout.addWidget(self.differences_title)
        self._differences_layout = QVBoxLayout()
        self._differences_layout.setContentsMargins(0, 0, 4, 0)
        self._differences_layout.setSpacing(6)
        container = QWidget()
        container.setLayout(self._differences_layout)
        self._differences_scroll = QScrollArea()
        self._differences_scroll.setWidgetResizable(True)
        self._differences_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._differences_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._differences_scroll.setWidget(container)
        self._differences_scroll.setVisible(False)
        layout.addWidget(self._differences_scroll, stretch=1)
        self._difference_cards: list[_ClickableCard] = []

        assumption = QLabel(
            "Dots are recognised by colour. The colour range is a configurable assumption "
            "(SRS §7 open item): confirm it matches your artwork's Braille dots."
        )
        assumption.setWordWrap(True)
        assumption.setProperty("secondary", "true")
        layout.addWidget(assumption)

        self.set_comparison(None)

    def set_differences(self, clusters: list[DiffCluster]) -> None:
        """The dot differences to list (the Braille findings), each clickable to show it in the viewer."""
        while self._differences_layout.count():
            item = self._differences_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()
                item.widget().setParent(None)
                item.widget().deleteLater()
        self._difference_cards = []
        for cluster in clusters:
            card = _ClickableCard(cluster.bbox_in_original)
            row = QVBoxLayout(card)
            row.setContentsMargins(10, 6, 10, 6)
            title = QLabel(f"#{cluster.id}  ·  {cluster.detail}")
            title.setWordWrap(True)
            row.addWidget(title)
            card.clicked.connect(self.regionSelected)
            self._differences_layout.addWidget(card)
            self._difference_cards.append(card)
        self._differences_layout.addStretch(1)
        self.differences_title.setVisible(bool(clusters))
        self._differences_scroll.setVisible(bool(clusters))

    def difference_cards(self) -> list[_ClickableCard]:
        return list(self._difference_cards)

    def set_comparison(self, comparison: BrailleComparison | None) -> None:
        self.verdict_card.box = comparison.region if comparison is not None else None
        for key, label in self._count_labels.items():
            label.setText(str(getattr(comparison, key)) if comparison is not None else "–")
        self.note_label.setVisible(False)

        if comparison is None:
            self._show_verdict(None, "Braille was not compared.")
            self.summary_pill.setText("")
            return

        if comparison.original_dot_count == 0 and comparison.suspect_dot_count == 0:
            self._show_verdict(None, "No Braille dots were detected in either file.")
            self.summary_pill.setText("nothing detected")
            set_variant(self.summary_pill, "pill", "warn")
            self.note_label.setText(
                "The dot colour may differ from the one assumed, so Braille may be present but not recognised."
            )
            self.note_label.setVisible(True)
            return

        differences = comparison.missing + comparison.added + comparison.moved
        if comparison.passed:
            self._show_verdict(True, f"Dot patterns match ({comparison.matched} dots).")
            self.summary_pill.setText(f"{TICK} match")
            set_variant(self.summary_pill, "pill", "loaded")
        else:
            self._show_verdict(False, f"{differences} dot difference{'s' if differences != 1 else ''} found.")
            self.summary_pill.setText(f"{CROSS} {differences} differ")
            set_variant(self.summary_pill, "pill", "warn")

    def _show_verdict(self, passed: bool | None, text: str) -> None:
        if self.verdict_badge is not None:
            self._verdict_row.removeWidget(self.verdict_badge)
            self.verdict_badge.deleteLater()
            self.verdict_badge = None
        if passed is not None:
            self.verdict_badge = _badge(passed)
            self._verdict_row.insertWidget(0, self.verdict_badge)
        self.verdict_label.setText(text)
