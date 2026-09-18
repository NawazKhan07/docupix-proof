"""Braille dot-pattern results (SRS FR-11.1).

Braille on artwork is marked as coloured dots; the comparison matches each dot
in the original to the nearest one in the suspect and reports what is missing,
added or moved. This panel gives the verdict and the counts behind it.

The colour that marks a dot is an open item in the SRS (§7), so the panel says
plainly when no dots were found at all -- the likeliest cause is that the
artwork uses a different colour from the one assumed, not that it has no Braille.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

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


class BrailleResultsPanel(QWidget):
    """The verdict on the dot patterns, with the counts behind it (FR-11.1)."""

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

        self.verdict_card = QFrame()
        self.verdict_card.setProperty("role", "card")
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

        assumption = QLabel(
            "Dots are recognised by colour. The colour range is a configurable assumption "
            "(SRS §7 open item): confirm it matches your artwork's Braille dots."
        )
        assumption.setWordWrap(True)
        assumption.setProperty("secondary", "true")
        layout.addWidget(assumption)
        layout.addStretch(1)

        self.set_comparison(None)

    def set_comparison(self, comparison: BrailleComparison | None) -> None:
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
