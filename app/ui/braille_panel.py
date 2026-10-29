"""Braille results (SRS FR-11.1).

Braille is read cell by cell in both files (`app.engine.detectors.braille_reader`) and compared cell for cell: the panel
gives the verdict, then each line as read in each file -- every cell shown as its Braille character and what it says,
green where both files have the same dots there, red where they do not (its dots in each file on hovering) -- and each
difference as listed among the findings. Clicking a line, a cell, a difference or the verdict shows it in the viewer.
Dots that are not set to a Braille grid are compared dot by dot, and counted with the rest.

The colour that marks a dot is an open item in the SRS (§7), so the panel says plainly when no dots were found at all --
the likeliest cause is that the artwork uses a different colour from the one assumed, not that it has no Braille.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from app.models.diff import BoundingBox, DiffCluster
from app.engine.detectors.braille_reader import dots_text, glyph
from app.models.result import BrailleCellComparison, BrailleComparison, BrailleLineComparison
from app.ui.flow_layout import flow_widget
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
        self.setProperty("selected", "false")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_selected(self, selected: bool) -> None:
        set_variant(self, "selected", "true" if selected else "false")

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

        # The Braille as read, line by line and cell by cell, then each difference -- clickable, in one list that can be
        # long (it scrolls, like the findings list).
        body = QVBoxLayout()
        body.setContentsMargins(0, 0, 4, 0)
        body.setSpacing(8)
        self.lines_title = QLabel("Braille read, cell by cell")
        self.lines_title.setStyleSheet("font-weight: 700;")
        self.lines_title.setVisible(False)
        body.addWidget(self.lines_title)
        self._lines_layout = QVBoxLayout()
        self._lines_layout.setSpacing(6)
        body.addLayout(self._lines_layout)
        self.differences_title = QLabel("Differences")
        self.differences_title.setStyleSheet("font-weight: 700;")
        self.differences_title.setVisible(False)
        body.addWidget(self.differences_title)
        self._differences_layout = QVBoxLayout()
        self._differences_layout.setSpacing(6)
        body.addLayout(self._differences_layout)
        body.addStretch(1)
        container = QWidget()
        container.setLayout(body)
        self._differences_scroll = QScrollArea()
        self._differences_scroll.setWidgetResizable(True)
        self._differences_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._differences_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._differences_scroll.setWidget(container)
        self._differences_scroll.setVisible(False)
        layout.addWidget(self._differences_scroll, stretch=1)
        self._difference_cards: list[_ClickableCard] = []
        self._line_cards: list[_LineCard] = []
        self._selected: _ClickableCard | None = None  # the line or difference chosen (Up / Down step from it)

        assumption = QLabel(
            "Dots are recognised by their shape and by colour. The colour range is a configurable assumption "
            "(SRS §7 open item): confirm it matches your artwork's Braille dots. Cells are read as uncontracted "
            "Braille; the dots are compared exactly, whatever a cell is read as."
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
        self._selected = None
        for cluster in clusters:
            card = _ClickableCard(cluster.bbox_in_original)
            row = QVBoxLayout(card)
            row.setContentsMargins(10, 6, 10, 6)
            title = QLabel(f"#{cluster.id}  ·  {cluster.detail}")
            title.setWordWrap(True)
            row.addWidget(title)
            card.clicked.connect(lambda _box, c=card: self._select(c))
            self._differences_layout.addWidget(card)
            self._difference_cards.append(card)
        self.differences_title.setVisible(bool(clusters))
        self._sync_scroll()

    def _sync_scroll(self) -> None:
        self._differences_scroll.setVisible(bool(self._difference_cards) or bool(self._line_cards))

    def difference_cards(self) -> list[_ClickableCard]:
        return list(self._difference_cards)

    # -- choosing, and stepping with the arrows ---------------------------------------------------------------------
    def _items(self) -> list[_ClickableCard]:
        """What Up / Down step through, in the order shown: each line read, then each difference."""
        return [*self._line_cards, *self._difference_cards]

    def _select(self, card: _ClickableCard, show: bool = True) -> None:
        self._selected = card
        for item in self._items():
            item.set_selected(item is card)
        self._differences_scroll.ensureWidgetVisible(card)
        if show and card.box is not None:
            self.regionSelected.emit(card.box)

    @property
    def has_selection(self) -> bool:
        return self._selected is not None and any(item is self._selected for item in self._items())

    def step(self, delta: int) -> bool:
        """Choose the line or difference `delta` places from the one chosen (shown in the viewer, as clicking it would);
        False at either end, or with nothing chosen."""
        items = self._items()
        if not self.has_selection:
            return False
        index = next(i for i, item in enumerate(items) if item is self._selected) + delta
        if not 0 <= index < len(items):
            return False
        self._select(items[index])
        return True

    def line_cards(self) -> list["_LineCard"]:
        return list(self._line_cards)

    def _set_lines(self, lines: list[BrailleLineComparison]) -> None:
        while self._lines_layout.count():
            item = self._lines_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()
                item.widget().setParent(None)
                item.widget().deleteLater()
        self._line_cards = []
        self._selected = None
        for number, line in enumerate(lines, start=1):
            card = _LineCard(number, line)
            card.clicked.connect(lambda _box, c=card: self._select(c))
            card.cellClicked.connect(lambda box, c=card: (self._select(c, show=False), self.regionSelected.emit(box)))
            self._lines_layout.addWidget(card)
            self._line_cards.append(card)
        self.lines_title.setVisible(bool(lines))
        self._sync_scroll()

    def set_comparison(self, comparison: BrailleComparison | None) -> None:
        self.verdict_card.box = comparison.region if comparison is not None else None
        for key, label in self._count_labels.items():
            label.setText(str(getattr(comparison, key)) if comparison is not None else "–")
        self.note_label.setVisible(False)
        self._set_lines(comparison.lines if comparison is not None else [])

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

        cells, differing = comparison.cells_compared, comparison.cells_differing
        loose = comparison.missing + comparison.added + comparison.moved if not cells else 0
        if comparison.passed:
            if cells:
                lines = len(comparison.lines)
                self._show_verdict(True, f"Braille matches cell for cell ({cells} cells, {lines} line{'s' if lines != 1 else ''}).")
            else:
                self._show_verdict(True, f"Dot patterns match ({comparison.matched} dots).")
            self.summary_pill.setText(f"{TICK} match")
            set_variant(self.summary_pill, "pill", "loaded")
        elif cells:
            dot_level = comparison.missing + comparison.added + comparison.moved
            text = f"{differing} of {cells} Braille cells differ." if differing else f"{dot_level} dot difference{'s' if dot_level != 1 else ''} found."
            self._show_verdict(False, text)
            self.summary_pill.setText(f"{CROSS} {differing or dot_level} differ")
            set_variant(self.summary_pill, "pill", "warn")
        else:
            self._show_verdict(False, f"{loose} dot difference{'s' if loose != 1 else ''} found.")
            self.summary_pill.setText(f"{CROSS} {loose} differ")
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


_STATUS_COLOURS = {"match": ("#15803d", "#f0fdf4"), "moved": ("#b45309", "#fffbeb")}  # (else: red)


def _cell_text(meaning: str, pattern: int | None) -> str:
    if pattern is None:
        return "none"
    said = "a space" if meaning == " " else (f"'{meaning}'" if len(meaning) <= 3 else meaning)
    return f"{said} {glyph(pattern)} ({dots_text(pattern)})"


class _CellChip(QLabel):
    """One cell: its Braille character (the suspect's beside the original's where they differ) and what it says."""

    clicked = Signal(object)  # BoundingBox

    def __init__(self, cell: BrailleCellComparison, parent=None):
        super().__init__(parent)
        self.cell = cell
        border, background = _STATUS_COLOURS.get(cell.status, ("#b91c1c", "#fef2f2"))
        mine = glyph(cell.original_pattern) if cell.original_pattern is not None else "–"
        theirs = glyph(cell.suspect_pattern) if cell.suspect_pattern is not None else "–"
        shown = mine if cell.passed or cell.status == "moved" else f"{mine}→{theirs}"
        meaning = cell.original_meaning if cell.original_pattern is not None else cell.suspect_meaning
        meaning = "␣" if meaning == " " else (meaning if len(meaning) <= 3 else meaning.split()[0])
        self.setText(f'<div style="font-size:18px">{shown}</div><div style="font-size:11px">{meaning}</div>')
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"border: 1px solid {border}; background: {background}; border-radius: 6px; padding: 2px 6px;")
        self.setToolTip(
            f"{cell.status.capitalize()}\nOriginal: {_cell_text(cell.original_meaning, cell.original_pattern)}"
            f"\nSuspect: {_cell_text(cell.suspect_meaning, cell.suspect_pattern)}"
        )
        self.setAccessibleName(cell.status)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self.cell.bbox)
        super().mousePressEvent(event)


class _LineCard(_ClickableCard):
    """One line: pass or fail, what it says in each file, and its cells."""

    cellClicked = Signal(object)  # BoundingBox

    def __init__(self, number: int, line: BrailleLineComparison, parent=None):
        super().__init__(line.bbox, parent)
        self.line = line
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(6)
        heading = QHBoxLayout()
        self.badge = _badge(line.passed)
        heading.addWidget(self.badge)
        title = QLabel(f"Line {number}")
        title.setStyleSheet("font-weight: 700;")
        heading.addWidget(title)
        heading.addStretch(1)
        outer.addLayout(heading)
        for caption, text in (("Original", line.original_text), ("Suspect", line.suspect_text)):
            label = QLabel()
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setWordWrap(True)
            shown = text if text is not None else "<i>no such line</i>"
            colour = "#1f2430" if caption == "Original" or line.passed else "#b91c1c"
            label.setText(f'<span style="color:#6b7280">{caption}&nbsp;&nbsp;</span><span style="color:{colour}">{shown}</span>')
            outer.addWidget(label)
        cells_widget, cells_row = flow_widget(self, spacing=4)
        self.chips: list[_CellChip] = []
        for cell in line.cells:
            chip = _CellChip(cell)
            chip.clicked.connect(self.cellClicked)
            cells_row.addWidget(chip)
            self.chips.append(chip)
        outer.addWidget(cells_widget)
