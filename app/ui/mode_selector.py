"""Comparison mode selector (SRS §2.3). Mode is a parameter object handed to
the job, never a UI-level code fork (Architecture §3.1)."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
)

from app.models.job import MODE_DISPLAY_NAMES, ComparisonMode
from app.ui.theme import set_variant

# One short line on the card itself (six long paragraphs made the page too tall for a laptop screen); the
# fuller account is the card's tooltip.
_MODE_SUMMARIES = {
    ComparisonMode.AUTOMATIC: "Finds the artwork and compares everything: text, graphics and colour.",
    ComparisonMode.SCAN_VS_DIGITAL: "A scan or photo against the digital file: colour ignored, text read and compared.",
    ComparisonMode.BARCODE: "Only barcodes, Pharmacodes and QR codes, decoded and compared.",
    ComparisonMode.BRAILLE: "Only the Braille dots, read and compared.",
    ComparisonMode.SERIES: "One artwork against a printed sheet of its repeats, each compared.",
    ComparisonMode.LEAFLET: "Multi-page leaflets and inserts, open or closed, page by page.",
}

_MODE_DESCRIPTIONS = {
    ComparisonMode.AUTOMATIC: "Auto-detects the artwork area and compares every difference category.",
    ComparisonMode.SCAN_VS_DIGITAL: "Ignores colour: both files become black ink on white, and their text and numbers are read and compared, with barcode bars as marks.",
    ComparisonMode.BARCODE: "Restricts comparison to detected barcode/pharmacode/QR symbols, decoded and compared.",
    ComparisonMode.BRAILLE: "Restricts comparison to the coloured dot-matrix representing Braille content.",
    ComparisonMode.SERIES: "Original: one artwork, cropped or lifted as usual. Suspect: a sheet printed with several "
    "repeats of it -- every repeat is found automatically and compared against the original on its own.",
    ComparisonMode.LEAFLET: "For a multi-page leaflet or insert, open or closed, in any language. Pages are matched "
    "by what is on them (so an inserted or missing page does not throw the rest out of step) and every page is "
    "compared in full.",
}

_COLUMNS = 2  # two cards per row: six cards stacked in one column ran taller than the window on a modest screen
_CARD_MIN_WIDTH_PX = 190  # a card narrower than this wraps its description into too many lines
_CARD_MAX_WIDTH_PX = 360  # ...and one wider than this is mostly empty space on a large monitor

_MODE_ORDER = [
    ComparisonMode.AUTOMATIC,
    ComparisonMode.SCAN_VS_DIGITAL,
    ComparisonMode.BARCODE,
    ComparisonMode.BRAILLE,
    ComparisonMode.SERIES,
    ComparisonMode.LEAFLET,
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
        # As wide as the window allows, within limits: the description wraps to whatever width the card gets.
        self.setMaximumWidth(_CARD_MAX_WIDTH_PX)
        policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

        self.radio = QRadioButton(MODE_DISPLAY_NAMES[mode])
        group.addButton(self.radio)

        self.setToolTip(_MODE_DESCRIPTIONS[mode])
        self.description = QLabel(_MODE_SUMMARIES[mode])
        self.description.setProperty("secondary", "true")
        self.description.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(3)
        layout.addStretch(1)
        layout.addWidget(self.radio)
        layout.addWidget(self.description)
        layout.addStretch(1)

        self.radio.toggled.connect(self._refresh_selected_style)
        self._refresh_selected_style()
        # never so narrow that the mode's own name is cut off
        self.setMinimumWidth(max(_CARD_MIN_WIDTH_PX, self.radio.sizeHint().width() + 30))

    def _refresh_selected_style(self) -> None:
        set_variant(self, "selected", "true" if self.radio.isChecked() else "false")

    def mousePressEvent(self, event) -> None:
        self.radio.setChecked(True)
        super().mousePressEvent(event)

    def resizeEvent(self, event) -> None:
        """A word-wrapped QLabel nested two layouts deep (card -> group box -> the setup page's row) does not
        reliably get asked its height at the width it will actually end up with -- Qt's first layout pass can
        settle on a width narrower than final and never revisit the label's height, so a long description (e.g.
        Scan vs Digital, Series) is given too little room and paints over the next card. Once a real resize
        hands the label its true width, `heightForWidth` at that width is reliable; pinning the label's minimum
        height to it makes this card (and so the panel) report its correct size on the next layout pass."""
        super().resizeEvent(event)
        wanted = self.description.heightForWidth(self.description.width())
        if wanted > 0 and self.description.minimumHeight() != wanted:
            self.description.setMinimumHeight(wanted)


class ModeSelector(QGroupBox):
    """Selectable cards for the four comparison modes. Automatic is default."""

    modeChanged = Signal(ComparisonMode)

    def __init__(self, parent=None):
        super().__init__("Comparison Mode", parent)
        self._button_group = QButtonGroup(self)
        # Two cards per row: six cards in one column ran taller than a modest window, forcing a scroll (or
        # worse, clipping the Compare button below off the bottom of the window with no way to reach it) --
        # a grid trades some of that height for width the setup page already has room for.
        layout = QGridLayout(self)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)

        self._cards: dict[ComparisonMode, _ModeCard] = {}
        for index, mode in enumerate(_MODE_ORDER):
            card = _ModeCard(mode, self._button_group, self)
            layout.addWidget(card, index // _COLUMNS, index % _COLUMNS)
            self._cards[mode] = card
            card.radio.toggled.connect(self._on_toggled(mode))
        # The cards share the panel's height evenly (their text centred in them), so the panel is filled, not
        # left with a blank band under three short rows.
        for row in range(len(_MODE_ORDER) // _COLUMNS):
            layout.setRowStretch(row, 1)

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
