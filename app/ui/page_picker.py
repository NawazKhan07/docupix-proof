"""Which of a leaflet's pages to compare: a button that opens a list of its pages to tick.

"All pages" or any set of them. The list stays open while pages are ticked, and the choice is announced once, when it
closes -- so ticking three pages is one new comparison, not three. The choice is a page spec ("1,3-5"; "" = every
page), the same one `LeafletConfig.compare_pages` takes.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QMenu, QPushButton, QWidget


class _StayOpenMenu(QMenu):
    """A menu that stays open while its tick boxes are clicked."""

    def mouseReleaseEvent(self, event) -> None:
        action = self.activeAction()
        if action is not None and action.isCheckable() and action.isEnabled():
            action.trigger()
            return
        super().mouseReleaseEvent(event)


def page_spec(pages: set[int], count: int) -> str:
    """1-indexed `pages` of `count` as a spec, runs joined ("1-3,5"); "" for all of them (or none: nothing is not a choice)."""
    if not pages or len(pages) >= count:
        return ""
    ordered = sorted(pages)
    parts: list[str] = []
    start = previous = ordered[0]
    for page in ordered[1:] + [None]:
        if page is not None and page == previous + 1:
            previous = page
            continue
        parts.append(str(start) if start == previous else f"{start}-{previous}")
        if page is not None:
            start = previous = page
    return ",".join(parts)


def parse_spec(spec: str, count: int) -> set[int]:
    """The 1-indexed pages a spec names (every page for ""); parts that are not pages are skipped."""
    spec = spec.strip()
    if not spec:
        return set(range(1, count + 1))
    pages: set[int] = set()
    for part in spec.split(","):
        start_text, _, end_text = part.strip().partition("-")
        try:
            start, end = int(start_text), int(end_text) if end_text else int(start_text)
        except ValueError:
            continue
        pages.update(page for page in range(min(start, end), max(start, end) + 1) if 1 <= page <= count)
    return pages or set(range(1, count + 1))


class PagePicker(QPushButton):
    pagesChanged = Signal(str)  # the new spec ("" = every page), once the list is closed

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setToolTip("Which of the original's pages to compare. A page left out is not compared, and not reported as "
                        "missing or added.")
        self._menu = _StayOpenMenu(self)
        self.setMenu(self._menu)
        self._menu.aboutToHide.connect(self._announce)
        self._count = 0
        self._announced = ""
        self._all: QAction | None = None
        self._pages: list[QAction] = []
        self.set_page_count(0)

    # -- state ------------------------------------------------------------------------------------------------
    def set_page_count(self, count: int) -> None:
        """Offer `count` pages (all ticked)."""
        self._count = max(0, count)
        self._menu.clear()
        self._all = self._menu.addAction("All pages")
        self._all.setCheckable(True)
        self._all.triggered.connect(self._on_all)
        self._menu.addSeparator()
        self._pages = []
        for number in range(1, self._count + 1):
            action = self._menu.addAction(f"Page {number}")
            action.setCheckable(True)
            action.triggered.connect(self._on_page)
            self._pages.append(action)
        self.set_spec("")

    def set_spec(self, spec: str) -> None:
        """Show `spec` as the choice, without announcing it (it is already what was compared)."""
        chosen = parse_spec(spec, self._count)
        for number, action in enumerate(self._pages, start=1):
            action.setChecked(number in chosen)
        self._refresh()
        self._announced = self.spec()

    def spec(self) -> str:
        return page_spec({n for n, a in enumerate(self._pages, start=1) if a.isChecked()}, self._count)

    @property
    def page_count(self) -> int:
        return self._count

    # -- interaction --------------------------------------------------------------------------------------------
    def _on_all(self) -> None:
        for action in self._pages:
            action.setChecked(True)
        self._refresh()

    def _on_page(self) -> None:
        if not any(action.isChecked() for action in self._pages):
            self.sender().setChecked(True)  # at least one page is always compared
        self._refresh()

    def _refresh(self) -> None:
        spec = self.spec()
        if self._all is not None:
            self._all.setChecked(spec == "")
        self.setText(f"Pages: {'All' if spec == '' else spec.replace(',', ', ')}")

    def _announce(self) -> None:
        spec = self.spec()
        if spec != self._announced:
            self._announced = spec
            self.pagesChanged.emit(spec)


_BEST_MATCH = -1  # (the list's value for "Best match": Qt cannot look up an item by None)


class PageSelector(QWidget):
    """Which ONE page of a file to compare (every mode but Leaflet): "Page" and a list -- "Best match" (the page the other
    file's artwork is on, found when comparing) and each page's number. Typing a number picks that page. Shown only for a
    file of several pages. The choice is announced once per change: None for "Best match", else the 0-based page."""

    pageChosen = Signal(object)

    def __init__(self, label: str = "Page", with_best_match: bool = True, parent=None):
        super().__init__(parent)
        self._with_best_match = with_best_match
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.label = QLabel(label)
        self.combo = QComboBox()
        self.combo.setToolTip("Which page of this file to compare. Type a number to jump to that page.")
        self.combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        layout.addWidget(self.label)
        layout.addWidget(self.combo)
        self.combo.currentIndexChanged.connect(self._on_changed)
        self._count = 0

    @property
    def page_count(self) -> int:
        return self._count

    def set_page_count(self, count: int) -> None:
        """Offer `count` pages (the choice goes back to "Best match", or page 1 without it)."""
        self._count = count
        self.combo.blockSignals(True)
        self.combo.clear()
        if self._with_best_match:
            self.combo.addItem("Best match", _BEST_MATCH)
        for page in range(count):
            self.combo.addItem(str(page + 1), page)
        self.combo.setCurrentIndex(0)
        self.combo.blockSignals(False)

    @property
    def choice(self) -> int | None:
        data = self.combo.currentData()
        return None if data is None or data == _BEST_MATCH else int(data)

    def set_choice(self, page: int | None) -> None:
        """Show `page` (0-based; None: "Best match") as the choice, without announcing it."""
        index = self.combo.findData(_BEST_MATCH if page is None else page)
        if index < 0:
            return
        self.combo.blockSignals(True)
        self.combo.setCurrentIndex(index)
        self.combo.blockSignals(False)

    def _on_changed(self, _index: int) -> None:
        self.pageChosen.emit(self.choice)
