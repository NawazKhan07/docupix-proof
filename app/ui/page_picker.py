"""Which of a leaflet's pages to compare: a button that opens a list of its pages to tick.

"All pages" or any set of them. The list stays open while pages are ticked, and the choice is announced once, when it
closes -- so ticking three pages is one new comparison, not three. The choice is a page spec ("1,3-5"; "" = every
page), the same one `LeafletConfig.compare_pages` takes.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu, QPushButton


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
