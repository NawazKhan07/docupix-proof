"""A layout that lays its items out left to right and wraps onto the next line when the width runs out.

A `QHBoxLayout` never wraps: every control in one is width the window's minimum size must include, so a toolbar
that fits a large monitor can force the window wider than a small screen (or a window snapped to half of one).
This layout asks only for its widest single item, and takes as many lines as its width then needs -- so a row of
controls simply wraps instead of pushing the window out or being clipped.
"""

from __future__ import annotations

from PySide6.QtCore import QMargins, QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QLayoutItem, QSizePolicy, QWidget


class FlowLayout(QLayout):
    def __init__(self, parent: QWidget | None = None, spacing: int = 6, line_spacing: int | None = None,
                 align_right: bool = False):
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._spacing = spacing
        self._line_spacing = spacing if line_spacing is None else line_spacing
        self._align_right = align_right  # each line pushed to the right edge (a header's controls)
        self.setContentsMargins(0, 0, 0, 0)

    # -- QLayout's item bookkeeping --------------------------------------------------------------
    def addItem(self, item: QLayoutItem) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    # -- sizing ----------------------------------------------------------------------------------
    def expandingDirections(self) -> Qt.Orientation:
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._arrange(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self) -> QSize:
        """Everything on one line: what it would take never to wrap."""
        width = height = 0
        for item in self._visible():
            hint = item.sizeHint()
            width += hint.width() + (self._spacing if width else 0)
            height = max(height, hint.height())
        margins = self.contentsMargins()
        return QSize(width + margins.left() + margins.right(), height + margins.top() + margins.bottom())

    def minimumSize(self) -> QSize:
        """Only the widest single item: anything narrower than the line wraps."""
        size = QSize()
        for item in self._visible():
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    # -- placing ---------------------------------------------------------------------------------
    def _visible(self) -> list[QLayoutItem]:
        return [item for item in self._items if not (item.widget() is not None and item.widget().isHidden())]

    def _arrange(self, rect: QRect, apply: bool) -> int:
        margins: QMargins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        lines: list[list[tuple[QLayoutItem, QSize]]] = [[]]
        used = 0
        for item in self._visible():
            hint = item.sizeHint().boundedTo(QSize(max(area.width(), 1), 1 << 20)).expandedTo(item.minimumSize())
            needed = hint.width() + (self._spacing if lines[-1] else 0)
            if lines[-1] and used + needed > area.width():
                lines.append([])
                used, needed = 0, hint.width()
            lines[-1].append((item, hint))
            used += needed
        y = area.y()
        for line in lines:
            if not line:
                continue
            line_height = max(hint.height() for _item, hint in line)
            line_width = sum(hint.width() for _item, hint in line) + self._spacing * (len(line) - 1)
            x = area.x() + (max(0, area.width() - line_width) if self._align_right else 0)
            for item, hint in line:
                if apply:
                    # vertically centred on its line, so a checkbox sits level with the buttons beside it
                    item.setGeometry(QRect(QPoint(x, y + (line_height - hint.height()) // 2), hint))
                x += hint.width() + self._spacing
            y += line_height + self._line_spacing
        return y - self._line_spacing - rect.y() + margins.bottom() if any(lines) else 0


def flow_widget(parent: QWidget | None = None, spacing: int = 6, align_right: bool = False) -> tuple[QWidget, FlowLayout]:
    """A plain widget holding a `FlowLayout`, sized to the height its width needs."""
    widget = QWidget(parent)
    layout = FlowLayout(widget, spacing=spacing, align_right=align_right)
    policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
    policy.setHeightForWidth(True)
    widget.setSizePolicy(policy)
    return widget, layout
