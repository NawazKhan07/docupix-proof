"""A single scrollable, zoomable image pane for the comparison viewer.

Not in the Architecture's §10 module list -- split out of
`comparison_viewport.py` so the viewport can stay about modes and zoom while
this handles pixels, levels-of-detail and difference markers.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap, QTransform, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from app.engine.aggregation.severity import SEVERITY_COLOURS
from app.models.diff import BoundingBox, DiffCluster

_MIN_LEVEL_EDGE_PX = 64  # never reduce an image below this on its long edge
_CLICK_SLOP_PX = 5  # a press-release within this distance is a click, not a drag
_MARKER_Z = 10


def _to_pixmap(rgb: np.ndarray) -> QPixmap:
    height, width = rgb.shape[:2]
    contiguous = np.ascontiguousarray(rgb)
    image = QImage(contiguous.data, width, height, width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())  # .copy() detaches from the numpy buffer


class ImageStack:
    """One image plus lazily-built, area-averaged reductions of it.

    Showing a large page at a fraction of its size through Qt's bilinear
    scaling drops thin lines -- a dieline or a fine rule can vanish entirely
    at 30% -- that a proper area average keeps. So instead of always scaling
    the full-size bitmap, the pane picks the reduced level closest above the
    current zoom and scales *that* by at most 2x.

    Stacks are immutable once built and are shared between panes, so a page
    shown in two views is reduced and converted once.
    """

    def __init__(self, rgb: np.ndarray):
        self.height, self.width = rgb.shape[:2]
        self._arrays: dict[int, np.ndarray] = {0: rgb}
        self._pixmaps: dict[int, QPixmap] = {}

    @property
    def max_level(self) -> int:
        longest = max(self.width, self.height)
        return max(0, int(math.floor(math.log2(longest / _MIN_LEVEL_EDGE_PX))))

    def level_for_zoom(self, zoom: float) -> int:
        """The most-reduced level that is still at least as large as the display."""
        if zoom >= 1.0:
            return 0
        return min(self.max_level, int(math.floor(math.log2(1.0 / zoom))))

    def _array(self, level: int) -> np.ndarray:
        if level not in self._arrays:
            previous = self._array(level - 1)
            size = (max(1, previous.shape[1] // 2), max(1, previous.shape[0] // 2))
            self._arrays[level] = cv2.resize(previous, size, interpolation=cv2.INTER_AREA)
        return self._arrays[level]

    def pixmap(self, level: int) -> tuple[QPixmap, float, float]:
        """The level's pixmap and the (x, y) factors that scale it back to full size."""
        if level not in self._pixmaps:
            self._pixmaps[level] = _to_pixmap(self._array(level))
        pixmap = self._pixmaps[level]
        return pixmap, self.width / pixmap.width(), self.height / pixmap.height()


class PaneView(QGraphicsView):
    markerClicked = Signal(int)  # cluster id
    zoomStepRequested = Signal(int)  # +1 / -1, from Ctrl + wheel
    resized = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

        self._stacks: dict[str, ImageStack] = {}
        self._shown_key: str | None = None
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._zoom = 1.0
        self._marker_items: dict[int, QGraphicsRectItem] = {}
        self._press_pos = None

    # -- images --------------------------------------------------------------
    def set_stacks(self, stacks: dict[str, ImageStack]) -> None:
        """Replace the images this pane can show. Keeps zoom and scroll position."""
        self._stacks = dict(stacks)
        if self._shown_key not in self._stacks:
            self._shown_key = next(iter(self._stacks), None)
        self._render_shown()

    def show_stack(self, key: str) -> None:
        if key != self._shown_key and key in self._stacks:
            self._shown_key = key
            self._render_shown()

    @property
    def shown_key(self) -> str | None:
        return self._shown_key

    @property
    def image_size(self) -> tuple[int, int]:
        """(width, height) of the image in scene units."""
        stack = self._stacks.get(self._shown_key) if self._shown_key else None
        return (stack.width, stack.height) if stack else (0, 0)

    def _render_shown(self) -> None:
        stack = self._stacks.get(self._shown_key) if self._shown_key else None
        if stack is None:
            return

        level = stack.level_for_zoom(self._zoom)
        pixmap, scale_x, scale_y = stack.pixmap(level)

        if self._pixmap_item is None:
            self._pixmap_item = self._scene.addPixmap(pixmap)
            self._pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
            self._pixmap_item.setZValue(0)
        else:
            self._pixmap_item.setPixmap(pixmap)
        self._pixmap_item.setTransform(QTransform.fromScale(scale_x, scale_y))
        self._scene.setSceneRect(QRectF(0, 0, stack.width, stack.height))

    # -- zoom ----------------------------------------------------------------
    @property
    def zoom(self) -> float:
        return self._zoom

    def set_zoom(self, zoom: float) -> None:
        self._zoom = zoom
        self.setTransform(QTransform.fromScale(zoom, zoom))
        self._render_shown()  # a different reduced level may now be the right one

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoomStepRequested.emit(1 if event.angleDelta().y() > 0 else -1)
            event.accept()
            return
        super().wheelEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.resized.emit()

    # -- markers -------------------------------------------------------------
    def set_markers(self, clusters: list[DiffCluster], selected_id: int | None = None) -> None:
        """Draw one box per finding, in its severity colour (FR-7.2).

        Every pane draws the *same* shared-space box: both views are drawn from
        the one aligned pair, so a marker cannot land in different places in
        different panes (FR-7.4).
        """
        for item in self._marker_items.values():
            self._scene.removeItem(item)
        self._marker_items.clear()

        for cluster in clusters:
            box = cluster.bbox_in_original
            item = QGraphicsRectItem(box.x0, box.y0, box.width, box.height)
            colour = QColor(SEVERITY_COLOURS[cluster.severity])
            is_selected = cluster.id == selected_id

            pen = QPen(colour, 3 if is_selected else 2)
            pen.setCosmetic(True)  # constant on-screen width at any zoom
            item.setPen(pen)
            if is_selected:
                fill = QColor(colour)
                fill.setAlpha(60)
                item.setBrush(fill)
            item.setZValue(_MARKER_Z + (1 if is_selected else 0))
            item.setData(0, cluster.id)
            self._scene.addItem(item)
            self._marker_items[cluster.id] = item

    @property
    def marker_count(self) -> int:
        return len(self._marker_items)

    def center_on_box(self, box: BoundingBox) -> None:
        cx, cy = box.center
        self.centerOn(QPointF(cx, cy))

    # -- click-to-select -----------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._press_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        super().mouseReleaseEvent(event)
        if self._press_pos is None:
            return
        release_pos = event.position().toPoint()
        moved = (release_pos - self._press_pos).manhattanLength()
        self._press_pos = None
        if moved > _CLICK_SLOP_PX:
            return  # that was a pan, not a click

        # A marker's whole interior is clickable, so a large finding can sit
        # over a small one nested inside it. Prefer the smallest under the
        # cursor, which is the one the reviewer is pointing at.
        under_cursor = [
            item for item in self.items(release_pos) if item.data(0) is not None
        ]
        if under_cursor:
            smallest = min(
                under_cursor, key=lambda item: item.rect().width() * item.rect().height()
            )
            self.markerClicked.emit(int(smallest.data(0)))
