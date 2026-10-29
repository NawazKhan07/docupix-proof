"""A single scrollable, zoomable image pane for the comparison viewer.

Not in the Architecture's §10 module list -- split out of
`comparison_viewport.py` so the viewport can stay about modes and zoom while
this handles pixels, levels-of-detail and difference markers.
"""

from __future__ import annotations

import math
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from app.models.categories import finding_colour
from app.models.diff import BoundingBox, DiffCluster
from app.ui.gestures import enable_zoom_gestures, zoom_from_viewport_event, zoom_from_wheel

_MIN_LEVEL_EDGE_PX = 64  # never reduce an image below this on its long edge
_CLICK_SLOP_PX = 5  # a press-release within this distance is a click, not a drag
_MARKER_Z = 10
_HIGHLIGHT_COLOUR = "#1d64f2"  # a region chosen in a results panel (a symbol, the Braille)
_HIGHLIGHT_PAD_PX = 6
_COMPARED_AREA_Z = 2  # over the page and its detail tile, under the findings
_OUTSIDE_TINT = QColor(255, 255, 255)  # what the comparison left out is erased: plain white, as it was compared


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
    backgroundClicked = Signal()  # a click (not a pan) where there is no finding
    zoomStepRequested = Signal(int)  # +1 / -1, a fixed step (keyboard / buttons)
    zoomRequested = Signal(float, QPointF)  # scale factor about a viewport point (pinch, Ctrl + wheel)
    resized = Signal()
    regionSelected = Signal(QRectF)  # a rectangle dragged out while region selection is on (scene units)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        # No scroll bars: the picture is moved by dragging it and zoomed by pinch / Ctrl + wheel / the zoom buttons.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

        self._stacks: dict[str, ImageStack] = {}
        self._shown_key: str | None = None
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._zoom = 1.0
        self._marker_items: dict[int, QGraphicsItem] = {}  # each finding's box (its first, where it has several)
        self._extra_marker_items: list[QGraphicsItem] = []  # the rest (a Series finding common to several repeats)
        self._halo_items: list[QGraphicsItem] = []
        self._highlight_items: list[QGraphicsRectItem] = []
        self._press_pos = None
        self._selecting = False  # region selection: a drag draws a rectangle instead of panning
        self._region_from: QPointF | None = None
        self._region_item: QGraphicsRectItem | None = None
        self._outside_item: QGraphicsPathItem | None = None
        self._outline_item: QGraphicsPathItem | None = None
        self._compared_area: list[list[tuple[float, float]]] | None = None
        self._show_compared_area = True
        self._block_outline_item: QGraphicsPathItem | None = None
        self._block_outlines: list[list[tuple[float, float]]] | None = None
        self._show_block_outlines = True
        enable_zoom_gestures(self)

        # A screen-resolution tile drawn over the base image once zoomed in
        # (FR-9.1). See `set_detail`.
        self._detail_item: QGraphicsPixmapItem | None = None
        self._detail_pixmaps: dict[str, QPixmap] = {}
        self._detail_origin = (0, 0)
        self._detail_zoom: float | None = None

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
            self._refresh_detail_item()

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

    def set_zoom(self, zoom: float, anchor: QPointF | None = None) -> None:
        """Zoom to `zoom`, keeping the image point under `anchor` (a viewport position) still.

        Without an anchor the view keeps its centre, which is right for the zoom
        buttons; a pinch or wheel zooms about the fingers / cursor instead.
        """
        scene_point = self.mapToScene(anchor.toPoint()) if anchor is not None else None
        self._zoom = zoom
        self.setTransform(QTransform.fromScale(zoom, zoom))
        if scene_point is not None:
            drift = self.mapFromScene(scene_point) - anchor.toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + drift.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + drift.y())
        self._render_shown()  # a different reduced level may now be the right one
        self._refresh_detail_item()  # a tile made for another zoom is now the wrong size

    # -- detail tile (FR-9.1) ------------------------------------------------
    def set_detail(
        self, tiles: dict[str, np.ndarray], device_origin: tuple[int, int], zoom: float
    ) -> None:
        """Show a screen-resolution render of the visible region over the base image.

        `tiles` holds one RGB array per image key the pane can show, so the
        flicker and grayscale modes can alternate without re-rendering.
        `device_origin` is the tile's top-left in device pixels, i.e. its
        position in the image multiplied by `zoom`.

        The tile ignores the view's scaling and is drawn 1:1, so each of its
        pixels lands on exactly one screen pixel -- which is the whole point:
        going through the view's zoom transform again would resample it.
        """
        self._detail_pixmaps = {key: _to_pixmap(array) for key, array in tiles.items()}
        self._detail_origin = device_origin
        self._detail_zoom = zoom
        self._refresh_detail_item()

    def clear_detail(self) -> None:
        self._detail_pixmaps = {}
        self._detail_zoom = None
        self._refresh_detail_item()

    @property
    def has_detail(self) -> bool:
        return self._detail_item is not None and self._detail_item.isVisible()

    def _refresh_detail_item(self) -> None:
        pixmap = self._detail_pixmaps.get(self._shown_key or "")
        current = self._detail_zoom is not None and abs(self._detail_zoom - self._zoom) < 1e-9
        if pixmap is None or not current:
            if self._detail_item is not None:
                self._detail_item.setVisible(False)
            return

        if self._detail_item is None:
            self._detail_item = QGraphicsPixmapItem()
            self._detail_item.setFlag(QGraphicsPixmapItem.GraphicsItemFlag.ItemIgnoresTransformations)
            self._detail_item.setTransformationMode(Qt.TransformationMode.FastTransformation)
            self._detail_item.setZValue(1)  # above the base image, below the markers
            self._scene.addItem(self._detail_item)
        self._detail_item.setPixmap(pixmap)
        origin_x, origin_y = self._detail_origin
        # The item's own size is in device pixels (it ignores the zoom); only
        # its anchor point is in scene units.
        self._detail_item.setPos(origin_x / self._zoom, origin_y / self._zoom)
        self._detail_item.setVisible(True)

    def visible_device_rect(self) -> tuple[int, int, int, int]:
        """The visible region as (x0, y0, x1, y1) in device pixels, i.e. image position x zoom."""
        top_left = self.mapToScene(self.viewport().rect().topLeft())
        bottom_right = self.mapToScene(self.viewport().rect().bottomRight())
        return (
            math.floor(top_left.x() * self._zoom),
            math.floor(top_left.y() * self._zoom),
            math.ceil(bottom_right.x() * self._zoom),
            math.ceil(bottom_right.y() * self._zoom),
        )

    def wheelEvent(self, event: QWheelEvent) -> None:
        request = zoom_from_wheel(event)
        if request is not None:
            self.zoomRequested.emit(*request)
            event.accept()
            return
        super().wheelEvent(event)

    def viewportEvent(self, event: QEvent) -> bool:
        request = zoom_from_viewport_event(self, event)
        if request is not None:
            self.zoomRequested.emit(*request)
            event.accept()
            return True
        return super().viewportEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.resized.emit()

    # -- what was compared -----------------------------------------------------
    def set_compared_area(self, polygons: list[list[tuple[float, float]]] | None) -> None:
        """Erase what the comparison left out (to white) and outline what it covered (None: the whole page was compared)."""
        self._compared_area = polygons
        self._draw_compared_area()

    def show_compared_area(self, visible: bool) -> None:
        self._show_compared_area = visible
        self._draw_compared_area()

    @property
    def compared_area_visible(self) -> bool:
        return self._outside_item is not None and self._outside_item.isVisible()

    def _draw_compared_area(self) -> None:
        for item in (self._outside_item, self._outline_item):
            if item is not None and item.scene() is not None:
                self._scene.removeItem(item)
        self._outside_item = self._outline_item = None
        if not self._compared_area or not self._show_compared_area or self.image_size == (0, 0):
            return

        width, height = self.image_size
        outside = QPainterPath()
        outside.addRect(QRectF(0, 0, width, height))
        outline = QPainterPath()
        for polygon in self._compared_area:
            path = QPainterPath()
            path.addPolygon(QPolygonF([QPointF(x, y) for x, y in polygon]))
            path.closeSubpath()
            outside.addPath(path)  # odd-even: outside the outer boundary is erased, inside a hole is erased again
            outline.addPath(path)
        outside.setFillRule(Qt.FillRule.OddEvenFill)

        self._outside_item = QGraphicsPathItem(outside)
        self._outside_item.setBrush(QBrush(_OUTSIDE_TINT))
        self._outside_item.setPen(QPen(Qt.PenStyle.NoPen))
        self._outside_item.setZValue(_COMPARED_AREA_Z)
        self._outside_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._scene.addItem(self._outside_item)

        pen = QPen(QColor(47, 111, 237), 1.5, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self._outline_item = QGraphicsPathItem(outline)
        self._outline_item.setPen(pen)
        self._outline_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self._outline_item.setZValue(_COMPARED_AREA_Z + 0.1)
        self._outline_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._scene.addItem(self._outline_item)

    # -- block-wise segmentation: each panel's own outline ----------------------
    def set_block_outlines(self, polygons: list[list[tuple[float, float]]] | None) -> None:
        """Outline each panel a block-wise comparison found (None: no blocks, or block alignment is off)."""
        self._block_outlines = polygons
        self._draw_block_outlines()

    def show_block_outlines(self, visible: bool) -> None:
        self._show_block_outlines = visible
        self._draw_block_outlines()

    @property
    def block_outlines_visible(self) -> bool:
        return self._block_outline_item is not None and self._block_outline_item.isVisible()

    def _draw_block_outlines(self) -> None:
        if self._block_outline_item is not None and self._block_outline_item.scene() is not None:
            self._scene.removeItem(self._block_outline_item)
        self._block_outline_item = None
        if not self._block_outlines or not self._show_block_outlines or self.image_size == (0, 0):
            return

        path = QPainterPath()
        for polygon in self._block_outlines:
            sub = QPainterPath()
            sub.addPolygon(QPolygonF([QPointF(x, y) for x, y in polygon]))
            sub.closeSubpath()
            path.addPath(sub)

        pen = QPen(QColor(230, 140, 20), 1.5, Qt.PenStyle.DashLine)  # orange: distinct from the compared-area blue
        pen.setCosmetic(True)
        self._block_outline_item = QGraphicsPathItem(path)
        self._block_outline_item.setPen(pen)
        self._block_outline_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self._block_outline_item.setZValue(_COMPARED_AREA_Z + 0.2)
        self._block_outline_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._scene.addItem(self._block_outline_item)

    # -- markers -------------------------------------------------------------
    def set_markers(
        self,
        clusters: list[DiffCluster],
        selected_id: int | None = None,
        boxes_of: Callable[[DiffCluster], list[BoundingBox]] | None = None,
        by_source: bool = False,
    ) -> None:
        """Draw each finding's box, in its kind's colour (text, colour, graphics, barcode, Braille) -- or, `by_source`,
        in the colour of how it was found (OCR or pixels).

        Every pane draws the *same* shared-space box where both views are drawn from the one aligned pair, so a marker
        cannot land in different places in different panes (FR-7.4). Where this pane's picture is another frame (Series:
        the whole sheet), `boxes_of` says where each finding is on it -- one place or several.
        """
        for item in [*self._marker_items.values(), *self._extra_marker_items, *self._halo_items]:
            self._scene.removeItem(item)
        self._marker_items.clear()
        self._extra_marker_items = []
        self._halo_items.clear()

        for cluster in clusters:
            colour = QColor(finding_colour(cluster, by_source))
            is_selected = cluster.id == selected_id
            boxes = boxes_of(cluster) if boxes_of is not None else [cluster.bbox_in_original]
            for index, box in enumerate(boxes):
                item = QGraphicsRectItem(box.x0, box.y0, box.width, box.height)
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
                if index == 0:
                    self._marker_items[cluster.id] = item
                else:
                    self._extra_marker_items.append(item)

                # A colour alone vanishes over artwork of the same colour
                # (a red marker on a red block), so each marker sits on a white halo.
                halo = QGraphicsRectItem(box.x0, box.y0, box.width, box.height)
                halo_pen = QPen(QColor(255, 255, 255, 230), pen.widthF() + 3)
                halo_pen.setCosmetic(True)
                halo.setPen(halo_pen)
                halo.setZValue(item.zValue() - 0.5)
                halo.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                self._scene.addItem(halo)
                self._halo_items.append(halo)

    @property
    def marker_count(self) -> int:
        return len(self._marker_items)

    def set_highlight(self, box: BoundingBox | None) -> None:
        """Mark one region (a decoded symbol, the Braille, a dot difference) that was just chosen in a results panel:
        a bold dashed outline on a white halo, a little outside the region, over everything else. None clears it."""
        for item in self._highlight_items:
            self._scene.removeItem(item)
        self._highlight_items = []
        if box is None:
            return
        pad = _HIGHLIGHT_PAD_PX
        rect = (box.x0 - pad, box.y0 - pad, box.width + 2 * pad, box.height + 2 * pad)
        halo = QGraphicsRectItem(*rect)
        halo_pen = QPen(QColor(255, 255, 255, 235), 7)
        halo_pen.setCosmetic(True)
        halo.setPen(halo_pen)
        outline = QGraphicsRectItem(*rect)
        outline_pen = QPen(QColor(_HIGHLIGHT_COLOUR), 3, Qt.PenStyle.DashLine)
        outline_pen.setCosmetic(True)
        outline.setPen(outline_pen)
        fill = QColor(_HIGHLIGHT_COLOUR)
        fill.setAlpha(35)
        outline.setBrush(fill)
        for z, item in enumerate((halo, outline)):
            item.setZValue(_MARKER_Z + 10 + z)
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self._scene.addItem(item)
            self._highlight_items.append(item)

    @property
    def highlighted(self) -> bool:
        return bool(self._highlight_items)

    def center_on_box(self, box: BoundingBox) -> None:
        cx, cy = box.center
        self.centerOn(QPointF(cx, cy))

    # -- selecting a region --------------------------------------------------
    @property
    def selecting_region(self) -> bool:
        return self._selecting

    def set_region_selection(self, on: bool) -> None:
        """While on, dragging draws a rectangle (reported as `regionSelected`) instead of panning the page."""
        self._selecting = bool(on)
        self._cancel_region()
        self.setDragMode(QGraphicsView.DragMode.NoDrag if on else QGraphicsView.DragMode.ScrollHandDrag)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)

    def _cancel_region(self) -> None:
        if self._region_item is not None and self._region_item.scene() is not None:
            self._scene.removeItem(self._region_item)
        self._region_item = None
        self._region_from = None

    def _begin_region(self, scene_pos: QPointF) -> None:
        self._cancel_region()
        self._region_from = scene_pos
        pen = QPen(QColor(47, 111, 237), 2, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self._region_item = self._scene.addRect(QRectF(scene_pos, scene_pos), pen, QBrush(QColor(47, 111, 237, 40)))
        self._region_item.setZValue(_MARKER_Z + 5)
        self._region_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def _update_region(self, scene_pos: QPointF) -> None:
        if self._region_from is not None and self._region_item is not None:
            self._region_item.setRect(QRectF(self._region_from, scene_pos).normalized())

    def _end_region(self, scene_pos: QPointF) -> None:
        if self._region_from is None:
            return
        rect = QRectF(self._region_from, scene_pos).normalized()
        self._cancel_region()
        if rect.width() * self.transform().m11() >= _CLICK_SLOP_PX or rect.height() * self.transform().m22() >= _CLICK_SLOP_PX:
            self.regionSelected.emit(rect)

    # -- click-to-select -----------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._selecting and event.button() == Qt.MouseButton.LeftButton:
            self._begin_region(self.mapToScene(event.position().toPoint()))
            event.accept()
            return
        self._press_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._selecting and self._region_from is not None:
            self._update_region(self.mapToScene(event.position().toPoint()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._selecting and self._region_from is not None:
            self._end_region(self.mapToScene(event.position().toPoint()))
            event.accept()
            return
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
                under_cursor, key=lambda item: item.boundingRect().width() * item.boundingRect().height()
            )
            self.markerClicked.emit(int(smallest.data(0)))
        else:
            self.backgroundClicked.emit()

