"""Shared crop / click-to-trace polygon / freehand-erase canvas (Architecture
§3.1: "a shared InteractiveCanvas widget (QGraphicsView-based) that supports
three tool modes ... Emits normalized (0-1) coordinates so results are
resolution-independent").

The coordinate math (scene point <-> normalized 0-1) and the per-tool state
machines are factored into plain methods (`begin_crop_drag`, `click_point`,
`begin_erase_stroke`, ...) so they're directly testable without simulating
real Qt mouse events; `mousePressEvent`/`mouseMoveEvent`/`mouseReleaseEvent`
are thin translators from Qt events into those methods.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from app.models.isolation import IsolationTool, NormalizedPolygon, NormalizedRect

_CROP_PEN = QPen(QColor(0, 160, 255), 0)
_POLYGON_PEN = QPen(QColor(255, 140, 0), 0)
_ERASE_PEN = QPen(QColor(220, 0, 0, 160), 0)


def _numpy_rgb_to_qpixmap(image: np.ndarray) -> QPixmap:
    height, width = image.shape[:2]
    contiguous = np.ascontiguousarray(image)
    qimage = QImage(contiguous.data, width, height, width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())  # .copy() detaches from the numpy buffer


class InteractiveCanvas(QGraphicsView):
    cropRectChanged = Signal(object)  # NormalizedRect
    pointClicked = Signal(float, float)  # normalized (x, y) -- polygon-trace request
    eraseStrokeAdded = Signal(object)  # NormalizedPolygon

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._image_size = (0, 0)  # (width, height) px
        self.tool = IsolationTool.CROP

        self._crop_item: QGraphicsRectItem | None = None
        self._crop_origin: QPointF | None = None
        self._crop_rect_norm: NormalizedRect | None = None

        self._polygon_item: QGraphicsPolygonItem | None = None
        self._polygon_norm: NormalizedPolygon | None = None

        self._erase_items: list[QGraphicsPathItem] = []
        self._erase_strokes: list[NormalizedPolygon] = []
        self._current_stroke_path: QPainterPath | None = None
        self._current_stroke_item: QGraphicsPathItem | None = None
        self._current_stroke_points: list[tuple[float, float]] = []

    # -- image ---------------------------------------------------------------
    def set_image(self, image: np.ndarray) -> None:
        self._scene.clear()
        self._crop_item = None
        self._polygon_item = None
        self._erase_items = []
        self._crop_rect_norm = None
        self._polygon_norm = None
        pixmap = _numpy_rgb_to_qpixmap(image)
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._image_size = (image.shape[1], image.shape[0])
        self._scene.setSceneRect(QRectF(0, 0, *self._image_size))

    def set_tool(self, tool: IsolationTool) -> None:
        self.tool = tool

    # -- coordinate math (pure, testable) -------------------------------------
    def _clamp_to_image(self, pos: QPointF) -> QPointF:
        width, height = self._image_size
        return QPointF(min(max(pos.x(), 0), width), min(max(pos.y(), 0), height))

    def _to_normalized(self, pos: QPointF) -> tuple[float, float]:
        width, height = self._image_size
        if width == 0 or height == 0:
            return 0.0, 0.0
        clamped = self._clamp_to_image(pos)
        return clamped.x() / width, clamped.y() / height

    # -- crop tool -------------------------------------------------------------
    def begin_crop_drag(self, scene_pos: QPointF) -> None:
        self._crop_origin = self._clamp_to_image(scene_pos)
        if self._crop_item is not None:
            self._scene.removeItem(self._crop_item)
        self._crop_item = QGraphicsRectItem()
        self._crop_item.setPen(_CROP_PEN)
        self._scene.addItem(self._crop_item)
        self._update_crop_rect(scene_pos)

    def update_crop_drag(self, scene_pos: QPointF) -> None:
        if self._crop_origin is None:
            return
        self._update_crop_rect(scene_pos)

    def end_crop_drag(self, scene_pos: QPointF) -> None:
        if self._crop_origin is None:
            return
        self._update_crop_rect(scene_pos)
        self._crop_origin = None
        if self._crop_rect_norm is not None:
            self.cropRectChanged.emit(self._crop_rect_norm)

    def _update_crop_rect(self, scene_pos: QPointF) -> None:
        assert self._crop_origin is not None and self._crop_item is not None
        current = self._clamp_to_image(scene_pos)
        rect = QRectF(self._crop_origin, current).normalized()
        self._crop_item.setRect(rect)
        x0, y0 = self._to_normalized(rect.topLeft())
        x1, y1 = self._to_normalized(rect.bottomRight())
        self._crop_rect_norm = NormalizedRect(x0, y0, x1, y1)

    def crop_rect(self) -> NormalizedRect | None:
        return self._crop_rect_norm

    def clear_crop(self) -> None:
        if self._crop_item is not None:
            self._scene.removeItem(self._crop_item)
        self._crop_item = None
        self._crop_rect_norm = None

    # -- polygon-trace tool -------------------------------------------------------
    def click_point(self, scene_pos: QPointF) -> None:
        x, y = self._to_normalized(scene_pos)
        self.pointClicked.emit(x, y)

    def show_polygon(self, polygon: NormalizedPolygon) -> None:
        width, height = self._image_size
        if self._polygon_item is not None:
            self._scene.removeItem(self._polygon_item)
        qpoly = QPolygonF([QPointF(x * width, y * height) for x, y in polygon.points])
        self._polygon_item = self._scene.addPolygon(qpoly, _POLYGON_PEN)
        self._polygon_norm = polygon

    def polygon(self) -> NormalizedPolygon | None:
        return self._polygon_norm

    def clear_polygon(self) -> None:
        if self._polygon_item is not None:
            self._scene.removeItem(self._polygon_item)
        self._polygon_item = None
        self._polygon_norm = None

    # -- erase tool ------------------------------------------------------------
    def begin_erase_stroke(self, scene_pos: QPointF) -> None:
        clamped = self._clamp_to_image(scene_pos)
        self._current_stroke_points = [self._to_normalized(clamped)]
        self._current_stroke_path = QPainterPath(clamped)
        self._current_stroke_item = self._scene.addPath(self._current_stroke_path, _ERASE_PEN)

    def update_erase_stroke(self, scene_pos: QPointF) -> None:
        if self._current_stroke_path is None:
            return
        clamped = self._clamp_to_image(scene_pos)
        self._current_stroke_path.lineTo(clamped)
        self._current_stroke_points.append(self._to_normalized(clamped))
        assert self._current_stroke_item is not None
        self._current_stroke_item.setPath(self._current_stroke_path)

    def end_erase_stroke(self) -> None:
        if self._current_stroke_path is None or len(self._current_stroke_points) < 2:
            self._current_stroke_path = None
            self._current_stroke_item = None
            return
        stroke = NormalizedPolygon(points=self._current_stroke_points)
        self._erase_strokes.append(stroke)
        assert self._current_stroke_item is not None
        self._erase_items.append(self._current_stroke_item)
        self._current_stroke_path = None
        self._current_stroke_item = None
        self.eraseStrokeAdded.emit(stroke)

    def erase_strokes(self) -> list[NormalizedPolygon]:
        return list(self._erase_strokes)

    def clear_erase_strokes(self) -> None:
        for item in self._erase_items:
            self._scene.removeItem(item)
        self._erase_items = []
        self._erase_strokes = []

    # -- Qt event translation ----------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = self.mapToScene(event.position().toPoint())
        if self.tool == IsolationTool.CROP:
            self.begin_crop_drag(pos)
        elif self.tool == IsolationTool.POLYGON_TRACE:
            self.click_point(pos)
        elif self.tool == IsolationTool.ERASE:
            self.begin_erase_stroke(pos)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = self.mapToScene(event.position().toPoint())
        if self.tool == IsolationTool.CROP:
            self.update_crop_drag(pos)
        elif self.tool == IsolationTool.ERASE:
            self.update_erase_stroke(pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        pos = self.mapToScene(event.position().toPoint())
        if self.tool == IsolationTool.CROP:
            self.end_crop_drag(pos)
        elif self.tool == IsolationTool.ERASE:
            self.end_erase_stroke()
        super().mouseReleaseEvent(event)
