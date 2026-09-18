"""Shared crop / click-to-trace polygon / freehand-erase canvas (Architecture
§3.1: "a shared InteractiveCanvas widget (QGraphicsView-based) that supports
three tool modes ... Emits normalized (0-1) coordinates so results are
resolution-independent").

The coordinate math (scene point <-> normalized 0-1) and the per-tool state
machines are factored into plain methods (`begin_crop_drag`, `click_point`,
`begin_erase_stroke`, ...) so they're directly testable without simulating
real Qt mouse events; `mousePressEvent`/`mouseMoveEvent`/`mouseReleaseEvent`
are thin translators from Qt events into those methods.

Working on a large artwork sheet needs three things a bare view does not give:
the whole page visible when it opens, zoom that follows a pinch or the wheel,
and a way to move around without disturbing the selection. So the canvas opens
fitted to the window, zooms about the cursor / fingers, pans with the middle
button or Space + drag (or two-finger scroll), and the crop rectangle can be
grabbed by any edge or corner to resize, or by its body to move.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QKeyEvent,
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
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from app.models.isolation import IsolationTool, NormalizedPolygon, NormalizedRect
from app.ui.gestures import enable_zoom_gestures, zoom_from_viewport_event, zoom_from_wheel

_CROP_COLOUR = QColor(47, 111, 237)
_CROP_PEN = QPen(_CROP_COLOUR, 2)
_CROP_PEN.setCosmetic(True)  # constant on-screen width at any zoom
_DIM_BRUSH = QBrush(QColor(15, 23, 42, 120))  # the part of the page outside the crop
_POLYGON_PEN = QPen(QColor(255, 140, 0), 0)
_ERASE_PEN = QPen(QColor(220, 0, 0, 160), 0)

_HANDLE_PX = 10  # on-screen size of a resize handle
_GRAB_PX = 9  # how close (on screen) to an edge counts as grabbing it
_MIN_CROP_PX = 4  # a selection smaller than this (in image px) was a click, not a drag
_ZOOM_STEP = 1.25
_MIN_ZOOM = 0.02
_MAX_ZOOM = 16.0
_FIT_MARGIN_PX = 8

# Which edges a grab moves: any of "l", "r", "t", "b"; "move" is the whole rectangle.
_CURSORS = {
    "l": Qt.CursorShape.SizeHorCursor,
    "r": Qt.CursorShape.SizeHorCursor,
    "t": Qt.CursorShape.SizeVerCursor,
    "b": Qt.CursorShape.SizeVerCursor,
    "lt": Qt.CursorShape.SizeFDiagCursor,
    "rb": Qt.CursorShape.SizeFDiagCursor,
    "rt": Qt.CursorShape.SizeBDiagCursor,
    "lb": Qt.CursorShape.SizeBDiagCursor,
    "move": Qt.CursorShape.SizeAllCursor,
}


def _numpy_rgb_to_qpixmap(image: np.ndarray) -> QPixmap:
    height, width = image.shape[:2]
    contiguous = np.ascontiguousarray(image)
    qimage = QImage(contiguous.data, width, height, width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())  # .copy() detaches from the numpy buffer


class InteractiveCanvas(QGraphicsView):
    cropRectChanged = Signal(object)  # NormalizedRect, or None when the selection was cleared
    cropRectAdjusting = Signal(object)  # NormalizedRect, continuously while it is being dragged
    pointClicked = Signal(float, float)  # normalized (x, y) -- polygon-trace request
    eraseStrokeAdded = Signal(object)  # NormalizedPolygon
    zoomChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform | QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.viewport().setMouseTracking(True)  # so the cursor can show what a drag would do
        enable_zoom_gestures(self)

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._image_size = (0, 0)  # (width, height) px
        self.tool = IsolationTool.CROP

        self._zoom = 1.0
        self._fit_mode = True
        self._pan_from: QPointF | None = None
        self._space_down = False

        # Crop: one rectangle in scene units, adjustable until it is saved.
        self._crop: QRectF | None = None
        self._crop_grab: str | None = None  # "new", "move", or the edges being dragged
        self._crop_grab_scene = QPointF()
        self._crop_grab_rect = QRectF()
        self._crop_item: QGraphicsRectItem | None = None
        self._dim_item: QGraphicsPathItem | None = None
        self._handle_items: list[QGraphicsRectItem] = []
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
        self._dim_item = None
        self._handle_items = []
        self._polygon_item = None
        self._erase_items = []
        self._crop = None
        self._crop_rect_norm = None
        self._polygon_norm = None
        pixmap = _numpy_rgb_to_qpixmap(image)
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._image_size = (image.shape[1], image.shape[0])
        self._scene.setSceneRect(QRectF(0, 0, *self._image_size))
        self.fit_to_view()

    def set_tool(self, tool: IsolationTool) -> None:
        self.tool = tool
        self._update_cursor(None)

    # -- zoom ------------------------------------------------------------------
    @property
    def zoom(self) -> float:
        return self._zoom

    @property
    def zoom_percent(self) -> int:
        return round(self._zoom * 100)

    def _fit_zoom(self) -> float | None:
        width, height = self._image_size
        viewport = self.viewport()
        if width <= 0 or height <= 0 or viewport.width() <= 0 or viewport.height() <= 0:
            return None  # not laid out yet; the resize / show event will call back in
        return min(
            (viewport.width() - _FIT_MARGIN_PX) / width,
            (viewport.height() - _FIT_MARGIN_PX) / height,
        )

    def fit_to_view(self) -> None:
        """Show the whole page (and keep showing it as the window is resized)."""
        self._fit_mode = True
        fit = self._fit_zoom()
        if fit is None:
            return
        self._apply_zoom(fit)
        width, height = self._image_size
        self.centerOn(QPointF(width / 2, height / 2))

    def set_zoom(self, zoom: float, anchor: QPointF | None = None) -> None:
        """Zoom, keeping the page point under `anchor` (a viewport position) still."""
        self._fit_mode = False
        self._apply_zoom(zoom, anchor)

    def _apply_zoom(self, zoom: float, anchor: QPointF | None = None) -> None:
        fit = self._fit_zoom()
        floor = min(_MIN_ZOOM, fit) if fit else _MIN_ZOOM
        zoom = max(floor, min(_MAX_ZOOM, zoom))
        if anchor is None:
            centre = self.viewport().rect().center()
            anchor = QPointF(centre)
        scene_point = self.mapToScene(anchor.toPoint())
        self._zoom = zoom
        self.setTransform(QTransform.fromScale(zoom, zoom))
        drift = self.mapFromScene(scene_point) - anchor.toPoint()
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + drift.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() + drift.y())
        self._layout_crop_handles()  # the grab zone and handles are defined in screen pixels
        self.zoomChanged.emit(zoom)

    def zoom_in(self) -> None:
        self.set_zoom(self._zoom * _ZOOM_STEP)

    def zoom_out(self) -> None:
        self.set_zoom(self._zoom / _ZOOM_STEP)

    def zoom_actual(self) -> None:
        self.set_zoom(1.0)

    def zoom_to_selection(self) -> None:
        """Zoom so the crop rectangle fills the view, for checking its edges."""
        if self._crop is None or self._crop.isEmpty():
            return
        viewport = self.viewport()
        fit = min(
            (viewport.width() - _FIT_MARGIN_PX) / self._crop.width(),
            (viewport.height() - _FIT_MARGIN_PX) / self._crop.height(),
        )
        self.set_zoom(fit)
        self.centerOn(self._crop.center())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._fit_mode:
            self.fit_to_view()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._fit_mode:
            self.fit_to_view()

    def wheelEvent(self, event: QWheelEvent) -> None:
        request = zoom_from_wheel(event)
        if request is not None:
            self.set_zoom(self._zoom * request[0], request[1])
            event.accept()
            return
        super().wheelEvent(event)  # plain wheel / two-finger drag scrolls

    def viewportEvent(self, event: QEvent) -> bool:
        request = zoom_from_viewport_event(self, event)
        if request is not None:
            self.set_zoom(self._zoom * request[0], request[1])
            event.accept()
            return True
        return super().viewportEvent(event)

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
    def hit_test_crop(self, scene_pos: QPointF) -> str | None:
        """What a press at `scene_pos` would grab: an edge/corner ("l", "rb", ...), "move", or None."""
        if self._crop is None:
            return None
        tolerance = _GRAB_PX / max(self._zoom, 1e-6)
        rect = self._crop
        near_x = rect.left() - tolerance <= scene_pos.x() <= rect.right() + tolerance
        near_y = rect.top() - tolerance <= scene_pos.y() <= rect.bottom() + tolerance
        if not (near_x and near_y):
            return None

        # On a tiny rectangle the grab zones would swallow the whole thing, so
        # they are capped at a third of its size and its body stays movable.
        tol_x = min(tolerance, rect.width() / 3)
        tol_y = min(tolerance, rect.height() / 3)
        grab = ""
        if abs(scene_pos.x() - rect.left()) <= tol_x:
            grab += "l"
        elif abs(scene_pos.x() - rect.right()) <= tol_x:
            grab += "r"
        if abs(scene_pos.y() - rect.top()) <= tol_y:
            grab += "t"
        elif abs(scene_pos.y() - rect.bottom()) <= tol_y:
            grab += "b"
        if grab:
            return grab
        return "move" if rect.contains(scene_pos) else None

    def begin_crop_drag(self, scene_pos: QPointF) -> None:
        """Start a crop gesture: grab an edge/corner or the body of the existing
        selection, or -- pressing anywhere else -- start a new one."""
        grab = self.hit_test_crop(scene_pos)
        self._crop_grab_scene = self._clamp_to_image(scene_pos)
        if grab is None:
            self._crop_grab = "new"
            self._set_crop(QRectF(self._crop_grab_scene, self._crop_grab_scene))
        else:
            self._crop_grab = grab
            self._crop_grab_rect = QRectF(self._crop)

    def update_crop_drag(self, scene_pos: QPointF) -> None:
        if self._crop_grab is None:
            return
        current = self._clamp_to_image(scene_pos)
        if self._crop_grab == "new":
            rect = QRectF(self._crop_grab_scene, current).normalized()
        elif self._crop_grab == "move":
            rect = self._moved(current - self._crop_grab_scene)
        else:
            rect = self._resized(current)
        self._set_crop(rect)
        if self._crop_rect_norm is not None:
            self.cropRectAdjusting.emit(self._crop_rect_norm)

    def end_crop_drag(self, scene_pos: QPointF) -> None:
        if self._crop_grab is None:
            return
        self.update_crop_drag(scene_pos)
        self._crop_grab = None
        if self._crop is not None and (self._crop.width() < _MIN_CROP_PX or self._crop.height() < _MIN_CROP_PX):
            self.clear_crop()  # a click, not a drag
            self.cropRectChanged.emit(None)
            return
        self.cropRectChanged.emit(self._crop_rect_norm)

    def _moved(self, delta: QPointF) -> QRectF:
        """The grabbed rectangle moved by `delta`, kept inside the page."""
        width, height = self._image_size
        rect = QRectF(self._crop_grab_rect)
        dx = min(max(delta.x(), -rect.left()), width - rect.right())
        dy = min(max(delta.y(), -rect.top()), height - rect.bottom())
        return rect.translated(dx, dy)

    def _resized(self, current: QPointF) -> QRectF:
        """The grabbed rectangle with the grabbed edges pulled to `current`."""
        rect = self._crop_grab_rect
        left, top, right, bottom = rect.left(), rect.top(), rect.right(), rect.bottom()
        if "l" in self._crop_grab:
            left = current.x()
        if "r" in self._crop_grab:
            right = current.x()
        if "t" in self._crop_grab:
            top = current.y()
        if "b" in self._crop_grab:
            bottom = current.y()
        return QRectF(QPointF(left, top), QPointF(right, bottom)).normalized()  # pulling past the far edge flips it

    def _set_crop(self, rect: QRectF) -> None:
        self._crop = rect
        x0, y0 = self._to_normalized(rect.topLeft())
        x1, y1 = self._to_normalized(rect.bottomRight())
        self._crop_rect_norm = NormalizedRect(x0, y0, x1, y1)
        self._draw_crop()

    def set_crop_rect(self, rect: NormalizedRect) -> None:
        """Place the selection programmatically (e.g. to restore one)."""
        width, height = self._image_size
        self._set_crop(QRectF(rect.x0 * width, rect.y0 * height, (rect.x1 - rect.x0) * width, (rect.y1 - rect.y0) * height))
        self.cropRectChanged.emit(self._crop_rect_norm)

    def crop_rect(self) -> NormalizedRect | None:
        return self._crop_rect_norm

    def crop_size_px(self) -> tuple[int, int] | None:
        """(width, height) of the selection in image pixels."""
        if self._crop is None:
            return None
        return round(self._crop.width()), round(self._crop.height())

    def clear_crop(self) -> None:
        self._crop = None
        self._crop_rect_norm = None
        self._crop_grab = None
        for item in (self._crop_item, self._dim_item, *self._handle_items):
            if item is not None and item.scene() is not None:
                self._scene.removeItem(item)
        self._crop_item = None
        self._dim_item = None
        self._handle_items = []

    # The overlay: the page outside the selection dimmed, the selection outlined,
    # and a square handle on each corner and edge midpoint.
    def _draw_crop(self) -> None:
        if self._crop is None:
            return
        if self._crop_item is None:
            self._dim_item = QGraphicsPathItem()
            self._dim_item.setBrush(_DIM_BRUSH)
            self._dim_item.setPen(QPen(Qt.PenStyle.NoPen))
            self._dim_item.setZValue(5)
            self._dim_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self._scene.addItem(self._dim_item)

            self._crop_item = QGraphicsRectItem()
            self._crop_item.setPen(_CROP_PEN)
            self._crop_item.setZValue(6)
            self._crop_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self._scene.addItem(self._crop_item)

            for _ in range(8):
                handle = QGraphicsRectItem(-_HANDLE_PX / 2, -_HANDLE_PX / 2, _HANDLE_PX, _HANDLE_PX)
                handle.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)  # constant size
                handle.setBrush(QBrush(QColor(255, 255, 255)))
                handle.setPen(_CROP_PEN)
                handle.setZValue(7)
                handle.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                self._scene.addItem(handle)
                self._handle_items.append(handle)

        self._crop_item.setRect(self._crop)
        outside = QPainterPath()
        outside.addRect(self._scene.sceneRect())
        outside.addRect(self._crop)
        outside.setFillRule(Qt.FillRule.OddEvenFill)
        self._dim_item.setPath(outside)
        self._layout_crop_handles()

    def _layout_crop_handles(self) -> None:
        if self._crop is None or not self._handle_items:
            return
        rect = self._crop
        centre = rect.center()
        anchors = [
            rect.topLeft(), QPointF(centre.x(), rect.top()), rect.topRight(),
            QPointF(rect.right(), centre.y()), rect.bottomRight(),
            QPointF(centre.x(), rect.bottom()), rect.bottomLeft(), QPointF(rect.left(), centre.y()),
        ]
        for handle, anchor in zip(self._handle_items, anchors):
            handle.setPos(anchor)
        # A selection too small to show eight handles clearly shows only the corners.
        show_all = rect.width() * self._zoom > 4 * _HANDLE_PX and rect.height() * self._zoom > 4 * _HANDLE_PX
        for index, handle in enumerate(self._handle_items):
            handle.setVisible(show_all or index % 2 == 0)

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

    # -- panning ---------------------------------------------------------------
    def _wants_pan(self, event: QMouseEvent) -> bool:
        return event.button() == Qt.MouseButton.MiddleButton or (
            self._space_down and event.button() == Qt.MouseButton.LeftButton
        )

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_down = True
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Delete) and self.tool == IsolationTool.CROP:
            if self._crop is not None:
                self.clear_crop()
                self.cropRectChanged.emit(None)
                event.accept()
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_down = False
            self._update_cursor(None)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _update_cursor(self, scene_pos: QPointF | None) -> None:
        if self._space_down:
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        elif self.tool == IsolationTool.CROP:
            grab = self.hit_test_crop(scene_pos) if scene_pos is not None else None
            self.viewport().setCursor(_CURSORS.get(grab, Qt.CursorShape.CrossCursor))
        elif self.tool == IsolationTool.POLYGON_TRACE:
            self.viewport().setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    # -- Qt event translation ----------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.setFocus()
        if self._wants_pan(event):
            self._pan_from = event.position()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        pos = self.mapToScene(event.position().toPoint())
        if self.tool == IsolationTool.CROP:
            self.begin_crop_drag(pos)
        elif self.tool == IsolationTool.POLYGON_TRACE:
            self.click_point(pos)
        elif self.tool == IsolationTool.ERASE:
            self.begin_erase_stroke(pos)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._pan_from is not None:
            delta = event.position() - self._pan_from
            self._pan_from = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - round(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - round(delta.y()))
            event.accept()
            return
        pos = self.mapToScene(event.position().toPoint())
        if self.tool == IsolationTool.CROP:
            self.update_crop_drag(pos)
        elif self.tool == IsolationTool.ERASE:
            self.update_erase_stroke(pos)
        if not event.buttons():
            self._update_cursor(pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._pan_from is not None:
            self._pan_from = None
            self._update_cursor(None)
            event.accept()
            return
        pos = self.mapToScene(event.position().toPoint())
        if self.tool == IsolationTool.CROP:
            self.end_crop_drag(pos)
        elif self.tool == IsolationTool.ERASE:
            self.end_erase_stroke()
        self._update_cursor(pos)
        super().mouseReleaseEvent(event)
