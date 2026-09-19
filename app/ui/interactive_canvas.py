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

Lifting a subject adds three more: a press held still on the artwork (with a ring that
fills while it is held) asks for the artwork to be lifted; a correction brush paints
'keep' or 'drop' over what the lift got wrong; and a lifted cutout can be laid over the
page as a movable, rotatable, translucent overlay, whose outline then trims the page.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, Signal
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
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from app.engine.isolation.cutout import Cutout
from app.engine.isolation.placement import Placement
from app.models.isolation import IsolationTool, NormalizedPolygon, NormalizedRect
from app.ui.gestures import enable_zoom_gestures, zoom_from_viewport_event, zoom_from_wheel

_CROP_COLOUR = QColor(47, 111, 237)
_CROP_PEN = QPen(_CROP_COLOUR, 2)
_CROP_PEN.setCosmetic(True)  # constant on-screen width at any zoom
_DIM_BRUSH = QBrush(QColor(15, 23, 42, 120))  # the part of the page outside the crop
_OUTLINE_PEN = QPen(_CROP_COLOUR, 2)
_OUTLINE_PEN.setCosmetic(True)
_OUTLINE_HALO = QPen(QColor(255, 255, 255, 235), 5)
_OUTLINE_HALO.setCosmetic(True)
_KEEP_COLOUR = QColor(34, 160, 90, 120)  # painted over what the lift missed
_DROP_COLOUR = QColor(220, 50, 50, 120)  # painted over what it wrongly took
_HOLD_RING_MIN_PX = 8  # the ring that fills while a press is held
_HOLD_RING_MAX_PX = 30
_RING_TICK_MS = 30
_OVERLAY_Z = 4
_OUTLINE_Z = 4.5
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


class _CutoutItem(QGraphicsPixmapItem):
    """A cutout on the page: dragged by its artwork (not its transparent corners), reporting every move."""

    def __init__(self, pixmap: QPixmap, on_change) -> None:
        super().__init__(pixmap)
        self._on_change = on_change
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setShapeMode(QGraphicsPixmapItem.ShapeMode.MaskShape)
        self.setTransformationMode(Qt.TransformationMode.SmoothTransformation)

    def itemChange(self, change, value):
        result = super().itemChange(change, value)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and self._on_change is not None:
            self._on_change()
        return result


class InteractiveCanvas(QGraphicsView):
    cropRectChanged = Signal(object)  # NormalizedRect, or None when the selection was cleared
    cropRectAdjusting = Signal(object)  # NormalizedRect, continuously while it is being dragged
    subjectRequested = Signal(float, float)  # normalized (x, y): a press was held on the artwork -- lift it
    hintPainted = Signal(object, float, bool)  # points (image px), brush radius (px), keep?
    overlayChanged = Signal()  # the placed cutout was moved, rotated, scaled or replaced
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

        # The lifted / placed outline, drawn over a dimmed page.
        self._polygon_item: QGraphicsPolygonItem | None = None
        self._polygon_halo: QGraphicsPolygonItem | None = None
        self._outline_dim: QGraphicsPathItem | None = None
        self._polygon_norm: NormalizedPolygon | None = None

        # Press and hold to lift.
        self._hold_ms = 350
        self._hold_slop_px = 6
        self._hold_scene: QPointF | None = None
        self._hold_view: QPointF | None = None
        self._hold_elapsed_ms = 0
        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        self._hold_timer.timeout.connect(self._finish_hold)
        self._ring_timer = QTimer(self)
        self._ring_timer.setInterval(_RING_TICK_MS)
        self._ring_timer.timeout.connect(self._tick_ring)
        self._hold_ring: QGraphicsEllipseItem | None = None

        # Correction brush.
        self._brush_mode: str | None = None  # "keep" | "drop" | None
        self._brush_radius_px = 30.0
        self._hint_points: list[tuple[float, float]] = []
        self._hint_path: QPainterPath | None = None
        self._hint_item: QGraphicsPathItem | None = None
        self._hint_items: list[QGraphicsPathItem] = []

        # A cutout laid over the page.
        self._overlay: _CutoutItem | None = None
        self._overlay_cutout: Cutout | None = None

        self._erase_items: list[QGraphicsPathItem] = []
        self._erase_strokes: list[NormalizedPolygon] = []
        self._current_stroke_path: QPainterPath | None = None
        self._current_stroke_item: QGraphicsPathItem | None = None
        self._current_stroke_points: list[tuple[float, float]] = []

    # -- image ---------------------------------------------------------------
    def set_image(self, image: np.ndarray) -> None:
        self.cancel_hold()
        self._scene.clear()
        self._crop_item = None
        self._dim_item = None
        self._handle_items = []
        self._polygon_item = None
        self._polygon_halo = None
        self._outline_dim = None
        self._hold_ring = None
        self._hint_item = None
        self._hint_items = []
        self._overlay = None
        self._overlay_cutout = None
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
        self.cancel_hold()
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

    # -- the outline (a lifted subject, or a placed one) --------------------------------
    def show_polygon(self, polygon: NormalizedPolygon) -> None:
        """Outline the region and dim the rest of the page."""
        self.clear_polygon()
        width, height = self._image_size
        qpoly = QPolygonF([QPointF(x * width, y * height) for x, y in polygon.points])

        outside = QPainterPath()
        outside.addRect(self._scene.sceneRect())
        inside = QPainterPath()
        inside.addPolygon(qpoly)
        inside.closeSubpath()
        outside.addPath(inside)
        outside.setFillRule(Qt.FillRule.OddEvenFill)
        self._outline_dim = QGraphicsPathItem(outside)
        self._outline_dim.setBrush(_DIM_BRUSH)
        self._outline_dim.setPen(QPen(Qt.PenStyle.NoPen))
        self._outline_dim.setZValue(_OUTLINE_Z)
        self._outline_dim.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._scene.addItem(self._outline_dim)

        self._polygon_halo = self._scene.addPolygon(qpoly, _OUTLINE_HALO)
        self._polygon_item = self._scene.addPolygon(qpoly, _OUTLINE_PEN)
        for item in (self._polygon_halo, self._polygon_item):
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._polygon_halo.setZValue(_OUTLINE_Z + 0.1)
        self._polygon_item.setZValue(_OUTLINE_Z + 0.2)
        self._polygon_norm = polygon

    def polygon(self) -> NormalizedPolygon | None:
        return self._polygon_norm

    def clear_polygon(self) -> None:
        for item in (self._polygon_item, self._polygon_halo, self._outline_dim):
            if item is not None and item.scene() is not None:
                self._scene.removeItem(item)
        self._polygon_item = self._polygon_halo = self._outline_dim = None
        self._polygon_norm = None

    # -- press and hold to lift ------------------------------------------------------
    def set_hold_timing(self, hold_ms: int, slop_px: int) -> None:
        self._hold_ms, self._hold_slop_px = hold_ms, slop_px

    @property
    def holding(self) -> bool:
        return self._hold_scene is not None

    def begin_hold(self, scene_pos: QPointF, view_pos: QPointF | None = None) -> None:
        """A press has started on the artwork; if it is held still long enough, the artwork is lifted."""
        self.cancel_hold()
        self._hold_scene = self._clamp_to_image(scene_pos)
        self._hold_view = view_pos if view_pos is not None else QPointF(self.mapFromScene(scene_pos))
        self._hold_elapsed_ms = 0
        self._show_ring(self._hold_scene)
        self._hold_timer.start(self._hold_ms)
        self._ring_timer.start()

    def move_hold(self, view_pos: QPointF) -> None:
        """Drifting further than the slop while pressed is a drag, not a hold."""
        if self._hold_view is None:
            return
        drift = view_pos - self._hold_view
        if abs(drift.x()) + abs(drift.y()) > self._hold_slop_px:
            self.cancel_hold()

    def cancel_hold(self) -> None:
        self._hold_timer.stop()
        self._ring_timer.stop()
        self._hold_scene = self._hold_view = None
        if self._hold_ring is not None and self._hold_ring.scene() is not None:
            self._scene.removeItem(self._hold_ring)
        self._hold_ring = None

    def _finish_hold(self) -> None:
        held = self._hold_scene
        self.cancel_hold()
        if held is not None:
            x, y = self._to_normalized(held)
            self.subjectRequested.emit(x, y)

    def _show_ring(self, scene_pos: QPointF) -> None:
        ring = QGraphicsEllipseItem(-_HOLD_RING_MIN_PX, -_HOLD_RING_MIN_PX, 2 * _HOLD_RING_MIN_PX, 2 * _HOLD_RING_MIN_PX)
        ring.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)  # a constant size on screen
        pen = QPen(_CROP_COLOUR, 3)
        pen.setCosmetic(True)
        ring.setPen(pen)
        ring.setBrush(QBrush(QColor(47, 111, 237, 50)))
        ring.setZValue(20)
        ring.setPos(scene_pos)
        ring.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._scene.addItem(ring)
        self._hold_ring = ring

    def _tick_ring(self) -> None:
        """The ring grows as the hold goes on, so it is clear something is about to happen."""
        if self._hold_ring is None:
            return
        self._hold_elapsed_ms += _RING_TICK_MS
        progress = min(1.0, self._hold_elapsed_ms / max(1, self._hold_ms))
        radius = _HOLD_RING_MIN_PX + progress * (_HOLD_RING_MAX_PX - _HOLD_RING_MIN_PX)
        self._hold_ring.setRect(-radius, -radius, 2 * radius, 2 * radius)

    # -- correction brush --------------------------------------------------------------
    def set_brush(self, mode: str | None, radius_px: float | None = None) -> None:
        """`mode` is "keep" or "drop" to paint corrections with the lift tool, or None to press-and-hold instead."""
        if mode not in (None, "keep", "drop"):
            raise ValueError(mode)
        self._brush_mode = mode
        if radius_px is not None:
            self._brush_radius_px = max(1.0, float(radius_px))
        self._update_cursor(None)

    @property
    def brush_mode(self) -> str | None:
        return self._brush_mode

    @property
    def brush_radius_px(self) -> float:
        return self._brush_radius_px

    def begin_hint(self, scene_pos: QPointF) -> None:
        if self._brush_mode is None:
            return
        point = self._clamp_to_image(scene_pos)
        self._hint_points = [(point.x(), point.y())]
        self._hint_path = QPainterPath(point)
        self._hint_path.lineTo(point)
        self._hint_item = self._scene.addPath(self._hint_path, self._hint_pen())
        self._hint_item.setZValue(_OUTLINE_Z + 0.5)
        self._hint_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def update_hint(self, scene_pos: QPointF) -> None:
        if self._hint_path is None:
            return
        point = self._clamp_to_image(scene_pos)
        self._hint_points.append((point.x(), point.y()))
        self._hint_path.lineTo(point)
        self._hint_item.setPath(self._hint_path)

    def end_hint(self) -> None:
        if self._hint_path is None:
            return
        points, keep = self._hint_points, self._brush_mode == "keep"
        self._hint_items.append(self._hint_item)
        self._hint_path = self._hint_item = None
        self._hint_points = []
        self.hintPainted.emit(points, self._brush_radius_px, keep)

    def _hint_pen(self) -> QPen:
        pen = QPen(_KEEP_COLOUR if self._brush_mode == "keep" else _DROP_COLOUR, 2 * self._brush_radius_px)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        return pen

    def clear_hint_strokes(self) -> None:
        for item in self._hint_items:
            if item.scene() is not None:
                self._scene.removeItem(item)
        self._hint_items = []

    # -- a cutout laid over the page -----------------------------------------------------
    @property
    def has_overlay(self) -> bool:
        return self._overlay is not None

    def default_placement(self, cutout: Cutout) -> Placement:
        """Centred on the page, shrunk to fit if the cutout is bigger than the page."""
        width, height = self._image_size
        cut_w, cut_h = cutout.size
        scale = min(1.0, 0.9 * width / max(cut_w, 1), 0.9 * height / max(cut_h, 1))
        return Placement((width / 2, height / 2), 0.0, scale)

    def set_overlay(self, cutout: Cutout, placement: Placement | None = None) -> None:
        """Lay `cutout` over the page. It can be dragged; `placement` says where it starts."""
        self.clear_overlay()
        image = QImage(cutout.rgba.data, cutout.size[0], cutout.size[1], cutout.size[0] * 4, QImage.Format.Format_RGBA8888)
        item = _CutoutItem(QPixmap.fromImage(image.copy()), self._on_overlay_moved)
        item.setZValue(_OVERLAY_Z)
        item.setOpacity(0.6)
        # The outline is drawn on the cutout, so it moves with it and stays solid however faint the
        # cutout is made: over an identical file the cutout itself cannot be told from what is under it.
        outline = QPolygonF([QPointF(x, y) for x, y in cutout.polygon])
        for pen in (_OUTLINE_HALO, _OUTLINE_PEN):
            line = QGraphicsPolygonItem(outline, item)
            line.setPen(pen)
            line.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresParentOpacity, True)
            line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._scene.addItem(item)
        self._overlay, self._overlay_cutout = item, cutout
        self.set_overlay_placement(placement or self.default_placement(cutout))

    def clear_overlay(self) -> None:
        if self._overlay is not None and self._overlay.scene() is not None:
            self._scene.removeItem(self._overlay)
        self._overlay = None
        self._overlay_cutout = None
        self.overlayChanged.emit()

    def overlay_placement(self) -> Placement | None:
        if self._overlay is None:
            return None
        origin = self._overlay.transformOriginPoint()
        return Placement(
            (self._overlay.pos().x() + origin.x(), self._overlay.pos().y() + origin.y()),
            self._overlay.rotation(),
            self._overlay.scale(),
        )

    def set_overlay_placement(self, placement: Placement) -> None:
        if self._overlay is None:
            return
        width, height = self._overlay_cutout.size
        self._overlay.setTransformOriginPoint(width / 2, height / 2)
        self._overlay.setRotation(placement.rotation_deg)
        self._overlay.setScale(placement.scale)
        self._overlay.setPos(placement.center[0] - width / 2, placement.center[1] - height / 2)
        self.overlayChanged.emit()

    def set_overlay_opacity(self, opacity: float) -> None:
        if self._overlay is not None:
            self._overlay.setOpacity(min(1.0, max(0.05, opacity)))

    def nudge_overlay(self, dx: float, dy: float) -> None:
        placement = self.overlay_placement()
        if placement is not None:
            self.set_overlay_placement(Placement((placement.center[0] + dx, placement.center[1] + dy), placement.rotation_deg, placement.scale))

    def overlay_polygon_px(self) -> list[tuple[float, float]] | None:
        """The placed cutout's outline in this page's pixels -- what the page would be trimmed to."""
        if self._overlay is None or self._overlay_cutout is None:
            return None
        return [
            (mapped.x(), mapped.y())
            for mapped in (self._overlay.mapToScene(QPointF(x, y)) for x, y in self._overlay_cutout.polygon)
        ]

    def _on_overlay_moved(self) -> None:
        self.overlayChanged.emit()

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
        if self.tool == IsolationTool.PLACE_OUTLINE and self._overlay is not None:
            step = 10.0 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
            moves = {Qt.Key.Key_Left: (-step, 0), Qt.Key.Key_Right: (step, 0), Qt.Key.Key_Up: (0, -step), Qt.Key.Key_Down: (0, step)}
            if event.key() in moves:
                self.nudge_overlay(*moves[event.key()])
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
        elif self.tool == IsolationTool.LIFT_SUBJECT:
            self.viewport().setCursor(Qt.CursorShape.CrossCursor if self._brush_mode else Qt.CursorShape.PointingHandCursor)
        elif self.tool == IsolationTool.PLACE_OUTLINE:
            self.viewport().setCursor(Qt.CursorShape.SizeAllCursor if self._overlay is not None else Qt.CursorShape.ArrowCursor)
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
        elif self.tool == IsolationTool.LIFT_SUBJECT:
            if self._brush_mode is not None:
                self.begin_hint(pos)
            else:
                self.begin_hold(pos, event.position())
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
        elif self.tool == IsolationTool.LIFT_SUBJECT:
            if self._hint_path is not None:
                self.update_hint(pos)
            elif self.holding:
                self.move_hold(event.position())
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
        elif self.tool == IsolationTool.LIFT_SUBJECT:
            if self._hint_path is not None:
                self.end_hint()
            else:
                self.cancel_hold()  # let go before the hold completed: not a lift
        elif self.tool == IsolationTool.ERASE:
            self.end_erase_stroke()
        self._update_cursor(pos)
        super().mouseReleaseEvent(event)
