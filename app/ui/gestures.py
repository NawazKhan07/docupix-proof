"""Zoom input shared by every image view: touchpad pinch, touch pinch, Ctrl + wheel.

Each platform reports a pinch differently -- Windows precision touchpads as
Ctrl + wheel with small deltas, macOS as a native zoom gesture, touch screens as
a Qt pinch gesture. Left alone, a view only scrolls for some of them. All three
are turned here into one thing, "scale by this factor about this point", so zoom
follows the fingers continuously instead of jumping by a fixed step.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QNativeGestureEvent, QWheelEvent
from PySide6.QtWidgets import QGestureEvent, QGraphicsView, QPinchGesture

# A standard wheel notch is 120 units and should be ~1.2x; a touchpad pinch
# arrives as many small deltas and so zooms proportionally less per event.
WHEEL_ZOOM_BASE = 1.0015


def wheel_zoom_factor(angle_delta_y: int) -> float:
    return WHEEL_ZOOM_BASE**angle_delta_y


def enable_zoom_gestures(view: QGraphicsView) -> None:
    """Ask Qt to deliver pinch gestures (touch screens) to this view."""
    view.viewport().setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
    view.viewport().grabGesture(Qt.GestureType.PinchGesture)


def zoom_from_wheel(event: QWheelEvent) -> tuple[float, QPointF] | None:
    """(factor, viewport position) for a Ctrl + wheel event, else None."""
    if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
        return None
    delta = event.angleDelta().y()
    if delta == 0:
        return None
    return wheel_zoom_factor(delta), event.position()


def zoom_from_viewport_event(view: QGraphicsView, event: QEvent) -> tuple[float, QPointF] | None:
    """(factor, viewport position) if `event` is a pinch/zoom gesture, else None."""
    if isinstance(event, QNativeGestureEvent):
        if event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
            return 1.0 + event.value(), event.position()
        return None

    if isinstance(event, QGestureEvent):
        pinch = event.gesture(Qt.GestureType.PinchGesture)
        if isinstance(pinch, QPinchGesture) and (
            pinch.changeFlags() & QPinchGesture.ChangeFlag.ScaleFactorChanged
        ):
            event.accept(pinch)
            # scaleFactor() is relative to the previous event, which is what we want.
            centre = view.viewport().mapFromGlobal(pinch.hotSpot().toPoint())
            return pinch.scaleFactor(), QPointF(centre)
    return None
