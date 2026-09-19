from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF

from app.models.isolation import IsolationTool, NormalizedPolygon
from app.ui.interactive_canvas import InteractiveCanvas


def _image(width: int = 200, height: int = 100) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


def test_set_image_records_size(qapp):
    canvas = InteractiveCanvas()
    canvas.set_image(_image(200, 100))
    assert canvas._image_size == (200, 100)


def test_crop_drag_produces_correct_normalized_rect(qapp):
    canvas = InteractiveCanvas()
    canvas.set_image(_image(200, 100))
    received = []
    canvas.cropRectChanged.connect(lambda r: received.append(r))

    canvas.begin_crop_drag(QPointF(20, 10))
    canvas.update_crop_drag(QPointF(180, 90))
    canvas.end_crop_drag(QPointF(180, 90))

    rect = canvas.crop_rect()
    assert rect is not None
    assert rect.x0 == 20 / 200
    assert rect.y0 == 10 / 100
    assert rect.x1 == 180 / 200
    assert rect.y1 == 90 / 100
    assert len(received) == 1


def test_crop_drag_clamps_to_image_bounds(qapp):
    canvas = InteractiveCanvas()
    canvas.set_image(_image(200, 100))

    canvas.begin_crop_drag(QPointF(-50, -50))
    canvas.end_crop_drag(QPointF(9999, 9999))

    rect = canvas.crop_rect()
    assert rect.x0 == 0.0
    assert rect.y0 == 0.0
    assert rect.x1 == 1.0
    assert rect.y1 == 1.0


def test_a_held_press_asks_for_the_subject_at_that_point(qapp):
    import time

    canvas = InteractiveCanvas()
    canvas.set_image(_image(200, 100))
    canvas.set_tool(IsolationTool.LIFT_SUBJECT)
    canvas.set_hold_timing(20, 6)
    received = []
    canvas.subjectRequested.connect(lambda x, y: received.append((x, y)))

    canvas.begin_hold(QPointF(50, 25))
    assert canvas.holding and not received
    deadline = time.monotonic() + 2
    while not received and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)

    assert received == [(0.25, 0.25)]
    assert not canvas.holding


