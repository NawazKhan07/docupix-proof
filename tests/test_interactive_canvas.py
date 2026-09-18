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


def test_click_point_emits_normalized_coordinates(qapp):
    canvas = InteractiveCanvas()
    canvas.set_image(_image(200, 100))
    canvas.set_tool(IsolationTool.POLYGON_TRACE)
    received = []
    canvas.pointClicked.connect(lambda x, y: received.append((x, y)))

    canvas.click_point(QPointF(50, 25))

    assert received == [(0.25, 0.25)]


def test_show_polygon_and_clear(qapp):
    canvas = InteractiveCanvas()
    canvas.set_image(_image(200, 100))
    polygon = NormalizedPolygon(points=[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)])

    canvas.show_polygon(polygon)
    assert canvas.polygon() is polygon

    canvas.clear_polygon()
    assert canvas.polygon() is None


def test_erase_stroke_lifecycle(qapp):
    canvas = InteractiveCanvas()
    canvas.set_image(_image(200, 100))
    received = []
    canvas.eraseStrokeAdded.connect(lambda s: received.append(s))

    canvas.begin_erase_stroke(QPointF(10, 10))
    canvas.update_erase_stroke(QPointF(50, 10))
    canvas.update_erase_stroke(QPointF(90, 10))
    canvas.end_erase_stroke()

    strokes = canvas.erase_strokes()
    assert len(strokes) == 1
    assert len(received) == 1
    assert strokes[0].points[0] == (10 / 200, 10 / 100)
    assert strokes[0].points[-1] == (90 / 200, 10 / 100)


def test_erase_stroke_with_no_movement_is_discarded(qapp):
    """A click without a drag shouldn't register as a (degenerate) stroke."""
    canvas = InteractiveCanvas()
    canvas.set_image(_image(200, 100))
    received = []
    canvas.eraseStrokeAdded.connect(lambda s: received.append(s))

    canvas.begin_erase_stroke(QPointF(10, 10))
    canvas.end_erase_stroke()

    assert canvas.erase_strokes() == []
    assert received == []


def test_clear_erase_strokes(qapp):
    canvas = InteractiveCanvas()
    canvas.set_image(_image(200, 100))
    canvas.begin_erase_stroke(QPointF(10, 10))
    canvas.update_erase_stroke(QPointF(50, 50))
    canvas.end_erase_stroke()
    assert len(canvas.erase_strokes()) == 1

    canvas.clear_erase_strokes()
    assert canvas.erase_strokes() == []


def test_set_image_resets_prior_selections(qapp):
    canvas = InteractiveCanvas()
    canvas.set_image(_image(200, 100))
    canvas.begin_crop_drag(QPointF(20, 10))
    canvas.end_crop_drag(QPointF(180, 90))
    assert canvas.crop_rect() is not None

    canvas.set_image(_image(200, 100))
    assert canvas.crop_rect() is None
