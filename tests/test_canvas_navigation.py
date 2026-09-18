"""Seeing the whole page, zooming by pinch/wheel, and a crop that can be adjusted.

These pin the reported problems: the crop dialog opened at 1:1 so the page could
not be seen, pinching only scrolled, and a drawn crop could not be changed.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent, QNativeGestureEvent, QPointingDevice, QWheelEvent
from PySide6.QtWidgets import QMessageBox

from app.models.isolation import IsolationTool
from app.ui import gestures
from app.ui.interactive_canvas import InteractiveCanvas

PAGE_W, PAGE_H = 3000, 2000


def _canvas(qapp, width=800, height=600):
    canvas = InteractiveCanvas()
    canvas.resize(width, height)
    canvas.show()
    canvas.set_image(np.full((PAGE_H, PAGE_W, 3), 230, np.uint8))
    qapp.processEvents()
    return canvas


def _wheel(pos: QPointF, delta: int, modifiers=Qt.KeyboardModifier.ControlModifier) -> QWheelEvent:
    return QWheelEvent(
        pos, pos, QPoint(0, 0), QPoint(0, delta), Qt.MouseButton.NoButton, modifiers,
        Qt.ScrollPhase.NoScrollPhase, False,
    )


# ---- the whole page is visible when it opens ------------------------------------------


def test_a_large_page_opens_fitted_so_all_of_it_can_be_seen(qapp):
    canvas = _canvas(qapp)
    visible = canvas.mapToScene(canvas.viewport().rect()).boundingRect()

    assert canvas.zoom < 0.4  # a 3000 px page in an 800 px view
    assert visible.left() <= 0 and visible.top() <= 0
    assert visible.right() >= PAGE_W and visible.bottom() >= PAGE_H


def test_a_fitted_page_stays_fitted_as_the_window_grows(qapp):
    canvas = _canvas(qapp)
    before = canvas.zoom
    canvas.resize(1200, 900)
    qapp.processEvents()
    assert canvas.zoom > before


def test_a_zoom_the_user_chose_survives_a_resize(qapp):
    canvas = _canvas(qapp)
    canvas.set_zoom(1.0)
    canvas.resize(1200, 900)
    qapp.processEvents()
    assert canvas.zoom == 1.0


def test_fit_returns_to_the_whole_page(qapp):
    canvas = _canvas(qapp)
    fitted = canvas.zoom
    canvas.set_zoom(2.0)
    canvas.fit_to_view()
    assert canvas.zoom == pytest.approx(fitted)


# ---- zoom follows the cursor / fingers -------------------------------------------------


def test_zooming_keeps_the_point_under_the_cursor_still(qapp):
    canvas = _canvas(qapp)
    canvas.set_zoom(1.0)
    anchor = QPointF(300, 200)
    before = canvas.mapToScene(anchor.toPoint())

    canvas.set_zoom(2.0, anchor)

    after = canvas.mapToScene(anchor.toPoint())
    assert (after - before).manhattanLength() < 2.0


def test_ctrl_wheel_zooms_continuously_not_by_a_fixed_step(qapp):
    canvas = _canvas(qapp)
    canvas.set_zoom(1.0)

    canvas.wheelEvent(_wheel(QPointF(400, 300), 30))  # a touchpad pinch sends many small deltas
    small = canvas.zoom
    canvas.set_zoom(1.0)
    canvas.wheelEvent(_wheel(QPointF(400, 300), 120))  # a wheel notch is bigger
    notch = canvas.zoom

    assert 1.0 < small < notch
    assert notch == pytest.approx(gestures.wheel_zoom_factor(120), rel=0.02)


def test_a_plain_wheel_scrolls_and_does_not_zoom(qapp):
    canvas = _canvas(qapp)
    canvas.set_zoom(1.0)
    canvas.wheelEvent(_wheel(QPointF(400, 300), 120, Qt.KeyboardModifier.NoModifier))
    assert canvas.zoom == 1.0


def test_zoom_is_bounded(qapp):
    canvas = _canvas(qapp)
    canvas.set_zoom(10_000.0)
    assert canvas.zoom <= 16.0
    canvas.set_zoom(0.0000001)
    assert canvas.zoom > 0


def test_pinch_and_wheel_share_one_factor():
    assert gestures.wheel_zoom_factor(0) == 1.0
    assert gestures.wheel_zoom_factor(60) > 1.0 > gestures.wheel_zoom_factor(-60)
    assert gestures.wheel_zoom_factor(-120) == pytest.approx(1 / gestures.wheel_zoom_factor(120))


def test_a_native_touchpad_pinch_zooms_about_its_position(qapp):
    canvas = _canvas(qapp)
    canvas.set_zoom(1.0)
    try:
        device = QPointingDevice.primaryPointingDevice()
        pos = QPointF(300, 200)
        event = QNativeGestureEvent(Qt.NativeGestureType.ZoomNativeGesture, device, 2, pos, pos, pos, 0.5, QPointF(), 0)
    except TypeError:
        pytest.skip("this PySide6 build has no QNativeGestureEvent constructor")

    assert canvas.viewportEvent(event)
    assert canvas.zoom == pytest.approx(1.5)


# ---- panning without disturbing the selection -----------------------------------------


def _mouse(kind, pos: QPointF, button, buttons=None) -> QMouseEvent:
    return QMouseEvent(kind, pos, pos, button, buttons if buttons is not None else button, Qt.KeyboardModifier.NoModifier)


def test_middle_button_drag_pans_and_does_not_draw_a_crop(qapp):
    canvas = _canvas(qapp)
    canvas.set_zoom(1.0)
    start = canvas.horizontalScrollBar().value()

    canvas.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, QPointF(300, 300), Qt.MouseButton.MiddleButton))
    canvas.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, QPointF(240, 300), Qt.MouseButton.NoButton, Qt.MouseButton.MiddleButton))
    canvas.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, QPointF(240, 300), Qt.MouseButton.MiddleButton, Qt.MouseButton.NoButton))

    assert canvas.horizontalScrollBar().value() == start + 60
    assert canvas.crop_rect() is None


def test_space_plus_left_drag_pans(qapp):
    canvas = _canvas(qapp)
    canvas.set_zoom(1.0)
    start = canvas.verticalScrollBar().value()

    canvas.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier))
    canvas.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, QPointF(300, 300), Qt.MouseButton.LeftButton))
    canvas.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, QPointF(300, 250), Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton))
    canvas.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, QPointF(300, 250), Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton))
    canvas.keyReleaseEvent(QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier))

    assert canvas.verticalScrollBar().value() == start + 50
    assert canvas.crop_rect() is None


# ---- the crop can be adjusted from every side, and moved -------------------------------


def _with_crop(qapp):
    """A canvas with a 1000 x 600 crop at (500, 400) on the 3000 x 2000 page."""
    canvas = _canvas(qapp)
    canvas.set_tool(IsolationTool.CROP)
    canvas.begin_crop_drag(QPointF(500, 400))
    canvas.end_crop_drag(QPointF(1500, 1000))
    return canvas


def test_the_grab_zones_are_the_edges_the_corners_and_the_body(qapp):
    canvas = _with_crop(qapp)
    assert canvas.hit_test_crop(QPointF(1000, 700)) == "move"
    assert canvas.hit_test_crop(QPointF(500, 700)) == "l"
    assert canvas.hit_test_crop(QPointF(1500, 700)) == "r"
    assert canvas.hit_test_crop(QPointF(1000, 400)) == "t"
    assert canvas.hit_test_crop(QPointF(1000, 1000)) == "b"
    assert canvas.hit_test_crop(QPointF(500, 400)) == "lt"
    assert canvas.hit_test_crop(QPointF(1500, 400)) == "rt"
    assert canvas.hit_test_crop(QPointF(500, 1000)) == "lb"
    assert canvas.hit_test_crop(QPointF(1500, 1000)) == "rb"
    assert canvas.hit_test_crop(QPointF(2500, 1800)) is None


def test_the_grab_zone_is_the_same_size_on_screen_at_any_zoom(qapp):
    canvas = _with_crop(qapp)
    canvas.set_zoom(1.0)
    zoomed_in = canvas.hit_test_crop(QPointF(500 - 30, 700))  # 30 scene px = 30 screen px off the edge: far...
    canvas.set_zoom(0.1)
    zoomed_out = canvas.hit_test_crop(QPointF(500 - 30, 700))  # ...but only 3 screen px away when zoomed out
    assert zoomed_in is None and zoomed_out == "l"


def test_dragging_an_edge_resizes_only_that_edge(qapp):
    canvas = _with_crop(qapp)
    canvas.begin_crop_drag(QPointF(500, 700))  # the left edge
    canvas.update_crop_drag(QPointF(300, 700))
    canvas.end_crop_drag(QPointF(300, 700))

    rect = canvas.crop_rect()
    assert (rect.x0, rect.x1) == (300 / PAGE_W, 1500 / PAGE_W)
    assert (rect.y0, rect.y1) == (400 / PAGE_H, 1000 / PAGE_H)


def test_dragging_a_corner_resizes_both_edges(qapp):
    canvas = _with_crop(qapp)
    canvas.begin_crop_drag(QPointF(1500, 1000))
    canvas.end_crop_drag(QPointF(1800, 1300))

    rect = canvas.crop_rect()
    assert (rect.x0, rect.y0) == (500 / PAGE_W, 400 / PAGE_H)
    assert (rect.x1, rect.y1) == (1800 / PAGE_W, 1300 / PAGE_H)


def test_dragging_the_body_moves_the_crop_without_resizing_it(qapp):
    canvas = _with_crop(qapp)
    before = canvas.crop_size_px()
    canvas.begin_crop_drag(QPointF(1000, 700))
    canvas.end_crop_drag(QPointF(1200, 800))

    rect = canvas.crop_rect()
    assert canvas.crop_size_px() == before
    assert rect.x0 == pytest.approx(700 / PAGE_W)
    assert rect.y0 == pytest.approx(500 / PAGE_H)


def test_a_crop_cannot_be_moved_off_the_page(qapp):
    canvas = _with_crop(qapp)
    canvas.begin_crop_drag(QPointF(1000, 700))
    canvas.end_crop_drag(QPointF(9000, 9000))

    rect = canvas.crop_rect()
    assert rect.x1 == 1.0 and rect.y1 == 1.0
    assert canvas.crop_size_px() == (1000, 600)


def test_pulling_an_edge_past_the_opposite_one_flips_rather_than_breaks(qapp):
    canvas = _with_crop(qapp)
    canvas.begin_crop_drag(QPointF(500, 700))
    canvas.end_crop_drag(QPointF(2000, 700))

    rect = canvas.crop_rect()
    assert rect.x0 < rect.x1
    assert (rect.x0, rect.x1) == (1500 / PAGE_W, 2000 / PAGE_W)


def test_pressing_outside_the_crop_starts_a_new_one(qapp):
    canvas = _with_crop(qapp)
    canvas.begin_crop_drag(QPointF(2000, 1500))
    canvas.end_crop_drag(QPointF(2400, 1800))

    rect = canvas.crop_rect()
    assert (rect.x0, rect.y0) == (2000 / PAGE_W, 1500 / PAGE_H)


def test_a_click_without_a_drag_clears_rather_than_leaving_a_speck(qapp):
    canvas = _canvas(qapp)
    received = []
    canvas.cropRectChanged.connect(received.append)

    canvas.begin_crop_drag(QPointF(700, 700))
    canvas.end_crop_drag(QPointF(700, 700))

    assert canvas.crop_rect() is None
    assert received == [None]


def test_live_size_is_reported_while_dragging(qapp):
    canvas = _canvas(qapp)
    live = []
    canvas.cropRectAdjusting.connect(live.append)

    canvas.begin_crop_drag(QPointF(100, 100))
    canvas.update_crop_drag(QPointF(400, 300))
    canvas.update_crop_drag(QPointF(800, 600))

    assert len(live) == 2 and live[-1].x1 == 800 / PAGE_W


def test_escape_clears_the_selection(qapp):
    canvas = _with_crop(qapp)
    canvas.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
    assert canvas.crop_rect() is None


# ---- the dialog around it ---------------------------------------------------------------


@pytest.fixture
def dialog(qapp, synthetic_png):
    from app.engine.file_loader import load_artwork
    from app.ui.isolation_dialog import IsolationDialog

    artwork = load_artwork(synthetic_png)
    widget = IsolationDialog(artwork, "Original")
    widget.show()
    qapp.processEvents()
    yield widget
    widget.close()
    artwork.close()


def _draw(dialog):
    height, width = dialog._rendered.shape[:2]
    dialog.canvas.begin_crop_drag(QPointF(width * 0.2, height * 0.2))
    dialog.canvas.end_crop_drag(QPointF(width * 0.7, height * 0.6))


def test_the_dialog_shows_the_live_size_of_the_crop(dialog):
    assert dialog.crop_readout.text() == ""
    _draw(dialog)
    assert "Crop" in dialog.crop_readout.text() and "px" in dialog.crop_readout.text()
    assert "% of the page" in dialog.crop_readout.text()


def test_the_dialog_zoom_label_follows_the_canvas(dialog):
    dialog.canvas.set_zoom(1.0)
    assert dialog.zoom_label.text() == "100%"
    dialog.zoom_in_button.click()
    assert dialog.zoom_label.text() == "125%"
    dialog.fit_button.click()
    assert dialog.zoom_label.text() == f"{dialog.canvas.zoom_percent}%"


def test_reset_clears_the_crop_and_its_readout(dialog):
    _draw(dialog)
    dialog.reset_button.click()
    assert dialog.canvas.crop_rect() is None
    assert dialog.crop_readout.text() == ""


def test_saving_a_crop_asks_for_confirmation_and_no_keeps_the_dialog_open(dialog, monkeypatch):
    asked = []
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: asked.append(args) or QMessageBox.StandardButton.No
    )
    _draw(dialog)

    dialog._on_save_clicked()

    assert asked, "the reviewer was never asked"
    assert dialog.result_geometry() is None
    assert dialog.canvas.crop_rect() is not None  # the selection is still there to adjust


def test_confirming_the_crop_saves_it(dialog, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    _draw(dialog)

    dialog._on_save_clicked()

    geometry = dialog.result_geometry()
    assert geometry is not None and geometry.boundary is not None


def test_saving_without_a_crop_does_not_ask(dialog, monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: pytest.fail("asked to confirm a crop that does not exist")
    )
    dialog._on_save_clicked()
    assert dialog.result_geometry() is not None


def test_the_dialog_is_sized_to_the_screen(dialog):
    assert dialog.width() >= 900 and dialog.height() >= 650
