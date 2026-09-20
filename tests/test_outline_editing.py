"""Reshaping a lifted outline by hand: drag its corners, push its edges in or out, add and remove points.

A lift can miss part of the artwork or take something that is not part of it; painting Keep / Drop lifts again, which
can change more than was meant. Dragging the outline itself is direct: what is under the reviewer's hand moves.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import subject_scenes as scenes
from PIL import Image
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent

from app.engine.file_loader import load_artwork
from app.models.isolation import IsolationTool, NormalizedPolygon
from app.ui.interactive_canvas import InteractiveCanvas
from app.ui.isolation_dialog import IsolationDialog

WIDTH, HEIGHT = 800, 600


def _rectangle(x0=200, y0=150, x1=600, y1=450) -> NormalizedPolygon:
    return NormalizedPolygon([(x0 / WIDTH, y0 / HEIGHT), (x1 / WIDTH, y0 / HEIGHT), (x1 / WIDTH, y1 / HEIGHT), (x0 / WIDTH, y1 / HEIGHT)])


@pytest.fixture
def canvas(qapp):
    canvas = InteractiveCanvas()
    canvas.resize(1000, 800)
    canvas.show()
    canvas.set_image(np.full((HEIGHT, WIDTH, 3), 240, np.uint8))
    canvas.set_tool(IsolationTool.LIFT_SUBJECT)
    canvas.set_zoom(1.0)
    qapp.processEvents()
    canvas.show_polygon(_rectangle())
    canvas.set_outline_editing(True)
    return canvas


def _area(canvas) -> float:
    points = np.array(canvas.edit_points(), np.float32).reshape(-1, 1, 2)
    return float(cv2.contourArea(points))


def _points(canvas):
    return canvas.edit_points()


# ---- what is there to grab ---------------------------------------------------------------------------------------------


def test_the_outlines_corners_are_shown_as_handles_when_editing_starts(canvas):
    assert len(_points(canvas)) == 4 and len(canvas._edit_handles) == 4
    canvas.set_outline_editing(False)
    assert canvas._edit_handles == []


def test_a_press_near_a_corner_grabs_the_corner_and_near_an_edge_the_edge(canvas):
    assert canvas.hit_test_outline(QPointF(203, 152)) == ("vertex", 0)
    assert canvas.hit_test_outline(QPointF(400, 148)) == ("edge", 0)  # the top edge
    assert canvas.hit_test_outline(QPointF(602, 300)) == ("edge", 1)  # the right edge
    assert canvas.hit_test_outline(QPointF(400, 300)) is None  # inside, away from the outline


def test_how_close_counts_follows_the_zoom(canvas):
    canvas.set_zoom(0.25)  # zoomed out, a screen pixel is four image pixels: it is easier to hit
    assert canvas.hit_test_outline(QPointF(400, 138)) == ("edge", 0)
    canvas.set_zoom(4.0)  # zoomed in, it is harder
    assert canvas.hit_test_outline(QPointF(400, 145)) is None


def test_nothing_can_be_grabbed_while_editing_is_off(canvas):
    canvas.set_outline_editing(False)
    assert canvas.begin_outline_drag(QPointF(200, 150)) is False


# ---- dragging ---------------------------------------------------------------------------------------------------------------


def test_dragging_a_corner_moves_that_corner_and_reports_the_new_outline_once(canvas):
    edited = []
    canvas.outlineEdited.connect(edited.append)

    assert canvas.begin_outline_drag(QPointF(200, 150))
    canvas.update_outline_drag(QPointF(180, 120))
    canvas.update_outline_drag(QPointF(150, 100))
    assert edited == []  # nothing is reported until it is let go
    canvas.end_outline_drag(QPointF(150, 100))

    assert _points(canvas)[0] == pytest.approx((150, 100)) and _points(canvas)[1] == pytest.approx((600, 150))
    assert len(edited) == 1 and edited[0].points[0] == pytest.approx((150 / WIDTH, 100 / HEIGHT))
    assert canvas.polygon() == edited[0]


def test_dragging_an_edge_outward_takes_in_more_and_inward_drops_some(canvas):
    before = _area(canvas)
    canvas.begin_outline_drag(QPointF(400, 150))  # the top edge
    canvas.end_outline_drag(QPointF(400, 90))  # 60 px up
    assert _area(canvas) == pytest.approx(before + 400 * 60)
    assert _points(canvas)[0] == pytest.approx((200, 90)) and _points(canvas)[1] == pytest.approx((600, 90))
    assert _points(canvas)[2] == pytest.approx((600, 450))  # the other corners stay

    canvas.begin_outline_drag(QPointF(602, 300))  # the right edge
    canvas.end_outline_drag(QPointF(502, 300))  # 100 px in
    assert _area(canvas) == pytest.approx((500 - 200) * (450 - 90))


def test_a_corner_cannot_be_dragged_off_the_page(canvas):
    canvas.begin_outline_drag(QPointF(200, 150))
    canvas.end_outline_drag(QPointF(-500, -500))
    assert _points(canvas)[0] == pytest.approx((0, 0))


def test_the_dimming_and_the_outline_follow_the_drag(canvas):
    canvas.begin_outline_drag(QPointF(600, 450))
    canvas.update_outline_drag(QPointF(700, 550))
    assert canvas._polygon_item.polygon().at(2).x() == pytest.approx(700)
    assert canvas._outline_dim.path().contains(QPointF(650, 500)) is False  # inside the new outline: not dimmed
    assert canvas._outline_dim.path().contains(QPointF(20, 20)) is True
    canvas.end_outline_drag(QPointF(700, 550))


# ---- adding and removing points ---------------------------------------------------------------------------------------------------


def test_a_double_click_on_an_edge_adds_a_point_that_can_then_be_dragged(canvas):
    assert canvas.insert_outline_point(QPointF(400, 151))
    assert len(_points(canvas)) == 5 and _points(canvas)[1] == pytest.approx((400, 150))

    canvas.begin_outline_drag(QPointF(400, 150))
    canvas.end_outline_drag(QPointF(400, 60))  # a notch pulled out of the top edge
    assert _points(canvas)[1] == pytest.approx((400, 60)) and _area(canvas) > 400 * 300


def test_a_double_click_away_from_the_outline_adds_nothing(canvas):
    assert canvas.insert_outline_point(QPointF(400, 300)) is False
    assert len(_points(canvas)) == 4


def test_a_point_can_be_removed_but_never_below_a_triangle(canvas):
    assert canvas.remove_outline_point(QPointF(600, 450))
    assert len(_points(canvas)) == 3
    assert canvas.remove_outline_point(QPointF(200, 150)) is False  # a triangle is the least there is
    assert len(_points(canvas)) == 3


def test_an_edit_can_be_undone_step_by_step(canvas):
    original = _points(canvas)
    canvas.begin_outline_drag(QPointF(200, 150))
    canvas.end_outline_drag(QPointF(100, 100))
    canvas.insert_outline_point(QPointF(350, 125))  # on the top edge, which now runs from the moved corner
    assert len(_points(canvas)) == 5

    assert canvas.undo_outline_edit() and len(_points(canvas)) == 4
    assert canvas.undo_outline_edit() and _points(canvas) == pytest.approx(original)
    assert canvas.undo_outline_edit() is False


def test_ctrl_z_undoes_an_edit(canvas):
    canvas.begin_outline_drag(QPointF(200, 150))
    canvas.end_outline_drag(QPointF(100, 100))

    canvas.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier))

    assert _points(canvas)[0] == pytest.approx((200, 150))


def test_the_real_mouse_events_drag_a_corner_and_right_click_removes_one(canvas, qapp):
    def mouse(kind, scene_point, button=Qt.MouseButton.LeftButton, buttons=None):
        view = QPointF(canvas.mapFromScene(scene_point))
        return QMouseEvent(kind, view, view, button, buttons if buttons is not None else button, Qt.KeyboardModifier.NoModifier)

    canvas.mousePressEvent(mouse(QEvent.Type.MouseButtonPress, QPointF(600, 150)))
    canvas.mouseMoveEvent(mouse(QEvent.Type.MouseMove, QPointF(650, 120), Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton))
    canvas.mouseReleaseEvent(mouse(QEvent.Type.MouseButtonRelease, QPointF(650, 120), Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton))
    assert _points(canvas)[1] == pytest.approx((650, 120), abs=1.5)

    canvas.mousePressEvent(mouse(QEvent.Type.MouseButtonPress, QPointF(200, 450), Qt.MouseButton.RightButton))
    assert len(_points(canvas)) == 3


def test_editing_does_not_lift_and_does_not_start_a_hold(canvas):
    asked = []
    canvas.subjectRequested.connect(lambda x, y: asked.append((x, y)))
    canvas.set_hold_timing(30, 6)
    view = QPointF(canvas.mapFromScene(QPointF(400, 300)))
    canvas.mousePressEvent(QMouseEvent(QEvent.Type.MouseButtonPress, view, view, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    assert not canvas.holding and asked == []


# ---- an outline with many corners --------------------------------------------------------------------------------------------------


def test_an_outline_with_too_many_corners_to_grab_is_simplified_and_keeps_its_shape(qapp):
    canvas = InteractiveCanvas()
    canvas.resize(1000, 800)
    canvas.show()
    canvas.set_image(np.full((HEIGHT, WIDTH, 3), 240, np.uint8))
    canvas.set_tool(IsolationTool.LIFT_SUBJECT)
    angles = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    circle = NormalizedPolygon([((400 + 200 * np.cos(a)) / WIDTH, (300 + 200 * np.sin(a)) / HEIGHT) for a in angles])
    canvas.show_polygon(circle)

    canvas.set_outline_editing(True)

    assert 8 <= len(_points(canvas)) <= 100
    assert _area(canvas) == pytest.approx(np.pi * 200 * 200, rel=0.04)


def test_the_handles_appear_for_an_outline_lifted_while_editing_is_on(canvas):
    canvas.show_polygon(_rectangle(100, 100, 300, 300))
    assert len(_points(canvas)) == 4 and _points(canvas)[0] == pytest.approx((100, 100))


def test_clearing_the_outline_clears_the_handles(canvas):
    canvas.clear_polygon()
    assert canvas._edit_handles == [] and _points(canvas) == []


# ---- in the dialog ----------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def lifted_dialog(qapp, tmp_path):
    sheet, _truth, _not_artwork = scenes.dieline_sheet()
    path = tmp_path / "sheet.png"
    Image.fromarray(sheet).save(path)
    artwork = load_artwork(path)
    dialog = IsolationDialog(artwork, "Original")
    dialog.show()
    qapp.processEvents()
    dialog.lift_button.click()
    dialog._on_subject_requested(600 / 1300, 800 / 1560)
    yield dialog
    dialog.close()
    artwork.close()


def test_edit_outline_is_exclusive_with_keep_and_drop(lifted_dialog):
    dialog = lifted_dialog
    dialog.keep_button.setChecked(True)
    dialog.edit_button.setChecked(True)
    assert not dialog.keep_button.isChecked() and dialog.canvas.outline_editing and dialog.canvas.brush_mode is None

    dialog.drop_button.setChecked(True)
    assert not dialog.edit_button.isChecked() and not dialog.canvas.outline_editing and dialog.canvas.brush_mode == "drop"


def test_choosing_another_tool_stops_editing(lifted_dialog):
    dialog = lifted_dialog
    dialog.edit_button.setChecked(True)
    dialog.place_button.click()
    assert not dialog.canvas.outline_editing and not dialog.edit_button.isChecked()


def test_a_hand_edit_is_the_area_that_is_saved_and_the_cutout_that_is_offered(lifted_dialog, qapp):
    dialog = lifted_dialog
    dialog.edit_button.setChecked(True)
    before = dialog._subject.mask.sum()
    handles = dialog.canvas.edit_points()
    x, y = handles[0]

    assert dialog.canvas.begin_outline_drag(QPointF(x, y))
    dialog.canvas.end_outline_drag(QPointF(x + 200, y + 200))  # a corner pulled well in

    assert dialog._subject.mask.sum() < before
    assert dialog._outline[0] == dialog.canvas.polygon()
    assert "reshaped by hand" in dialog.status_label.text()

    dialog._on_save()
    geometry = dialog.result_geometry()
    assert geometry.boundary == dialog.canvas.polygon()
    assert dialog.result_cutout() is not None
    edited_points = np.array([(px * dialog._rendered.shape[1], py * dialog._rendered.shape[0]) for px, py in geometry.boundary.points])
    assert cv2.contourArea(edited_points.astype(np.float32).reshape(-1, 1, 2)) < before


def test_the_edit_button_explains_itself(lifted_dialog):
    tip = lifted_dialog.edit_button.toolTip()
    assert "Drag a corner" in tip and "edge" in tip and "Ctrl+Z" in tip
