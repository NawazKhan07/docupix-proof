"""Lift Subject, Place Subject and Erase in the canvas, the dialog and the upload slots."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import subject_scenes as scenes
from PIL import Image
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QFileDialog, QMessageBox

from app.config import SubjectConfig
from app.engine.file_loader import load_artwork
from app.engine.isolation.cutout import Cutout
from app.engine.isolation.placement import Placement
from app.engine.isolation.subject_lifter import lift_subject
from app.models.isolation import IsolationGeometry, IsolationTool, NormalizedPolygon, NormalizedRect
from app.ui.interactive_canvas import InteractiveCanvas
from app.ui.isolation_dialog import IsolationDialog, SubjectReference
from app.ui.upload_panel import UploadPanel


def _wait(qapp, condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    return condition()


def _canvas(qapp, tool=IsolationTool.LIFT_SUBJECT, size=(600, 400)):
    canvas = InteractiveCanvas()
    canvas.resize(800, 600)
    canvas.show()
    canvas.set_image(np.full((size[1], size[0], 3), 240, np.uint8))
    canvas.set_tool(tool)
    qapp.processEvents()
    return canvas


def _mouse(kind, pos, button, buttons=None):
    return QMouseEvent(kind, pos, pos, button, buttons if buttons is not None else button, Qt.KeyboardModifier.NoModifier)


def _cutout() -> Cutout:
    rgba = np.zeros((60, 100, 4), np.uint8)
    rgba[..., :3] = (200, 60, 50)
    rgba[..., 3] = 255
    return Cutout(rgba=rgba, polygon=[(0.0, 0.0), (100.0, 0.0), (100.0, 60.0), (0.0, 60.0)])


# ---- press and hold ------------------------------------------------------------------------------------------------------------


def test_a_press_held_still_lifts_and_shows_a_ring_meanwhile(qapp):
    canvas = _canvas(qapp)
    canvas.set_hold_timing(60, 6)
    asked = []
    canvas.subjectRequested.connect(lambda x, y: asked.append((x, y)))

    canvas.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, QPointF(300, 200), Qt.MouseButton.LeftButton))
    assert canvas.holding and canvas._hold_ring is not None  # feedback while it is held
    assert _wait(qapp, lambda: asked)
    assert canvas._hold_ring is None and not canvas.holding
    assert len(asked) == 1


def test_letting_go_early_is_not_a_lift(qapp):
    canvas = _canvas(qapp)
    canvas.set_hold_timing(150, 6)
    asked = []
    canvas.subjectRequested.connect(lambda x, y: asked.append((x, y)))

    canvas.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, QPointF(300, 200), Qt.MouseButton.LeftButton))
    canvas.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, QPointF(300, 200), Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton))
    time.sleep(0.25)
    qapp.processEvents()

    assert asked == [] and not canvas.holding and canvas._hold_ring is None


def test_drifting_while_pressed_is_a_drag_not_a_hold(qapp):
    canvas = _canvas(qapp)
    canvas.set_hold_timing(80, 6)
    asked = []
    canvas.subjectRequested.connect(lambda x, y: asked.append((x, y)))

    canvas.begin_hold(QPointF(50, 50), QPointF(100, 100))
    canvas.move_hold(QPointF(103, 102))  # within the slop: still a hold
    assert canvas.holding
    canvas.move_hold(QPointF(140, 100))  # too far
    assert not canvas.holding
    time.sleep(0.15)
    qapp.processEvents()
    assert asked == []


def test_choosing_another_tool_abandons_a_hold(qapp):
    canvas = _canvas(qapp)
    canvas.begin_hold(QPointF(50, 50))
    canvas.set_tool(IsolationTool.CROP)
    assert not canvas.holding and canvas._hold_ring is None


def test_the_press_point_is_reported_relative_to_the_page(qapp):
    canvas = _canvas(qapp, size=(600, 400))
    canvas.set_hold_timing(10, 6)
    asked = []
    canvas.subjectRequested.connect(lambda x, y: asked.append((x, y)))
    canvas.begin_hold(QPointF(150, 100))
    assert _wait(qapp, lambda: asked)
    assert asked[0] == pytest.approx((0.25, 0.25))


# ---- the correction brush -------------------------------------------------------------------------------------------------------


def test_the_brush_paints_a_stroke_and_reports_it(qapp):
    canvas = _canvas(qapp)
    canvas.set_brush("keep", 12)
    painted = []
    canvas.hintPainted.connect(lambda points, radius, keep: painted.append((points, radius, keep)))

    canvas.begin_hint(QPointF(10, 10))
    canvas.update_hint(QPointF(60, 40))
    canvas.end_hint()

    (points, radius, keep), = painted
    assert points == [(10.0, 10.0), (60.0, 40.0)] and radius == 12 and keep is True
    assert len(canvas._hint_items) == 1  # left on screen so the correction can be seen


def test_a_drop_brush_reports_drop_and_strokes_can_be_cleared(qapp):
    canvas = _canvas(qapp)
    canvas.set_brush("drop", 5)
    painted = []
    canvas.hintPainted.connect(lambda points, radius, keep: painted.append(keep))
    canvas.begin_hint(QPointF(1, 1))
    canvas.end_hint()
    assert painted == [False]
    canvas.clear_hint_strokes()
    assert canvas._hint_items == []


def test_with_no_brush_nothing_is_painted_and_a_bad_mode_is_refused(qapp):
    canvas = _canvas(qapp)
    canvas.begin_hint(QPointF(1, 1))
    assert canvas._hint_path is None
    with pytest.raises(ValueError):
        canvas.set_brush("erase")


def test_with_a_brush_selected_dragging_paints_instead_of_holding(qapp):
    canvas = _canvas(qapp)
    canvas.set_brush("keep", 10)
    canvas.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, QPointF(100, 100), Qt.MouseButton.LeftButton))
    assert not canvas.holding and canvas._hint_path is not None
    canvas.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, QPointF(100, 100), Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton))
    assert canvas._hint_path is None


# ---- the outline ---------------------------------------------------------------------------------------------------------------------


def test_showing_an_outline_dims_the_rest_of_the_page_and_clearing_removes_it(qapp):
    canvas = _canvas(qapp)
    before = len(canvas._scene.items())
    canvas.show_polygon(NormalizedPolygon([(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]))
    assert len(canvas._scene.items()) == before + 3  # dimming, halo and line
    assert canvas.polygon() is not None

    canvas.show_polygon(NormalizedPolygon([(0.3, 0.3), (0.7, 0.3), (0.5, 0.7)]))  # replaced, not stacked
    assert len(canvas._scene.items()) == before + 3

    canvas.clear_polygon()
    assert len(canvas._scene.items()) == before and canvas.polygon() is None


# ---- a cutout laid over the page ----------------------------------------------------------------------------------------------------


def test_a_placed_cutout_reports_where_it_sits_and_its_outline_on_the_page(qapp):
    canvas = _canvas(qapp, IsolationTool.PLACE_OUTLINE, size=(600, 400))
    cutout = _cutout()
    placement = Placement((300.0, 200.0), rotation_deg=20.0, scale=1.5)

    canvas.set_overlay(cutout, placement)

    got = canvas.overlay_placement()
    assert got.center == pytest.approx(placement.center) and got.rotation_deg == pytest.approx(20.0) and got.scale == pytest.approx(1.5)
    shown = canvas.overlay_polygon_px()
    expected = placement.map_points(cutout.polygon, cutout.size)
    assert shown == pytest.approx(expected, abs=1e-6)


def test_the_overlay_starts_centred_and_shrinks_to_fit_a_small_page(qapp):
    canvas = _canvas(qapp, IsolationTool.PLACE_OUTLINE, size=(200, 100))
    canvas.set_overlay(_cutout())  # 100 x 60 on a 200 x 100 page
    placement = canvas.overlay_placement()
    assert placement.center == pytest.approx((100.0, 50.0))
    assert placement.scale == 1.0  # it fits, so it is shown at its own size
    canvas.set_overlay(_cutout().__class__(np.zeros((300, 400, 4), np.uint8) + 255, [(0, 0), (400, 0), (400, 300), (0, 300)]))
    assert canvas.overlay_placement().scale < 0.4  # a cutout larger than the page is shrunk to fit


def test_moving_the_overlay_is_reported(qapp):
    canvas = _canvas(qapp, IsolationTool.PLACE_OUTLINE)
    canvas.set_overlay(_cutout(), Placement((300.0, 200.0)))
    moved = []
    canvas.overlayChanged.connect(lambda: moved.append(1))

    canvas._overlay.setPos(canvas._overlay.pos() + QPointF(15, 5))  # what dragging it does

    assert moved
    assert canvas.overlay_placement().center == pytest.approx((315.0, 205.0))


def test_arrow_keys_nudge_the_overlay_and_shift_moves_it_further(qapp):
    canvas = _canvas(qapp, IsolationTool.PLACE_OUTLINE)
    canvas.set_overlay(_cutout(), Placement((300.0, 200.0)))
    canvas.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier))
    canvas.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Down, Qt.KeyboardModifier.ShiftModifier))
    assert canvas.overlay_placement().center == pytest.approx((301.0, 210.0))


def test_the_overlay_can_be_faded_and_removed(qapp):
    canvas = _canvas(qapp, IsolationTool.PLACE_OUTLINE)
    canvas.set_overlay(_cutout())
    canvas.set_overlay_opacity(0.3)
    assert canvas._overlay.opacity() == pytest.approx(0.3)
    canvas.set_overlay_opacity(0.0)
    assert canvas._overlay.opacity() >= 0.05  # never invisible: it could not be found again
    canvas.clear_overlay()
    assert not canvas.has_overlay and canvas.overlay_placement() is None and canvas.overlay_polygon_px() is None


# ---- the dialog: lifting a sheet ----------------------------------------------------------------------------------------------------


@pytest.fixture
def sheet_file(tmp_path):
    sheet, truth, not_artwork = scenes.dieline_sheet()
    path = tmp_path / "sheet.png"
    Image.fromarray(sheet).save(path)
    return path, truth, not_artwork


@pytest.fixture
def sheet_dialog(qapp, sheet_file):
    path, truth, not_artwork = sheet_file
    artwork = load_artwork(path)
    dialog = IsolationDialog(artwork, "Original")
    dialog.show()
    qapp.processEvents()
    yield dialog, truth, not_artwork
    dialog.close()
    artwork.close()


def _polygon_mask(dialog) -> np.ndarray:
    height, width = dialog._rendered.shape[:2]
    points = np.round(np.array([(x * width, y * height) for x, y in dialog.canvas.polygon().points])).astype(np.int32)
    return cv2.fillPoly(np.zeros((height, width), np.uint8), [points], 1).astype(bool)


def _dilated(truth):
    return cv2.dilate(truth.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


def test_lift_subject_replaces_trace_dieline_in_the_tool_row(sheet_dialog):
    dialog, _truth, _not_artwork = sheet_dialog
    labels = [button.text() for button, _tool in dialog._tool_buttons]
    assert labels == ["Crop", "Lift Subject", "Place Subject", "Erase"]
    assert "Trace Dieline" not in labels


def test_pressing_and_holding_inside_the_carton_lifts_only_the_carton(sheet_dialog):
    dialog, truth, not_artwork = sheet_dialog
    dialog.lift_button.click()

    dialog._on_subject_requested(600 / 1300, 800 / 1560)

    mask = _polygon_mask(dialog)
    assert dialog.canvas.polygon() is not None
    assert scenes.iou(mask, _dilated(truth)) > 0.97
    assert (mask & not_artwork).sum() < 60  # the title, colour legend, crop marks, dots and the leader line + label
    assert "closed border line" in dialog.status_label.text()
    assert "Outline" in dialog.crop_readout.text() and "% of the page" in dialog.crop_readout.text()


def test_the_tool_options_follow_the_tool(sheet_dialog):
    dialog, *_ = sheet_dialog
    assert dialog.options.currentIndex() == 0
    dialog.lift_button.click()
    assert dialog.options.currentIndex() == 1 and dialog.method_combo.currentData() == "auto"
    dialog.place_button.click()
    assert dialog.options.currentIndex() == 2
    dialog.erase_button.click()
    assert dialog.options.currentIndex() == 3
    dialog.crop_button.click()
    assert dialog.options.currentIndex() == 0


def test_auto_detect_lifts_the_largest_border_without_a_press(sheet_dialog):
    dialog, truth, _ = sheet_dialog
    dialog.lift_button.click()
    dialog.autodetect_button.click()
    assert scenes.iou(_polygon_mask(dialog), _dilated(truth)) > 0.97


def test_the_method_can_be_forced_to_colour(sheet_dialog):
    dialog, *_ = sheet_dialog
    dialog.lift_button.click()
    dialog.method_combo.setCurrentIndex(dialog.method_combo.findData("colour"))
    dialog._on_subject_requested(600 / 1300, 800 / 1560)
    assert dialog._subject is not None and dialog._subject.source == "segmented"
    assert "by colour" in dialog.status_label.text()


def test_a_drop_stroke_corrects_the_outline_and_is_kept_with_it(sheet_dialog):
    dialog, *_ = sheet_dialog
    dialog.lift_button.click()
    dialog._on_subject_requested(600 / 1300, 800 / 1560)
    before = _polygon_mask(dialog).sum()

    dialog.drop_button.click()
    assert dialog.canvas.brush_mode == "drop" and not dialog.keep_button.isChecked()
    dialog.canvas.hintPainted.emit([(1140.0, 800.0)], 120.0, False)  # over the right end of the carton

    assert _polygon_mask(dialog).sum() < before - 20_000
    assert len(dialog._hints) == 1 and dialog.canvas._hint_items == []


def test_keep_and_drop_are_exclusive_and_off_when_another_tool_is_chosen(sheet_dialog):
    dialog, *_ = sheet_dialog
    dialog.lift_button.click()
    dialog.keep_button.click()
    assert dialog.canvas.brush_mode == "keep"
    dialog.drop_button.click()
    assert dialog.canvas.brush_mode == "drop" and not dialog.keep_button.isChecked()
    dialog.drop_button.click()  # off again
    assert dialog.canvas.brush_mode is None
    dialog.keep_button.click()
    dialog.crop_button.click()
    assert dialog.canvas.brush_mode is None and not dialog.keep_button.isChecked()


def test_a_correction_before_anything_is_lifted_asks_for_a_press_first(sheet_dialog):
    dialog, *_ = sheet_dialog
    dialog.lift_button.click()
    dialog.canvas.hintPainted.emit([(10.0, 10.0)], 20.0, True)
    assert dialog.canvas.polygon() is None and "Press and hold" in dialog.status_label.text()


def test_a_new_press_starts_a_new_subject_and_forgets_old_corrections(sheet_dialog):
    dialog, *_ = sheet_dialog
    dialog.lift_button.click()
    dialog._on_subject_requested(600 / 1300, 800 / 1560)
    dialog.canvas.hintPainted.emit([(1140.0, 800.0)], 120.0, False)
    assert dialog._hints
    dialog._on_subject_requested(600 / 1300, 800 / 1560)
    assert dialog._hints == []


def test_the_lifted_artwork_is_saved_as_a_transparent_png(sheet_dialog, tmp_path, monkeypatch):
    dialog, *_ = sheet_dialog
    dialog.lift_button.click()
    assert not dialog.save_cutout_button.isEnabled()  # nothing lifted yet
    dialog._on_subject_requested(600 / 1300, 800 / 1560)
    assert dialog.save_cutout_button.isEnabled()

    target = tmp_path / "carton_cutout.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), "PNG image (*.png)"))
    dialog.save_cutout_button.click()

    saved = Cutout.load_png(target)
    assert target.is_file() and "Saved the cutout" in dialog.status_label.text()
    assert saved.rgba[..., 3].min() == 0 and saved.rgba[..., 3].max() == 255  # transparent outside, opaque inside
    assert saved.size[0] > 800 and saved.size[1] > 500  # the carton, cropped to its box


def test_cancelling_the_save_dialog_saves_nothing(sheet_dialog, tmp_path, monkeypatch):
    dialog, *_ = sheet_dialog
    dialog.lift_button.click()
    dialog._on_subject_requested(600 / 1300, 800 / 1560)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    dialog.save_cutout_button.click()
    assert list(tmp_path.glob("*_cutout.png")) == [] and "Saved the cutout" not in dialog.status_label.text()


def test_reset_in_the_lift_tool_clears_the_lift(sheet_dialog):
    dialog, *_ = sheet_dialog
    dialog.lift_button.click()
    dialog._on_subject_requested(600 / 1300, 800 / 1560)
    dialog.reset_button.click()
    assert dialog.canvas.polygon() is None and dialog._subject is None and not dialog.save_cutout_button.isEnabled()


def test_applying_a_lift_saves_a_polygon_that_masks_but_never_registers(sheet_dialog):
    dialog, *_ = sheet_dialog
    dialog.lift_button.click()
    dialog._on_subject_requested(600 / 1300, 800 / 1560)

    dialog._on_save()

    geometry = dialog.result_geometry()
    assert isinstance(geometry.boundary, NormalizedPolygon) and geometry.boundary_origin == "lifted"
    assert not geometry.anchors_alignment
    assert dialog.result_cutout() is not None  # available to lay over the other file


def test_drawing_a_crop_after_a_lift_replaces_it(sheet_dialog):
    dialog, *_ = sheet_dialog
    dialog.lift_button.click()
    dialog._on_subject_requested(600 / 1300, 800 / 1560)
    dialog.crop_button.click()
    dialog.canvas.begin_crop_drag(QPointF(200, 200))
    dialog.canvas.end_crop_drag(QPointF(600, 500))
    assert dialog.canvas.polygon() is None and dialog._outline is None and dialog.canvas.crop_rect() is not None


def test_a_lift_after_a_crop_replaces_the_crop(sheet_dialog):
    dialog, *_ = sheet_dialog
    dialog.canvas.begin_crop_drag(QPointF(200, 200))
    dialog.canvas.end_crop_drag(QPointF(600, 500))
    dialog.lift_button.click()
    dialog._on_subject_requested(600 / 1300, 800 / 1560)
    assert dialog.canvas.crop_rect() is None and dialog.canvas.polygon() is not None


# ---- the dialog: placing another file's outline ---------------------------------------------------------------------------------------


@pytest.fixture
def photo_pair(tmp_path):
    """The artwork on a grey desk (the original), and the same artwork photographed differently, no border of its own."""
    original, _ = scenes.compose(scenes.artwork(edge="coloured"), "grey desk", rotation_deg=0.0, offset=(0, 0))
    suspect, suspect_truth = scenes.compose(
        scenes.artwork(edge="coloured"), "wood", canvas=(1500, 1700), rotation_deg=6.0, scale=0.85, offset=(50, -30), seed=9
    )
    a, b = tmp_path / "original.png", tmp_path / "suspect.png"
    Image.fromarray(original).save(a)
    Image.fromarray(suspect).save(b)
    return a, b, suspect_truth


def _lift_original_cutout(path) -> Cutout:
    artwork = load_artwork(path)
    try:
        image = artwork.render()
        subject = lift_subject(image, (image.shape[1] / 2, image.shape[0] / 2), SubjectConfig())
        return Cutout.from_subject(image, subject)
    finally:
        artwork.close()


@pytest.fixture
def suspect_dialog(qapp, photo_pair):
    original_path, suspect_path, truth = photo_pair
    reference = SubjectReference(_lift_original_cutout(original_path), "original.png")
    artwork = load_artwork(suspect_path)
    dialog = IsolationDialog(artwork, "Suspect", reference=reference)
    dialog.show()
    dialog.place_button.click()
    qapp.processEvents()
    yield dialog, truth
    dialog.close()
    artwork.close()


def test_the_place_tool_starts_with_only_what_can_be_done_yet(qapp, photo_pair):
    _a, suspect_path, _truth = photo_pair
    artwork = load_artwork(suspect_path)
    try:
        dialog = IsolationDialog(artwork, "Suspect")  # no reference: nothing was lifted in the other file
        assert not dialog.use_reference_button.isEnabled()
        assert dialog.open_cutout_button.isEnabled()
        dialog._update_option_states()
        assert not dialog.trim_button.isEnabled() and not dialog.autofit_button.isEnabled()
    finally:
        artwork.close()


def test_the_other_files_outline_arrives_already_on_the_artwork(suspect_dialog):
    dialog, truth = suspect_dialog
    assert dialog.use_reference_button.isEnabled() and "original.png" in dialog.use_reference_button.text()

    dialog.use_reference_button.click()

    assert dialog.canvas.has_overlay
    height, width = dialog._rendered.shape[:2]
    placed = cv2.fillPoly(np.zeros((height, width), np.uint8), [np.round(np.array(dialog.canvas.overlay_polygon_px())).astype(np.int32)], 1).astype(bool)
    assert scenes.iou(placed, truth) > 0.96
    assert "Fitted by matching the artwork" in dialog.status_label.text()
    # the controls show where it sits
    assert dialog.scale_spin.value() == pytest.approx(85.0, abs=3.0)
    assert dialog.rotation_spin.value() == pytest.approx(-6.0, abs=0.8)  # the scene was turned 6 degrees counter-clockwise


def test_trimming_makes_the_placed_outline_the_area_and_saving_keeps_it(suspect_dialog, monkeypatch):
    dialog, truth = suspect_dialog
    dialog.use_reference_button.click()

    dialog.trim_button.click()
    assert dialog._outline is not None and dialog._outline[1] == "placed" and dialog.canvas.polygon() is not None

    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a[1]) or QMessageBox.StandardButton.Yes)
    dialog._on_save_clicked()

    geometry = dialog.result_geometry()
    assert asked and "Apply this area?" in asked
    assert geometry.boundary_origin == "placed" and not geometry.anchors_alignment
    height, width = dialog._rendered.shape[:2]
    mask = cv2.fillPoly(np.zeros((height, width), np.uint8), [np.round(np.array([(x * width, y * height) for x, y in geometry.boundary.points])).astype(np.int32)], 1).astype(bool)
    assert scenes.iou(mask, truth) > 0.96


def test_saving_with_an_untrimmed_overlay_offers_to_trim(suspect_dialog, monkeypatch):
    dialog, _truth = suspect_dialog
    dialog.use_reference_button.click()
    titles = []

    def answer(_parent, title, *_a, **_k):
        titles.append(title)
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", answer)
    dialog._on_save_clicked()

    assert titles[0] == "Trim to the placed outline?"
    assert dialog.result_geometry().boundary_origin == "placed"


def test_declining_to_trim_saves_the_file_without_the_outline(suspect_dialog, monkeypatch):
    dialog, _truth = suspect_dialog
    dialog.use_reference_button.click()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    dialog._on_save_clicked()
    assert dialog.result_geometry().boundary is None


def test_cancelling_at_the_trim_question_stays_in_the_dialog(suspect_dialog, monkeypatch):
    dialog, _truth = suspect_dialog
    dialog.use_reference_button.click()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Cancel)
    dialog._on_save_clicked()
    assert dialog.result_geometry() is None


def test_the_rotation_and_scale_controls_move_the_overlay(suspect_dialog):
    dialog, _truth = suspect_dialog
    dialog.use_reference_button.click()
    centre = dialog.canvas.overlay_placement().center

    dialog.rotation_spin.setValue(12.5)
    dialog.scale_spin.setValue(60.0)

    placement = dialog.canvas.overlay_placement()
    assert placement.rotation_deg == pytest.approx(12.5) and placement.scale == pytest.approx(0.6)
    assert placement.center == pytest.approx(centre)  # turned and scaled about its centre


def test_dragging_the_overlay_updates_the_controls_and_auto_fit_undoes_it(suspect_dialog):
    dialog, truth = suspect_dialog
    dialog.use_reference_button.click()
    good = dialog.canvas.overlay_placement()

    dialog.canvas.nudge_overlay(120, -80)
    dialog.canvas.nudge_overlay(0, 0)
    assert dialog.canvas.overlay_placement().center != pytest.approx(good.center)

    dialog.autofit_button.click()
    assert dialog.canvas.overlay_placement().center == pytest.approx(good.center, abs=2.0)


def test_when_the_artwork_cannot_be_matched_the_outline_is_left_for_placing_by_hand(qapp, tmp_path):
    reference = SubjectReference(_cutout_of_text(), "original.png")
    blank = np.full((600, 800, 3), 200, np.uint8)
    path = tmp_path / "nothing.png"
    Image.fromarray(blank).save(path)
    artwork = load_artwork(path)
    try:
        dialog = IsolationDialog(artwork, "Suspect", reference=reference)
        dialog.place_button.click()
        dialog.use_reference_button.click()
        assert dialog.canvas.has_overlay  # still there to drag
        assert "could not be matched" in dialog.status_label.text()
        dialog.trim_button.click()  # and it can still be used
        assert dialog._outline[1] == "placed"
    finally:
        artwork.close()


def _cutout_of_text() -> Cutout:
    face = scenes.artwork(edge="coloured")
    rgba = np.dstack([face, np.full(face.shape[:2], 255, np.uint8)])
    h, w = face.shape[:2]
    return Cutout(rgba=rgba, polygon=[(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))])


def test_a_saved_cutout_can_be_opened_and_placed(suspect_dialog, tmp_path, monkeypatch):
    dialog, truth = suspect_dialog
    saved = _cutout_of_text()
    path = saved.save_png(tmp_path / "saved")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), ""))

    dialog.open_cutout_button.click()

    assert dialog.canvas.has_overlay and "saved.png" in dialog.status_label.text()
    assert dialog.trim_button.isEnabled()


def test_opening_something_that_is_not_a_cutout_explains(suspect_dialog, tmp_path, monkeypatch):
    dialog, _truth = suspect_dialog
    flat = tmp_path / "flat.png"
    Image.fromarray(np.full((40, 40, 3), 128, np.uint8)).save(flat)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(flat), ""))
    dialog.reset_button.click()  # the fixture's dialog already laid the other file's artwork over the page
    assert not dialog.canvas.has_overlay
    dialog.open_cutout_button.click()
    assert not dialog.canvas.has_overlay and "no transparent background" in dialog.status_label.text()


def test_reset_in_the_place_tool_removes_the_overlay_and_a_placed_trim(suspect_dialog):
    dialog, _truth = suspect_dialog
    dialog.use_reference_button.click()
    dialog.trim_button.click()
    dialog.reset_button.click()
    assert not dialog.canvas.has_overlay and dialog._outline is None and dialog.canvas.polygon() is None


def test_opacity_slider_fades_the_overlay(suspect_dialog):
    dialog, _truth = suspect_dialog
    dialog.use_reference_button.click()
    dialog.opacity_slider.setValue(25)
    assert dialog.canvas._overlay.opacity() == pytest.approx(0.25)


# ---- the upload slots pass a lifted subject to the other file --------------------------------------------------------------------------


def test_each_file_is_offered_the_others_lifted_subject(qapp, tmp_path, monkeypatch, synthetic_png):
    import app.ui.upload_panel as upload_module

    seen = {}

    class FakeDialog:
        class DialogCode:
            Accepted = 1

        def __init__(self, artwork, title, parent=None, reference=None, config=None, partner_title=None):
            seen[title] = reference

        def exec(self):
            return 0  # cancelled

    monkeypatch.setattr(upload_module, "IsolationDialog", FakeDialog)
    panel = UploadPanel()
    panel.original_slot.load_path(synthetic_png)
    panel.suspect_slot.load_path(synthetic_png)
    panel.original_slot.cutout = _cutout()

    panel.original_slot._on_edit_artwork_area()
    panel.suspect_slot._on_edit_artwork_area()

    assert seen["Original"] is None  # the suspect has lifted nothing yet
    assert seen["Suspect"] is not None
    assert seen["Suspect"].source_name == synthetic_png.name and seen["Suspect"].cutout is panel.original_slot.cutout


def test_a_lifted_subject_is_remembered_by_its_slot_and_dropped_with_its_file(qapp, synthetic_png, monkeypatch):
    import app.ui.upload_panel as upload_module

    cutout = _cutout()

    class FakeDialog:
        class DialogCode:
            Accepted = 1

        def __init__(self, *a, **k):
            pass

        def exec(self):
            return 1

        def result_geometry(self):
            return IsolationGeometry(boundary=NormalizedPolygon([(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)]), boundary_origin="lifted")

        def result_cutout(self):
            return cutout

    monkeypatch.setattr(upload_module, "IsolationDialog", FakeDialog)
    panel = UploadPanel()
    panel.original_slot.load_path(synthetic_png)

    panel.original_slot._on_edit_artwork_area()

    assert panel.original_slot.cutout is cutout
    assert "artwork lifted" in panel.original_slot.detail_label.text()

    panel.original_slot.load_path(synthetic_png)  # a different file: the old subject no longer applies
    assert panel.original_slot.cutout is None


def test_the_status_line_says_when_a_file_is_trimmed_to_a_placed_outline(qapp, synthetic_png):
    panel = UploadPanel()
    panel.suspect_slot.load_path(synthetic_png)
    panel.suspect_slot.geometry = IsolationGeometry(
        boundary=NormalizedPolygon([(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)]), boundary_origin="placed"
    )
    panel.suspect_slot._refresh_status()
    assert "trimmed to the placed outline" in panel.suspect_slot.detail_label.text()


# ---- placing is offered, not hidden ---------------------------------------------------------------------------------------------------------


@pytest.fixture
def open_dialog(qapp, photo_pair):
    """Builds the suspect's dialog, with or without the original's lifted artwork to offer."""
    made = []

    def build(with_reference=True, **kwargs):
        original_path, suspect_path, _truth = photo_pair
        reference = SubjectReference(_lift_original_cutout(original_path), "original.png") if with_reference else None
        artwork = load_artwork(suspect_path)
        dialog = IsolationDialog(artwork, "Suspect", reference=reference, **kwargs)
        dialog.show()
        qapp.processEvents()
        made.append((dialog, artwork))
        return dialog

    yield build
    for dialog, artwork in made:
        dialog.close()
        artwork.close()


def test_the_other_files_lifted_artwork_is_offered_in_a_banner_when_the_dialog_opens(open_dialog):
    dialog = open_dialog(partner_title="Original")
    assert dialog.reference_banner.isVisible() and "original.png" in dialog.reference_label.text()
    assert dialog.canvas.tool == IsolationTool.CROP  # nothing was done on the user's behalf yet

    dialog.reference_button.click()

    assert dialog.canvas.tool == IsolationTool.PLACE_OUTLINE and dialog.canvas.has_overlay
    assert not dialog.reference_banner.isVisible()  # it has been taken up: nothing left to offer


def test_no_banner_when_the_other_file_has_lifted_nothing(open_dialog):
    dialog = open_dialog(with_reference=False, partner_title="Original")
    assert not dialog.reference_banner.isVisible()


def test_with_nothing_lifted_yet_the_place_tool_says_what_to_do_first_instead_of_only_being_greyed_out(open_dialog):
    dialog = open_dialog(with_reference=False, partner_title="Original")
    dialog.place_button.click()

    hint = dialog.place_hint_label.text()
    assert "Nothing has been lifted from Original" in hint and "Lift Subject" in hint
    assert not dialog.use_reference_button.isEnabled() and "Original" in dialog.use_reference_button.toolTip()
    assert dialog.open_cutout_button.isEnabled() and not dialog.canvas.has_overlay


def test_choosing_place_subject_lays_the_other_files_artwork_over_the_page_without_a_second_click(open_dialog):
    dialog = open_dialog()
    dialog.place_button.click()
    assert dialog.canvas.has_overlay and "Fitted by matching the artwork" in dialog.status_label.text()


def test_the_banner_stays_until_the_file_has_an_outline_of_its_own(open_dialog):
    dialog = open_dialog()
    dialog.crop_button.click()
    dialog.canvas.begin_crop_drag(QPointF(100, 100))
    dialog.canvas.end_crop_drag(QPointF(400, 400))
    assert dialog.reference_banner.isVisible()  # a crop is not an outline: the offer stands

    dialog._set_outline(NormalizedPolygon([(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)]), "lifted")
    assert not dialog.reference_banner.isVisible()


def test_a_lift_with_no_other_file_lifted_says_where_it_goes_next(sheet_dialog):
    dialog, *_ = sheet_dialog
    dialog.lift_button.click()
    dialog._on_subject_requested(600 / 1300, 800 / 1560)
    assert "place this artwork over it" in dialog.status_label.text()


def test_a_colour_lift_much_smaller_than_the_other_files_artwork_offers_to_use_that_instead(qapp, photo_pair):
    """The scan of a die-cut sheet: colour finds only the printed panels. The dialog says so, and offers the
    outline of the artwork already lifted from the digital file."""
    _a, suspect_path, _truth = photo_pair
    huge = np.zeros((3000, 3000, 4), np.uint8)
    huge[..., 3] = 255
    reference = SubjectReference(
        Cutout(rgba=huge, polygon=[(0.0, 0.0), (3000.0, 0.0), (3000.0, 3000.0), (0.0, 3000.0)]), "digital.pdf"
    )
    artwork = load_artwork(suspect_path)
    try:
        dialog = IsolationDialog(artwork, "Suspect", reference=reference, partner_title="Original")
        dialog.show()
        dialog.lift_button.click()
        dialog.method_combo.setCurrentIndex(dialog.method_combo.findData("colour"))
        dialog._on_subject_requested(0.5, 0.5)
        qapp.processEvents()

        assert dialog._subject is not None and dialog.canvas.polygon() is not None
        assert dialog.reference_banner.isVisible() and "digital.pdf" in dialog.reference_label.text()
        assert "only" in dialog.reference_label.text() and "%" in dialog.reference_label.text()

        dialog.reference_button.click()  # take the offer

        assert dialog.canvas.tool == IsolationTool.PLACE_OUTLINE
        assert dialog._subject is None and dialog.canvas.polygon() is None  # the partial lift is forgotten
        dialog.close()
    finally:
        artwork.close()


def test_a_lift_close_to_the_other_files_size_does_not_second_guess_itself(open_dialog):
    dialog = open_dialog()
    dialog.lift_button.click()
    dialog.method_combo.setCurrentIndex(dialog.method_combo.findData("colour"))
    dialog._on_subject_requested(0.5, 0.5)
    assert dialog._subject is not None and dialog._banner_hint is None


def test_a_lifted_area_keeps_the_width_of_the_line_it_followed_so_the_line_is_not_compared(sheet_dialog):
    dialog, *_ = sheet_dialog
    dialog.lift_button.click()
    dialog._on_subject_requested(600 / 1300, 800 / 1560)
    dialog._on_save()

    geometry = dialog.result_geometry()
    height, width = dialog._rendered.shape[:2]
    assert geometry.edge_inset_norm * max(height, width) >= 2.0  # at least the 2 px line the sheet was drawn with


def test_a_cropped_area_is_not_inset(sheet_dialog):
    dialog, *_ = sheet_dialog
    dialog.canvas.begin_crop_drag(QPointF(200, 200))
    dialog.canvas.end_crop_drag(QPointF(900, 900))
    dialog._on_save()
    assert dialog.result_geometry().edge_inset_norm == 0.0


def test_a_placed_outline_is_inset_by_a_margin_for_how_exact_the_fit_is(open_dialog):
    dialog = open_dialog()
    dialog.place_button.click()
    dialog.trim_button.click()
    dialog._on_save()

    geometry = dialog.result_geometry()
    assert geometry.edge_inset_norm * max(dialog._rendered.shape[:2]) >= 1.5


# ---- erase: select a rectangle like a crop -----------------------------------------------------------------------------------------------------


def test_the_erase_tool_selects_a_rectangle_like_a_crop_and_applies_it(qapp):
    canvas = _canvas(qapp, IsolationTool.ERASE)
    seen = []
    canvas.eraseSelectionChanged.connect(seen.append)

    canvas.begin_erase_drag(QPointF(100, 80))
    canvas.update_erase_drag(QPointF(300, 200))
    canvas.end_erase_drag(QPointF(300, 200))

    selection = canvas.erase_selection()
    assert (selection.x0, selection.y0, selection.x1, selection.y1) == pytest.approx((100 / 600, 80 / 400, 300 / 600, 200 / 400))
    assert canvas.erase_selection_size_px() == (200, 120) and seen[-1] is selection
    assert canvas.erase_regions() == []  # selected, not yet applied

    applied = canvas.apply_erase_selection()

    assert applied is selection and canvas.erase_selection() is None and canvas.erase_regions() == [selection]


def test_the_erase_selection_can_be_resized_and_moved_before_it_is_applied(qapp):
    canvas = _canvas(qapp, IsolationTool.ERASE)
    canvas.begin_erase_drag(QPointF(100, 80))
    canvas.end_erase_drag(QPointF(300, 200))

    assert canvas.hit_test_erase(QPointF(300, 140)) == "r" and canvas.hit_test_erase(QPointF(200, 140)) == "move"
    canvas.begin_erase_drag(QPointF(300, 140))  # the right edge
    canvas.update_erase_drag(QPointF(400, 140))
    canvas.end_erase_drag(QPointF(400, 140))
    assert canvas.erase_selection_size_px() == (300, 120)

    canvas.begin_erase_drag(QPointF(250, 140))  # the body
    canvas.update_erase_drag(QPointF(270, 190))
    canvas.end_erase_drag(QPointF(270, 190))
    selection = canvas.erase_selection()
    assert (selection.x0, selection.y0) == pytest.approx((120 / 600, 130 / 400))


def test_a_click_with_the_erase_tool_selects_nothing(qapp):
    canvas = _canvas(qapp, IsolationTool.ERASE)
    seen = []
    canvas.eraseSelectionChanged.connect(seen.append)
    canvas.begin_erase_drag(QPointF(100, 100))
    canvas.end_erase_drag(QPointF(101, 101))
    assert canvas.erase_selection() is None and seen[-1] is None


def test_erased_areas_can_be_undone_one_by_one_or_cleared(qapp):
    canvas = _canvas(qapp, IsolationTool.ERASE)
    changes = []
    canvas.eraseRegionsChanged.connect(lambda: changes.append(len(canvas.erase_regions())))
    for x in (50, 250, 450):
        canvas.begin_erase_drag(QPointF(x, 50))
        canvas.end_erase_drag(QPointF(x + 100, 150))
        canvas.apply_erase_selection()
    assert len(canvas.erase_regions()) == 3

    canvas.undo_erase()
    assert len(canvas.erase_regions()) == 2
    canvas.clear_erase_regions()
    assert canvas.erase_regions() == [] and changes[-1] == 0
    canvas.undo_erase()  # nothing left: not an error
    assert canvas.erase_regions() == []


def test_enter_applies_the_erase_selection_and_escape_drops_it(qapp):
    canvas = _canvas(qapp, IsolationTool.ERASE)
    canvas.begin_erase_drag(QPointF(100, 100))
    canvas.end_erase_drag(QPointF(200, 200))
    canvas.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier))
    assert len(canvas.erase_regions()) == 1 and canvas.erase_selection() is None

    canvas.begin_erase_drag(QPointF(300, 100))
    canvas.end_erase_drag(QPointF(400, 200))
    canvas.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
    assert len(canvas.erase_regions()) == 1 and canvas.erase_selection() is None


def test_the_erase_selection_and_the_crop_are_independent(qapp):
    canvas = _canvas(qapp, IsolationTool.CROP)
    canvas.begin_crop_drag(QPointF(50, 50))
    canvas.end_crop_drag(QPointF(500, 350))
    canvas.set_tool(IsolationTool.ERASE)
    canvas.begin_erase_drag(QPointF(100, 100))
    canvas.end_erase_drag(QPointF(200, 200))

    assert canvas.crop_rect() is not None and canvas.erase_selection() is not None
    canvas.clear_erase_selection()
    assert canvas.crop_rect() is not None


def test_a_new_image_clears_the_erased_areas(qapp):
    canvas = _canvas(qapp, IsolationTool.ERASE)
    canvas.begin_erase_drag(QPointF(100, 100))
    canvas.end_erase_drag(QPointF(200, 200))
    canvas.apply_erase_selection()
    canvas.set_image(np.full((400, 600, 3), 240, np.uint8))
    assert canvas.erase_regions() == [] and canvas.erase_selection() is None


def test_erasing_is_a_rectangle_not_a_pen(qapp):
    """The reported problem: the Erase button gave a pen to draw with; it should select an area like a crop."""
    canvas = _canvas(qapp, IsolationTool.ERASE)
    assert not hasattr(canvas, "begin_erase_stroke")
    canvas.begin_erase_drag(QPointF(100, 100))
    canvas.update_erase_drag(QPointF(150, 250))
    canvas.update_erase_drag(QPointF(300, 120))  # wandering does not paint anything: it is a rectangle
    canvas.end_erase_drag(QPointF(300, 120))
    assert canvas.erase_selection_size_px() == (200, 20)


@pytest.fixture
def erase_dialog(sheet_dialog):
    dialog, *_ = sheet_dialog
    dialog.erase_button.click()
    return dialog


def _select_erase(dialog, x0=0.3, y0=0.4, x1=0.5, y1=0.5):
    width, height = dialog.canvas._image_size
    dialog.canvas.begin_erase_drag(QPointF(width * x0, height * y0))
    dialog.canvas.end_erase_drag(QPointF(width * x1, height * y1))


def test_the_erase_options_follow_the_selection(erase_dialog):
    dialog = erase_dialog
    assert dialog.options.currentIndex() == 3 and not dialog.erase_apply_button.isEnabled()
    assert not dialog.erase_undo_button.isEnabled() and not dialog.erase_clear_button.isEnabled()

    _select_erase(dialog)
    assert dialog.erase_apply_button.isEnabled() and dialog.crop_readout.text().startswith("Erase ")

    dialog.erase_apply_button.click()
    assert not dialog.erase_apply_button.isEnabled() and dialog.erase_undo_button.isEnabled()
    assert dialog.erase_count_label.text() == "1 area erased"

    _select_erase(dialog, 0.6, 0.6, 0.8, 0.7)
    dialog.erase_apply_button.click()
    assert dialog.erase_count_label.text() == "2 areas erased"

    dialog.erase_undo_button.click()
    assert dialog.erase_count_label.text() == "1 area erased"
    dialog.erase_clear_button.click()
    assert dialog.erase_count_label.text() == "" and not dialog.erase_clear_button.isEnabled()


def test_erased_rectangles_are_saved_with_the_area(erase_dialog):
    dialog = erase_dialog
    _select_erase(dialog)
    dialog.erase_apply_button.click()
    dialog._on_save()

    (region,) = dialog.result_geometry().erase_regions
    assert isinstance(region, NormalizedRect)
    assert (region.x0, region.y0, region.x1, region.y1) == pytest.approx((0.3, 0.4, 0.5, 0.5), abs=1e-3)


def test_reset_in_the_erase_tool_brings_back_everything_erased(erase_dialog):
    dialog = erase_dialog
    _select_erase(dialog)
    dialog.erase_apply_button.click()
    _select_erase(dialog, 0.6, 0.6, 0.8, 0.7)

    dialog.reset_button.click()

    assert dialog.canvas.erase_regions() == [] and dialog.canvas.erase_selection() is None and dialog.crop_readout.text() == ""


def test_saving_with_a_selected_but_unapplied_erase_area_asks_whether_to_apply_it(erase_dialog, monkeypatch):
    dialog = erase_dialog
    _select_erase(dialog)
    titles = []

    def answer(_parent, title, *_a, **_k):
        titles.append(title)
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", answer)
    dialog._on_save_clicked()

    assert titles == ["Erase the selected area?"] and len(dialog.result_geometry().erase_regions) == 1


def test_declining_to_erase_the_selection_saves_without_it(erase_dialog, monkeypatch):
    dialog = erase_dialog
    _select_erase(dialog)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    dialog._on_save_clicked()
    assert dialog.result_geometry().erase_regions == []


def test_cancelling_at_the_erase_question_stays_in_the_dialog(erase_dialog, monkeypatch):
    dialog = erase_dialog
    _select_erase(dialog)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Cancel)
    dialog._on_save_clicked()
    assert dialog.result_geometry() is None


def test_the_upload_slot_says_how_many_areas_were_erased(qapp, synthetic_png):
    panel = UploadPanel()
    panel.original_slot.load_path(synthetic_png)
    panel.original_slot.geometry = IsolationGeometry(
        erase_regions=[NormalizedRect(0.1, 0.1, 0.2, 0.2), NormalizedRect(0.5, 0.5, 0.6, 0.6)]
    )
    panel.original_slot._refresh_status()
    assert "2 areas erased" in panel.original_slot.detail_label.text()


def test_a_slot_says_its_lifted_artwork_can_be_placed_on_the_other_file(qapp, synthetic_png):
    panel = UploadPanel()
    panel.original_slot.load_path(synthetic_png)
    panel.suspect_slot.load_path(synthetic_png)
    panel.original_slot.cutout = _cutout()
    panel.original_slot.geometry = IsolationGeometry(
        boundary=NormalizedPolygon([(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)]), boundary_origin="lifted"
    )
    panel.original_slot._refresh_status()
    assert "place it on the Suspect" in panel.original_slot.detail_label.text()

    panel.suspect_slot.geometry = IsolationGeometry(
        boundary=NormalizedPolygon([(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)]), boundary_origin="placed"
    )
    panel.original_slot._refresh_status()  # the suspect has its area now: nothing left to suggest
    assert "place it on" not in panel.original_slot.detail_label.text()


# ---- the outline is laid over at the size the two files' resolutions say ---------------------------------------------------------------


def test_the_other_files_outline_starts_at_the_size_the_two_resolutions_say():
    from types import SimpleNamespace

    cutout = _cutout_of_text()
    pdf_at = lambda dpi: SimpleNamespace(kind="pdf", working_dpi=lambda: dpi)  # noqa: E731
    twice = SimpleNamespace(_reference=SubjectReference(cutout, "a.pdf", source_dpi=150.0), _artwork=pdf_at(300.0))
    assert IsolationDialog._expected_scale(twice) == pytest.approx(2.0)
    same = SimpleNamespace(_reference=SubjectReference(cutout, "a.pdf", source_dpi=300.0), _artwork=pdf_at(300.0))
    assert IsolationDialog._expected_scale(same) == pytest.approx(1.0)
    # a photograph has no known resolution, nor has an outline lifted from one
    photo = SimpleNamespace(_reference=SubjectReference(cutout, "a.png"), _artwork=pdf_at(300.0))
    assert IsolationDialog._expected_scale(photo) is None
    scan = SimpleNamespace(_reference=SubjectReference(cutout, "a.pdf", source_dpi=300.0), _artwork=SimpleNamespace(kind="image", working_dpi=lambda: 300.0))
    assert IsolationDialog._expected_scale(scan) is None
    alone = SimpleNamespace(_reference=None, _artwork=pdf_at(300.0))
    assert IsolationDialog._expected_scale(alone) is None
