"""A lifted subject as a cutout, and laying it over another file (trim a suspect that has no border of its own)."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
import subject_scenes as scenes
from PIL import Image

from app.config import AlignmentConfig, SubjectConfig
from app.engine.alignment.aligner import AlignmentError, align
from app.engine.isolation.cutout import Cutout, CutoutError
from app.engine.isolation.placement import Placement, fit_placement, placement_from_matrix
from app.engine.isolation.subject_lifter import lift_subject
from app.models.isolation import IsolationGeometry, NormalizedPolygon

CONFIG = SubjectConfig()


def _lifted_cutout():
    image, truth = scenes.compose(scenes.artwork(edge="coloured"), "grey desk", rotation_deg=3.0, offset=(20, -30))
    ys, xs = np.nonzero(truth)
    subject = lift_subject(image, (xs.mean(), ys.mean()), CONFIG)
    return image, truth, subject, Cutout.from_subject(image, subject)


# ---- the cutout ------------------------------------------------------------------------------------------


def test_a_cutout_is_the_artwork_alone_cropped_to_its_box():
    image, _truth, subject, cutout = _lifted_cutout()
    x0, y0, x1, y1 = subject.bbox

    assert cutout.size == (x1 - x0, y1 - y0)
    assert cutout.origin == (x0, y0)
    assert cutout.rgba.shape[2] == 4
    # inside the outline it is the artwork, opaque; outside it is transparent
    inside = subject.mask[y0:y1, x0:x1]
    assert (cutout.rgba[..., 3][inside & cv2.erode(inside.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)] == 255).all()
    assert (cutout.rgba[..., 3][~cv2.dilate(inside.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)] == 0).all()
    assert np.array_equal(cutout.rgba[..., :3][inside], image[y0:y1, x0:x1][inside])  # the pixels are untouched


def test_the_outline_is_carried_in_the_cutouts_own_pixels():
    _image, _truth, subject, cutout = _lifted_cutout()
    x0, y0, _, _ = subject.bbox
    for (sx, sy), (cx, cy) in zip(subject.polygon, cutout.polygon):
        assert (cx, cy) == pytest.approx((sx - x0, sy - y0))


def test_on_white_flattens_the_transparency():
    _image, _truth, _subject, cutout = _lifted_cutout()
    flat = cutout.on_white()
    assert flat.shape[2] == 3 and flat.dtype == np.uint8
    assert (flat[cutout.rgba[..., 3] == 0] == 255).all()


def test_a_saved_cutout_reads_back_with_its_outline(tmp_path):
    _image, _truth, subject, cutout = _lifted_cutout()

    saved = cutout.save_png(tmp_path / "carton")  # the .png suffix is supplied
    loaded = Cutout.load_png(saved, CONFIG)

    assert saved.suffix == ".png"
    assert loaded.size == cutout.size
    assert loaded.origin is None  # where it came from is not in the file
    assert np.array_equal(loaded.rgba[..., :3][loaded.mask], cutout.rgba[..., :3][cutout.mask])
    mask_a = cv2.fillPoly(np.zeros(cutout.rgba.shape[:2], np.uint8), [np.round(np.array(cutout.polygon)).astype(np.int32)], 1).astype(bool)
    mask_b = cv2.fillPoly(np.zeros(loaded.rgba.shape[:2], np.uint8), [np.round(np.array(loaded.polygon)).astype(np.int32)], 1).astype(bool)
    assert scenes.iou(mask_a, mask_b) > 0.99


def test_a_saved_cutout_is_lossless(tmp_path):
    _image, _truth, _subject, cutout = _lifted_cutout()
    reread = np.array(Image.open(cutout.save_png(tmp_path / "c.png")).convert("RGBA"))
    assert np.array_equal(reread, cutout.rgba)  # no lossy re-encoding of the artwork


def test_a_picture_with_no_transparency_is_refused_with_advice(tmp_path):
    Image.fromarray(np.full((50, 50, 3), 200, np.uint8)).save(tmp_path / "flat.png")
    with pytest.raises(CutoutError) as caught:
        Cutout.load_png(tmp_path / "flat.png")
    assert "no transparent background" in str(caught.value)


def test_a_fully_transparent_picture_is_refused(tmp_path):
    Image.fromarray(np.zeros((50, 50, 4), np.uint8), "RGBA").save(tmp_path / "empty.png")
    with pytest.raises(CutoutError):
        Cutout.load_png(tmp_path / "empty.png")


def test_something_that_is_not_an_image_is_refused(tmp_path):
    (tmp_path / "x.png").write_bytes(b"not a png")
    with pytest.raises(CutoutError):
        Cutout.load_png(tmp_path / "x.png")


# ---- where a cutout sits ------------------------------------------------------------------------------------


def test_a_placement_puts_the_cutouts_centre_where_it_says():
    size = (200, 100)
    placement = Placement((500.0, 400.0), rotation_deg=30.0, scale=1.7)
    (cx, cy), = placement.map_points([(100.0, 50.0)], size)
    assert (cx, cy) == pytest.approx((500.0, 400.0))


def test_a_positive_rotation_is_clockwise_on_screen_and_scales_about_the_centre():
    size = (200, 200)
    turned = Placement((0.0, 0.0), rotation_deg=90.0, scale=1.0)
    right_of_centre, = turned.map_points([(200.0, 100.0)], size)  # the point 100 px right of the centre
    assert right_of_centre == pytest.approx((0.0, 100.0))  # ...is now 100 px BELOW it (y grows downward)

    doubled = Placement((0.0, 0.0), rotation_deg=0.0, scale=2.0)
    corner, = doubled.map_points([(0.0, 0.0)], size)
    assert corner == pytest.approx((-200.0, -200.0))


@pytest.mark.parametrize("rotation,scale,centre", [(0.0, 1.0, (10.0, 20.0)), (7.5, 0.8, (300.0, 250.0)), (-31.0, 1.9, (-40.0, 90.0))])
def test_a_matrix_is_recovered_as_the_placement_that_made_it(rotation, scale, centre):
    size = (240, 180)
    made = Placement(centre, rotation, scale)

    recovered, error = placement_from_matrix(made.matrix(size), size)

    assert recovered.rotation_deg == pytest.approx(rotation, abs=1e-3)
    assert recovered.scale == pytest.approx(scale, abs=1e-4)
    assert recovered.center == pytest.approx(centre, abs=1e-2)
    assert error < 1e-2


def test_a_matrix_that_is_not_quite_a_similarity_is_approximated_and_the_error_reported():
    size = (400, 300)
    skewed = Placement((200.0, 150.0), 3.0, 1.0).matrix(size).copy()
    skewed[2, 0] = 1e-5  # a touch of perspective
    placement, error = placement_from_matrix(skewed, size)
    assert placement.scale == pytest.approx(1.0, abs=0.05)
    assert 0.0 < error < 3.0  # close, and honest about how close


# ---- snapping it into place -----------------------------------------------------------------------------------


def _suspect_scene(rotation_deg: float, scale: float, offset, bg: str = "wood", canvas=(1500, 1700)):
    """The same artwork photographed differently: turned, at another scale, on another surface."""
    return scenes.compose(scenes.artwork(edge="coloured"), bg, canvas=canvas, rotation_deg=rotation_deg, scale=scale, offset=offset, seed=9)


@pytest.mark.parametrize(
    "rotation,scale,offset",
    [(0.0, 1.0, (0, 0)), (7.0, 0.8, (60, -40)), (-4.0, 1.25, (-80, 50))],
    ids=["same place", "turned and smaller", "turned the other way, larger"],
)
def test_the_outline_is_snapped_onto_the_artwork_by_matching_what_is_printed_on_it(rotation, scale, offset):
    _image, _truth, _subject, cutout = _lifted_cutout()
    target, target_truth = _suspect_scene(rotation, scale, offset)

    fit = fit_placement(cutout, target, AlignmentConfig())

    placed = cv2.fillPoly(
        np.zeros(target.shape[:2], np.uint8),
        [np.round(np.array(fit.placement.map_points(cutout.polygon, cutout.size))).astype(np.int32)],
        1,
    ).astype(bool)
    assert scenes.iou(placed, target_truth) > 0.96
    assert fit.outline_error_px < 5.0  # worst point of the cutout, px: how far a pure similarity is from the registration
    assert not fit.alignment.low_confidence


def test_a_cutout_that_is_not_in_the_target_cannot_be_placed_and_says_so():
    _image, _truth, _subject, cutout = _lifted_cutout()
    unrelated = scenes.background("wood", (900, 1100))
    cv2.putText(unrelated, "nothing here", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    with pytest.raises(AlignmentError):
        fit_placement(cutout, unrelated, AlignmentConfig())


def _features_fail(monkeypatch):
    """Plain artwork, few marks: the features cannot be matched (so the pictures' edges have to be)."""
    import app.engine.isolation.placement as placement_module

    def refuse(*_args, **_kwargs):
        raise AlignmentError("too few features")

    monkeypatch.setattr(placement_module, "align", refuse)


@pytest.mark.parametrize("scale", [0.85, 1.0], ids=["a size smaller", "the same size"])
def test_where_the_features_cannot_be_matched_the_outline_is_found_by_its_edges_at_its_size(monkeypatch, scale):
    _features_fail(monkeypatch)
    _image, _truth, _subject, cutout = _lifted_cutout()
    target, target_truth = scenes.compose(scenes.artwork(edge="coloured"), "grey desk", canvas=(1500, 1700), rotation_deg=0.0, scale=scale, offset=(0, 0), seed=9)

    fit = fit_placement(cutout, target, AlignmentConfig(), expected_scale=scale)

    assert fit.alignment.method == "template"
    assert fit.placement.scale == pytest.approx(scale, abs=0.03)
    placed = cv2.fillPoly(
        np.zeros(target.shape[:2], np.uint8), [np.round(np.array(fit.placement.map_points(cutout.polygon, cutout.size))).astype(np.int32)], 1
    ).astype(bool)
    assert scenes.iou(placed, target_truth) > 0.9


def test_the_edges_are_looked_for_at_the_same_resolution_if_nothing_says_otherwise(monkeypatch):
    _features_fail(monkeypatch)
    _image, _truth, _subject, cutout = _lifted_cutout()
    target, _ = scenes.compose(scenes.artwork(edge="coloured"), "grey desk", canvas=(1500, 1700), rotation_deg=0.0, scale=1.0, offset=(0, 0), seed=9)
    assert fit_placement(cutout, target, AlignmentConfig()).placement.scale == pytest.approx(1.0, abs=0.03)


def test_an_outline_that_is_nowhere_in_the_picture_is_not_placed_by_its_edges_either(monkeypatch):
    _features_fail(monkeypatch)
    _image, _truth, _subject, cutout = _lifted_cutout()
    unrelated = scenes.background("wood", (1500, 1700))
    with pytest.raises(AlignmentError):
        fit_placement(cutout, unrelated, AlignmentConfig())


def test_the_cutouts_own_outline_never_anchors_the_registration():
    """The fit rests on the artwork's features: a polygon that was lifted (or placed) is only a mask."""
    _image, _truth, _subject, cutout = _lifted_cutout()
    target, _ = _suspect_scene(5.0, 0.9, (30, 30))
    fit = fit_placement(cutout, target, AlignmentConfig())
    assert "dieline_polygon" not in fit.alignment.method
    assert fit.alignment.residual_basis == "keypoints"


# ---- a lifted or placed polygon masks the comparison but never registers it --------------------------------------------


def _squares(width=600, height=400):
    image = np.full((height, width, 3), 255, np.uint8)
    cv2.rectangle(image, (60, 60), (540, 340), (30, 30, 30), 3)
    cv2.putText(image, "PARACETAMOL 500mg", (90, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
    cv2.rectangle(image, (90, 200), (260, 300), (200, 30, 40), -1)
    cv2.circle(image, (430, 250), 50, (30, 90, 200), -1)
    return image


def _polygon_geometry(origin: str | None):
    polygon = NormalizedPolygon(points=[(60 / 600, 60 / 400), (540 / 600, 60 / 400), (540 / 600, 340 / 400), (60 / 600, 340 / 400)])
    return IsolationGeometry(boundary=polygon) if origin is None else IsolationGeometry(boundary=polygon, boundary_origin=origin)


def test_a_traced_polygon_registers_the_files_as_before():
    image = _squares()
    result = align(image, image.copy(), AlignmentConfig(), _polygon_geometry(None), _polygon_geometry(None))
    assert result.method.startswith("dieline_polygon")


@pytest.mark.parametrize("origin", ["lifted", "placed"])
def test_a_lifted_or_placed_polygon_does_not_register_the_files(origin):
    image = _squares()
    result = align(image, image.copy(), AlignmentConfig(), _polygon_geometry(origin), _polygon_geometry(origin))
    assert result.method == "features" and result.residual_basis == "keypoints"


def test_only_the_side_with_an_exact_polygon_cannot_register_alone():
    """Registration needs BOTH files' polygons to be exact; one lifted polygon is enough to fall back to features."""
    image = _squares()
    result = align(image, image.copy(), AlignmentConfig(), _polygon_geometry(None), _polygon_geometry("placed"))
    assert result.method == "features"


def test_the_geometry_says_which_boundaries_may_anchor_registration():
    assert IsolationGeometry().anchors_alignment
    assert IsolationGeometry(boundary_origin="manual").anchors_alignment
    assert not IsolationGeometry(boundary_origin="lifted").anchors_alignment
    assert not IsolationGeometry(boundary_origin="placed").anchors_alignment


def test_a_placed_outline_trims_the_comparison_area():
    """The point of placing it: everything outside the outline is masked out of the comparison."""
    from app.engine.isolation.boundary_mask import resolve_keep_mask

    image = _squares()
    keep = resolve_keep_mask(image.shape[:2], _polygon_geometry("placed"))
    assert keep[200, 300] and not keep[20, 20] and not keep[380, 580]
    assert math.isclose(keep.mean(), (480 * 280) / (600 * 400), rel_tol=0.02)
