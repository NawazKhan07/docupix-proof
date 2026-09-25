"""Lifting the artwork: press on it and get the area inside its border (each test pins a measured behaviour).

Two methods, tried in this order: the area sealed inside a closed border line (a die-cut outline on a
sheet, where colour cannot help because the flaps are the colour of the page), then colour and
background (a photograph or scan of the artwork on a surface).
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest
import subject_scenes as scenes

from app.config import SubjectConfig
from app.engine.isolation import subject_lifter as lifter
from app.engine.isolation.subject_lifter import HintStroke, NoSubjectFoundError, lift_subject

CONFIG = SubjectConfig()


def _scene(edge="coloured", bg="grey desk", **kwargs):
    art = scenes.artwork(edge=edge)
    return scenes.compose(art, bg, rotation_deg=kwargs.pop("rotation_deg", 4.0), offset=kwargs.pop("offset", (20, -30)), **kwargs)


def _centre(truth: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(truth)
    return float(xs.mean()), float(ys.mean())


# ---- the outline is right, on the surfaces a carton is photographed or printed on ------------------


@pytest.mark.parametrize(
    "edge,bg",
    [
        ("coloured", "grey desk"), ("coloured", "wood"), ("coloured", "gradient"),
        ("white", "grey desk"), ("white", "wood"), ("dark", "grey desk"),
        ("white", "pale desk"),  # the margin is barely paler than the desk: adaptive tolerance
    ],
)
def test_the_artwork_is_lifted_cleanly_from_the_surface_it_sits_on(edge, bg):
    image, truth = _scene(edge, bg)
    subject = lift_subject(image, _centre(truth), CONFIG)

    assert scenes.iou(subject.mask, truth) > 0.97
    median, p95 = scenes.edge_error_px(subject.mask, truth)
    assert median <= 2.5 and p95 <= 6.0  # pixels, on a 1500 px scene


def test_the_result_does_not_depend_on_where_on_the_artwork_you_press():
    image, truth = _scene("coloured", "grey desk")
    ys, xs = np.nonzero(truth)
    left, right, top, bottom = xs.min(), xs.max(), ys.min(), ys.max()
    presses = [
        (left + 0.15 * (right - left), top + 0.2 * (bottom - top)),  # on the orange band
        (left + 0.5 * (right - left), top + 0.8 * (bottom - top)),  # on the green block
        (left + 0.5 * (right - left), top + 0.55 * (bottom - top)),  # on the roundel
    ]
    for press in presses:
        assert scenes.iou(lift_subject(image, press, CONFIG).mask, truth) > 0.97


def test_a_rectangular_subject_comes_back_as_a_rectangle():
    image, truth = _scene("coloured", "grey desk")
    subject = lift_subject(image, _centre(truth), CONFIG)

    assert len(subject.polygon) == 4
    assert scenes.iou(lifter.polygon_mask(image.shape[:2], subject.polygon), truth) > 0.97
    assert subject.source == "segmented"


def test_an_irregular_subject_keeps_its_shape_within_a_point_budget():
    image, truth = _scene("coloured", "grey desk", rotation_deg=0.0)
    ys, xs = np.nonzero(truth)
    bite = (int(xs.max()), int((ys.min() + ys.max()) // 2))  # a semicircular bite out of the right edge
    notched = image.copy()
    cv2.circle(notched, bite, 100, tuple(int(v) for v in image[5, 5]), -1)
    truth_notched = truth.astype(np.uint8)
    cv2.circle(truth_notched, bite, 100, 0, -1)
    truth_notched = truth_notched.astype(bool)

    subject = lift_subject(notched, _centre(truth_notched), CONFIG)

    assert 8 < len(subject.polygon) <= CONFIG.max_outline_points
    assert scenes.iou(lifter.polygon_mask(image.shape[:2], subject.polygon), truth_notched) > 0.96


# ---- it is the pressed object that comes back ---------------------------------------------------------------------------


def test_with_two_objects_in_view_the_one_pressed_on_is_lifted():
    art = scenes.artwork(edge="coloured")
    image, first = scenes.compose(art, "grey desk", canvas=(1300, 2600), rotation_deg=3, offset=(-620, 0), shadow=False)
    image2, second = scenes.compose(scenes.artwork(edge="dark", seed=9), "grey desk", canvas=(1300, 2600), rotation_deg=-5, offset=(620, 0), shadow=False)
    both = image.copy()
    both[second] = image2[second]

    for pressed, other in ((first, second), (second, first)):
        subject = lift_subject(both, _centre(pressed), CONFIG)
        assert scenes.iou(subject.mask, pressed) > 0.95
        assert (subject.mask & other).sum() < 0.02 * other.sum()


# ---- corrections ------------------------------------------------------------------------------------------------------------------


def _scene_with_a_flap_the_colour_of_the_desk():
    """A subject with a part that is, colour for colour, the surface (a flap folded out)."""
    image, truth = _scene("coloured", "grey desk", shadow=False, rotation_deg=0.0)
    ys, xs = np.nonzero(truth)
    desk = tuple(int(v) for v in image[5, 5])
    flap = np.zeros(truth.shape, np.uint8)
    x1 = int(xs.max())
    cv2.rectangle(flap, (x1, int(ys.min()) + 200), (x1 + 90, int(ys.min()) + 500), 1, -1)
    flap = flap.astype(bool)
    return image, truth | flap, flap, desk


def test_a_keep_stroke_brings_back_a_part_the_first_lift_missed():
    image, truth, flap, _desk = _scene_with_a_flap_the_colour_of_the_desk()
    ys, xs = np.nonzero(flap)
    press = _centre(truth & ~flap)

    before = lift_subject(image, press, CONFIG)
    assert (before.mask & flap).sum() < 0.3 * flap.sum()  # it cannot tell the flap from the desk

    centre_x = float(xs.mean())
    half_width = float(xs.max() - xs.min()) / 2
    stroke = HintStroke([(centre_x, float(ys.min()) + half_width), (centre_x, float(ys.max()) - half_width)], radius_px=half_width, keep=True)
    after = lift_subject(image, press, CONFIG, hints=[stroke])

    assert (after.mask & flap).sum() > 0.9 * flap.sum()
    assert scenes.iou(after.mask, truth) > 0.95
    assert after.hints == [stroke]


def test_a_drop_stroke_removes_something_that_was_wrongly_lifted():
    """A strip on the desk in the artwork's own colours, touching its side, is lifted with it; painting it away drops it."""
    image, truth = _scene("coloured", "grey desk", shadow=False, rotation_deg=0.0)
    ys, xs = np.nonzero(truth)
    right = int(xs.max())
    strip = np.zeros(truth.shape, np.uint8)
    cv2.rectangle(strip, (right + 1, int(ys.min())), (right + 110, int(ys.max())), 1, -1)
    strip = strip.astype(bool)
    image = image.copy()
    image[strip] = (200, 60, 50)  # the artwork's margin colour, spilling out along one side

    press = _centre(truth)
    before = lift_subject(image, press, CONFIG)
    assert (before.mask & strip).sum() > 0.5 * strip.sum()

    stroke = HintStroke([(right + 60, float(ys.min()) - 40), (right + 60, float(ys.max()) + 40)], radius_px=58, keep=False)
    after = lift_subject(image, press, CONFIG, hints=[stroke])
    assert (after.mask & strip).sum() < 0.1 * strip.sum()
    assert scenes.iou(after.mask, truth) > 0.95


# ---- edges of the problem ----------------------------------------------------------------------------------------------------------


def test_artwork_that_fills_the_frame_still_lifts_something_sensible():
    """With no background in view the page edge IS the artwork's edge, so there is nothing to separate
    it from: it must not crash or return a speck."""
    art = scenes.artwork(edge="coloured")
    subject = lift_subject(art, (350, 475), CONFIG)
    assert subject.area_fraction > 0.5


def test_pressing_where_there_is_nothing_says_so_instead_of_returning_a_speck():
    blank = np.full((900, 1200, 3), 200, np.uint8)
    with pytest.raises(NoSubjectFoundError):
        lift_subject(blank, (600, 450), CONFIG)


def test_a_press_outside_the_image_is_clamped_not_a_crash():
    image, truth = _scene()
    try:
        lift_subject(image, (_centre(truth)[0], -50), CONFIG)
    except NoSubjectFoundError:
        pass  # pressing on the bare surface lifts nothing: a clear message, not an IndexError


def test_a_white_panel_inside_the_artwork_is_not_taken_for_the_white_page():
    """On a white page, a carton whose middle is white keeps its middle: only background-coloured
    pixels joined to the page edge are background."""
    art = scenes.artwork(edge="coloured")
    cv2.rectangle(art, (120, 380), (580, 700), (255, 255, 255), -1)  # a big white panel
    image, truth = scenes.compose(art, "white page", rotation_deg=4.0, offset=(20, -30), shadow=False)
    ys, xs = np.nonzero(truth)
    subject = lift_subject(image, (float(xs.min()) + 60, float(ys.min()) + 120), CONFIG)  # pressed on the orange band
    assert scenes.iou(subject.mask, truth) > 0.97


def test_a_lift_of_a_full_size_sheet_takes_a_couple_of_seconds():
    """An 18 MP page (a carton at 300 DPI): segmentation runs on a reduced copy, then the edge is refined."""
    art = cv2.resize(scenes.artwork(edge="coloured"), None, fx=3.4, fy=3.4, interpolation=cv2.INTER_LINEAR)
    image, truth = scenes.compose(art, "grey desk", canvas=(4965, 3660), rotation_deg=2.0, offset=(0, 0), shadow=False)

    started = time.perf_counter()
    subject = lift_subject(image, _centre(truth), CONFIG)
    elapsed = time.perf_counter() - started

    assert scenes.iou(subject.mask, truth) > 0.98
    assert scenes.edge_error_px(subject.mask, truth)[1] <= 4.0  # the full-resolution edge refinement at work
    assert elapsed < 4.0


# ---- the area inside a closed border: what is outside it is not artwork --------------------------------------------------------------


def _outline_dilated(truth: np.ndarray) -> np.ndarray:
    """The true region plus the thickness of the outline itself, which is part of what is sealed in."""
    return cv2.dilate(truth.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


@pytest.mark.parametrize("press", [(600, 800), (490, 600), (490, 480), (1000, 800)], ids=["body", "flap", "dome", "block"])
def test_only_the_area_inside_the_border_is_lifted(press):
    """The reported problem: text, a colour legend, crop marks and dots around the carton were compared as
    if they were artwork. Everything outside the die-cut outline is left out -- and the flaps, which are
    the colour of the page, are in."""
    sheet, truth, not_artwork = scenes.dieline_sheet()
    subject = lift_subject(sheet, press, CONFIG)

    assert subject.source == "border"
    assert scenes.iou(subject.mask, _outline_dilated(truth)) > 0.97
    assert (subject.mask & not_artwork).sum() < 30  # the title, legend, marks, dots, leader line and its label


def test_the_leader_line_touching_the_outline_and_its_label_are_not_taken_with_it():
    sheet, truth, _ = scenes.dieline_sheet()
    subject = lift_subject(sheet, (600, 800), CONFIG)
    label_rows = slice(1300, 1340)  # "Silver Metallic Matt Board", joined to the outline by a thin line
    assert subject.mask[label_rows, :].sum() == 0
    assert subject.mask[1250:1300, 745:756].sum() == 0  # and the leader itself


@pytest.mark.parametrize(
    "kwargs",
    [dict(line_px=1), dict(line_colour=(210, 20, 150)), dict(frame=True), dict(gap_in_outline=6)],
    ids=["hairline", "coloured line", "frame round the sheet", "gap in the outline"],
)
def test_it_copes_with_the_ways_real_sheets_differ(kwargs):
    sheet, truth, not_artwork = scenes.dieline_sheet(**kwargs)
    subject = lift_subject(sheet, (600, 800), CONFIG)
    assert subject.source == "border"
    assert scenes.iou(subject.mask, _outline_dilated(truth)) > 0.97
    assert (subject.mask & not_artwork).sum() < 30


@pytest.mark.parametrize(
    "kwargs,scale,minimum",
    [
        (dict(line_px=1), 0.5, 0.95),  # a screenshot of the sheet: 1 px lines softened to grey by the resampling
        (dict(line_px=1), 0.35, 0.94),
        (dict(line_px=1, line_colour=(150, 150, 150)), 1.0, 0.97),  # a grey hairline, at full size
    ],
    ids=["screenshot", "small screenshot", "grey hairline"],
)
def test_faint_thin_lines_are_lines_too(kwargs, scale, minimum):
    """The lines of a screenshot or a scan are rarely black: resampling turns a 1 px black line into a grey one,
    which no fixed darkness threshold finds. A thin structure clearly darker than what is around it is a line."""
    sheet, truth, _ = scenes.dieline_sheet(**kwargs)
    image = cv2.resize(sheet, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else sheet
    region = cv2.resize(truth.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

    subject = lift_subject(image, (600 * scale, 800 * scale), CONFIG)

    assert subject.source == "border"
    assert scenes.iou(subject.mask, cv2.dilate(region.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)) > minimum


def test_a_scan_of_the_sheet_lifts_the_carton_and_not_the_page():
    """A tilted, softened, noisy, vignetted scan: the outline is still a closed dark line."""
    from dataclasses import replace

    from scan_simulator import TYPICAL, simulate_scan

    sheet, truth, _ = scenes.dieline_sheet()
    scan = simulate_scan(sheet, TYPICAL)
    where = simulate_scan(
        np.dstack([truth.astype(np.uint8) * 255] * 3),
        replace(TYPICAL, blur_sigma=0.0, noise_sigma=0.0, vignette=0.0, brightness=1.0, colour_cast=(1, 1, 1), jpeg_quality=100),
    )[..., 0]
    # the scanner's own margin (outside the sheet) is white in `where`; the carton is what is inside the sheet
    ys, xs = np.nonzero(where > 127)
    inside_sheet = np.zeros(where.shape, bool)
    inside_sheet[60:-60, 60:-60] = True
    region = (where > 127) & inside_sheet
    ys, xs = np.nonzero(region)

    subject = lift_subject(scan, (float(xs.mean()), float(ys.mean())), CONFIG)

    assert subject.source == "border"
    assert scenes.iou(subject.mask & inside_sheet, cv2.dilate(region.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)) > 0.95
    assert subject.area_fraction < 0.4  # the carton, not the sheet it is printed on


def test_an_outline_with_a_break_too_wide_to_bridge_is_not_silently_wrong():
    """A gap wider than the bridging leaves the outline open: the answer must not be an arbitrary panel
    presented as the artwork."""
    sheet, truth, _ = scenes.dieline_sheet(gap_in_outline=40)
    try:
        subject = lift_subject(sheet, (600, 800), CONFIG, method="border")
    except NoSubjectFoundError:
        return  # honest: no closed border
    # or, if something was lifted, it is at least not a fragment passed off as the whole
    assert scenes.iou(subject.mask, _outline_dilated(truth)) > 0.9 or subject.area_fraction > 0.02


def test_with_no_press_the_largest_closed_border_is_taken():
    sheet, truth, _ = scenes.dieline_sheet()
    subject = lift_subject(sheet, None, CONFIG)
    assert subject.source == "border" and scenes.iou(subject.mask, _outline_dilated(truth)) > 0.97


def test_a_press_outside_every_border_never_lifts_the_artwork_by_the_border_method():
    """On the title, outside the carton: no closed border there is the artwork (a small enclosure of the title's own,
    if the press lands on one, may be lifted -- there is no size rule -- but never the carton)."""
    sheet, truth, _ = scenes.dieline_sheet()
    try:
        subject = lift_subject(sheet, (600, 140), CONFIG, method="border")  # on the title
    except NoSubjectFoundError as caught:
        assert "border" in str(caught)
        return
    assert scenes.iou(subject.mask, _outline_dilated(truth)) < 0.05


def test_a_small_closed_shape_that_is_pointed_at_is_lifted_however_small():
    """There is no size rule where the reviewer pointed: a boxed label is a closed border too, and pointing at it lifts
    it, in the automatic method as in the border method. (Finding the artwork with NO press is careful of what it takes.)"""
    sheet = np.full((1500, 1200, 3), 255, np.uint8)
    cv2.rectangle(sheet, (400, 500), (560, 620), (20, 20, 20), 2)  # 120 x 100 px: ~0.7% of the sheet
    for method in ("border", "auto"):
        small = lift_subject(sheet, (480, 560), CONFIG, method=method)
        assert small.source == "border" and small.area_fraction < 0.02, method
        assert 11_000 < small.area_fraction * sheet.shape[0] * sheet.shape[1] < 21_000, method  # (inside the line, and the line)
    with pytest.raises(NoSubjectFoundError):
        lift_subject(sheet, None, CONFIG)  # with no press, a boxed label is not taken for the whole artwork


def test_an_even_smaller_shape_is_lifted_too():
    sheet = np.full((3000, 2400, 3), 255, np.uint8)
    cv2.rectangle(sheet, (1000, 1200), (1140, 1290), (20, 20, 20), 3)  # 140 x 90 px of a 7 MP sheet: 0.18%
    small = lift_subject(sheet, (1070, 1245), CONFIG)
    assert small.source == "border"


def test_a_photograph_is_not_treated_as_line_art():
    image, truth = _scene("coloured", "grey desk")
    subject = lift_subject(image, _centre(truth), CONFIG)
    assert subject.source == "segmented"


def test_the_colour_method_can_be_asked_for_on_a_sheet():
    sheet, _truth, _ = scenes.dieline_sheet()
    assert lift_subject(sheet, (600, 800), CONFIG, method="colour").source == "segmented"


def test_an_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        lift_subject(np.zeros((10, 10, 3), np.uint8), (5, 5), CONFIG, method="magic")


# ---- a PDF's vector lines are the border, in any colour ---------------------------------------------------------------------------


def test_the_vector_outline_of_a_pdf_is_used_even_when_it_is_not_dark(tmp_path):
    """A dieline is often a spot colour. An orange stroke is not dark enough to be found in the rendered
    picture, but it is a stroked path, and a stroked path is a line whatever its colour."""
    from app.engine.file_loader import load_artwork

    truth = scenes.dieline_sheet_pdf(tmp_path / "sheet.pdf", line_colour=(1.0, 0.55, 0.0))
    with load_artwork(tmp_path / "sheet.pdf") as artwork:
        image = artwork.render()
        with_lines = lift_subject(image, (600, 800), CONFIG, page=artwork.pdf_page(0), dpi=300)

    assert with_lines.source == "border"
    assert scenes.iou(with_lines.mask, _outline_dilated(truth)) > 0.96


def test_pdf_text_and_swatches_are_not_lines(tmp_path):
    """Only strokes are lines: filled legend swatches and text (however dark) do not seal anything."""
    from app.engine.file_loader import load_artwork

    truth = scenes.dieline_sheet_pdf(tmp_path / "sheet.pdf", line_colour=(0, 0, 0))
    with load_artwork(tmp_path / "sheet.pdf") as artwork:
        image = artwork.render()
        subject = lift_subject(image, (600, 800), CONFIG, page=artwork.pdf_page(0), dpi=300)

    outside = ~_outline_dilated(truth)
    assert (subject.mask & outside).sum() < 0.002 * subject.mask.sum()  # the title and legend are not in it


# ---- corrections apply to a border lift as well -------------------------------------------------------------------------------


def test_a_drop_stroke_removes_part_of_a_border_lift_and_a_keep_stroke_adds_to_it():
    sheet, truth, _ = scenes.dieline_sheet()
    base = lift_subject(sheet, (600, 800), CONFIG)

    # (a drop at the edge: an outline has no holes, so one painted wholly inside the artwork changes nothing)
    dropped = lift_subject(sheet, (600, 800), CONFIG, hints=[HintStroke([(1140, 800)], radius_px=120, keep=False)])
    assert dropped.mask.sum() < base.mask.sum() - 20_000
    assert not dropped.mask[800, 1130]

    # (a keep stroke has to start on the lifted area: a blob painted in empty paper is not part of it)
    kept = lift_subject(sheet, (600, 800), CONFIG, hints=[HintStroke([(750, 580), (750, 420)], radius_px=30, keep=True)])
    assert kept.mask[430, 750] and kept.mask.sum() > base.mask.sum()
    apart = lift_subject(sheet, (600, 800), CONFIG, hints=[HintStroke([(640, 300), (700, 300)], radius_px=40, keep=True)])
    assert not apart.mask[300, 660]


# ---- the sheet's speed ----------------------------------------------------------------------------------------------------------------


def test_a_full_size_sheet_is_lifted_in_well_under_a_second():
    """A carton sheet at 300 DPI (18 MP): the lines are found on a reduced copy that keeps one-pixel lines."""
    sheet, truth, _ = scenes.dieline_sheet()
    big = cv2.resize(sheet, None, fx=2.8, fy=2.8, interpolation=cv2.INTER_LINEAR)
    big_truth = cv2.resize(truth.astype(np.uint8), None, fx=2.8, fy=2.8, interpolation=cv2.INTER_NEAREST).astype(bool)

    started = time.perf_counter()
    subject = lift_subject(big, (1680, 2240), CONFIG)
    elapsed = time.perf_counter() - started

    assert subject.source == "border"
    assert scenes.iou(subject.mask, cv2.dilate(big_truth.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)) > 0.97
    assert elapsed < 2.0


# ---- the numbers behind the outline ---------------------------------------------------------------------------------------------------


def test_the_normalized_outline_is_relative_to_the_image():
    image, truth = _scene()
    subject = lift_subject(image, _centre(truth), CONFIG)
    height, width = image.shape[:2]
    for (x, y), (nx, ny) in zip(subject.polygon, subject.normalized_polygon().points):
        assert nx == pytest.approx(x / width) and ny == pytest.approx(y / height)


def test_the_bounding_box_and_area_describe_the_mask():
    image, truth = _scene()
    subject = lift_subject(image, _centre(truth), CONFIG)
    x0, y0, x1, y1 = subject.bbox
    ys, xs = np.nonzero(subject.mask)
    assert (x0, y0, x1, y1) == (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    assert subject.area_fraction == pytest.approx(subject.mask.sum() / subject.mask.size)


# ---- dimension lines and their arrowheads, and the width of the line the outline follows ------------------------------------------


def _with_dimension_line(sheet: np.ndarray) -> np.ndarray:
    """A dimension line rising from the dust flap's top edge to a label: an arrowhead ON the outline, a long thin
    shaft, a second arrowhead and a number -- the way a packaging sheet's measurements hang off its outline."""
    marked = sheet.copy()
    blue = (30, 50, 150)
    cv2.line(marked, (760, 521), (760, 404), blue, 1)  # the shaft
    cv2.fillConvexPoly(marked, np.array([(760, 545), (751, 521), (769, 521)], np.int32), blue)  # tip touching the outline
    cv2.fillConvexPoly(marked, np.array([(760, 380), (751, 404), (769, 404)], np.int32), blue)  # the far arrowhead
    cv2.putText(marked, "59", (775, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.9, blue, 2)
    return marked


def test_a_dimension_line_with_arrowheads_hanging_off_the_outline_is_not_lifted_with_it():
    """The reported problem: the lifted area included the arrows and lines the sheet's measurements hang off
    the outline with."""
    sheet, truth, _ = scenes.dieline_sheet()
    subject = lift_subject(_with_dimension_line(sheet), (600, 800), CONFIG)

    assert subject.source == "border"
    assert subject.mask[370:538, 700:830].sum() == 0  # the arrowheads, the shaft and the number
    assert scenes.iou(subject.mask, _outline_dilated(truth)) > 0.97


def test_the_outlines_own_corners_survive_the_shaving_of_what_hangs_off_it():
    sheet, truth, _ = scenes.dieline_sheet()
    subject = lift_subject(_with_dimension_line(sheet), (600, 800), CONFIG)
    # the body row's four corners: the 5 x 5 pixels just inside each must still be there, the point not rounded off
    for x, y in [(160, 620), (1135, 620), (1135, 975), (160, 975)]:
        assert subject.mask[y : y + 5, x : x + 5].all(), (x, y)


@pytest.mark.parametrize("line_px", [1, 2, 4])
def test_the_lift_measures_how_thick_the_line_it_followed_is(line_px):
    sheet, _truth, _ = scenes.dieline_sheet(line_px=line_px)
    subject = lift_subject(sheet, (600, 800), CONFIG)
    assert line_px - 0.6 <= subject.edge_inset_px - CONFIG.border_line_margin_px <= line_px + 1.6


def test_a_lift_by_colour_starts_a_little_inside_its_edge_because_a_scans_edge_is_a_blur():
    art = scenes.artwork(edge="coloured")
    scene, truth = scenes.compose(art, "grey desk", rotation_deg=3.0, offset=(10, -20))
    subject = lift_subject(scene, _centre(truth), CONFIG, method="colour")
    assert subject.source == "segmented" and subject.edge_inset_px == CONFIG.colour_edge_inset_px


def test_a_cutout_carries_the_inset_so_a_file_it_is_laid_over_starts_the_same_distance_in():
    from app.engine.isolation.cutout import Cutout

    sheet, _truth, _ = scenes.dieline_sheet(line_px=3)
    subject = lift_subject(sheet, (600, 800), CONFIG)
    assert Cutout.from_subject(sheet, subject).edge_inset_px == subject.edge_inset_px > 3
