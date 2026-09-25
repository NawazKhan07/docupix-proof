"""The deep local comparison, the finer verdicts and the borders of findings (Automatic mode).

A small real change (a '36' become '24' in six-point type, a 0 become a 1 in a serial number, the stripes of a logo)
was hidden by the tolerance of the detectors that work on a reduced copy of the page, and the borders they drew round
what they did find were a few pixels off. `local_diff` looks where the two files differ at full size, tile by tile;
`verify` decides which differences are the same artwork drawn a little differently; the borders are then fitted to the
pixels that differ.
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from app.config import AppConfig, ClusteringConfig, DetectionConfig
from app.engine.aggregation.clustering import cluster_candidates
from app.engine.detectors import local_diff, verify
from app.engine.worker import _drop_die_lines, _fit_borders, run_comparison
from app.models.diff import BoundingBox, DiffCandidate, DiffCluster, DiffSubtype, DiffType
from app.models.job import ComparisonMode

CONFIG = DetectionConfig()
WHITE = 255


def _page(width=700, height=400) -> np.ndarray:
    return np.full((height, width, 3), WHITE, np.uint8)


def _text(page, text, origin, scale=1.0, thickness=2, colour=(20, 20, 20)):
    cv2.putText(page, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)
    return page


def _shifted(page: np.ndarray, dx: float, dy: float = 0.0) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(page, matrix, (page.shape[1], page.shape[0]), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))


# ---- local_diff: what it finds -------------------------------------------------------------------------------------------------------


def test_identical_pages_have_no_local_difference():
    page = _text(_page(), "Store below 25 C", (30, 120))
    result = local_diff.detect(page, page.copy(), CONFIG)
    assert result.candidates == [] and not result.residual.any()


def test_a_changed_digit_is_found_where_it_is_and_only_there():
    a = _text(_page(), "Batch 2500 EXP", (30, 120), 0.8)
    b = _text(_page(), "Batch 2600 EXP", (30, 120), 0.8)
    result = local_diff.detect(a, b, CONFIG)
    assert result.candidates
    (x0, y0, x1, y1) = (min(c.bbox.x0 for c in result.candidates), min(c.bbox.y0 for c in result.candidates),
                        max(c.bbox.x1 for c in result.candidates), max(c.bbox.y1 for c in result.candidates))
    (digits_x, _), _ = cv2.getTextSize("Batch 2", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    (changed_x, _), _ = cv2.getTextSize("Batch 25", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    # the box hugs the '5' / '6' (a few px of slack), it is not the whole line
    assert 30 + digits_x - 3 <= x0 and x1 <= 30 + changed_x + 6
    assert (x1 - x0) < 40 and (y1 - y0) < 40


def test_text_a_pixel_off_is_not_a_difference():
    page = _text(_page(), "Store below 25 C", (30, 120))
    assert local_diff.detect(page, _shifted(page, 1.0, 0.0), CONFIG).candidates == []
    assert local_diff.detect(page, _shifted(page, 0.0, -1.0), CONFIG).candidates == []


def test_a_tile_shifted_by_two_pixels_is_registered_and_what_is_left_is_no_change():
    """A carton bent by its folds is off by a different amount in different places: each tile finds its own shift.
    What a tile edge still leaves is the slice of a letter, which the verdict on each difference dismisses."""
    page = _text(_page(), "Store below 25 C in a dry place", (30, 120), 0.9)
    page = _text(page, "Keep out of reach of children", (30, 220), 0.9)
    shifted = _shifted(page, 2.0, 0.0)
    left = local_diff.detect(page, shifted, CONFIG)
    assert int(left.residual.sum()) < 40  # (unregistered, the two lines of type differ in thousands of pixels)
    assert verify.drop_residue(left.candidates, page, shifted, CONFIG) == []


def test_a_softer_copy_is_no_change_once_each_difference_is_judged():
    page = _text(_page(), "Store below 25 C", (30, 120))
    softer = cv2.GaussianBlur(page, (0, 0), 1.0)
    found = local_diff.detect(page, softer, CONFIG)
    assert verify.drop_residue(found.candidates, page, softer, CONFIG) == []


def test_a_digit_changed_in_small_type_is_found_at_any_page_size():
    """The tiles are 128 px and a page is rarely a whole number of them: a difference must be looked for in the tile it
    is in, at whatever size the page is (an index by a rounded ratio put it in a neighbouring one)."""
    for width, height in ((700, 300), (733, 411), (1290, 517)):
        a, b = _page(width, height), _page(width, height)
        _text(a, "Batch 2500 EXP", (width - 300, 120), 0.8)
        _text(b, "Batch 2600 EXP", (width - 300, 120), 0.8)
        assert local_diff.detect(a, b, CONFIG).candidates, (width, height)


def test_a_missing_dot_is_a_difference_and_says_it_is_missing():
    a = _page(200, 200)
    cv2.circle(a, (100, 100), 6, (0, 0, 0), -1)
    result = local_diff.detect(a, _page(200, 200), CONFIG)
    (found,) = result.candidates
    assert found.type is DiffType.GRAPHICS and found.subtype is DiffSubtype.REMOVED
    assert found.bbox.x0 >= 92 and found.bbox.x1 <= 108 and found.bbox.y0 >= 92 and found.bbox.y1 <= 108


def test_a_recoloured_block_is_a_colour_finding_at_its_own_exact_edge():
    a = _page(300, 300)
    b = a.copy()
    cv2.rectangle(a, (100, 100), (180, 160), (30, 60, 200), -1)
    cv2.rectangle(b, (100, 100), (180, 160), (200, 60, 30), -1)
    result = local_diff.detect(a, b, CONFIG)
    (found,) = result.candidates
    assert found.type is DiffType.COLOUR and found.region
    box = found.bbox
    assert abs(box.x0 - 100) <= 3 and abs(box.y0 - 100) <= 3 and abs(box.x1 - 181) <= 3 and abs(box.y1 - 161) <= 3


def test_only_the_compared_area_is_compared():
    a = _page(300, 300)
    b = a.copy()
    cv2.rectangle(b, (20, 20), (60, 60), (0, 0, 0), -1)  # outside the area
    cv2.rectangle(b, (200, 200), (240, 240), (0, 0, 0), -1)  # inside it
    mask = np.zeros((300, 300), bool)
    mask[150:, 150:] = True
    found = local_diff.detect(a, b, CONFIG, mask).candidates
    assert found and all(c.bbox.x0 >= 190 and c.bbox.y0 >= 190 for c in found)


# ---- verify: how much of a colour difference is a colour profile -------------------------------------------------------------------


def _solid_letter_pages(colour_a, colour_b):
    a, b = _page(300, 200), _page(300, 200)
    _text(a, "Ba", (40, 150), 4.0, 12, colour_a)
    _text(b, "Ba", (40, 150), 4.0, 12, colour_b)
    return a, b


def _colour_candidate(box, kind=DiffType.COLOUR, magnitude=30.0):
    return DiffCandidate(BoundingBox(*box), kind, DiffSubtype.CHANGED, 0.9, "Colour changed", magnitude)


def test_two_exports_that_render_one_ink_a_shade_apart_are_the_same_artwork():
    """A pink or black a few dE off across the whole page (a colour profile) is not a change to a piece of it."""
    a, b = _solid_letter_pages((236, 42, 140), (236, 12, 140))  # dE ~4, the same shapes
    assert verify.drop_residue([_colour_candidate((60, 60, 130, 140))], a, b, CONFIG) == []
    tight = replace(CONFIG, residue_max_ink_colour_delta=2.0)
    assert verify.drop_residue([_colour_candidate((60, 60, 130, 140))], a, b, tight) != []  # (so it is the allowance that decides)


def test_a_recolour_well_beyond_a_shade_is_kept():
    a, b = _solid_letter_pages((236, 42, 140), (30, 60, 200))
    assert verify.drop_residue([_colour_candidate((60, 60, 130, 140))], a, b, CONFIG) != []


def test_a_shape_finding_over_the_same_shapes_yields_to_the_colour_comparison_up_to_a_bigger_delta():
    """The similarity map says nothing about colour: where the marks coincide its finding is dropped for a colour
    difference that a colour finding over the same place would keep."""
    a, b = _solid_letter_pages((0, 90, 0), (60, 70, 0))
    shape = _colour_candidate((60, 60, 130, 140), DiffType.GRAPHICS)
    colour = _colour_candidate((60, 60, 130, 140), DiffType.COLOUR)
    kept_colour = verify.drop_residue([colour], a, b, CONFIG)
    kept_shape = verify.drop_residue([shape], a, b, CONFIG)
    assert len(kept_colour) >= len(kept_shape)
    strict = replace(CONFIG, residue_max_shape_colour_delta=0.0)
    assert len(verify.drop_residue([shape], a, b, strict)) == len(kept_colour)


def test_a_thin_line_a_pixel_and_a_half_off_is_not_a_change():
    a, b = _page(400, 100), _page(400, 100)
    cv2.line(a, (20, 50), (380, 50), (0, 0, 0), 1)
    cv2.line(b, (20, 52), (380, 52), (0, 0, 0), 1)
    assert verify.drop_residue([_colour_candidate((20, 47, 380, 56), DiffType.GRAPHICS)], a, b, CONFIG) == []


# ---- the detection profile: an exactly registered pair forgives one pixel only ----------------------------------------------------------


def _carton(serial: str) -> np.ndarray:
    """A sheet with enough on it to register exactly, and a serial number in type a few pixels tall."""
    page = np.full((900, 1400, 3), 235, np.uint8)
    for row in range(12):
        cv2.putText(page, f"Store below 25 C in a dry place, protect from light {row}", (60, 90 + row * 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.rectangle(page, (900, 100), (1300, 400), (30, 60, 200), -1)
    cv2.circle(page, (1100, 650), 120, (200, 120, 30), -1)
    cv2.putText(page, serial, (300, 800), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0), 1, cv2.LINE_AA)
    return page


def test_a_zero_become_a_one_in_tiny_type_on_an_exactly_registered_pair_is_a_change():
    outcome = run_comparison(_carton("SN EMJ3TLBE204300"), _carton("SN EMJ3TLBE204301"), ComparisonMode.AUTOMATIC, AppConfig())
    assert outcome.detection_profile == "exact"
    assert len(outcome.clusters) == 1
    box = outcome.clusters[0].bbox_in_original
    (start, _), _ = cv2.getTextSize("SN EMJ3TLBE20430", cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)
    assert 300 + start - 3 <= box.x0 and box.x1 <= 300 + start + 14 and box.width < 20 and box.height < 20


# ---- borders fitted to what differs ---------------------------------------------------------------------------------------------------


def _cluster(box: BoundingBox) -> DiffCluster:
    member = DiffCandidate(box, DiffType.GRAPHICS, DiffSubtype.CHANGED, 0.9, "Graphic changed", 10.0)
    return DiffCluster(
        id=1, bbox_in_original=box, bbox_in_suspect=box, type=DiffType.GRAPHICS,
        members=[member], detail="Graphic changed",
    )


def test_a_border_is_pulled_in_to_the_pixels_that_differ():
    residual = np.zeros((300, 300), bool)
    residual[115:145, 105:155] = True
    cluster = _cluster(BoundingBox(100, 110, 160, 150))  # a box drawn a little wide of the change
    _fit_borders([cluster], residual, np.eye(3))
    box = cluster.bbox_in_original
    assert (box.x0, box.y0, box.x1, box.y1) == (105, 115, 155, 145)
    assert cluster.bbox_in_suspect.x0 == pytest.approx(105)


def test_a_border_is_drawn_a_halo_outside_the_pixels_that_differ():
    residual = np.zeros((300, 300), bool)
    residual[115:145, 105:155] = True
    cluster = _cluster(BoundingBox(100, 110, 160, 150))
    _fit_borders([cluster], residual, np.eye(3), halo=2)
    box = cluster.bbox_in_original
    assert (box.x0, box.y0, box.x1, box.y1) == (103, 113, 157, 147)


def test_a_border_drawn_with_a_halo_stays_on_the_page():
    residual = np.zeros((100, 120), bool)
    residual[0:20, 0:30] = True  # a change at the corner of the page
    residual[80:100, 90:120] = True
    corner, far = _cluster(BoundingBox(0, 0, 40, 30)), _cluster(BoundingBox(80, 70, 120, 100))
    _fit_borders([corner, far], residual, np.eye(3), halo=3)
    assert (corner.bbox_in_original.x0, corner.bbox_in_original.y0) == (0, 0)
    assert (far.bbox_in_original.x1, far.bbox_in_original.y1) == (120, 100)


def test_a_border_is_left_alone_where_too_little_of_it_is_backed_by_differing_pixels():
    residual = np.zeros((300, 300), bool)
    residual[120:122, 110:114] = True  # a faint change: only a few pixels clear the tolerance
    original = BoundingBox(50, 50, 250, 250)
    cluster = _cluster(original)
    _fit_borders([cluster], residual, np.eye(3))
    assert cluster.bbox_in_original == original


def test_a_border_is_left_alone_where_nothing_differs_at_full_size():
    cluster = _cluster(BoundingBox(50, 50, 120, 120))
    before = cluster.bbox_in_original
    _fit_borders([cluster], np.zeros((300, 300), bool), np.eye(3))
    assert cluster.bbox_in_original == before


def test_a_box_scaled_by_its_own_factor_each_way():
    box = BoundingBox(10, 20, 30, 60).scaled_xy(2.0, 3.0)
    assert (box.x0, box.y0, box.x1, box.y1) == (20, 60, 60, 180)


# ---- a recoloured area is one finding, and what changed inside it is found on its own -------------------------------------------------------


def _panel(colour, digits="0"):
    page = np.full((500, 800, 3), 255, np.uint8)
    cv2.rectangle(page, (100, 80), (700, 420), colour, -1)
    cv2.rectangle(page, (100, 80), (700, 420), (30, 30, 30), 3)  # its frame
    for row, text in enumerate(("Nutrition table 12 34 56", "Contains: sugar, salt, water")):
        cv2.putText(page, text, (130, 160 + row * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(page, f"Lot 1001{digits}", (130, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    return page


def test_a_recoloured_panel_is_one_colour_finding_with_its_own_exact_box():
    result = local_diff.detect(_panel((250, 190, 50)), _panel((243, 165, 20)), CONFIG)
    regions = [c for c in result.candidates if c.region]
    (found,) = regions
    assert found.type is DiffType.COLOUR
    box = found.bbox
    assert abs(box.x0 - 100) <= 6 and abs(box.y0 - 80) <= 6 and abs(box.x1 - 700) <= 6 and abs(box.y1 - 420) <= 6
    assert result.colour_regions == 1


def test_what_changed_inside_a_recoloured_panel_is_still_found_on_its_own():
    """A digit changed in the panel that was recoloured: the panel is one finding, the digit is another."""
    result = local_diff.detect(_panel((250, 190, 50), "7"), _panel((243, 165, 20), "3"), CONFIG)
    assert len([c for c in result.candidates if c.region]) == 1
    digits = [c for c in result.candidates if not c.region]
    assert digits, "the digit changed"
    x0 = min(c.bbox.x0 for c in digits)
    x1 = max(c.bbox.x1 for c in digits)
    assert x1 - x0 < 60 and all(c.bbox.height < 60 for c in digits)  # the digit, not the panel or the line of type


def test_the_type_of_a_recoloured_panel_is_not_reported_as_changed_because_the_background_moved():
    result = local_diff.detect(_panel((250, 190, 50)), _panel((243, 165, 20)), CONFIG)
    assert [c for c in result.candidates if not c.region] == []


def test_a_graphic_that_is_the_colour_of_the_page_in_one_file_is_added_or_removed_not_recoloured():
    a, b = _page(400, 300), _page(400, 300)
    cv2.circle(a, (200, 150), 70, (30, 90, 200), -1)
    found = [c for c in local_diff.detect(a, b, CONFIG).candidates if c.bbox.width > 100]
    assert found and all(c.type is DiffType.GRAPHICS and c.subtype is DiffSubtype.REMOVED for c in found)
    again = [c for c in local_diff.detect(b, a, CONFIG).candidates if c.bbox.width > 100]
    assert again and all(c.type is DiffType.GRAPHICS and c.subtype is DiffSubtype.ADDED for c in again)


def test_a_shade_of_colour_profile_over_a_whole_page_is_no_finding_once_judged():
    a = _page(400, 300)
    cv2.circle(a, (200, 150), 60, (236, 42, 140), -1)
    b = a.copy()
    cv2.circle(b, (200, 150), 60, (236, 12, 140), -1)
    found = local_diff.detect(a, b, CONFIG).candidates
    assert verify.drop_residue(found, a, b, CONFIG) == []


# ---- die lines are not the artwork --------------------------------------------------------------------------------------------------------


def test_a_long_thin_line_that_one_file_draws_is_flagged_as_line_work_and_a_letter_is_not():
    a, b = _page(700, 400), _page(700, 400)
    cv2.line(a, (50, 200), (650, 200), (255, 0, 255), 2)  # a fold line only the first file has
    _text(b, "Batch 2500", (60, 100), 0.8)
    _text(a, "Batch 2600", (60, 100), 0.8)
    result = local_diff.detect(a, b, CONFIG)
    lines = [c for c in result.candidates if c.line_work]
    assert len(lines) == 1 and lines[0].bbox.width > 500
    letters = [c for c in result.candidates if not c.line_work]
    assert letters and all(c.bbox.width < 60 for c in letters)


def test_a_die_line_on_a_blank_flap_is_ignored_and_a_rule_beside_type_is_not():
    a, b = _page(900, 500), _page(900, 500)
    cv2.line(a, (60, 100), (500, 100), (255, 0, 255), 2)  # across blank paper
    for row in range(6):  # a table: type on both sides of its rule
        _text(a, "Cell text and more", (60, 260 + row * 30), 0.5, 1)
        _text(b, "Cell text and more", (60, 260 + row * 30), 0.5, 1)
    cv2.line(a, (50, 340), (450, 340), (0, 0, 0), 2)
    mask = np.ones(a.shape[:2], bool)
    found = local_diff.detect(a, b, CONFIG, mask).candidates
    kept, ignored = _drop_die_lines(found, a, b, mask, CONFIG)
    lines_kept = [c for c in kept if c.line_work]
    assert ignored >= 1
    assert all(c.bbox.y0 > 300 for c in lines_kept)  # the flap's line is gone; the table's rule stays


def test_a_fold_line_that_runs_the_whole_way_across_is_ignored_even_among_printing():
    a, b = _page(900, 1000), _page(900, 1000)
    for x in (100, 700):
        for row in range(0, 900, 40):
            _text(a, "printed panel text", (x, 60 + row), 0.5, 1)
            _text(b, "printed panel text", (x, 60 + row), 0.5, 1)
    cv2.line(a, (450, 20), (450, 980), (255, 0, 255), 2)  # a crease between two printed panels
    mask = np.ones(a.shape[:2], bool)
    kept, ignored = _drop_die_lines(local_diff.detect(a, b, CONFIG, mask).candidates, a, b, mask, CONFIG)
    assert ignored >= 1 and [c for c in kept if c.line_work] == []


def _dashes(y, x0, x1, dash=24, gap=6, thickness=7):
    """The findings a dashed line one file draws comes out as: a difference for every dash."""
    return [
        DiffCandidate(BoundingBox(x, y, min(x + dash, x1), y + thickness), DiffType.GRAPHICS, DiffSubtype.ADDED, 0.9, "Graphic added in suspect")
        for x in range(x0, x1, dash + gap)
    ]


def test_a_dashed_fold_line_one_file_draws_is_ignored_dash_by_dash_across_blank_paper():
    a, b = _page(900, 500), _page(900, 500)
    dashes = _dashes(100, 60, 60 + 22 * 30 - 6)
    mask = np.ones(a.shape[:2], bool)
    kept, ignored = _drop_die_lines(dashes, a, b, mask, CONFIG)
    assert kept == [] and ignored == len(dashes)


def test_a_long_dashed_line_is_ignored_even_among_printing_but_a_short_one_beside_type_is_not():
    a, b = _page(1000, 1000), _page(1000, 1000)
    for row in range(0, 900, 40):
        for x in (60, 620):
            _text(a, "printed panel text", (x, 60 + row), 0.5, 1)
            _text(b, "printed panel text", (x, 60 + row), 0.5, 1)
    crease, rule = _dashes(40, 40, 960), _dashes(500, 60, 200)  # 920 px across the top of the panels; 140 px among the type
    mask = np.ones(a.shape[:2], bool)
    kept, ignored = _drop_die_lines(crease + rule, a, b, mask, CONFIG)
    assert ignored == len(crease)  # the crease is gone
    assert kept == rule  # the short dashed rule beside type is still a difference


def test_a_few_marks_in_a_row_are_not_a_dashed_line():
    a, b = _page(900, 500), _page(900, 500)
    three = _dashes(100, 60, 60 + 3 * 30 - 6)
    assert len(three) == 3
    kept, ignored = _drop_die_lines(three, a, b, np.ones(a.shape[:2], bool), CONFIG)
    assert kept == three and ignored == 0
