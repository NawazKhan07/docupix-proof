"""Comparing the artwork of each file: each is cleared to white outside its own area, then aligned and compared.

The reported need: "take the lifted area from both files, erase everything outside it, make a white
background, align them and compare them". Where the two areas differ, nothing is hidden: what only one
of them contains is compared too. What was erased by hand, in either file, is left out.
"""

from __future__ import annotations

import cv2
import numpy as np
import subject_scenes as scenes

from app.config import AppConfig, SubjectConfig
from app.engine.isolation.boundary_mask import boundary_mask, boundary_to_mask, erased_mask, resolve_keep_mask
from app.engine.isolation.subject_lifter import lift_subject
from app.engine.worker import _Area, _build_compare_mask, run_comparison
from app.models.isolation import IsolationGeometry, NormalizedPolygon, NormalizedRect
from app.models.job import ComparisonMode

CONFIG = AppConfig()
IDENTITY = np.eye(3)


def _rect_geometry(x0, y0, x1, y1, **kwargs) -> IsolationGeometry:
    return IsolationGeometry(boundary=NormalizedRect(x0, y0, x1, y1), **kwargs)


def _square_polygon(x0, y0, x1, y1) -> NormalizedPolygon:
    return NormalizedPolygon(points=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


# ---- a file's own area ----------------------------------------------------------------------------------


def test_the_file_is_cleared_to_white_outside_its_area_and_only_there():
    image = np.full((100, 200, 3), 90, np.uint8)
    area = _Area.of(image, _rect_geometry(0.25, 0.25, 0.75, 0.75))

    blanked = area.blanked(image)

    assert (blanked[:20] == 255).all() and (blanked[:, :40] == 255).all()  # outside: white
    assert (blanked[30:70, 60:140] == 90).all()  # inside: untouched
    assert (image == 90).all()  # the original picture is not modified


def test_nothing_chosen_is_the_whole_page_and_the_same_picture():
    image = np.full((50, 50, 3), 90, np.uint8)
    area = _Area.of(image, None)
    assert area.is_whole_page and area.blanked(image) is image


def test_erased_parts_are_cleared_too():
    image = np.full((100, 100, 3), 90, np.uint8)
    geometry = _rect_geometry(0.0, 0.0, 1.0, 1.0, erase_regions=[NormalizedRect(0.4, 0.4, 0.6, 0.6)])
    blanked = _Area.of(image, geometry).blanked(image)
    assert (blanked[45:55, 45:55] == 255).all() and (blanked[10:20, 10:20] == 90).all()


# ---- what is compared -----------------------------------------------------------------------------------


def _masks(original_geometry, suspect_geometry, shape=(100, 200), transform=IDENTITY):
    image = np.zeros((*shape, 3), np.uint8)
    return _build_compare_mask(
        shape, shape, transform, _Area.of(image, original_geometry), _Area.of(image, suspect_geometry)
    )


def test_the_compared_area_is_where_either_files_area_is_not_only_where_they_agree():
    """Two outlines that differ (a flap one file has and the other lacks): comparing only where they agree
    would hide whatever the flap carries."""
    mask = _masks(_rect_geometry(0.0, 0.0, 0.5, 1.0), _rect_geometry(0.4, 0.0, 1.0, 1.0))
    assert mask[:, :100].all() and mask[:, 100:].all()  # the original's half AND the suspect's, and their overlap


def test_a_file_with_no_area_of_its_own_is_taken_to_have_the_others():
    mask = _masks(_rect_geometry(0.25, 0.25, 0.75, 0.75), None)
    assert mask[30:70, 60:140].all() and not mask[:20].any() and not mask[:, :40].any()

    mask = _masks(None, _rect_geometry(0.25, 0.25, 0.75, 0.75))
    assert mask[30:70, 60:140].all() and not mask[:20].any() and not mask[:, :40].any()


def test_what_either_file_erased_is_left_out_of_the_comparison():
    original = _rect_geometry(0.0, 0.0, 1.0, 1.0, erase_regions=[NormalizedRect(0.0, 0.0, 0.25, 1.0)])
    suspect = _rect_geometry(0.0, 0.0, 1.0, 1.0, erase_regions=[NormalizedRect(0.75, 0.0, 1.0, 1.0)])
    mask = _masks(original, suspect)
    assert not mask[:, :50].any() and not mask[:, 150:].any() and mask[:, 60:140].all()


def test_only_the_part_the_suspect_covers_is_compared():
    shift = np.array([[1, 0, 50], [0, 1, 0], [0, 0, 1]], float)  # suspect pixels land 50 px to the right
    mask = _masks(None, _rect_geometry(0.0, 0.0, 1.0, 1.0), transform=shift)
    assert not mask[:, :50].any() and mask[:, 50:].all()


# ---- the line the outline is drawn along is not artwork ------------------------------------------------------


def test_a_polygon_area_starts_inside_its_own_edge_by_the_inset():
    polygon = _square_polygon(0.2, 0.2, 0.8, 0.8)
    plain = boundary_to_mask((200, 200), polygon)
    inset = boundary_to_mask((200, 200), polygon, inset_px=5)

    assert plain[40:160, 40:160].all()
    assert not inset[41:44, 50:150].any() and not inset[157:159, 50:150].any()  # the rim is out
    assert inset[52:148, 52:148].all()  # everything further in is still compared
    assert not inset[~plain].any()  # never larger than the outline


def test_a_rectangle_is_not_inset_a_crops_edge_is_not_a_line():
    geometry = _rect_geometry(0.2, 0.2, 0.8, 0.8, edge_inset_norm=0.05)
    assert boundary_mask((200, 200), geometry)[40:160, 40:160].all()


def test_the_inset_is_a_share_of_the_pages_long_edge():
    geometry = IsolationGeometry(boundary=_square_polygon(0.2, 0.2, 0.8, 0.8), edge_inset_norm=0.025)
    small = boundary_mask((200, 200), geometry)
    large = boundary_mask((400, 400), geometry)
    assert small[41:44, 100].sum() == 0 and large[82:88, 200].sum() == 0  # 5 px on 200, 10 px on 400
    assert small[48:60, 100].all() and large[92:110, 200].all()


def test_erased_and_boundary_masks_agree_with_the_combined_keep_mask():
    geometry = IsolationGeometry(
        boundary=_square_polygon(0.1, 0.1, 0.9, 0.9), erase_regions=[NormalizedRect(0.4, 0.4, 0.6, 0.6)]
    )
    keep = resolve_keep_mask((100, 100), geometry)
    assert np.array_equal(keep, boundary_mask((100, 100), geometry) & ~erased_mask((100, 100), geometry))
    assert erased_mask((100, 100), IsolationGeometry()) is None and boundary_mask((100, 100), None) is None


# ---- end to end -------------------------------------------------------------------------------------------------


def _sheet_pair():
    original, truth, _ = scenes.dieline_sheet()
    suspect = original.copy()
    cv2.rectangle(suspect, (240, 675), (520, 715), (226, 238, 246), -1)  # body copy removed, inside the carton
    return original, suspect, truth


def _lifted(image) -> IsolationGeometry:
    subject = lift_subject(image, (600, 800), SubjectConfig())
    return IsolationGeometry(
        boundary=subject.normalized_polygon(),
        boundary_origin="lifted",
        edge_inset_norm=subject.edge_inset_px / max(image.shape[:2]),
    )


def _outside_reports(outcome, truth) -> int:
    region = cv2.dilate(truth.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    count = 0
    for cluster in outcome.clusters:
        box = cluster.bbox_in_original
        x0, y0, x1, y1 = (int(v) for v in (box.x0, box.y0, box.x1, box.y1))
        count += int(not region[max(0, y0) : y1 + 1, max(0, x0) : x1 + 1].all())
    return count


def test_comparing_the_lifted_artwork_of_both_files_finds_the_change_inside_and_nothing_outside():
    original, suspect, truth = _sheet_pair()
    cv2.putText(suspect, "Carton Size : 99", (330, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)  # outside
    outcome = run_comparison(
        original, suspect, ComparisonMode.AUTOMATIC, CONFIG, _lifted(original), _lifted(suspect)
    )

    assert outcome.clusters and _outside_reports(outcome, truth) == 0
    assert any(c.bbox_in_original.y0 < 720 and c.bbox_in_original.y1 > 675 for c in outcome.clusters)


def test_the_viewer_still_gets_the_whole_suspect_page():
    """Only the comparison uses the cleared copies; the reviewer sees each file as it is."""
    original, suspect, _ = _sheet_pair()
    cv2.putText(suspect, "OUTSIDE", (330, 1500), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)
    outcome = run_comparison(original, suspect, ComparisonMode.AUTOMATIC, CONFIG, _lifted(original), _lifted(suspect))
    assert (outcome.aligned_suspect[1470:1510, 330:520] < 100).any()  # the text outside the carton is on show
    assert outcome.compared_outline is not None


def test_a_change_only_the_suspects_outline_contains_is_still_reported():
    """The suspect has a panel the original's outline does not take in: comparing only where the outlines
    agree would never look at it."""
    original, suspect, _ = _sheet_pair()
    height, width = original.shape[:2]
    cv2.rectangle(suspect, (200, 1200), (500, 1260), (30, 30, 30), -1)  # a printed block, well outside the carton

    narrow = _lifted(original)  # the original's outline: the carton
    wide = IsolationGeometry(boundary=NormalizedRect(0.0, 0.5, 1.0, 1.0))  # the suspect's: everything below the body

    outcome = run_comparison(original, suspect, ComparisonMode.AUTOMATIC, CONFIG, narrow, wide)

    assert any(c.bbox_in_original.y0 > 1150 for c in outcome.clusters), [c.bbox_in_original for c in outcome.clusters]


def test_an_erased_area_hides_a_change_inside_it():
    original, suspect, _ = _sheet_pair()
    geometry = _lifted(original)
    erased = IsolationGeometry(
        boundary=geometry.boundary,
        boundary_origin="lifted",
        edge_inset_norm=geometry.edge_inset_norm,
        erase_regions=[NormalizedRect(230 / 1300, 665 / 1560, 530 / 1300, 725 / 1560)],  # where the copy was removed
    )
    outcome = run_comparison(original, suspect, ComparisonMode.AUTOMATIC, CONFIG, erased, _lifted(suspect))
    assert not any(c.bbox_in_original.y0 < 720 and c.bbox_in_original.y1 > 675 for c in outcome.clusters)


def test_the_die_line_itself_is_not_compared():
    """A scan of the printed sheet has no die line, and a difference along the whole outline is not a
    difference in the artwork: comparison starts inside the line."""
    original = np.full((700, 900, 3), 255, np.uint8)
    cv2.rectangle(original, (100, 100), (800, 600), (20, 20, 20), 4)  # the die line
    cv2.putText(original, "ARTWORK 500 mg", (170, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (30, 30, 30), 3)
    suspect = original.copy()
    cv2.rectangle(suspect, (100, 100), (800, 600), (255, 255, 255), 4)  # the same artwork, printed without it

    subject = lift_subject(original, (450, 350), SubjectConfig(), method="border")
    assert subject.edge_inset_px >= 4  # the lift measured the line it followed

    def compared(inset_scale: float):
        geometry = IsolationGeometry(
            boundary=subject.normalized_polygon(),
            boundary_origin="lifted",
            edge_inset_norm=inset_scale * subject.edge_inset_px / max(original.shape[:2]),
        )
        outcome = run_comparison(original, suspect, ComparisonMode.AUTOMATIC, CONFIG, geometry, geometry)
        (polygon,) = outcome.compared_outline
        return cv2.contourArea(np.array(polygon, np.float32)), outcome.clusters

    plain_area, _ = compared(0.0)
    inset_area, findings = compared(1.0)

    assert plain_area - inset_area > 0.7 * 2 * (700 + 500) * subject.edge_inset_px  # it starts a line's width further in
    assert not findings  # and nothing is reported along the outline
