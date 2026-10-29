"""Comparing one artwork against every repeat of it on a sheet (`app.engine.series`)."""

from __future__ import annotations

import numpy as np
import pytesseract
import pytest
import series_scenes as scenes

from app.config import AppConfig
from app.engine.ocr_locator import find_tesseract
from app.engine.series import NoRepeatsFoundError, compare_series, find_repeats
from app.models.diff import DiffSubtype
from app.models.isolation import IsolationGeometry, NormalizedRect
from app.models.job import ComparisonMode

CONFIG = AppConfig().series


@pytest.fixture(autouse=True)
def _ocr(monkeypatch):
    """So a text defect reads as a text finding, as it would in the app -- scoped to this test (like every other
    file's OCR fixture), so it does not leave Tesseract configured for whatever test runs next."""
    tesseract = find_tesseract()
    if tesseract is not None:
        monkeypatch.setattr(pytesseract.pytesseract, "tesseract_cmd", str(tesseract))


def _geometry(box: tuple[int, int, int, int], size: tuple[int, int]) -> IsolationGeometry:
    x0, y0, x1, y1 = box
    width, height = size
    return IsolationGeometry(boundary=NormalizedRect(x0 / width, y0 / height, x1 / width, y1 / height))


# ---- finding the repeats ---------------------------------------------------------------------------------------


def test_every_clean_repeat_is_found_in_reading_order():
    template = scenes.artwork(None)
    sheet = scenes.sheet([None] * 6)

    boxes = find_repeats(template, sheet, CONFIG)

    assert len(boxes) == 6
    # top row's boxes come before the bottom row's, and left to right within a row
    rows = [round(box.y0 / scenes.CELL_H) for box in boxes]
    assert rows == sorted(rows)
    top_row = [box.x0 for box in boxes if round(box.y0 / scenes.CELL_H) == rows[0]]
    assert top_row == sorted(top_row)


def test_a_repeat_that_was_edited_is_still_found_by_its_edges():
    template = scenes.artwork(None)
    sheet = scenes.sheet([None, "text", "colour", "graphic"])
    assert len(find_repeats(template, sheet, CONFIG)) == 4


def test_a_sheet_with_only_one_repeat_still_works():
    template = scenes.artwork(None)
    sheet = scenes.sheet([None])
    boxes = find_repeats(template, sheet, CONFIG)
    assert len(boxes) == 1


def test_nothing_on_an_unrelated_sheet_is_found():
    template = scenes.artwork(None)
    unrelated = np.full((900, 700, 3), (40, 90, 150), np.uint8)
    assert find_repeats(template, unrelated, CONFIG) == []


def test_a_box_covers_its_repeat_with_the_configured_margin():
    template = scenes.artwork(None)
    sheet = scenes.sheet([None])
    (box,) = find_repeats(template, sheet, CONFIG)
    # the one repeat sits at (40, 40) to (40+360, 40+520) on this 1x1 sheet
    assert box.x0 <= 40 and box.y0 <= 40
    assert box.x1 >= 40 + scenes.CELL_W and box.y1 >= 40 + scenes.CELL_H


def test_the_expected_scale_narrows_the_search_and_still_finds_it():
    template = scenes.artwork(None)
    sheet = scenes.sheet([None])
    assert len(find_repeats(template, sheet, CONFIG, expected_scale=1.0)) == 1


# ---- comparing every repeat -------------------------------------------------------------------------------------


def test_each_defect_is_found_and_attributed_to_its_own_repeat():
    page, box = scenes.original_page()
    geometry = _geometry(box, (page.shape[1], page.shape[0]))
    sheet = scenes.sheet([None, "text", None, "colour", None, "graphic"])

    outcome = compare_series(page, sheet, ComparisonMode.AUTOMATIC, AppConfig(), geometry)

    assert outcome.notes and "6 repeats" in outcome.notes[0]
    by_repeat = {}
    for cluster in outcome.clusters:
        for n in range(1, 7):
            if cluster.detail.startswith(f"Repeat {n} of 6:"):
                by_repeat.setdefault(n, []).append(cluster)
    assert set(by_repeat) == {2, 4, 6}  # the clean repeats (1, 3, 5) have no findings
    assert by_repeat[2][0].category == "text"
    assert by_repeat[4][0].category == "colour"
    assert by_repeat[6][0].category == "graphics"


def test_findings_are_shown_where_their_repeat_actually_sits_on_the_sheet():
    page, box = scenes.original_page()
    geometry = _geometry(box, (page.shape[1], page.shape[0]))
    sheet = scenes.sheet([None, None, None, "colour"])  # repeat 4: bottom-left cell

    outcome = compare_series(page, sheet, ComparisonMode.AUTOMATIC, AppConfig(), geometry)

    (cluster,) = outcome.clusters
    # the bottom-left cell of a 3-wide grid starts at x=40, y=40+520+40
    assert cluster.bbox_in_suspect.x0 < scenes.CELL_W
    assert cluster.bbox_in_suspect.y0 > scenes.CELL_H


def test_the_suspect_pane_is_the_whole_sheet_and_the_original_pane_the_whole_page():
    page, box = scenes.original_page()
    geometry = _geometry(box, (page.shape[1], page.shape[0]))
    sheet = scenes.sheet([None, None])

    outcome = compare_series(page, sheet, ComparisonMode.AUTOMATIC, AppConfig(), geometry)

    assert outcome.aligned_suspect.shape[:2] == sheet.shape[:2]
    assert outcome.aligned_original.shape[:2] == page.shape[:2]


def test_no_repeats_found_says_so_rather_than_returning_nothing():
    page, box = scenes.original_page()
    geometry = _geometry(box, (page.shape[1], page.shape[0]))
    unrelated = np.full((900, 700, 3), (40, 90, 150), np.uint8)

    with pytest.raises(NoRepeatsFoundError):
        compare_series(page, unrelated, ComparisonMode.AUTOMATIC, AppConfig(), geometry)


def test_ids_are_renumbered_in_reading_order_across_every_repeat():
    page, box = scenes.original_page()
    geometry = _geometry(box, (page.shape[1], page.shape[0]))
    sheet = scenes.sheet(["graphic", None, "colour", None, "text", None])

    outcome = compare_series(page, sheet, ComparisonMode.AUTOMATIC, AppConfig(), geometry)

    ids = [cluster.id for cluster in outcome.clusters]
    assert ids == list(range(1, len(ids) + 1))
    ys = [cluster.bbox_in_suspect.y0 for cluster in outcome.clusters]
    assert ys == sorted(ys)  # reading order top to bottom (this sheet's defects are one per row)


# ---- a reel as wide as its artwork; an original that is a whole job sheet -------------------------------------------


def _reel(count: int, defects: dict[int, str] | None = None, gap: int = 30) -> np.ndarray:
    """A reel (a foil, a label roll): copies of the artwork one under the other, the reel exactly as wide as a copy."""
    defects = defects or {}
    parts = []
    for index in range(count):
        parts.append(np.full((gap, scenes.CELL_W, 3), scenes.SHEET_BG, np.uint8))
        parts.append(scenes.artwork(defects.get(index)))
    parts.append(np.full((gap, scenes.CELL_W, 3), scenes.SHEET_BG, np.uint8))
    return np.vstack(parts)


def _job_sheet() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """The original as a design house sends it: the one artwork, with dimension lines round it and a job sheet
    (a table of approvals) under it, on a page much bigger than the artwork. Returns the page and the artwork's box."""
    import cv2

    page = np.full((1500, 1000, 3), 255, np.uint8)
    x0, y0 = 300, 160
    page[y0 : y0 + scenes.CELL_H, x0 : x0 + scenes.CELL_W] = scenes.artwork(None)
    cv2.line(page, (x0, y0 - 40), (x0 + scenes.CELL_W, y0 - 40), (0, 0, 0), 2)  # a dimension line
    cv2.putText(page, "132 mm", (x0 + 130, y0 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    for row in range(8):  # the job sheet
        y = 800 + row * 70
        cv2.rectangle(page, (60, y), (940, y + 60), (0, 0, 0), 2)
        cv2.putText(page, f"Approved by {'ABCDEFGH'[row]} on 0{row + 1}/05/2025", (80, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    return page, (x0, y0, x0 + scenes.CELL_W, y0 + scenes.CELL_H)


def test_every_copy_on_a_reel_as_wide_as_the_artwork_is_found():
    boxes = find_repeats(scenes.artwork(None), _reel(8), AppConfig().series)
    assert len(boxes) == 8


def test_half_a_copy_at_a_cut_end_of_the_reel_is_not_a_repeat():
    reel = _reel(6)[: -(scenes.CELL_H // 2)]  # cut through the last copy
    assert len(find_repeats(scenes.artwork(None), reel, AppConfig().series)) == 5


def test_the_one_artwork_is_found_on_a_whole_job_sheet_from_the_copies_on_the_reel():
    from app.engine.series import find_on_original, sheet_unit

    page, (x0, y0, x1, y1) = _job_sheet()
    reel = _reel(8)
    unit = sheet_unit(reel)
    assert unit is not None
    ux0, uy0, ux1, uy1 = unit
    assert ux1 - ux0 == scenes.CELL_W and abs((uy1 - uy0) - (scenes.CELL_H + 30)) <= 6  # a copy's length, and its gap
    found = find_on_original(page, reel, AppConfig().series)
    assert found is not None
    fx0, fy0, fx1, fy1 = found[1]
    assert fx0 <= x0 + 8 and fy0 <= y0 + 8 and fx1 >= x1 - 8 and fy1 >= y1 - 8  # the artwork, whole
    assert fy1 < 800  # and not the job sheet


def test_a_sheet_that_does_not_repeat_offers_no_copy():
    from app.engine.series import sheet_unit

    assert sheet_unit(scenes.artwork(None)) is None


def test_a_whole_job_sheet_original_is_compared_against_every_copy_on_the_reel():
    page, _box = _job_sheet()
    outcome = compare_series(page, _reel(6, {2: "text"}), ComparisonMode.SERIES, AppConfig())
    assert any("6 repeats" in note for note in outcome.notes)
    assert any("found on its page" in note for note in outcome.notes)
    repeats_with_findings = {cluster.detail.split(":")[0] for cluster in outcome.clusters}
    assert "Repeat 3 of 6" in repeats_with_findings


def test_the_artwork_is_compared_cut_out_of_its_page_and_its_area_still_means_the_same_pixels():
    from app.engine.isolation.boundary_mask import resolve_keep_mask
    from app.engine.series import _artwork_frame, _in_frame
    from app.models.isolation import NormalizedPolygon

    shape = (1500, 1000)
    geometry = IsolationGeometry(
        boundary=NormalizedRect(0.3, 0.1, 0.66, 0.45),
        erase_regions=[NormalizedPolygon([(0.4, 0.2), (0.5, 0.25)])],
        erase_brush_radius_norm=0.01,
    )
    frame = _artwork_frame(shape, geometry)
    assert frame is not None
    x0, y0, x1, y1 = frame
    assert x0 < 300 and y0 < 150 and x1 > 660 and y1 > 675  # the area, with a margin...
    assert (x1 - x0) * (y1 - y0) < 0.5 * shape[0] * shape[1]  # ...not the page
    whole = resolve_keep_mask(shape, geometry)[y0:y1, x0:x1]
    cut = resolve_keep_mask((y1 - y0, x1 - x0), _in_frame(geometry, shape, frame))
    assert np.mean(whole != cut) < 0.002  # the same pixels kept, the same erased (to within an edge pixel)


def test_an_artwork_that_is_most_of_its_page_is_compared_as_it_is():
    from app.engine.series import _artwork_frame

    assert _artwork_frame((1000, 800), IsolationGeometry(boundary=NormalizedRect(0.02, 0.02, 0.98, 0.98))) is None
    assert _artwork_frame((1000, 800), None) is None


# ---- one finding per difference of the sheet; each drawn where it is on the sheet ---------------------------------------


def _finding(repeat_x: float, original_box=(10, 10, 40, 30), kind=None):
    from app.models.diff import BoundingBox, DiffCandidate, DiffCluster, DiffSubtype, DiffType

    kind = kind or DiffType.GRAPHICS
    member = DiffCandidate(BoundingBox(*original_box), kind, DiffSubtype.REMOVED, 1.0, "Graphic missing from suspect")
    return DiffCluster(0, BoundingBox(*original_box), BoundingBox(repeat_x, 10, repeat_x + 30, 30), kind, [member], detail=member.detail)


def test_a_difference_in_every_repeat_is_one_finding_marked_on_each():
    from app.engine.series import _gathered

    found = [(n, _finding(100 * n)) for n in range(1, 6)]  # the same place of the original, all five repeats
    found.append((3, _finding(300, original_box=(200, 200, 230, 230))))  # a misprint in repeat 3 alone
    gathered = _gathered(found, total=5, compared=5)
    assert len(gathered) == 2
    common = next(c for c in gathered if c.also_in_suspect)
    assert common.detail == "All 5 repeats: Graphic missing from suspect"
    assert [b.x0 for b in common.suspect_boxes] == [100, 200, 300, 400, 500]
    alone = next(c for c in gathered if not c.also_in_suspect)
    assert alone.detail == "Repeat 3 of 5: Graphic missing from suspect"


def test_a_difference_in_only_some_repeats_stays_one_finding_per_repeat():
    from app.engine.series import _gathered

    found = [(n, _finding(100 * n)) for n in (1, 2)]  # 2 of 6: prints going wrong, not the sheet
    gathered = _gathered(found, total=6, compared=6)
    assert [c.detail for c in gathered] == ["Repeat 1 of 6: Graphic missing from suspect", "Repeat 2 of 6: Graphic missing from suspect"]


def test_the_suspect_pane_draws_a_series_finding_on_the_sheet_not_at_the_originals_place(qapp):
    import numpy as np

    from app.engine.series import _gathered
    from app.ui.comparison_viewport import ComparisonViewport

    common = _gathered([(n, _finding(100 * n)) for n in range(1, 4)], total=3, compared=3)[0]
    common.id = 1
    viewport = ComparisonViewport()
    viewport.set_images(np.full((100, 100, 3), 255, np.uint8), np.full((60, 400, 3), 255, np.uint8))
    viewport.set_composite_modes_enabled(False)  # two different pictures, as Series gives
    viewport.set_marker_clusters([common])
    in_original = viewport.original_pane._marker_items[1].rect()
    assert (in_original.x(), in_original.y()) == (10, 10)
    in_sheet = [viewport.suspect_pane._marker_items[1].rect().x()] + [
        item.rect().x() for item in viewport.suspect_pane._extra_marker_items
    ]
    assert sorted(in_sheet) == [100, 200, 300]  # each repeat's own place on the sheet
    assert not viewport.scroll_lock  # the two panes are not one frame
