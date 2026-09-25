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
