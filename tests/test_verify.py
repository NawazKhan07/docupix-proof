"""Automatic mode finds differences in the pixels; reading only describes them.

On real carton pairs the per-pixel detectors reported hairlines round letters that were the same in both files (13 of
19 'critical colour changes' on one pair: one file drew its type a little softer), and comparing two readings of a
page reported whole areas that had not changed (one file read differently from the other although the pixels were
the same). So each pixel finding is checked by what the two files really have inside it, and the words are read only
from inside a finding's own box. Tests that need the real engine are skipped where Tesseract is not installed.
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytesseract
import pytest

from app.config import AppConfig, DetectionConfig
from app.engine.detectors import text_diff, verify
from app.engine.ocr_locator import find_tesseract
from app.engine.worker import run_comparison
from app.models.diff import BoundingBox, DiffCandidate, DiffSubtype, DiffType
from app.models.job import ComparisonMode

CONFIG = DetectionConfig()

tesseract_required = pytest.mark.skipif(find_tesseract() is None, reason="Tesseract OCR binary is not installed")


@pytest.fixture
def real_ocr(monkeypatch):
    monkeypatch.setattr(pytesseract.pytesseract, "tesseract_cmd", str(find_tesseract()))


def _page(text="Store below 25 C", colour=(20, 20, 20), background=255, size=(900, 220)) -> np.ndarray:
    page = np.full((size[1], size[0], 3), background, np.uint8)
    cv2.putText(page, text, (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 2.0, colour, 4, cv2.LINE_AA)
    return page


def _span(prefix: str, text: str) -> BoundingBox:
    """Where `text` sits on a page drawn by `_page` when it follows `prefix`."""
    (start, _), _ = cv2.getTextSize(prefix, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
    (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
    return BoundingBox(30 + start, 80, 30 + start + width, 140)


def _softer(page: np.ndarray, sigma=1.4) -> np.ndarray:
    return cv2.GaussianBlur(page, (0, 0), sigma)


def _shifted(page: np.ndarray, dx: float, dy: float = 0.0) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(page, matrix, (page.shape[1], page.shape[0]), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def _candidate(box, diff_type=DiffType.COLOUR, magnitude=40.0) -> DiffCandidate:
    detail = f"Colour changed (dE {magnitude:.1f})" if diff_type is DiffType.COLOUR else "Graphic changed"
    return DiffCandidate(BoundingBox(*box), diff_type, DiffSubtype.CHANGED, 0.9, detail, magnitude)


WORD = (30, 80, 300, 140)  # over the first word of the page above


def _kept(candidate, original, suspect, config=CONFIG):
    return verify.drop_residue([candidate], original, suspect, config)


# ---- the same artwork, drawn a little differently ---------------------------------------------------------------------------


@pytest.mark.parametrize("diff_type", [DiffType.COLOUR, DiffType.GRAPHICS])
def test_a_softer_copy_of_the_same_text_is_not_a_change(diff_type):
    page = _page()
    assert _kept(_candidate((60, 84, 90, 132), diff_type), page, _softer(page)) == []


def test_text_drawn_a_fraction_of_a_pixel_off_is_not_a_change():
    page = _page()
    assert _kept(_candidate(WORD), page, _shifted(page, 0.6, 0.4)) == []


def test_a_shift_is_a_change_where_no_registration_error_is_forgiven():
    page = _page()
    exact = replace(CONFIG, residue_slack_px=0)
    assert _kept(_candidate(WORD), page, _shifted(page, 1.0), exact) != []


def test_text_that_moved_beyond_the_registration_error_is_a_change():
    page = _page()
    assert _kept(_candidate(WORD), page, _shifted(page, 9.0, 3.0)) != []


# ---- a real difference is kept ----------------------------------------------------------------------------------------------------


def test_a_changed_digit_is_a_change():
    a, b = _page("Store below 25 C"), _page("Store below 30 C")
    assert _kept(_candidate((400, 80, 500, 140), DiffType.GRAPHICS), a, b) != []


def test_a_word_removed_is_a_change():
    a, b = _page("Store below 25 C"), _page("Store 25 C")
    assert _kept(_candidate((130, 80, 400, 140), DiffType.GRAPHICS), a, b) != []


def test_text_recoloured_is_a_change_although_its_shape_is_the_same():
    a, b = _page(colour=(20, 20, 20)), _page(colour=(200, 30, 30))
    assert _kept(_candidate(WORD), a, b) != []


def test_a_background_tint_that_changed_is_a_change_even_under_the_same_text():
    a, b = _page(background=245), _page(background=228)
    assert _kept(_candidate(WORD), a, b) != []


def test_a_flat_area_that_changed_colour_is_a_change_and_an_unchanged_one_is_not():
    a = np.full((200, 200, 3), (200, 40, 40), np.uint8)
    b = np.full((200, 200, 3), (40, 40, 200), np.uint8)
    same = _candidate((20, 20, 180, 180))
    assert _kept(same, a, b) != []
    assert _kept(same, a, a.copy()) == []


def test_a_dot_missing_is_a_change():
    a = np.full((120, 120, 3), 255, np.uint8)
    cv2.circle(a, (60, 60), 5, (0, 0, 0), -1)
    assert _kept(_candidate((54, 54, 66, 66), DiffType.GRAPHICS), a, np.full_like(a, 255)) != []


# ---- other kinds of finding, and panels, are not judged here ----------------------------------------------------------------------


def test_a_barcode_finding_is_never_dropped():
    page = _page()
    barcode = DiffCandidate(BoundingBox(*WORD), DiffType.BARCODE, DiffSubtype.CHANGED, 1.0, "different value")
    assert verify.drop_residue([barcode], page, page.copy(), CONFIG) == [barcode]


def test_a_panel_sized_difference_is_not_checked():
    page = np.full((1800, 1800, 3), 255, np.uint8)
    panel = _candidate((0, 0, 1700, 1700), DiffType.GRAPHICS)
    assert verify.drop_residue([panel], page, page.copy(), CONFIG) == [panel]


# ---- how a colour change is rated ---------------------------------------------------------------------------------------------------


def test_a_colour_finding_is_rated_by_how_different_the_area_looks_not_by_its_worst_pixel():
    a, b = _page(background=245), _page(background=228)
    (kept,) = _kept(_candidate(WORD, magnitude=90.0), a, b)
    assert kept.type is DiffType.COLOUR
    assert 3.0 < kept.magnitude < 15.0 and f"{kept.magnitude:.1f}" in kept.detail


def test_a_recoloured_panel_keeps_its_full_delta():
    a = np.full((200, 200, 3), (200, 40, 40), np.uint8)
    b = np.full((200, 200, 3), (40, 40, 200), np.uint8)
    (kept,) = _kept(_candidate((0, 0, 200, 200), magnitude=90.0), a, b)
    assert kept.magnitude > 40.0
    assert kept.type is DiffType.COLOUR


def test_a_shape_that_differs_in_position_only_is_reported_as_a_shape_not_a_colour():
    a = np.full((200, 400, 3), 255, np.uint8)
    b = a.copy()
    cv2.rectangle(a, (50, 60), (90, 140), (0, 0, 0), -1)
    cv2.rectangle(b, (120, 60), (160, 140), (0, 0, 0), -1)
    # a candidate over a stretch that the bar leaves and the other file's bar does not reach: same average colour
    (kept,) = _kept(_candidate((40, 50, 170, 150), magnitude=90.0), a, b)
    assert kept.type is DiffType.GRAPHICS and "colour unchanged" in kept.detail


# ---- what a finding says is read from its own box -------------------------------------------------------------------------------------


@tesseract_required
def test_a_changed_number_is_described_with_its_old_and_new_words(real_ocr):
    a, b = _page("Take 500 mg daily"), _page("Take 250 mg daily")
    (found,) = text_diff.describe([_span("Take ", "500")], a, b, CONFIG)
    assert found.type is DiffType.NUMBER and found.subtype is DiffSubtype.CHANGED
    assert "500" in found.detail and "250" in found.detail
    assert found.type is DiffType.NUMBER


@tesseract_required
def test_text_that_reads_the_same_in_both_is_not_described_as_a_text_change(real_ocr):
    page = _page()
    assert text_diff.describe([BoundingBox(*WORD)], page, _softer(page), CONFIG) == [None]


@tesseract_required
def test_a_finding_over_a_removed_word_names_the_word(real_ocr):
    a, b = _page("Store below room temp"), _page("Store room temp")
    (found,) = text_diff.describe([_span("Store ", "below")], a, b, CONFIG)
    assert "below" in found.detail


def _reading(*words):
    return [text_diff.OcrWord(text, BoundingBox(x, 100, x + 60, 130), 92.0) for text, x in words]


def _bars(*positions, size=(500, 300)) -> np.ndarray:
    """A page with a dark bar (a word's marks) at each x."""
    page = np.full((size[1], size[0], 3), 255, np.uint8)
    for x in positions:
        cv2.rectangle(page, (x, 105), (x + 55, 125), (0, 0, 0), -1)
    return page


def test_a_dropped_word_is_the_one_that_is_missing_not_everything_after_it():
    """The words beside it move along when one goes; only the word that went is the change."""
    original = _reading(("Store", 20), ("below", 100), ("room", 180), ("temp", 260))
    suspect = _reading(("Store", 20), ("room", 100), ("temp", 180))
    found = text_diff._described(BoundingBox(100, 100, 160, 130), original, suspect, _bars(20, 100, 180, 260), _bars(20, 100, 180), CONFIG)
    assert found.subtype is DiffSubtype.REMOVED and found.detail == "'below' missing from suspect"


def test_a_word_added_is_the_one_that_is_new():
    original = _reading(("Store", 20), ("room", 100), ("temp", 180))
    suspect = _reading(("Store", 20), ("cool", 100), ("room", 180), ("temp", 260))
    found = text_diff._described(BoundingBox(100, 100, 160, 130), original, suspect, _bars(20, 100, 180), _bars(20, 100, 180, 260), CONFIG)
    assert found.subtype is DiffSubtype.ADDED and found.detail == "'cool' added in suspect"


def test_a_dropped_digit_reads_as_the_number_it_was():
    """'100g' -> '10 g': the box is only the digit that went; the number is what the reviewer needs to read."""
    original = _reading(("100g", 100))
    suspect = _reading(("10", 100), ("g", 190))
    found = text_diff._described(BoundingBox(150, 100, 172, 130), original, suspect, _bars(100), _bars(100, 190), CONFIG)
    assert found.subtype is DiffSubtype.CHANGED and found.type is DiffType.TEXT
    assert found.detail == "'100g' -> '10 g'"


def test_a_hyphen_that_was_dropped_is_a_change_although_the_word_is_the_same_letters():
    original = _reading(("eco-pack", 100))
    suspect = _reading(("ecopack", 100))
    a, b = _bars(100), _bars(100)
    cv2.rectangle(a, (140, 112), (150, 116), (255, 255, 255), -1)  # (the gap where the hyphen was)
    found = text_diff._described(BoundingBox(138, 100, 152, 130), original, suspect, a, b, CONFIG)
    assert found is not None and "eco-pack" in found.detail and "ecopack" in found.detail


def test_words_far_from_a_finding_are_not_part_of_it():
    original = _reading(("Store", 20), ("below", 700))
    suspect = _reading(("Stone", 20), ("below", 700))
    assert text_diff._described(BoundingBox(690, 100, 770, 130), original, suspect, _bars(20, 700, size=(900, 300)), _bars(20, 700, size=(900, 300)), CONFIG) is None


def test_a_word_read_differently_over_the_same_marks_is_the_readers_and_not_a_change():
    """A word beside a finding, read differently in the two files though its marks are the same, is not part of it."""
    original = _reading(("Store", 20))
    suspect = _reading(("Stone", 20))
    same = _bars(20)
    assert text_diff._described(BoundingBox(100, 100, 130, 130), original, suspect, same, same.copy(), CONFIG) is None


def test_a_word_the_other_file_could_not_be_read_at_all_is_not_missing_if_its_marks_are_still_there():
    """White lettering on red is read in one file and not the other: the words are 'missing' only in the reading."""
    original = _reading(("ecopack", 100))
    assert text_diff._described(BoundingBox(100, 100, 160, 130), original, [], _bars(100), _bars(100), CONFIG) is None
    assert text_diff._described(BoundingBox(100, 100, 160, 130), original, [], _bars(100), _bars(300), CONFIG) is not None


def test_a_word_read_shakily_is_not_claimed_as_a_change():
    shaky = [text_diff.OcrWord("cindll", BoundingBox(100, 100, 160, 130), 70.0)]
    assert text_diff._described(BoundingBox(100, 100, 160, 130), [], shaky, _bars(300), _bars(100), CONFIG) is None


def test_a_finding_over_a_blank_crop_reads_nothing_and_a_panel_is_not_read_at_all(monkeypatch):
    empty = {"text": [], "conf": [], "left": [], "top": [], "width": [], "height": [], "block_num": [], "par_num": [], "line_num": []}
    calls = []
    monkeypatch.setattr(text_diff, "_ocr_data", lambda *_a, **_k: calls.append(1) or empty)
    blank = np.full((300, 300, 3), 255, np.uint8)

    assert text_diff.describe([BoundingBox(0, 0, 300, 300)], blank, blank.copy(), CONFIG) == [None]
    assert calls == []  # nothing printed in either file's window: there is nothing to read

    calls.clear()
    assert text_diff.describe([BoundingBox(0, 0, 3000, 3000)], blank, blank.copy(), CONFIG) == [None]
    assert calls == []


# ---- the whole comparison ---------------------------------------------------------------------------------------------------------------


def test_two_readings_that_disagree_over_identical_pixels_are_no_finding_at_all(monkeypatch):
    """The reported failure: one file's text read differently from the other's, so whole areas were flagged.

    Nothing that reads the pixels differently can make a finding where the pixels are the same."""
    page = _page()
    answers = iter(
        [
            {"text": ["Store", "below", "junk"], "conf": [90, 90, 90], "left": [30, 200, 400], "top": [80, 80, 80],
             "width": [150, 190, 90], "height": [50, 50, 50], "block_num": [1, 1, 1], "par_num": [1, 1, 1], "line_num": [1, 1, 1]},
        ]
        * 50
    )
    monkeypatch.setattr(text_diff, "_ocr_data", lambda *_a, **_k: next(answers))

    outcome = run_comparison(page, _softer(page, 1.0), ComparisonMode.AUTOMATIC, AppConfig())

    assert outcome.clusters == []


@tesseract_required
def test_a_real_text_change_is_found_in_the_pixels_and_described_by_reading(real_ocr):
    a, b = _page("Take 500 mg daily"), _page("Take 250 mg daily")

    outcome = run_comparison(a, b, ComparisonMode.AUTOMATIC, AppConfig())

    assert len(outcome.clusters) == 1
    (cluster,) = outcome.clusters
    assert cluster.type is DiffType.NUMBER and cluster.category == "text"
    assert "500" in cluster.detail and "250" in cluster.detail
    assert cluster.bbox_in_original.area < 0.3 * a.shape[0] * a.shape[1]


@tesseract_required
def test_softer_text_with_a_real_change_beside_it_gives_one_finding_at_the_change(real_ocr):
    a = _page("Take 500 mg daily")
    b = _softer(_page("Take 250 mg daily"), 1.2)

    outcome = run_comparison(a, b, ComparisonMode.AUTOMATIC, AppConfig())

    assert len(outcome.clusters) == 1
    assert outcome.clusters[0].bbox_in_original.x1 < 600  # the digits, not the whole line


def test_a_comparison_without_ocr_still_finds_the_change_and_says_the_words_were_not_read(monkeypatch):
    monkeypatch.setattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract-that-is-not-installed")
    a, b = _page("Take 500 mg daily"), _page("Take 250 mg daily")

    outcome = run_comparison(a, b, ComparisonMode.AUTOMATIC, AppConfig())

    assert outcome.clusters and any("Tesseract" in warning for warning in outcome.warnings)
