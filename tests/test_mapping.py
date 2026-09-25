"""How differences are read and mapped, checked against the failures found on real pairs (how they are grouped into
findings is in `test_clustering.py`).

OCR read rotated text and barcode digits as junk 'changes', and identical pixels were reported as changed words.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.config import AppConfig, DetectionConfig
from app.engine.detectors import braille_diff, structure_diff, text_diff
from app.engine.detectors.text_diff import OcrWord, _findings, _runs, _trustworthy, corroborated, marks_agreement
from app.models.diff import BoundingBox, DiffCandidate, DiffSubtype, DiffType

CONFIG = DetectionConfig()


def _candidate(x0, y0, x1, y1, diff_type=DiffType.GRAPHICS, subtype=DiffSubtype.CHANGED, magnitude=0.0) -> DiffCandidate:
    return DiffCandidate(BoundingBox(x0, y0, x1, y1), diff_type, subtype, 1.0, f"{diff_type.value} at {x0},{y0}", magnitude)


# ---- text: no junk, no phantom changes, no page-sized 'change' --------------------------------------------------------------------------


def _word(text, x, y=0, confidence=90.0) -> OcrWord:
    return OcrWord(text, BoundingBox(x, y, x + 12 * len(text), y + 20), confidence)


def test_a_run_of_far_apart_words_is_separate_phrases():
    words = [_word("alpha", 0), _word("beta", 80), _word("gamma", 2000), _word("delta", 2080), _word("epsilon", 100, y=900)]
    assert [[w.text for w in run] for run in _runs(words)] == [["alpha", "beta"], ["gamma", "delta"], ["epsilon"]]


def test_misaligned_readings_do_not_make_one_change_the_size_of_the_page():
    gone = [_word("alpha", 0, 0), _word("bravo", 3000, 2500)]
    new = [_word("charlie", 1500, 1200)]
    assert all(f.bbox.area < 200 * 40 * 3 for f in _findings(gone, new))


def test_a_real_changed_phrase_is_still_one_change():
    (found,) = _findings([_word("500mg", 0), _word("tablets", 80)], [_word("250mg", 0), _word("tablets", 80)])
    assert found.subtype is DiffSubtype.CHANGED and "500mg" in found.detail and "250mg" in found.detail


def test_words_read_as_marks_are_not_words_in_any_mode():
    junk = [_word("|", 0), _word("!!", 30), _word("e", 60, confidence=80), _word("Store", 90)]
    assert [w.text for w in _trustworthy(junk, CONFIG)] == ["Store"]


def _page_with_text(text, width=700, height=160) -> np.ndarray:
    page = np.full((height, width, 3), 255, np.uint8)
    cv2.putText(page, text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (20, 20, 20), 4, cv2.LINE_AA)
    return page


def test_marks_that_are_the_same_agree_and_different_ones_do_not():
    a, b = _page_with_text("500 mg"), _page_with_text("250 mg")
    box = BoundingBox(0, 0, 700, 160)
    assert marks_agreement(a, a.copy(), box) >= 0.999
    assert marks_agreement(a, b, box) < 0.95
    assert marks_agreement(a, np.full_like(a, 255), box) is None  # a blank crop cannot be judged


def test_a_finding_over_identical_pixels_is_dropped_and_one_over_different_pixels_is_kept():
    same = _page_with_text("500 mg")
    phantom = DiffCandidate(BoundingBox(0, 0, 700, 160), DiffType.TEXT, DiffSubtype.CHANGED, 0.9, "'x' -> 'a'")
    assert corroborated([phantom], same, same.copy(), CONFIG) == []
    assert corroborated([phantom], same, _page_with_text("250 mg"), CONFIG) == [phantom]


def test_a_finding_over_a_blank_crop_is_kept_because_it_cannot_be_checked():
    blank = np.full((160, 700, 3), 255, np.uint8)
    finding = DiffCandidate(BoundingBox(0, 0, 700, 160), DiffType.TEXT, DiffSubtype.REMOVED, 0.9, "'x' missing")
    assert corroborated([finding], blank, blank.copy(), CONFIG) == [finding]


def test_a_moved_word_reports_how_far_it_moved():
    from app.engine.detectors.text_diff import _moved_candidates

    (moved,) = _moved_candidates([_word("Injection", 0)], [_word("Injection", 110)], CONFIG)
    assert moved.subtype is DiffSubtype.MOVED and moved.magnitude == pytest.approx(110.0)


# ---- graphics: a chain of differing ink is not one object -----------------------------------------------------------------------------


def test_differing_ink_scattered_across_the_sheet_is_grouped_into_its_pieces():
    different = np.zeros((1200, 1200), bool)
    for i in range(10):  # ten small marks along a diagonal, 10 px apart edge to edge: one chain at the default reach
        different[100 + i * 100 : 130 + i * 100, 100 + i * 100 : 130 + i * 100] = True
        if i:
            different[100 + i * 100 - 60 : 100 + i * 100 - 40, 100 + i * 100 - 60 : 100 + i * 100 - 40] = True
    config = DetectionConfig(structure_group_px=64, structure_split_min_area_px=100000)

    count, labels = structure_diff._group_ink(different, config)

    objects = {int(v) for v in np.unique(labels[different])}
    assert len(objects) >= 5  # not one object the size of the sheet


def test_a_solid_object_is_not_split_however_big():
    different = np.zeros((1000, 1000), bool)
    different[100:900, 100:900] = True
    count, labels = structure_diff._group_ink(different, DetectionConfig(structure_split_min_area_px=100000))
    assert len({int(v) for v in np.unique(labels[different])}) == 1


def test_nearby_ink_is_still_one_object_when_it_is_small():
    different = np.zeros((300, 300), bool)
    different[50:60, 50:60] = True
    different[50:60, 66:76] = True
    count, labels = structure_diff._group_ink(different, DetectionConfig())
    assert len({int(v) for v in np.unique(labels[different])}) == 1


# ---- Braille: the dots are red on real artwork ----------------------------------------------------------------------------------------------


def _dots(colour, radius=9, centres=((100, 100), (140, 100), (100, 140), (180, 180), (220, 100), (260, 140), (300, 100), (340, 140), (380, 180))):
    page = np.full((300, 500, 3), 255, np.uint8)
    for x, y in centres:
        cv2.circle(page, (x, y), radius, colour, -1)
    return page


@pytest.mark.parametrize(
    "colour,name",
    [((239, 65, 53), "red, as measured on the real cartons"), ((200, 20, 140), "magenta"), ((240, 60, 120), "pink-red")],
)
def test_braille_dots_are_found_in_red_and_in_magenta(colour, name):
    assert len(braille_diff.detect_dots(_dots(colour), CONFIG)) == 9, name


@pytest.mark.parametrize("colour", [(247, 148, 29), (0, 160, 90), (30, 90, 200), (20, 20, 20)], ids=["orange", "green", "blue", "black"])
def test_dots_of_other_colours_are_not_braille(colour):
    assert len(braille_diff.detect_dots(_dots(colour), CONFIG)) == 0


def test_a_red_letter_or_bar_is_not_a_dot():
    page = np.full((300, 500, 3), 255, np.uint8)
    cv2.rectangle(page, (50, 50), (450, 80), (239, 65, 53), -1)  # a red bar
    cv2.putText(page, "Unvarnished space", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (239, 65, 53), 2, cv2.LINE_AA)
    dots = braille_diff.detect_dots(page, CONFIG)
    assert len(dots) <= 2  # the dot of an i at most, never the bar or the letters as such


def test_the_specks_of_red_text_are_not_counted_among_real_dots():
    page = _dots((239, 65, 53))
    for x in range(30, 480, 25):  # tiny round red specks (the dot of an i)
        cv2.circle(page, (x, 250), 3, (239, 65, 53), -1)
    assert len(braille_diff.detect_dots(page, CONFIG)) == 9


def test_braille_mode_compares_red_dots():
    original, suspect = _dots((239, 65, 53)), _dots((239, 65, 53), centres=((100, 100), (140, 100), (100, 140), (180, 180), (220, 100), (260, 140), (300, 100), (340, 140)))
    candidates, comparison = braille_diff.compare(original, suspect, CONFIG)
    assert comparison.original_dot_count == 9 and comparison.missing == 1 and comparison.passed is False
    assert [c.subtype for c in candidates] == [DiffSubtype.REMOVED]
