"""Reading a scan quickly and comparing it with its digital file by what it says and where.

Scan vs Digital took 19-25 s on real pairs, nearly all of it reading: every line of the sheet was read again, twice,
one Tesseract call each. Now the whole page is read once per file, in strips side by side, the words are matched by
what they say and where they are, and only the lines that do not match are read again, stacked into one picture.
The lettering reversed out of dark bars is read the right way round, and a scan that sits a few tens of pixels
from its digital file is compared where it actually is.
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytesseract
import pytest

from app.config import DetectionConfig
from app.engine.detectors import text_diff
from app.engine.detectors.text_diff import (
    OcrWord,
    _cut_rows,
    _findings,
    _fewest_differences,
    _local_offset,
    _marked_if_misread,
    _stroked,
    _unmatched,
    detect_scan,
    ink_agreement,
    read_pages,
    read_windows,
    reading_picture,
)
from app.engine.detectors.text_diff import LIKELY_MISREAD_CONFIDENCE
from app.engine.ocr_locator import find_tesseract
from app.engine.preprocessing.scan_normalizer import ink_image
from app.models.diff import BoundingBox, DiffCandidate, DiffSubtype, DiffType

CONFIG = DetectionConfig()

tesseract_required = pytest.mark.skipif(find_tesseract() is None, reason="Tesseract OCR binary is not installed")


@pytest.fixture
def real_ocr(monkeypatch):
    monkeypatch.setattr(pytesseract.pytesseract, "tesseract_cmd", str(find_tesseract()))


def _word(text, x, y=100, confidence=90.0, width=None, height=30) -> OcrWord:
    width = width or 14 * len(text)
    return OcrWord(text, BoundingBox(x, y, x + width, y + height), confidence)


def _ink(height=400, width=600) -> np.ndarray:
    return np.full((height, width), 255, np.uint8)


# ---- lettering reversed out of a bar is read the right way round -------------------------------------------------------------


def test_white_lettering_on_a_black_bar_is_reversed_to_dark_on_light_for_reading():
    ink = _ink()
    cv2.rectangle(ink, (50, 100), (550, 220), 0, -1)
    cv2.putText(ink, "AMPHOTRET", (80, 185), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 255, 5, cv2.LINE_AA)

    reading = reading_picture(ink)

    assert (reading[110:125, 60:540] > 200).mean() > 0.9  # the bar is now light...
    assert (reading[140:190, 80:520] < 60).any()  # ...and the lettering dark


def test_ordinary_lettering_and_bold_headings_are_left_alone():
    ink = _ink()
    cv2.putText(ink, "Store below 25 C", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3, cv2.LINE_AA)
    cv2.putText(ink, "HEADING", (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 0, 12, cv2.LINE_AA)  # bold, big

    assert np.array_equal(reading_picture(ink), ink)


def test_a_paragraph_is_not_taken_for_a_block():
    ink = _ink(500, 800)
    for row in range(8):
        cv2.putText(ink, "Each tablet contains 500 mg", (30, 50 + row * 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 3, cv2.LINE_AA)
    assert np.array_equal(reading_picture(ink), ink)


def _strokes(width: int) -> np.ndarray:
    """Letter-like strokes (bars and a bowl) drawn `width` pixels wide."""
    page = _ink()
    for x in range(60, 540, 60):
        cv2.line(page, (x, 100), (x, 220), 0, width)
        cv2.line(page, (x, 100), (x + 30, 100), 0, width)
        cv2.circle(page, (x + 20, 170), 25, 0, width)
    return page


def test_hairline_lettering_is_thickened_for_reading_and_ordinary_lettering_is_not():
    thin, fat = _strokes(1), _strokes(5)

    assert (_stroked(thin) < 128).sum() > (thin < 128).sum() * 1.5
    assert np.array_equal(_stroked(fat), fat)


# ---- a page is cut into strips at blank rows -------------------------------------------------------------------------------


def test_a_page_is_cut_through_empty_rows_not_through_a_line_of_text():
    page = _ink(3000, 800)
    for y in range(100, 2900, 120):
        page[y : y + 40, 50:750] = 0  # lines of ink, every 120 px, with 80 blank rows between them
    cuts = _cut_rows(page, 3)

    assert cuts[0] == 0 and cuts[-1] == 3000 and len(cuts) == 4
    for cut in cuts[1:-1]:
        assert not (page[cut] < 128).any()  # a blank row


def test_a_short_page_is_one_strip():
    assert _cut_rows(_ink(500, 500), 1) == [0, 500]


def test_strips_are_read_side_by_side_and_their_words_put_back_in_page_coordinates(monkeypatch):
    calls = []

    def fake(image, page_segmentation=None, lang=None):
        calls.append(image.shape)
        return {"text": ["", "word"], "conf": [-1, 91], "left": [0, 10], "top": [0, 20], "width": [0, 60], "height": [0, 20],
                "block_num": [1, 1], "par_num": [1, 1], "line_num": [1, 1]}

    monkeypatch.setattr(text_diff, "_ocr_data", fake)
    page = _ink(2400, 500)
    first, second = read_pages([page, page.copy()], 40.0)

    assert len(calls) >= 4  # both pages, several strips each
    tops = sorted(round(word.bbox.y0) for _line, word in first)
    assert tops[0] == 20 and tops[-1] > 1000  # a later strip's word is placed lower on the page, not at the top
    assert len({line for line, _word in first}) == len(first)  # every strip's line is its own


# ---- a residual tilt is levelled before the whole page is read, then turned back (`preprocessing.deskew`) --------------------


def _dense_tilted_page(angle: float, rows: int = 60, width: int = 1600, height: int = 1600) -> np.ndarray:
    """Many tightly-packed, near-identical lines (a repeated batch/expiry stamp, worst case for line segmentation)
    turned by `angle` about its own centre -- as a reading picture (255 = paper, dark = ink)."""
    page = np.full((height, width, 3), 255, np.uint8)
    spacing = (height - 160) // rows
    for row in range(rows):
        cv2.putText(
            page, f"BATCH 2109{row:04d} EXP 2027-06 MNB/05/109 Mfd by CIPLA reach protect moisture", (60, 80 + row * spacing),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA,
        )
    grey = cv2.cvtColor(page, cv2.COLOR_RGB2GRAY)
    if angle:
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        grey = cv2.warpAffine(grey, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=255)
    return grey


@tesseract_required
def test_a_severely_tilted_dense_page_still_reads_every_line_once_levelled(real_ocr):
    """Left as it is, a residual tilt on packed, near-identical lines makes Tesseract's OWN line segmentation lose
    its footing (lines merge or drop words); levelled first, it reads the page cleanly."""
    tilted = _dense_tilted_page(6.0)

    (untouched,) = read_pages([tilted], CONFIG.ocr_min_confidence, CONFIG.ocr_language)
    lines_untouched = {line for line, _word in untouched}

    from app.engine.preprocessing import deskew

    level = deskew.measured(tilted, CONFIG)
    (raw,) = read_pages([level.picture(tilted)], CONFIG.ocr_min_confidence, CONFIG.ocr_language)
    levelled = [(line, OcrWord(word.text, level.unrotate_box(word.bbox), word.confidence)) for line, word in raw]
    lines_levelled = {line for line, _word in levelled}

    assert len(lines_levelled) == 60  # every one of the 60 lines is its own, not merged with a neighbour
    assert len(lines_levelled) > len(lines_untouched)
    assert len(levelled) > len(untouched)  # words lost to a merged/broken line are recovered


@tesseract_required
def test_deskewing_does_not_change_a_reading_that_was_already_level(real_ocr):
    """The common case -- a scan already close enough to square -- must not be perturbed by this at all."""
    level_page = _dense_tilted_page(0.0)

    (with_deskew,) = read_pages([level_page], CONFIG.ocr_min_confidence, CONFIG.ocr_language)

    from app.engine.preprocessing import deskew

    level = deskew.measured(level_page, CONFIG)
    assert level.is_identity  # nothing measured worth correcting
    (without_deskew,) = read_pages([level.picture(level_page)], CONFIG.ocr_min_confidence, CONFIG.ocr_language)

    assert len(with_deskew) == len(without_deskew)  # identity: reading it again changes nothing


_VARIED_WORDS = (
    "Each", "capsule", "contains", "Indacaterol", "Glycopyrronium", "Store", "below", "protect", "from",
    "moisture", "Keep", "out", "reach", "children", "Mfd", "CIPLA", "Batch", "Expiry", "Rx", "only",
    "inhalation", "powder", "dosage", "physician", "not", "swallow", "MNB", "109", "2027", "500", "mcg",
)


def _varied_page(rows: int = 40, seed: int = 4) -> np.ndarray:
    """Rows of GENUINELY different small print (not one template with a digit changed): the ordinary case, as
    opposed to `_dense_tilted_page`'s worst-case repetition, which stresses word MATCHING as much as reading."""
    rng = np.random.default_rng(seed)
    page = np.full((1600, 1600, 3), 255, np.uint8)
    for row in range(rows):
        words = " ".join(rng.choice(_VARIED_WORDS, size=int(rng.integers(6, 11))))
        cv2.putText(page, words, (60, 80 + row * 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    return page


@tesseract_required
def test_detect_scan_reports_fewer_shown_findings_on_a_tilted_scan_once_levelled(real_ocr):
    """End to end, on the same tilted scan: levelling it before the whole page is read leaves fewer findings shown
    by default than reading it as it is, since fewer real lines are lost to a segmentation failure the levelling
    avoids (what is left over from a genuinely poorer reading is ranked down and hidden by default either way --
    `_marked_if_misread` -- so this compares what stays SHOWN, which is what the reviewer actually sees)."""
    digital_rgb = _varied_page()
    matrix = cv2.getRotationMatrix2D((800, 800), 2.0, 1.0)
    scan_rgb = cv2.warpAffine(digital_rgb, matrix, (1600, 1600), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    digital_ink, scan_ink = ink_image(digital_rgb, CONFIG), ink_image(scan_rgb, CONFIG)

    from dataclasses import replace

    levelled = detect_scan(digital_ink, scan_ink, replace(CONFIG, deskew_enabled=True))
    as_is = detect_scan(digital_ink, scan_ink, replace(CONFIG, deskew_enabled=False))
    shown_levelled = sum(1 for c in levelled.candidates if not c.uncertain)
    shown_as_is = sum(1 for c in as_is.candidates if not c.uncertain)

    assert shown_levelled <= shown_as_is


# ---- words are matched by what they say and where they are ----------------------------------------------------------------------------


def test_words_match_wherever_they_come_in_the_reading_order():
    digital = [_word("PARACETAMOL", 50), _word("500", 250), _word("mg", 320), _word("TABLETS", 380)]
    scan = [_word("TABLETS", 382), _word("mg", 322), _word("500", 252), _word("PARACETAMOL", 52)]  # read in another order

    missing, extra, pairs = _unmatched(digital, scan)

    assert missing == set() and extra == set() and len(pairs) == 4


def test_a_word_that_differs_or_is_absent_is_the_one_left_over():
    digital = [_word("Each", 50), _word("tablet", 130), _word("contains", 240), _word("500", 400), _word("mg", 470)]
    scan = [_word("Each", 50), _word("tablet", 130), _word("contains", 240), _word("250", 400), _word("mg", 470)]

    missing, extra, _pairs = _unmatched(digital, scan)

    assert missing == {3} and extra == {3}


def test_the_same_word_elsewhere_on_the_page_is_not_a_match():
    digital = [_word("Store", 50, y=100)]
    scan = [_word("Store", 50, y=900)]
    missing, extra, _pairs = _unmatched(digital, scan)
    assert missing == {0} and extra == {0}


def test_a_word_matches_only_once():
    digital = [_word("mg", 50), _word("mg", 60)]
    scan = [_word("mg", 55)]
    missing, _extra, pairs = _unmatched(digital, scan)
    assert len(missing) == 1 and len(pairs) == 1


def test_case_and_the_confusable_letters_still_match():
    missing, extra, _pairs = _unmatched([_word("Ambemath", 50)], [_word("AMBEMATH", 52)])
    assert missing == set() and extra == set()


# ---- a scan that sits off its digital file ---------------------------------------------------------------------------------------------------


def test_how_far_the_scan_is_off_is_judged_from_the_words_that_matched_nearby():
    digital = [_word(t, 100 + 90 * i) for i, t in enumerate(["alpha", "bravo", "charlie", "delta", "echo"])]
    scan = [_word(w.text, w.bbox.x0 + 45, w.bbox.y0 - 8) for w in digital]  # the scan is 45 px right and 8 up here
    _missing, _extra, pairs = _unmatched(digital, scan)

    offset = _local_offset(pairs, digital, scan, BoundingBox(100, 100, 500, 130))

    assert offset == pytest.approx((45.0, -8.0))


def test_with_too_few_matches_to_judge_the_scan_is_taken_to_be_where_it_should_be():
    digital = [_word("alpha", 100), _word("bravo", 200)]
    scan = [_word(w.text, w.bbox.x0 + 45, w.bbox.y0) for w in digital]
    _missing, _extra, pairs = _unmatched(digital, scan)
    assert _local_offset(pairs, digital, scan, BoundingBox(100, 100, 300, 130)) == (0.0, 0.0)


def test_ink_is_compared_where_the_scan_actually_is():
    digital = _ink()
    cv2.putText(digital, "Store", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3, cv2.LINE_AA)
    scan = np.roll(digital, (12, 40), axis=(0, 1))  # the same word, 40 px right and 12 down
    box = BoundingBox(90, 150, 260, 220)

    assert ink_agreement(digital, scan, box) < 0.5
    assert ink_agreement(digital, scan, box, offset=(40, 12)) > 0.95


def test_the_same_word_in_fatter_strokes_still_agrees():
    thin = _ink()
    cv2.putText(thin, "Cilnidipine", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 1, cv2.LINE_AA)
    fat = 255 - cv2.dilate(255 - thin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))  # dot gain: the same strokes, wider
    assert ink_agreement(thin, fat, BoundingBox(50, 150, 400, 230), slack_px=3) > 0.95


# ---- differences that are the reading's ----------------------------------------------------------------------------------------------------


def test_a_shaky_reading_of_the_digital_file_beside_a_sure_one_of_the_scan_is_a_likely_misread():
    (found,) = _findings([_word("Cilnidipene", 100, confidence=67.0)], [_word("Cilnidipine", 100, confidence=87.0)])
    assert found.confidence == LIKELY_MISREAD_CONFIDENCE and "likely a misread" in found.detail


def test_a_word_cut_short_in_the_digital_file_is_a_change_however_shakily_it_was_read():
    """'Cilnidip' for 'Cilnidipine' is the digital file really ending the word there (the reported Met XL change): a
    reader misreads letters, it does not drop the end of a word it can see."""
    (found,) = _findings([_word("Cilnidip", 100, confidence=67.0)], [_word("Cilnidipine", 100, confidence=87.0)])
    assert found.confidence > LIKELY_MISREAD_CONFIDENCE and "likely a misread" not in found.detail


def test_a_surely_read_digital_word_that_the_scan_reads_differently_is_still_a_change():
    (found,) = _findings([_word("500", 100, confidence=96.0)], [_word("250", 100, confidence=90.0)])
    assert found.confidence > LIKELY_MISREAD_CONFIDENCE and "500" in found.detail and "250" in found.detail


def _page_with(word_at, size=(200, 500)) -> np.ndarray:
    page = np.full(size, 255, np.uint8)
    if word_at:
        cv2.putText(page, "Tablets", word_at, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3, cv2.LINE_AA)
    return page


def test_a_word_said_to_be_new_where_the_digital_file_has_ink_is_ranked_down():
    added = DiffCandidate(BoundingBox(60, 80, 260, 130), DiffType.TEXT, DiffSubtype.ADDED, 0.9, "'Tablets' added in suspect")
    kept = _marked_if_misread(added, _page_with((60, 120)), _page_with((60, 120)), CONFIG)
    assert kept.confidence == LIKELY_MISREAD_CONFIDENCE and "not read" in kept.detail


def test_a_word_really_new_where_the_digital_file_has_nothing_stays_serious():
    added = DiffCandidate(BoundingBox(60, 80, 260, 130), DiffType.TEXT, DiffSubtype.ADDED, 0.9, "'Tablets' added in suspect")
    assert _marked_if_misread(added, _page_with(None), _page_with((60, 120)), CONFIG) is added


def test_a_word_really_gone_from_the_scan_stays_serious_and_one_with_ink_still_there_is_ranked_down():
    gone = DiffCandidate(BoundingBox(60, 80, 260, 130), DiffType.TEXT, DiffSubtype.REMOVED, 0.9, "'Tablets' missing from suspect")
    assert _marked_if_misread(gone, _page_with((60, 120)), _page_with(None), CONFIG) is gone
    assert _marked_if_misread(gone, _page_with((60, 120)), _page_with((60, 120)), CONFIG).confidence == LIKELY_MISREAD_CONFIDENCE


def test_any_reading_of_a_line_that_agrees_makes_it_no_difference():
    """A window that misses a word the page reading found is not a change; a real change differs in every reading."""
    digital = [[_word("Release,", 100), _word("Cilnidipine", 200)][:1], [_word("Release,", 100), _word("Cilnidipine", 200)]]
    scan = [[_word("Release,", 100), _word("Cilnidipine", 200)]]
    assert _fewest_differences(digital, scan) == []

    real = [[_word("Release,", 100), _word("Cilnidipene", 200)]]
    assert _fewest_differences(real, scan) != []


# ---- windows are read together, each on its own ------------------------------------------------------------------------------------------------


def test_each_window_is_read_on_its_own_in_one_batch_and_its_words_go_back_to_their_own_line(monkeypatch):
    """Each line's window is its own picture (how one line reads must not depend on which others are read with it),
    all of them handed to the reader in one batch; each word goes back to its own line of the page."""
    batches = []

    def fake(pictures, page_segmentation=None, lang=None):
        batches.append(len(pictures))
        found = []
        for number, image in enumerate(pictures):
            rows = np.flatnonzero((image < 128).any(axis=1))
            found.append({
                "text": [f"line{number}"], "conf": [90], "left": [30], "top": [int(rows[0])], "width": [60],
                "height": [int(rows[-1] - rows[0] + 1)], "block_num": [1], "par_num": [1], "line_num": [1],
            })
        return found

    monkeypatch.setattr(text_diff, "_ocr_many", fake)
    picture = _ink(600, 800)
    for y in (100, 250, 400):
        picture[y : y + 30, 60:200] = 0
    boxes = [BoundingBox(60, y, 200, y + 30) for y in (100, 250, 400)]

    readings = read_windows(picture, boxes, CONFIG)

    assert batches == [3]  # one batch of three pictures, not one stacked picture
    assert [words[0].text for words in readings] == ["line0", "line1", "line2"]
    assert [round(words[0].bbox.y0 / 50) for words in readings] == [2, 5, 8]  # each is at its own line of the page


def test_a_lone_number_is_compared_but_a_lone_speck_is_not():
    """A line of one short word is usually a speck read as letters -- but '10' of '10 g', whose unit was read too
    unsurely to count, is a quantity, and a change to it ('10 g' -> '100 g') must be found."""
    from app.engine.detectors.text_diff import _lone_number

    assert _lone_number("10") and _lone_number("500") and _lone_number("24.")
    assert not _lone_number("5") and not _lone_number("ab") and not _lone_number("B2")
