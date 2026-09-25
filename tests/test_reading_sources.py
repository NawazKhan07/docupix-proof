"""What Scan vs Digital reads its words from, text on its side, and a scan a half pixel off its digital file.

Reading was always from the ink picture (background removed, ink black). Other pictures (the file's own grey, grey with
the paper divided out, ...) can be chosen for each file (`scan_reading_digital`, `scan_reading_scan`), and lines that
differ can be read from a second picture as well. A serial number that runs up a flap is read turned on its side, and a
scan that sits a half pixel off (the median of an even number of offsets) must not be able to crash the ink check.
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytesseract
import pytest

from app.config import AppConfig, DetectionConfig
from app.engine.detectors import text_diff
from app.engine.detectors.text_diff import detect_scan, ink_agreement, reading_picture
from app.engine.ocr_locator import find_tesseract
from app.engine.preprocessing.scan_normalizer import READING_SOURCES, ink_image, reading_source
from app.engine.worker import run_comparison
from app.models.diff import BoundingBox, DiffSubtype, DiffType
from app.models.job import ComparisonMode

CONFIG = DetectionConfig()

tesseract_required = pytest.mark.skipif(find_tesseract() is None, reason="Tesseract OCR binary is not installed")


@pytest.fixture
def real_ocr(monkeypatch):
    monkeypatch.setattr(pytesseract.pytesseract, "tesseract_cmd", str(find_tesseract()))


def _page(text="Store below 25 C", colour=(20, 20, 20), background=(250, 246, 238), size=(900, 260)) -> np.ndarray:
    page = np.empty((size[1], size[0], 3), np.uint8)
    page[:] = background
    cv2.putText(page, text, (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 2.0, colour, 4, cv2.LINE_AA)
    return page


# ---- the picture a file is read from -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("source", READING_SOURCES)
def test_every_reading_source_is_a_grey_picture_of_the_same_size_with_light_paper_and_dark_ink(source):
    page = _page()
    picture = reading_source(page, CONFIG, None, source)
    assert picture.shape == page.shape[:2] and picture.dtype == np.uint8
    assert np.median(picture) > 200  # the paper
    assert picture[100:150, 30:400].min() < 100  # the lettering


@pytest.mark.parametrize("source", READING_SOURCES)
def test_what_lies_outside_the_compared_area_reads_as_blank_paper(source):
    page = _page(background=(120, 110, 100))  # a dark scan
    valid = np.zeros(page.shape[:2], bool)
    valid[:, :450] = True
    picture = reading_source(page, CONFIG, valid, source)
    assert (picture[:, 460:] == 255).all()


def test_the_ink_source_is_the_ink_picture_and_an_unknown_one_is_refused():
    page = _page()
    assert np.array_equal(reading_source(page, CONFIG, None, "ink"), ink_image(page, CONFIG, None))
    with pytest.raises(ValueError):
        reading_source(page, CONFIG, None, "sepia")


def test_a_dark_scan_has_its_paper_divided_out_except_where_the_source_is_the_files_own_grey():
    dark = _page(background=(120, 116, 108), colour=(15, 15, 15))
    assert np.median(reading_source(dark, CONFIG, None, "gray")) < 150
    for source in ("flat", "darkest", "ink"):
        assert np.median(reading_source(dark, CONFIG, None, source)) > 220, source


# ---- a scan half a pixel off its digital file ------------------------------------------------------------------------------------------


def test_the_ink_check_does_not_fail_for_an_offset_of_half_a_pixel_and_a_box_of_odd_size():
    """The offset is the middle of the offsets of the words that matched, which is a half pixel for an even number of
    them; rounding the two edges of the window separately made the windows differ by one pixel in size."""
    ink = np.full((300, 500), 255, np.uint8)
    cv2.putText(ink, "Store", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3, cv2.LINE_AA)
    box = BoundingBox(91, 101, 170, 165)  # 79 x 64
    for offset in ((0.5, 0.5), (10.5, -3.5), (1.5, 2.5), (-0.5, 0.5)):
        agreement = ink_agreement(ink, ink.copy(), box, offset)
        assert 0.0 <= agreement <= 1.0


# ---- lines that differ are read from a second picture too ---------------------------------------------------------------------------------


@tesseract_required
def test_identical_lines_give_no_finding_whichever_picture_is_read(real_ocr):
    page = _page()
    ink = ink_image(page, CONFIG, None)
    gray = reading_source(page, CONFIG, None, "gray")
    result = detect_scan(ink, ink.copy(), CONFIG, reading=(gray, gray.copy()), alternate=(gray, gray.copy()))
    assert result.candidates == []


@tesseract_required
def test_a_scan_compared_from_its_own_grey_finds_a_changed_number_like_the_ink_picture_does(real_ocr):
    digital, scan = _page("Take 500 mg daily", background=(255, 255, 255)), _page("Take 250 mg daily")
    for source in ("ink", "gray", "flat"):
        config = replace(CONFIG, scan_reading_digital=source, scan_reading_scan=source)
        outcome = run_comparison(digital, scan, ComparisonMode.SCAN_VS_DIGITAL, AppConfig(detection=config))
        assert any("250" in c.detail or "500" in c.detail for c in outcome.clusters), source


# ---- text turned on its side -----------------------------------------------------------------------------------------------------------------


_SERIAL_SCALE, _SERIAL_THICKNESS = 0.55, 1  # a serial number down a flap is small: ten pixels a letter


def _vertical_serial(serial: str) -> tuple[np.ndarray, BoundingBox]:
    """A sheet with a serial number running UP a flap, and the box a comparison would put round its last digit."""
    page = np.full((600, 500, 3), 255, np.uint8)
    strip = np.full((30, 260, 3), 255, np.uint8)
    cv2.putText(strip, serial, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, _SERIAL_SCALE, (0, 0, 0), _SERIAL_THICKNESS, cv2.LINE_AA)
    page[200:460, 300:330] = cv2.rotate(strip, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 260 high, 30 wide
    (before, _), _ = cv2.getTextSize(serial[:-1], cv2.FONT_HERSHEY_SIMPLEX, _SERIAL_SCALE, _SERIAL_THICKNESS)
    (through, _), _ = cv2.getTextSize(serial, cv2.FONT_HERSHEY_SIMPLEX, _SERIAL_SCALE, _SERIAL_THICKNESS)
    # (strip x, y) becomes page (300 + y, 459 - x): the text starts at the bottom and runs up
    top, bottom = 459 - (6 + through) - 1, 459 - (6 + before) + 1
    return page, BoundingBox(300 + 4, top, 300 + 24, bottom)


@tesseract_required
def test_a_digit_changed_in_text_running_up_the_page_is_described(real_ocr):
    original, box = _vertical_serial("EMJ3TLBE204300")
    suspect, _ = _vertical_serial("EMJ3TLBE204301")
    config = replace(CONFIG, text_describe_min_confidence=60.0)  # (a stroke font reads less surely than a typeface)
    (found,) = text_diff.describe([box], original, suspect, config)
    assert found is not None, "the text is on its side, and reading it upright gives nothing"
    assert found.type in (DiffType.TEXT, DiffType.NUMBER) and found.subtype is DiffSubtype.CHANGED
    assert "204300" in found.detail and "204301" in found.detail


@tesseract_required
def test_a_finding_over_upright_words_that_read_the_same_is_still_no_text_change_when_turned_too(real_ocr):
    page = _page("Store below 25 C", background=(255, 255, 255))
    assert text_diff.describe([BoundingBox(30, 90, 300, 150)], page, cv2.GaussianBlur(page, (0, 0), 1.2), CONFIG) == [None]
