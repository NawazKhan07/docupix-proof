"""A scan's own faults, and type set close to bars, in Scan vs Digital.

* A scanner leaves STREAKS: narrow bands down the page where the lamp or the sensor gave less, every stroke crossing one
  lighter -- half of a letter missing, read as a change ('d montré' for 'démontré'). They are evened out against the
  digital file, which has none; a stripe printed down the artwork, in both files, is not a streak.
* A heading BAR set a few pixels over its text is still a bar: its lettering is read, and judged, the right way round.
* A word ADDED at the end of a line is found, however close the lines above and below it.
* In the view, the text between two coloured bars is shown as it is (not made part of one panel with them).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytesseract
import pytest

from app.config import DetectionConfig
from app.engine.detectors.text_diff import detect_scan, reading_picture
from app.engine.ocr_locator import find_tesseract
from app.engine.preprocessing import scan_normalizer
from app.engine.preprocessing.scan_normalizer import _ink_contrast, _streaks_evened, ink_pair
from app.models.diff import DiffSubtype

CONFIG = DetectionConfig()

tesseract_required = pytest.mark.skipif(find_tesseract() is None, reason="Tesseract OCR binary is not installed")


@pytest.fixture
def real_ocr(monkeypatch):
    monkeypatch.setattr(pytesseract.pytesseract, "tesseract_cmd", str(find_tesseract()))


def _leaflet(height=1400, width=1000) -> np.ndarray:
    """A page of small print: many lines of text, all the way down."""
    page = np.full((height, width, 3), 255, np.uint8)
    for row, y in enumerate(range(60, height - 40, 34)):
        cv2.putText(page, f"Each tablet contains {row} mg of paracetamol BP", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
    return page


def _streaked(page: np.ndarray, x0: int, x1: int, keep: float) -> np.ndarray:
    """`page` scanned through a streak at columns x0..x1: only `keep` of the ink there (the paper unchanged)."""
    scan = page.copy()
    band = scan[:, x0:x1].astype(np.float32)
    scan[:, x0:x1] = (255.0 - (255.0 - band) * keep).astype(np.uint8)
    return scan


def _ink_share(picture: np.ndarray, x0: int, x1: int) -> float:
    return float((picture[:, x0:x1] < 128).mean())


# ---- a scanner's streaks ----------------------------------------------------------------------------------------------


def test_the_letters_crossing_a_scanners_streak_come_out_whole():
    page = _leaflet()
    scan = _streaked(page, 400, 412, keep=0.3)

    pictures = ink_pair(page, scan, CONFIG)

    # as much ink in the streak as the file has there, and as the scan has beside it
    assert _ink_share(pictures.first, 400, 412) > 0.05
    assert _ink_share(pictures.second, 400, 412) > 0.8 * _ink_share(pictures.first, 400, 412)


def test_without_evening_the_streak_breaks_the_letters(monkeypatch):
    """(The test above is the streak's doing: left as it is, the letters crossing it are broken.)"""
    page = _leaflet()
    scan = _streaked(page, 400, 412, keep=0.3)
    monkeypatch.setattr(scan_normalizer, "_streaks_evened", lambda _scan, _digital: None)

    pictures = ink_pair(page, scan, CONFIG)

    assert _ink_share(pictures.second, 400, 412) < 0.5 * _ink_share(pictures.first, 400, 412)


def test_a_stripe_printed_down_the_artwork_is_not_a_streak():
    """A grey stripe down the artwork is weaker than the black type round it in BOTH files: nothing to even out."""
    page = _leaflet()
    page[:, 600:640] = (200, 200, 200)  # a pale grey stripe, the type running over it
    for y in range(60, page.shape[0] - 40, 34):
        cv2.putText(page, "x", (605, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
    scan = page.copy()
    digital_contrast, _ = _ink_contrast(page, CONFIG, None)
    scan_contrast, _ = _ink_contrast(scan, CONFIG, None)
    before = scan_contrast.copy()

    _streaks_evened(scan_contrast, digital_contrast)

    assert np.array_equal(scan_contrast, before)


def test_a_digital_file_is_not_evened_however_its_columns_differ():
    """Only the scan is evened: the digital file is what it is measured against."""
    page = _leaflet()
    scan = _streaked(page, 400, 412, keep=0.3)
    digital_before, _ = _ink_contrast(page, CONFIG, None)

    pictures = ink_pair(page, scan, CONFIG)
    alone = ink_pair(page, page, CONFIG)

    assert np.array_equal(pictures.first, alone.first)
    assert digital_before.shape == page.shape[:2]


# ---- a bar set close over its text -----------------------------------------------------------------------------------


def _bar_over_text(gap: int) -> np.ndarray:
    """An ink picture: a black heading bar with white lettering, and its paragraph of light type `gap` pixels below it."""
    ink = np.full((560, 900), 255, np.uint8)
    cv2.rectangle(ink, (40, 60), (600, 110), 0, -1)
    cv2.putText(ink, "CATEGORY", (60, 98), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 3, cv2.LINE_AA)
    for row in range(8):
        y = 110 + gap + 22 + row * 30
        cv2.putText(ink, "Anti-inflammatory analgesic combination", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 1, cv2.LINE_AA)
    return ink


@pytest.mark.parametrize("gap", [6, 30])
def test_a_heading_bar_close_over_its_paragraph_is_read_the_right_way_round(gap):
    """The bar used to be found by closing its lettering over; a paragraph a few pixels under it was closed into it too,
    the whole was no bar, and its lettering was never read."""
    ink = _bar_over_text(gap)

    reading = reading_picture(ink)

    assert (reading[64:72, 50:590] > 200).mean() > 0.9  # the bar is light...
    assert (reading[72:100, 60:300] < 60).any()  # ...its lettering dark
    paragraph = (slice(110 + gap + 2, 110 + gap + 92), slice(40, 700))
    assert (reading[paragraph] > 200).mean() > 0.6  # and the paragraph dark on light still (hairlines thickened, not reversed)


# ---- a word added at the end of a line -------------------------------------------------------------------------------


def _lines(texts: list[str]) -> np.ndarray:
    ink = np.full((260, 1000), 255, np.uint8)
    for row, text in enumerate(texts):
        cv2.putText(ink, text, (30, 60 + row * 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2, cv2.LINE_AA)
    return ink


@tesseract_required
def test_a_word_added_at_the_end_of_a_line_is_found_between_close_lines(real_ocr):
    """'STORE BELOW' matched word for word; the scan's '30C.' after it lies within reach of the lines above and below,
    whose ink is not its own."""
    digital = _lines(["Available in strips of 10 tablets", "STORE BELOW", "KEEP OUT OF THE REACH OF CHILDREN"])
    scan = _lines(["Available in strips of 10 tablets", "STORE BELOW 30C.", "KEEP OUT OF THE REACH OF CHILDREN"])

    found = detect_scan(digital, scan, CONFIG).candidates

    added = [c for c in found if c.subtype is DiffSubtype.ADDED and "30" in c.detail]
    assert added, [c.detail for c in found]
    assert added[0].bbox.x0 > 200 and 70 < added[0].bbox.center[1] < 110


@tesseract_required
def test_lines_that_read_the_same_have_nothing_added(real_ocr):
    lines = _lines(["Available in strips of 10 tablets", "STORE BELOW 30C.", "KEEP OUT OF THE REACH OF CHILDREN"])
    assert detect_scan(lines, lines.copy(), CONFIG).candidates == []


# ---- the view: text between two coloured bars --------------------------------------------------------------------------


def _two_bars() -> np.ndarray:
    page = np.full((1600, 1600, 3), 255, np.uint8)
    for top, words in ((200, "CATEGORIE"), (262, "PROPRIETE PHARMACOLOGIQUES")):
        cv2.rectangle(page, (100, top), (900, top + 40), (110, 20, 10)[::-1], -1)  # a navy bar
        cv2.putText(page, words, (120, top + 31), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(page, "Combinaison d'anti-inflammatoire", (100, 256), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
    return page


def test_the_text_between_two_coloured_bars_is_shown_as_it_is_not_as_a_black_band():
    page = _two_bars()

    pictures = ink_pair(page, page.copy(), CONFIG)

    for shown in (pictures.shown_first, pictures.shown_second):
        between = shown[244:259, 100:900]  # the line of text between the bars
        assert (between > 200).mean() > 0.7  # its paper is paper
        assert (between < 100).any()  # its type is there
        assert shown[206:236, 300:880].mean() > 150  # each bar: paper, its type black on it
        assert (shown[206:236, 115:300] < 60).any()


def test_white_type_on_a_coloured_bar_is_shown_solid_not_hollow():
    page = _two_bars()
    letters = (page[200:240, 100:900].min(axis=2) > 200)  # the white type's own pixels

    shown = ink_pair(page, page.copy(), CONFIG).shown_first[200:240, 100:900]

    assert (shown[letters] < 100).mean() > 0.85


def test_white_type_on_a_panel_with_a_few_dark_marks_on_it_is_shown():
    """A green panel with white type and a few dark dots on it (Braille marked in ink, a scan's grain): the dots made it
    a panel of dark type, and its white type was cut away as the panel's own edges -- shown as nothing."""
    page = np.full((900, 1200, 3), 255, np.uint8)
    cv2.rectangle(page, (100, 200), (1100, 600), (40, 170, 60), -1)  # a green panel
    cv2.putText(page, "BLACIDEC 10", (300, 430), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (255, 255, 255), 7, cv2.LINE_AA)
    for y in (480, 520, 560):
        for x in range(130, 1080, 40):
            cv2.circle(page, (x, y), 10, (90, 20, 90), -1)  # dark dots, more of them than of the white type
    letters = page[330:450, 290:820].min(axis=2) > 230  # the white type's own pixels

    pictures = ink_pair(page, page.copy(), CONFIG)

    for shown in (pictures.shown_first, pictures.shown_second):
        assert (shown[330:450, 290:820][letters] < 100).mean() > 0.8


def test_bold_type_in_colour_beside_a_coloured_box_is_shown_as_type_not_a_smear():
    """Bold purple type on paper, an orange box set against its last letter: the type is no panel, and is not grown into
    the box's panel either -- it is shown as it is printed."""
    page = np.full((500, 700, 3), 255, np.uint8)
    cv2.putText(page, "Equirab", (80, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (70, 20, 120), 2, cv2.LINE_AA)  # (strokes ~8 px)
    cv2.rectangle(page, (232, 255), (290, 315), (230, 140, 20), -1)  # an orange box against its last letter
    type_ink = page[260:312, 78:231].min(axis=2) < 150

    pictures = ink_pair(page, page.copy(), CONFIG)

    for shown in (pictures.shown_first, pictures.shown_second):
        crop = shown[260:312, 78:231]
        assert (crop[type_ink] < 100).mean() > 0.85  # the type black...
        assert (crop[~type_ink] > 200).mean() > 0.95  # ...and the paper round it paper, not a smear
