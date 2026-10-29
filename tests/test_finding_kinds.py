"""A finding says what actually changed: the font's colour, weight, size or typeface, text or a graphic that moved
(`text_diff._set_differently`, `worker._mark_moves`). Real fonts are drawn with PIL, so the words are read as they are in
artwork; the tests are skipped where the fonts or Tesseract are not there."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pytesseract

from app.config import AppConfig
from app.engine.ocr_locator import find_tesseract
from app.engine.worker import run_comparison
from app.models.diff import DiffSubtype
from app.models.job import ComparisonMode

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

FONTS = Path(r"C:\Windows\Fonts")
SENTENCE = "Read the enclosed leaflet carefully before use."
PAPER = (245, 245, 240)

pytestmark = pytest.mark.skipif(
    not (FONTS / "arial.ttf").is_file() or find_tesseract() is None, reason="needs Arial and Tesseract"
)


@pytest.fixture(autouse=True)
def _tesseract(monkeypatch):
    monkeypatch.setattr(pytesseract.pytesseract, "tesseract_cmd", str(find_tesseract()))


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / name), size)


def _page(font_file="arial.ttf", size=34, colour=(30, 30, 30), dx=0, dy=0, text=SENTENCE, icon_at=(900, 700), icon=True) -> np.ndarray:
    """A page of small print (which keeps the two files aligned whatever is changed), one sentence to change, and an icon."""
    image = Image.new("RGB", (1600, 1200), PAPER)
    draw = ImageDraw.Draw(image)
    draw.text((100, 100), "PARACETAMOL TABLETS 500 mg", font=_font("arialbd.ttf", 52), fill=(20, 20, 20))
    for row in range(10):
        draw.text((100, 200 + row * 26), f"Batch no. {row + 11} Mfd. 08/2026 Exp. 07/2028 keep out of reach of children", font=_font("arial.ttf", 18), fill=(60, 60, 60))
    draw.rectangle((60, 60, 1540, 1140), outline=(90, 90, 90), width=3)
    draw.text((100 + dx, 520 + dy), text, font=_font(font_file, size), fill=colour)
    draw.text((100, 620), "Store below 25 C in a dry place.", font=_font("arial.ttf", 30), fill=(40, 40, 40))
    draw.ellipse((220, 940, 380, 1100), fill=(40, 90, 200))
    draw.rectangle((500, 950, 700, 1080), fill=(30, 150, 60))
    if icon:
        x, y = icon_at
        draw.rectangle((x, y, x + 120, y + 90), fill=(200, 60, 30))
        draw.ellipse((x + 35, y + 20, x + 85, y + 70), fill=(255, 255, 255))
        draw.line((x + 10, y + 80, x + 110, y + 10), fill=(20, 20, 120), width=5)
    return np.array(image)


def _compare(suspect: np.ndarray, original: np.ndarray | None = None):
    return run_comparison(_page() if original is None else original, suspect, ComparisonMode.AUTOMATIC, AppConfig()).clusters


def _only(clusters):
    assert len(clusters) == 1, [(c.category, c.detail) for c in clusters]
    return clusters[0]


def test_identical_pages_have_no_findings():
    assert _compare(_page()) == []


def test_a_recoloured_sentence_is_a_text_finding_that_says_font_colour():
    cluster = _only(_compare(_page(colour=(200, 30, 30))))
    assert cluster.category == "text"
    assert "font colour changed" in cluster.detail and cluster.subtype is DiffSubtype.CHANGED


def test_small_type_that_is_recoloured_says_font_colour_too():
    """Thin strokes are all edge: only their hue can be compared, and a change of it is a change of colour."""
    cluster = _only(_compare(_page(size=16, colour=(200, 30, 30)), _page(size=16)))
    assert cluster.category == "text" and "font colour changed" in cluster.detail


def test_type_that_is_only_softer_is_not_a_different_colour():
    """A blurred render of the same black type is greyer: that is not a change of the type's colour."""
    import cv2

    original = _page(size=16)
    for cluster in _compare(cv2.GaussianBlur(original, (0, 0), 1.0), original):
        assert "colour" not in cluster.detail


def test_a_bolder_sentence_says_font_weight():
    cluster = _only(_compare(_page(font_file="arialbd.ttf")))
    assert cluster.category == "text"
    assert "font weight changed" in cluster.detail and "bolder" in cluster.detail


def test_a_bigger_sentence_says_font_size():
    cluster = _only(_compare(_page(size=38)))
    assert cluster.category == "text"
    assert "font size changed" in cluster.detail and "bigger" in cluster.detail
    assert "weight" not in cluster.detail  # strokes grow with the type: that is not a heavier weight


def test_a_smaller_sentence_says_font_size_and_not_moved():
    cluster = _only(_compare(_page(size=30)))
    assert "font size changed" in cluster.detail and "smaller" in cluster.detail
    assert cluster.subtype is not DiffSubtype.MOVED


def test_another_typeface_says_font_changed():
    cluster = _only(_compare(_page(font_file="times.ttf", size=36)))
    assert cluster.category == "text"
    assert "font changed" in cluster.detail


def test_a_sentence_set_a_little_to_the_right_is_text_moved():
    cluster = _only(_compare(_page(dx=20)))
    assert cluster.category == "text" and cluster.subtype is DiffSubtype.MOVED
    assert "moved 20 px right" in cluster.detail


def test_a_sentence_set_lower_is_text_moved_down():
    cluster = _only(_compare(_page(dy=25)))
    assert cluster.subtype is DiffSubtype.MOVED and "25 px down" in cluster.detail


def test_a_sentence_moved_far_is_one_finding_shown_where_it_was_and_where_it_is():
    cluster = _only(_compare(_page(dy=180)))
    assert cluster.category == "text" and cluster.subtype is DiffSubtype.MOVED
    assert "moved 180 px down" in cluster.detail
    assert cluster.bbox_in_original.y0 < 540 and cluster.bbox_in_suspect.y0 > 690  # boxes at each file's own place


def test_a_changed_word_is_still_a_changed_word():
    cluster = _only(_compare(_page(text="Read the enclosed leaflet carefully after use.")))
    assert cluster.category == "text" and "before" in cluster.detail and "after" in cluster.detail


def test_an_icon_set_a_little_aside_is_one_graphic_moved_not_a_strip_here_and_a_strip_there():
    cluster = _only(_compare(_page(icon_at=(915, 705))))
    assert cluster.category == "graphics" and cluster.subtype is DiffSubtype.MOVED
    assert cluster.detail.startswith("Graphic moved 15 px right, 5 px down")


def test_an_icon_moved_far_is_one_finding_shown_where_it_was_and_where_it_is():
    cluster = _only(_compare(_page(icon_at=(900, 900))))
    assert cluster.category == "graphics" and cluster.subtype is DiffSubtype.MOVED
    assert cluster.detail == "Graphic moved 200 px down"
    assert cluster.bbox_in_original.y1 < 800 and cluster.bbox_in_suspect.y0 > 890


def test_an_icon_taken_away_is_removed_not_moved():
    cluster = _only(_compare(_page(icon=False)))
    assert cluster.subtype is DiffSubtype.REMOVED
