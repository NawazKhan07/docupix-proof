"""Comparing the words of two digital pages where they are printed (`document_text`)."""

from __future__ import annotations

import cv2
import numpy as np

from app.config import AppConfig
from app.engine.detectors import document_text
from app.engine.detectors.text_diff import OcrWord
from app.models.diff import BoundingBox, DiffSubtype

CONFIG = AppConfig().detection
_FONT, _SCALE, _THICK = cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2


def _page(lines: list[list[str]], shift: tuple[int, int] = (0, 0), size=(700, 1400)) -> tuple[np.ndarray, list[OcrWord]]:
    """A page of lines of words, and its own words (as a PDF's text layer gives them) where they are printed."""
    image = np.full((*size, 3), 255, np.uint8)
    words = []
    for row, line in enumerate(lines):
        x, baseline = 60 + shift[0], 100 + row * 70 + shift[1]
        for text in line:
            (width, height), below = cv2.getTextSize(text, _FONT, _SCALE, _THICK)
            cv2.putText(image, text, (x, baseline), _FONT, _SCALE, (20, 20, 20), _THICK, cv2.LINE_AA)
            words.append(OcrWord(text, BoundingBox(x, baseline - height - 4, x + width, baseline + below + 2), 99.0, True))
            x += width + 24
    return image, words


def _compare(original, suspect, **kwargs):
    (a, words_a), (b, words_b) = original, suspect
    return document_text.compare_words(words_a, words_b, a, b, CONFIG, words_a, words_b, **kwargs)


def test_the_same_page_has_no_changes():
    page = _page([["Take", "one", "tablet", "daily"], ["Store", "below", "25", "C"]])
    assert _compare(page, page) == []


def test_a_changed_number_is_one_change_saying_what_it_was_and_is():
    original = _page([["Take", "500", "mg", "daily"]])
    suspect = _page([["Take", "250", "mg", "daily"]])

    (change,) = _compare(original, suspect)

    assert change.subtype is DiffSubtype.CHANGED
    assert "'500' -> '250'" in change.detail


def test_a_word_that_reads_differently_over_the_same_print_is_no_change():
    """A reading can be wrong: where the one file's letters are printed the same in the other, nothing changed."""
    image, words = _page([["Information", "leaflet"]])
    misread = [OcrWord("lnformation" if w.text == "Information" else w.text, w.bbox, 70.0, True) for w in words]
    assert document_text.compare_words(words, misread, image, image.copy(), CONFIG, words, None) == []


def test_a_line_that_moved_a_little_is_the_same_words():
    original = _page([["Keep", "out", "of", "reach"]])
    suspect = _page([["Keep", "out", "of", "reach"]], shift=(5, 3))
    assert _compare(original, suspect) == []


def test_a_word_gone_is_missing_and_a_word_new_is_added():
    original = _page([["Do", "not", "freeze"], ["Shake", "well"]])
    suspect = _page([["Do", "freeze"], ["Shake", "well", "first"]])

    details = sorted(change.detail for change in _compare(original, suspect))

    assert any("'not'" in d for d in details)
    assert any("'first'" in d and "added" in d for d in details)


def test_one_letter_changed_in_a_long_word_is_not_the_same_print():
    image, words = _page([["Paracetamol"]])
    other, _ = _page([["Paracetamal"]])
    box = words[0].bbox
    assert document_text._found_in(image, box, other, box.height) is False
    assert document_text._found_in(image, box, image.copy(), box.height) is True


def test_text_in_a_script_the_reader_cannot_read_is_compared_by_its_print():
    """Arabic on a page read for English: what a reading made of it is not words, so the lines are compared by ink."""
    image, words = _page([["Take", "daily"], ["XXXX", "YYYY"]])
    arabic = [OcrWord("العلاج" if w.text in ("XXXX", "YYYY") else w.text, w.bbox, 99.0, True) for w in words]
    garbled = [OcrWord("mn" if w.text in ("XXXX", "YYYY") else w.text, w.bbox, 60.0, True) for w in words]

    same = document_text.compare_words(arabic, garbled, image, image.copy(), CONFIG, arabic, None, language="eng")
    assert same == []

    changed_image, _ = _page([["Take", "daily"], ["XXXX", "ZZZZZZ"]])
    changed = document_text.compare_words(arabic, garbled, image, changed_image, CONFIG, arabic, None, language="eng")
    assert changed and all("script" in c.detail for c in changed)


def test_only_where_the_pages_differ_is_read():
    original, _ = _page([["Take", "one", "tablet"], ["Store", "below", "25", "C"]])
    suspect, _ = _page([["Take", "one", "tablet"], ["Store", "below", "30", "C"]])

    windows = document_text.difference_windows(original, suspect, 40.0)

    assert len(windows) == 1
    (window,) = windows
    assert window.y0 > 100  # the second line, not the first
    assert document_text.difference_windows(original, original.copy(), 40.0) == []


def test_the_type_height_is_measured_from_the_print():
    image, words = _page([["Take", "one", "tablet", "daily", "with", "water"]] * 4)
    measured = document_text.type_height(image)
    assert 0.5 * words[0].bbox.height < measured < 2.0 * words[0].bbox.height
