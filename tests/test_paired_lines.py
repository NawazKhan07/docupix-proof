"""A finding's text read line by line, each line set against the suspect's line most like it
(`text_diff._paired_lines`): a block of type re-set, or lines that moved with the panel they are on."""

from __future__ import annotations

import numpy as np

from app.config import AppConfig
from app.engine.detectors.text_diff import _NOT_PAIRED, OcrWord, _paired_lines
from app.models.diff import BoundingBox, DiffSubtype

CONFIG = AppConfig().detection
BLANK = np.full((600, 1200, 3), 255, np.uint8)  # (no ink to compare: every reading difference stands)


def _line(text: str, y: float, x: float = 100.0) -> list[OcrWord]:
    words = []
    for word in text.split():
        width = 14 * len(word)
        words.append(OcrWord(word, BoundingBox(x, y, x + width, y + 24), 95.0))
        x += width + 12
    return words


def test_a_block_of_lines_is_read_line_by_line():
    original = _line("BAHAN: Kentang ukuran 7 x 7 mm", 100) + _line("RAMUAN: Kentang saiz 7 X 7 mm", 140)
    suspect = _line("BAHAN: Kentang ukuran 9 x 9 mm", 100) + _line("RAMUAN: Kentang saiz 9 X 9 mm", 140)
    found = _paired_lines(BoundingBox(90, 95, 700, 170), original, suspect, BLANK, BLANK, CONFIG)
    assert found.subtype is DiffSubtype.CHANGED
    assert found.detail == "'7 x 7' -> '9 x 9'; '7 X 7' -> '9 X 9'"


def test_a_line_that_moved_is_set_against_itself_not_the_line_now_in_its_place():
    """The block moved down a line: the line now where the finding is, is the one above it moved there -- not new."""
    original = _line("Store below 30 C", 60) + _line("INGREDIENTS: Potato cut-size 7 X 7 mm", 100)
    suspect = _line("Store below 30 C", 100) + _line("INGREDIENTS: Potato cut-size 9 X 9 mm", 140)
    found = _paired_lines(BoundingBox(90, 95, 700, 128), original, suspect, BLANK, BLANK, CONFIG)
    assert found is not _NOT_PAIRED and found.detail == "'7 X 7' -> '9 X 9'"


def test_lines_only_moved_say_no_change_in_their_words():
    original = _line("Store below 30 C", 100) + _line("Keep out of reach of children", 140)
    suspect = _line("Store below 30 C", 130) + _line("Keep out of reach of children", 170)
    assert _paired_lines(BoundingBox(90, 95, 700, 170), original, suspect, BLANK, BLANK, CONFIG) is None


def test_one_line_in_place_is_left_to_the_ordinary_reading():
    original = _line("Store below 30 C", 100)
    suspect = _line("Store below 25 C", 100)
    assert _paired_lines(BoundingBox(90, 95, 400, 128), original, suspect, BLANK, BLANK, CONFIG) is _NOT_PAIRED


def test_a_line_rewritten_is_set_against_what_replaced_it():
    original = _line("MARKETED BY", 100) + _line("ALLANA CONSUMER PRODUCTS LTD", 140)
    suspect = _line("MARKETED BY", 100) + _line("PEPIZO FOODS LLP", 140)
    found = _paired_lines(BoundingBox(90, 95, 700, 170), original, suspect, BLANK, BLANK, CONFIG)
    assert found.detail == "'ALLANA CONSUMER PRODUCTS LTD' -> 'PEPIZO FOODS LLP'"
