"""Braille read and compared cell by cell (Braille mode, FR-11.1).

Each file's Braille is read as a Braille reader would: dots set to cells of two columns of three, the cells to lines.
Each cell is its dots (compared exactly) and what they say (uncontracted Braille: letters, numbers after the number
sign, capitals). The original's cells are set against the suspect's where they are, so a finding says which letter
each file has, which dots differ, and where. Braille printed turned on the pack (a panel upside down) is read the right
way round.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.config import DetectionConfig
from app.engine.detectors import braille_diff, braille_reader
from app.engine.detectors.braille_reader import DOT_BITS, glyph, text_of, translate
from app.models.diff import DiffSubtype, DiffType
from app.models.result import BrailleComparison
from app.ui.braille_panel import BrailleResultsPanel

CONFIG = DetectionConfig()
RED = (239, 65, 53)
SPACING, CELL, LINE, RADIUS = 30, 72, 120, 8  # px, at 300 DPI: 2.5 mm dots, 6 mm cells, 10 mm lines

_LETTER_DOTS = {
    "a": (1,), "b": (1, 2), "c": (1, 4), "d": (1, 4, 5), "e": (1, 5), "f": (1, 2, 4), "g": (1, 2, 4, 5), "h": (1, 2, 5),
    "i": (2, 4), "j": (2, 4, 5), "k": (1, 3), "l": (1, 2, 3), "m": (1, 3, 4), "n": (1, 3, 4, 5), "o": (1, 3, 5),
    "p": (1, 2, 3, 4), "q": (1, 2, 3, 4, 5), "r": (1, 2, 3, 5), "s": (2, 3, 4), "t": (2, 3, 4, 5), "u": (1, 3, 6),
    "v": (1, 2, 3, 6), "w": (2, 4, 5, 6), "x": (1, 3, 4, 6), "y": (1, 3, 4, 5, 6), "z": (1, 3, 5, 6),
}
_DIGIT_LETTERS = dict(zip("1234567890", "abcdefghij"))


def _cells(text: str) -> list[tuple[int, ...]]:
    """Uncontracted Braille for `text`: each cell's dots (a space is no dots)."""
    cells, number = [], False
    for char in text:
        if char == " ":
            cells.append(())
            number = False
        elif char.isdigit():
            if not number:
                cells.append((3, 4, 5, 6))
                number = True
            cells.append(_LETTER_DOTS[_DIGIT_LETTERS[char]])
        else:
            number = False
            cells.append(_LETTER_DOTS[char])
    return cells


def _page(lines: list[str], origin=(120, 150), turned: bool = False, size=(700, 1100), shift=(0, 0)) -> np.ndarray:
    """A page with these lines in red Braille dots (`turned`: the whole page upside down, as a carton panel is)."""
    page = np.full((size[0], size[1], 3), 255, np.uint8)
    x0, y0 = origin[0] + shift[0], origin[1] + shift[1]
    for row, text in enumerate(lines):
        for place, dots in enumerate(_cells(text)):
            for dot in dots:
                column, down = (0, dot - 1) if dot <= 3 else (1, dot - 4)
                centre = (x0 + place * CELL + column * SPACING, y0 + row * LINE + down * SPACING)
                cv2.circle(page, centre, RADIUS, RED, -1, cv2.LINE_AA)
    return np.ascontiguousarray(page[::-1, ::-1]) if turned else page


# ---- reading --------------------------------------------------------------------------------------------------------------


def test_uncontracted_braille_is_translated_letters_numbers_and_signs():
    patterns = [sum(DOT_BITS[d] for d in dots) for dots in _cells("metformin 500 mg")]
    assert text_of(patterns) == "metformin 500 mg"
    assert translate(patterns)[10] == "number sign"
    assert glyph(patterns[0]) == "⠍"


@pytest.mark.parametrize("turned", [False, True], ids=["upright", "upside down"])
def test_braille_is_read_cell_by_cell_the_right_way_round(turned):
    page = _page(["metformin", "500 mg", "tablets"], turned=turned)

    blocks, loose = braille_reader.read_dots(braille_reader.find_braille_dots(page, CONFIG))

    assert len(blocks) == 1 and len(loose) == 0
    assert [line.text for line in blocks[0].lines] == ["metformin", "500 mg", "tablets"]
    assert blocks[0].turn == (180 if turned else 0) and not blocks[0].mirrored


# ---- comparing ------------------------------------------------------------------------------------------------------------


def test_the_same_braille_matches_cell_for_cell():
    page = _page(["metformin", "500 mg"])

    candidates, comparison = braille_diff.compare(page, page.copy(), CONFIG)

    assert candidates == [] and comparison.passed
    assert comparison.cells_compared == len(_cells("metformin")) + len(_cells("500 mg")) - 1  # (a space is no cell)
    assert [(line.original_text, line.suspect_text) for line in comparison.lines] == [("metformin", "metformin"), ("500 mg", "500 mg")]


def test_a_letter_changed_is_one_finding_saying_both_letters_and_the_dots():
    candidates, comparison = braille_diff.compare(_page(["metformin", "500 mg"]), _page(["metformin", "800 mg"]), CONFIG)

    assert len(candidates) == 1 and comparison.cells_differing == 1 and not comparison.passed
    (finding,) = candidates
    assert finding.type is DiffType.BRAILLE and finding.subtype is DiffSubtype.CHANGED
    # '5' is the letter e (dots 1-5), '8' the letter h (dots 1-2-5): dot 2 added
    assert "'5'" in finding.detail and "'8'" in finding.detail and "dot 2 added" in finding.detail
    assert comparison.lines[1].suspect_text == "800 mg"
    (changed,) = [cell for cell in comparison.lines[1].cells if not cell.passed]
    assert (changed.original_meaning, changed.suspect_meaning, changed.status) == ("5", "8", "changed")


def test_a_letter_left_out_at_the_end_of_a_word_is_missing_from_the_suspect():
    candidates, comparison = braille_diff.compare(_page(["tablets"]), _page(["tablet"]), CONFIG)

    assert [c.subtype for c in candidates] == [DiffSubtype.REMOVED]
    assert "'s'" in candidates[0].detail and "missing from suspect" in candidates[0].detail
    assert comparison.lines[0].suspect_text == "tablet"


def test_upside_down_braille_is_compared_letter_by_letter_too():
    original = _page(["metformin", "tablets"], turned=True)
    suspect = _page(["metformin", "tablats"], turned=True)

    candidates, comparison = braille_diff.compare(original, suspect, CONFIG)

    assert len(candidates) == 1 and "'e'" in candidates[0].detail and "'a'" in candidates[0].detail
    assert [line.suspect_text for line in comparison.lines] == ["metformin", "tablats"]


def test_a_line_moved_is_one_finding_not_every_cell_of_it():
    candidates, comparison = braille_diff.compare(_page(["metformin"]), _page(["metformin"], shift=(40, 0)), CONFIG)

    assert [c.subtype for c in candidates] == [DiffSubtype.MOVED]
    assert "moved 40 px" in candidates[0].detail
    assert all(cell.status == "moved" for cell in comparison.lines[0].cells)


def test_a_line_only_one_file_has_is_one_finding():
    candidates, comparison = braille_diff.compare(_page(["metformin", "tablets"]), _page(["metformin"]), CONFIG)

    assert [c.subtype for c in candidates] == [DiffSubtype.REMOVED]
    assert "'tablets'" in candidates[0].detail and "missing from suspect" in candidates[0].detail
    assert comparison.lines[1].suspect_text is None


# ---- the panel ------------------------------------------------------------------------------------------------------------


def test_the_panel_shows_each_line_and_each_cell_marked(qapp):
    _candidates, comparison = braille_diff.compare(_page(["metformin", "500 mg"]), _page(["metformin", "800 mg"]), CONFIG)
    panel = BrailleResultsPanel()
    chosen = []
    panel.regionSelected.connect(chosen.append)

    panel.set_comparison(comparison)

    first, second = panel.line_cards()
    assert first.badge.accessibleName() == "pass" and second.badge.accessibleName() == "fail"
    assert [chip.accessibleName() for chip in second.chips].count("changed") == 1
    assert "1 of" in panel.verdict_label.text() and "cells differ" in panel.verdict_label.text()
    changed = next(chip for chip in second.chips if chip.accessibleName() == "changed")
    assert "Original: '5'" in changed.toolTip() and "Suspect: '8'" in changed.toolTip()
    changed.clicked.emit(changed.cell.bbox)
    assert chosen == [changed.cell.bbox]


def test_the_panel_still_shows_a_dot_count_comparison_without_cells(qapp):
    panel = BrailleResultsPanel()
    panel.set_comparison(BrailleComparison(original_dot_count=24, suspect_dot_count=23, matched=20, missing=2, added=1, moved=1))
    assert panel.line_cards() == [] and "4 dot differences" in panel.verdict_label.text()
