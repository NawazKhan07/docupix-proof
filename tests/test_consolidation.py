"""One finding for one changed thing, once the findings are described (`worker._consolidated`)."""

from __future__ import annotations

import numpy as np
import cv2

from app.engine.worker import _consolidated, _same_recolour
from app.models.diff import BoundingBox, DiffCandidate, DiffCluster, DiffSubtype, DiffType


def _cluster(number, box, diff_type, subtype=DiffSubtype.CHANGED, detail="") -> DiffCluster:
    b = BoundingBox(*box)
    member = DiffCandidate(bbox=b, type=diff_type, subtype=subtype, confidence=1.0, detail=detail or diff_type.value)
    return DiffCluster(id=number, bbox_in_original=b, bbox_in_suspect=b, type=diff_type, members=[member], detail=member.detail)


def _blank() -> np.ndarray:
    return np.full((400, 600, 3), 255, np.uint8)


def test_a_piece_of_a_word_that_changed_is_part_of_that_change():
    """The hyphen gone from inside 'eco-pack' -> 'ecopack' is that change, not a second finding."""
    word = _cluster(1, (100, 100, 300, 140), DiffType.TEXT, detail="'eco-pack' -> 'ecopack'")
    hyphen = _cluster(2, (160, 115, 175, 125), DiffType.GRAPHICS, DiffSubtype.ADDED)
    (one,) = _consolidated([word, hyphen], _blank(), _blank())
    assert one.detail == "'eco-pack' -> 'ecopack'" and len(one.members) == 2


def test_a_word_changed_inside_a_rewritten_paragraph_stays_its_own():
    paragraph = _cluster(1, (100, 100, 400, 300), DiffType.TEXT, detail="'Expiry Date' -> 'BEST BEFORE'")
    moved = _cluster(2, (200, 200, 260, 225), DiffType.TEXT, DiffSubtype.MOVED, detail="Text moved 19 px right: 'NVZ'")
    assert len(_consolidated([paragraph, moved], _blank(), _blank())) == 2


def test_two_halves_of_one_word_running_up_the_page_are_one():
    top = _cluster(1, (500, 100, 540, 220), DiffType.GRAPHICS, DiffSubtype.REMOVED)
    bottom = _cluster(2, (500, 210, 540, 300), DiffType.TEXT, DiffSubtype.REMOVED, detail="'MATRIX' missing from suspect")
    (one,) = _consolidated([top, bottom], _blank(), _blank())
    assert "MATRIX" in one.detail and one.bbox_in_original.y0 == 100 and one.bbox_in_original.y1 == 300


def test_lines_stacked_on_top_of_each_other_are_not_one_upright_word():
    first = _cluster(1, (100, 100, 400, 120), DiffType.TEXT, detail="'a' -> 'b'")
    second = _cluster(2, (100, 125, 400, 145), DiffType.TEXT, detail="'c' -> 'd'")
    assert len(_consolidated([first, second], _blank(), _blank())) == 2


def _panel(recoloured: bool, symbol: bool) -> np.ndarray:
    image = _blank()
    cv2.rectangle(image, (50, 50), (550, 350), (200, 30, 40) if recoloured else (215, 215, 215), -1)
    cv2.putText(image, "DRUG", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (20, 20, 20), 4, cv2.LINE_AA)
    if symbol:
        cv2.circle(image, (450, 200), 40, (255, 255, 255), 4)
    return image


def test_a_letter_whose_counter_shows_the_panels_new_colour_is_part_of_the_recolour():
    original, suspect = _panel(False, symbol=False), _panel(True, symbol=False)
    assert _same_recolour(BoundingBox(50, 50, 550, 350), BoundingBox(150, 140, 200, 205), original, suspect)


def test_a_symbol_removed_from_a_recoloured_panel_is_its_own_change():
    original, suspect = _panel(False, symbol=True), _panel(True, symbol=False)
    assert not _same_recolour(BoundingBox(50, 50, 550, 350), BoundingBox(400, 150, 500, 250), original, suspect)


def test_words_changed_on_a_recoloured_panel_are_never_folded_into_it():
    panel = _cluster(1, (50, 50, 550, 350), DiffType.COLOUR, detail="Colour changed (dE 6.9)")
    words = _cluster(2, (100, 150, 250, 210), DiffType.TEXT, detail="'236,' -> '208,'")
    image = _panel(False, symbol=False)
    assert len(_consolidated([panel, words], image, image.copy())) == 2
