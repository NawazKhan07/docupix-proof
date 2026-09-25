"""Something that moved is one finding that says so -- not a strip missing here and a strip added there
(`app.engine.detectors.movement`, `worker._mark_moves`)."""

from __future__ import annotations

import cv2
import numpy as np

from app.engine.detectors import movement
from app.models.diff import BoundingBox

PAPER = (245, 245, 240)


def _logo(image: np.ndarray, x: int, y: int) -> None:
    cv2.rectangle(image, (x, y), (x + 120, y + 90), (200, 60, 30), -1)
    cv2.circle(image, (x + 60, y + 45), 25, (255, 255, 255), -1)
    cv2.line(image, (x + 10, y + 80), (x + 110, y + 10), (20, 20, 120), 5)


def _sheet(*logos: tuple[int, int]) -> np.ndarray:
    image = np.full((600, 900, 3), PAPER, np.uint8)
    for x, y in logos:
        _logo(image, x, y)
    return image


def test_a_picture_that_lies_further_along_is_found_at_its_shift():
    original, suspect = _sheet((300, 200)), _sheet((315, 206))
    shift = movement.moved_by(original, suspect, BoundingBox(300, 200, 420, 290))
    assert shift is not None
    assert abs(shift[0] - 15) <= 1 and abs(shift[1] - 6) <= 1


def test_a_picture_that_stays_where_it_is_has_not_moved():
    original = _sheet((300, 200))
    assert movement.moved_by(original, original.copy(), BoundingBox(300, 200, 420, 290)) is None


def test_a_picture_that_changed_is_not_a_picture_that_moved():
    original, suspect = _sheet((300, 200)), _sheet((315, 200))
    cv2.circle(suspect, (375, 245), 40, (30, 160, 60), -1)  # and something new in the middle of it
    assert movement.moved_by(original, suspect, BoundingBox(300, 200, 420, 290)) is None


def test_a_colour_change_is_not_a_movement():
    original, suspect = _sheet((300, 200)), _sheet((300, 200))
    cv2.rectangle(suspect, (300, 200), (420, 290), (30, 140, 200), -1)
    assert movement.moved_by(original, suspect, BoundingBox(300, 200, 420, 290)) is None


def test_a_flat_crop_fits_anywhere_so_it_says_nothing():
    original = np.full((300, 300, 3), PAPER, np.uint8)
    assert movement.moved_by(original, original.copy(), BoundingBox(100, 100, 200, 200)) is None


def test_one_of_a_row_taken_away_is_not_one_moved_along_the_row():
    """A dot missing from a row of dots fits the dot beside it; but the original already has a dot there."""
    original = np.full((200, 700, 3), PAPER, np.uint8)
    for x in range(60, 640, 60):
        cv2.circle(original, (x, 100), 18, (30, 30, 30), -1)
        cv2.circle(original, (x, 100), 8, (240, 240, 240), -1)
    suspect = original.copy()
    cv2.circle(suspect, (300, 100), 24, PAPER, -1)  # the dot at x=300 is gone
    assert movement.moved_by(original, suspect, BoundingBox(280, 80, 320, 120)) is None


def test_the_strips_a_movement_is_found_in_are_explained_by_it():
    original, suspect = _sheet((300, 200)), _sheet((315, 206))
    # the strip of the logo's left edge that the original has and the suspect has moved away
    assert movement.explained_by_shift(original, suspect, BoundingBox(300, 205, 312, 285), (15, 6))
    # a change that is not the movement is not explained by it
    changed = suspect.copy()
    cv2.circle(changed, (330, 250), 12, (30, 160, 60), -1)
    assert not movement.explained_by_shift(original, changed, BoundingBox(318, 238, 342, 262), (15, 6))


def test_how_far_says_only_the_parts_that_are_a_movement():
    assert movement.how_far(15, 6) == "15 px right, 6 px down"
    assert movement.how_far(-20, 0.4) == "20 px left"
    assert movement.how_far(0.5, -30) == "30 px up"
