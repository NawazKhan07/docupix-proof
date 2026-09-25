"""Whether what a finding marks has MOVED: the same picture, a little further along in the other file.

Something that shifted by more than the registration forgives is a difference to every pixel comparison -- a ring of
pixels where it was and where it is -- and reads as 'changed'. But it is the same thing: the crop of the original,
laid over the suspect near it, fits (almost) perfectly at one shift that is not zero. That is what is asked here, from
the pixels alone, so it holds for a graphic, a logo, a line of type in a script nobody has taught the reader.

Only a confident fit says 'moved': the picture must have structure of its own (a flat crop fits anywhere), the fit must
be nearly perfect and clearly better than where it lies already, and the original must not already have the picture
where it is found (one dot of a row of dots removed is not a dot moved to its neighbour's place). Where any of that is in
doubt the finding stays what it was.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from app.logging_config import StageTimer
from app.models.diff import BoundingBox

logger = logging.getLogger(__name__)

MIN_SHIFT_PX = 4.0  # a smaller shift is the registration's (and is forgiven before a finding is made)
SEARCH_PX = 80  # how far from where it lies the picture is looked for (further, and it is gone from here and new there)
MIN_FIT = 0.92  # normalised correlation of the picture with the suspect where it fits best
MIN_GAIN = 0.12  # the fit must beat the fit where it lies by at least this
MAX_ALREADY_THERE = 0.8  # the original's own fit where the picture is found in the suspect: more, and it is a repeat
MIN_STRUCTURE = 14.0  # a crop whose grey levels vary by less than this (standard deviation) is flat: it fits anywhere
MAX_SIDE_PX = 700  # a finding bigger than this is looked at reduced
_PAD_PX = 3


def moved_by(original: np.ndarray, suspect: np.ndarray, box: BoundingBox) -> tuple[float, float] | None:
    """The (dx, dy) by which the picture in `box` of `original` lies in `suspect` -- None unless it is, plainly, the
    same picture moved (see the module's account). Both images are RGB (or grey) in the same pixel frame."""
    height, width = original.shape[:2]
    x0, y0 = max(0, int(box.x0) - _PAD_PX), max(0, int(box.y0) - _PAD_PX)
    x1, y1 = min(width, math.ceil(box.x1) + _PAD_PX), min(height, math.ceil(box.y1) + _PAD_PX)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    scale = min(1.0, MAX_SIDE_PX / max(x1 - x0, y1 - y0))
    sx0, sy0 = max(0, x0 - SEARCH_PX), max(0, y0 - SEARCH_PX)
    sx1, sy1 = min(width, x1 + SEARCH_PX), min(height, y1 + SEARCH_PX)

    def grey(image: np.ndarray, ax0: int, ay0: int, ax1: int, ay1: int) -> np.ndarray:
        crop = image[ay0:ay1, ax0:ax1]
        crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
        if scale < 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return crop

    template = grey(original, x0, y0, x1, y1)
    if float(template.std()) < MIN_STRUCTURE:
        return None
    search = grey(suspect, sx0, sy0, sx1, sy1)
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return None
    fit = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _low, best, _low_at, best_at = cv2.minMaxLoc(fit)
    if best < MIN_FIT:
        return None
    # where the template lies unmoved, in the search window's frame
    rest_x, rest_y = int(round((x0 - sx0) * scale)), int(round((y0 - sy0) * scale))
    at_rest = float(fit[rest_y, rest_x]) if 0 <= rest_y < fit.shape[0] and 0 <= rest_x < fit.shape[1] else 0.0
    if best - at_rest < MIN_GAIN:
        return None
    dx, dy = (best_at[0] - rest_x) / scale, (best_at[1] - rest_y) / scale
    if math.hypot(dx, dy) < MIN_SHIFT_PX:
        return None
    # The original must not already have the picture where the suspect has it (a repeat, not a move).
    there = grey(original, sx0, sy0, sx1, sy1)[best_at[1] : best_at[1] + template.shape[0], best_at[0] : best_at[0] + template.shape[1]]
    if there.shape == template.shape and float(there.std()) >= MIN_STRUCTURE:
        already = float(cv2.matchTemplate(there, template, cv2.TM_CCOEFF_NORMED)[0, 0])
        if already >= MAX_ALREADY_THERE:
            return None
    return dx, dy


EXPLAINED_TOLERANCE = 32  # levels: a pixel this close to its counterpart at the shift is the same
EXPLAINED_SHARE = 0.9  # ... and this many of a piece's pixels must be


def explained_by_shift(original: np.ndarray, suspect: np.ndarray, box: BoundingBox, shift: tuple[float, float]) -> bool:
    """Whether the difference in `box` is only that a picture moved by `shift`: what the original has in it lies, moved,
    in the suspect (the picture left from here) or what the suspect has in it lay, moved back, in the original (it arrived
    here). The pieces a movement is found in -- the strip where it was, the strip where it is -- are not differences of their own."""
    height, width = original.shape[:2]
    dx, dy = int(round(shift[0])), int(round(shift[1]))
    x0, y0, x1, y1 = max(0, int(box.x0)), max(0, int(box.y0)), min(width, math.ceil(box.x1)), min(height, math.ceil(box.y1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return False

    def similar(a: np.ndarray, b: np.ndarray) -> bool:
        if a.shape != b.shape or a.size == 0:
            return False
        return float((np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=-1 if a.ndim == 3 else None) <= EXPLAINED_TOLERANCE).mean()) >= EXPLAINED_SHARE

    def window(image: np.ndarray, ox: int, oy: int) -> np.ndarray | None:
        wx0, wy0, wx1, wy1 = x0 + ox, y0 + oy, x1 + ox, y1 + oy
        if wx0 < 0 or wy0 < 0 or wx1 > width or wy1 > height:
            return None
        return image[wy0:wy1, wx0:wx1]

    here_original, there_suspect = window(original, 0, 0), window(suspect, dx, dy)
    if here_original is not None and there_suspect is not None and similar(here_original, there_suspect):
        return True
    there_original, here_suspect = window(original, -dx, -dy), window(suspect, 0, 0)
    return there_original is not None and here_suspect is not None and similar(there_original, here_suspect)


def find_moves(
    boxes: list[BoundingBox], original: np.ndarray, suspect: np.ndarray
) -> list[tuple[float, float] | None]:
    """`moved_by` for each box."""
    with StageTimer(logger, "find_moves"):
        return [moved_by(original, suspect, box) for box in boxes]


def how_far(dx: float, dy: float) -> str:
    """'12 px right, 3 px down' -- the parts that are more than the registration's."""
    parts = []
    if abs(dx) >= 2:
        parts.append(f"{abs(dx):.0f} px {'right' if dx > 0 else 'left'}")
    if abs(dy) >= 2:
        parts.append(f"{abs(dy):.0f} px {'down' if dy > 0 else 'up'}")
    return ", ".join(parts)
