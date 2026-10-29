"""Pharmacode (Laetus) reading (SRS FR-10.1).

Neither bundled decoder (ZBar, ZXing) implements Pharmacode, so this does. It is
a very simple symbology -- a row of bars that are each thin or thick, separated
by equal spaces, no start/stop pattern and no check digit -- which is also why
finding one takes some care: a row of equal ticks is not necessarily a code.

How a candidate is accepted:

* **Bars are solid rectangles** of one length (same ends within a couple of
  pixels), several times longer than thick.
* **The spaces between them are all equal.** This is what separates Pharmacode
  from every other bar symbology (EAN, Code 128 have spaces of several widths),
  and from text.
* **The bars come in at most two thicknesses**, thin and thick.
* **There is quiet space at both ends**, so a run of bars taken from the middle
  of a longer pattern (say, part of an EAN symbol) is not mistaken for a code.

At least four bars are needed (a value of 15 or more): two or three equal ticks
cannot be told from ordinary marks, and a false code is worse than an unread one.

Value: reading from the right end, a thin bar in place n adds 2^n and a thick
one adds 2 * 2^n, giving 3 to 131070. When the code is printed vertically the
reading starts at the top. (A symbol that reads the same in both directions, as
one on the sample carton does, has one value either way; for the rest the
convention only matters for showing a number -- comparing two files reads both
the same way, so a difference is a difference whichever end is "first".)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import DetectionConfig
from app.engine.detectors.structure_diff import local_background
from app.models.diff import BoundingBox

logger = logging.getLogger(__name__)

MIN_VALUE, MAX_VALUE = 3, 131070

_BACKGROUND_WINDOW_PX = 200  # large enough that a whole symbol is a minority of it
_DARK_LEVEL = 80  # a bar is darker than this, or than half its local background (a coloured guide line crossing the symbol must not join its bars)
_CROSSING_LINE_PX = 5  # a line crossing the bars (thinner than this) does not link them...
_BRIDGE_PX = 7  # ...and one crossing a bar (up to 6 px wide) does not split it
_MIN_THICKNESS_PX = 2
_FILL_RATIO = 0.75  # a bar is a solid rectangle, not a glyph
_SAME_END_PX = 2  # bars of one symbol end at the same place, give or take this...
_SAME_END_FRACTION = 0.08  # ...or this share of their length, whichever is more
_GAP_TOLERANCE = 0.30  # spaces agree within this share (plus a pixel and a half)
_MAX_FIRST_GAP_FACTOR = 3.0  # the first space is at most this many bar thicknesses
_TWO_CLASSES_RATIO = 1.6  # thick / thin, at least, to be two thicknesses
_MAX_CLASS_RATIO = 4.5  # ...and at most
_CLASS_SPREAD = 0.3  # each thickness is tight within this share (and a pixel)
_SINGLE_CLASS_WIDE_OVER_GAP = 0.85  # when every bar is alike, wide if thicker than this x the space
_QUIET_GAPS = 2.5  # free space beyond each end, in space-widths, at least...
_QUIET_WIDE_GAPS = 5.0  # ...and beyond one of them at least this much (a code stands clear; lettering has its
# neighbours a word-space away: the stems of 'mm' in '268 mm (' had none at all)
_QUIET_INK_SHARE = 0.05
_EDGE_PX = 2


@dataclass(frozen=True)
class Pharmacode:
    value: int
    bbox: BoundingBox
    vertical: bool  # bars lie horizontally, stacked -- the symbol is printed sideways


def encode(value: int) -> list[bool]:
    """Bars for `value` in reading order (place 0 first); True is a thick bar."""
    if not MIN_VALUE <= value <= MAX_VALUE:
        raise ValueError(f"Pharmacode values run from {MIN_VALUE} to {MAX_VALUE}")
    bars = []
    while value > 0:
        if value % 2:
            bars.append(False)
            value = (value - 1) // 2
        else:
            bars.append(True)
            value = (value - 2) // 2
    return bars


def decode(thick: list[bool]) -> int:
    """The value of bars given in reading order (place 0 first); True is a thick bar."""
    return sum((2 if wide else 1) << place for place, wide in enumerate(thick))


def _dark_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    background = local_background(gray, _BACKGROUND_WINDOW_PX)
    # "darker than half the background", in whole numbers (gray < bg / 2 is gray < ceil(bg / 2)): kept in 8 bits,
    # where `background * 0.5` made a float64 copy of the page (763 MB on a 2 m shipper)
    half = (background >> 1) + (background & 1)
    return (gray < np.minimum(np.uint8(_DARK_LEVEL), half)).astype(np.uint8)


def _bar_candidates(dark: np.ndarray, config: DetectionConfig, along_y: bool) -> np.ndarray:
    """Bar-like components as rows of (x, y, w, h): horizontal bars if `along_y`, else vertical ones.

    A guide or fold line crossing a symbol does two things: it joins the bars
    into one blob (through the spaces), and it can cut each bar in two. So
    features thinner than a few pixels ACROSS a bar's length are removed first,
    then gaps of a few pixels ALONG it are closed. The spaces between bars are
    never bridged.
    """
    along = (lambda n: (n, 1)) if along_y else (lambda n: (1, n))
    cleaned = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, along(_CROSSING_LINE_PX)))
    joined = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, along(_BRIDGE_PX)))

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(joined, connectivity=8)
    x, y, w, h, area = (stats[:, index] for index in range(5))
    length, thickness = (w, h) if along_y else (h, w)
    keep = (
        (np.arange(count) > 0)
        & (length >= config.pharmacode_min_bar_length_px)
        & (length <= config.pharmacode_max_bar_length_px)
        & (thickness >= _MIN_THICKNESS_PX)
        & (length >= config.pharmacode_min_aspect * thickness)
        & (area >= _FILL_RATIO * w * h)
    )
    return stats[keep][:, :4]


def _same_length_groups(bars: np.ndarray) -> list[np.ndarray]:
    """Group bars (rows of along0, along1, cross0, cross1) that end at the same place."""
    groups: list[list[int]] = []
    order = np.argsort(bars[:, 2], kind="stable")
    for index in order:
        _a0, _a1, c0, c1 = bars[index]
        tolerance = max(_SAME_END_PX, _SAME_END_FRACTION * (c1 - c0))
        for group in groups:
            g0, g1 = bars[group[0], 2], bars[group[0], 3]
            if abs(c0 - g0) <= tolerance and abs(c1 - g1) <= tolerance:
                group.append(int(index))
                break
        else:
            groups.append([int(index)])
    return [bars[group] for group in groups]


def _runs(group: np.ndarray) -> list[np.ndarray]:
    """Split a group, ordered along the row, into runs of bars with equal spaces."""
    group = group[np.argsort(group[:, 0])]
    runs: list[list[np.ndarray]] = []
    current: list[np.ndarray] = [group[0]]
    gaps: list[float] = []
    for bar in group[1:]:
        previous = current[-1]
        gap = float(bar[0] - previous[1])
        fits = False
        if gap >= 1:  # touching or overlapping bars are not a spaced symbol
            if gaps:
                expected = float(np.median(gaps))
                fits = abs(gap - expected) <= max(1.5, _GAP_TOLERANCE * expected)
            else:
                thickest = max(bar[1] - bar[0], previous[1] - previous[0])
                fits = gap <= _MAX_FIRST_GAP_FACTOR * thickest
        if fits:
            current.append(bar)
            gaps.append(gap)
        else:
            runs.append(current)
            current, gaps = [bar], []
    runs.append(current)
    return [np.array(run) for run in runs]


def _tight(thicknesses: np.ndarray) -> bool:
    """Whether bars meant to be one thickness agree, allowing a pixel of print/scan slop."""
    return thicknesses.size == 0 or (thicknesses.max() - thicknesses.min()) <= _CLASS_SPREAD * thicknesses.mean() + 1


def _classify(run: np.ndarray) -> list[bool] | None:
    """Thick (True) / thin (False) per bar in order along the row, or None if not two clean thicknesses."""
    thickness = (run[:, 1] - run[:, 0]).astype(float)
    gap = float(np.median(run[1:, 0] - run[:-1, 1]))
    thin, thick = thickness.min(), thickness.max()

    if thick < _TWO_CLASSES_RATIO * thin:  # every bar alike: all thin, or all thick
        if not _tight(thickness):
            return None
        return [bool(thickness.mean() > _SINGLE_CLASS_WIDE_OVER_GAP * gap)] * len(run)

    if thick > _MAX_CLASS_RATIO * thin:
        return None
    wide = thickness > float(np.sqrt(thin * thick))
    if not (_tight(thickness[~wide]) and _tight(thickness[wide])):
        return None
    return [bool(flag) for flag in wide]


def _has_quiet_zone(dark: np.ndarray, run: np.ndarray, along_y: bool) -> bool:
    """Free space beyond both ends of the row (`along_y`: the bars are stacked, not side by side): at least
    `_QUIET_GAPS` spaces at each end and `_QUIET_WIDE_GAPS` at one -- measured line by line outward from the end bars,
    each line across the bars' length all but clear of ink (a page's edge is free space)."""
    gap = float(np.median(run[1:, 0] - run[:-1, 1]))
    a0, a1 = int(run[:, 0].min()), int(run[:, 1].max())
    c0, c1 = int(run[:, 2].min()), int(run[:, 3].max())
    limit = int(np.ceil(_QUIET_WIDE_GAPS * gap))
    extent = dark.shape[0] if along_y else dark.shape[1]

    def free(step: int) -> int:
        # (from a couple of lines past the end bar: its own soft edge is not something beside it)
        position, clear = (a0 - 1 - _EDGE_PX, _EDGE_PX) if step < 0 else (a1 + _EDGE_PX, _EDGE_PX)
        while clear < limit and 0 <= position < extent:
            line = dark[position, c0:c1] if along_y else dark[c0:c1, position]
            if line.size and line.mean() > _QUIET_INK_SHARE:
                return clear
            clear += 1
            position += step
        return limit if clear < limit and not 0 <= position < extent else clear

    before, after = free(-1), free(1)
    return min(before, after) >= _QUIET_GAPS * gap and max(before, after) >= _QUIET_WIDE_GAPS * gap - 0.5


def find_pharmacodes(image: np.ndarray, config: DetectionConfig) -> list[Pharmacode]:
    """Every Pharmacode in an RGB image, in either orientation."""
    dark = _dark_mask(image)
    found: list[Pharmacode] = []
    considered = 0

    # Vertical bars lie side by side along x; horizontal bars are stacked along y.
    for along_y in (False, True):
        bars = _bar_candidates(dark, config, along_y)
        considered += len(bars)
        if len(bars) < config.pharmacode_min_bars:
            continue
        if along_y:
            arranged = np.column_stack([bars[:, 1], bars[:, 1] + bars[:, 3], bars[:, 0], bars[:, 0] + bars[:, 2]])
        else:
            arranged = np.column_stack([bars[:, 0], bars[:, 0] + bars[:, 2], bars[:, 1], bars[:, 1] + bars[:, 3]])

        for group in _same_length_groups(arranged):
            if len(group) < config.pharmacode_min_bars:
                continue
            for run in _runs(group):
                if not config.pharmacode_min_bars <= len(run) <= config.pharmacode_max_bars:
                    continue
                thick = _classify(run)
                if thick is None or not _has_quiet_zone(dark, run, along_y=along_y):
                    continue
                # Reading starts at the right end; a symbol lying on its side is read from the top.
                in_reading_order = thick if along_y else thick[::-1]
                value = decode(in_reading_order)
                if not MIN_VALUE <= value <= MAX_VALUE:
                    continue
                a0, a1 = float(run[:, 0].min()), float(run[:, 1].max())
                c0, c1 = float(run[:, 2].min()), float(run[:, 3].max())
                box = BoundingBox(c0, a0, c1, a1) if along_y else BoundingBox(a0, c0, a1, c1)
                found.append(Pharmacode(value=value, bbox=box, vertical=along_y))

    logger.info("pharmacode search finished", extra={"found": len(found), "bar_candidates": considered})
    return found
