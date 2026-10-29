"""A deep, local comparison of two aligned files at full resolution (Automatic mode).

The other pixel detectors work on a reduced copy of the page and forgive a few pixels of misregistration all
over it, which is what hides a small real change (a '36' become '24' in six-point type disappeared when the
alignment residual went from 0.9 px to 1.2 px) and makes their boxes coarse. This looks where the two files
differ AT ALL (a cheap overlay at full size) and there, tile by tile, does what a person would do with two
transparencies: slide one over the other until the type lines up HERE, then see what still differs.

* Each tile that has differing pixels is registered on its own: the best whole-pixel shift (a couple of pixels
  either way) between the two files' greys, taken only if it clearly helps. A carton bent by its folds, or
  re-exported with a different transform, is off by a different amount in different places.
* What is left is compared with a tolerance of `local_forgive_px` each way and both ways round, after a small blur to
  soften what one file draws sharper than the other. (Two files that register exactly forgive nothing and are
  neither shifted nor blurred: the tolerance would hide a digit changed in small type, whose strokes lie within a pixel
  of a different digit's.) What differs beyond that is real: the outline of a letter
  the same in both is not, a different letter, a missing dot or a moved word is.
* A recoloured AREA (a panel, a tint) is found first, by colour over flat pixels, and its offset is allowed for: a
  pixel that differs by no more than that offset (or lies between the two colours, as an anti-aliased edge does) is not
  a difference. So the area is ONE colour finding with its own exact box, and whatever else changed inside it -- a
  digit, a word -- is found on its own instead of being lost in it.
* The pixels that differ are grouped into differences at their own, full-resolution, extent, so the box drawn
  hugs what changed. They are also returned as a mask, which is what the findings' borders are fitted to.
* A difference that is one long thin stroke is flagged as line-work (a die line or fold line): the worker decides
  whether it is ignored.

It only finds; whether a difference is a change at all (a softer copy, a shade of registration) is still judged
by `verify`, and what its words say by reading.
"""

from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import cv2
import numpy as np

from app.config import DetectionConfig
from app.engine.detectors.regions import padded_window
from app.engine.detectors.verify import marks_within_pair
from app.logging_config import StageTimer
from app.models.diff import BoundingBox, DiffCandidate, DiffSubtype, DiffType

logger = logging.getLogger(__name__)

_TILE_PX = 128  # the unit that is registered and compared
_MIN_TILE_DIFFERING = 4  # a tile with fewer differing pixels than this is left alone
_SHIFT_GAIN = 0.9  # a shift must cut the difference to this share of the unshifted one to be taken
_SHIFT_MIN_GAIN = 0.5  # ...and by at least this many grey levels
_MAX_GROUPS = 600  # the largest differences are kept when there are more than this
_DIRECTIONAL_SHARE = 0.9  # this much of the differing marks on one side makes a difference 'added' or 'missing'
_SAME_SHAPE = 0.97  # marks that coincide this well are the same shape: what differs is its colour (a digit changed for another one, whose strokes lie close to it, coincides less)
_SPLIT_AREA_PX = 250_000  # a group whose box is bigger than this may be a chain of unrelated differences: it is grouped again, closer
_SPLIT_MIN_REACH_PX = 2
_SPARSE_FILL = 0.08  # a group whose pixels fill less of its box than this is a drawing of lines, not a region
_HOLLOW_SHARE = 0.25  # a group round a hole this big (a share of its box) is a frame round something that did not change


@dataclass
class LocalResult:
    candidates: list[DiffCandidate]
    # Where the two files differ after each tile was registered and forgiven (full size, boolean).
    residual: np.ndarray
    colour_regions: int = 0  # how many recoloured regions were found


def _max_abs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per pixel, the largest difference in any channel (uint8)."""
    difference = cv2.absdiff(a, b)
    return np.maximum(np.maximum(difference[..., 0], difference[..., 1]), difference[..., 2])


def _shift_of(gray_a: np.ndarray, gray_b: np.ndarray, margin: int, radius: int) -> tuple[int, int]:
    """The whole-pixel shift of file B (within `radius`) that makes it lie best over file A in the tile's core."""
    if radius <= 0:
        return 0, 0
    height, width = gray_a.shape
    core = gray_a[margin : height - margin, margin : width - margin]
    best, best_error = (0, 0), float("inf")
    zero_error = None
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            moved = gray_b[margin + dy : height - margin + dy, margin + dx : width - margin + dx]
            error = float(np.abs(core - moved).mean())
            if dy == 0 and dx == 0:
                zero_error = error
            if error < best_error:
                best, best_error = (dy, dx), error
    assert zero_error is not None
    if best_error < _SHIFT_GAIN * zero_error and zero_error - best_error >= _SHIFT_MIN_GAIN:
        return best
    return 0, 0


def _from_offset(difference: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """How far each pixel's difference (int16, H x W x 3) lies from the way between 'no change' and its area's colour
    offset (same shape): the largest channel (uint8 range). A pixel of the recoloured background differs by the offset,
    the fill of a letter by nothing, an edge by something between: all of them are 0 here."""
    d = difference.astype(np.float32)
    o = offset.astype(np.float32)
    length = (o * o).sum(axis=-1)
    share = np.where(length > 0, np.clip((d * o).sum(axis=-1) / np.maximum(length, 1e-6), 0.0, 1.0), 0.0)
    return np.abs(d - share[..., None] * o).max(axis=-1)


def _tile_residual(
    a: np.ndarray, b: np.ndarray, margin: int, config: DetectionConfig, offsets: np.ndarray | None = None
) -> np.ndarray:
    """Where the tile's core differs between A and B once B is slid into place and one pixel is forgiven (bool).
    `offsets` (the tile's core, H x W x 3, int16) is the colour offset of the recoloured area each pixel lies in."""
    return _tile_difference(a, b, margin, config, offsets) > config.local_tolerance


def _tile_difference(
    a: np.ndarray, b: np.ndarray, margin: int, config: DetectionConfig, offsets: np.ndarray | None = None
) -> np.ndarray:
    """How much the tile's core still differs between A and B, pixel by pixel (levels), once B is slid into place and
    one pixel is forgiven: what `_tile_residual` holds against `local_tolerance`."""
    height, width = a.shape[:2]
    if config.local_blur_px > 0:
        a = cv2.GaussianBlur(a, (0, 0), config.local_blur_px)
        b = cv2.GaussianBlur(b, (0, 0), config.local_blur_px)
    gray_a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    shift_y, shift_x = _shift_of(gray_a, gray_b, margin, config.local_shift_radius_px)

    core_a = a[margin : height - margin, margin : width - margin]
    core_b = b[margin : height - margin, margin : width - margin]
    nearest_a = nearest_b = None
    forgive = config.local_forgive_px
    for oy in range(-forgive, forgive + 1):
        for ox in range(-forgive, forgive + 1):
            dy, dx = shift_y + oy, shift_x + ox
            moved_b = b[margin + dy : height - margin + dy, margin + dx : width - margin + dx]
            moved_a = a[margin - dy : height - margin - dy, margin - dx : width - margin - dx]
            if offsets is None:
                from_a = _max_abs(core_a, moved_b)  # A here, against B a pixel around where it should be
                from_b = _max_abs(core_b, moved_a)  # and B here, against A a pixel around
            else:
                from_a = _from_offset(moved_b.astype(np.int16) - core_a, offsets)
                from_b = _from_offset(core_b.astype(np.int16) - moved_a, offsets)
            nearest_a = from_a if nearest_a is None else np.minimum(nearest_a, from_a)
            nearest_b = from_b if nearest_b is None else np.minimum(nearest_b, from_b)
    return np.maximum(nearest_a, nearest_b)


def _tiles(coarse: np.ndarray) -> list[tuple[int, int]]:
    """(row, column) of every tile that has differing pixels."""
    height, width = coarse.shape
    rows, columns = math.ceil(height / _TILE_PX), math.ceil(width / _TILE_PX)
    padded = np.zeros((rows * _TILE_PX, columns * _TILE_PX), np.uint8)
    padded[:height, :width] = coarse
    counts = padded.reshape(rows, _TILE_PX, columns, _TILE_PX).sum(axis=(1, 3), dtype=np.int64)  # (exactly per tile)
    return [(int(r), int(c)) for r, c in zip(*np.nonzero(counts >= _MIN_TILE_DIFFERING))]


def _boxes(mask: np.ndarray, reach: int) -> tuple[list[tuple[int, int, int, int, int, int]], np.ndarray]:
    """Groups of pixels of `mask` within `reach` of each other, as (x0, y0, x1, y1, pixels, label), largest first, and
    the picture of labels (of the mask grown by `reach`) the last of each refers to."""
    size = 2 * reach + 1
    count, labels = cv2.connectedComponents(
        cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))), connectivity=8
    )
    if count <= 1:
        return [], labels
    ys, xs = np.nonzero(mask)
    owner = labels[ys, xs]
    order = np.argsort(owner, kind="stable")
    owner, ys, xs = owner[order], ys[order], xs[order]
    starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
    pixels = np.diff(np.r_[starts, len(owner)])
    x0, x1 = np.minimum.reduceat(xs, starts), np.maximum.reduceat(xs, starts) + 1
    y0, y1 = np.minimum.reduceat(ys, starts), np.maximum.reduceat(ys, starts) + 1
    groups = list(zip(x0.tolist(), y0.tolist(), x1.tolist(), y1.tolist(), pixels.tolist(), owner[starts].tolist()))
    return sorted(groups, key=lambda group: -group[4]), labels


def _in_cells(own: np.ndarray, left: int, top: int) -> list[tuple[int, int, int, int, int, bool]]:
    """The pixels of `own` (whose corner is at `left`, `top`) as one group per `_TILE_PX` cell that has any."""
    cells = []
    for cy in range(0, own.shape[0], _TILE_PX):
        for cx in range(0, own.shape[1], _TILE_PX):
            ys, xs = np.nonzero(own[cy : cy + _TILE_PX, cx : cx + _TILE_PX])
            if len(xs):
                cells.append((left + cx + int(xs.min()), top + cy + int(ys.min()), left + cx + int(xs.max()) + 1, top + cy + int(ys.max()) + 1, len(xs), False))
    return cells


_THIN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  # a stroke up to eight pixels thick does not survive it (the two files' lines, side by side, differ over a width of both)
_LINE_MAX_THICKNESS_PX = 12  # the box of a straight line is no thicker than this
_LINE_ELONGATION = 10.0  # a single stroke is this many times as long as it is thick
_LINE_NETWORK_FILL = 0.06  # ...and a drawing of joined strokes fills less of its box than this


def _is_line_work(own: np.ndarray, minimum_length: int) -> bool:
    """Whether the pixels of one difference (a crop, True = differs) are only long thin strokes: a die line or a fold
    line that one file draws and the other does not, or draws in another colour. Nothing thicker than a few pixels,
    and long: a letter is thin but short, so a difference in type is never line-work."""
    height, width = own.shape
    longest = max(height, width)
    if longest < minimum_length:
        return False
    pixels = int(own.sum())
    if pixels == 0:
        return False
    # (padded with nothing: erosion treats what lies beyond the crop as full, and the crop is the stroke's own box)
    padded = cv2.copyMakeBorder(own.astype(np.uint8), 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)
    thick = int(cv2.erode(padded, _THIN_KERNEL).sum())
    if thick > 0.08 * pixels:
        return False
    if min(height, width) <= _LINE_MAX_THICKNESS_PX:
        return longest >= _LINE_ELONGATION * min(height, width)  # a straight line: its box is as thin as it is
    # a diagonal or a drawing of joined strokes: a big box that is almost empty (a line of type is thin-stroked too, but its box
    # is a good fifth ink)
    return pixels < _LINE_NETWORK_FILL * height * width


def _grouped(mask: np.ndarray, reach: int, minimum_line: int = 60, top_level: bool = True) -> list[tuple[int, int, int, int, int, bool]]:
    """The differences in `mask` as groups (x0, y0, x1, y1, pixels, is line-work), largest first.

    Pixels within `reach` of each other are one group -- the strokes of a letter. (Letters are joined into words and
    lines when the differences are clustered, only where they really adjoin.) But where two files differ all over (two
    different cartons) or by a drawing of thin lines, the pixels chain into one group as big as the page, and a small
    difference lying inside its box (a block, a dot) is lost in it. So a group with a big box is grouped again at half
    the reach, on its own pixels, down to `_SPLIT_MIN_REACH_PX`; and a group still big at that, and mostly empty (a
    drawing of joined lines), is cut along the grid of tiles. A solid region that differs stays whole.
    """
    groups = []
    found, labels = _boxes(mask, reach)
    for x0, y0, x1, y1, pixels, label in found:
        area = (x1 - x0) * (y1 - y0)
        long_enough = max(x1 - x0, y1 - y0) >= minimum_line
        own = None
        if area > _SPLIT_AREA_PX or long_enough:
            own = mask[y0:y1, x0:x1] & (labels[y0:y1, x0:x1] == label)
        if area > _SPLIT_AREA_PX:
            if reach > _SPLIT_MIN_REACH_PX:
                groups += _regrouped(own, reach, x0, y0, minimum_line)
                continue
            if pixels < _SPARSE_FILL * area or _hollow(own):
                groups += [(a, b, c, d, n, _is_line_work(own[b - y0 : d - y0, a - x0 : c - x0], minimum_line)) for a, b, c, d, n, _ in _in_cells(own, x0, y0)]
                continue
        line = long_enough and own is not None and _is_line_work(own, minimum_line)
        groups.append((x0, y0, x1, y1, pixels, bool(line)))
    return sorted(groups, key=lambda group: -group[4])


def _hollow(own: np.ndarray) -> bool:
    """Whether the differing pixels `own` (one group's box) run round a big hole: a frame -- a panel's border pattern
    re-laid round a panel that did not change -- whose box would be the whole panel, holding every finding inside it."""
    contours, hierarchy = cv2.findContours(own.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return False
    holes = [cv2.contourArea(contour) for contour, links in zip(contours, hierarchy[0]) if links[3] >= 0]
    return bool(holes) and max(holes) >= _HOLLOW_SHARE * own.shape[0] * own.shape[1]


def _regrouped(own: np.ndarray, reach: int, left: int, top: int, minimum_line: int) -> list[tuple[int, int, int, int, int, bool]]:
    inner = _grouped(own, max(_SPLIT_MIN_REACH_PX, reach // 2), minimum_line, top_level=False)
    return [(left + a, top + b, left + c, top + d, n, line) for a, b, c, d, n, line in inner]


def _candidate(
    original: np.ndarray, suspect: np.ndarray, group: tuple[int, int, int, int, int, bool], config: DetectionConfig
) -> DiffCandidate:
    """One difference as a finding: a colour change if the two files' marks are the same shape (only the colour
    differs), else a graphic that was added, is missing or changed."""
    x0, y0, x1, y1, pixels, line_work = group
    box = BoundingBox(float(x0), float(y0), float(x1), float(y1))
    window = padded_window(box, 4, original.shape)
    wx0, wy0, wx1, wy1 = window
    marks = marks_within_pair(original[wy0:wy1, wx0:wx1], suspect[wy0:wy1, wx0:wx1], config.residue_slack_px)
    if marks is not None and marks.agreement >= _SAME_SHAPE:
        peak = float(_max_abs(original[y0:y1, x0:x1], suspect[y0:y1, x0:x1]).max())
        return DiffCandidate(
            box, DiffType.COLOUR, DiffSubtype.CHANGED, min(1.0, peak / 255.0), f"Colour changed (dE {peak * 100 / 255:.1f})", peak * 100 / 255,
            line_work=line_work,
        )
    subtype, wording = DiffSubtype.CHANGED, "Graphic changed"
    if marks is not None:
        only_original, only_suspect = marks.only_original, marks.only_suspect
        total = only_original + only_suspect
        if total > 0 and only_suspect >= _DIRECTIONAL_SHARE * total:
            subtype, wording = DiffSubtype.ADDED, "Graphic added in suspect"
        elif total > 0 and only_original >= _DIRECTIONAL_SHARE * total:
            subtype, wording = DiffSubtype.REMOVED, "Graphic missing from suspect"
    return DiffCandidate(
        box, DiffType.GRAPHICS, subtype, min(1.0, pixels / max(1.0, box.area)), wording, float(pixels), line_work=line_work
    )


@dataclass
class _Regions:
    candidates: list[DiffCandidate]
    ids: np.ndarray | None  # H x W uint16: which recoloured AREA a pixel lies in (0 = none)
    offsets: np.ndarray | None  # (areas + 1) x 3 int16: each area's colour offset (suspect minus original)


_BAND_PX = 512
_AREA_JOIN_PX = 12  # pieces of one colour change this close (or inside each other) are one recoloured area...
_EDGE_GROW_PX = 2  # the box of a recoloured area is grown by this much: its flat pixels stop short of its edge
_AREA_EDGE_PX = 11  # a recoloured area's allowance for its offset reaches this many pixels (odd) past its flat part: to its frame line's far side
_AREA_JOIN_LEVELS = 8  # ...if their colour offsets differ by no more than this many levels


_SURROUND_PX = 8  # the ring of page round a colour region whose colour is compared with the region's in each file
_LIKE_PAPER = 9.0  # a region whose colour is this close (CIE76) to what surrounds it is not there, in that file


def _lab_of(colour: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(np.uint8(colour).reshape(1, 1, 3), cv2.COLOR_RGB2LAB).reshape(3).astype(np.float32)
    lab[0] *= 100.0 / 255.0
    return lab


def _added_or_removed(original: np.ndarray, suspect: np.ndarray, known: dict) -> DiffSubtype | None:
    """ADDED or REMOVED if, in one file, the region is the colour of the page round it (so nothing is there), and in the
    other it is not: the region is a graphic that one file has. None where both have something there (a recolour)."""
    height, width = original.shape[:2]
    x0, y0, x1, y1 = known["box"]
    left, top, right, bottom = max(0, x0 - _SURROUND_PX), max(0, y0 - _SURROUND_PX), min(width, x1 + _SURROUND_PX), min(height, y1 + _SURROUND_PX)
    inside = np.zeros((bottom - top, right - left), bool)
    for x, y, w, h, part in known["parts"]:
        inside[y - top : y - top + h, x - left : x - left + w] |= part
    ring = ~cv2.dilate(inside.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if ring.sum() < 16 or not inside.any():
        return None
    colours = []
    for picture in (original, suspect):
        crop = picture[top:bottom, left:right]
        colours.append((_lab_of(np.median(crop[inside], axis=0)), _lab_of(np.median(crop[ring], axis=0))))
    (a_inside, a_ring), (b_inside, b_ring) = colours
    a_gone = float(np.linalg.norm(a_inside - a_ring)) < _LIKE_PAPER
    b_gone = float(np.linalg.norm(b_inside - b_ring)) < _LIKE_PAPER
    if a_gone != b_gone and _a_background(original, suspect, inside, (left, top, right, bottom), colours):
        # A panel's background, with the type and marks printed on it in both files, that has turned the colour of the
        # paper (a grey panel now white): its colour changed -- the panel is still there, and so is what is on it, each
        # a finding of its own where it differs -- not a graphic removed with everything on it.
        return None
    if b_gone and not a_gone:
        return DiffSubtype.REMOVED
    if a_gone and not b_gone:
        return DiffSubtype.ADDED
    return None


_BACKGROUND_HOLES = 0.1  # a coloured area whose outline holds at least this share of other pixels (type, marks)...
_BACKGROUND_PRINTED = 0.15  # ...at least this share of them printed in BOTH files, is a background, not a graphic


def _a_background(original: np.ndarray, suspect: np.ndarray, inside: np.ndarray, window: tuple, colours: list) -> bool:
    """Whether the coloured area (`inside`, within `window` of the page) is a background things are printed on in both
    files: holes in it (where the type is) that hold print -- off the area's own colour -- in the original and in the
    suspect alike. A shape removed (a logo, a badge) has nothing printed in its holes in the file it is gone from."""
    left, top, right, bottom = window
    contours, _hierarchy = cv2.findContours(inside.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(inside.shape, np.uint8)
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    holes = filled.astype(bool) & ~inside
    if holes.sum() < _BACKGROUND_HOLES * filled.sum():
        return False
    for picture, (area_colour, _ring) in zip((original, suspect), colours):
        crop = picture[top:bottom, left:right]
        lab = cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[..., 0] *= 100.0 / 255.0
        printed = np.linalg.norm(lab[holes] - area_colour, axis=1) >= _LIKE_PAPER
        if printed.mean() < _BACKGROUND_PRINTED:
            return False
    return True


def _colour_regions(
    original: np.ndarray, suspect: np.ndarray, mask: np.ndarray | None, config: DetectionConfig
) -> _Regions:
    """Where the two files differ in colour over FLAT pixels, as regions with their own exact boxes (see the module's
    account). CIE76 delta-E on 8-bit Lab is enough to find them: whether a region is a change or only a shade of profile
    is judged after, on its average delta-E (CIEDE2000), by `verify`."""
    height, width = original.shape[:2]
    faint = _max_abs(original, suspect) > config.local_faint_threshold
    if mask is not None:
        faint &= mask
    if not faint.any():
        return _Regions([], None, None)

    gray_a, gray_b = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY), cv2.cvtColor(suspect, cv2.COLOR_RGB2GRAY)
    kernel3 = np.ones((3, 3), np.uint8)
    edge = (cv2.morphologyEx(gray_a, cv2.MORPH_GRADIENT, kernel3) > config.local_flat_gradient) | (
        cv2.morphologyEx(gray_b, cv2.MORPH_GRADIENT, kernel3) > config.local_flat_gradient
    )
    flat = ~cv2.dilate(edge.astype(np.uint8), kernel3).astype(bool)

    delta = np.zeros((height, width), np.float32)

    def band(y0: int) -> None:
        y1 = min(height, y0 + _BAND_PX)
        if not faint[y0:y1].any():
            return
        lab_a = cv2.cvtColor(original[y0:y1], cv2.COLOR_RGB2LAB).astype(np.int16)
        lab_b = cv2.cvtColor(suspect[y0:y1], cv2.COLOR_RGB2LAB).astype(np.int16)
        d = (lab_a - lab_b).astype(np.float32)
        d[..., 0] *= 100.0 / 255.0
        delta[y0:y1] = np.sqrt((d * d).sum(axis=-1))

    with ThreadPoolExecutor(max_workers=max(1, min(8, os.cpu_count() or 2))) as pool:
        list(pool.map(band, range(0, height, _BAND_PX)))

    coloured = (delta >= config.local_colour_delta_e) & flat & faint
    coloured = cv2.morphologyEx(coloured.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    coloured = cv2.morphologyEx(coloured, cv2.MORPH_OPEN, kernel3)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(coloured, connectivity=8)
    # The biggest first. Pieces of one colour change that lie next to each other (or inside each other) are ONE recoloured
    # area: the type printed on a recoloured panel leaves only slivers of its background flat, and the panel's frame is
    # flat apart from it, but they are all the panel.
    areas: list[dict] = []
    for label in sorted(range(1, count), key=lambda index: -int(stats[index][4])):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < config.local_colour_min_area_px:
            continue
        inside = labels[y : y + h, x : x + w] == label
        crop_a, crop_b = original[y : y + h, x : x + w], suspect[y : y + h, x : x + w]
        shift = np.median(crop_b[inside].astype(np.int16) - crop_a[inside].astype(np.int16), axis=0).astype(np.int16)
        mean_delta = float(delta[y : y + h, x : x + w][inside].mean())
        for known in areas:
            bx0, by0, bx1, by1 = known["box"]
            gap = max(x - bx1, bx0 - (x + w), y - by1, by0 - (y + h), 0)
            if gap <= _AREA_JOIN_PX and int(np.abs(shift - known["shift"]).max()) <= _AREA_JOIN_LEVELS:
                known["box"] = (min(bx0, x), min(by0, y), max(bx1, x + w), max(by1, y + h))
                known["pixels"] += area
                known["delta"] += mean_delta * area
                known["parts"].append((x, y, w, h, inside))
                break
        else:
            areas.append({"box": (x, y, x + w, y + h), "shift": shift, "pixels": area, "delta": mean_delta * area, "parts": [(x, y, w, h, inside)]})

    candidates: list[DiffCandidate] = []
    ids = np.zeros((height, width), np.uint16)
    offsets = [np.zeros(3, np.int16)]
    for known in areas:
        x0, y0, x1, y1 = known["box"]
        mean_delta = known["delta"] / known["pixels"]
        # (the flat pixels stop a couple of pixels short of the edge they lie inside of: the area reaches that far)
        box = BoundingBox(float(max(0, x0 - _EDGE_GROW_PX)), float(max(0, y0 - _EDGE_GROW_PX)), float(min(width, x1 + _EDGE_GROW_PX)), float(min(height, y1 + _EDGE_GROW_PX)))
        gone = _added_or_removed(original, suspect, known)
        if gone is not None:
            # One file has the colour of the paper round it where the other has a shape of colour: a graphic added or removed
            candidates.append(
                DiffCandidate(
                    box, DiffType.GRAPHICS, gone, min(1.0, mean_delta / 25.0),
                    "Graphic added in suspect" if gone is DiffSubtype.ADDED else "Graphic missing from suspect", float(known["pixels"]),
                )
            )
        else:
            candidates.append(
                DiffCandidate(
                    box, DiffType.COLOUR, DiffSubtype.CHANGED, min(1.0, mean_delta / 25.0), f"Colour changed (dE {mean_delta:.1f})", mean_delta,
                    region=True,
                )
            )
        if known["pixels"] >= config.local_offset_min_area_px and gone is None and len(offsets) < 65000:
            edge = _AREA_EDGE_PX // 2
            for x, y, w, h, inside in known["parts"]:
                # the whole area the part spans, holes (the type printed on it) included, and `edge` pixels beyond its edge
                contours, _hierarchy = cv2.findContours(inside.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                filled = np.zeros((h + 2 * edge, w + 2 * edge), np.uint8)
                cv2.drawContours(filled, contours, -1, 1, cv2.FILLED, offset=(edge, edge))
                filled = cv2.dilate(filled, np.ones((_AREA_EDGE_PX, _AREA_EDGE_PX), np.uint8)).astype(bool)
                top, left = max(0, y - edge), max(0, x - edge)
                bottom, right = min(height, y + h + edge), min(width, x + w + edge)
                window = filled[top - (y - edge) : bottom - (y - edge), left - (x - edge) : right - (x - edge)]
                ids[top:bottom, left:right][window] = len(offsets)
            offsets.append(known["shift"])
    if len(offsets) == 1:
        return _Regions(candidates, None, None)
    return _Regions(candidates, ids, np.array(offsets, np.int16))


def detect(
    original: np.ndarray, suspect: np.ndarray, config: DetectionConfig, compare_mask: np.ndarray | None = None
) -> LocalResult:
    """Differences between two aligned pictures, found tile by tile at full size (see the module's account).

    Worked out on the box round what is compared -- on a proof sheet the artwork can be a third of the page -- laid on
    the tiles' own grid, so every tile is the tile it is on the page and what is found is exactly what the whole page
    would give; then put back on the page."""
    box = _compared_box(compare_mask)
    if box is None:
        return _detect(original, suspect, config, compare_mask)
    x0, y0, x1, y1 = box
    found = _detect(original[y0:y1, x0:x1], suspect[y0:y1, x0:x1], config, compare_mask[y0:y1, x0:x1])
    residual = np.zeros(original.shape[:2], bool)
    residual[y0:y1, x0:x1] = found.residual
    for candidate in found.candidates:
        b = candidate.bbox
        candidate.bbox = BoundingBox(b.x0 + x0, b.y0 + y0, b.x1 + x0, b.y1 + y0)
    return LocalResult(found.candidates, residual, found.colour_regions)


_BOX_WORTH = 0.85  # (the box is used where it is less than this share of the page: otherwise the page, as it is)


def _compared_box(compare_mask: np.ndarray | None) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) the box round what `compare_mask` compares, out to whole tiles (and a tile more, for what a
    tile reads round itself) -- None where there is no mask, nothing in it, or the box is nearly the page."""
    if compare_mask is None or not compare_mask.any():
        return None
    height, width = compare_mask.shape
    rows = np.flatnonzero(compare_mask.any(axis=1))
    columns = np.flatnonzero(compare_mask.any(axis=0))
    x0 = max(0, (int(columns[0]) // _TILE_PX - 1) * _TILE_PX)
    y0 = max(0, (int(rows[0]) // _TILE_PX - 1) * _TILE_PX)
    x1 = min(width, (int(columns[-1]) // _TILE_PX + 2) * _TILE_PX)
    y1 = min(height, (int(rows[-1]) // _TILE_PX + 2) * _TILE_PX)
    if (x1 - x0) * (y1 - y0) >= _BOX_WORTH * height * width:
        return None
    return x0, y0, x1, y1


def _detect(
    original: np.ndarray, suspect: np.ndarray, config: DetectionConfig, compare_mask: np.ndarray | None = None
) -> LocalResult:
    height, width = original.shape[:2]
    residual = np.zeros((height, width), bool)
    proof = config.local_proof_tolerance
    strong = np.zeros((height, width), bool) if proof > config.local_tolerance else None
    with StageTimer(logger, "detector_local"):
        coarse = _max_abs(original, suspect) > config.local_coarse_threshold
        if compare_mask is not None:
            coarse &= compare_mask
        regions = _colour_regions(original, suspect, compare_mask, config)
        if not coarse.any():
            return LocalResult(regions.candidates, residual, len(regions.candidates))

        margin = config.local_shift_radius_px + config.local_forgive_px + int(math.ceil(3 * config.local_blur_px))

        def work(tile: tuple[int, int]) -> None:
            row, column = tile
            y0, x0 = row * _TILE_PX, column * _TILE_PX
            y1, x1 = min(height, y0 + _TILE_PX), min(width, x0 + _TILE_PX)
            top, left = max(0, y0 - margin), max(0, x0 - margin)
            bottom, right = min(height, y1 + margin), min(width, x1 + margin)
            a, b = original[top:bottom, left:right], suspect[top:bottom, left:right]
            pad = (margin - (y0 - top), margin - (bottom - y1), margin - (x0 - left), margin - (right - x1))
            if any(pad):  # a tile at the edge of the page: repeat its border so a shift never reaches outside
                a = cv2.copyMakeBorder(a, pad[0], pad[1], pad[2], pad[3], cv2.BORDER_REPLICATE)
                b = cv2.copyMakeBorder(b, pad[0], pad[1], pad[2], pad[3], cv2.BORDER_REPLICATE)
            tile_offsets = None
            if regions.ids is not None:
                tile_ids = regions.ids[y0:y1, x0:x1]
                if tile_ids.any():  # (a tile inside a recoloured area: its pixels differ by that area's offset)
                    tile_offsets = regions.offsets[tile_ids]
                    full = np.zeros((a.shape[0] - 2 * margin, a.shape[1] - 2 * margin, 3), np.int16)
                    full[: y1 - y0, : x1 - x0] = tile_offsets
                    tile_offsets = full
            difference = _tile_difference(a, b, margin, config, tile_offsets)[: y1 - y0, : x1 - x0]
            different = difference > config.local_tolerance
            if compare_mask is not None:
                different = different & compare_mask[y0:y1, x0:x1]
            residual[y0:y1, x0:x1] = different  # (tiles do not overlap: each writes only its own part)
            if strong is not None:
                strong[y0:y1, x0:x1] = different & (difference > proof)

        tiles = _tiles(coarse)
        with ThreadPoolExecutor(max_workers=max(1, min(8, os.cpu_count() or 2))) as pool:
            list(pool.map(work, tiles))

        groups = [
            group
            for group in _grouped(residual, config.local_group_px, config.local_line_min_length_px)
            if group[4] >= config.local_min_pixels
            # (calibrated: only a difference some of which stands clear of the pair's own noise -- see `calibrated`)
            and (strong is None or int(strong[group[1] : group[3], group[0] : group[2]].sum()) >= config.local_proof_min_pixels)
        ]
        candidates = [_candidate(original, suspect, group, config) for group in groups[:_MAX_GROUPS]]
        candidates += regions.candidates
    logger.info(
        "local comparison finished",
        extra={"tiles": len(tiles), "groups": len(groups), "colour_regions": len(regions.candidates), "candidates": len(candidates)},
    )
    return LocalResult(candidates, residual, len(regions.candidates))


_NOISE_TILES = 150  # the pair's own noise is measured on at most this many tiles...
_NOISE_TILE_PERCENTILE = 99.5  # ...each by how far its strongest pixels (bar a few) still differ...
_NOISE_MIN_INK = 0.02  # ...among tiles with at least this share printed (paper against paper says nothing)
_NOISE_QUIET_SHARE = 25.0  # (percentile over tiles)


def measure_noise(
    original: np.ndarray, suspect: np.ndarray, config: DetectionConfig, compare_mask: np.ndarray | None = None
) -> float | None:
    """How much two aligned pictures differ where they print the same thing -- the pair's own noise (rendering,
    anti-aliasing, registration, softness), in grey levels, measured the way `_tile_residual` compares: over printed
    tiles spread across the page, each tile's strongest remaining difference, and of those the quietest quarter's (real
    changes are in some tiles, never in all). None if too little is printed to say."""
    height, width = original.shape[:2]
    margin = config.local_shift_radius_px + config.local_forgive_px + int(math.ceil(3 * config.local_blur_px))
    rows, columns = math.ceil(height / _TILE_PX), math.ceil(width / _TILE_PX)
    small = cv2.resize(cv2.cvtColor(original, cv2.COLOR_RGB2GRAY), (columns, rows), interpolation=cv2.INTER_AREA)
    ink = cv2.resize(((cv2.cvtColor(original, cv2.COLOR_RGB2GRAY) < 200).astype(np.float32)), (columns, rows), interpolation=cv2.INTER_AREA)
    candidates = [(r, c) for r in range(rows) for c in range(columns) if ink[r, c] >= _NOISE_MIN_INK]
    if compare_mask is not None:
        candidates = [(r, c) for r, c in candidates if compare_mask[min(height - 1, r * _TILE_PX + _TILE_PX // 2), min(width - 1, c * _TILE_PX + _TILE_PX // 2)]]
    del small
    if len(candidates) < 5:
        return None
    step = max(1, len(candidates) // _NOISE_TILES)
    chosen = candidates[::step][:_NOISE_TILES]

    def level(tile: tuple[int, int]) -> float:
        row, column = tile
        y0, x0 = row * _TILE_PX, column * _TILE_PX
        y1, x1 = min(height, y0 + _TILE_PX), min(width, x0 + _TILE_PX)
        top, left, bottom, right = max(0, y0 - margin), max(0, x0 - margin), min(height, y1 + margin), min(width, x1 + margin)
        a, b = original[top:bottom, left:right], suspect[top:bottom, left:right]
        pad = (margin - (y0 - top), margin - (bottom - y1), margin - (x0 - left), margin - (right - x1))
        if any(pad):
            a = cv2.copyMakeBorder(a, *pad, cv2.BORDER_REPLICATE)
            b = cv2.copyMakeBorder(b, *pad, cv2.BORDER_REPLICATE)
        difference = _tile_difference(a, b, margin, config)[: y1 - y0, : x1 - x0]
        return float(np.percentile(difference, _NOISE_TILE_PERCENTILE))

    with ThreadPoolExecutor(max_workers=max(1, min(8, os.cpu_count() or 2))) as pool:
        levels = list(pool.map(level, chosen))
    # the quietest quarter of the printed tiles: a pair whose text changed in most tiles (another strength, another
    # language) still has a quarter that print the same -- the median of all tiles read 191 on such a pair, this 7
    return float(np.percentile(levels, _NOISE_QUIET_SHARE))


def calibrated(config: DetectionConfig, noise: float | None) -> DetectionConfig:
    """`config` with the bar a difference must prove itself against set from the pair's measured `noise` (see
    `DetectionConfig.local_calibrate`)."""
    if not config.local_calibrate or noise is None:
        return config
    proof = int(round(min(config.local_tolerance_max, max(config.local_tolerance, config.local_noise_margin * noise))))
    return replace(config, local_proof_tolerance=proof if proof > config.local_tolerance else 0)
