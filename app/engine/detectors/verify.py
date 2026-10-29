"""Is a difference between two aligned files a change, or only how each renders the same thing?

Two files that register to within a pixel or two, or where one is a little softer than the other (a
raster where the other has live type, another resampling), differ along every edge: hairlines of 'colour
changed' and 'graphic changed' around letters that are the same in both, and a per-pixel detector cannot tell
them from real changes -- both are pixels that differ. So each finding is checked by what is INSIDE its box,
with a margin wide enough to take in the whole edge:

* the average colour of the two files there. An edge that moved or blurred by a pixel or two conserves it;
  ink added, removed or recoloured does not.
* whether the marks (the ink) coincide, within a couple of pixels. The same letters, even softer, do; a
  different letter, a word that moved or a missing dot do not.

A difference that leaves both unchanged is the same artwork drawn slightly differently, not a change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import cv2
import numpy as np

from app.config import DetectionConfig
from app.engine.detectors.colour_math import ciede2000, rgb_to_lab
from app.engine.detectors.regions import padded_window
from app.models.diff import BoundingBox, DiffCandidate, DiffSubtype, DiffType

logger = logging.getLogger(__name__)

_AGREEMENT_SLACK_PX = 2  # ink this close counts as the same ink
_CONTEXT_PX = 12  # how much wider than a finding its marks are cut (see `_marks_within`)
_MIN_BACKGROUND_PX = 16  # fewer pixels of background than this say nothing about the tint
_PIXEL_TYPES = frozenset({DiffType.COLOUR, DiffType.GRAPHICS})
_THIN_PX = 6  # a difference this thin (and at least `_THIN_ASPECT` times as long) is a line, whose edge residue is a pixel wider than a letter's
_THIN_ASPECT = 3.0
_SMALL_REGION_PX = 4000  # a colour region smaller than this (px^2) is a patch, not a panel
_MAX_CHECKED_AREA_PX = 1_500_000  # a difference this big (a panel) is not a hairline: never checked


@dataclass(frozen=True)
class Evidence:
    """What two files really have inside a finding's box."""

    colour_delta: float  # CIEDE2000 between the two files' AVERAGE colour there
    agreement: float | None  # how far the marks coincide (0-1); None where either is blank and they cannot be compared
    background_delta: float = 0.0  # the same, over the part of the box away from any mark: a tint that changed shows here
    contrast: float = 0.0  # CIEDE2000 between the original's ink and its background there: how much an edge that moved can shift the average
    extent: int = 1  # the narrower side of the window (px): the narrower it is, the more a moved edge shifts its average


def _marks(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """The marks of two colour crops of one place: what departs from the background by a third of the range.

    Both are cut with ONE background and ONE threshold, taken from the two crops together. Cut on their own, a
    softer copy of a busy window has a different median tone, the marks flip from the ink to the paper in one of
    them, and the same letters 'disagree' (a page blurred by 1.3 px gave a finding this way). None if the two
    together have no marks to speak of (both blank).
    """
    grays = [cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop for crop in (first, second)]
    pooled = np.concatenate([gray.ravel() for gray in grays])
    low, high = np.percentile(pooled, (2, 98))
    if high - low < 30:
        return None
    background = int(np.argmax(np.bincount(pooled // 8, minlength=32))) * 8 + 4  # the commonest tone
    threshold = (high - low) / 3
    return tuple(np.abs(gray.astype(np.int16) - background) > threshold for gray in grays)  # type: ignore[return-value]


def _dilated(mask: np.ndarray, slack_px: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * slack_px + 1, 2 * slack_px + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def _delta_of_means(mean_a, mean_b) -> float:
    """CIEDE2000 between two average colours (RGB, 0..255)."""
    return ciede2000(rgb_to_lab(tuple(float(c) / 255.0 for c in mean_a[:3])), rgb_to_lab(tuple(float(c) / 255.0 for c in mean_b[:3])))


def _window_means(a: np.ndarray, b: np.ndarray, where: np.ndarray | None = None) -> tuple[tuple, tuple]:
    """The average colour of two windows (of their pixels in `where`, if given), in one pass each (`cv2.mean`)."""
    mask = None if where is None else where.astype(np.uint8)
    return cv2.mean(np.ascontiguousarray(a), mask=mask), cv2.mean(np.ascontiguousarray(b), mask=mask)


@dataclass(frozen=True)
class _Marks:
    """The two files' marks inside a finding's window, and how they line up."""

    agreement: float
    near_any: np.ndarray  # window-sized: within reach of a mark of either file
    ink: np.ndarray  # window-sized: a mark of either file


def _marks_within(original: np.ndarray, suspect: np.ndarray, box: BoundingBox, pad: int, slack_px: int) -> _Marks | None:
    """The marks of the two images inside `box` (plus `pad`), or None if either has none there.

    The marks are cut, and the reach looked for, in a window wider than the one judged: in a window of just the
    finding a mark close to its edge has its counterpart outside it, and a crop of a few pixels has too little in
    it to say what is a mark and what is background.
    """
    x0, y0, x1, y1 = padded_window(box, pad, original.shape)
    cx0, cy0, cx1, cy1 = padded_window(box, pad + _CONTEXT_PX, original.shape)
    both = _marks(original[cy0:cy1, cx0:cx1], suspect[cy0:cy1, cx0:cx1])
    if both is None:
        return None
    a, b = both
    inside = (slice(y0 - cy0, y1 - cy0), slice(x0 - cx0, x1 - cx0))
    a_here, b_here = a[inside], b[inside]
    if not a_here.any() or not b_here.any():
        return None
    a_near, b_near = _dilated(a, slack_px), _dilated(b, slack_px)
    agreement = min((a_here & b_near[inside]).sum() / a_here.sum(), (b_here & a_near[inside]).sum() / b_here.sum())
    return _Marks(float(agreement), _dilated(a | b, slack_px + 1)[inside], (a | b)[inside])


@dataclass(frozen=True)
class PairMarks:
    """How the marks of two crops of one place line up."""

    agreement: float  # the smaller of the share of each one's marks lying within the slack of the other's (0-1)
    only_original: int  # marks of the first that lie beyond the slack of the second's
    only_suspect: int


def marks_within_pair(original: np.ndarray, suspect: np.ndarray, slack_px: int) -> PairMarks | None:
    """The marks of two colour crops of the same place (cut with one background and threshold), and how far they line
    up. None if the two together have no marks to speak of."""
    both = _marks(original, suspect)
    if both is None:
        return None
    a, b = both
    if not a.any() and not b.any():
        return None
    a_near, b_near = _dilated(a, slack_px), _dilated(b, slack_px)
    a_in_b = float((a & b_near).sum() / a.sum()) if a.any() else 0.0
    b_in_a = float((b & a_near).sum() / b.sum()) if b.any() else 0.0
    return PairMarks(min(a_in_b, b_in_a), int((a & ~b_near).sum()), int((b & ~a_near).sum()))


def marks_agreement(
    original: np.ndarray, suspect: np.ndarray, box: BoundingBox, pad: int = 6, slack_px: int = _AGREEMENT_SLACK_PX
) -> float | None:
    """How closely the marks of the two images coincide inside `box` (plus `pad`): the smaller of the share of each
    one's marks lying within `slack_px` of the other's (1.0 = the same marks). None if either has none there."""
    found = _marks_within(original, suspect, box, pad, slack_px)
    return None if found is None else found.agreement


def evidence(
    original: np.ndarray, suspect: np.ndarray, box: BoundingBox, margin: int, slack_px: int = _AGREEMENT_SLACK_PX
) -> Evidence:
    """What the two images have inside `box` (plus `margin`): their average colours, whether their marks coincide
    (within `slack_px`), and whether the background between the marks is the same colour."""
    x0, y0, x1, y1 = padded_window(box, margin, original.shape)
    a, b = original[y0:y1, x0:x1], suspect[y0:y1, x0:x1]
    if a.size == 0:
        return Evidence(0.0, None)
    extent = min(a.shape[0], a.shape[1])
    delta = _delta_of_means(*_window_means(a, b))
    found = _marks_within(original, suspect, box, margin, slack_px)
    if found is None:
        return Evidence(delta, None, delta, 0.0, extent)
    background = ~found.near_any
    enough_background = int(background.sum()) >= _MIN_BACKGROUND_PX
    background_delta = _delta_of_means(*_window_means(a, b, background)) if enough_background else 0.0
    contrast = 0.0
    if enough_background and int(found.ink.sum()) >= _MIN_BACKGROUND_PX // 4:
        contrast = _delta_of_means(cv2.mean(np.ascontiguousarray(a), mask=found.ink.astype(np.uint8)), cv2.mean(np.ascontiguousarray(a), mask=background.astype(np.uint8)))
    return Evidence(delta, found.agreement, background_delta, contrast, extent)


def is_residue(found: Evidence, config: DetectionConfig, shape_only: bool = False) -> bool:
    """Whether the evidence says the two files have the same content there.

    Where the marks coincide, the letters and lines are the same and only how they are drawn differs: the average
    colour may move a little (`residue_max_ink_colour_delta`: a thin line drawn softer or a shade lighter), but the
    background between them must not (`residue_max_colour_delta`: a tint that changed is a change). Where there
    are no marks to compare (a flat area) the average colour must not differ at all by more than that.

    `shape_only` is a finding about shape (the similarity map): if the marks coincide there is no shape difference, and
    a colour one is for the colour comparison to report, so it is allowed a larger colour difference.
    """
    if found.agreement is None:
        return found.colour_delta < config.residue_max_colour_delta
    # An edge across the window that moved or softened by `residue_edge_shift_px` changes the window's average by that
    # much of the contrast, over the window's width: the stronger the edge (a black letter on light blue) and the
    # narrower the window (a thin line), the further the average can shift although nothing changed.
    shift_allowance = found.contrast * config.residue_edge_shift_px / max(found.extent, 1)
    allowed = max(config.residue_max_ink_colour_delta, shift_allowance)
    if shape_only:
        allowed = max(allowed, config.residue_max_shape_colour_delta)
    return (
        found.agreement >= config.residue_min_agreement
        and found.colour_delta < allowed
        and found.background_delta < config.residue_max_colour_delta
    )


def _rated_by_average(candidate: DiffCandidate, original: np.ndarray, suspect: np.ndarray, config: DetectionConfig) -> DiffCandidate:
    """A colour finding rated by how different the two files look over the finding, not by its most different pixel.

    The peak is the worst pixel of an edge (a hairline of black against white is ~dE 100 whatever changed), which
    made a line drawn a little lighter 'Critical'. The average over the finding's box is the change as seen: a
    recoloured area keeps its full delta, a hairline's is small.
    """
    delta = evidence(original, suspect, candidate.bbox, 0).colour_delta
    if delta < config.residue_max_colour_delta:
        # The colour did not change: what differs is the shape or where it sits (its marks do not coincide), so
        # that is what is said, not 'colour changed (dE 0.3)'.
        return replace(
            candidate,
            type=DiffType.GRAPHICS,
            subtype=DiffSubtype.MOVED,  # (rated like a small shift: Minor, never Major for the size of the box)
            magnitude=0.5,
            confidence=0.5,
            detail="Shape or position differs (colour unchanged)",
        )
    return replace(
        candidate,
        magnitude=delta,
        confidence=min(1.0, delta / (config.color_delta_e_threshold * 4)),
        detail=f"Colour changed (dE {delta:.1f})",
    )


def _is_thin(box: BoundingBox) -> bool:
    """Whether a box is a line (a few pixels across and much longer) rather than a speck the size of a glyph."""
    short, long = sorted((box.width, box.height))
    return short <= _THIN_PX and long >= _THIN_ASPECT * short


def drop_residue(
    candidates: list[DiffCandidate], original: np.ndarray, suspect: np.ndarray, config: DetectionConfig
) -> list[DiffCandidate]:
    """The candidates that are real differences: colour and graphics findings that are only how the two files
    draw the same thing are dropped, and the colour ones that stay are rated by their average change.
    Anything else (barcodes, Braille, a panel-sized difference) is kept as it is.

    `original` and `suspect` are the two whole aligned pages, not the cleared copies that were compared: where an
    outline only one file has leaves a sliver of one page and blank in the other, the artwork is the same on both
    pages, and that sliver is the outline's, not a change.
    """
    kept = []
    dropped = 0
    for candidate in candidates:
        if candidate.type not in _PIXEL_TYPES or candidate.bbox.area > _MAX_CHECKED_AREA_PX:
            kept.append(candidate)
            continue
        # A thin difference (a line one file draws a pixel or two from where the other does) is forgiven one pixel more.
        slack = config.residue_slack_px + (1 if _is_thin(candidate.bbox) else 0)
        found = evidence(original, suspect, candidate.bbox, config.residue_margin_px, slack)
        # A small patch of colour that differs by a shade is what two exports of one artwork do to a solid ink (a colour
        # profile), not a change to a patch of it; a whole panel differing by that much is still a change.
        small_shade = candidate.region and candidate.bbox.area < _SMALL_REGION_PX and found.colour_delta < config.residue_max_ink_colour_delta
        if small_shade or is_residue(found, config, shape_only=candidate.type is DiffType.GRAPHICS):
            dropped += 1
            continue
        kept.append(_rated_by_average(candidate, original, suspect, config) if candidate.type is DiffType.COLOUR else candidate)
    logger.info("residue check finished", extra={"checked": len(candidates), "dropped": dropped})
    return kept
