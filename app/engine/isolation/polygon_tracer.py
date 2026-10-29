"""Click-to-trace dieline polygon extraction (SRS FR-4.1/FR-4.2, Architecture §3.3.2), and the artwork's blocks.

Given a single click near a dieline in a picture, this follows the connected contour it lies on back to the start and
returns the enclosed polygon -- a traversal of the boundary ("follow the border and loop back"), not a segmentation
heuristic. `find_blocks` divides an artwork into the panels its fold and cut lines enclose.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.models.diff import BoundingBox

Point = tuple[float, float]

_MAX_RASTER_CLICK_DISTANCE_PX = 25  # how far a click may be from a contour and still hit it


class NoDielineFoundError(Exception):
    """No closed boundary loop could be traced from the given click point."""


@dataclass
class TracedPolygon:
    points: list[Point]  # ordered closed loop, in the same coordinate space as the input


# ---------------------------------------------------------------------------
# Raster (image) tracing
# ---------------------------------------------------------------------------


def trace_raster_dieline(image: np.ndarray, click_point: tuple[float, float]) -> TracedPolygon:
    """Contour-based trace for raster-only sources.

    `cv2.findContours` already returns an ordered walk of connected boundary
    pixels, so tracing here is "find the contour nearest the click" rather
    than a separate walk implementation.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise NoDielineFoundError("No closed contours found in the image.")

    click = (float(click_point[0]), float(click_point[1]))

    def distance_to_contour(contour: np.ndarray) -> float:
        return abs(cv2.pointPolygonTest(contour, click, True))

    nearest = min(contours, key=distance_to_contour)
    if distance_to_contour(nearest) > _MAX_RASTER_CLICK_DISTANCE_PX:
        raise NoDielineFoundError("No closed contour found near the clicked point.")
    points = [(float(p[0][0]), float(p[0][1])) for p in nearest]
    return TracedPolygon(points=points)


# ---------------------------------------------------------------------------
# Block-wise segmentation: every interior panel, not just one clicked loop
# ---------------------------------------------------------------------------


@dataclass
class ArtworkBlock:
    """One interior panel of an already-lifted artwork, found by tracing every enclosed region its dielines
    form (not just the outer boundary): a folded carton's panels are not always coplanar in a photograph or
    scan, so each can benefit from its own local alignment refinement (`app.engine.alignment.block_alignment`).

    `polygon` is a simplified quadrilateral-ish outline (`cv2.approxPolyDP`), not the raw pixel-level contour:
    a block's own local homography is fit from feature keypoints inside it, not from its edge, so the extra
    precision of the raw contour buys nothing and only costs more points to warp and rasterize later.
    """

    polygon: list[Point]  # ordered closed loop, in the same picture's own pixels
    bbox: BoundingBox


_BLOCK_MIN_AREA_FRACTION = 0.02  # smaller than this is a logo, a line of text, or noise -- not a panel
_BLOCK_MAX_AREA_FRACTION = 0.6  # larger than this is the artwork's own outer boundary (or most of it), not a panel inside it
_BLOCK_MIN_RECTANGULARITY = 0.85  # area / its own minimum-area rotated rectangle: a panel is bounded by straight fold lines
_BLOCK_MAX_VERTICES = 6  # simplified at _BLOCK_APPROX_EPSILON: a panel is a quadrilateral, not a decorative curve's contour
_BLOCK_APPROX_EPSILON = 0.03  # share of the contour's own perimeter: coarse enough that a fold line's corners survive
# but a curved decorative edge (a ribbon, a wave graphic) still needs many more points to approximate -- see
# `_BLOCK_MAX_VERTICES`. Both were raised together from a finer epsilon after a real carton's decorative
# swoosh graphic traced a closed loop that passed rectangularity alone (an irregular blob can still fill a
# large share of its own bounding rotated rectangle) but never simplified down anywhere near four corners.
_BLOCK_MAX_OVERLAP = 0.9  # a second contour this much inside an already-kept block's box is the same panel again
_BLOCK_MIN_MASK_COVERAGE = 0.9  # this much of a candidate must sit inside the given mask to count as one of its panels
_BLOCK_MIN_EDGE_COVERAGE = 0.85  # a panel's own four sides (its min-area rectangle) are real edge pixels this often
_BLOCK_EDGE_SAMPLE_STEP_PX = 6  # a point checked this often along each side, in the picture's own pixels
_BLOCK_EDGE_TOLERANCE_PX = 4  # how far a sampled point may be from a real edge pixel and still count as confirmed


def _rect_edge_coverage(rect: np.ndarray, edges: np.ndarray) -> float:
    """What share of a candidate rectangle's own four sides (`rect`, its four corners) sit on a real edge pixel
    in `edges`, sampled every few pixels along each side.

    Falls back on this, rather than requiring the whole traced contour to itself be near-rectangular, when
    artwork drawn right up to a panel's edge (a logo, an illustration) merges into the same contour: the
    contour's enclosed shape is then corrupted by that content, but the panel's own straight sides are still
    there in the edge map underneath it. A decorative curve's own bounding rectangle fails this the same way it
    fails rectangularity -- its edge does not run along the rectangle's straight sides -- so this does not
    reopen the false positive `_BLOCK_MAX_VERTICES` was added for.
    """
    height, width = edges.shape[:2]
    confirmed = total = 0
    for start, end in zip(rect, np.roll(rect, -1, axis=0)):
        length = float(np.linalg.norm(end - start))
        steps = max(2, int(length / _BLOCK_EDGE_SAMPLE_STEP_PX))
        for t in np.linspace(0.0, 1.0, steps):
            x, y = (start + (end - start) * t).round().astype(int)
            y0, y1 = max(0, y - _BLOCK_EDGE_TOLERANCE_PX), min(height, y + _BLOCK_EDGE_TOLERANCE_PX + 1)
            x0, x1 = max(0, x - _BLOCK_EDGE_TOLERANCE_PX), min(width, x + _BLOCK_EDGE_TOLERANCE_PX + 1)
            total += 1
            if edges[y0:y1, x0:x1].any():
                confirmed += 1
    return confirmed / total if total else 0.0


def _auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Canny edges with thresholds set from `gray`'s own median (the standard auto-Canny rule), rather than
    fixed numbers that suit one picture's contrast and not another's."""
    median = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(gray, lower, upper)


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
    ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def find_blocks(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    min_area_fraction: float = _BLOCK_MIN_AREA_FRACTION,
    max_area_fraction: float = _BLOCK_MAX_AREA_FRACTION,
    min_rectangularity: float = _BLOCK_MIN_RECTANGULARITY,
) -> list[ArtworkBlock]:
    """Every interior panel `image`'s dielines enclose (empty if there are none worth the name).

    Works from the rendered picture alone -- Canny edges, the same starting point `trace_raster_dieline` uses
    for a raster source -- rather than a PDF's own vector paths, so it applies equally to a scan and needs
    nothing threaded in from the file loader. `mask`, if given, restricts the search to where the artwork
    actually is (its own outer boundary, e.g.), so a page's crop marks or a neighbouring artwork are not
    mistaken for a panel.
    """
    # Edge detection runs on the picture exactly as it is, `mask` unapplied: a fold line that reaches the
    # artwork's own outer edge (an ordinary thing for one to do) has its last pixel or two masked away if the
    # mask is used to blank the picture, or clipped from the edge map before tracing -- either breaks that
    # fold line's contour from closing, and a panel it bounds is lost entirely. `mask` is applied afterwards
    # instead, as a coverage check on each already-traced candidate below.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    # Local contrast is stretched first (a fold line printed in a light colour, or on a light background, can
    # otherwise sit below a fixed threshold everywhere), and Canny's own thresholds are set from the picture's
    # own contrast (its median, the classic auto-Canny rule) rather than fixed numbers tuned to one sample.
    enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    edges = _auto_canny(enhanced)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # A panel is a fraction of the ARTWORK, not of the whole page it may have been lifted from (an
    # auto-found or manually cropped/lifted artwork is often a small share of the page it sits on).
    picture_area = float(np.count_nonzero(mask)) if mask is not None else float(image.shape[0] * image.shape[1])
    blocks: list[ArtworkBlock] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if not (min_area_fraction * picture_area <= area <= max_area_fraction * picture_area):
            continue
        rect = cv2.boxPoints(cv2.minAreaRect(contour))
        rect_area = cv2.contourArea(rect)
        # A panel is a quadrilateral: a decorative graphic's edge (a ribbon, a wave) can still fill a large
        # enough share of its own bounding rotated rectangle to pass rectangularity alone, but it needs far
        # more than a handful of points to approximate at this coarse a tolerance, where a fold line's own
        # corners survive untouched.
        approx = cv2.approxPolyDP(contour, _BLOCK_APPROX_EPSILON * cv2.arcLength(contour, True), True)
        clean_shape = rect_area > 0 and area / rect_area >= min_rectangularity and len(approx) <= _BLOCK_MAX_VERTICES
        if clean_shape:
            polygon = [(float(p[0][0]), float(p[0][1])) for p in approx]
            x0, y0, w, h = cv2.boundingRect(contour)
            bbox = BoundingBox(float(x0), float(y0), float(x0 + w), float(y0 + h))
        else:
            # The contour itself may be a real panel's four sides with artwork drawn right up to one of them
            # (a logo, an illustration) merged in, corrupting its shape: the sides are checked directly against
            # the edge map instead of asking the whole contour to look like a rectangle, and the panel is taken
            # to be the clean rectangle those sides trace, not the corrupted shape that also picked up the
            # artwork touching it.
            if rect_area <= 0 or _rect_edge_coverage(rect, edges) < _BLOCK_MIN_EDGE_COVERAGE:
                continue
            polygon = [(float(x), float(y)) for x, y in rect]
            bbox = BoundingBox(float(rect[:, 0].min()), float(rect[:, 1].min()), float(rect[:, 0].max()), float(rect[:, 1].max()))
        if mask is not None and not _mostly_inside(contour, mask, area):
            continue  # mostly outside where the artwork actually is -- not one of its own panels
        if any(_iou(bbox, existing.bbox) > _BLOCK_MAX_OVERLAP for existing in blocks):
            continue  # the same panel, found again as a second (inner/outer) edge of its own dilated line
        blocks.append(ArtworkBlock(polygon=polygon, bbox=bbox))
    return blocks


def _mostly_inside(contour: np.ndarray, mask: np.ndarray, area: float) -> bool:
    filled = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
    covered = np.count_nonzero((filled > 0) & mask)
    return covered >= _BLOCK_MIN_MASK_COVERAGE * area
