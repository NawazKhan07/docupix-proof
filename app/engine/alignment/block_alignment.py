"""Block-wise alignment: every block of the artwork aligned on its own, the way cropping just that block in both
files and aligning the two crops would (Architecture §3.3.3 extended).

One transform for a whole sheet cannot fit two cartons whose panels are different sizes (a 500 mg carton's front
panel 148 mm wide where the 200 mg one's is 118 mm), a fold not quite flat in a photograph, or a panel printed a
little out of register: whatever it fits well, it leaves the rest out, by hundreds of pixels. So each block (the
carton's own panels and flaps, `block_grid.divide_into_blocks`) finds its counterpart in the other file and takes
its own transform:

* **its counterpart is looked for where it really is**, not only where the whole-sheet transform predicts: in the
  other file's own blocks nearest that prediction (or failing those, a wide window round it) -- and among near-equal matches
  the one nearer the prediction wins, so a carton's two identical front panels are never swapped;
* **its own transform comes from the printing inside it** (keypoints, RANSAC, the quality gate), and is taken only
  where it beats the whole-sheet transform on the block's own points by a clear margin -- where the whole-sheet one
  already fits (two files the same size, the ordinary case) the block keeps it, and no seam is drawn for nothing;
* **a block with too little printing to match** (a plain flap) is stretched onto its counterpart's outline, or
  failing that shifted by phase correlation, each only if it measurably brings the two closer together.

Blocks are put back together into one picture (`composite_warp`), feathered where they meet.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import cv2
import numpy as np

from app.config import AlignmentConfig
from app.engine.alignment import quality_gate
from app.engine.alignment.feature_matcher import FeatureMatches
from app.engine.alignment.homography import apply_transform, estimate_homography, warp_to_reference
from app.engine.isolation.polygon_tracer import ArtworkBlock
from app.models.diff import BoundingBox

Point = tuple[float, float]

_MAX_THREADS = 8  # blocks aligned at once
_SEAM_FEATHER_PX = 12  # blend width where a block's own warp meets its neighbours': see `composite_warp`
_PHASE_MIN_WINDOW_PX = 32  # a phase-correlation window smaller than this either way gives too unstable a shift to trust
_COUNTERPART_PAD_PX = 24  # a counterpart block is searched a little past its own outline
_NEAR_EQUAL_INLIERS = 0.85  # matches within this share of the best one's inliers are near-equal: the nearer wins
_MIN_AREA_RATIO = 0.4  # a counterpart block is between this and its inverse of the block's own size
_WINDOW_GAIN = 0.97  # a block's own transform must leave it at most this share as different as the whole-sheet one
_STRONG_WINDOW_GAIN = 0.5  # a part's own transform this much closer in pixels than the whole-sheet one is right
# Where the edges land (see `_EdgeScorer`): what says whether a block lines up, whatever the tone of its paper.
_EDGE_GAIN = 0.85  # a block's own transform must bring its edges at least this much closer than the whole-sheet one...
_EDGE_MIN_GAIN_PX = 0.5  # ...and by at least this many pixels
_EDGE_PIXEL_SLACK = 1.1  # ...without leaving its pixels more than this share more different
_EDGE_CAP_PX = 15.0  # an edge further than this from any edge of the other file counts as this far (it is simply not there)
_EDGE_SCORE_MAX_PX = 1600  # edges are compared on a copy of the block no longer than this
_EDGE_BAND_PX = 12  # ...and in a band this wide round it (a flap's die line is its edge)
_REFINE_ROUNDS = 12  # rounds of iterative closest point refining a texture-poor block onto its edges
_REFINE_MIN_POINTS = 30
_KEYPOINT_INSET_PX = 12  # a block's keypoints are taken this far inside its outline
_MAX_FIT_SHEAR = 0.08  # a block's own fit shears against the whole sheet's by no more than this (a panel is a rectangle)
_REFINE_MAX_SCALE_CHANGE = 0.05  # a refinement rescales by no more than this share...
_REFINE_MAX_SHEAR = 0.03  # ...nor shears by more than this
_REFINE_MIN_GAIN_PX = 0.1  # a refinement is kept if it brings the edges at least this much closer
_REFINE_MAX_POINTS = 3000  # a round fits its correction to at most this many edge points (evenly spread)
_OUTLINE_CANDIDATES = 2  # the nearest counterparts whose outlines are tried
_REFINE_MIN_SCALE = 0.9  # one round may not rescale by more than this (or its inverse): a slide, not a leap
_MIN_SHIFT_PX = 1.0  # a phase-correlation shift smaller than this is not worth a block of its own
_MAX_DISPLACEMENT_SHARE = 1.0  # a counterpart at most this many times the block's own size from where it is predicted
_SECOND_GROUP_SHARE = 0.2  # a second group of matches moving differently is at least this share of the first...
_SECOND_GROUP_EXTENT = 0.15  # ...spread over at least this share of the block's width and of its height...
_SECOND_GROUP_MIN_OFFSET_PX = 3.0  # ...and puts its own points at least this far from where the first fit would
_MIN_PANEL_SIZE_CHANGE_PX = 10.0  # counterpart panels differing in size by less than this are the same size
_MAX_GROUPS = 3  # at most this many differently-moving groups in one block
_SHIFT_TRIALS = 300  # candidate shifts tried for each further group
_SHIFT_TOLERANCE_PX = 4.0  # points within this of the second group's shift belong to it
_PART_CELL_PX = 16  # a block is divided between its two groups on a grid of cells this size
_PART_MIN_SHARE = 0.01  # a piece smaller than this share of the block is not a part of its own


@dataclass
class ArtworkBlockAlignment:
    """One block, with the transform to use for it -- its own local fit if one could be trusted, else the
    whole artwork's (so a caller need not treat `locally_aligned=False` as a special case when warping)."""

    block: ArtworkBlock
    transform: np.ndarray
    locally_aligned: bool  # False: no local fit cleared the quality gate, so `transform` is the global one


def _polygon_mask(
    shape: tuple[int, int], polygon: list[Point], dilate_px: int = 0, scale: float = 1.0, shift: Point = (0.0, 0.0)
) -> np.ndarray:
    """`polygon`'s mask at `shape`. `scale`/`shift` map it into another frame first (a device-pixel tile at some
    zoom, e.g.): `device = polygon_point * scale - shift`, the same convention `tile_renderer` uses."""
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.array([(round(x * scale - shift[0]), round(y * scale - shift[1])) for x, y in polygon], dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    if dilate_px > 0:
        mask = cv2.dilate(mask, np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8))
    return mask.astype(bool)


def _block_config(config: AlignmentConfig) -> AlignmentConfig:
    """A block is smaller than the whole artwork and credibly has fewer of its own keypoints: its own, more
    forgiving match/inlier counts stand in for the whole-artwork ones everything else here reads from."""
    return replace(config, min_match_count=config.block_min_match_count, min_inlier_count=config.block_min_inlier_count)


@dataclass
class _Keypoints:
    points: np.ndarray  # (N, 2) float32, in the whole picture's own pixels
    descriptors: np.ndarray | None


@dataclass
class _Fit:
    transform: np.ndarray  # suspect pixels -> original pixels
    inliers: int
    residual: float  # on its own inlier points
    global_residual: float  # the whole-sheet transform's, on the very same points
    displacement: float  # how far from the whole-sheet transform's prediction it finds the block, in the suspect
    matches: FeatureMatches | None = None  # every match it was fitted from
    inlier_mask: np.ndarray | None = None  # which of `matches` it fits
    counterpart: ArtworkBlock | None = None  # the other file's block it was found in, if it was found in one


def _detector(config: AlignmentConfig):
    return cv2.AKAZE_create() if config.feature_detector.upper() == "AKAZE" else cv2.ORB_create(nfeatures=config.orb_features)


def _keypoints(
    gray: np.ndarray, box: tuple[float, float, float, float], config: AlignmentConfig, polygon: list[Point] | None = None
) -> _Keypoints:
    """Keypoints and descriptors inside `box` (x0, y0, x1, y1), and inside `polygon` if one is given -- found on just
    that crop of the picture, never the whole page."""
    height, width = gray.shape[:2]
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    x1, y1 = min(width, int(np.ceil(box[2]))), min(height, int(np.ceil(box[3])))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return _Keypoints(np.empty((0, 2), np.float32), None)
    mask = None
    if polygon is not None:
        mask = _polygon_mask((y1 - y0, x1 - x0), polygon, shift=(x0, y0)).astype(np.uint8) * 255
        # Not on the block's own outline: a fold is shared with the next panel (and a die line with the paper beyond it),
        # so where it went says nothing of where THIS panel's printing went.
        inset = 2 * _KEYPOINT_INSET_PX + 1
        mask = cv2.erode(mask, np.ones((inset, inset), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    found, descriptors = _detector(config).detectAndCompute(gray[y0:y1, x0:x1], mask)
    points = np.array([k.pt for k in found], np.float32).reshape(-1, 2) + np.array([x0, y0], np.float32)
    return _Keypoints(points, descriptors)


def _matches(suspect: _Keypoints, original: _Keypoints, config: AlignmentConfig) -> FeatureMatches:
    empty = FeatureMatches(np.empty((0, 2), np.float32), np.empty((0, 2), np.float32))
    if suspect.descriptors is None or original.descriptors is None or len(suspect.points) < 2 or len(original.points) < 2:
        return empty
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(suspect.descriptors, original.descriptors, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < config.ratio_test * pair[1].distance]
    if not good:
        return empty
    return FeatureMatches(
        src_points=suspect.points[[m.queryIdx for m in good]], dst_points=original.points[[m.trainIdx for m in good]]
    )


def _fit(matches: FeatureMatches, config: AlignmentConfig, global_transform: np.ndarray, centre: np.ndarray) -> _Fit | None:
    """The block's own transform from these matches, if one clears the quality gate."""
    estimate = estimate_homography(matches, config)
    if estimate is None:
        return None
    transform, inliers = estimate
    if quality_gate.rejection_reason(transform, int(inliers.sum()), config) is not None:
        return None
    if _shear(transform, global_transform) > _MAX_FIT_SHEAR:
        # A panel is a rectangle: against the whole sheet it may be bigger or smaller (another carton size), or turned
        # a little (a scan), but not sheared -- keypoints agreeing on a sheared fit have matched the wrong things
        # (repeated lines of text slid past one another).
        return None
    src, dst = matches.src_points[inliers], matches.dst_points[inliers]
    try:
        found_at = apply_transform(np.linalg.inv(transform), centre)[0]
        predicted_at = apply_transform(np.linalg.inv(global_transform), centre)[0]
    except np.linalg.LinAlgError:
        return None
    return _Fit(
        transform=transform,
        inliers=int(inliers.sum()),
        residual=quality_gate.mean_residual_px(transform, src, dst),
        global_residual=quality_gate.mean_residual_px(global_transform, src, dst),
        displacement=float(np.linalg.norm(found_at - predicted_at)),
        matches=matches,
        inlier_mask=inliers,
    )


def _split(
    block: ArtworkBlock, first: _Fit, config: AlignmentConfig, global_transform: np.ndarray, shape: tuple[int, int]
) -> list[tuple[ArtworkBlock, _Fit]]:
    """The block as parts that move differently between the files, each with its own fit (just the block itself,
    with `first`, if it moves as one).

    Two cartons of different widths put a panel's left-aligned text and its right-aligned barcode (or logo) at
    different places relative to each other, and a taller panel spaces its lower lines further down: one transform
    fits one group and leaves the rest out. So what the first fit leaves over is looked at again for further
    groups, each the first fit plus one shift (content set against another edge moves all together) -- each
    substantial and spread over a real share of the block, so a single small element that moved stays a finding
    rather than something aligned away. Every part of the block then goes with whichever group's own points are
    nearest it (on a coarse grid, smoothed), so a title stays with the text it is set with and a logo with the
    edge it is set against.
    """
    matches, inliers = first.matches, first.inlier_mask
    if matches is None or inliers is None or first.counterpart is None:
        return [(block, first)]
    # How much bigger (or smaller) the counterpart panel is, seen through the first fit: content set against its
    # far edge can have moved by up to that much, along that axis, and by nothing more. Two panels the same size
    # (the ordinary case) leave no room at all -- lines that moved there moved because the text changed (a line
    # removed, the rest reflowed), which is a difference to report, not something to align away.
    outline = apply_transform(first.transform, _box_corners(first.counterpart).astype(np.float64))
    room = np.abs((outline.max(axis=0) - outline.min(axis=0)) - np.array([block.bbox.width, block.bbox.height]))
    if room.max() < _MIN_PANEL_SIZE_CHANGE_PX:
        return [(block, first)]
    src, dst = matches.src_points.astype(np.float64), matches.dst_points.astype(np.float64)
    offsets = dst - apply_transform(first.transform, src)
    left = ~inliers
    groups: list[tuple[np.ndarray, _Fit]] = [(dst[inliers], first)]
    rng = np.random.default_rng(0)
    while len(groups) < _MAX_GROUPS and left.sum() >= 3 * config.min_inlier_count:
        candidates = np.flatnonzero(left)
        trials = offsets[rng.choice(candidates, size=min(len(candidates), _SHIFT_TRIALS), replace=False)]
        best = max((left & (np.linalg.norm(offsets - trial, axis=1) <= _SHIFT_TOLERANCE_PX) for trial in trials), key=lambda m: int(m.sum()))
        shift = np.median(offsets[best], axis=0)
        members = left & (np.linalg.norm(offsets - shift, axis=1) <= _SHIFT_TOLERANCE_PX)
        if members.sum() < max(3 * config.min_inlier_count, _SECOND_GROUP_SHARE * first.inliers):
            break
        left &= ~members
        points = dst[members]
        extent = points.max(axis=0) - points.min(axis=0)
        if np.any(np.abs(shift) > room + _SHIFT_TOLERANCE_PX * 2) or float(np.linalg.norm(shift)) < _SECOND_GROUP_MIN_OFFSET_PX or (
            extent[0] < _SECOND_GROUP_EXTENT * block.bbox.width and extent[1] < _SECOND_GROUP_EXTENT * block.bbox.height
        ):
            continue  # the same fit again, or one small element that moved: a finding, not a part
        transform = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]], [0.0, 0.0, 1.0]]) @ first.transform
        groups.append((points, _Fit(
            transform=transform,
            inliers=int(members.sum()),
            residual=quality_gate.mean_residual_px(transform, src[members], points),
            global_residual=quality_gate.mean_residual_px(global_transform, src[members], points),
            displacement=first.displacement,
        )))
    if len(groups) == 1:
        return [(block, first)]

    x0, y0 = int(block.bbox.x0), int(block.bbox.y0)
    columns = max(1, int(np.ceil(block.bbox.width / _PART_CELL_PX)))
    rows = max(1, int(np.ceil(block.bbox.height / _PART_CELL_PX)))
    distances = []
    for points, _fit_of_group in groups:
        grid = np.full((rows, columns), 255, np.uint8)
        cells = np.clip(((points - [x0, y0]) / _PART_CELL_PX).astype(int), 0, [columns - 1, rows - 1])
        grid[cells[:, 1], cells[:, 0]] = 0
        distances.append(cv2.distanceTransform(grid, cv2.DIST_L2, 3))
    labels = np.argmin(np.stack(distances), axis=0).astype(np.uint8)
    if min(rows, columns) >= 5:
        labels = cv2.medianBlur(labels, 5)
    # Worked out within the block's own box, in its own coordinates, then placed back.
    bx0, by0, bx1, by1 = _region(shape, block, 0)
    full = cv2.resize(labels, (columns * _PART_CELL_PX, rows * _PART_CELL_PX), interpolation=cv2.INTER_NEAREST)
    local = np.full((by1 - by0, bx1 - bx0), 255, np.uint8)
    oy, ox = y0 - by0, x0 - bx0
    h, w = min(full.shape[0], local.shape[0] - oy), min(full.shape[1], local.shape[1] - ox)
    local[oy : oy + h, ox : ox + w] = full[:h, :w]
    whole = _polygon_mask(local.shape, block.polygon, shift=(bx0, by0))

    parts = []
    minimum = _PART_MIN_SHARE * float(whole.sum())
    for index, (_points, fit) in enumerate(groups):
        count, components, stats, _ = cv2.connectedComponentsWithStats((whole & (local == index)).astype(np.uint8), connectivity=8)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < minimum:
                continue
            contours, _ = cv2.findContours((components == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(contour)
            polygon = [(float(p[0][0] + bx0), float(p[0][1] + by0)) for p in cv2.approxPolyDP(contour, 2.0, True)]
            if len(polygon) >= 3:
                parts.append((ArtworkBlock(polygon=polygon, bbox=BoundingBox(float(x + bx0), float(y + by0), float(x + bx0 + bw), float(y + by0 + bh))), fit))
    return parts if any(fit is not first for _part, fit in parts) else [(block, first)]


def _box_corners(block: ArtworkBlock) -> np.ndarray:
    b = block.bbox
    return np.float32([[b.x0, b.y0], [b.x1, b.y0], [b.x1, b.y1], [b.x0, b.y1]])


def _counterparts(
    block: ArtworkBlock, suspect_blocks: list[ArtworkBlock], inverse_transform: np.ndarray, count: int
) -> list[int]:
    """The other file's blocks nearest where the whole-sheet transform predicts this one, of a plausible size."""
    predicted = apply_transform(inverse_transform, np.array(block.polygon, dtype=np.float64))
    predicted_centre = predicted.mean(axis=0)
    predicted_area = max(1.0, abs(cv2.contourArea(predicted.astype(np.float32))))
    ranked = []
    for index, other in enumerate(suspect_blocks):
        ratio = other.bbox.area / predicted_area
        if _MIN_AREA_RATIO <= ratio <= 1.0 / _MIN_AREA_RATIO:
            ranked.append((float(np.linalg.norm(np.array(other.bbox.center) - predicted_centre)), index))
    return [index for _distance, index in sorted(ranked)[:count]]


def _darkest(image: np.ndarray) -> np.ndarray:
    """Each pixel's darkest channel (ink of any colour, however light it looks, is dark in one)."""
    if image.ndim != 3:
        return image
    channels = cv2.split(np.ascontiguousarray(image[:, :, :3]))
    return cv2.min(cv2.min(channels[0], channels[1]), channels[2])


class _EdgeScorer:
    """How far apart the two files' edges lie inside one block, with the suspect warped by a given transform (pixels;
    lower is better): the symmetric chamfer distance between the edges of the original's block and those of the
    warped suspect, each capped at `_EDGE_CAP_PX`.

    Edges are found on each picture's DARKEST channel, not its grey: a die line is often a light tint (pale red, cyan)
    that is nearly paper-white in grey but dark in one channel. The block's mean grey difference cannot tell a good fit
    from a poor one where the block is mostly paper -- a flap
    carries a die line and nothing else, and a line off by 20 px moves the mean by a fraction of a grey level -- nor on
    a scan, whose paper tone, shading and background dwarf any misregistration. Edges are what alignment is about,
    and where they land does not depend on either.
    """

    def __init__(self, original_gray: np.ndarray, suspect_gray: np.ndarray, block: ArtworkBlock, interior: bool = False):
        """`interior`: score only the edges well inside the block (its printing), not its outline and the band round it
        (which a flap needs, its die line being all there is, but which a panel's printing moves within)."""
        height, width = original_gray.shape
        self._suspect = suspect_gray
        self._last: tuple[bytes, np.ndarray] | None = None  # the transform whose suspect edges were found last, and them
        self._scores: dict[bytes, float] = {}  # each transform scored so far (a block's candidates are scored again and again)
        self.valid = min(block.bbox.width, block.bbox.height) >= 16
        self.judges = False
        if not self.valid:
            return
        # The block and a narrow band round it: a flap's own die line runs along the block's edge (it IS the flap), and
        # must be scored too. The window is widened by that band, so the band is inside it.
        band = _EDGE_BAND_PX
        self.x0, self.y0 = max(0, int(block.bbox.x0) - band), max(0, int(block.bbox.y0) - band)
        x1, y1 = min(width, int(np.ceil(block.bbox.x1)) + band), min(height, int(np.ceil(block.bbox.y1)) + band)
        self.size = (x1 - self.x0, y1 - self.y0)
        self.scale = min(1.0, _EDGE_SCORE_MAX_PX / max(1, *self.size))
        self.small = (max(1, round(self.size[0] * self.scale)), max(1, round(self.size[1] * self.scale)))
        inside = _polygon_mask((self.size[1], self.size[0]), block.polygon, shift=(self.x0, self.y0)).astype(np.uint8)
        grow = 2 * max(1, round(band * self.scale)) + 1
        self._inside = cv2.dilate(cv2.resize(inside, self.small, interpolation=cv2.INTER_NEAREST), np.ones((grow, grow), np.uint8)) > 0
        self._edges = self._edges_of(original_gray[self.y0 : y1, self.x0 : x1])
        # Whether the block has edges of its own to judge a fit by, well inside its outline (the outline itself is often
        # just where the neighbouring panel starts, and does not move with what is printed inside it).
        own = cv2.erode(cv2.resize(inside, self.small, interpolation=cv2.INTER_NEAREST), np.ones((grow, grow), np.uint8)) > 0
        self.judges = bool((self._edges & own).any())
        if interior:
            self._inside = own
            self._edges &= own
        self._distance, self._labels = cv2.distanceTransformWithLabels(
            (~self._edges).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
        )
        self._edge_ys, self._edge_xs = np.nonzero(self._edges)  # (labels number the edge pixels in raster order, from 1)
        self._to_window = np.array(
            [[self.scale, 0.0, -self.x0 * self.scale], [0.0, self.scale, -self.y0 * self.scale], [0.0, 0.0, 1.0]]
        )

    def _edges_of(self, window: np.ndarray) -> np.ndarray:
        return self._edges_of_small(cv2.resize(window, self.small, interpolation=cv2.INTER_AREA) if self.scale < 1.0 else window)

    def _suspect_edges(self, transform: np.ndarray) -> np.ndarray:
        # A refinement scores a transform and then starts its next round from it: the same edges, found once.
        key = np.asarray(transform, np.float64).tobytes()
        if self._last is not None and self._last[0] == key:
            return self._last[1]
        warped = cv2.warpPerspective(self._suspect, self._to_window @ transform, self.small, flags=cv2.INTER_LINEAR, borderValue=255)
        edges = self._edges_of_small(warped)
        self._last = (key, edges)
        return edges

    def __call__(self, transform: np.ndarray) -> float:
        if not self.valid:
            return float("inf")
        key = np.asarray(transform, np.float64).tobytes()
        if key in self._scores:
            return self._scores[key]
        edges = self._suspect_edges(transform)
        if not edges.any() or not self._edges.any():
            return float("inf")
        cap = _EDGE_CAP_PX * self.scale
        to_suspect = np.minimum(self._distance[edges], cap).mean()
        suspect_distance = cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 3)
        to_original = np.minimum(suspect_distance[self._edges], cap).mean()
        self._scores[key] = score = float((to_suspect + to_original) / 2.0 / self.scale)
        return score

    def _edges_of_small(self, small: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(small, (0, 0), 1.2)
        return (cv2.Canny(blurred, 20, 60) > 0) & self._inside

    def refine(self, transform: np.ndarray) -> tuple[np.ndarray, float]:
        """`transform` refined so the suspect's edges land on the original's (iterative closest point, an affine
        correction each round), and its score: whichever round scored best, the start included.

        For a block with too little printing for keypoints -- a flap, whose die line is nearly all there is -- an
        outline stretch puts it roughly in place, and this puts its lines on the original's."""
        best, best_score = transform, self(transform)
        if not self.valid or not np.isfinite(best_score) or len(self._edge_xs) < _REFINE_MIN_POINTS:
            return best, best_score
        from_window = np.linalg.inv(self._to_window)
        cap = _EDGE_CAP_PX * self.scale
        current = transform
        for _round in range(_REFINE_ROUNDS):
            ys, xs = np.nonzero(self._suspect_edges(current))
            near = self._distance[ys, xs] < cap
            if near.sum() < _REFINE_MIN_POINTS:
                break
            ys, xs = ys[near], xs[near]
            if len(xs) > _REFINE_MAX_POINTS:
                keep = np.linspace(0, len(xs) - 1, _REFINE_MAX_POINTS).astype(int)
                ys, xs = ys[keep], xs[keep]
            nearest = self._labels[ys, xs] - 1
            source = np.column_stack([xs, ys]).astype(np.float32)
            target = np.column_stack([self._edge_xs[nearest], self._edge_ys[nearest]]).astype(np.float32)
            step, _inliers = cv2.estimateAffine2D(source, target, method=cv2.LMEDS)
            if step is None:
                break
            scale = np.sqrt(abs(np.linalg.det(step[:, :2])))
            if not _REFINE_MIN_SCALE <= scale <= 1.0 / _REFINE_MIN_SCALE:
                break
            current = from_window @ np.vstack([step, [0.0, 0.0, 1.0]]) @ self._to_window @ current
            if not _small_correction(current, transform):
                break  # a polish, never a new fit: a slide along repeated lines into a stretch or a shear is not one
            score = self(current)
            if score < best_score - 0.01:
                best, best_score = current, score
            elif score > best_score + 0.5:
                break  # drifting away: stop
        return best, best_score


def _shear(transform: np.ndarray, reference: np.ndarray) -> float:
    """How sheared `transform` is against `reference` (0 for a pure scale and turn, whatever the scale)."""
    try:
        change = (transform @ np.linalg.inv(reference))[:2, :2]
    except np.linalg.LinAlgError:
        return float("inf")
    # a scale-and-turn has columns at right angles: the cosine of the angle between them is the shear
    first, second = change[:, 0], change[:, 1]
    norms = float(np.linalg.norm(first) * np.linalg.norm(second))
    return abs(float(first @ second)) / norms if norms > 0 else float("inf")


def _small_correction(transform: np.ndarray, start: np.ndarray) -> bool:
    """Whether `transform` is within a small correction of `start` (their linear parts, compared at the same place):
    scaled by no more than `_REFINE_MAX_SCALE_CHANGE` either way, sheared by no more than `_REFINE_MAX_SHEAR`."""
    try:
        change = (transform @ np.linalg.inv(start))[:2, :2]
    except np.linalg.LinAlgError:
        return False
    singular = np.linalg.svd(change, compute_uv=False)
    return bool(np.all(np.abs(singular - 1.0) <= _REFINE_MAX_SCALE_CHANGE) and _shear(transform, start) <= _REFINE_MAX_SHEAR)


def _outlines(block: ArtworkBlock, nearest: list[int], suspect_blocks: list[ArtworkBlock]) -> list[np.ndarray]:
    """Each of these counterpart blocks' outlines stretched onto this block's (suspect pixels -> original pixels)."""
    return [
        cv2.getPerspectiveTransform(_box_corners(suspect_blocks[index]), _box_corners(block)).astype(np.float64)
        for index in nearest
    ]


def _outline_on_edges(
    edges: "_EdgeScorer",
    block: ArtworkBlock,
    nearest: list[int],
    suspect_blocks: list[ArtworkBlock],
    global_transform: np.ndarray,
    extra: list[np.ndarray] | None = None,
) -> tuple[np.ndarray | None, float]:
    """The best of the nearest counterparts' outlines stretched onto this block (and the whole-sheet fit, and `extra`),
    each refined on the edges: (transform, its edge score), or (None, inf) if there are no edges to go by."""
    best, best_edges = None, float("inf")
    if not edges.judges:
        return best, best_edges
    for start in _outlines(block, nearest[:_OUTLINE_CANDIDATES], suspect_blocks) + [global_transform] + list(extra or []):
        refined, score = edges.refine(start)
        if score < best_edges:
            best, best_edges = refined, score
    return best, best_edges


def _edges_closer(own: float, whole: float, own_pixels: float, whole_pixels: float) -> bool:
    """Whether a block's own transform lines its edges up clearly better than the whole-sheet one, its pixels no worse."""
    return (
        np.isfinite(own)
        and own <= _EDGE_GAIN * whole
        and whole - own >= _EDGE_MIN_GAIN_PX
        and own_pixels <= _EDGE_PIXEL_SLACK * whole_pixels
    )


def _window_difference(
    original_gray: np.ndarray, suspect_gray: np.ndarray, block: ArtworkBlock, transform: np.ndarray
) -> float:
    """How different the block still is (mean grey difference inside it) with the suspect warped by `transform`."""
    height, width = original_gray.shape
    x0, y0 = max(0, int(block.bbox.x0)), max(0, int(block.bbox.y0))
    x1, y1 = min(width, int(np.ceil(block.bbox.x1))), min(height, int(np.ceil(block.bbox.y1)))
    if x1 <= x0 or y1 <= y0:
        return float("inf")
    to_window = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]]) @ transform
    warped = cv2.warpPerspective(suspect_gray, to_window, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderValue=255)
    inside = _polygon_mask((y1 - y0, x1 - x0), block.polygon, shift=(x0, y0))
    if not inside.any():
        return float("inf")
    return float(np.abs(original_gray[y0:y1, x0:x1].astype(np.float32) - warped.astype(np.float32))[inside].mean())


def align_blocks(
    original_image: np.ndarray,
    suspect_image: np.ndarray,
    blocks: list[ArtworkBlock],
    global_transform: np.ndarray,
    config: AlignmentConfig,
    suspect_blocks: list[ArtworkBlock] | None = None,
) -> list[ArtworkBlockAlignment]:
    """Align each of `blocks` (in `original_image`'s own frame) on its own, against `suspect_image`.

    `global_transform` maps suspect pixels -> original pixels (the whole-artwork alignment already settled): it says
    roughly where to look, and it is what a block keeps if its own fit is no better. `suspect_blocks` (the suspect's
    own blocks, in its own frame) are where each block's counterpart is looked for first; without them, only a wide
    window round the predicted place is searched.
    """
    try:
        inverse_transform = np.linalg.inv(global_transform)
    except np.linalg.LinAlgError:
        return [ArtworkBlockAlignment(block, global_transform, False) for block in blocks]

    block_config = _block_config(config)
    suspect_blocks = suspect_blocks or []
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY) if original_image.ndim == 3 else original_image
    suspect_gray = cv2.cvtColor(suspect_image, cv2.COLOR_RGB2GRAY) if suspect_image.ndim == 3 else suspect_image
    original_dark, suspect_dark = _darkest(original_image), _darkest(suspect_image)
    nearest_of = [_counterparts(block, suspect_blocks, inverse_transform, config.block_candidates) for block in blocks]

    def counterpart_keypoints(index: int) -> _Keypoints:
        b = suspect_blocks[index].bbox
        pad = _COUNTERPART_PAD_PX
        return _keypoints(suspect_gray, (b.x0 - pad, b.y0 - pad, b.x1 + pad, b.y1 + pad), block_config)

    def align_one(block: ArtworkBlock, nearest: list[int]) -> list[ArtworkBlockAlignment]:
        results: list[ArtworkBlockAlignment] = []
        centre = np.array([block.bbox.center], dtype=np.float64)
        own = _keypoints(original_gray, (block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1), block_config, block.polygon)

        # Where to look: the other file's own blocks nearest the prediction, else a wide window round it.
        predicted = apply_transform(inverse_transform, np.array(block.polygon, dtype=np.float64))
        margin = config.block_search_margin_share * max(block.bbox.width, block.bbox.height)
        (px0, py0), (px1, py1) = predicted.min(axis=0), predicted.max(axis=0)
        reach = _MAX_DISPLACEMENT_SHARE * max(block.bbox.width, block.bbox.height)

        def search(found: _Keypoints, other: ArtworkBlock | None) -> _Fit | None:
            fit = _fit(_matches(found, own, block_config), block_config, global_transform, centre)
            # A counterpart further from where the whole-sheet transform puts it than the block is big is another
            # copy of it (a sheet of repeated units, a carton's second front panel), not this block moved.
            if fit is None or fit.displacement > reach:
                return None
            fit.counterpart = other
            return fit

        fits = [fit for fit in (search(counterparts[index], suspect_blocks[index]) for index in nearest) if fit]
        if not fits:
            # The other file's blocks did not hold it (divided differently there, or not divided at all): the wide
            # window round the prediction, which costs several times as much to search, is only looked in then.
            window = (px0 - margin, py0 - margin, px1 + margin, py1 + margin)
            fits = [fit for fit in [search(_keypoints(suspect_gray, window, block_config), None)] if fit]

        transform, locally_aligned = global_transform, False
        if fits:
            best_inliers = max(fit.inliers for fit in fits)
            best = min((fit for fit in fits if fit.inliers >= _NEAR_EQUAL_INLIERS * best_inliers), key=lambda fit: fit.displacement)
            # Each part only where the whole-sheet transform is measurably wrong for it, judged on the part's own
            # points: one whose printing the whole-sheet transform already fits (or whose few keypoints sit round
            # the real difference being looked for) keeps it, rather than be nudged into manufacturing a difference.
            for part, fit in _split(block, best, block_config, global_transform, original_gray.shape):
                # And the part as a whole must look more like the original with its own transform than with the
                # whole-sheet one: keypoints can agree on the wrong thing (a repeated character, another copy of a
                # unit) and still fit each other well, but pulling in content from somewhere else never makes the
                # whole part match better -- so a real difference there is not aligned away.
                # A whole-sheet transform already sub-pixel on the part's own text can still be pixels out at its
                # far edges (a slight scale or shear, pulled by some other panel): a part whose pixels come far
                # closer with its own transform is taken on that alone.
                own_difference = _window_difference(original_gray, suspect_gray, part, fit.transform)
                global_difference = _window_difference(original_gray, suspect_gray, part, global_transform)
                improved = own_difference < _STRONG_WINDOW_GAIN * global_difference or (
                    fit.global_residual - fit.residual >= config.block_min_improvement_px
                    and own_difference < _WINDOW_GAIN * global_difference
                )
                edges = _EdgeScorer(original_dark, suspect_dark, part, interior=True)
                if improved and edges.judges:
                    # ...but pixels are swayed by what is NOT the artwork -- how much of a scan's grey background or
                    # shading a fit happens to cover -- and a fit locked onto one bold logo can look better by them while
                    # doubling every thin line round it. Where it lands the part's edges clearly further off than the
                    # whole-sheet fit does, it is not taken.
                    improved = edges(fit.transform) <= edges(global_transform) + _EDGE_MIN_GAIN_PX
                elif not improved and fit.global_residual - fit.residual >= config.block_min_improvement_px:
                    # A scan's paper and shading (or a flap's blank paper) can hide a real improvement from the pixel
                    # difference: where the edges land says whether the part is lined up.
                    improved = _edges_closer(edges(fit.transform), edges(global_transform), own_difference, global_difference)
                transform = fit.transform
                if improved and edges.judges:
                    # A fit from a block's keypoints can carry a little spurious stretch or shear (few points, a full
                    # homography): refined on the block's own printed edges, it lands them where they belong.
                    refined, refined_edges = edges.refine(transform)
                    if refined_edges < edges(transform) - _REFINE_MIN_GAIN_PX and _window_difference(
                        original_gray, suspect_gray, part, refined
                    ) <= own_difference:
                        transform = refined
                results.append(ArtworkBlockAlignment(part, transform if improved else global_transform, improved))
            if len(results) == 1 and not results[0].locally_aligned:
                # The block's keypoint fit was not taken (a flap whose few keypoints sit on a dimension note, or a fit
                # locked onto one logo): the outline stretched onto its counterpart, and the whole-sheet fit, each
                # refined on the edges, are tried instead, and taken where they land the edges clearly closer. (A fit
                # that passed on the block's own printing is not second-guessed by its outline.)
                chosen = results[0].transform
                edges = _EdgeScorer(original_dark, suspect_dark, block)
                outlined, outlined_edges = _outline_on_edges(edges, block, nearest, suspect_blocks, global_transform)
                if outlined is not None and _edges_closer(
                    outlined_edges,
                    edges(chosen),
                    _window_difference(original_gray, suspect_gray, block, outlined),
                    _window_difference(original_gray, suspect_gray, block, chosen),
                ):
                    results = [ArtworkBlockAlignment(block, outlined, True)]
            return results
        else:
            # Too little printing to match (a plain flap, a flat colour panel): the counterpart block's outline
            # stretched onto this one's, and a shift by phase correlation on top of that or of the whole-sheet fit.
            # Whichever lands the block's edges (its die line, mostly) closest is taken -- if it is clearly closer
            # than the whole-sheet fit.
            edges = _EdgeScorer(original_dark, suspect_dark, block)
            whole_edges = edges(global_transform)
            whole_pixels = _window_difference(original_gray, suspect_gray, block, global_transform)
            phased = [
                shifted
                for shifted in (
                    _phase_shift_transform(original_image, suspect_image, block, base, config)
                    for base in [global_transform] + _outlines(block, nearest[:1], suspect_blocks)
                )
                if shifted is not None
            ]
            best, best_edges = _outline_on_edges(edges, block, nearest, suspect_blocks, global_transform, phased)
            if best is not None and _edges_closer(
                best_edges, whole_edges, _window_difference(original_gray, suspect_gray, block, best), whole_pixels
            ):
                transform, locally_aligned = best, True
            elif not edges.judges and phased:
                # No edges to judge by at all (a soft, flat colour panel): the phase shift, which only ever passes if
                # it measurably brings the panel's pixels closer (see `_phase_shift_transform`), is taken on that.
                transform, locally_aligned = phased[0], True
        results.append(ArtworkBlockAlignment(block, transform, locally_aligned))
        return results

    # Every block is aligned on its own, so all of them at once (OpenCV lets go of the GIL while it works).
    with ThreadPoolExecutor(max_workers=min(_MAX_THREADS, max(1, len(blocks)))) as pool:
        wanted = sorted({index for nearest in nearest_of for index in nearest})
        counterparts = dict(zip(wanted, pool.map(counterpart_keypoints, wanted)))
        aligned = list(pool.map(lambda job: align_one(*job), zip(blocks, nearest_of)))
    results = [alignment for parts in aligned for alignment in parts]
    return _neighbours_taken(results, original_gray, suspect_gray)


_NEIGHBOUR_GAP_SHARE = 0.02  # blocks this close (a share of the larger's longer side) are neighbours


def _neighbours_taken(
    results: list[ArtworkBlockAlignment], original_gray: np.ndarray, suspect_gray: np.ndarray
) -> list[ArtworkBlockAlignment]:
    """A block that could not be aligned on its own (a flap, a strip above or below a panel, too plain to match by)
    kept the whole sheet's transform; it is joined on to the panel it borders, and moves with it on the carton -- so the
    transform of a neighbour that was aligned on its own is tried for it, and taken where it brings the block's pixels
    clearly closer than the whole sheet's. (A whole-sheet transform of a re-laid proof is a compromise between panels
    moved each their own way: it fits no strip well.)"""
    own = [result for result in results if result.locally_aligned]
    if not own:
        return results
    out = []
    for result in results:
        if result.locally_aligned:
            out.append(result)
            continue
        neighbours = [other for other in own if _bordering(result.block.bbox, other.block.bbox)]
        if not neighbours:
            out.append(result)
            continue
        current = _window_difference(original_gray, suspect_gray, result.block, result.transform)
        tried = [(_window_difference(original_gray, suspect_gray, result.block, other.transform), other) for other in neighbours]
        difference, best = min(tried, key=lambda item: item[0])
        if difference < _WINDOW_GAIN * current:
            out.append(ArtworkBlockAlignment(result.block, best.transform, True))
        else:
            out.append(result)
    return out


def _bordering(a, b) -> bool:
    """Whether two blocks' boxes touch (or nearly) along a side they share."""
    gap = _NEIGHBOUR_GAP_SHARE * max(a.width, a.height, b.width, b.height)
    overlap_x = min(a.x1, b.x1) - max(a.x0, b.x0)
    overlap_y = min(a.y1, b.y1) - max(a.y0, b.y0)
    beside = overlap_y > 0 and (abs(a.x1 - b.x0) <= gap or abs(b.x1 - a.x0) <= gap)
    above = overlap_x > 0 and (abs(a.y1 - b.y0) <= gap or abs(b.y1 - a.y0) <= gap)
    return beside or above


def _phase_shift_transform(
    original_image: np.ndarray,
    suspect_image: np.ndarray,
    block: ArtworkBlock,
    base_transform: np.ndarray,
    config: AlignmentConfig,
) -> np.ndarray | None:
    """A pure-translation correction for a block feature matching found nothing to work with at all (a flat
    colour panel, with no corners of its own): phase correlation between the block's own window in the
    original and the same window of the suspect already warped by `base_transform` (the whole-sheet transform,
    or the block's outline stretch), which reads a reliable shift from the window's bulk content (an edge, a line
    of text) rather than distinct keypoints.

    `None` if the block is too small for a stable window, the two windows do not correlate confidently enough
    to trust (`block_phase_min_response`), the shift is implausibly large (`block_phase_max_drift_px`) or too
    small to be worth a block of its own (`_MIN_SHIFT_PX`), or -- the same "must actually be better, not just
    plausible" bar the feature-matched path holds itself to -- applying it does not bring the two windows closer.
    """
    height, width = original_image.shape[:2]
    x0, y0 = max(0, round(block.bbox.x0)), max(0, round(block.bbox.y0))
    x1, y1 = min(width, round(block.bbox.x1)), min(height, round(block.bbox.y1))
    if x1 - x0 < _PHASE_MIN_WINDOW_PX or y1 - y0 < _PHASE_MIN_WINDOW_PX:
        return None

    to_window = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]]) @ base_transform
    warped_window = cv2.warpPerspective(
        suspect_image, to_window, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255)
    )
    original_window = cv2.cvtColor(original_image[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32)
    suspect_window = cv2.cvtColor(warped_window, cv2.COLOR_RGB2GRAY).astype(np.float32)

    hann = cv2.createHanningWindow((x1 - x0, y1 - y0), cv2.CV_32F)
    # `cv2.phaseCorrelate(a, b)` returns the shift that describes b AS a shifted by (dx, dy) -- so it is the
    # NEGATIVE of that shift which must be applied to b to move it back onto a (verified directly: applying
    # the raw (dx, dy) here made two synthetically-shifted windows match worse, not better).
    (raw_dx, raw_dy), response = cv2.phaseCorrelate(original_window * hann, suspect_window * hann)
    dx, dy = -raw_dx, -raw_dy
    if response < config.block_phase_min_response or max(abs(dx), abs(dy)) > config.block_phase_max_drift_px:
        return None
    if max(abs(dx), abs(dy)) < _MIN_SHIFT_PX:
        return None

    shift_matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    shifted_window = cv2.warpAffine(suspect_window, shift_matrix, (x1 - x0, y1 - y0), borderMode=cv2.BORDER_REPLICATE)
    if np.mean(np.abs(original_window - shifted_window)) >= np.mean(np.abs(original_window - suspect_window)):
        return None  # the shift did not actually bring the two windows closer together

    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]]) @ base_transform


def feathered_alpha(shape: tuple[int, int], polygon: list[Point], scale: float = 1.0, shift: Point = (0.0, 0.0)) -> np.ndarray:
    """A block's mask with its edge softened into a narrow ramp (0 outside, 1 well inside, in between only
    within `_SEAM_FEATHER_PX` of the edge) -- ready to alpha-blend rather than hard-cut, so where one block's
    own warp meets its neighbour's the seam reads as ordinary registration softness, not a new, sharp edge
    that was never in either file (a hard cut there is exactly what a structural/graphics detector is built
    to flag, manufacturing a difference that is really just this composite's own construction).

    `scale`/`shift` map the mask into another frame first (a device-pixel tile at some zoom, e.g. -- see
    `_polygon_mask`), the feather width scaled the same way, so a zoomed-in per-block re-render blends at its
    seam exactly as the non-zoomed composite already does.
    """
    hard = _polygon_mask(shape, polygon, scale=scale, shift=shift).astype(np.float32)
    sigma = max(0.5, (_SEAM_FEATHER_PX / 2.0) * scale)
    return np.clip(cv2.GaussianBlur(hard, (0, 0), sigmaX=sigma), 0.0, 1.0)


def _region(shape: tuple[int, int], block: ArtworkBlock, pad: int) -> tuple[int, int, int, int]:
    """The block's bounding box grown by `pad`, clipped to the picture: (x0, y0, x1, y1)."""
    height, width = shape
    b = block.bbox
    return (
        max(0, int(b.x0) - pad),
        max(0, int(b.y0) - pad),
        min(width, int(np.ceil(b.x1)) + pad),
        min(height, int(np.ceil(b.y1)) + pad),
    )


def seam_margin_mask(
    shape: tuple[int, int], block_alignments: list[ArtworkBlockAlignment], margin_px: int
) -> np.ndarray:
    """True within `margin_px` of any locally-aligned block's own edge.

    A block warped by its own transform meets its neighbours (warped by theirs, or by the whole artwork's) at
    a seam `composite_warp` can only feather, not remove -- and, just as the outer artwork boundary's own
    `edge_inset_norm` already accepts that the die line drawn on a lifted outline is not printed artwork, this
    margin is not compared either: whatever a seam looks like there is this composite's own construction,
    never a difference in either real file.
    """
    near_edge = np.zeros(shape, dtype=bool)
    kernel = np.ones((margin_px * 2 + 1, margin_px * 2 + 1), np.uint8)
    for alignment in block_alignments:
        if not alignment.locally_aligned:
            continue
        x0, y0, x1, y1 = _region(shape, alignment.block, margin_px + 1)
        mask = _polygon_mask((y1 - y0, x1 - x0), alignment.block.polygon, shift=(x0, y0)).astype(np.uint8) * 255
        near_edge[y0:y1, x0:x1] |= (cv2.dilate(mask, kernel) & ~cv2.erode(mask, kernel)).astype(bool)
    return near_edge


def composite_warp(
    suspect_image: np.ndarray,
    block_alignments: list[ArtworkBlockAlignment],
    global_transform: np.ndarray,
    reference_shape: tuple[int, int],
) -> np.ndarray:
    """The suspect warped into the reference (original) frame, each locally-aligned block's own region taken
    from its own local warp instead of the whole-artwork one -- one composite image the rest of the pipeline
    (which knows nothing about blocks) compares exactly as it always has. Each block is warped and blended over
    just its own region (a page can be 100 megapixels: a whole-page warp per block does not fit in memory)."""
    composite = warp_to_reference(suspect_image, global_transform, reference_shape)
    border = (255,) * composite.shape[2] if composite.ndim == 3 else 255
    for alignment in block_alignments:
        if not alignment.locally_aligned:
            continue
        x0, y0, x1, y1 = _region(reference_shape, alignment.block, 3 * _SEAM_FEATHER_PX)
        if x1 <= x0 or y1 <= y0:
            continue
        to_region = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]]) @ alignment.transform
        block_warp = cv2.warpPerspective(
            suspect_image, to_region, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderValue=border
        ).astype(np.float32)
        alpha = feathered_alpha((y1 - y0, x1 - x0), alignment.block.polygon, shift=(x0, y0))
        if composite.ndim == 3:
            alpha = alpha[..., None]
        region = composite[y0:y1, x0:x1].astype(np.float32)
        composite[y0:y1, x0:x1] = np.clip(alpha * block_warp + (1.0 - alpha) * region + 0.5, 0, 255).astype(np.uint8)
    return composite
