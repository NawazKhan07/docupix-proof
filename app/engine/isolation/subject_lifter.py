"""Lifting the artwork off its background -- "Lift subject".

The user presses and holds on the artwork; this returns the outline of the thing
under the press, separated from whatever surrounds it. Two ways of finding it are
tried, in this order:

1. **The area inside a closed border line** (`border_lifter.py`). Packaging artwork is a
   sheet: a carton's die-cut outline with the printed panels inside it and, around it,
   a title, a colour legend, crop marks and dots that are not artwork. The outline
   is what defines the artwork -- a carton's flaps are the colour of the paper, so
   colour cannot -- and what is sealed inside it is lifted, and nothing else.
2. **Colour and background** (this module): for a photograph or a scan of an object
   on a surface, where there is no line to follow.

The colour method, and why it is built the way it is:

* **Grab-cut on a reduced copy.** GrabCut (a graph cut over colour models of
  "subject" and "background") is what gives a clean edge, but it is slow on a
  5000 px sheet, so it runs on an ~720 px copy where it takes under a second.
* **The press seeds it, the page edge informs it.** A disc under the press is
  definite subject. The background's colours are learned from the outer strip of
  the page, including that background darkened by a cast shadow; pixels of those
  colours joined to the edge, and nearer to them than to the pressed colour, start
  out as background. So no rectangle has to be drawn, and the thing pressed on is
  what comes back even when several objects are in view.
* **The edge is then refined at full resolution.** Upscaling a 720 px outline to a
  5000 px sheet leaves it several pixels off, and for a comparison area that
  matters: a fringe of background inside the outline shows up as a difference. So
  the same colour models classify the pixels in a narrow band around the edge at
  full resolution.

**Hold-and-drag corrections** (keep this / drop this) are applied to either result as edits.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from app.config import SubjectConfig
from app.engine.isolation.border_lifter import border_region
from app.engine.isolation.mask_ops import NoSubjectFoundError, component_at, disc, fill_holes
from app.logging_config import StageTimer

logger = logging.getLogger(__name__)

Point = tuple[float, float]

_GMM_COMPONENTS = 5  # GrabCut's model size
_MIN_SUBJECT_PX = 40  # a speck of fewer pixels than this is not a subject (there is no rule about size beyond that)
_REFINE_PRIOR = 1.5  # how strongly the coarse outline is trusted against colour, in log-likelihood
_REFINE_SMOOTHING_PX = 1.2
_MAX_REFINE_BAND_PX = 24
# An edge pixel is a blend of the two colours, which neither model has seen; a tight model (a
# flat, noiseless background) would call every blend impossible and push the edge outward. Each
# component is therefore assumed to be at least this many grey levels wide.
_COLOUR_FLOOR_LEVELS = 20.0


@dataclass
class HintStroke:
    """A brushed correction: everything under it is (or is not) part of the subject."""

    points: list[Point]  # image pixels
    radius_px: float
    keep: bool  # True: part of the subject. False: background.


@dataclass
class LiftedSubject:
    mask: np.ndarray  # bool (H, W): the subject, in the pixels of the image it was lifted from
    polygon: list[Point]  # its outer outline, simplified, in those pixels
    source: str  # "border" (inside a closed line) | "print" (the printed extent) | "segmented" (by colour)
    image_size: tuple[int, int]  # (width, height) of the image it was lifted from
    hints: list[HintStroke] = field(default_factory=list)  # the corrections that produced it
    # How far inside the outline comparing should start (pixels of the image it was lifted from): the
    # width of the border line the outline traces, or the blurred fringe of an edge found by colour.
    edge_inset_px: float = 0.0

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        ys, xs = np.nonzero(self.mask)
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    @property
    def area_fraction(self) -> float:
        return float(self.mask.sum()) / self.mask.size

    def normalized_polygon(self):
        from app.models.isolation import NormalizedPolygon

        width, height = self.image_size
        return NormalizedPolygon(points=[(x / width, y / height) for x, y in self.polygon])


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def _to_lab(bgr: np.ndarray) -> np.ndarray:
    """CIELAB with L in 0-100 and a/b centred, so distances are roughly delta-E."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def _in_shadow(small_bgr: np.ndarray, palette_bgr: list[np.ndarray], config: SubjectConfig) -> np.ndarray:
    """Pixels that are a palette colour darkened without changing hue: the background in shadow."""
    low, high = config.shadow_brightness_range
    pixels = small_bgr.astype(np.float32)
    shadowed = np.zeros(small_bgr.shape[:2], bool)
    for colour in palette_bgr:
        length_squared = float(colour @ colour)
        if length_squared < 1.0:
            continue
        brightness = (pixels @ colour) / length_squared  # how much of the colour this pixel is
        residual = np.linalg.norm(pixels - brightness[..., None] * colour, axis=2) / math.sqrt(length_squared)
        shadowed |= (brightness >= low) & (brightness < high) & (residual < config.shadow_hue_tolerance)
    return shadowed


def _initial_labels(small_bgr: np.ndarray, seed: Point, config: SubjectConfig) -> np.ndarray:
    """GrabCut's starting labels: definite subject at the press, background from the page edge."""
    height, width = small_bgr.shape[:2]
    lab = _to_lab(small_bgr)

    band = max(2, int(round(config.border_band_fraction * min(height, width))))
    border = np.zeros((height, width), bool)
    border[:band], border[-band:], border[:, :band], border[:, -band:] = True, True, True, True

    # The background's palette: a few colour centres of the outer strip.
    edge_positions = np.flatnonzero(border.ravel())
    if len(edge_positions) > 8000:
        edge_positions = np.random.default_rng(0).choice(edge_positions, 8000, replace=False)
    lab_samples = np.ascontiguousarray(lab.reshape(-1, 3)[edge_positions])
    bgr_samples = small_bgr.reshape(-1, 3)[edge_positions].astype(np.float32)
    clusters = min(3, len(lab_samples))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, cluster_of, centres = cv2.kmeans(lab_samples, clusters, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
    cluster_of = cluster_of.ravel()
    distance_to_background = np.min(
        [np.linalg.norm(lab - centre, axis=2) for centre in centres], axis=0
    )
    palette_bgr = [bgr_samples[cluster_of == k].mean(axis=0) for k in range(clusters) if (cluster_of == k).any()]

    seed_disc = disc((height, width), seed, max(3.0, config.seed_radius_fraction * min(height, width)))
    seed_colour = np.median(lab[seed_disc], axis=0)
    distance_to_seed = np.linalg.norm(lab - seed_colour, axis=2)

    # How alike is "the same background"? As alike as the page edge is to itself.
    spread = float(np.percentile(distance_to_background[border], 90))
    tolerance = float(np.clip(config.background_spread_factor * spread, config.background_tolerance_min, config.background_tolerance))

    # "Nearer the background than the pressed colour" only means something when the press is
    # on something else. Pressed on a background-coloured feature (a white panel on a white
    # page), every pixel is as near to it as to the background, and edge-connectivity below
    # is what keeps such a panel from being taken for background.
    press_is_background_coloured = min(float(np.linalg.norm(seed_colour - c)) for c in centres) < tolerance
    if press_is_background_coloured:
        closer_to_background = np.ones((height, width), bool)
    else:
        closer_to_background = distance_to_background < distance_to_seed
    background_like = (distance_to_background < tolerance) & closer_to_background
    background_like |= _in_shadow(small_bgr, palette_bgr, config) & closer_to_background
    # Only background-coloured pixels joined to the page edge are background: a white
    # panel inside a carton on a white page is not.
    _, joined = cv2.connectedComponents(background_like.astype(np.uint8), connectivity=4)
    touching = np.unique(joined[border & background_like])
    reaches_edge = np.isin(joined, touching[touching > 0])

    labels = np.full((height, width), cv2.GC_PR_FGD, np.uint8)
    labels[reaches_edge] = cv2.GC_PR_BGD
    labels[border & (distance_to_background < 2 * tolerance) & closer_to_background] = cv2.GC_BGD
    labels[seed_disc] = cv2.GC_FGD

    return labels


def _segment(small_bgr: np.ndarray, seed: Point, config: SubjectConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(subject mask, foreground colour model, background colour model), at working resolution."""
    height, width = small_bgr.shape[:2]
    labels = _initial_labels(small_bgr, seed, config)
    has_background = ((labels == cv2.GC_BGD) | (labels == cv2.GC_PR_BGD)).any()
    if not has_background:
        # Nothing looks like the page edge's colours: the artwork fills the frame.
        return np.ones((height, width), bool), np.zeros((1, 65)), np.zeros((1, 65))

    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        small_bgr, labels, None, background_model, foreground_model,
        config.grabcut_iterations, cv2.GC_INIT_WITH_MASK,
    )
    subject = (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)
    subject = cv2.morphologyEx(subject.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)

    seed_radius = max(3.0, config.seed_radius_fraction * min(height, width))
    subject = fill_holes(component_at(subject, seed, seed_radius))
    return subject, foreground_model, background_model


# ---------------------------------------------------------------------------
# Full-resolution edge refinement
# ---------------------------------------------------------------------------


def _mixture_log_density(model: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Log-density of pixels (N, 3, BGR) under a GrabCut colour model (a 5-component Gaussian mixture)."""
    flat = model.reshape(-1)
    count = _GMM_COMPONENTS
    weights = flat[:count]
    means = flat[count : 4 * count].reshape(count, 3)
    covariances = flat[4 * count : 13 * count].reshape(count, 3, 3)

    terms = []
    for weight, mean, covariance in zip(weights, means, covariances):
        if weight <= 0:
            continue
        covariance = covariance + np.eye(3) * (_COLOUR_FLOOR_LEVELS**2)
        determinant = np.linalg.det(covariance)
        inverse = np.linalg.inv(covariance)
        delta = pixels - mean
        mahalanobis = np.einsum("ni,ij,nj->n", delta, inverse, delta)
        terms.append(math.log(weight) - 0.5 * math.log(determinant) - 0.5 * mahalanobis)
    if not terms:
        return np.full(len(pixels), -1e9)
    stacked = np.stack(terms)
    peak = stacked.max(axis=0)
    return peak + np.log(np.exp(stacked - peak).sum(axis=0))


def _refine_edge(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    foreground_model: np.ndarray,
    background_model: np.ndarray,
    seed: Point,
    scale: float,
    config: SubjectConfig,
) -> np.ndarray:
    """Re-decide the pixels along the outline at full resolution, from the colour models."""
    if not foreground_model.any() or not background_model.any():
        return mask
    reach = int(min(_MAX_REFINE_BAND_PX, max(2, round(config.refine_band_working_px / scale))))

    ys, xs = np.nonzero(mask)
    pad = reach + 3
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + 1 + pad)
    region = mask[y0:y1, x0:x1].astype(np.uint8)
    crop = image_bgr[y0:y1, x0:x1]

    # The band along the edge is drawn as a thick line along the outline. (Eroding and dilating
    # a 18 MP mask by a 20 px disc took longer than everything else put together.)
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ring = np.zeros(region.shape, np.uint8)
    cv2.drawContours(ring, contours, -1, 1, thickness=2 * reach + 1)
    band = ring.astype(bool)
    core = region.astype(bool) & ~band
    band_ys, band_xs = np.nonzero(band)
    if band_ys.size == 0:
        return mask

    pixels = crop[band_ys, band_xs].astype(np.float64)
    evidence = _mixture_log_density(foreground_model, pixels) - _mixture_log_density(background_model, pixels)
    evidence += np.where(region[band_ys, band_xs] > 0, _REFINE_PRIOR, -_REFINE_PRIOR)

    field_ = np.zeros(region.shape, np.float32)
    field_[band_ys, band_xs] = np.clip(evidence, -30, 30)
    weight = band.astype(np.float32)
    sigma = _REFINE_SMOOTHING_PX
    smoothed = cv2.GaussianBlur(field_, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(weight, (0, 0), sigma), 1e-3)
    refined = core | (band & (smoothed > 0))
    refined = cv2.morphologyEx(refined.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)

    full = np.zeros_like(mask)
    full[y0:y1, x0:x1] = refined
    try:
        full = fill_holes(component_at(full, seed, max(3.0, config.seed_radius_fraction * min(mask.shape) )))
    except NoSubjectFoundError:
        return mask
    # A refinement that changes the subject wholesale is a mistake, not a refinement.
    if abs(int(full.sum()) - int(mask.sum())) > 0.25 * int(mask.sum()):
        return mask
    return full


# ---------------------------------------------------------------------------
# Outline
# ---------------------------------------------------------------------------


def outline_from_mask(mask: np.ndarray, config: SubjectConfig) -> list[Point]:
    """The outer outline of `mask` as a simplified polygon (a rectangle, if it is nearly one)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise NoSubjectFoundError("The lifted area is empty.")
    contour = max(contours, key=cv2.contourArea)

    (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(contour)
    if rect_w > 0 and rect_h > 0 and cv2.contourArea(contour) / (rect_w * rect_h) >= config.rectangle_snap_fill:
        corners = cv2.boxPoints(cv2.minAreaRect(contour))
        return [(float(x), float(y)) for x, y in corners]

    epsilon = config.outline_epsilon_px
    approx = cv2.approxPolyDP(contour, epsilon, True)
    while len(approx) > config.max_outline_points:
        epsilon *= 1.4
        approx = cv2.approxPolyDP(contour, epsilon, True)
    return [(float(p[0][0]), float(p[0][1])) for p in approx]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


METHODS = ("auto", "border", "colour")


def _stroke_mask(shape: tuple[int, int], stroke: HintStroke) -> np.ndarray:
    canvas = np.zeros(shape, np.uint8)
    points = np.round(np.array(stroke.points, dtype=np.float32)).astype(np.int32)
    thickness = max(1, int(round(2 * stroke.radius_px)))
    if len(points) == 1:
        cv2.circle(canvas, tuple(int(v) for v in points[0]), max(1, thickness // 2), 1, -1)
    else:
        cv2.polylines(canvas, [points], False, 1, thickness)
    return canvas.astype(bool)


def apply_hints(mask: np.ndarray, hints: list[HintStroke]) -> np.ndarray:
    """Edit a lifted region by hand: add what was painted 'keep', remove what was painted 'drop'."""
    edited = mask.copy()
    for stroke in hints:
        painted = _stroke_mask(mask.shape, stroke)
        edited = (edited | painted) if stroke.keep else (edited & ~painted)
    return edited


def _corrected(mask: np.ndarray, hints: list[HintStroke], seed: Point | None) -> np.ndarray:
    """Apply the user's corrections to a lifted region and keep it one piece.

    Corrections are edits, not evidence: painting over a part that was missed adds exactly that
    part, and painting over something wrongly lifted removes exactly that, with nothing else
    moving. (Fed to the colour models as labels instead, a stroke of the artwork's own colour on
    the wrong side of its edge taught the model that the artwork was background.)
    """
    if not hints:
        return mask
    edited = apply_hints(mask, hints)
    if not edited.any():
        raise NoSubjectFoundError("Nothing is left of the lifted area after the corrections.")
    if seed is None:
        count, labels = cv2.connectedComponents(edited.astype(np.uint8), connectivity=8)
        sizes = np.bincount(labels.ravel(), minlength=count)
        sizes[0] = 0
        return fill_holes(labels == int(sizes.argmax()))
    return fill_holes(component_at(edited, seed, max(3.0, 0.005 * min(edited.shape))))


_PRINT_SCALE_LONG_EDGE_PX = 1200  # the printed extent is found on a copy this big
_PRINT_FROM_PAPER = 40  # (darkest channel) this much darker than the paper is print
_PRINT_GATHER_SHARES = (0.03, 0.06, 0.1)  # print this close (a share of the long edge) is one piece, tried in turn...
_PRINT_MOST = 0.6  # ...until one piece holds this share of the page's print
_PRINT_SPECK_SHARE = 0.00005  # a mark smaller than this share of the page is a speck (dust, a stray dot)
_PRINT_MIN_AREA = 0.05  # ...and the piece is at least this share of the page


def printed_extent(image_rgb: np.ndarray, config: SubjectConfig) -> np.ndarray | None:
    """The artwork as what is printed: the page's print gathered into pieces (what lies close together is one piece),
    holes filled, the biggest piece -- a carton whose dieline is not closed, on a light page. None on a dark page, or
    where no piece is big enough to be an artwork."""
    height, width = image_rgb.shape[:2]
    scale = min(1.0, _PRINT_SCALE_LONG_EDGE_PX / max(height, width))
    small = cv2.resize(image_rgb, (max(2, round(width * scale)), max(2, round(height * scale))), interpolation=cv2.INTER_AREA)
    darkest = small.min(axis=2)
    paper = float(np.percentile(darkest, 90))
    if paper < config.border_min_page_brightness:
        return None
    ink = (darkest.astype(np.int16) < paper - _PRINT_FROM_PAPER).astype(np.uint8)
    total = int(ink.sum())
    if total == 0:
        return None
    # gathered ever wider -- a carton's panels lie apart, blank paper between their blocks of print -- until one piece
    # holds most of the page's print (a job sheet or a colour bar beside the artwork stays out of it)
    for share in _PRINT_GATHER_SHARES:
        gather = max(3, int(round(share * max(small.shape[:2])))) | 1
        pieces = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((gather, gather), np.uint8))
        count, labels, stats, _centres = cv2.connectedComponentsWithStats(pieces, connectivity=8)
        if count <= 1:
            continue
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        piece = (labels == biggest).astype(np.uint8)
        if int((ink & piece).sum()) >= _PRINT_MOST * total:
            break
    else:
        return None
    # the artwork's extent: the outline round all the page's print but specks -- a flap's barcode, a side panel lie far
    # from the rest, and must not be cut away; the paper this takes in round the corners is paper in both files
    count, labels, stats, _centres = cv2.connectedComponentsWithStats(ink, connectivity=8)
    big = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= _PRINT_SPECK_SHARE * ink.size)
    big = big[big > 0]
    points = cv2.findNonZero(np.isin(labels, big).astype(np.uint8)) if big.size else cv2.findNonZero(piece)
    if points is None:
        return None
    filled = np.zeros_like(piece)
    cv2.fillConvexPoly(filled, cv2.convexHull(points), 1)
    if filled.mean() < _PRINT_MIN_AREA:
        return None
    return cv2.resize(filled, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)


def lift_subject(
    image_rgb: np.ndarray,
    seed: Point | None = None,
    config: SubjectConfig | None = None,
    hints: list[HintStroke] | None = None,
    page=None,
    dpi: float | None = None,
    method: str = "auto",
) -> LiftedSubject:
    """Lift the artwork at `seed` (image pixels), or, with no seed, the largest closed border.

    `method`: "auto" tries the area inside a closed border line and falls back to colour;
    "border" only the former (with a lower bar for how big an enclosure must be);
    "colour" only the latter. `page` and `dpi`, when the image is a rendered PDF page, let its
    vector lines be used as the border.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    config = config or SubjectConfig()
    hints = list(hints or [])
    height, width = image_rgb.shape[:2]
    if seed is not None:
        seed = (min(max(float(seed[0]), 0.0), width - 1.0), min(max(float(seed[1]), 0.0), height - 1.0))

    with StageTimer(logger, "lift_subject"):
        if method in ("auto", "border"):
            region = border_region(image_rgb, seed, config, page=page, dpi=dpi, strict=(method == "auto"))
            if region is not None:
                mask = _corrected(region.mask, hints, seed)
                inset = region.line_width_px + config.border_line_margin_px
                return LiftedSubject(mask, outline_from_mask(mask, config), "border", (width, height), hints, inset)
            if method == "border":
                raise NoSubjectFoundError(
                    "No closed border line was found around that spot. Press and hold inside the outlined artwork, "
                    "or switch to the colour method."
                )
        if seed is None:
            # No closed border (a dieline dashed, faint, or broken off where a flattened file lost it): the artwork is
            # what is printed -- all of the page's print gathered into one piece, its holes filled.
            extent = printed_extent(image_rgb, config)
            if extent is not None:
                mask = _corrected(extent, hints, seed)
                return LiftedSubject(mask, outline_from_mask(mask, config), "print", (width, height), hints, 0.0)
            raise NoSubjectFoundError("Press and hold on the artwork to lift it.")

        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        scale = min(1.0, config.working_long_edge_px / max(height, width))
        if scale < 1.0:
            small = cv2.resize(bgr, (max(2, round(width * scale)), max(2, round(height * scale))), interpolation=cv2.INTER_AREA)
        else:
            small = bgr
        seed_small = (seed[0] * scale, seed[1] * scale)

        # The press marks a disc of definite subject; where the area pressed on is smaller than the disc, the result is
        # the disc and nothing has been lifted. So a smaller disc is tried, and smaller again: a small area is lifted like a big one.
        mask = None
        for step in config.seed_radius_steps:
            trial = replace(config, seed_radius_fraction=config.seed_radius_fraction * step)
            small_mask, foreground_model, background_model = _segment(small, seed_small, trial)

            if scale < 1.0:
                # Bilinear, not blurred first: smoothing a reduced-size mask rounds an artwork's corners by
                # several full-resolution pixels, more than the edge refinement then reaches to correct.
                candidate = cv2.resize(small_mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR) > 0.5
            else:
                candidate = small_mask
            candidate = _refine_edge(bgr, candidate, foreground_model, background_model, seed, scale, trial)

            seed_area = math.pi * (max(3.0, trial.seed_radius_fraction * min(small.shape[:2])) / scale) ** 2
            if candidate.sum() >= max(_MIN_SUBJECT_PX, 2 * seed_area):
                mask = candidate
                break
        if mask is None:
            raise NoSubjectFoundError("Nothing could be lifted at that spot. Press and hold on the artwork itself.")
        mask = _corrected(mask, hints, seed)

        polygon = outline_from_mask(mask, config)
        logger.info(
            "lifted a subject",
            extra={"area_fraction": round(float(mask.sum()) / mask.size, 4), "points": len(polygon), "hints": len(hints)},
        )
        return LiftedSubject(mask, polygon, "segmented", (width, height), hints, config.colour_edge_inset_px)
