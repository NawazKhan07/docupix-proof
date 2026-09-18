"""Structural ("ink") comparison that tolerates misregistration (SRS FR-6.3).

Built for the case where a straight comparison fails: a scan against a digital
original. Scanning shifts colour, blurs, adds noise, darkens the corners and
never registers perfectly, so this compares *where ink is*, not what colour it
is, and only calls something a difference when the other file has no ink
anywhere near it. Each design choice below was made against a measured failure:

* **Ink is contrast against the LOCAL background**, symmetric (darker or
  lighter), so orange text on grey counts exactly like black on white. A single
  global black/white threshold does not: orange sits near the cutoff and can
  vanish, taking a whole sentence with it.
* **The background is a median, not a maximum.** An earlier version took the
  brightest nearby pixel as "paper"; a grey background beside white elements
  then flipped in and out of "ink" under scan noise, and 4-34% of the page
  came back as different. A median follows the dominant local tone and is
  stable.
* **Masked-out areas are filled from their neighbours** before the background
  is estimated. Otherwise the fill (white) sits against a scan's darkened
  corner, the estimate cannot follow the shadow, and the corner reads as ink.
* **Hysteresis.** A pixel is a difference only if it is STRONG ink in one file
  and not even WEAKLY present in the other, so faint content that flickers
  across a single threshold between two files (a pale tint, a vignette) is
  never reported.
* **Two background scales, unioned.** One window cannot serve both: a large
  one sees a small coloured box as ink but hides a change inside it (a hyphen
  removed from white-on-red text was missed), while a small one sees inside
  an added box and reports only its outline as four thin strips. Each scale
  is clean on its own, and overlapping detections merge into one finding.
* **Tolerance is positional, not morphological.** A pixel of ink matches if the
  other file has ink within `slack` pixels, so a registration shift of that
  size flags nothing -- yet a stroke the other file lacks is kept whole, thin or
  not. (Erasing thin differences to suppress noise is what turned a missing
  sentence into four boxes on stray letters.) The slack follows the measured
  alignment residual: it is chosen from how well the files register, not from
  what they are.
* **Thresholds follow each image's measured noise.** A clean digital file keeps
  the sensitive defaults; a noisy scan is judged against its own noise floor,
  so its speckle is not mistaken for content. Nothing is tuned to a file --
  the floor is measured from the pixels.
* **"Is there ink over there?" is asked at the softness of the softer image.** A
  scan is a little out of focus, and a soft edge reads as ink some pixels wider
  than a sharp one -- more than registration slack can absorb, so a box's
  outline came back as a difference. The sharper image is blurred to match, by
  an amount measured from the pixels, but ONLY to answer that question. Whether
  a stroke is strong ink is judged on the image as it is: blurring a thin
  stroke to a third of its contrast would make a missing thin sentence vanish.
* **Detail the scan cannot hold is not compared.** Tiny text in a soft scan is
  simply gone from the scan, and reporting all of it as "removed" would bury
  the reviewer in false alarms. Where one image has fine detail and the other
  has only a soft version of it -- the same LAYOUT at a coarser scale -- the
  area is excluded and reported as unverifiable. Blur keeps layout and loses
  detail; a real change alters the layout (a missing sentence leaves nothing,
  a new box adds an edge), so it stays a difference.
* **Direction is preserved.** Ink only in the original was removed; ink only in
  the suspect was added; both is a change (FR-6.1's changed / added / removed).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import DetectionConfig
from app.logging_config import StageTimer
from app.models.diff import BoundingBox, DiffCandidate, DiffSubtype, DiffType

logger = logging.getLogger(__name__)

_DIRECTIONAL_SHARE = 0.9  # this much of the differing ink on one side makes it added/removed
_DETAIL_SIGMA_PX = 1.6  # the band-pass that isolates stroke-scale detail
_DETAIL_POOL_PX = 8  # detail and ink mass are compared per 8 px cell...
_DETAIL_BOX_CELLS = 9  # ...averaged over 9 x 9 of them (~6 mm at 300 DPI)
_COARSE_SIGMAS_PX = (2.5, 6.0)  # the band that survives blur: word and line structure
_BACKGROUND_WINDOW_CELLS = 11  # median window, in cells of the reduced copy it runs on


def estimate_noise(gray: np.ndarray) -> float:
    """Sensor-noise standard deviation in grey levels, robustly estimated.

    Convolves with a Laplacian-difference kernel that cancels smooth image
    content and leaves noise (Immerkaer), then takes the MEDIAN of the response
    rather than the mean: text and edges are a small minority of pixels, so the
    median stays on the flat background where the noise is. Returns 0 for a clean
    digital render.
    """
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    # A strided sample is plenty for a global statistic and keeps this cheap.
    sample = gray[::2, ::2]
    response = np.abs(cv2.filter2D(sample, cv2.CV_32F, kernel))
    return float(np.median(response) / (0.6745 * 6.0))


def local_background(
    gray: np.ndarray, window_px: int, keep_mask: np.ndarray | None = None
) -> np.ndarray:
    """The dominant local tone of `gray` (uint8), robust to text, noise and shadows.

    `window_px` is the size of the neighbourhood the tone is taken over. It is
    a property of the artwork's feature size, not of the page: too large and a
    small coloured box the text sits in is averaged into its surroundings, so
    the whole box reads as ink in both files and a change inside it vanishes
    (a hyphen removed from white-on-red text was missed for exactly this
    reason); too small and thick strokes become their own background.

    `keep_mask` marks pixels that are really part of the image; the rest (outside
    a scan, outside the artwork boundary) are filled from their neighbours so
    they cannot pull the estimate towards their own, meaningless, value.
    """
    height, width = gray.shape
    factor = max(1, window_px // _BACKGROUND_WINDOW_CELLS)
    small = cv2.resize(
        gray, (max(1, width // factor), max(1, height // factor)), interpolation=cv2.INTER_AREA
    )

    if keep_mask is not None and not keep_mask.all():
        valid = (
            cv2.resize(
                keep_mask.astype(np.uint8) * 255,
                (small.shape[1], small.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            > 127
        )
        if valid.any():
            small = cv2.inpaint(small, (~valid).astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)

    window = min(_BACKGROUND_WINDOW_CELLS, (min(small.shape) // 2) * 2 - 1)
    median = cv2.medianBlur(small, max(3, window | 1))
    return cv2.resize(median, (width, height), interpolation=cv2.INTER_LINEAR)


class InkField:
    """An image's contrast against its local background, from which ink masks are cut."""

    def __init__(self, image: np.ndarray, window_px: int, keep_mask: np.ndarray | None = None) -> None:
        self.gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.background = local_background(self.gray, window_px, keep_mask)
        self.contrast = cv2.absdiff(self.gray, self.background)
        self.noise = estimate_noise(self.gray)

    def _above(self, relative: float, floor: float) -> np.ndarray:
        # The bar scales with the local tone: a fixed number of grey levels means
        # far more on a dark panel than on white paper.
        threshold = np.maximum(floor, self.background.astype(np.float32) * relative).astype(np.uint8)
        return cv2.compare(self.contrast, threshold, cv2.CMP_GT).astype(bool)

    def strong(self, config: DetectionConfig) -> np.ndarray:
        return self._above(
            config.ink_strong_relative, max(config.ink_strong_floor, config.ink_noise_strong_sigmas * self.noise)
        )

    def weak(self, config: DetectionConfig, other_noise: float = 0.0) -> np.ndarray:
        """Ink at the level the OTHER image could still show, given its own noise."""
        noise = max(self.noise, other_noise)
        return self._above(config.ink_weak_relative, max(config.ink_weak_floor, config.ink_noise_weak_sigmas * noise))


@dataclass
class StructureResult:
    candidates: list[DiffCandidate]
    # Share of the compared area where the two images cannot be compared at fine
    # scale because one is much softer than the other; those areas are skipped.
    unverifiable_fraction: float = 0.0
    # Gaussian blur (px) applied to the sharper image to match the softer one.
    matched_softness_px: float = 0.0


def _pool(values: np.ndarray, factor: int) -> np.ndarray:
    return cv2.resize(
        values,
        (max(1, values.shape[1] // factor), max(1, values.shape[0] // factor)),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)


def _box(values: np.ndarray) -> np.ndarray:
    return cv2.boxFilter(values, -1, (_DETAIL_BOX_CELLS, _DETAIL_BOX_CELLS))


def _fine_and_coarse(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell energy at stroke scale, and the word/line-scale structure map."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    fine = cv2.absdiff(gray, cv2.GaussianBlur(gray, (0, 0), _DETAIL_SIGMA_PX))
    # Word-and-line-scale structure: what a blur leaves behind.
    coarse = cv2.GaussianBlur(gray, (0, 0), _COARSE_SIGMAS_PX[0]).astype(np.int16) - cv2.GaussianBlur(
        gray, (0, 0), _COARSE_SIGMAS_PX[1]
    ).astype(np.int16)
    return _box(_pool(fine, _DETAIL_POOL_PX)), _pool(coarse.astype(np.float32), _DETAIL_POOL_PX)


def _local_correlation(a: np.ndarray, b: np.ndarray, variance_floor: float) -> np.ndarray:
    """Correlation of two maps over a sliding window; 0 where either is flat."""
    mean_a, mean_b = _box(a), _box(b)
    var_a = _box(a * a) - mean_a * mean_a
    var_b = _box(b * b) - mean_b * mean_b
    covariance = _box(a * b) - mean_a * mean_b
    informative = (var_a > variance_floor) & (var_b > variance_floor)
    denominator = np.sqrt(np.maximum(var_a * var_b, 1e-6))
    return np.where(informative, covariance / denominator, 0.0)


def unresolvable_mask(original: np.ndarray, suspect: np.ndarray, config: DetectionConfig) -> np.ndarray:
    """Where one image is a soft version of the other (boolean, full size).

    True where fine detail differs a lot yet the coarse layout of the two still
    agrees. A region that is blank in one image has no layout to agree with, and
    a box added around existing text changes the layout, so neither is masked.
    """
    fine_a, coarse_a = _fine_and_coarse(original)
    fine_b, coarse_b = _fine_and_coarse(suspect)

    sharp = np.maximum(fine_a, fine_b)
    soft = np.minimum(fine_a, fine_b)
    correlation = _local_correlation(coarse_a, coarse_b, config.structure_layout_variance_floor)

    blurred = (
        (sharp >= config.structure_detail_floor)
        & (soft < config.structure_detail_ratio * sharp)
        & (correlation >= config.structure_layout_correlation)
    )
    height, width = original.shape[:2]
    full = cv2.resize(blurred.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return cv2.dilate(full, _ellipse(_DETAIL_POOL_PX)).astype(bool)


_SOFTNESS_TILE_PX = 96  # softness is judged on patches this size...
_SOFTNESS_TILES = 24  # ...taking the busiest ones, since blur only shows where there is detail
_SOFTNESS_MARGIN_PX = 12  # context so a blur does not bleed in from the patch edge
_SOFTNESS_SEARCH_PX = 4  # residual misregistration each patch is allowed to absorb
_SOFTNESS_BUSY_FRACTION = 0.25  # a patch must have this share of the busiest one's detail


def _busiest_tiles(a: np.ndarray, b: np.ndarray, valid: np.ndarray | None) -> list[tuple[int, int]]:
    """Top-left corners of the patches with the most detail in BOTH images."""
    size, edge = _SOFTNESS_TILE_PX, _SOFTNESS_MARGIN_PX
    height, width = a.shape
    rows, cols = height // size, width // size
    if rows < 1 or cols < 1:
        return []

    def energy(gray: np.ndarray) -> np.ndarray:
        detail = cv2.absdiff(gray, cv2.GaussianBlur(gray, (0, 0), 1.0)).astype(np.float32)
        return cv2.resize(detail, (cols, rows), interpolation=cv2.INTER_AREA)

    score = np.minimum(energy(a), energy(b))
    if valid is not None:
        inside = cv2.resize(valid.astype(np.uint8) * 255, (cols, rows), interpolation=cv2.INTER_AREA) > 250
        score = np.where(inside, score, 0.0)
    flat = score.ravel()
    # "Busy" is relative to this page: a soft, faint scan has little detail in absolute terms.
    bar = _SOFTNESS_BUSY_FRACTION * float(flat.max())
    tiles = []
    for index in np.argsort(flat)[::-1]:
        y, x = int(index // cols) * size, int(index % cols) * size
        if flat[index] <= bar or len(tiles) >= _SOFTNESS_TILES:
            break
        if y >= edge and x >= edge and y + size + edge <= height and x + size + edge <= width:
            tiles.append((y, x))
    return tiles


def match_softness(
    original: np.ndarray,
    suspect: np.ndarray,
    config: DetectionConfig,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Blur the sharper image to the softness of the other. Returns (original, suspect, sigma).

    The blur is the one that makes the two agree best (normalised correlation on
    the busiest patches, so lighting and colour differences do not matter), tried
    in both directions; it is only applied if it clearly beats not blurring. Each
    patch may slide a few pixels while it is scored: a registration error would
    otherwise be "explained" by blur, which flatters the wrong answer.
    """
    gray_a = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(suspect, cv2.COLOR_RGB2GRAY)
    tiles = _busiest_tiles(gray_a, gray_b, valid)
    if not tiles:
        return original, suspect, 0.0

    size, edge, search = _SOFTNESS_TILE_PX, _SOFTNESS_MARGIN_PX, _SOFTNESS_SEARCH_PX

    def context(gray: np.ndarray) -> list[np.ndarray]:
        return [gray[y - edge : y + size + edge, x - edge : x + size + edge].astype(np.float32) for y, x in tiles]

    def agreement(sharp: list[np.ndarray], other: list[np.ndarray], sigma: float) -> float:
        total = 0.0
        for s, o in zip(sharp, other):
            blurred = cv2.GaussianBlur(s, (0, 0), sigma) if sigma > 0 else s
            template = blurred[edge : edge + size, edge : edge + size]
            window = o[edge - search : edge + size + search, edge - search : edge + size + search]
            total += float(cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED).max())
        return total / len(tiles)

    a_context, b_context = context(gray_a), context(gray_b)
    baseline = agreement(a_context, b_context, 0.0)
    best_sigma, best_score, blur_original = 0.0, baseline, True
    for sigma in np.geomspace(0.5, max(0.6, config.structure_softness_max_px), 9):
        for blur_a in (True, False):
            value = agreement(a_context, b_context, sigma) if blur_a else agreement(b_context, a_context, sigma)
            if value > best_score:
                best_sigma, best_score, blur_original = float(sigma), value, blur_a

    if best_sigma == 0.0 or best_score - baseline < config.structure_softness_min_gain:
        return original, suspect, 0.0

    logger.info(
        "matching softness",
        extra={
            "sigma_px": round(best_sigma, 2),
            "blurred": "original" if blur_original else "suspect",
            "gain": round(best_score - baseline, 3),
        },
    )
    if blur_original:
        return cv2.GaussianBlur(original, (0, 0), best_sigma), suspect, best_sigma
    return original, cv2.GaussianBlur(suspect, (0, 0), best_sigma), best_sigma


def slack_for(residual_px: float, config: DetectionConfig) -> int:
    """How much registration error to tolerate, from the measured alignment residual."""
    wanted = math.ceil(residual_px * config.structure_slack_factor)
    return int(min(config.structure_max_slack_px, max(config.structure_min_slack_px, wanted)))


def _ellipse(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _differences_at(
    original: np.ndarray,
    suspect: np.ndarray,
    matched_original: np.ndarray,
    matched_suspect: np.ndarray,
    config: DetectionConfig,
    slack_px: int,
    window_px: int,
    compare_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """(only_original, only_suspect) ink at one background scale.

    Strong ink is read from each image as it is. Whether the OTHER image has ink
    nearby is read from its softness-matched copy (the same image when the two are
    equally sharp). And a stroke only counts as missing if the other image could
    have shown it: blurred to that image's softness and held against its noise
    floor, a tiny stroke may be gone, and "gone" is then not "removed".
    """
    original_field = InkField(original, window_px, compare_mask)
    suspect_field = InkField(suspect, window_px, compare_mask)
    original_matched = original_field if matched_original is original else InkField(matched_original, window_px, compare_mask)
    suspect_matched = suspect_field if matched_suspect is suspect else InkField(matched_suspect, window_px, compare_mask)

    # What each image's ink would look like in the other's softness and noise.
    original_visible_there = original_matched.weak(config, suspect_matched.noise)
    suspect_visible_there = suspect_matched.weak(config, original_matched.noise)

    tolerance = _ellipse(max(slack_px, window_px // _BACKGROUND_WINDOW_CELLS))
    near_suspect = cv2.dilate(suspect_matched.weak(config).astype(np.uint8), tolerance).astype(bool)
    near_original = cv2.dilate(original_matched.weak(config).astype(np.uint8), tolerance).astype(bool)
    return (
        original_field.strong(config) & original_visible_there & ~near_suspect,
        suspect_field.strong(config) & suspect_visible_there & ~near_original,
    )


def detect(
    original: np.ndarray,
    suspect: np.ndarray,
    config: DetectionConfig,
    slack_px: int,
    compare_mask: np.ndarray | None = None,
) -> StructureResult:
    """Structural differences between two images already in the shared aligned space."""
    with StageTimer(logger, "detector_structure"):
        # The edge of a scan (or of the artwork boundary) is contaminated by
        # interpolation and vignetting for a few pixels; keep clear of it.
        if compare_mask is not None:
            compare_mask = cv2.erode(
                compare_mask.astype(np.uint8), _ellipse(config.structure_edge_margin_factor * slack_px)
            ).astype(bool)

        matched_original, matched_suspect, matched_softness = match_softness(
            original, suspect, config, compare_mask
        )

        # Detail is only "lost" if it is missing at the softness both images share.
        unverifiable = unresolvable_mask(matched_original, matched_suspect, config)
        compared = compare_mask if compare_mask is not None else np.ones(original.shape[:2], bool)
        unverifiable_fraction = float((unverifiable & compared).sum() / max(1, compared.sum()))
        compare_mask = compared & ~unverifiable

        only_original = np.zeros(original.shape[:2], dtype=bool)
        only_suspect = np.zeros(original.shape[:2], dtype=bool)
        for window_px in config.structure_background_windows_px:
            removed_here, added_here = _differences_at(
                original, suspect, matched_original, matched_suspect, config, slack_px, window_px, compare_mask
            )
            only_original |= removed_here
            only_suspect |= added_here

        only_original &= compare_mask
        only_suspect &= compare_mask
        different = only_original | only_suspect
        if not different.any():
            return StructureResult([], unverifiable_fraction, matched_softness)

        # Group differing ink into objects; the box is then the tight extent of
        # the differing pixels themselves, not of the grouping dilation. Done
        # with per-label reductions rather than a loop over labels: a noisy scan
        # can have thousands of components, and masking the whole image for each
        # one took a harsh scan from ~1 s to ~6 s.
        grouped = cv2.dilate(different.astype(np.uint8), _ellipse(config.structure_group_px))
        count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(grouped, connectivity=8)

        ys, xs = np.nonzero(different)
        owner = labels[ys, xs]
        pixels = np.bincount(owner, minlength=count)
        removed = np.bincount(owner, weights=only_original[ys, xs], minlength=count)
        added = np.bincount(owner, weights=only_suspect[ys, xs], minlength=count)

        min_area = max(
            config.structure_min_area_px, math.pi * (config.structure_footprint_sigmas * matched_softness) ** 2
        )
        big_enough = np.nonzero(pixels >= min_area)[0]
        big_enough = big_enough[big_enough > 0]  # label 0 is the background
        order = np.argsort(owner, kind="stable")
        sorted_owner = owner[order]
        starts = np.searchsorted(sorted_owner, big_enough, side="left")
        ends = np.searchsorted(sorted_owner, big_enough, side="right")

        candidates = []
        for label, first, last in zip(big_enough, starts, ends):
            members = order[first:last]
            x0, x1 = xs[members].min(), xs[members].max() + 1
            y0, y1 = ys[members].min(), ys[members].max() + 1
            total = float(pixels[label])
            if added[label] >= _DIRECTIONAL_SHARE * total:
                subtype, wording = DiffSubtype.ADDED, "Graphic added in suspect"
            elif removed[label] >= _DIRECTIONAL_SHARE * total:
                subtype, wording = DiffSubtype.REMOVED, "Graphic missing from suspect"
            else:
                subtype, wording = DiffSubtype.CHANGED, "Graphic changed"

            box = BoundingBox(float(x0), float(y0), float(x1), float(y1))
            candidates.append(
                DiffCandidate(
                    bbox=box,
                    type=DiffType.GRAPHICS,
                    subtype=subtype,
                    confidence=min(1.0, total / max(1.0, box.area)),
                    detail=wording,
                    magnitude=total,
                )
            )

    logger.info(
        "structure detector finished",
        extra={
            "candidates": len(candidates),
            "slack_px": slack_px,
            "unverifiable_fraction": round(unverifiable_fraction, 4),
            "matched_softness_px": round(matched_softness, 2),
        },
    )
    return StructureResult(candidates, unverifiable_fraction, matched_softness)
