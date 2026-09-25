"""Comparing one artwork against every repeat of it on a sheet.

The reviewer crops or lifts the ONE artwork in the original file, exactly as for any other comparison, and leaves the
suspect file as it is: a sheet printed with the same artwork over and over (a press sheet, a multi-up proof). Every
repeat is found on it (`find_repeats`) and each is compared against the original on its own (`compare_series`) --
alignment, every detector, clustering, all of it -- so a repeat that is itself imperfectly printed is judged on its
own pixels, not blamed on its neighbours.

The findings all come back as ONE list, each saying which repeat it is in. The suspect pane shows the whole sheet, so
a finding sits where its repeat actually is; the original pane shows the original's own whole page (as every other
mode's does), so several repeats' findings can land on the same spot of it -- expected, since they are all about the
same master artwork.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from app.config import AppConfig, SeriesConfig
from app.engine.detectors.regions import padded_window
from app.engine.isolation.boundary_mask import apply_mask, resolve_keep_mask
from app.engine.worker import ComparisonOutcome, run_comparison
from app.logging_config import StageTimer
from app.models.diff import BoundingBox, DiffCluster
from app.models.isolation import IsolationGeometry
from app.models.job import ComparisonMode

logger = logging.getLogger(__name__)


class NoRepeatsFoundError(Exception):
    """No repeat of the original's artwork could be found on the suspect sheet."""


def _edge_picture(rgb: np.ndarray) -> np.ndarray:
    """The edges of a picture, softened, as a float image (what is printed, whatever its colours)."""
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    edges = cv2.Canny(cv2.GaussianBlur(grey, (0, 0), 1.0), 40, 120)
    return cv2.GaussianBlur(edges.astype(np.float32), (0, 0), 1.5)


_MIN_LOCAL_EDGE_STD = 3.0  # a window of the target whose edges vary by less than this is blank paper: matched nowhere


def _blank_windows(target: np.ndarray, kh: int, kw: int) -> np.ndarray:
    """Where a `kh` x `kw` window of `target`, at each of a `matchTemplate(target, template)`-shaped map's positions,
    is too flat to mean anything: normalised correlation over a nearly constant window is numerically unstable and can
    read as a near-perfect match anywhere blank (a wide margin, a gap between repeats) once the template is shrunk
    small enough -- exactly the accident a search across scales is prone to finding, so it must be ruled out, not just
    hoped past."""
    mean = cv2.boxFilter(target, -1, (kw, kh), borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(target * target, -1, (kw, kh), borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.clip(mean_sq - mean * mean, 0, None))
    # `matchTemplate`'s output at (y, x) is the window target[y:y+kh, x:x+kw]; the centred box filter's value at
    # (y + kh//2, x + kw//2) is that same window's, near enough for a threshold mask.
    out_h, out_w = target.shape[0] - kh + 1, target.shape[1] - kw + 1
    return std[kh // 2 : kh // 2 + out_h, kw // 2 : kw // 2 + out_w] < _MIN_LOCAL_EDGE_STD


def _peak(target: np.ndarray, template: np.ndarray) -> tuple[float, float, tuple[int, int], np.ndarray]:
    """(score, how many standard deviations it stands above the correlation's own average, top-left, the correlation
    map with blank windows and everything a caller has already claimed set to that average) of the best fit."""
    correlation = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
    blank = _blank_windows(target, *template.shape)
    floor = float(correlation[~blank].mean()) if (~blank).any() else float(correlation.mean())
    correlation[blank] = floor
    spread = float(correlation[~blank].std()) if (~blank).any() else float(correlation.std())
    _low, score, _low_at, at = cv2.minMaxLoc(correlation)
    return float(score), (float(score) - floor) / (spread + 1e-9), at, correlation


def _settle_scale(target: np.ndarray, template: np.ndarray, config: SeriesConfig, expected_scale: float | None) -> float | None:
    """The one scale, of the template against the target, that the strongest match anywhere on the target picks out."""
    centre = expected_scale if expected_scale and expected_scale > 0 else config.default_expected_scale
    best: tuple[float, float] | None = None  # (peak, scale)
    for step in range(-config.scale_steps_each_way, config.scale_steps_each_way + 1):
        scale = centre * config.scale_step**step
        resized = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        if min(resized.shape) < 12 or resized.shape[0] > target.shape[0] or resized.shape[1] > target.shape[1]:
            continue
        if float(resized.std()) < 1e-3:
            continue
        score, peak, _at, _map = _peak(target, resized)
        if score < config.min_match_score:
            continue
        if best is None or peak > best[0]:
            best = (peak, scale)
    return best[1] if best is not None else None


def find_repeats(
    template_rgb: np.ndarray, sheet_rgb: np.ndarray, config: SeriesConfig, expected_scale: float | None = None
) -> list[BoundingBox]:
    """Every place on `sheet_rgb` that holds a copy of `template_rgb` (the original's one artwork), in reading order.

    The scale is settled once, by the strongest match anywhere on the sheet (near `expected_scale` first, if the two
    files' resolutions say what it is likely to be); every repeat is then found AT that one scale -- they are printed
    copies of each other, so it does not vary from one to the next -- by finding the strongest remaining match,
    blanking out where it was found, and looking again, until nothing left stands out enough or `max_repeats` is
    reached. No rotation is looked for: a multi-up sheet's copies are printed upright to each other.

    Boxes are in `sheet_rgb`'s own pixels, each padded by `config.instance_margin_px` beyond the template's size so
    the comparison's own registration (features, not this search) can settle its exact position.
    """
    with StageTimer(logger, "find_repeats"):
        height, width = sheet_rgb.shape[:2]
        search_scale = min(1.0, config.search_long_edge_px / max(height, width))
        target = _edge_picture(cv2.resize(sheet_rgb, None, fx=search_scale, fy=search_scale, interpolation=cv2.INTER_AREA))
        template_full = _edge_picture(template_rgb)
        template_search = cv2.resize(
            template_full, None, fx=search_scale, fy=search_scale, interpolation=cv2.INTER_AREA if search_scale < 1 else cv2.INTER_LINEAR
        )
        scale = _settle_scale(target, template_search, config, expected_scale)
        if scale is None:
            return []

        template = cv2.resize(template_search, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        th, tw = template.shape
        if th < 8 or tw < 8 or th > target.shape[0] or tw > target.shape[1]:
            return []

        # Every instance is found in ONE correlation map (blanking the picture and matching again is no good: a
        # template that is mostly blank paper matches its own just-blanked footprint about as well as the real thing,
        # and a window of blank paper is already ruled out above). Classic multi-peak non-max suppression instead:
        # take the strongest peak, rule out another top-left within a fraction of the template's size of it (a real
        # neighbour, printed beside this one with a gap between them, sits further off than that; this only stops the
        # same instance being picked a second time a few pixels away), and look again.
        score, peak, at, working = _peak(target, template)
        floor, spread = float(working.mean()), float(working.std())
        radius_x, radius_y = max(4, int(config.suppression_fraction * tw)), max(4, int(config.suppression_fraction * th))
        found: list[tuple[int, int]] = []  # (left, top) in the search copy's pixels
        for _ in range(config.max_repeats):
            if score < config.min_match_score or peak < config.min_match_peak:
                break
            left, top = at
            found.append((left, top))
            x0, y0 = max(0, left - radius_x), max(0, top - radius_y)
            x1, y1 = min(working.shape[1], left + radius_x + 1), min(working.shape[0], top + radius_y + 1)
            working[y0:y1, x0:x1] = floor
            _low, score, _low_at, at = cv2.minMaxLoc(working)
            peak = (score - floor) / (spread + 1e-9)

        if not found:
            return []

        margin = config.instance_margin_px
        boxes = [
            BoundingBox(
                max(0.0, left / search_scale - margin),
                max(0.0, top / search_scale - margin),
                min(float(width), (left + tw) / search_scale + margin),
                min(float(height), (top + th) / search_scale + margin),
            )
            for left, top in found
        ]
        # Reading order: top to bottom, then left to right -- rows are as tall as a repeat, so two in the same row of
        # a slightly uneven grid still round to the same one.
        row_height = max(1.0, (th / search_scale) * 0.6)
        boxes.sort(key=lambda box: (round(box.y0 / row_height), box.x0))
        logger.info("repeats found", extra={"count": len(boxes), "scale": scale})
        return boxes


def _artwork_crop(image: np.ndarray, geometry: IsolationGeometry | None, margin_px: int = 8) -> np.ndarray:
    """The one artwork the reviewer chose in `image` (cropped/lifted), on white, cropped tight to its own box.

    The whole page if nothing was chosen -- the repeat search still works from a whole page, just more slowly.
    """
    if geometry is None or geometry.boundary is None:
        return image
    height, width = image.shape[:2]
    keep = resolve_keep_mask((height, width), geometry)
    ys, xs = np.nonzero(keep)
    if len(xs) == 0:
        return image
    x0, y0 = max(0, int(xs.min()) - margin_px), max(0, int(ys.min()) - margin_px)
    x1, y1 = min(width, int(xs.max()) + 1 + margin_px), min(height, int(ys.max()) + 1 + margin_px)
    return apply_mask(image, keep)[y0:y1, x0:x1]


def compare_series(
    original_image: np.ndarray,
    sheet_rgb: np.ndarray,
    mode: ComparisonMode,
    config: AppConfig,
    original_geometry: IsolationGeometry | None = None,
    expected_scale: float | None = None,
) -> ComparisonOutcome:
    """Compare the one artwork the reviewer chose in `original_image` against every repeat of it found on the
    suspect sheet `sheet_rgb`.

    Each repeat is compared on its own by the ordinary single-pair pipeline (`run_comparison`, with the same
    `original_geometry` every time -- the original is the same file throughout), so a repeat's own misprint is its
    own finding, not blamed on its neighbours. The findings come back as one list in one `ComparisonOutcome`, each
    saying which repeat (of how many) it is in; the suspect pane is the whole sheet, the original pane the one
    artwork every repeat was measured against.
    """
    template = _artwork_crop(original_image, original_geometry)
    boxes = find_repeats(template, sheet_rgb, config.series, expected_scale)
    del template  # only needed to search for the repeats; `run_comparison` below re-derives its own crop from the geometry
    if not boxes:
        raise NoRepeatsFoundError(
            "No repeat of the original's artwork could be found on the suspect sheet. Check that the original was "
            "cropped or lifted to just the one artwork, and that the suspect really holds copies of it."
        )
    total = len(boxes)
    height, width = sheet_rgb.shape[:2]
    all_clusters: list[DiffCluster] = []
    warnings: list[str] = []
    notes = [f"Found {total} repeat{'s' if total != 1 else ''} of the artwork on the suspect sheet; each was compared on its own."]
    profiles: list[str] = []
    graphics_compared = True
    reference_alignment = None
    for number, box in enumerate(boxes, start=1):
        x0, y0, x1, y1 = padded_window(box, 0, (height, width))
        repeat_crop = sheet_rgb[y0:y1, x0:x1]
        try:
            outcome = run_comparison(original_image, repeat_crop, mode, config, original_geometry, None)
        except Exception as exc:  # noqa: BLE001 -- one bad repeat (too little overlap to align) must not sink the rest
            logger.warning("repeat %d/%d could not be compared: %s", number, total, exc)
            warnings.append(f"Repeat {number} of {total}: could not be aligned with the original, so it was skipped.")
            continue
        if reference_alignment is None:
            reference_alignment = outcome.alignment
        profiles.append(outcome.detection_profile)
        graphics_compared = graphics_compared and outcome.graphics_compared
        for cluster in outcome.clusters:
            shifted = cluster.bbox_in_suspect
            cluster.bbox_in_suspect = BoundingBox(shifted.x0 + x0, shifted.y0 + y0, shifted.x1 + x0, shifted.y1 + y0)
            cluster.detail = f"Repeat {number} of {total}: {cluster.detail}"
            all_clusters.append(cluster)
        warnings.extend(f"Repeat {number} of {total}: {warning}" for warning in outcome.warnings)

    if reference_alignment is None:
        raise NoRepeatsFoundError("Every repeat that was found failed to align with the original; none could be compared.")

    all_clusters.sort(key=lambda cluster: (cluster.bbox_in_suspect.y0, cluster.bbox_in_suspect.x0))
    for number, cluster in enumerate(all_clusters, start=1):
        cluster.id = number

    return ComparisonOutcome(
        alignment=reference_alignment,
        candidates=[],
        clusters=all_clusters,
        aligned_original=original_image,
        aligned_suspect=sheet_rgb,
        warnings=warnings,
        detection_profile="exact" if profiles and all(p == "exact" for p in profiles) else "tolerant",
        used_change_regions=False,
        mode=mode,
        compared_outline=None,
        graphics_compared=graphics_compared,
        notes=notes,
    )
