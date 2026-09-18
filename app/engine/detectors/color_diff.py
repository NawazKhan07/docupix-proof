"""Colour difference detection (Architecture §3.3.5).

Uses CIEDE2000 in Lab rather than a raw RGB delta so the numbers track what
a reviewer would actually perceive as a colour shift: an equal RGB step is
far more visible in some parts of the gamut than others.

Skipped entirely when the mode sets `compare_color=False` (FR-6.3) -- the
caller simply never invokes this module, rather than it being filtered out
of the report later.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab

from app.config import DetectionConfig
from app.engine.detectors.regions import mask_to_boxes, unmatched_within_tolerance
from app.logging_config import StageTimer
from app.models.diff import DiffCandidate, DiffSubtype, DiffType

logger = logging.getLogger(__name__)


def _downscale_factor(shape: tuple[int, int], target_long_edge: int) -> float:
    long_edge = max(shape[:2])
    if long_edge <= target_long_edge:
        return 1.0
    return target_long_edge / long_edge


def _colour_unmatched_nearby(
    original: np.ndarray, suspect: np.ndarray, config: DetectionConfig
) -> np.ndarray:
    """Whether each pixel's colour is absent from the other image's neighbourhood.

    Uses OpenCV's uint8 Lab (cheap for morphology) rather than the float Lab
    the delta-E maths needs; a pixel counts as changed if any one channel has
    no near match nearby.
    """
    original_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB)
    suspect_lab = cv2.cvtColor(suspect, cv2.COLOR_RGB2LAB)
    unmatched = np.zeros(original.shape[:2], dtype=bool)
    for channel in range(3):
        unmatched |= unmatched_within_tolerance(
            original_lab[:, :, channel],
            suspect_lab[:, :, channel],
            config.registration_tolerance_px,
            config.registration_intensity_tolerance,
        )
    return unmatched


def detect(
    original: np.ndarray,
    suspect: np.ndarray,
    config: DetectionConfig,
    compare_mask: np.ndarray | None = None,
) -> list[DiffCandidate]:
    """Colour differences between two images already in the shared aligned space."""
    with StageTimer(logger, "detector_color"):
        factor = _downscale_factor(original.shape, config.color_working_long_edge_px)
        if factor < 1.0:
            size = (int(original.shape[1] * factor), int(original.shape[0] * factor))
            small_original = cv2.resize(original, size, interpolation=cv2.INTER_AREA)
            small_suspect = cv2.resize(suspect, size, interpolation=cv2.INTER_AREA)
            small_mask = (
                cv2.resize(compare_mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
                if compare_mask is not None
                else None
            )
        else:
            small_original, small_suspect, small_mask = original, suspect, compare_mask

        delta_e = deltaE_ciede2000(rgb2lab(small_original), rgb2lab(small_suspect))
        changed = delta_e > config.color_delta_e_threshold

        # Same registration problem as the graphics channel: a shifted edge
        # puts ink against background and reads as a maximal colour change.
        # Testing each Lab channel for a near match nearby tolerates that,
        # while a genuinely recoloured area has no matching colour nearby --
        # including an equal-lightness change like red to green, which the
        # a/b channels catch even though L alone would not.
        changed &= _colour_unmatched_nearby(small_original, small_suspect, config)
        if small_mask is not None:
            changed &= small_mask

        # Areas are compared at the downscaled resolution, so scale the
        # minimum with it rather than silently demanding far larger regions.
        min_area = max(4, int(config.color_min_area_px * factor * factor))
        candidates = []
        for box, _area in mask_to_boxes(
            changed, min_area, config.edge_tolerance_px, config.min_fill_ratio
        ):
            region = delta_e[int(box.y0) : int(box.y1), int(box.x0) : int(box.x1)]
            peak_delta = float(region.max()) if region.size else 0.0
            candidates.append(
                DiffCandidate(
                    bbox=box.scaled(1 / factor),
                    type=DiffType.COLOUR,
                    subtype=DiffSubtype.CHANGED,
                    confidence=min(1.0, peak_delta / (config.color_delta_e_threshold * 4)),
                    # ASCII label deliberately: this string reaches a Windows
                    # console log handler, which is cp1252 and cannot encode a
                    # literal delta.
                    detail=f"dE {peak_delta:.1f}",
                )
            )

    logger.info("colour detector finished", extra={"candidates": len(candidates)})
    return candidates
