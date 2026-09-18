"""Structural/graphics difference detection (Architecture §3.3.5).

SSIM compares local structure rather than absolute pixel values, so it
catches a moved or reshaped logo without firing on the small intensity
differences that separate two renderings of the same artwork.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from app.config import DetectionConfig
from app.engine.detectors.regions import mask_to_boxes, unmatched_within_tolerance
from app.logging_config import StageTimer
from app.models.diff import DiffCandidate, DiffSubtype, DiffType

logger = logging.getLogger(__name__)


def detect(
    original: np.ndarray,
    suspect: np.ndarray,
    config: DetectionConfig,
    compare_mask: np.ndarray | None = None,
) -> list[DiffCandidate]:
    """Structural differences between two images already in the shared aligned space."""
    with StageTimer(logger, "detector_graphics"):
        original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        suspect_gray = cv2.cvtColor(suspect, cv2.COLOR_RGB2GRAY)

        _score, similarity_map = structural_similarity(
            original_gray, suspect_gray, full=True, data_range=255
        )
        dissimilar = similarity_map < config.ssim_threshold

        # SSIM's window smears a one-pixel edge shift across its whole
        # neighbourhood, so require that a flagged pixel also has genuinely
        # unmatched content near it rather than merely displaced content.
        dissimilar &= unmatched_within_tolerance(
            original_gray,
            suspect_gray,
            config.registration_tolerance_px,
            config.registration_intensity_tolerance,
        )
        if compare_mask is not None:
            dissimilar &= compare_mask

        candidates = []
        for box, _area in mask_to_boxes(
            dissimilar, config.ssim_min_area_px, config.edge_tolerance_px, config.min_fill_ratio
        ):
            region = similarity_map[int(box.y0) : int(box.y1), int(box.x0) : int(box.x1)]
            worst_similarity = float(region.min()) if region.size else 1.0
            candidates.append(
                DiffCandidate(
                    bbox=box,
                    type=DiffType.GRAPHICS,
                    subtype=DiffSubtype.CHANGED,
                    confidence=float(np.clip(1.0 - worst_similarity, 0.0, 1.0)),
                    magnitude=float(np.clip(1.0 - worst_similarity, 0.0, 2.0)),
                    detail=f"Graphic changed (similarity {worst_similarity:.2f})",
                )
            )

    logger.info("graphics detector finished", extra={"candidates": len(candidates)})
    return candidates
