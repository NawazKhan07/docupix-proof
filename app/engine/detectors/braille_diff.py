"""Braille dot-pattern comparison (Architecture §3.3.5, FR-11.1).

Braille on packaging artwork is marked as coloured dots, so detection is a
colour-blob problem rather than an OCR one: find dot centroids in both
files (already in the shared aligned space) and match them by nearest
neighbour.

PENDING SIGN-OFF (SRS §7): the dot colour convention used in our source
artwork is an open item, so the HSV ranges live in `DetectionConfig`. The default is red
(measured on the only real Braille files so far) and magenta/pink, not a signed-off value.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from app.config import DetectionConfig
from app.logging_config import StageTimer
from app.models.diff import BoundingBox, DiffCandidate, DiffSubtype, DiffType
from app.models.result import BrailleComparison

logger = logging.getLogger(__name__)


def detect_dots(image: np.ndarray, config: DetectionConfig) -> np.ndarray:
    """Centroids of the coloured Braille dots, as an (N, 2) array.

    A dot is a blob in the dot colour that is small and roundish; a letter, a logo or a panel in the same
    colour is not one.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lower, upper in config.braille_hsv_ranges:
        mask |= cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for index in range(1, count):  # 0 is the background
        area = stats[index, cv2.CC_STAT_AREA]
        width, height = stats[index, cv2.CC_STAT_WIDTH], stats[index, cv2.CC_STAT_HEIGHT]
        if not config.braille_dot_min_area_px <= area <= config.braille_dot_max_area_px:
            continue
        if max(width, height) > config.braille_dot_max_aspect * min(width, height):
            continue
        if area < config.braille_dot_min_fill * width * height:
            continue
        x, y = stats[index, cv2.CC_STAT_LEFT], stats[index, cv2.CC_STAT_TOP]
        blob = (_labels[y : y + height, x : x + width] == index).astype(np.uint8)
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = cv2.arcLength(max(contours, key=cv2.contourArea), True) if contours else 0.0
        if perimeter <= 0 or 4 * np.pi * area / perimeter**2 < config.braille_dot_min_circularity:
            continue
        blobs.append((area, centroids[index]))
    if len(blobs) >= config.braille_dot_min_count_for_size:
        median = float(np.median([area for area, _centre in blobs]))
        factor = config.braille_dot_size_factor
        blobs = [(area, centre) for area, centre in blobs if median / factor <= area <= median * factor]
    dots = [centre for _area, centre in blobs]
    return np.array(dots, dtype=np.float64) if dots else np.empty((0, 2))


def _pair_dots(
    original_dots: np.ndarray, suspect_dots: np.ndarray, tolerance: float
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedy nearest-neighbour pairing within a generous search radius.

    The radius is wider than the move tolerance so that a dot which shifted
    is reported as one moved dot rather than as a missing/added pair.
    """
    if len(original_dots) == 0 or len(suspect_dots) == 0:
        return [], list(range(len(original_dots))), list(range(len(suspect_dots)))

    distances = np.linalg.norm(original_dots[:, None, :] - suspect_dots[None, :, :], axis=2)
    search_radius = tolerance * 3

    pairs: list[tuple[int, int, float]] = []
    used_original: set[int] = set()
    used_suspect: set[int] = set()
    for original_index, suspect_index in zip(*np.unravel_index(np.argsort(distances, axis=None), distances.shape)):
        distance = distances[original_index, suspect_index]
        if distance > search_radius:
            break
        if original_index in used_original or suspect_index in used_suspect:
            continue
        used_original.add(int(original_index))
        used_suspect.add(int(suspect_index))
        pairs.append((int(original_index), int(suspect_index), float(distance)))

    missing = [index for index in range(len(original_dots)) if index not in used_original]
    added = [index for index in range(len(suspect_dots)) if index not in used_suspect]
    return pairs, missing, added


def _dot_box(center: np.ndarray, radius: float = 6.0) -> BoundingBox:
    x, y = float(center[0]), float(center[1])
    return BoundingBox(x - radius, y - radius, x + radius, y + radius)


def _region(*dot_sets: np.ndarray) -> BoundingBox | None:
    """The box round every dot found (None if there are none)."""
    dots = [d for d in dot_sets if len(d)]
    if not dots:
        return None
    every = np.vstack(dots)
    (x0, y0), (x1, y1) = every.min(axis=0), every.max(axis=0)
    return _dot_box(np.array([x0, y0])).union(_dot_box(np.array([x1, y1])))


def compare(
    original: np.ndarray, suspect: np.ndarray, config: DetectionConfig
) -> tuple[list[DiffCandidate], BrailleComparison]:
    """Compare Braille dot patterns between two aligned images (FR-11.1)."""
    with StageTimer(logger, "detector_braille"):
        original_dots = detect_dots(original, config)
        suspect_dots = detect_dots(suspect, config)
        pairs, missing, added = _pair_dots(
            original_dots, suspect_dots, config.braille_match_tolerance_px
        )

        candidates: list[DiffCandidate] = []
        moved = 0
        for original_index, suspect_index, distance in pairs:
            if distance > config.braille_match_tolerance_px:
                moved += 1
                candidates.append(
                    DiffCandidate(
                        bbox=_dot_box(original_dots[original_index]).union(
                            _dot_box(suspect_dots[suspect_index])
                        ),
                        type=DiffType.BRAILLE,
                        subtype=DiffSubtype.MOVED,
                        confidence=1.0,
                        detail=f"Braille dot moved {distance:.0f} px",
                    )
                )
        for index in missing:
            candidates.append(
                DiffCandidate(
                    bbox=_dot_box(original_dots[index]),
                    type=DiffType.BRAILLE,
                    subtype=DiffSubtype.REMOVED,
                    confidence=1.0,
                    detail="Braille dot missing from suspect",
                )
            )
        for index in added:
            candidates.append(
                DiffCandidate(
                    bbox=_dot_box(suspect_dots[index]),
                    type=DiffType.BRAILLE,
                    subtype=DiffSubtype.ADDED,
                    confidence=1.0,
                    detail="Braille dot added in suspect",
                )
            )

        comparison = BrailleComparison(
            original_dot_count=len(original_dots),
            suspect_dot_count=len(suspect_dots),
            matched=len(pairs) - moved,
            missing=len(missing),
            added=len(added),
            moved=moved,
            region=_region(original_dots, suspect_dots),
        )

    logger.info(
        "braille detector finished",
        extra={
            "original_dots": comparison.original_dot_count,
            "suspect_dots": comparison.suspect_dot_count,
            "missing": comparison.missing,
            "added": comparison.added,
            "moved": comparison.moved,
        },
    )
    return candidates, comparison
