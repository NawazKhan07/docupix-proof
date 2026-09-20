"""Shared helper: turn a boolean difference mask into bounding boxes.

Used by the pixel-level detectors (colour, graphics, Braille) so they agree
on how a mask becomes regions instead of each rolling its own.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from app.models.diff import BoundingBox


def padded_window(box: BoundingBox, pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """`box` grown by `pad` pixels on every side and clipped to an image of `shape`, as (x0, y0, x1, y1)."""
    height, width = shape[:2]
    return max(0, int(box.x0) - pad), max(0, int(box.y0) - pad), min(width, math.ceil(box.x1) + pad), min(height, math.ceil(box.y1) + pad)


def _min_difference_over_shifts(
    reference: np.ndarray, other: np.ndarray, radius_px: int
) -> np.ndarray:
    """Per pixel, the smallest |reference - other| over every shift within the radius.

    Deliberately compares actual shifted values rather than the
    neighbourhood's min/max range. A range test treats every value between
    the local darkest and lightest as "present nearby", which in a text area
    spans ink to paper and therefore excuses *any* difference -- it blinded
    the detector precisely where artwork detail matters most.
    """
    reference_signed = reference.astype(np.int16)
    padded = cv2.copyMakeBorder(
        other, radius_px, radius_px, radius_px, radius_px, cv2.BORDER_REPLICATE
    ).astype(np.int16)

    height, width = reference.shape
    best = np.full(reference.shape, 255, dtype=np.int16)
    for dy in range(2 * radius_px + 1):
        for dx in range(2 * radius_px + 1):
            shifted = padded[dy : dy + height, dx : dx + width]
            np.minimum(best, np.abs(reference_signed - shifted), out=best)
    return best


def unmatched_within_tolerance(
    first: np.ndarray, second: np.ndarray, radius_px: int, intensity_tolerance: int
) -> np.ndarray:
    """Pixels with no near-equal counterpart within `radius_px` in the other image.

    Registration is never perfect, so a stroke sitting half a pixel off would
    otherwise mismatch along its whole length. Allowing a small positional
    search absorbs that while still flagging content that is genuinely
    absent or different, and it assumes nothing about whether the artwork is
    dark-on-light or light-on-dark.
    """
    if radius_px <= 0:
        return np.abs(first.astype(np.int16) - second.astype(np.int16)) > intensity_tolerance

    return (_min_difference_over_shifts(first, second, radius_px) > intensity_tolerance) | (
        _min_difference_over_shifts(second, first, radius_px) > intensity_tolerance
    )


def mask_to_boxes(
    mask: np.ndarray,
    min_area_px: int,
    edge_tolerance_px: int = 0,
    min_fill_ratio: float = 0.0,
) -> list[tuple[BoundingBox, int]]:
    """Connected components of `mask`, as (box, pixel_area), largest first.

    `edge_tolerance_px` morphologically opens the mask first, which drops the
    hairline mismatches that sub-pixel resampling leaves along every edge
    when the two files are rotated relative to each other. Without it a
    single 1 px halo traces a whole shape's outline and connected components
    returns it as one enormous region.

    Components smaller than `min_area_px` are then dropped: at working
    resolution a handful of stray pixels is rendering noise between two
    files, not a difference a reviewer should be asked to adjudicate.
    """
    mask = mask.astype(np.uint8)
    if edge_tolerance_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * edge_tolerance_px + 1, 2 * edge_tolerance_px + 1)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for index in range(1, count):  # 0 is the background component
        x, y, width, height, area = stats[index]
        if area < min_area_px:
            continue
        # A hairline that traces a shape's outline has a bounding box the
        # size of the whole shape but almost no pixels in it. Reporting that
        # as one region is worse than useless -- a 1.5 degree rotation
        # produced a 123,000 px "difference" holding a few thousand pixels.
        # A genuine change (a blob, a line, a cluster of glyphs) fills its
        # own box; an outline does not.
        if min_fill_ratio > 0 and area < min_fill_ratio * width * height:
            continue
        boxes.append((BoundingBox(float(x), float(y), float(x + width), float(y + height)), int(area)))
    return sorted(boxes, key=lambda item: item[1], reverse=True)
