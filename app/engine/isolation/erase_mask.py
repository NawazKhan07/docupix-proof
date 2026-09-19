"""Erasing parts of the artwork (SRS FR-5.1, Architecture §3.3.4).

A part to leave out is either a *rectangle* (selected like a crop, and adjustable until it is
applied) or a *freehand stroke* (a polyline painted with a round brush). Both are rasterized
onto a mask rather than baked into the image pixels, so `boundary_mask.resolve_keep_mask` can
combine them with the artwork boundary before anything is handed to a detector.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.models.isolation import NormalizedPolygon, NormalizedRect


def rasterize_erase_regions(
    shape: tuple[int, int], regions: list[NormalizedRect | NormalizedPolygon], brush_radius_norm: float
) -> np.ndarray:
    """A boolean mask (True = erased): rectangles filled, polyline strokes painted with the brush."""
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    radius_px = max(1, round(brush_radius_norm * max(width, height)))
    for region in regions:
        if isinstance(region, NormalizedRect):
            x0, y0, x1, y1 = region.to_pixels(width, height)
            mask[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)] = 255
            continue
        pixel_points = np.array(region.to_pixels(width, height), dtype=np.int32)
        if len(pixel_points) == 0:
            continue
        if len(pixel_points) == 1:
            cv2.circle(mask, tuple(pixel_points[0]), radius_px, 255, -1)
        else:
            cv2.polylines(mask, [pixel_points], isClosed=False, color=255, thickness=radius_px * 2)
    return mask.astype(bool)

