"""Freehand erase-mask tool (SRS FR-5.1, Architecture §3.3.4).

Strokes are rasterized onto an alpha mask rather than baked into the image
pixels directly, so `boundary_mask.resolve_keep_mask` can combine them with
the artwork boundary before anything is handed to a detector.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.models.isolation import NormalizedPolygon


def rasterize_erase_strokes(
    shape: tuple[int, int], strokes: list[NormalizedPolygon], brush_radius_norm: float
) -> np.ndarray:
    """A boolean mask (True = erased) painted from freehand stroke polylines."""
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    radius_px = max(1, round(brush_radius_norm * max(width, height)))
    for stroke in strokes:
        pixel_points = np.array(stroke.to_pixels(width, height), dtype=np.int32)
        if len(pixel_points) == 0:
            continue
        if len(pixel_points) == 1:
            cv2.circle(mask, tuple(pixel_points[0]), radius_px, 255, -1)
        else:
            cv2.polylines(
                mask, [pixel_points], isClosed=False, color=255, thickness=radius_px * 2
            )
    return mask.astype(bool)
