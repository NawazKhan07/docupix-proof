"""Central artwork-boundary masking (SRS FR-2.1, Architecture §3.3.4:
"everything outside the resolved artwork boundary is hard-masked out before
any detector runs -- this is enforced once, centrally, so no detector can
accidentally compare furniture").

Every isolation path -- manual crop, click-to-trace dieline polygon, and
(from Phase 3) automatic dieline detection -- resolves to the same
`IsolationGeometry`, so they all funnel through `resolve_keep_mask` /
`apply_mask` here rather than each detector doing its own boundary check.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.engine.isolation.erase_mask import rasterize_erase_strokes
from app.models.isolation import IsolationGeometry, NormalizedPolygon, NormalizedRect


def boundary_to_mask(
    shape: tuple[int, int], boundary: NormalizedRect | NormalizedPolygon
) -> np.ndarray:
    """A boolean mask (True = inside the artwork boundary, keep)."""
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    if isinstance(boundary, NormalizedRect):
        x0, y0, x1, y1 = boundary.to_pixels(width, height)
        mask[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)] = 255
    else:
        points = np.array(boundary.to_pixels(width, height), dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
    return mask.astype(bool)


def resolve_keep_mask(shape: tuple[int, int], geometry: IsolationGeometry) -> np.ndarray:
    """The final boolean mask for one file: inside the boundary AND not erased."""
    if geometry.boundary is None:
        keep = np.ones(shape, dtype=bool)
    else:
        keep = boundary_to_mask(shape, geometry.boundary)
    if geometry.erase_regions:
        erased = rasterize_erase_strokes(
            shape, geometry.erase_regions, geometry.erase_brush_radius_norm
        )
        keep = keep & ~erased
    return keep


def apply_mask(
    image: np.ndarray, keep_mask: np.ndarray, background: tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """Hard-mask everything outside `keep_mask` to `background`.

    This is the single place isolation results get applied to pixels, so no
    detector downstream can accidentally see furniture (SRS FR-2.1).
    """
    result = image.copy()
    result[~keep_mask] = background
    return result
