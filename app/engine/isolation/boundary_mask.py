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

from app.engine.isolation.erase_mask import rasterize_erase_regions
from app.models.isolation import IsolationGeometry, NormalizedPolygon, NormalizedRect


def boundary_to_mask(
    shape: tuple[int, int], boundary: NormalizedRect | NormalizedPolygon, inset_px: float = 0.0
) -> np.ndarray:
    """A boolean mask (True = inside the artwork boundary, keep).

    `inset_px` starts a polygon's mask that far inside its edge: the edge is drawn over as a
    line twice that wide, which is far cheaper than eroding a page-sized mask and just as round.
    """
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    if isinstance(boundary, NormalizedRect):
        x0, y0, x1, y1 = boundary.to_pixels(width, height)
        mask[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)] = 255
    else:
        points = np.array(boundary.to_pixels(width, height), dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        if inset_px > 0:
            cv2.polylines(mask, [points], True, 0, thickness=max(1, int(round(2 * inset_px))))
    return mask.astype(bool)


def boundary_mask(shape: tuple[int, int], geometry: IsolationGeometry | None) -> np.ndarray | None:
    """The artwork area chosen for one file, or None if no area was chosen (the whole page)."""
    if geometry is None or geometry.boundary is None:
        return None
    inset_px = geometry.edge_inset_norm * max(shape)
    return boundary_to_mask(shape, geometry.boundary, inset_px)


def erased_mask(shape: tuple[int, int], geometry: IsolationGeometry | None) -> np.ndarray | None:
    """What was erased by hand from one file, or None if nothing was."""
    if geometry is None or not geometry.erase_regions:
        return None
    return rasterize_erase_regions(shape, geometry.erase_regions, geometry.erase_brush_radius_norm)


def resolve_keep_mask(shape: tuple[int, int], geometry: IsolationGeometry) -> np.ndarray:
    """The final boolean mask for one file: inside the boundary AND not erased."""
    keep = boundary_mask(shape, geometry)
    if keep is None:
        keep = np.ones(shape, dtype=bool)
    erased = erased_mask(shape, geometry)
    if erased is not None:
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
    fill = np.asarray(background, dtype=image.dtype) if image.ndim == 3 else np.asarray(background[0], dtype=image.dtype)
    # in place: `result[~keep_mask] = ...` lists every masked pixel's index first (548 MB on a 100 MP page)
    np.copyto(result, fill, where=~keep_mask[..., None] if image.ndim == 3 else ~keep_mask)
    return result
