"""Where do two aligned images differ at all? (Architecture §6.3, coarse-to-fine).

Most of two artwork files is normally identical, so running CIEDE2000 and
SSIM over the whole page spends nearly all its time confirming that nothing
changed. A plain pixel difference is orders of magnitude cheaper, and tells
the expensive detectors exactly where to look -- at full resolution, which
also stops thin strokes being averaged away by a downscale.

If the differences are so widespread that the regions would cover much of the
page (a scan, or a pair that is not registered), region-limited work stops
paying off and this returns None so the caller uses the whole-page pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import DetectionConfig
from app.logging_config import StageTimer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChangeRegion:
    """A rectangle of the shared frame, in pixels, including its context margin."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return self.width * self.height


def changed_pixels(
    original: np.ndarray, suspect: np.ndarray, threshold: int, keep_mask: np.ndarray | None = None
) -> np.ndarray:
    """A boolean mask of pixels where any channel differs by more than `threshold`."""
    difference = cv2.absdiff(original, suspect)
    blue, green, red = cv2.split(difference)
    strongest = cv2.max(cv2.max(blue, green), red)
    changed = strongest > threshold
    if keep_mask is not None:
        changed &= keep_mask
    return changed


def find_change_regions(
    original: np.ndarray,
    suspect: np.ndarray,
    config: DetectionConfig,
    keep_mask: np.ndarray | None = None,
    threshold: int | None = None,
) -> list[ChangeRegion] | None:
    """The regions of the page that differ, or None if region-limited work won't help.

    An empty list means the two images are identical (within the threshold):
    there is nothing for any detector to do.
    """
    with StageTimer(logger, "find_change_regions"):
        height, width = original.shape[:2]
        changed = changed_pixels(
            original,
            suspect,
            config.region_diff_threshold if threshold is None else threshold,
            keep_mask,
        )
        if not changed.any():
            return []

        # Group on a coarse grid. Dilating by one block then sampling at the
        # block stride marks a cell if ANY changed pixel falls inside it, and
        # keeps everything after this on an image 1/64th the size.
        block = config.region_block_px
        occupied = cv2.dilate(changed.astype(np.uint8), np.ones((block, block), np.uint8))
        coarse = occupied[::block, ::block]

        margin_cells = max(1, -(-config.region_margin_px // block))
        spread = cv2.dilate(
            coarse, np.ones((2 * margin_cells + 1, 2 * margin_cells + 1), np.uint8)
        )
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(spread, connectivity=8)

        regions = []
        for index in range(1, count):  # 0 is the background
            x, y, w, h, _area = stats[index]
            regions.append(
                ChangeRegion(
                    x0=max(0, int(x) * block),
                    y0=max(0, int(y) * block),
                    x1=min(width, int(x + w) * block),
                    y1=min(height, int(y + h) * block),
                )
            )

        covered = sum(region.area for region in regions) / float(width * height)
        logger.info(
            "change regions found",
            extra={"regions": len(regions), "covered_fraction": round(covered, 4)},
        )
        if covered > config.region_max_area_fraction:
            return None
        return regions
