"""Coarse-to-fine detection: find where two aligned images differ, cheaply."""

from __future__ import annotations

import time

import cv2
import numpy as np

from app.config import DetectionConfig
from app.engine.detectors.change_regions import changed_pixels, find_change_regions

CONFIG = DetectionConfig()


def _page(height: int = 1200, width: int = 1600) -> np.ndarray:
    page = np.full((height, width, 3), 220, dtype=np.uint8)
    for row in range(20):
        cv2.putText(page, f"BATCH {row:04d} EXP 2027-06", (80, 60 + row * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 60), 2)
    return page


def test_identical_images_have_no_regions():
    page = _page()
    assert find_change_regions(page, page.copy(), CONFIG) == []


def test_a_single_change_gives_one_region_with_context_around_it():
    original = _page()
    suspect = original.copy()
    cv2.rectangle(suspect, (1000, 600), (1100, 660), (200, 30, 40), -1)

    regions = find_change_regions(original, suspect, CONFIG)

    assert len(regions) == 1
    region = regions[0]
    assert region.x0 <= 1000 - CONFIG.region_margin_px + CONFIG.region_block_px
    assert region.x1 >= 1100 + CONFIG.region_margin_px - CONFIG.region_block_px
    assert region.y0 <= 600 and region.y1 >= 660  # and always contains the change


def test_every_changed_pixel_lies_inside_some_region():
    """The regions are only useful if nothing that changed can fall outside them."""
    original = _page()
    suspect = original.copy()
    for x, y in [(100, 100), (900, 300), (1400, 1000), (200, 1100)]:
        cv2.circle(suspect, (x, y), 15, (0, 0, 255), -1)

    regions = find_change_regions(original, suspect, CONFIG)

    changed = changed_pixels(original, suspect, CONFIG.region_diff_threshold)
    ys, xs = np.nonzero(changed)
    for x, y in zip(xs, ys):
        assert any(r.x0 <= x < r.x1 and r.y0 <= y < r.y1 for r in regions)


def test_distant_changes_stay_separate_and_near_ones_merge():
    original = _page()
    suspect = original.copy()
    cv2.circle(suspect, (100, 100), 12, (0, 0, 255), -1)
    cv2.circle(suspect, (1500, 1100), 12, (0, 0, 255), -1)  # far away
    assert len(find_change_regions(original, suspect, CONFIG)) == 2

    cv2.circle(suspect, (140, 100), 12, (0, 0, 255), -1)  # right beside the first
    assert len(find_change_regions(original, suspect, CONFIG)) == 2  # merged into it, not a third


def test_a_pair_that_differs_everywhere_is_left_to_the_whole_page_pass():
    """A scan or an unregistered pair: region-limited work stops paying off."""
    original = _page()
    rng = np.random.default_rng(1)
    noisy = np.clip(original.astype(int) + rng.integers(-60, 60, original.shape), 0, 255).astype(np.uint8)
    assert find_change_regions(original, noisy, CONFIG) is None


def test_changes_outside_the_keep_mask_are_ignored():
    """FR-2.1: furniture outside the artwork boundary must not open a region."""
    original = _page()
    suspect = original.copy()
    cv2.rectangle(suspect, (1000, 600), (1100, 660), (200, 30, 40), -1)
    keep = np.zeros(original.shape[:2], dtype=bool)
    keep[:300, :300] = True  # nowhere near the change

    assert find_change_regions(original, suspect, CONFIG, keep_mask=keep) == []


def test_differences_below_the_threshold_are_not_changes():
    original = _page()
    faint = np.clip(original.astype(int) + 5, 0, 255).astype(np.uint8)
    assert find_change_regions(original, faint, CONFIG) == []
    assert find_change_regions(original, faint, CONFIG, threshold=3) is None  # ...unless asked


def test_regions_stay_inside_the_image():
    original = _page()
    suspect = original.copy()
    cv2.rectangle(suspect, (0, 0), (30, 30), (0, 0, 255), -1)  # in the very corner
    cv2.rectangle(suspect, (1570, 1170), (1599, 1199), (0, 0, 255), -1)

    for region in find_change_regions(original, suspect, CONFIG):
        assert 0 <= region.x0 < region.x1 <= 1600
        assert 0 <= region.y0 < region.y1 <= 1200


def test_finding_regions_on_a_full_sheet_is_fast():
    """The whole point: a cheap first pass over an 18 MP sheet (measured ~110 ms)."""
    original = np.full((4965, 3660, 3), 220, dtype=np.uint8)
    suspect = original.copy()
    cv2.rectangle(suspect, (1000, 2000), (1100, 2100), (0, 0, 255), -1)

    started = time.perf_counter()
    regions = find_change_regions(original, suspect, CONFIG)
    elapsed = time.perf_counter() - started

    assert len(regions) == 1
    assert elapsed < 1.0
