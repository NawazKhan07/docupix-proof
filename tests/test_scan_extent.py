"""A scan with no area set is an item laid on the glass: only the item is compared (`auto_area.scan_extent`), so what
of a digital proof sheet lies beyond it -- the sheet's other artworks, its dimension lines and notes -- is never
reported missing."""

from __future__ import annotations

import cv2
import numpy as np

from app.engine.isolation.auto_area import scan_extent


def _scanner_page() -> np.ndarray:
    """A scanner's page: a label (dense print) with a barcode just below it, the glass's thin edge lines, and specks."""
    page = np.full((3300, 2500, 3), 246, np.uint8)
    cv2.rectangle(page, (900, 1500), (1700, 2700), (30, 70, 160), -1)  # the label's print
    for x in range(920, 1240, 14):  # a barcode beside it
        cv2.rectangle(page, (x, 2760), (x + 6, 2880), (20, 20, 20), -1)
    cv2.line(page, (4, 0), (4, 3299), (40, 40, 40), 3)  # the glass's edges
    cv2.line(page, (2490, 0), (2490, 3299), (60, 60, 60), 4)
    for x, y in ((150, 160), (2300, 300), (200, 3150)):  # dust
        cv2.circle(page, (x, y), 9, (90, 90, 90), -1)
    return page


def test_the_item_and_what_is_beside_it_are_the_extent():
    extent = scan_extent(_scanner_page())
    assert extent is not None
    assert extent[2000, 1300] and extent[2820, 1000]  # the label and its barcode
    assert not extent[300, 300] and not extent[1500, 4] and not extent[3150, 200]  # glass, edge, dust
    assert extent.mean() < 0.3


def test_a_scan_that_is_all_artwork_is_not_narrowed():
    page = np.full((2000, 1500, 3), 246, np.uint8)
    cv2.rectangle(page, (20, 20), (1480, 1980), (30, 70, 160), -1)
    assert scan_extent(page) is None
