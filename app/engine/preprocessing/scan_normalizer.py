"""Scan-vs-Digital preprocessing (SRS FR-6.3, Architecture §3.3.4).

Flattens both files to the same black-on-white representation so a scan can
be compared against a digital original without every lighting and ink
variation reading as a difference. Colour comparison is disabled separately,
by the mode's `DetectorOptions`, so it is skipped rather than merely hidden.
"""

from __future__ import annotations

import cv2
import numpy as np

# The illumination field is estimated on a small copy of the page. Working
# at this size keeps the estimate smooth and makes a very wide neighbourhood
# affordable, which matters: any background estimator whose window is
# narrower than a solid artwork element will read that element as background
# and erase it. An early version used a fixed 31 px window and deleted a
# 100 px solid roundel outright.
_ILLUMINATION_LONG_EDGE_PX = 256
_ILLUMINATION_RADIUS_DIVISOR = 8


def _illumination_field(gray: np.ndarray) -> np.ndarray:
    """A smooth estimate of the page's lighting, ignoring artwork detail."""
    height, width = gray.shape
    scale = _ILLUMINATION_LONG_EDGE_PX / max(height, width)
    small = (
        cv2.resize(gray, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        if scale < 1
        else gray.copy()
    )

    # Local maximum: on a page, the brightest value nearby is the paper.
    radius = max(2, max(small.shape) // _ILLUMINATION_RADIUS_DIVISOR)
    paper = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
    paper = cv2.GaussianBlur(paper, (0, 0), max(1.0, radius / 2))
    return cv2.resize(paper, (width, height), interpolation=cv2.INTER_LINEAR)


def normalize_for_scan(image: np.ndarray) -> np.ndarray:
    """Return an RGB image with a white background and pure-black foreground.

    The illumination is estimated and divided out first: a scan is rarely lit
    evenly, and thresholding a vignetted page directly would erase pale
    corners or keep grey ones.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    flattened = cv2.divide(gray, _illumination_field(gray), scale=255)
    _, binary = cv2.threshold(flattened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


def is_effectively_blank(image: np.ndarray) -> bool:
    """Whether a normalized image carries essentially no foreground."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return bool(np.count_nonzero(gray < 128) < 0.0001 * gray.size)
