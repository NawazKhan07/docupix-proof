"""Rectangular crop + Save-Crop auto-rotate/align (SRS FR-3.1/FR-3.2).

Rotation is estimated from the minimum-area bounding rectangle of the
crop's foreground content (Architecture §3.3.3: "estimate rotation via
minimum-area-rect of the artwork mask"), then warp-corrected so the
content's dominant edges become axis-aligned. This is what lets two
independently-drawn crops (one per file, FR-3.1) end up consistently
oriented before Phase 3's alignment has to reconcile them.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.models.isolation import NormalizedRect


def crop_to_rect(image: np.ndarray, rect: NormalizedRect) -> np.ndarray:
    """Crop `image` (H, W, 3) to the pixel region described by a normalized rect."""
    height, width = image.shape[:2]
    x0, y0, x1, y1 = rect.to_pixels(width, height)
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Crop rect resolves to an empty region: {rect}")
    return image[y0:y1, x0:x1].copy()


def _content_mask(image: np.ndarray) -> np.ndarray:
    """Binary mask of non-background pixels, via Otsu threshold on grayscale."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def estimate_rotation_deg(image: np.ndarray) -> float:
    """Estimate the crop's skew, in degrees, from the minimum-area rect of its
    content mask. Returns 0.0 if no content contour is found (e.g. a blank crop).
    """
    mask = _content_mask(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 4:
        return 0.0
    (_, _), (rect_w, rect_h), angle = cv2.minAreaRect(largest)
    # minAreaRect's (width, height, angle) triple isn't unique -- the same
    # box can be described with its axes swapped and the angle offset by
    # 90 degrees. Canonicalize to "the smallest rotation that snaps the
    # rect to axis-aligned", in (-45, 45], independent of which axis
    # OpenCV happened to call "width".
    if rect_w < rect_h:
        angle += 90
    if angle <= -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    return angle


def auto_rotate_align(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotate `image` to straighten its content (FR-3.2).

    Returns (corrected_image, angle_applied_deg). Exposed rotated-in
    corners are filled white rather than left black, consistent with
    packaging artwork's white production background.
    """
    angle = estimate_rotation_deg(image)
    if abs(angle) < 1e-3:
        return image.copy(), 0.0
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    corrected = cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255)
    )
    return corrected, angle
