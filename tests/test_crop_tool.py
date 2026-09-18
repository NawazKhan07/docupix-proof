from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.engine.isolation.crop_tool import (
    auto_rotate_align,
    crop_to_rect,
    estimate_rotation_deg,
)
from app.models.isolation import NormalizedRect


def _blank(width: int = 200, height: int = 150) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


def _rect_image(width: int = 300, height: int = 300) -> np.ndarray:
    image = _blank(width, height)
    cv2.rectangle(image, (60, 80), (240, 220), (0, 0, 0), -1)
    return image


def test_crop_to_rect_extracts_correct_region():
    image = _blank()
    image[40:110, 60:140] = (10, 20, 30)
    rect = NormalizedRect(x0=60 / 200, y0=40 / 150, x1=140 / 200, y1=110 / 150)
    cropped = crop_to_rect(image, rect)
    assert cropped.shape == (70, 80, 3)
    assert np.all(cropped == (10, 20, 30))


def test_crop_to_rect_rejects_empty_region():
    image = _blank()
    rect = NormalizedRect(x0=0.5, y0=0.5, x1=0.5, y1=0.5)
    with pytest.raises(ValueError):
        crop_to_rect(image, rect)


def test_estimate_rotation_is_near_zero_for_axis_aligned_content():
    angle = estimate_rotation_deg(_rect_image())
    assert abs(angle) < 1.0


@pytest.mark.parametrize("skew_deg", [8, -8, 20, -25])
def test_estimate_rotation_detects_a_known_skew(skew_deg: float):
    image = _rect_image()
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), skew_deg, 1.0)
    rotated = cv2.warpAffine(image, matrix, (width, height), borderValue=(255, 255, 255))
    angle = estimate_rotation_deg(rotated)
    assert abs(angle - (-skew_deg)) < 1.5


def test_auto_rotate_align_straightens_a_skewed_crop():
    image = _rect_image()
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 15, 1.0)
    rotated = cv2.warpAffine(image, matrix, (width, height), borderValue=(255, 255, 255))

    corrected, angle_applied = auto_rotate_align(rotated)
    assert abs(angle_applied) > 5  # a real correction was applied
    residual = estimate_rotation_deg(corrected)
    assert abs(residual) < 1.5  # corrected content is now close to axis-aligned


def test_auto_rotate_align_is_a_noop_when_already_straight():
    image = _rect_image()
    corrected, angle_applied = auto_rotate_align(image)
    assert angle_applied == 0.0
    assert np.array_equal(corrected, image)
