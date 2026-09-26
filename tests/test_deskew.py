"""Levelling a picture before it is read (`app.engine.preprocessing.deskew`)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.config import DetectionConfig
from app.engine.preprocessing import deskew
from app.models.diff import BoundingBox

CONFIG = DetectionConfig()


def _dense_page(angle: float = 0.0, rows: int = 40, width: int = 1200, height: int = 1500) -> np.ndarray:
    """Many tightly-packed, near-identical lines -- the case a residual tilt confuses most -- as a grey reading
    picture (255 = paper, dark = ink), turned by `angle` about its own centre."""
    page = np.full((height, width, 3), 255, np.uint8)
    spacing = (height - 160) // rows
    for row in range(rows):
        cv2.putText(
            page, f"BATCH 2109{row:04d} EXP 2027-06 MNB/05/109 Mfd by CIPLA", (60, 80 + row * spacing),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA,
        )
    grey = cv2.cvtColor(page, cv2.COLOR_RGB2GRAY)
    if angle:
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        grey = cv2.warpAffine(grey, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=255)
    return grey


# ---- measuring the angle -----------------------------------------------------------------------------------------


def test_a_level_picture_needs_no_correction():
    assert deskew.measured(_dense_page(0.0), CONFIG).is_identity


def test_a_blank_picture_is_left_alone():
    blank = np.full((900, 700), 255, np.uint8)
    assert deskew.measured(blank, CONFIG).is_identity


@pytest.mark.parametrize("angle", [0.6, 1.2, 2.5, -1.8, 4.0])
def test_a_tilted_pictures_correction_levels_it(angle):
    tilted = _dense_page(angle)
    level = deskew.measured(tilted, CONFIG)

    assert not level.is_identity
    assert level.angle_deg == pytest.approx(-angle, abs=0.15)
    corrected = level.picture(tilted)
    assert abs(deskew.estimate_skew_deg(corrected, CONFIG)) <= CONFIG.deskew_min_correction_deg


def test_a_barely_tilted_picture_is_not_worth_correcting():
    from dataclasses import replace

    generous = replace(CONFIG, deskew_min_correction_deg=1.0)
    assert deskew.measured(_dense_page(0.2), generous).is_identity


def test_the_search_never_returns_more_than_the_configured_limit():
    from dataclasses import replace

    narrow = replace(CONFIG, deskew_max_angle_deg=2.0)
    level = deskew.measured(_dense_page(8.0), narrow)
    assert abs(level.angle_deg) <= 2.0


def test_disabled_returns_identity_however_tilted_the_picture_is():
    from dataclasses import replace

    off = replace(CONFIG, deskew_enabled=False)
    assert deskew.measured(_dense_page(3.0), off).is_identity


# ---- turning a picture, and turning a box back ---------------------------------------------------------------


def test_an_identity_deskew_leaves_the_picture_and_a_box_untouched():
    picture = _dense_page(0.0)
    level = deskew.measured(picture, CONFIG)
    assert level.picture(picture) is picture
    box = BoundingBox(10, 20, 100, 40)
    assert level.unrotate_box(box) == box


def test_a_corrected_pictures_word_maps_back_near_where_it_really_is():
    """A single WORD's box (the size `_unrotated` actually deals in), read from the LEVELLED picture and turned
    back, lands close to where that same word is in the original, un-levelled one. Only a few pixels of slack are
    allowed: a wide SPAN turned back is not axis-aligned any more and grows a box round it (`width * sin(angle)`,
    tens of pixels for a whole sentence at a couple of degrees) -- but a single word is narrow enough that this
    stays small, which is exactly why `_unrotated` is applied per word rather than to a whole line's box."""
    angle = 2.0
    width = 260  # about one word wide
    original = np.full((120, width, 3), 255, np.uint8)
    cv2.putText(original, "CIPLA", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 2, cv2.LINE_AA)
    original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    matrix = cv2.getRotationMatrix2D((width / 2, 60), angle, 1.0)
    tilted = cv2.warpAffine(original, matrix, (width, 120), flags=cv2.INTER_LINEAR, borderValue=255)

    level = deskew.measured(tilted, CONFIG)
    corrected = level.picture(tilted)

    ys, xs = np.nonzero(original < 128)
    true_box = BoundingBox(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    ys, xs = np.nonzero(corrected < 128)
    box_in_corrected = BoundingBox(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    back = level.unrotate_box(box_in_corrected)

    assert back.x0 == pytest.approx(true_box.x0, abs=6) and back.x1 == pytest.approx(true_box.x1, abs=6)
    assert back.y0 == pytest.approx(true_box.y0, abs=6) and back.y1 == pytest.approx(true_box.y1, abs=6)
