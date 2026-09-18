from __future__ import annotations

import shutil

import cv2
import numpy as np
import pytest

from app.config import AppConfig
from app.engine.alignment.homography import warp_validity_mask
from app.engine.worker import run_comparison
from app.models.diff import DiffType
from app.models.isolation import IsolationGeometry, NormalizedPolygon
from app.models.job import ComparisonMode

CONFIG = AppConfig()


def _artwork(width: int = 600, height: int = 400) -> np.ndarray:
    """Textured enough for alignment, with a distinct colour block to change."""
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (40, 30), (width - 40, height - 30), (0, 0, 0), 3)
    cv2.putText(image, "PARACETAMOL 500mg", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
    cv2.putText(image, "Tablets BP x16", (70, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)
    cv2.rectangle(image, (70, 200), (240, 290), (200, 30, 40), -1)
    cv2.circle(image, (430, 240), 50, (30, 90, 200), -1)
    for index in range(12):
        x = 300 + index * 18
        cv2.line(image, (x, 300), (x, 350), (0, 0, 0), 3)
    return image


def _shifted(image: np.ndarray, angle: float = 1.5, tx: float = 10, ty: float = 6) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    return cv2.warpAffine(image, matrix, (width, height), borderValue=(255, 255, 255))


def _recolour_block(image: np.ndarray) -> np.ndarray:
    changed = image.copy()
    cv2.rectangle(changed, (70, 200), (240, 290), (40, 160, 60), -1)  # red -> green
    return changed


def _covers(candidates, x: float, y: float) -> bool:
    return any(
        candidate.bbox.x0 <= x <= candidate.bbox.x1 and candidate.bbox.y0 <= y <= candidate.bbox.y1
        for candidate in candidates
    )


def test_end_to_end_finds_a_colour_change_in_the_originals_coordinate_space():
    original = _artwork()
    suspect = _shifted(_recolour_block(original))

    outcome = run_comparison(original, suspect, ComparisonMode.AUTOMATIC, CONFIG)

    assert outcome.candidates
    colour = [c for c in outcome.candidates if c.type is DiffType.COLOUR]
    assert colour
    # The block sits at (70..240, 200..290) in the ORIGINAL; the suspect was
    # rotated and shifted, so finding it there proves the finding came back
    # through the alignment transform into the shared space.
    assert _covers(colour, 155, 245)


def test_identical_pair_reports_no_differences():
    original = _artwork()
    outcome = run_comparison(original, original.copy(), ComparisonMode.AUTOMATIC, CONFIG)
    assert outcome.candidates == []


def test_rotated_copy_does_not_produce_false_differences_at_the_edges():
    """The warped suspect covers less than the full frame; the uncovered band
    must be excluded rather than compared against the original's content."""
    original = _artwork()
    rotated = _shifted(original, angle=4.0, tx=0, ty=0)

    outcome = run_comparison(original, rotated, ComparisonMode.AUTOMATIC, CONFIG)

    page_area = original.shape[0] * original.shape[1]
    flagged_area = sum(candidate.bbox.area for candidate in outcome.candidates)
    assert flagged_area < 0.05 * page_area


def test_warp_validity_mask_excludes_uncovered_corners():
    transform = cv2.getRotationMatrix2D((300.0, 200.0), 10.0, 1.0)
    transform = np.vstack([transform, [0, 0, 1]])

    mask = warp_validity_mask((400, 600), transform, (400, 600))

    assert mask[200, 300]  # centre is covered
    assert not mask[0, 0]  # a corner rotated out of frame is not


def test_scan_vs_digital_mode_excludes_colour_differences():
    """FR-6.3: colour is not compared at all when a scan is involved."""
    original = _artwork()
    suspect = _recolour_block(original)

    automatic = run_comparison(original, suspect, ComparisonMode.AUTOMATIC, CONFIG)
    scan = run_comparison(original, suspect, ComparisonMode.SCAN_VS_DIGITAL, CONFIG)

    assert any(c.type is DiffType.COLOUR for c in automatic.candidates)
    assert not any(c.type is DiffType.COLOUR for c in scan.candidates)


def test_scan_mode_still_sees_a_structural_change():
    original = _artwork()
    suspect = original.copy()
    cv2.circle(suspect, (430, 240), 50, (255, 255, 255), -1)  # remove the roundel

    outcome = run_comparison(original, suspect, ComparisonMode.SCAN_VS_DIGITAL, CONFIG)

    assert any(c.type is DiffType.GRAPHICS for c in outcome.candidates)


def test_barcode_mode_compares_only_symbols():
    """FR-10.1: no other artwork content is compared in this mode."""
    original = _artwork()
    suspect = _recolour_block(original)  # a blatant colour change, and no symbols

    outcome = run_comparison(original, suspect, ComparisonMode.BARCODE, CONFIG)

    assert all(c.type is DiffType.BARCODE for c in outcome.candidates)
    assert outcome.warnings  # told the reviewer nothing was decodable


def test_braille_mode_compares_only_dots():
    original = _artwork()
    suspect = _recolour_block(original)

    outcome = run_comparison(original, suspect, ComparisonMode.BRAILLE, CONFIG)

    assert all(c.type is DiffType.BRAILLE for c in outcome.candidates)
    assert outcome.braille is not None


def test_content_outside_the_dieline_is_never_compared():
    """FR-2.1: production furniture must not reach any detector."""
    original = _artwork()
    suspect = original.copy()
    # A blatant change in the margin, outside the dieline boundary.
    cv2.rectangle(suspect, (5, 5), (35, 25), (255, 0, 255), -1)

    dieline = NormalizedPolygon(
        points=[(40 / 600, 30 / 400), (560 / 600, 30 / 400), (560 / 600, 370 / 400), (40 / 600, 370 / 400)]
    )
    geometry = IsolationGeometry(boundary=dieline)

    unrestricted = run_comparison(original, suspect, ComparisonMode.AUTOMATIC, CONFIG)
    restricted = run_comparison(
        original, suspect, ComparisonMode.AUTOMATIC, CONFIG, geometry, geometry
    )

    assert unrestricted.candidates  # visible without isolation
    assert restricted.candidates == []  # invisible once the dieline is applied


@pytest.mark.skipif(
    shutil.which("tesseract") is not None, reason="Tesseract installed; no warning expected"
)
def test_missing_ocr_is_reported_as_a_warning_without_stopping_other_detectors():
    original = _artwork()
    suspect = _recolour_block(original)

    outcome = run_comparison(original, suspect, ComparisonMode.AUTOMATIC, CONFIG)

    assert any("Tesseract" in warning for warning in outcome.warnings)
    assert any(c.type is DiffType.COLOUR for c in outcome.candidates)  # others still ran


def test_alignment_result_is_carried_through_to_the_outcome():
    original = _artwork()
    outcome = run_comparison(original, _shifted(original), ComparisonMode.AUTOMATIC, CONFIG)
    assert outcome.alignment.residual_error_px >= 0
    assert outcome.alignment.low_confidence is False
