from __future__ import annotations


import cv2
import numpy as np
import pytesseract

from dataclasses import replace

from app.config import AppConfig
from app.engine.alignment.homography import warp_validity_mask
from app.engine.worker import run_comparison
from app.models.diff import DiffType
from app.models.isolation import IsolationGeometry, NormalizedPolygon
from app.models.job import ComparisonMode

CONFIG = AppConfig()
GRAPHICS = replace(CONFIG, detection=replace(CONFIG.detection, scan_compare_graphics=True))


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


def test_scan_vs_digital_mode_ignores_a_colour_shift_but_not_a_different_ink():
    """FR-6.3: a scan's colours are not compared -- scanning shifts every one -- but an area printed in a different ink
    (red in the file, green in the print) is not a shift: it is the one colour finding a scan gets."""
    original = _artwork()
    shifted = original.copy()
    cv2.rectangle(shifted, (70, 200), (240, 290), (225, 60, 55), -1)  # the same red, as a scan renders it
    scan_shift = run_comparison(original, shifted, ComparisonMode.SCAN_VS_DIGITAL, CONFIG)
    assert not any(c.type is DiffType.COLOUR for c in scan_shift.candidates)

    swapped = _recolour_block(original)  # red -> green
    automatic = run_comparison(original, swapped, ComparisonMode.AUTOMATIC, CONFIG)
    scan = run_comparison(original, swapped, ComparisonMode.SCAN_VS_DIGITAL, CONFIG)
    assert any(c.type is DiffType.COLOUR for c in automatic.candidates)
    colour = [c for c in scan.candidates if c.type is DiffType.COLOUR]
    assert len(colour) == 1 and "red in the digital file, green in the scan" in colour[0].detail


def test_scan_mode_still_sees_a_structural_change():
    original = _artwork()
    suspect = original.copy()
    cv2.circle(suspect, (430, 240), 50, (255, 255, 255), -1)  # remove the roundel

    outcome = run_comparison(original, suspect, ComparisonMode.SCAN_VS_DIGITAL, GRAPHICS)

    assert any(c.type is DiffType.GRAPHICS for c in outcome.candidates)


def test_barcode_mode_compares_only_symbols():
    """FR-10.1: no other artwork content is compared in this mode."""
    original = _artwork()
    suspect = _recolour_block(original)  # a blatant colour change; the symbol is untouched

    outcome = run_comparison(original, suspect, ComparisonMode.BARCODE, CONFIG)

    assert outcome.candidates == []  # the colour change is not this mode's business
    # The artwork's twelve equal bars are, by the symbology's definition, a Pharmacode (2^12 - 1).
    assert [(c.symbol_type, c.original_value, c.passed) for c in outcome.symbol_comparisons] == [
        ("Pharmacode", "4095", True)
    ]
    assert outcome.mode is ComparisonMode.BARCODE


def test_barcode_mode_says_so_when_nothing_could_be_decoded():
    blank = np.full((400, 600, 3), 255, dtype=np.uint8)
    cv2.putText(blank, "PARACETAMOL", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)

    outcome = run_comparison(blank, blank.copy(), ComparisonMode.BARCODE, CONFIG)

    assert outcome.symbol_comparisons == []
    assert any("No barcode" in warning for warning in outcome.warnings)  # told the reviewer nothing was read


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


def test_missing_ocr_is_reported_as_a_warning_without_stopping_other_detectors(monkeypatch):
    monkeypatch.setattr(pytesseract.pytesseract, "tesseract_cmd", "no-such-tesseract-executable")  # installed or not
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


def test_small_text_edges_do_not_produce_spurious_findings(review_pair):
    """Regression: on a 1.2 degree pair, anti-aliased glyph edges read as
    ~dE 25 and produced tiny colour findings on small text, each rated
    CRITICAL. Exactly the two genuine changes may be reported -- and each once,
    with its colour and graphics evidence merged into a single finding."""
    original, suspect = review_pair

    outcome = run_comparison(original, suspect, ComparisonMode.AUTOMATIC, CONFIG)

    assert len(outcome.clusters) == 2  # the recoloured block and the removed roundel
    assert all(c.bbox_in_original.area > 5000 for c in outcome.clusters)


def test_the_outcome_carries_the_aligned_pair_the_viewer_draws_from():
    """FR-7.4: one aligned pair, at the original's size, for every view."""
    original = _artwork()
    outcome = run_comparison(original, _shifted(original), ComparisonMode.AUTOMATIC, CONFIG)

    assert outcome.aligned_original.shape == original.shape
    assert outcome.aligned_suspect.shape == original.shape


def test_the_same_pair_gives_the_same_alignment_and_findings_every_time():
    """A reviewer comparing the same two files again must get the same answer, whatever was compared in between."""
    original = _artwork()
    suspect = _shifted(_recolour_block(original))

    def run():
        outcome = run_comparison(original, suspect, ComparisonMode.SCAN_VS_DIGITAL, GRAPHICS)
        return outcome.alignment.transform, [(c.type, c.detail, c.bbox_in_original) for c in outcome.all_clusters]

    first = run()
    run_comparison(original, _shifted(original, angle=-2.0), ComparisonMode.AUTOMATIC, CONFIG)  # something else in between
    second = run()
    assert np.array_equal(first[0], second[0]) and first[1] == second[1]
