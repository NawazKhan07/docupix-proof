from __future__ import annotations

import logging

import cv2
import numpy as np
import pytest

from app.config import AlignmentConfig
from app.engine.alignment import homography, quality_gate
from app.engine.alignment.aligner import AlignmentError, align
from app.engine.alignment.feature_matcher import match_features
from app.engine.alignment.homography import (
    apply_transform,
    polygon_correspondence,
    resample_polygon,
    similarity_transform,
)
from app.models.isolation import IsolationGeometry, NormalizedPolygon, NormalizedRect

CONFIG = AlignmentConfig()


def _artwork_image(width: int = 600, height: int = 400) -> np.ndarray:
    """A synthetic carton front with enough corner detail for ORB to work on."""
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (40, 30), (width - 40, height - 30), (0, 0, 0), 3)  # dieline
    cv2.putText(image, "PARACETAMOL 500mg", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
    cv2.putText(image, "Tablets BP  x16", (70, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)
    cv2.rectangle(image, (70, 200), (240, 290), (200, 30, 40), -1)
    cv2.circle(image, (430, 240), 50, (30, 90, 200), -1)
    for i in range(12):  # barcode-ish stripes
        x = 300 + i * 18
        cv2.line(image, (x, 300), (x, 350), (0, 0, 0), 3)
    return image


def _dieline_polygon(width: int = 600, height: int = 400) -> list[tuple[float, float]]:
    """The `_artwork_image` dieline rectangle, in normalized coordinates."""
    return [
        (40 / width, 30 / height),
        ((width - 40) / width, 30 / height),
        ((width - 40) / width, (height - 30) / height),
        (40 / width, (height - 30) / height),
    ]


def _warp(image: np.ndarray, angle: float, tx: float, ty: float) -> tuple[np.ndarray, np.ndarray]:
    """Warp an image by a known rotation+translation; returns (warped, 3x3 matrix)."""
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    warped = cv2.warpAffine(image, matrix, (width, height), borderValue=(255, 255, 255))
    return warped, np.vstack([matrix, [0, 0, 1]])


def _warp_stretched(
    image: np.ndarray, angle: float, tx: float, ty: float, scale_x: float, scale_y: float, canvas_scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Warp an image by a known rotation+translation+INDEPENDENT x/y scale, onto a canvas of a different overall
    size (`canvas_scale` of the original's): the lifted subject is not always printed at the same size or
    proportions as its counterpart. Returns (warped, 3x3 matrix)."""
    height, width = image.shape[:2]
    stretch = np.diag([scale_x, scale_y, 1.0])
    rotation = np.vstack([cv2.getRotationMatrix2D((width * scale_x / 2, height * scale_y / 2), angle, 1.0), [0, 0, 1]])
    matrix = rotation @ stretch
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    canvas = (round(width * scale_x * canvas_scale), round(height * scale_y * canvas_scale))
    warped = cv2.warpAffine(image, matrix[:2], canvas, borderValue=(255, 255, 255))
    return warped, matrix


def _polygon_geometry(points) -> IsolationGeometry:
    return IsolationGeometry(boundary=NormalizedPolygon(points=list(points)))


# ---------------------------------------------------------------------------
# Feature matching and homography
# ---------------------------------------------------------------------------


def test_feature_matching_finds_correspondences_between_shifted_copies():
    original = _artwork_image()
    suspect, _ = _warp(original, angle=0, tx=12, ty=-7)
    matches = match_features(suspect, original, CONFIG)
    assert len(matches) >= CONFIG.min_match_count


def test_homography_recovers_a_known_transform():
    original = _artwork_image()
    suspect, forward = _warp(original, angle=3.0, tx=15, ty=9)

    matches = match_features(suspect, original, CONFIG)
    estimate = homography.estimate_homography(matches, CONFIG)
    assert estimate is not None
    transform, inliers = estimate

    # Recovered transform should undo the warp: suspect pixels map back to
    # where they came from in the original.
    probe = np.array([[100.0, 100.0], [500.0, 120.0], [300.0, 350.0]])
    expected = probe
    moved = apply_transform(forward, probe)
    recovered = apply_transform(transform, moved)
    assert np.allclose(recovered, expected, atol=3.0)
    assert inliers.sum() >= CONFIG.min_match_count


def test_estimate_homography_returns_none_with_too_few_matches():
    blank = np.full((200, 200, 3), 255, dtype=np.uint8)
    matches = match_features(blank, blank.copy(), CONFIG)
    assert homography.estimate_homography(matches, CONFIG) is None


def test_align_registers_a_stretched_rescaled_file_on_a_differently_sized_canvas():
    """One lifted artwork is not always the same size or proportions as its counterpart -- printed a little
    smaller, or unevenly stretched, on a sheet of a different size altogether. A full homography (not a rigid
    rotation+translation) must still register the two well, whichever way each file was resized."""
    original = _artwork_image()
    suspect, forward = _warp_stretched(
        original, angle=2.0, tx=20, ty=-15, scale_x=1.18, scale_y=0.88, canvas_scale=0.9
    )

    result = align(original, suspect, CONFIG)

    assert not result.low_confidence
    probe = np.array([[100.0, 100.0], [500.0, 120.0], [300.0, 350.0]])
    moved = apply_transform(forward, probe)
    recovered = apply_transform(result.transform, moved)
    assert np.allclose(recovered, probe, atol=4.0)


# ---------------------------------------------------------------------------
# Polygon correspondence
# ---------------------------------------------------------------------------


def test_resample_polygon_returns_evenly_spaced_points():
    square = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    resampled = resample_polygon(square, 64)
    assert resampled.shape == (64, 2)
    spacings = np.linalg.norm(np.diff(np.vstack([resampled, resampled[:1]]), axis=0), axis=1)
    assert np.allclose(spacings, spacings[0], atol=1e-6)  # perimeter 400 / 64 each


def test_similarity_transform_recovers_rotation_scale_and_translation():
    src = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]])
    expected = np.array([[0.0, 0.0], [0.0, 20.0], [-10.0, 20.0], [-10.0, 0.0]])  # 90deg, 2x
    transform = similarity_transform(src, expected)
    assert np.allclose(apply_transform(transform, src), expected, atol=1e-6)


def test_polygon_correspondence_handles_rotated_start_and_reversed_winding():
    """The two files' traces need not start at the same vertex or run the same way."""
    # Deliberately asymmetric: a square would fit equally well at four
    # orientations (see the ambiguity test below) and prove nothing here.
    l_shape = np.array(
        [[0.0, 0.0], [120.0, 0.0], [120.0, 60.0], [60.0, 60.0], [60.0, 140.0], [0.0, 140.0]]
    )
    scrambled = np.roll(l_shape[::-1], 2, axis=0)  # reversed winding, different start

    fit = polygon_correspondence(scrambled, l_shape, count=64)

    residual = quality_gate.mean_residual_px(fit.transform, fit.src_points, fit.dst_points)
    assert residual < 0.5
    assert np.allclose(apply_transform(fit.transform, scrambled), scrambled, atol=1.0)


def test_symmetric_dieline_fit_is_orientation_ambiguous():
    """A square dieline fits its own outline equally well at 0/90/180/270 degrees.

    The boundary alone cannot say which, so the fit may legitimately come back
    rotated. This is precisely why the dieline path refines with artwork
    features afterwards -- content resolves what the outline can't.
    """
    square = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])

    fit = polygon_correspondence(np.roll(square[::-1], 2, axis=0), square, count=64)

    residual = quality_gate.mean_residual_px(fit.transform, fit.src_points, fit.dst_points)
    assert residual < 0.5  # lands exactly on the outline...
    mapped = apply_transform(fit.transform, square)
    # ...though possibly onto a rotated labelling of the same four corners.
    assert np.allclose(
        mapped[np.lexsort(mapped.T)], square[np.lexsort(square.T)], atol=1.0
    )


def test_polygon_correspondence_recovers_a_shifted_scaled_loop():
    square = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    target = square * 2.0 + np.array([30.0, 50.0])

    fit = polygon_correspondence(square, target, count=64)

    assert np.allclose(apply_transform(fit.transform, square), target, atol=0.5)


# ---------------------------------------------------------------------------
# End-to-end alignment paths
# ---------------------------------------------------------------------------


def test_dieline_path_registers_a_known_warp():
    original = _artwork_image()
    suspect, forward = _warp(original, angle=2.0, tx=10, ty=6)
    suspect_polygon = apply_transform(forward, np.array(_dieline_polygon()) * [600, 400])

    result = align(
        original,
        suspect,
        CONFIG,
        original_geometry=_polygon_geometry(_dieline_polygon()),
        suspect_geometry=_polygon_geometry([(x / 600, y / 400) for x, y in suspect_polygon]),
    )

    assert result.method.startswith("dieline_polygon")
    assert result.low_confidence is False
    assert result.residual_error_px < CONFIG.residual_threshold_px

    probe = np.array([[150.0, 120.0], [420.0, 260.0]])
    recovered = apply_transform(result.transform, apply_transform(forward, probe))
    assert np.allclose(recovered, probe, atol=3.0)


def test_feature_fallback_path_when_only_one_file_has_a_dieline():
    """FR-2.3: align the non-dielined file to the dielined one via feature matching."""
    original = _artwork_image()
    suspect, forward = _warp(original, angle=1.5, tx=8, ty=-5)

    result = align(
        original,
        suspect,
        CONFIG,
        original_geometry=_polygon_geometry(_dieline_polygon()),
        suspect_geometry=None,
    )

    assert result.method == "features"
    assert result.low_confidence is False
    probe = np.array([[200.0, 150.0], [400.0, 300.0]])
    recovered = apply_transform(result.transform, apply_transform(forward, probe))
    assert np.allclose(recovered, probe, atol=3.0)


def test_alignment_works_with_no_geometry_at_all():
    original = _artwork_image()
    suspect, _ = _warp(original, angle=0, tx=6, ty=4)
    result = align(original, suspect, CONFIG)
    assert result.method == "features"
    assert result.low_confidence is False


def test_alignment_error_when_nothing_can_be_matched():
    blank = np.full((300, 300, 3), 255, dtype=np.uint8)
    with pytest.raises(AlignmentError):
        align(blank, blank.copy(), CONFIG)


# ---------------------------------------------------------------------------
# Quality gate (NFR-3)
# ---------------------------------------------------------------------------


def test_quality_gate_flags_a_deliberately_misaligned_pair():
    """Regression test required by Build Prompt §5.

    A scan traced with a badly wrong dieline: the suspect has no content to
    refine against, so the polygon fit stands on its own, and a rectangle
    cannot be fitted onto a trapezoid by a similarity transform.
    """
    original = _artwork_image()
    blank_suspect = np.full((400, 600, 3), 255, dtype=np.uint8)
    trapezoid = [(0.3, 0.1), (0.7, 0.1), (0.95, 0.9), (0.05, 0.9)]

    result = align(
        original,
        blank_suspect,
        CONFIG,
        original_geometry=_polygon_geometry(_dieline_polygon()),
        suspect_geometry=_polygon_geometry(trapezoid),
    )

    assert result.residual_basis == "polygon"
    assert result.residual_error_px > CONFIG.residual_threshold_px
    assert result.low_confidence is True


def test_threshold_comes_from_config_not_a_constant():
    """The same pair flips to low-confidence purely by tightening the config."""
    original = _artwork_image()
    suspect, _ = _warp(original, angle=1.0, tx=5, ty=3)

    lenient = align(original, suspect, CONFIG)
    assert lenient.low_confidence is False

    strict = align(original, suspect, AlignmentConfig(residual_threshold_px=0.0001))
    assert strict.low_confidence is True
    assert strict.threshold_px == 0.0001


def test_gate_compares_residual_against_the_configured_threshold():
    assert quality_gate.is_low_confidence(2.5, CONFIG)
    assert not quality_gate.is_low_confidence(1.5, CONFIG)


def test_rejection_reason_accepts_a_sane_registration():
    sane = np.array([[1.0, 0.02, 15.0], [-0.02, 1.0, 9.0], [0.0, 0.0, 1.0]])
    assert quality_gate.rejection_reason(sane, inlier_count=400, config=CONFIG) is None


@pytest.mark.parametrize(
    "transform, inliers, expected",
    [
        (np.eye(3), 8, "consistent feature matches"),
        (np.diag([-1.0, 1.0, 1.0]), 400, "mirrors"),  # negative determinant
        (np.diag([0.001, 0.001, 1.0]), 400, "rescales"),  # collapsed
        (np.diag([50.0, 50.0, 1.0]), 400, "rescales"),  # blown up
    ],
)
def test_rejection_reason_catches_untrustworthy_fits(transform, inliers, expected):
    reason = quality_gate.rejection_reason(transform, inliers, CONFIG)
    assert reason is not None and expected in reason


def test_unregistrable_pair_fails_rather_than_reporting_a_confident_fit():
    """Two different products must not quietly produce a plausible-looking fit.

    Measured behaviour: a genuinely different layout yields zero RANSAC
    inliers, so it fails here rather than reaching the gate with a small
    (RANSAC-bounded) residual.
    """
    carton = _artwork_image()
    leaflet = np.full((400, 600, 3), 255, dtype=np.uint8)
    cv2.rectangle(leaflet, (200, 20), (400, 200), (0, 0, 0), 5)
    cv2.putText(leaflet, "INHALATION", (210, 300), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 0), 3)
    cv2.circle(leaflet, (300, 360), 30, (200, 30, 40), -1)

    with pytest.raises(AlignmentError):
        align(carton, leaflet, CONFIG)


def test_repetitive_artwork_still_registers_confidently():
    """Regression: barcode stripes and repeated text lines generate many
    wrong-instance matches; a good registration must not be flagged over it."""
    original = _artwork_image()
    suspect, _ = _warp(original, angle=1.5, tx=20, ty=12)

    result = align(original, suspect, CONFIG)

    assert result.low_confidence is False
    assert result.residual_error_px < CONFIG.residual_threshold_px


def test_residual_is_logged_on_every_alignment(caplog):
    """Build Prompt §3: never silently claim perfect alignment."""
    original = _artwork_image()
    suspect, _ = _warp(original, angle=0, tx=5, ty=5)

    with caplog.at_level(logging.INFO, logger="app.engine.alignment.aligner"):
        result = align(original, suspect, CONFIG)

    logged = [r for r in caplog.records if r.message == "alignment complete"]
    assert len(logged) == 1
    assert logged[0].residual_error_px == pytest.approx(result.residual_error_px, abs=1e-3)
    assert logged[0].low_confidence == result.low_confidence


def test_crop_rect_geometry_masks_without_taking_the_dieline_path():
    """A rect boundary isn't a traced dieline, so it masks but doesn't correspond."""
    original = _artwork_image()
    suspect, _ = _warp(original, angle=0, tx=4, ty=2)
    rect_geometry = IsolationGeometry(boundary=NormalizedRect(0.05, 0.05, 0.95, 0.95))

    result = align(original, suspect, CONFIG, rect_geometry, rect_geometry)

    assert result.method == "features"
