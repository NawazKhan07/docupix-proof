"""Alignment quality gate (NFR-3, Architecture §3.3.3 / §9).

NFR-3 replaced an unverifiable "100% alignment" claim with a measurable one:
every alignment computes a numeric registration residual, that number is
logged for every job, and if it exceeds the configured acceptance threshold
the result is flagged low-confidence and surfaced to the user rather than
proceeding as though registration were perfect.
"""

from __future__ import annotations

import numpy as np

from app.config import AlignmentConfig
from app.engine.alignment.homography import apply_transform


def reprojection_errors(
    transform: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray
) -> np.ndarray:
    """Per-point distance, in pixels, between projected source and actual destination."""
    projected = apply_transform(transform, src_points)
    return np.linalg.norm(projected - np.asarray(dst_points, dtype=np.float64), axis=1)


def mean_residual_px(
    transform: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray
) -> float:
    if len(src_points) == 0:
        return float("inf")
    return float(np.mean(reprojection_errors(transform, src_points, dst_points)))


def rejection_reason(
    transform: np.ndarray, inlier_count: int, config: AlignmentConfig
) -> str | None:
    """Why this fit cannot be trusted at all, or None if it is usable.

    Distinct from the low-confidence gate below: these are fits with no
    geometric basis rather than poor-but-usable registrations, so the caller
    fails the job instead of reporting a reassuring residual. Measured on
    synthetic pairs, genuine registrations produced 124-768 inliers with a
    2x2 determinant within a few thousandths of 1.0, while two different
    products produced 8 inliers and a mirrored, near-collapsed transform
    (determinant -0.011) that the residual alone reported as 0.96 px.
    """
    if inlier_count < config.min_inlier_count:
        return (
            f"only {inlier_count} consistent feature matches were found "
            f"(at least {config.min_inlier_count} are needed to trust a fit)"
        )

    determinant = float(np.linalg.det(np.asarray(transform, dtype=np.float64)[:2, :2]))
    if determinant <= 0:
        return "the fitted transform mirrors the artwork"
    max_determinant = config.max_scale_factor**2
    if not 1 / max_determinant <= determinant <= max_determinant:
        return f"the fitted transform rescales the artwork implausibly (x{determinant**0.5:.1f})"
    return None


def distortion(transform: np.ndarray, points: np.ndarray) -> tuple[float, float, float]:
    """How a fit bends the artwork over the area its matched `points` (in the frame it maps from) span: (the ratio of
    its largest to its smallest local scale there, its largest stretch -- scale along one way over the other -- and its
    largest shear, as the cosine between where two square directions go). A flat print, rendered, scanned or
    photographed square on, is turned and scaled the same all over: (1, 1, 0), give or take a little."""
    points = np.asarray(points, np.float64).reshape(-1, 2)
    if len(points) < 3:
        return 1.0, 1.0, 0.0
    h = np.asarray(transform, np.float64)
    (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)
    scales, stretches, shears = [], [], []
    for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1), ((x0 + x1) / 2, (y0 + y1) / 2)):
        w = h[2, 0] * x + h[2, 1] * y + h[2, 2]
        u = h[0, 0] * x + h[0, 1] * y + h[0, 2]
        v = h[1, 0] * x + h[1, 1] * y + h[1, 2]
        local = np.array(
            [[h[0, 0] * w - u * h[2, 0], h[0, 1] * w - u * h[2, 1]], [h[1, 0] * w - v * h[2, 0], h[1, 1] * w - v * h[2, 1]]]
        ) / (w * w)
        singular = np.linalg.svd(local, compute_uv=False)
        if singular[1] <= 0:
            return float("inf"), float("inf"), 1.0
        scales.append(float(np.sqrt(singular[0] * singular[1])))
        stretches.append(float(singular[0] / singular[1]))
        a, b = local[:, 0], local[:, 1]
        shears.append(abs(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))))
    return max(scales) / min(scales), max(stretches), max(shears)


def is_low_confidence(residual_px: float, config: AlignmentConfig) -> bool:
    """Whether this registration should be surfaced as low-confidence.

    Known limitation to weigh when setting the threshold with QA: a
    keypoint-basis residual is averaged over RANSAC inliers, so it is
    bounded above by `ransac_reproj_threshold_px` and has a compressed
    dynamic range. A pair that cannot be registered at all doesn't reach
    this gate -- it fails earlier with no consistent matches -- but the
    number itself should not be read as a full measure of correctness.
    Polygon-basis residuals have no such ceiling.
    """
    return residual_px > config.residual_threshold_px
