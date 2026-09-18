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
