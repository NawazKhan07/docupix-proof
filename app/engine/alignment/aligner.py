"""Alignment stage entry point (Architecture §3.3.3).

Picks between the two registration paths the SRS calls for and always
returns a measured, logged residual:

* **dieline-to-dieline** (FR-4.2) when both files have a traced polygon
  boundary -- correspond the two loops, fit a similarity, then refine with
  keypoint matching inside the artwork region for sub-pixel correction.
* **feature matching** (FR-2.3) when only one side (or neither) has a
  dieline -- ORB/AKAZE keypoints plus a RANSAC homography, with each side
  masked to whatever artwork boundary it does have.

The returned transform maps suspect image pixels -> original image pixels;
that original space is the single shared aligned space (FR-7.4).
"""

from __future__ import annotations

import logging

import numpy as np

from app.config import AlignmentConfig
from app.engine.alignment import homography, quality_gate
from app.engine.alignment.feature_matcher import match_features
from app.engine.isolation.boundary_mask import resolve_keep_mask
from app.logging_config import StageTimer
from app.models.isolation import IsolationGeometry, NormalizedPolygon
from app.models.result import AlignmentResult

logger = logging.getLogger(__name__)


class AlignmentError(Exception):
    """The two files could not be registered at all (distinct from a poor fit)."""


def align(
    original_image: np.ndarray,
    suspect_image: np.ndarray,
    config: AlignmentConfig,
    original_geometry: IsolationGeometry | None = None,
    suspect_geometry: IsolationGeometry | None = None,
) -> AlignmentResult:
    with StageTimer(logger, "alignment"):
        original_mask = _keep_mask(original_image, original_geometry)
        suspect_mask = _keep_mask(suspect_image, suspect_geometry)

        original_polygon = _polygon_px(original_image, original_geometry)
        suspect_polygon = _polygon_px(suspect_image, suspect_geometry)

        if original_polygon is not None and suspect_polygon is not None:
            result = _align_via_dielines(
                original_image,
                suspect_image,
                original_polygon,
                suspect_polygon,
                original_mask,
                suspect_mask,
                config,
            )
        else:
            result = _align_via_features(
                original_image, suspect_image, original_mask, suspect_mask, config
            )

    # Non-negotiable (Build Prompt §3): the residual is logged on every job.
    logger.info(
        "alignment complete",
        extra={
            "residual_error_px": round(result.residual_error_px, 4),
            "residual_basis": result.residual_basis,
            "method": result.method,
            "match_count": result.match_count,
            "inlier_count": result.inlier_count,
            "inlier_ratio": round(result.inlier_ratio, 4),
            "threshold_px": result.threshold_px,
            "low_confidence": result.low_confidence,
        },
    )
    return result


def _keep_mask(image: np.ndarray, geometry: IsolationGeometry | None) -> np.ndarray | None:
    if geometry is None:
        return None
    return resolve_keep_mask(image.shape[:2], geometry)


def _polygon_px(image: np.ndarray, geometry: IsolationGeometry | None) -> np.ndarray | None:
    """The geometry's boundary as pixel coordinates, if it is a traced polygon."""
    if geometry is None or not isinstance(geometry.boundary, NormalizedPolygon):
        return None
    height, width = image.shape[:2]
    return np.array(
        [(x * width, y * height) for x, y in geometry.boundary.points], dtype=np.float64
    )


def _align_via_dielines(
    original_image: np.ndarray,
    suspect_image: np.ndarray,
    original_polygon: np.ndarray,
    suspect_polygon: np.ndarray,
    original_mask: np.ndarray | None,
    suspect_mask: np.ndarray | None,
    config: AlignmentConfig,
) -> AlignmentResult:
    fit = homography.polygon_correspondence(
        suspect_polygon, original_polygon, config.polygon_resample_points
    )
    polygon_residual = quality_gate.mean_residual_px(
        fit.transform, fit.src_points, fit.dst_points
    )

    refined = _refine_with_features(
        original_image, suspect_image, original_mask, suspect_mask, fit.transform, config
    )
    if refined is not None:
        return refined

    return _build_result(
        transform=fit.transform,
        residual_px=polygon_residual,
        residual_basis="polygon",
        method="dieline_polygon",
        match_count=len(fit.src_points),
        inlier_count=len(fit.src_points),
        config=config,
    )


def _refine_with_features(
    original_image: np.ndarray,
    suspect_image: np.ndarray,
    original_mask: np.ndarray | None,
    suspect_mask: np.ndarray | None,
    polygon_transform: np.ndarray,
    config: AlignmentConfig,
) -> AlignmentResult | None:
    """Sub-pixel refinement of a polygon fit using keypoints inside the artwork.

    Only accepted when it actually beats the polygon transform on the *same*
    keypoint correspondences -- otherwise matching noise could quietly make
    registration worse than the boundary fit it started from.
    """
    matches = match_features(suspect_image, original_image, config, suspect_mask, original_mask)
    estimate = homography.estimate_homography(matches, config)
    if estimate is None:
        return None

    transform, inliers = estimate
    if quality_gate.rejection_reason(transform, int(inliers.sum()), config) is not None:
        return None  # decline the refinement; the polygon fit stands

    src, dst = matches.src_points[inliers], matches.dst_points[inliers]
    feature_residual = quality_gate.mean_residual_px(transform, src, dst)
    polygon_residual_on_keypoints = quality_gate.mean_residual_px(polygon_transform, src, dst)
    if polygon_residual_on_keypoints <= feature_residual:
        return None

    return _build_result(
        transform=transform,
        residual_px=feature_residual,
        residual_basis="keypoints",
        method="dieline_polygon+features",
        match_count=len(matches),
        inlier_count=int(inliers.sum()),
        config=config,
    )


def _align_via_features(
    original_image: np.ndarray,
    suspect_image: np.ndarray,
    original_mask: np.ndarray | None,
    suspect_mask: np.ndarray | None,
    config: AlignmentConfig,
) -> AlignmentResult:
    matches = match_features(suspect_image, original_image, config, suspect_mask, original_mask)
    estimate = homography.estimate_homography(matches, config)
    if estimate is None:
        raise AlignmentError(
            "Could not register these two files: too few matching features were found. "
            "Try cropping to the artwork area or tracing the dieline on both files."
        )

    transform, inliers = estimate
    reason = quality_gate.rejection_reason(transform, int(inliers.sum()), config)
    if reason is not None:
        raise AlignmentError(
            f"Could not register these two files: {reason}. Check that both files are the "
            f"same artwork, or crop/trace the dieline on both to narrow the comparison."
        )

    src, dst = matches.src_points[inliers], matches.dst_points[inliers]
    return _build_result(
        transform=transform,
        residual_px=quality_gate.mean_residual_px(transform, src, dst),
        residual_basis="keypoints",
        method="features",
        match_count=len(matches),
        inlier_count=int(inliers.sum()),
        config=config,
    )


def _build_result(
    transform: np.ndarray,
    residual_px: float,
    residual_basis: str,
    method: str,
    match_count: int,
    inlier_count: int,
    config: AlignmentConfig,
) -> AlignmentResult:
    return AlignmentResult(
        transform=transform,
        residual_error_px=residual_px,
        residual_basis=residual_basis,
        method=method,
        match_count=match_count,
        inlier_count=inlier_count,
        # Reported and logged for QA diagnostics only -- measured inlier
        # ratios vary hugely with how repetitive the artwork is (barcodes and
        # repeated text lines generate wrong-instance matches), so they don't
        # discriminate a good registration from a bad one.
        inlier_ratio=inlier_count / match_count if match_count else 0.0,
        low_confidence=quality_gate.is_low_confidence(residual_px, config),
        threshold_px=config.residual_threshold_px,
    )
