"""Transform estimation: dieline-polygon correspondence and RANSAC homography
(Architecture §3.3.3)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import AlignmentConfig
from app.engine.alignment.feature_matcher import FeatureMatches

logger = logging.getLogger(__name__)


@dataclass
class PolygonFit:
    """The best correspondence found between two traced dieline loops."""

    src_points: np.ndarray  # (N, 2) resampled source loop, reordered to match dst
    dst_points: np.ndarray  # (N, 2) resampled destination loop
    transform: np.ndarray  # 3x3 similarity mapping src -> dst


def apply_transform(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Project (N, 2) points through a 3x3 homogeneous transform."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.hstack([pts, np.ones((len(pts), 1))])
    projected = homogeneous @ np.asarray(transform, dtype=np.float64).T
    w = projected[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return projected[:, :2] / w


def polygon_perimeter(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    closed = np.vstack([pts, pts[:1]])
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())


def resample_polygon(points: np.ndarray, count: int, offset: float = 0.0) -> np.ndarray:
    """Resample a closed polygon to `count` points evenly spaced by arc length.

    Two files' dieline traces have no reason to share a vertex count (one may
    be vector paths, the other contour pixels), so they are resampled onto a
    common parameterization before anything tries to correspond them.
    `offset` slides the sampling start around the loop, in arc-length units.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        raise ValueError("A polygon needs at least 3 points to resample.")
    closed = np.vstack([pts, pts[:1]])
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    perimeter = cumulative[-1]
    if perimeter <= 0:
        raise ValueError("Polygon has zero perimeter.")

    targets = (np.linspace(0.0, perimeter, count, endpoint=False) + offset) % perimeter
    index = np.clip(np.searchsorted(cumulative, targets, side="right") - 1, 0, len(segment_lengths) - 1)
    t = (targets - cumulative[index]) / np.maximum(segment_lengths[index], 1e-12)
    return closed[index] + (closed[index + 1] - closed[index]) * t[:, None]


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity (rotation + uniform scale + translation), Umeyama.

    Used instead of `estimateAffinePartial2D` for the polygon search below
    because it is a plain closed-form fit: RANSAC could make a badly-ordered
    correspondence look good by quietly discarding most of the loop, which
    would corrupt the ranking of candidate orderings.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_centered, dst_centered = src - src_mean, dst - dst_mean

    covariance = dst_centered.T @ src_centered / len(src)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ correction @ vt

    src_variance = (src_centered**2).sum() / len(src)
    scale = float((singular_values * np.diag(correction)).sum() / src_variance) if src_variance > 0 else 1.0

    transform = np.eye(3)
    transform[:2, :2] = scale * rotation
    transform[:2, 2] = dst_mean - scale * rotation @ src_mean
    return transform


_SUBSAMPLE_STEPS = 21


def _fit_at_offset(
    src_polygon: np.ndarray, dst_points: np.ndarray, count: int, offset: float
) -> tuple[float, PolygonFit]:
    src = resample_polygon(src_polygon, count, offset=offset)
    transform = similarity_transform(src, dst_points)
    residual = float(np.mean(np.linalg.norm(apply_transform(transform, src) - dst_points, axis=1)))
    return residual, PolygonFit(src_points=src, dst_points=dst_points, transform=transform)


def polygon_correspondence(
    src_polygon: np.ndarray, dst_polygon: np.ndarray, count: int
) -> PolygonFit:
    """Correspond two dieline loops by searching start offset and winding.

    The two traces may start at different vertices and run in opposite
    directions, so the source loop is swept around its whole perimeter in
    both windings and the lowest-residual fit wins.
    """
    dst = resample_polygon(dst_polygon, count)
    source = np.asarray(src_polygon, dtype=np.float64).reshape(-1, 2)
    perimeter = polygon_perimeter(source)
    step = perimeter / count

    best_residual, best_fit = np.inf, None
    for winding in (source, source[::-1]):
        coarse_residual, _, coarse_offset = min(
            ((*_fit_at_offset(winding, dst, count, offset), offset)
             for offset in np.linspace(0.0, perimeter, count, endpoint=False)),
            key=lambda candidate: candidate[0],
        )
        # The true correspondence rarely lands exactly on a sample boundary,
        # and the leftover error scales with the dieline's perimeter -- on
        # full-size artwork that alone could exceed the quality-gate
        # threshold, so refine within one sample spacing of the best offset.
        residual, fit, _ = min(
            ((*_fit_at_offset(winding, dst, count, offset), offset)
             for offset in coarse_offset + np.linspace(-step, step, _SUBSAMPLE_STEPS)),
            key=lambda candidate: candidate[0],
        )
        if residual < best_residual:
            best_residual, best_fit = residual, fit

    assert best_fit is not None  # both windings always produce a candidate
    logger.info("fitted dieline polygons", extra={"polygon_residual_px": round(best_residual, 3)})
    return best_fit


def warp_to_reference(
    image: np.ndarray,
    transform: np.ndarray,
    output_shape: tuple[int, int],
    border_value: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Project an image into the reference (original) frame."""
    height, width = output_shape
    return cv2.warpPerspective(
        image, transform, (width, height), flags=cv2.INTER_LINEAR, borderValue=border_value
    )


def warp_validity_mask(
    source_shape: tuple[int, int], transform: np.ndarray, output_shape: tuple[int, int]
) -> np.ndarray:
    """Which reference pixels the warped source actually covers.

    Without this, everywhere the suspect has no data becomes border fill and
    then differs from the original's content -- a band of false differences
    around every edge, growing with the rotation between the two files.
    """
    height, width = output_shape
    covered = cv2.warpPerspective(
        np.full(source_shape, 255, dtype=np.uint8),
        transform,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    return covered > 0


def estimate_homography(
    matches: FeatureMatches, config: AlignmentConfig
) -> tuple[np.ndarray, np.ndarray] | None:
    """RANSAC homography from matched keypoints, or None if it can't be fitted.

    Returns (3x3 transform, boolean inlier mask over `matches`).
    """
    if len(matches) < config.min_match_count:
        return None
    transform, mask = cv2.findHomography(
        matches.src_points,
        matches.dst_points,
        cv2.RANSAC,
        config.ransac_reproj_threshold_px,
    )
    if transform is None:
        return None
    return transform, mask.ravel().astype(bool)
