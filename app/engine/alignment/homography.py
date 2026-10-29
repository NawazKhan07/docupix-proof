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
    matches: FeatureMatches, config: AlignmentConfig, whole_sheet: bool = False, extent: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """The transform matched keypoints say takes one file onto the other, or None if none can be fitted.

    Returns (3x3 transform, boolean inlier mask over `matches`).

    A part of the sheet (a panel, `whole_sheet` False) takes the perspective RANSAC fits: on a scan, paper not quite
    flat on the glass bends each panel a little its own way, and a panel's fit follows it (with checks of its own).
    The whole sheet (`whole_sheet`) is fitted as follows.

    The perspective RANSAC fits is kept if a flat print could take it -- print, rendered or scanned or photographed, is
    turned and scaled the same all over, stretched perhaps (a carton of other proportions, a scanner's two
    resolutions), seen in perspective or sheared only a little. A perspective fitted to a re-laid proof bends the sheet
    into a compromise between panels moved each their own way (the panels are aligned each on its own after); fitted
    to a few matches it can fold the page. Such a fit is not kept: the matches it agreed on are fitted again as an
    affine, or a turn-and-scale, instead.
    """
    if len(matches) < config.min_match_count:
        return None
    if not whole_sheet:
        transform, mask = cv2.findHomography(matches.src_points, matches.dst_points, cv2.RANSAC, config.ransac_reproj_threshold_px)
        return None if transform is None else (transform, mask.ravel().astype(bool))
    src = np.asarray(matches.src_points, np.float32).reshape(-1, 2)
    dst = np.asarray(matches.dst_points, np.float32).reshape(-1, 2)
    threshold = config.ransac_reproj_threshold_px
    transform, mask = cv2.findHomography(src, dst, cv2.RANSAC, threshold)
    if transform is None:
        return None
    inliers = mask.ravel().astype(bool)
    # judged over all it will be applied to (`extent`: the corners of what is compared), not only where it matched: a
    # fit can be mild across the matches in the middle and fold the sheet's far edges
    judged = extent if extent is not None else src[inliers]
    spread, stretch, shear = _distortion(transform, judged)
    if spread <= _MAX_PERSPECTIVE and shear <= _MAX_SHEAR:
        return transform, inliers
    logger.info(
        "implausible fit declined",
        extra={"model": "perspective", "inliers": int(inliers.sum()), "spread": round(spread, 3), "stretch": round(stretch, 3), "shear": round(shear, 3)},
    )
    # The matches it agreed on are kept -- refitted as no more than a flat print can bend. (A fresh search over all the
    # matches could settle on another consensus altogether: artwork that repeats -- a logo printed thrice -- matches
    # each copy against the others.)
    chosen = np.flatnonzero(inliers)
    if len(chosen) < config.min_match_count:
        return None
    for kind in ("affine", "similarity"):
        fit = cv2.estimateAffine2D if kind == "affine" else cv2.estimateAffinePartial2D
        try:
            fitted, sub_mask = fit(src[chosen], dst[chosen], method=cv2.RANSAC, ransacReprojThreshold=threshold, maxIters=4000, confidence=0.999)
        except cv2.error:  # (degenerate points: all on a line, all alike)
            continue
        if fitted is None or sub_mask is None:
            continue
        refit = np.vstack([fitted, [0.0, 0.0, 1.0]])
        kept = np.zeros(len(src), bool)
        kept[chosen[sub_mask.ravel().astype(bool)]] = True
        if kept.sum() >= 4 and _plausible(refit, extent if extent is not None else src[kept]):
            logger.info("transform model chosen", extra={"model": kind, "inliers": int(kept.sum()), "matches": len(src)})
            return refit, kept
    return None


_MAX_PERSPECTIVE = 1.15  # flat print's scale differs across what is compared by at most this ratio...
_MAX_SHEAR = 0.08  # ...and is sheared by at most this (cosine between two square directions). Measured over the whole
# compared area of every sample pair, at full resolution: fits that are right reach perspective 1.063 and shear 0.042
# (a scanner's, a proof's); fits that fold the sheet into a compromise between panels moved each their own way reach
# 1.33 / 0.144 and 1.57 / 0.22 (a re-laid proof), 1.12 / 0.097 (two cartons of other proportions); a scanned label
# fitted to a few matches, 0.129. A stretch is allowed (a carton of other proportions, 15% and more).


def _distortion(transform: np.ndarray, points: np.ndarray) -> tuple[float, float, float]:
    from app.engine.alignment.quality_gate import distortion  # (quality_gate imports this module)

    return distortion(transform, points)


def _plausible(transform: np.ndarray, points: np.ndarray) -> bool:
    spread, _stretch, shear = _distortion(transform, points)
    return spread <= _MAX_PERSPECTIVE and shear <= _MAX_SHEAR

