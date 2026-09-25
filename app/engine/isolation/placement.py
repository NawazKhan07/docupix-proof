"""Laying one file's lifted outline over another, and snapping it into place.

A cutout lifted from the original can be placed over the suspect to trim the
suspect to the same outline when it has no clear border of its own. Where it lands
is a `Placement`: the point its centre sits on, a rotation and a scale (a
similarity, which is all that moving a cutout by hand can express).

Placing by hand is the fallback. The app already knows how to register two files
from the artwork's own features, without needing a border, so `fit_placement`
asks it, and the outline arrives already on the artwork with the reviewer only
adjusting it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import AlignmentConfig
from app.engine.alignment.aligner import AlignmentError, align
from app.engine.isolation.cutout import Cutout
from app.models.isolation import IsolationGeometry, NormalizedPolygon
from app.models.result import AlignmentResult

Point = tuple[float, float]


@dataclass(frozen=True)
class Placement:
    """Where a cutout sits on a target image (y down, degrees clockwise)."""

    center: Point  # where the cutout's centre lands, in target pixels
    rotation_deg: float = 0.0
    scale: float = 1.0

    def matrix(self, size: tuple[int, int]) -> np.ndarray:
        """3x3 matrix taking the cutout's own pixels to target pixels."""
        width, height = size
        angle = math.radians(self.rotation_deg)
        c, s = math.cos(angle) * self.scale, math.sin(angle) * self.scale
        half_w, half_h = width / 2.0, height / 2.0
        return np.array(
            [
                [c, -s, self.center[0] - (c * half_w - s * half_h)],
                [s, c, self.center[1] - (s * half_w + c * half_h)],
                [0.0, 0.0, 1.0],
            ]
        )

    def map_points(self, points: list[Point], size: tuple[int, int]) -> list[Point]:
        matrix = self.matrix(size)
        return [
            (float(matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]), float(matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]))
            for x, y in points
        ]


def placement_from_matrix(matrix: np.ndarray, size: tuple[int, int]) -> tuple[Placement, float]:
    """The similarity that best matches a 3x3 transform on a cutout, and how far it is out.

    Returns (placement, the largest distance in pixels between where the transform and
    the similarity put a point of the cutout).
    """
    width, height = size
    xs, ys = np.meshgrid(np.linspace(0, width, 5), np.linspace(0, height, 5))
    source = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    homogeneous = np.column_stack([source, np.ones(len(source))]) @ matrix.T
    target = (homogeneous[:, :2] / homogeneous[:, 2:3]).astype(np.float32)

    fitted, _ = cv2.estimateAffinePartial2D(source, target, method=cv2.LMEDS)
    if fitted is None:
        raise ValueError("the transform cannot be reduced to a placement")
    scale = math.hypot(fitted[0, 0], fitted[1, 0])
    rotation = math.degrees(math.atan2(fitted[1, 0], fitted[0, 0]))
    centre = fitted @ np.array([width / 2.0, height / 2.0, 1.0])
    placement = Placement((float(centre[0]), float(centre[1])), rotation, scale)

    reproduced = np.array(placement.map_points([tuple(p) for p in source], size))
    return placement, float(np.max(np.linalg.norm(reproduced - target, axis=1)))


@dataclass
class PlacementFit:
    placement: Placement
    alignment: AlignmentResult
    outline_error_px: float  # how far the similarity is from the registration, worst point of the cutout


def fit_placement(
    cutout: Cutout, target_rgb: np.ndarray, config: AlignmentConfig | None = None, expected_scale: float | None = None
) -> PlacementFit:
    """Find where `cutout` belongs on `target_rgb` by matching the artwork itself -- its features first, and, where those
    cannot be matched, its edges (near `expected_scale` if the two files' resolutions say what the scale should be).

    Raises `AlignmentError` if the two do not share enough artwork to be matched.
    """
    config = config or AlignmentConfig()
    width, height = cutout.size
    # The cutout's own outline only masks the search for features; it never anchors the fit.
    outline = IsolationGeometry(
        boundary=NormalizedPolygon(points=[(x / width, y / height) for x, y in cutout.polygon]),
        boundary_origin="lifted",
    )
    try:
        alignment = align(cutout.on_white(), target_rgb, config, original_geometry=outline)
        # `alignment.transform` takes target pixels to cutout pixels; placing needs the inverse.
        placement, error = placement_from_matrix(np.linalg.inv(alignment.transform), cutout.size)
        return PlacementFit(placement, alignment, error)
    except (AlignmentError, ValueError) as features_failed:
        try:
            return fit_by_matching(cutout, target_rgb, expected_scale)
        except AlignmentError:
            raise features_failed if isinstance(features_failed, AlignmentError) else AlignmentError(str(features_failed))


# --- matching the picture itself, when its features cannot be matched -------------------------------------------------------

_MATCH_LONG_EDGE_PX = 900  # the search is made on copies this big (long edge)
_MATCH_REFINE_LONG_EDGE_PX = 1800
_MATCH_MIN_SIDE_PX = 24  # a template smaller than this (at search size) matches anything
_MATCH_MIN_COVER = 0.25  # ...and so does one whose long side is less than this share of the target's: the artwork fills the picture
_MATCH_STEPS = 6  # scales tried: this many steps (of `_MATCH_STEP`) either side of the expected one
_MATCH_STEP = 1.08
_MATCH_MIN_SCORE = 0.38  # normalised correlation of the two edge pictures: unrelated pictures give ~0.1-0.3
_MATCH_MIN_PEAK = 6.5  # ...and the best fit must stand this many standard deviations above the correlation's average
_MATCH_CONFIDENT_SCORE = 0.6
_MATCH_EDGE_BLUR_PX = 1.5


def _edge_picture(rgb: np.ndarray) -> np.ndarray:
    """The edges of a picture, softened, as a float image (what is printed, whatever its colours)."""
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    edges = cv2.Canny(cv2.GaussianBlur(grey, (0, 0), 1.0), 40, 120)
    return cv2.GaussianBlur(edges.astype(np.float32), (0, 0), _MATCH_EDGE_BLUR_PX)


def _best_scale(target_edges: np.ndarray, template: np.ndarray, factors: list[float]) -> tuple[float, float, tuple[int, int], float] | None:
    """(score, factor, top-left, peak) of the best of `template` resized by each factor, laid over `target_edges`; `peak` is
    how many standard deviations the score stands above the average of the correlation over the whole target."""
    best: tuple[float, float, tuple[int, int], float] | None = None
    for factor in factors:
        scaled = cv2.resize(template, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR)
        if min(scaled.shape[:2]) < _MATCH_MIN_SIDE_PX or scaled.shape[0] > target_edges.shape[0] or scaled.shape[1] > target_edges.shape[1]:
            continue
        if max(scaled.shape[:2]) < _MATCH_MIN_COVER * max(target_edges.shape[:2]):
            continue
        if float(scaled.std()) < 1e-3:
            continue
        correlation = cv2.matchTemplate(target_edges, scaled, cv2.TM_CCOEFF_NORMED)
        _low, score, _low_at, at = cv2.minMaxLoc(correlation)
        peak = (float(score) - float(correlation.mean())) / (float(correlation.std()) + 1e-9)
        if best is None or score > best[0]:
            best = (float(score), factor, at, peak)
    return best


def fit_by_matching(cutout: Cutout, target_rgb: np.ndarray, expected_scale: float | None = None) -> PlacementFit:
    """Find where `cutout` lies on `target_rgb` from the two pictures' edges, at whatever scale it is there.

    For when the features of the artwork cannot be matched (plain artwork, few marks): the cutout's edges are laid over
    the target's at many scales -- near `expected_scale` first, if that is known (the two files' resolutions say what it is
    likely to be) -- and the best fit is refined on bigger copies. No rotation is looked for (the reviewer turns it by hand).
    Raises `AlignmentError` if nothing fits well enough.
    """
    height, width = target_rgb.shape[:2]
    cut_w, cut_h = cutout.size
    inside = cv2.erode(cutout.rgba[..., 3], np.ones((9, 9), np.uint8)) > 0  # (the cutout's own border is not artwork)
    flat = cutout.on_white()

    def pictures(image_scale: float) -> tuple[np.ndarray, np.ndarray]:
        target = _edge_picture(cv2.resize(target_rgb, None, fx=image_scale, fy=image_scale, interpolation=cv2.INTER_AREA))
        template = _edge_picture(cv2.resize(flat, None, fx=image_scale, fy=image_scale, interpolation=cv2.INTER_AREA))
        keep = cv2.resize(inside.astype(np.uint8), (template.shape[1], template.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        template[~keep] = 0
        return target, template

    search = min(1.0, _MATCH_LONG_EDGE_PX / max(height, width))
    target_edges, template = pictures(search)
    centre_scale = expected_scale if expected_scale is not None and expected_scale > 0 else 1.0  # (the same resolution, if nothing says)
    best = _best_scale(target_edges, template, [centre_scale * _MATCH_STEP**k for k in range(-_MATCH_STEPS, _MATCH_STEPS + 1)])
    if best is None or best[0] < _MATCH_MIN_SCORE or best[3] < _MATCH_MIN_PEAK:
        raise AlignmentError("The artwork could not be found in this file by its edges either.")
    score, scale, (left, top), used = best[0], best[1], best[2], search

    fine = min(1.0, _MATCH_REFINE_LONG_EDGE_PX / max(height, width))
    if fine > search:
        fine_target, fine_template = pictures(fine)
        refined = _best_scale(fine_target, fine_template, [scale * (1 + 0.006 * k) for k in range(-8, 9)])
        if refined is not None and refined[0] >= 0.9 * score:
            score, scale, (left, top), used = refined[0], refined[1], refined[2], fine
    centre = (left / used + cut_w * scale / 2.0, top / used + cut_h * scale / 2.0)
    placement = Placement(centre, 0.0, float(scale))
    alignment = AlignmentResult(
        transform=np.linalg.inv(placement.matrix(cutout.size)),  # target pixels -> cutout pixels
        residual_error_px=0.0,
        residual_basis="edges",
        method="template",
        match_count=0,
        inlier_count=0,
        inlier_ratio=score,
        low_confidence=score < _MATCH_CONFIDENT_SCORE,
        threshold_px=0.0,
    )
    return PlacementFit(placement, alignment, 0.0)
