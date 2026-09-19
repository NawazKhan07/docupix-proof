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
from app.engine.alignment.aligner import align
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


def fit_placement(cutout: Cutout, target_rgb: np.ndarray, config: AlignmentConfig | None = None) -> PlacementFit:
    """Find where `cutout` belongs on `target_rgb` by matching the artwork itself.

    Raises `AlignmentError` if the two do not share enough artwork to be matched.
    """
    config = config or AlignmentConfig()
    width, height = cutout.size
    # The cutout's own outline only masks the search for features; it never anchors the fit.
    outline = IsolationGeometry(
        boundary=NormalizedPolygon(points=[(x / width, y / height) for x, y in cutout.polygon]),
        boundary_origin="lifted",
    )
    alignment = align(cutout.on_white(), target_rgb, config, original_geometry=outline)
    # `alignment.transform` takes target pixels to cutout pixels; placing needs the inverse.
    placement, error = placement_from_matrix(np.linalg.inv(alignment.transform), cutout.size)
    return PlacementFit(placement, alignment, error)
