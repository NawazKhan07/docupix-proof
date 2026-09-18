"""Isolation geometry: what bounds the compared artwork area in one file
(Architecture §3.3.2-3.3.4). Coordinates are normalized (0-1) so a geometry
resolved at one render resolution still applies after a re-render at another
DPI/zoom (Architecture §3.1: "Emits normalized (0-1) coordinates so results
are resolution-independent")."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class IsolationTool(Enum):
    CROP = "crop"
    POLYGON_TRACE = "polygon_trace"
    ERASE = "erase"


@dataclass
class NormalizedRect:
    """A rectangle in normalized (0-1) image coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.x1 < self.x0:
            self.x0, self.x1 = self.x1, self.x0
        if self.y1 < self.y0:
            self.y0, self.y1 = self.y1, self.y0

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (
            round(self.x0 * width),
            round(self.y0 * height),
            round(self.x1 * width),
            round(self.y1 * height),
        )


@dataclass
class NormalizedPolygon:
    """An ordered point sequence in normalized (0-1) coordinates.

    Used both for closed boundaries (a traced dieline loop) and open
    strokes (a freehand erase path) — callers know which they asked for.
    """

    points: list[tuple[float, float]]

    def to_pixels(self, width: int, height: int) -> list[tuple[int, int]]:
        return [(round(x * width), round(y * height)) for x, y in self.points]


@dataclass
class IsolationGeometry:
    """The resolved artwork boundary + exclusions for one file (original or suspect)."""

    boundary: NormalizedRect | NormalizedPolygon | None = None
    erase_regions: list[NormalizedPolygon] = field(default_factory=list)
    erase_brush_radius_norm: float = 0.01
    rotation_deg: float = 0.0  # auto-rotate correction applied on Save Crop (FR-3.2)
