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
    LIFT_SUBJECT = "lift_subject"  # press and hold on the artwork to lift it from its background
    PLACE_OUTLINE = "place_outline"  # lay another file's lifted outline over this one, and trim to it
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
    # Parts to leave out: a NormalizedRect is a region, a NormalizedPolygon a freehand stroke
    # painted with `erase_brush_radius_norm`.
    erase_regions: list[NormalizedRect | NormalizedPolygon] = field(default_factory=list)
    erase_brush_radius_norm: float = 0.01
    rotation_deg: float = 0.0  # auto-rotate correction applied on Save Crop (FR-3.2)
    # Where the boundary came from: "manual" (a crop, or an exact traced dieline), "lifted"
    # (segmented from its background) or "placed" (another file's outline laid over this one).
    boundary_origin: str = "manual"
    # How far inside a polygon boundary comparing starts, as a share of the page's long edge: a lifted
    # outline includes the die line drawn on it (or the blurred fringe of a photographed edge), which is
    # not printed artwork. Rectangles are not inset -- a crop's edge is not a line.
    edge_inset_norm: float = 0.0

    @property
    def anchors_alignment(self) -> bool:
        """Whether a polygon boundary may be used to REGISTER the files, not just to mask them.

        A lifted outline is only as exact as the segmentation, and a placed one is the
        other file's outline moved by hand: registering two files on such outlines would
        report a fit that is largely an artefact of how they were drawn, and a residual
        that means nothing ("do not silently claim perfect alignment"). They trim the
        comparison area; the artwork's own features do the registering.
        """
        return self.boundary_origin not in ("lifted", "placed")
