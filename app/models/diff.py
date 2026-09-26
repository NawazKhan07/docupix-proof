"""Difference objects produced by the detectors (Architecture §3.4).

Every detector reports `DiffCandidate`s whose boxes live in the shared
aligned space (the original image's pixel frame), which is what lets Phase
5 project one difference into both files' own coordinates (FR-7.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DiffType(Enum):
    TEXT = "text"
    NUMBER = "number"
    COLOUR = "colour"
    GRAPHICS = "graphics"
    BARCODE = "barcode"
    BRAILLE = "braille"


class DiffSubtype(Enum):
    """FR-6.1 distinguishes a thing changing from the same thing moving."""

    CHANGED = "changed"
    MOVED = "moved"
    ADDED = "added"
    REMOVED = "removed"


# What a finding IS, in the words a reviewer uses. Each has its own colour (`app.models.categories`), so what kind of
# difference a box marks is seen at a glance. (There is no severity: a difference is described, not ranked.)
_CATEGORIES = {
    DiffType.TEXT: "text",
    DiffType.NUMBER: "text",
    DiffType.COLOUR: "colour",
    DiffType.GRAPHICS: "graphics",
    DiffType.BARCODE: "barcode",
    DiffType.BRAILLE: "braille",
}


@dataclass(frozen=True)
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    def union(self, other: "BoundingBox") -> "BoundingBox":
        return BoundingBox(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    def intersection_area(self, other: "BoundingBox") -> float:
        width = min(self.x1, other.x1) - max(self.x0, other.x0)
        height = min(self.y1, other.y1) - max(self.y0, other.y0)
        return max(0.0, width) * max(0.0, height)

    def iou(self, other: "BoundingBox") -> float:
        intersection = self.intersection_area(other)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def scaled(self, factor: float) -> "BoundingBox":
        return BoundingBox(self.x0 * factor, self.y0 * factor, self.x1 * factor, self.y1 * factor)

    def scaled_xy(self, factor_x: float, factor_y: float) -> "BoundingBox":
        """Scaled by one factor across and another down: a reduced picture's width and height are whole numbers, so its
        two scales are not quite the same, and one factor for both puts a box a few pixels off at the far edge."""
        return BoundingBox(self.x0 * factor_x, self.y0 * factor_y, self.x1 * factor_x, self.y1 * factor_y)


@dataclass
class DiffCandidate:
    bbox: BoundingBox  # in the shared aligned space
    type: DiffType
    subtype: DiffSubtype
    confidence: float
    detail: str = ""
    # Detector-specific strength, kept numeric so nothing has
    # to parse `detail`: peak delta-E for colour, 1 - SSIM for graphics, and
    # 0 where a detector has no meaningful scale (text, symbols, Braille).
    magnitude: float = 0.0
    # A long thin stroke that is all there is to the difference (a die line or fold line drawn in one file and not the
    # other, or in another colour): whether it is ignored depends on where it lies (see `worker._drop_die_lines`).
    line_work: bool = False
    # A difference in what was READ that the pixels do not bear out (the words differ, the ink matches: most likely the
    # reader's mistake, not the artwork's). Listed apart, and only shown on request.
    uncertain: bool = False
    # A recoloured AREA (a panel, a tint): its finding is the colour; what is written on it is findings of their own, so
    # the words in its box are not read into it.
    region: bool = False

    @property
    def category(self) -> str:
        return _CATEGORIES[self.type]


class DiffStatus(Enum):
    ACTIVE = "active"
    IGNORED = "ignored"


@dataclass
class DiffCluster:
    """A group of related candidates reported as one finding (FR-7.3).

    Carries its box in *both* files' coordinates (FR-7.4), so the two
    viewers mark the same difference in the corresponding place rather than
    each deriving a position of its own.
    """

    id: int
    bbox_in_original: BoundingBox
    bbox_in_suspect: BoundingBox
    type: DiffType
    members: list[DiffCandidate] = field(default_factory=list)
    status: DiffStatus = DiffStatus.ACTIVE
    detail: str = ""

    @property
    def category(self) -> str:
        return _CATEGORIES[self.type]

    @property
    def subtype(self) -> DiffSubtype:
        """What happened to it: changed, moved, added or removed (that of the strongest member that has one)."""
        for member in sorted(self.members, key=lambda m: (m.type is not self.type, -m.magnitude)):
            return member.subtype
        return DiffSubtype.CHANGED

    @property
    def uncertain(self) -> bool:
        """Whether every member is a reading the pixels do not bear out (see `DiffCandidate.uncertain`)."""
        return bool(self.members) and all(member.uncertain for member in self.members)

    @property
    def types(self) -> set[DiffType]:
        """Every kind of difference merged into this finding (usually just one)."""
        return {member.type for member in self.members} or {self.type}

    @property
    def is_ignored(self) -> bool:
        return self.status is DiffStatus.IGNORED

    def ignore(self) -> None:
        """Soft-dismiss (FR-7.5). Never deletes, so the finding stays auditable."""
        self.status = DiffStatus.IGNORED

    def restore(self) -> None:
        self.status = DiffStatus.ACTIVE


# Scan vs Digital always finds both graphics and text changes (see `app.engine.worker.run_comparison`); its
# results page just filters which are SHOWN, so ticking a checkbox back on is instant, nothing recompared.
_ALWAYS_SHOWN_TYPES = {DiffType.BARCODE, DiffType.BRAILLE}
_TEXT_TYPES = {DiffType.TEXT, DiffType.NUMBER}
_GRAPHICS_TYPES = {DiffType.COLOUR, DiffType.GRAPHICS}


def visible_clusters(clusters: list[DiffCluster], *, text_wanted: bool, graphics_wanted: bool) -> list[DiffCluster]:
    """`clusters` restricted to what Scan vs Digital's graphics/text checkboxes ask to see.

    Barcode and Braille are compared as marks either way and are always shown. A finding merged from more
    than one kind of detector (a change that is both structural and textual) is shown if either is wanted:
    hiding graphics must not hide a real text change that happened to land on the same spot.
    """
    def shown(cluster: DiffCluster) -> bool:
        types = cluster.types
        if types & _ALWAYS_SHOWN_TYPES:
            return True
        return bool((text_wanted and types & _TEXT_TYPES) or (graphics_wanted and types & _GRAPHICS_TYPES))

    return [cluster for cluster in clusters if shown(cluster)]
