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


# FR-7.1 asks for three reporting categories; the finer DiffTypes fold into
# them. Barcode and Braille findings have their own dedicated result panels
# (FR-10.2 / FR-11.1) but still read as "graphics" in the general diff list.
_CATEGORIES = {
    DiffType.TEXT: "text",
    DiffType.NUMBER: "text",
    DiffType.COLOUR: "colour",
    DiffType.GRAPHICS: "graphics",
    DiffType.BARCODE: "graphics",
    DiffType.BRAILLE: "graphics",
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


@dataclass
class DiffCandidate:
    bbox: BoundingBox  # in the shared aligned space
    type: DiffType
    subtype: DiffSubtype
    confidence: float
    detail: str = ""
    # Detector-specific strength, kept numeric so severity rules never have
    # to parse `detail`: peak delta-E for colour, 1 - SSIM for graphics, and
    # 0 where a detector has no meaningful scale (text, symbols, Braille).
    magnitude: float = 0.0

    @property
    def category(self) -> str:
        return _CATEGORIES[self.type]


class Severity(Enum):
    """FR-7.2 tiers. PENDING SIGN-OFF: the SRS says "Critical and others"
    without finalising the list (SRS §7). The rules that assign these, and
    the colour for each, live together in `engine/aggregation/severity.py`."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"

    @property
    def rank(self) -> int:
        """Lower is more severe, so sorting by rank puts Critical first."""
        return _SEVERITY_RANK[self]


_SEVERITY_RANK = {Severity.CRITICAL: 0, Severity.MAJOR: 1, Severity.MINOR: 2}


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
    severity: Severity
    members: list[DiffCandidate] = field(default_factory=list)
    status: DiffStatus = DiffStatus.ACTIVE
    detail: str = ""

    @property
    def category(self) -> str:
        return _CATEGORIES[self.type]

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
