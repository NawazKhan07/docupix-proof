"""Spatial clustering of raw difference candidates into findings (SRS FR-7.3, Architecture §3.3.6).

Each difference is a finding of its own, exactly where it is. Two are joined only where they are really one thing
(`ClusteringConfig` says which): the bars of a barcode and the dots of Braille; the letters and words of one changed
line of type that adjoin along it; the same place found by two detectors. Nothing is joined because it is merely near
another: a digit changed inside a recoloured panel is not part of the panel's finding, and two differences a few
millimetres apart are two findings. (Where several small ones really are one -- the bars of a barcode drawn as marks --
the reviewer merges them by hand.)
"""

from __future__ import annotations

import math
from collections import defaultdict

from app.models.diff import BoundingBox, DiffCandidate, DiffSubtype, DiffType

_SYMBOL_TYPES = frozenset({DiffType.BARCODE, DiffType.BRAILLE})
_WORD_SIZE_PX = 60  # a box this tall or less is a letter or a word of ordinary type
_LETTER_SPACING = 0.3  # bigger letters are joined only this share of their height apart
_SAME_PLACE_RATIO = 4.0  # a box inside another of at most this many times its area is the same place found with another extent


def box_gap(a: BoundingBox, b: BoundingBox) -> float:
    """Shortest distance between two boxes' edges; 0 if they touch or overlap."""
    dx = max(a.x0 - b.x1, b.x0 - a.x1, 0.0)
    dy = max(a.y0 - b.y1, b.y0 - a.y1, 0.0)
    return math.hypot(dx, dy)


class _DisjointSet:
    def __init__(self, size: int):
        self._parent = list(range(size))

    def find(self, item: int) -> int:
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]  # path halving
            item = self._parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self._parent[root_b] = root_a


def _inside(small: BoundingBox, large: BoundingBox, share: float = 0.9) -> bool:
    """Whether at least `share` of `small` lies inside `large` (and `large` is the bigger)."""
    return large.area > small.area and small.area > 0 and small.intersection_area(large) >= share * small.area


def _same_line(a: BoundingBox, b: BoundingBox, overlap: float, gap_heights: float, max_height: float, max_gap: float) -> bool:
    """Whether two boxes are letters/words of one line: level with each other, and close along it."""
    tall = max(a.height, b.height)
    if tall <= 0 or tall > max_height:
        return False
    shared = min(a.y1, b.y1) - max(a.y0, b.y0)
    if shared < overlap * min(a.height, b.height):
        return False
    gap = max(a.x0 - b.x1, b.x0 - a.x1, 0.0)
    # Small type has word spaces between its letters' groups (a line's words are about as far apart as it is tall); the
    # letters of a big heading are a fraction of their height apart, and two big blocks side by side are not its letters.
    allowed = gap_heights * tall if tall <= _WORD_SIZE_PX else _LETTER_SPACING * tall
    return gap <= max(3.0, min(allowed, max_gap))


def _moved_alike(tolerance_px: float, tolerance_share: float, candidates: list[DiffCandidate]):
    """A link test: words that moved join only words that moved by about the same (their displacement is `magnitude`)."""

    def compatible(i: int, j: int) -> bool:
        a, b = candidates[i], candidates[j]
        a_moved, b_moved = a.subtype is DiffSubtype.MOVED, b.subtype is DiffSubtype.MOVED
        if not (a_moved or b_moved):
            return True
        if a_moved != b_moved:
            return False  # a moved word and a changed one are different findings
        return abs(a.magnitude - b.magnitude) <= max(tolerance_px, tolerance_share * max(a.magnitude, b.magnitude))

    return compatible


def cluster_candidates(
    candidates: list[DiffCandidate],
    distance_px: float,
    line_overlap: float = 0.5,
    line_gap_heights: float = 1.2,
    line_max_height_px: float = 400.0,
    touch_px: float = 2.0,
    duplicate_iou: float = 0.6,
    moved_tolerance: tuple[float, float] = (6.0, 0.3),
    line_max_gap_px: float = 60.0,
) -> list[list[DiffCandidate]]:
    """Group candidates into findings (see the module's account). Groups come back in reading order (top-to-bottom, then
    left-to-right) so callers get stable, human-friendly numbering."""
    if not candidates:
        return []
    sets = _DisjointSet(len(candidates))
    compatible = _moved_alike(*moved_tolerance, candidates)
    reach = max(distance_px, 3.0, line_max_gap_px)
    order = sorted(range(len(candidates)), key=lambda index: candidates[index].bbox.x0)
    for offset, i in enumerate(order):
        a = candidates[i]
        for j in order[offset + 1 :]:
            b = candidates[j]
            if b.bbox.x0 - a.bbox.x1 > reach:
                break
            if _joined(a, b, distance_px, line_overlap, line_gap_heights, line_max_height_px, touch_px, duplicate_iou, line_max_gap_px) and compatible(i, j):
                sets.union(i, j)

    components: dict[int, list[DiffCandidate]] = defaultdict(list)
    for position, candidate in enumerate(candidates):
        components[sets.find(position)].append(candidate)
    groups = list(components.values())
    for members in groups:
        members.sort(key=lambda c: (c.bbox.y0, c.bbox.x0))
    return sorted(groups, key=_reading_order_key)


def _joined(
    a: DiffCandidate,
    b: DiffCandidate,
    distance_px: float,
    line_overlap: float,
    line_gap_heights: float,
    line_max_height: float,
    touch_px: float,
    duplicate_iou: float,
    line_max_gap_px: float,
) -> bool:
    """Whether two candidates are one finding."""
    box_a, box_b = a.bbox, b.bbox
    if a.type in _SYMBOL_TYPES or b.type in _SYMBOL_TYPES:
        # the bars of a barcode, the dots of Braille: one symbol
        return a.type is b.type and box_gap(box_a, box_b) <= distance_px
    nested = _inside(box_a, box_b) or _inside(box_b, box_a)
    similar = nested and max(box_a.area, box_b.area) <= _SAME_PLACE_RATIO * max(1.0, min(box_a.area, box_b.area))
    if a.category != b.category:
        # two detectors' boxes for one place (a new logo is both a colour and a graphics difference; a dot that is gone is
        # a graphic missing and the colour of its fill differing)
        return box_a.iou(box_b) >= duplicate_iou or similar
    if nested:
        if similar:
            return True  # one inside a box only a little bigger is that box again
        outer, inner = (a, b) if _inside(box_b, box_a) else (b, a)
        # what lies inside a graphic that was added or removed is part of it (the type on a removed panel); a small
        # difference inside a recoloured area, or inside a graphic that both files have, is a finding of its own
        return outer.type is DiffType.GRAPHICS and outer.subtype in (DiffSubtype.ADDED, DiffSubtype.REMOVED) and inner.subtype is outer.subtype
    if box_a.iou(box_b) >= duplicate_iou:
        return True  # the same place, found twice
    if box_gap(box_a, box_b) <= touch_px and _shares_a_side(box_a, box_b):
        return True  # one difference cut in two (by a tile, a fold): boxes that meet along a side
    if a.region or b.region:
        return False  # a recoloured area is not a letter of a line
    return _same_line(box_a, box_b, line_overlap, line_gap_heights, line_max_height, line_max_gap_px)


def _shares_a_side(a: BoundingBox, b: BoundingBox) -> bool:
    """Whether two touching boxes lie side by side (or one above the other) with a good share of their extent in common,
    rather than meeting at a corner."""
    across = min(a.x1, b.x1) - max(a.x0, b.x0)
    down = min(a.y1, b.y1) - max(a.y0, b.y0)
    if across > 0 and down > 0:  # overlapping: they must overlap a good part of the smaller one
        return a.intersection_area(b) >= 0.25 * min(a.area, b.area)
    if across > 0:
        return across >= 0.5 * min(a.width, b.width)
    if down > 0:
        return down >= 0.5 * min(a.height, b.height)
    return False


def _reading_order_key(group: list[DiffCandidate]) -> tuple[float, float]:
    return (min(c.bbox.y0 for c in group), min(c.bbox.x0 for c in group))


def union_box(candidates: list[DiffCandidate]) -> BoundingBox:
    box = candidates[0].bbox
    for candidate in candidates[1:]:
        box = box.union(candidate.bbox)
    return box
