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
    groups = _merge_elements(list(components.values()))
    for members in groups:
        members.sort(key=lambda c: (c.bbox.y0, c.bbox.x0))
    return sorted(groups, key=_reading_order_key)


_ELEMENT_OVERLAP = 0.6  # two parts of one element (a word, a wordmark) share at least this much of the shorter's height...
_ELEMENT_HEIGHT_RATIO = 1.6  # ...are about as tall as each other (a letter and the line it is in, not a panel beside it)...
_ELEMENT_GAP = 0.35  # ...and, found as the same kind, are at most this share of their height apart (a letter's space)...
_ELEMENT_CROSS_OVERLAP = 0.25  # ...or, found as different kinds (a recoloured letter, a reshaped one), overlap this much
_ELEMENT_NESTED = 0.5  # pieces side by side overlap by no more than this share of the smaller (else one is inside the other)
_ELEMENT_PART_HEIGHT = 0.2  # a finding inside a joined element, at least this share of its height, is a part of it
_SPECK_HOST_MAX_PX = 300  # a changed graphic takes in the colour specks inside it only if it is a mark (a badge), not a panel
_ELEMENT_MAX_ASPECT = 25.0  # an element is not a rule or a band drawn across the whole panel
_ELEMENT_TOUCH_SHARE = 0.06  # pieces of different kinds this close (a share of the line's height, plus...
_ELEMENT_TOUCH_PX = 2.0  # ...this) along one line of type...
_ELEMENT_TOUCH_LINE = 0.8  # ...sharing this much of its height are pieces of one word
_PATTERN_MAX_PX = 40  # the marks of a pattern (the dots of a Braille cell) are small...
_PATTERN_SIZE_RATIO = 1.3  # ...the same size and round or square (either way within this)...
_PATTERN_GAP = 2.0  # ...and at most this many times their size apart (a cell's dots are set on a grid, rows and diagonals)


def _merge_elements(groups: list[list[DiffCandidate]]) -> list[list[DiffCandidate]]:
    """The findings that are pieces of ONE changed element, joined (a word recoloured or retyped, found letter by letter
    by different detectors -- a colour change on one letter, a graphics change on the next two, the counter of an A;
    the dots of one Braille cell): see `_one_element`. Then whatever lies inside a joined element and is a real part of
    it (a letter's counter). A changed word is one finding, not one per letter; two changes a word's width apart, a
    digit changed in a recoloured panel, a moved word beside a changed one, stay what they were."""
    pieces = [group for group in groups if not any(c.type in _SYMBOL_TYPES for c in group)]
    symbols = [group for group in groups if any(c.type in _SYMBOL_TYPES for c in group)]
    if len(pieces) < 2:
        return groups
    boxes = [union_box(group) for group in pieces]
    kinds = [{c.category for c in group} for group in pieces]
    moves = [_move_of(group) for group in pieces]
    dots = [_dot_size(group) for group in pieces]
    sets = _DisjointSet(len(pieces))
    order = sorted(range(len(pieces)), key=lambda index: boxes[index].x0)
    for offset, i in enumerate(order):
        a = boxes[i]
        reach = max(_ELEMENT_GAP * _ELEMENT_HEIGHT_RATIO * a.height, _PATTERN_GAP * _PATTERN_MAX_PX)
        for j in order[offset + 1 :]:
            b = boxes[j]
            if b.x0 - a.x1 > reach:
                break
            if not _moves_agree(moves[i], moves[j]):
                continue
            if _one_element(a, b, same_kind=bool(kinds[i] & kinds[j])) or _one_dot_pattern(a, b, dots[i], dots[j]):
                sets.union(i, j)
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(pieces)):
        components[sets.find(index)].append(index)
    merged: list[list[DiffCandidate]] = []
    merged_boxes: list[BoundingBox] = []
    joined: list[bool] = []
    for indices in components.values():
        merged.append([c for index in indices for c in pieces[index]])
        box = boxes[indices[0]]
        for index in indices[1:]:
            box = box.union(boxes[index])
        merged_boxes.append(box)
        joined.append(len(indices) > 1)
    # what lies inside a joined element and is a real part of it (a letter's counter, the rest of a letter) joins it; so
    # does a colour speck inside a changed graphic (a recoloured badge seen in pieces) -- but a digit or a mark changed
    # inside a recoloured panel is still its own finding
    order = sorted(range(len(merged)), key=lambda index: merged_boxes[index].x0)
    for i in range(len(merged)):
        outer = merged_boxes[i]
        outer_graphic = max(outer.width, outer.height) <= _SPECK_HOST_MAX_PX and any(
            c.type is DiffType.GRAPHICS and c.subtype is DiffSubtype.CHANGED for c in merged[i]
        )
        if not (joined[i] or outer_graphic):
            continue
        for j in order:
            inner = merged_boxes[j]
            if inner.x0 > outer.x1:
                break
            if j == i or not merged[j] or not merged[i] or not _inside(inner, outer):
                continue
            part = joined[i] and inner.height >= _ELEMENT_PART_HEIGHT * outer.height
            speck = outer_graphic and all(c.type is DiffType.COLOUR for c in merged[j])
            if part or speck:
                merged[i] = merged[i] + merged[j]
                merged[j] = []
    return [group for group in merged if group] + symbols


def _move_of(group: list[DiffCandidate]) -> float | None:
    """How far a finding's words moved (None if it is not a move)."""
    moved = [c.magnitude for c in group if c.subtype is DiffSubtype.MOVED]
    return max(moved) if moved and len(moved) == len(group) else (None if not moved else -1.0)


def _moves_agree(a: float | None, b: float | None) -> bool:
    """A moved word joins only a word moved about as far; never a changed one (see `_moved_alike`)."""
    if a is None and b is None:
        return True
    if a is None or b is None or a < 0 or b < 0:
        return False
    return abs(a - b) <= max(6.0, 0.3 * max(a, b))


def _one_pattern(a: BoundingBox, b: BoundingBox) -> bool:
    """Two marks of one pattern: small, the same round or square size, close together."""
    for box in (a, b):
        if max(box.width, box.height) > _PATTERN_MAX_PX or min(box.width, box.height) <= 0:
            return False
        if max(box.width, box.height) > _PATTERN_SIZE_RATIO * min(box.width, box.height):
            return False  # round or square: a word or a line of type is not a dot
    if max(a.width, b.width) > _PATTERN_SIZE_RATIO * min(a.width, b.width):
        return False
    size = max(a.width, a.height, b.width, b.height)
    return 0 < box_gap(a, b) <= _PATTERN_GAP * size  # (touching marks are one mark cut in two, handled elsewhere)


def _dot_size(group: list[DiffCandidate]) -> float | None:
    """The size of the marks a finding is made of, if every one is a small round or square mark (a dot of a pattern)."""
    sizes = []
    for candidate in group:
        box = candidate.bbox
        big, small = max(box.width, box.height), min(box.width, box.height)
        if big > _PATTERN_MAX_PX or small <= 0 or big > _PATTERN_SIZE_RATIO * small:
            return None
        sizes.append(big)
    return max(sizes) if sizes else None


def _one_dot_pattern(a: BoundingBox, b: BoundingBox, dots_a: float | None, dots_b: float | None) -> bool:
    """Two findings made of dots of one size, a few dots apart: one pattern (a row of dots found, then the next row)."""
    if dots_a is None or dots_b is None or max(dots_a, dots_b) > _PATTERN_SIZE_RATIO * min(dots_a, dots_b):
        return False
    return 0 < box_gap(a, b) <= _PATTERN_GAP * max(dots_a, dots_b)


def _one_element(a: BoundingBox, b: BoundingBox, same_kind: bool) -> bool:
    """Whether two findings' boxes are pieces of one element on one line, or two marks of one pattern."""
    if _one_pattern(a, b):
        return True
    short, tall = sorted((max(a.height, 1.0), max(b.height, 1.0)))
    if tall > _ELEMENT_HEIGHT_RATIO * short:
        return False
    if max(a.width / max(a.height, 1.0), b.width / max(b.height, 1.0)) > _ELEMENT_MAX_ASPECT:
        return False
    shared = min(a.y1, b.y1) - max(a.y0, b.y0)
    if shared < _ELEMENT_OVERLAP * short:
        return False
    overlap = a.intersection_area(b)
    smaller = max(min(a.area, b.area), 1.0)
    if overlap > _ELEMENT_NESTED * smaller:
        return False  # one inside the other (an icon on a recoloured band) is not two pieces side by side
    gap = max(a.x0 - b.x0 - b.width, b.x0 - a.x0 - a.width, 0.0)
    if not same_kind:
        # different kinds: where they really overlap -- or where they butt up against each other along one line of type
        # (a word recoloured letter by letter: one letter's colour, the next two's shape), a letter space apart at most
        touching = gap <= _ELEMENT_TOUCH_SHARE * short + _ELEMENT_TOUCH_PX and shared >= _ELEMENT_TOUCH_LINE * short
        return overlap >= _ELEMENT_CROSS_OVERLAP * smaller or touching
    return gap <= _ELEMENT_GAP * tall


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
