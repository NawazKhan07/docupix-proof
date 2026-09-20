"""Spatial clustering of raw difference candidates (SRS FR-7.3, Architecture §3.3.6).

FR-7.3 wants "every word in a moved sentence, every missing line of a
barcode" reported as one bounding region rather than N. That is single-linkage
clustering: two candidates belong together if their boxes are within a
distance of each other, and grouping is transitive through chains. This is
exactly DBSCAN with `min_samples=1` (every point is a core point, so
clusters are the connected components of the epsilon-neighbourhood graph),
written directly rather than pulling in scikit-learn and its dependency tree
for it -- the trade-off decided at project setup.

There are two ways candidates join a cluster:

* **Proximity, within one type.** Nearby differences of the same kind are one
  finding (the words of a missing sentence).
* **Overlap, across types.** Different detectors see the same physical change
  differently -- a new logo is both a colour and a graphics difference -- so
  boxes that substantially overlap are the same finding, reported once. On a
  real carton pair, keeping them apart gave a colour box nested inside a
  graphics box for every change.

Distance is the gap between the two *boxes*, not between their centroids: a
long word and a short one sit far apart centre-to-centre while nearly
touching, and centroid distance would split them.

Single linkage has one weakness: it chains. Where two files' layouts differ, every word
is near another, and hundreds of separate changes link into one 'finding' the size of the
carton. So a finding whose members cover little of its own box is re-clustered with a
shorter link until it falls into its real pieces (`_split_sparse`), and words that moved are
linked only to words that moved alike -- one sentence moved is one move.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from app.models.diff import BoundingBox, DiffCandidate, DiffSubtype, DiffType


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


def _link_within(
    boxes: list[BoundingBox],
    positions: list[int],
    distance_px: float,
    sets: _DisjointSet,
    compatible=None,
) -> None:
    """Union into `sets` every pair of `boxes` whose gap is within `distance_px`.

    `positions[i]` is where `boxes[i]` lives in `sets`. Sweeps left to right by
    x0 so each box is only compared with those that could possibly be near it,
    instead of every pair: a page of moved words can produce hundreds of
    candidates, and a noisy pair thousands.
    """
    order = sorted(range(len(boxes)), key=lambda index: boxes[index].x0)
    for offset, i in enumerate(order):
        box_i = boxes[i]
        for j in order[offset + 1 :]:
            box_j = boxes[j]
            # Sorted by x0, so once a box starts beyond reach of box_i every
            # later one does too.
            if box_j.x0 - box_i.x1 > distance_px:
                break
            if box_gap(box_i, box_j) <= distance_px and (compatible is None or compatible(positions[i], positions[j])):
                sets.union(positions[i], positions[j])


def _link_overlapping(
    boxes: list[BoundingBox], overlap_ratio: float, max_area_ratio: float, sets: _DisjointSet
) -> None:
    """Union boxes that substantially overlap, whatever kind of difference they are.

    "Substantially" is measured against the SMALLER box, so a small finding
    nested inside a larger one counts as the same change. Pairs whose areas
    differ enormously are left alone: that is a stray outline artefact around
    genuine findings, and merging through it would fuse everything it touches.
    """
    order = sorted(range(len(boxes)), key=lambda index: boxes[index].x0)
    for offset, i in enumerate(order):
        box_i = boxes[i]
        for j in order[offset + 1 :]:
            box_j = boxes[j]
            if box_j.x0 > box_i.x1:
                break
            overlap = box_i.intersection_area(box_j)
            if overlap <= 0:
                continue
            smaller = max(1.0, min(box_i.area, box_j.area))
            larger = max(box_i.area, box_j.area)
            if overlap / smaller >= overlap_ratio and larger / smaller <= max_area_ratio:
                sets.union(i, j)


def _moved_alike(tolerance_px: float, tolerance_share: float, candidates: list[DiffCandidate]):
    """A link test: words that moved join only words that moved by about the same (their displacement is `magnitude`)."""

    def compatible(i: int, j: int) -> bool:
        a, b = candidates[i], candidates[j]
        a_moved, b_moved = a.subtype is DiffSubtype.MOVED, b.subtype is DiffSubtype.MOVED
        if not (a_moved or b_moved):
            return True
        if a_moved != b_moved:
            return False  # a moved word and a changed one are different findings, unless their boxes overlap
        return abs(a.magnitude - b.magnitude) <= max(tolerance_px, tolerance_share * max(a.magnitude, b.magnitude))

    return compatible


def _cluster(
    candidates: list[DiffCandidate],
    distance_px: float,
    overlap_ratio: float,
    max_area_ratio: float,
    moved_tolerance: tuple[float, float],
) -> list[list[DiffCandidate]]:
    """Single-linkage groups (unsorted): same-type neighbours, alike-moved words, and overlapping boxes."""
    sets = _DisjointSet(len(candidates))

    by_type: dict[DiffType, list[int]] = defaultdict(list)
    for position, candidate in enumerate(candidates):
        by_type[candidate.type].append(position)
    compatible = _moved_alike(*moved_tolerance, candidates)
    for positions in by_type.values():
        _link_within([candidates[p].bbox for p in positions], positions, distance_px, sets, compatible)

    _link_overlapping([c.bbox for c in candidates], overlap_ratio, max_area_ratio, sets)

    components: dict[int, list[DiffCandidate]] = defaultdict(list)
    for position, candidate in enumerate(candidates):
        components[sets.find(position)].append(candidate)
    return list(components.values())


def _fill(members: list[DiffCandidate], box: BoundingBox) -> float:
    """The share of `box` that the members' own boxes cover (a chain across a page covers almost none of it)."""
    cell = max(2.0, max(box.width, box.height) / 160.0)
    columns, rows = max(1, math.ceil(box.width / cell)), max(1, math.ceil(box.height / cell))
    covered = np.zeros((rows, columns), bool)
    for member in members:
        b = member.bbox
        covered[
            max(0, int((b.y0 - box.y0) // cell)) : min(rows, int((b.y1 - box.y0) // cell) + 1),
            max(0, int((b.x0 - box.x0) // cell)) : min(columns, int((b.x1 - box.x0) // cell) + 1),
        ] = True
    return float(covered.mean())


def _split_sparse(
    members: list[DiffCandidate],
    distance_px: float,
    overlap_ratio: float,
    max_area_ratio: float,
    moved_tolerance: tuple[float, float],
    min_fill: float,
    min_area_px: float,
    min_link_px: float,
) -> list[list[DiffCandidate]]:
    """Break a chain into its pieces: a group whose members fill little of its box is re-linked more tightly."""
    if len(members) < 2 or min_fill <= 0:
        return [members]
    box = union_box(members)
    if box.area < min_area_px or _fill(members, box) >= min_fill:
        return [members]
    link = distance_px / 2
    while link >= min_link_px:
        parts = _cluster(members, link, overlap_ratio, max_area_ratio, moved_tolerance)
        if len(parts) > 1:
            return [
                piece
                for part in parts
                for piece in _split_sparse(
                    part, link, overlap_ratio, max_area_ratio, moved_tolerance, min_fill, min_area_px, min_link_px
                )
            ]
        link /= 2
    return [members]


def cluster_candidates(
    candidates: list[DiffCandidate],
    distance_px: float,
    overlap_ratio: float = 0.2,
    max_area_ratio: float = 60.0,
    moved_tolerance: tuple[float, float] = (6.0, 0.3),
    min_fill: float = 0.0,
    split_min_area_px: float = 60000.0,
    min_link_px: float = 4.0,
    huge_area_px: float = float("inf"),
) -> list[list[DiffCandidate]]:
    """Group candidates into findings.

    Same-type candidates within `distance_px` of each other are joined (words that moved only to words
    that moved alike), and any candidates (of any type) whose boxes overlap by `overlap_ratio` of the
    smaller are joined too. With `min_fill` set, a group that is a chain rather than one thing is split
    (see `_split_sparse`), and a candidate over `huge_area_px` is a finding of its own. Groups come back in reading order (top-to-bottom, then left-to-right) so
    callers get stable, human-friendly numbering.
    """
    if not candidates:
        return []

    # A candidate the size of a panel (where everything differs, one 'graphic changed' box spans most of the
    # sheet) overlaps everything and would fuse the whole page into one finding. Those stay findings of their
    # own (huge ones only touching each other, as duplicates of one change) and the rest cluster normally.
    huge = [c for c in candidates if c.bbox.area > huge_area_px]
    normal = [c for c in candidates if c.bbox.area <= huge_area_px]
    groups = _cluster(huge, 0.0, overlap_ratio, max_area_ratio, moved_tolerance) if huge else []
    for members in _cluster(normal, distance_px, overlap_ratio, max_area_ratio, moved_tolerance) if normal else []:
        groups += _split_sparse(
            members, distance_px, overlap_ratio, max_area_ratio, moved_tolerance, min_fill, split_min_area_px, min_link_px
        )
    for members in groups:
        members.sort(key=lambda c: (c.bbox.y0, c.bbox.x0))
    return sorted(groups, key=_reading_order_key)


def _reading_order_key(group: list[DiffCandidate]) -> tuple[float, float]:
    return (min(c.bbox.y0 for c in group), min(c.bbox.x0 for c in group))


def union_box(candidates: list[DiffCandidate]) -> BoundingBox:
    box = candidates[0].bbox
    for candidate in candidates[1:]:
        box = box.union(candidate.bbox)
    return box
