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
"""

from __future__ import annotations

import math
from collections import defaultdict

from app.models.diff import BoundingBox, DiffCandidate, DiffType


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
    boxes: list[BoundingBox], positions: list[int], distance_px: float, sets: _DisjointSet
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
            if box_gap(box_i, box_j) <= distance_px:
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


def cluster_candidates(
    candidates: list[DiffCandidate],
    distance_px: float,
    overlap_ratio: float = 0.2,
    max_area_ratio: float = 60.0,
) -> list[list[DiffCandidate]]:
    """Group candidates into findings.

    Same-type candidates within `distance_px` of each other are joined, and any
    candidates (of any type) whose boxes overlap by `overlap_ratio` of the
    smaller are joined too. Groups come back in reading order (top-to-bottom,
    then left-to-right) so callers get stable, human-friendly numbering.
    """
    if not candidates:
        return []

    sets = _DisjointSet(len(candidates))

    by_type: dict[DiffType, list[int]] = defaultdict(list)
    for position, candidate in enumerate(candidates):
        by_type[candidate.type].append(position)
    for positions in by_type.values():
        _link_within([candidates[p].bbox for p in positions], positions, distance_px, sets)

    _link_overlapping([c.bbox for c in candidates], overlap_ratio, max_area_ratio, sets)

    components: dict[int, list[DiffCandidate]] = defaultdict(list)
    for position, candidate in enumerate(candidates):
        components[sets.find(position)].append(candidate)

    groups = []
    for members in components.values():
        members.sort(key=lambda c: (c.bbox.y0, c.bbox.x0))
        groups.append(members)
    return sorted(groups, key=_reading_order_key)


def _reading_order_key(group: list[DiffCandidate]) -> tuple[float, float]:
    return (min(c.bbox.y0 for c in group), min(c.bbox.x0 for c in group))


def union_box(candidates: list[DiffCandidate]) -> BoundingBox:
    box = candidates[0].bbox
    for candidate in candidates[1:]:
        box = box.union(candidate.bbox)
    return box
