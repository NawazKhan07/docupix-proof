"""Spatial clustering of raw difference candidates (SRS FR-7.3, Architecture §3.3.6).

FR-7.3 wants "every word in a moved sentence, every missing line of a
barcode" reported as one bounding region rather than N. That is single-linkage
clustering: two candidates belong together if their boxes are within a
distance of each other, and grouping is transitive through chains. This is
exactly DBSCAN with `min_samples=1` (every point is a core point, so
clusters are the connected components of the epsilon-neighbourhood graph),
written directly rather than pulling in scikit-learn and its dependency tree
for it -- the trade-off decided at project setup.

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


def _link_within(boxes: list[BoundingBox], distance_px: float) -> list[list[int]]:
    """Connected components of "gap <= distance" over `boxes`, as index lists.

    Sweeps left to right by x0 so each box is only compared with those that
    could possibly be near it, instead of every pair: a page of moved words
    can produce hundreds of candidates, and a noisy pair thousands.
    """
    order = sorted(range(len(boxes)), key=lambda index: boxes[index].x0)
    sets = _DisjointSet(len(boxes))

    for position, i in enumerate(order):
        box_i = boxes[i]
        for j in order[position + 1 :]:
            box_j = boxes[j]
            # Sorted by x0, so once a box starts beyond reach of box_i every
            # later one does too.
            if box_j.x0 - box_i.x1 > distance_px:
                break
            if box_gap(box_i, box_j) <= distance_px:
                sets.union(i, j)

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(boxes)):
        components[sets.find(index)].append(index)
    return list(components.values())


def cluster_candidates(
    candidates: list[DiffCandidate], distance_px: float
) -> list[list[DiffCandidate]]:
    """Group candidates into clusters of the same type within `distance_px`.

    Different types are never merged (Architecture §3.3.6: "spatially
    cluster raw DiffCandidate boxes of the same type"), which keeps the
    text / colour / graphics categories of FR-7.1 meaningful. Groups come
    back in reading order (top-to-bottom, then left-to-right) so callers get
    stable, human-friendly numbering.
    """
    by_type: dict[DiffType, list[DiffCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_type[candidate.type].append(candidate)

    groups: list[list[DiffCandidate]] = []
    for same_type in by_type.values():
        boxes = [candidate.bbox for candidate in same_type]
        for component in _link_within(boxes, distance_px):
            members = [same_type[index] for index in component]
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
