from __future__ import annotations

import numpy as np

from app.config import ClusteringConfig
from app.engine.aggregation.aggregator import build_clusters
from app.models.diff import (
    BoundingBox,
    DiffCandidate,
    DiffStatus,
    DiffSubtype,
    DiffType,
)

CONFIG = ClusteringConfig(distance_px=30.0)
IDENTITY = np.eye(3)


def _candidate(x0, y0, x1, y1, diff_type=DiffType.TEXT, subtype=DiffSubtype.CHANGED, magnitude=0.0):
    return DiffCandidate(
        bbox=BoundingBox(x0, y0, x1, y1),
        type=diff_type,
        subtype=subtype,
        confidence=1.0,
        detail=f"{diff_type.value} @ {x0},{y0}",
        magnitude=magnitude,
    )


def test_candidates_become_clusters_with_both_boxes_and_a_category():
    clusters = build_clusters([_candidate(100, 100, 140, 120)], IDENTITY, CONFIG)

    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.category == "text" and cluster.subtype is DiffSubtype.CHANGED
    assert cluster.bbox_in_original == BoundingBox(100, 100, 140, 120)
    assert cluster.bbox_in_suspect == BoundingBox(100, 100, 140, 120)  # identity transform
    assert cluster.status is DiffStatus.ACTIVE


def test_suspect_box_is_carried_through_the_inverse_transform():
    shift = np.array([[1.0, 0.0, 30.0], [0.0, 1.0, 12.0], [0.0, 0.0, 1.0]])
    (cluster,) = build_clusters([_candidate(100, 100, 140, 120)], shift, CONFIG)

    assert cluster.bbox_in_original == BoundingBox(100, 100, 140, 120)
    assert cluster.bbox_in_suspect == BoundingBox(70, 88, 110, 108)


def test_each_kind_of_difference_is_its_own_category():
    kinds = {
        DiffType.TEXT: "text",
        DiffType.NUMBER: "text",
        DiffType.COLOUR: "colour",
        DiffType.GRAPHICS: "graphics",
        DiffType.BARCODE: "barcode",
        DiffType.BRAILLE: "braille",
    }
    for diff_type, category in kinds.items():
        (cluster,) = build_clusters([_candidate(0, 0, 40, 20, diff_type)], IDENTITY, CONFIG)
        assert cluster.category == category


def test_a_cluster_is_labelled_by_its_most_specific_member():
    """One place seen by two detectors: a changed word is a text finding before it is a colour one."""
    candidates = [
        _candidate(0, 0, 40, 20, DiffType.COLOUR, DiffSubtype.CHANGED, magnitude=30.0),
        _candidate(0, 0, 40, 20, DiffType.TEXT, DiffSubtype.CHANGED),
    ]
    (cluster,) = build_clusters(candidates, IDENTITY, CONFIG)
    assert cluster.category == "text"
    assert len(cluster.members) == 2


def test_ids_follow_reading_order_starting_at_one():
    candidates = [
        _candidate(500, 500, 540, 520),
        _candidate(0, 0, 40, 20),
        _candidate(300, 0, 340, 20),
    ]
    clusters = build_clusters(candidates, IDENTITY, CONFIG)

    assert [c.id for c in clusters] == [1, 2, 3]
    assert [c.bbox_in_original.x0 for c in clusters] == [0, 300, 500]


def test_the_words_of_one_line_are_one_finding_and_it_says_what_the_strongest_says():
    words = [_candidate(x, 0, x + 40, 20, subtype=DiffSubtype.MOVED) for x in (0, 50, 100)]  # a few pixels apart, one line
    (cluster,) = build_clusters(words, IDENTITY, CONFIG)
    assert len(cluster.members) == 3
    assert cluster.detail.startswith("text @")  # one line of description, not a count


def test_differences_on_different_lines_are_findings_of_their_own_however_near():
    lines = [_candidate(0, 0, 40, 20), _candidate(0, 26, 40, 46)]  # six pixels between them
    assert len(build_clusters(lines, IDENTITY, CONFIG)) == 2


def test_only_the_bars_of_a_barcode_are_merged_by_distance():
    bars = [_candidate(x, 0, x + 6, 100, DiffType.BARCODE) for x in (0, 16, 32)]
    assert len(build_clusters(bars, IDENTITY, ClusteringConfig(distance_px=30))) == 1
    assert len(build_clusters(bars, IDENTITY, ClusteringConfig(distance_px=5))) == 3
    far_apart = [_candidate(x, 0, x + 6, 100, DiffType.BARCODE) for x in (0, 200)]
    assert len(build_clusters(far_apart, IDENTITY, ClusteringConfig(distance_px=30))) == 2


def test_no_candidates_means_no_clusters():
    assert build_clusters([], IDENTITY, CONFIG) == []


def test_ignoring_a_finding_is_a_soft_state_not_a_deletion():
    """FR-7.5, and the Architecture's requirement that it stays auditable."""
    clusters = build_clusters([_candidate(0, 0, 40, 20), _candidate(500, 500, 540, 520)], IDENTITY, CONFIG)
    target = clusters[0]

    target.ignore()

    assert target.is_ignored
    assert target.status is DiffStatus.IGNORED
    assert len(clusters) == 2  # still present in the result
    assert target.members  # and still carrying everything it was built from

    target.restore()
    assert not target.is_ignored
