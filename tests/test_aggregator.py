from __future__ import annotations

import numpy as np

from app.config import ClusteringConfig
from app.engine.aggregation.aggregator import build_clusters
from app.engine.aggregation.severity import SeverityRule
from app.models.diff import (
    BoundingBox,
    DiffCandidate,
    DiffStatus,
    DiffSubtype,
    DiffType,
    Severity,
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


def test_candidates_become_clusters_with_severity_and_both_boxes():
    clusters = build_clusters([_candidate(100, 100, 140, 120)], IDENTITY, CONFIG)

    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.severity is Severity.CRITICAL  # a changed text run
    assert cluster.bbox_in_original == BoundingBox(100, 100, 140, 120)
    assert cluster.bbox_in_suspect == BoundingBox(100, 100, 140, 120)  # identity transform
    assert cluster.status is DiffStatus.ACTIVE


def test_suspect_box_is_carried_through_the_inverse_transform():
    shift = np.array([[1.0, 0.0, 30.0], [0.0, 1.0, 12.0], [0.0, 0.0, 1.0]])
    (cluster,) = build_clusters([_candidate(100, 100, 140, 120)], shift, CONFIG)

    assert cluster.bbox_in_original == BoundingBox(100, 100, 140, 120)
    assert cluster.bbox_in_suspect == BoundingBox(70, 88, 110, 108)


def test_a_cluster_takes_its_worst_members_severity():
    candidates = [
        _candidate(0, 0, 40, 20, DiffType.COLOUR, DiffSubtype.CHANGED, magnitude=3.0),  # minor
        _candidate(50, 0, 90, 20, DiffType.COLOUR, DiffSubtype.CHANGED, magnitude=30.0),  # critical
    ]
    (cluster,) = build_clusters(candidates, IDENTITY, CONFIG)
    assert cluster.severity is Severity.CRITICAL
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


def test_a_multi_member_cluster_summarises_itself():
    words = [_candidate(x, 0, x + 40, 20, subtype=DiffSubtype.MOVED) for x in (0, 50, 100)]
    (cluster,) = build_clusters(words, IDENTITY, CONFIG)
    assert "3 text differences" in cluster.detail


def test_a_custom_rule_table_is_honoured():
    lenient = (SeverityRule(Severity.MINOR),)
    (cluster,) = build_clusters([_candidate(0, 0, 40, 20)], IDENTITY, CONFIG, rules=lenient)
    assert cluster.severity is Severity.MINOR


def test_clustering_distance_comes_from_config():
    pair = [_candidate(0, 0, 40, 20), _candidate(100, 0, 140, 20)]  # 60 px apart
    assert len(build_clusters(pair, IDENTITY, ClusteringConfig(distance_px=30))) == 2
    assert len(build_clusters(pair, IDENTITY, ClusteringConfig(distance_px=80))) == 1


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
