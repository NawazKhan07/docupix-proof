from __future__ import annotations

import time

from app.engine.aggregation.clustering import box_gap, cluster_candidates, union_box
from app.models.diff import BoundingBox, DiffCandidate, DiffSubtype, DiffType


def _candidate(x0, y0, x1, y1, diff_type=DiffType.TEXT, subtype=DiffSubtype.MOVED) -> DiffCandidate:
    return DiffCandidate(
        bbox=BoundingBox(x0, y0, x1, y1),
        type=diff_type,
        subtype=subtype,
        confidence=1.0,
        detail=f"{diff_type.value} at {x0},{y0}",
    )


def test_box_gap_is_zero_for_overlapping_and_touching_boxes():
    a = BoundingBox(0, 0, 10, 10)
    assert box_gap(a, BoundingBox(5, 5, 20, 20)) == 0.0  # overlap
    assert box_gap(a, BoundingBox(10, 0, 20, 10)) == 0.0  # touching edges


def test_box_gap_measures_edge_distance_not_centre_distance():
    long_word = BoundingBox(0, 0, 200, 20)
    short_word = BoundingBox(205, 0, 215, 20)
    # Centres are ~110 px apart, but the boxes nearly touch.
    assert box_gap(long_word, short_word) == 5.0


def test_box_gap_uses_the_diagonal_for_corner_neighbours():
    assert box_gap(BoundingBox(0, 0, 10, 10), BoundingBox(13, 14, 20, 20)) == 5.0  # 3-4-5


def test_words_of_a_moved_sentence_become_one_cluster():
    """FR-7.3: every word in a moved sentence is one finding, not N."""
    words = [_candidate(x, 100, x + 40, 120) for x in (0, 50, 100, 150, 200)]
    groups = cluster_candidates(words, distance_px=30)
    assert len(groups) == 1
    assert len(groups[0]) == 5


def test_distant_differences_stay_separate():
    groups = cluster_candidates(
        [_candidate(0, 0, 40, 20), _candidate(1000, 1000, 1040, 1020)], distance_px=30
    )
    assert len(groups) == 2


def test_clustering_is_transitive_through_a_chain():
    """A-B and B-C are near, A-C are not; single linkage still joins all three."""
    chain = [_candidate(0, 0, 40, 20), _candidate(60, 0, 100, 20), _candidate(120, 0, 160, 20)]
    assert box_gap(chain[0].bbox, chain[2].bbox) > 30
    assert len(cluster_candidates(chain, distance_px=30)) == 1


def test_different_types_are_never_merged_even_when_overlapping():
    """Same-type clustering keeps the text / colour / graphics categories meaningful."""
    overlapping = [
        _candidate(0, 0, 100, 100, DiffType.COLOUR, DiffSubtype.CHANGED),
        _candidate(10, 10, 90, 90, DiffType.GRAPHICS, DiffSubtype.CHANGED),
    ]
    assert len(cluster_candidates(overlapping, distance_px=30)) == 2


def test_the_distance_is_a_tunable_parameter():
    pair = [_candidate(0, 0, 40, 20), _candidate(100, 0, 140, 20)]  # 60 px gap
    assert len(cluster_candidates(pair, distance_px=30)) == 2
    assert len(cluster_candidates(pair, distance_px=80)) == 1


def test_groups_come_back_in_reading_order():
    groups = cluster_candidates(
        [
            _candidate(500, 500, 540, 520),
            _candidate(0, 0, 40, 20),
            _candidate(300, 0, 340, 20),
            _candidate(0, 300, 40, 320),
        ],
        distance_px=10,
    )
    starts = [(min(c.bbox.y0 for c in g), min(c.bbox.x0 for c in g)) for g in groups]
    assert starts == sorted(starts)


def test_members_within_a_cluster_are_in_reading_order():
    scrambled = [_candidate(100, 0, 140, 20), _candidate(0, 0, 40, 20), _candidate(50, 0, 90, 20)]
    (group,) = cluster_candidates(scrambled, distance_px=30)
    assert [c.bbox.x0 for c in group] == [0, 50, 100]


def test_empty_input_gives_no_clusters():
    assert cluster_candidates([], distance_px=30) == []


def test_union_box_encloses_every_member():
    box = union_box([_candidate(0, 10, 40, 20), _candidate(100, 0, 140, 30)])
    assert (box.x0, box.y0, box.x1, box.y1) == (0, 0, 140, 30)


def test_thousands_of_candidates_cluster_quickly():
    """A noisy pair can produce thousands of candidates; this must stay fast."""
    grid = [_candidate(x, y, x + 20, y + 10) for x in range(0, 2400, 60) for y in range(0, 3400, 40)]
    assert len(grid) > 3000

    started = time.perf_counter()
    groups = cluster_candidates(grid, distance_px=30)
    elapsed = time.perf_counter() - started

    assert elapsed < 2.0
    assert sum(len(g) for g in groups) == len(grid)  # nothing lost or duplicated
