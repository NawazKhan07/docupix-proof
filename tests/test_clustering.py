"""Each difference is a finding of its own; two are joined only where they are really one thing: the bars of a barcode,
the letters and words of one line of type, the same place found twice (see `app.engine.aggregation.clustering`)."""

from __future__ import annotations

import time

from app.engine.aggregation.clustering import box_gap, cluster_candidates, union_box
from app.models.diff import BoundingBox, DiffCandidate, DiffSubtype, DiffType


def _candidate(x0, y0, x1, y1, diff_type=DiffType.TEXT, subtype=DiffSubtype.CHANGED, magnitude=0.0) -> DiffCandidate:
    return DiffCandidate(
        bbox=BoundingBox(x0, y0, x1, y1),
        type=diff_type,
        subtype=subtype,
        confidence=1.0,
        detail=f"{diff_type.value} at {x0},{y0}",
        magnitude=magnitude,
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


# ---- type: the words of one line ---------------------------------------------------------------------------------------------


def test_the_words_of_one_line_are_one_finding():
    """FR-7.3: every word of a line that changed is one finding, not N."""
    words = [_candidate(x, 100, x + 40, 120) for x in (0, 50, 100, 150, 200)]
    groups = cluster_candidates(words, distance_px=30)
    assert len(groups) == 1 and len(groups[0]) == 5


def test_words_further_apart_than_a_word_or_so_are_separate_findings():
    groups = cluster_candidates([_candidate(0, 0, 40, 20), _candidate(200, 0, 240, 20)], distance_px=30)
    assert len(groups) == 2


def test_lines_are_not_joined_to_each_other_however_near():
    lines = [_candidate(0, 0, 200, 20), _candidate(0, 24, 200, 44), _candidate(0, 48, 200, 68)]
    assert len(cluster_candidates(lines, distance_px=30)) == 3


def test_a_line_of_big_letters_is_one_finding_too():
    letters = [_candidate(x, 843, x + 90, 958) for x in (0, 100, 200, 300)]  # a heading: 115 px tall, 10 px between letters
    assert len(cluster_candidates(letters, distance_px=30)) == 1


def test_letters_are_joined_along_a_line_only_up_to_the_gap_limit():
    letters = [_candidate(0, 0, 90, 300), _candidate(400, 0, 490, 300)]  # 310 px apart: not one word, whatever their height
    assert len(cluster_candidates(letters, distance_px=30)) == 2


def test_distant_differences_stay_separate():
    groups = cluster_candidates([_candidate(0, 0, 40, 20), _candidate(1000, 1000, 1040, 1020)], distance_px=30)
    assert len(groups) == 2


def test_a_chain_of_near_differences_down_a_page_is_not_one_finding():
    """Nothing is joined because it is merely near another (a chain of single linkage covered a third of a carton)."""
    chain = [_candidate(0, y, 40, y + 20) for y in range(0, 600, 40)]  # twenty pixels apart, one below another
    assert len(cluster_candidates(chain, distance_px=30)) == len(chain)


# ---- barcodes: by distance -----------------------------------------------------------------------------------------------------


def test_the_bars_of_a_barcode_are_one_finding_by_distance():
    bars = [_candidate(x, 0, x + 6, 100, DiffType.BARCODE) for x in (0, 16, 32, 48)]
    assert len(cluster_candidates(bars, distance_px=30)) == 1
    assert len(cluster_candidates(bars, distance_px=5)) == 4


def test_the_dots_of_braille_are_joined_by_distance_too_but_not_to_a_barcode():
    dots = [_candidate(x, 0, x + 8, 8, DiffType.BRAILLE) for x in (0, 20, 40)]
    assert len(cluster_candidates(dots, distance_px=30)) == 1
    mixed = [_candidate(0, 0, 8, 8, DiffType.BRAILLE), _candidate(12, 0, 20, 8, DiffType.BARCODE)]
    assert len(cluster_candidates(mixed, distance_px=30)) == 2


# ---- the same place, found twice ---------------------------------------------------------------------------------------------


def test_different_types_that_cover_the_same_place_are_one_finding():
    """A new logo is both a colour and a graphics difference: report it once."""
    same = [
        _candidate(0, 0, 100, 100, DiffType.COLOUR),
        _candidate(10, 10, 90, 90, DiffType.GRAPHICS),
    ]
    (group,) = cluster_candidates(same, distance_px=30)
    assert {c.type for c in group} == {DiffType.COLOUR, DiffType.GRAPHICS}


def test_different_types_that_are_merely_near_each_other_stay_separate():
    near = [
        _candidate(0, 0, 100, 100, DiffType.COLOUR),
        _candidate(110, 0, 200, 100, DiffType.GRAPHICS),  # 10 px away
    ]
    assert len(cluster_candidates(near, distance_px=30)) == 2


def test_a_barely_overlapping_pair_of_different_types_is_not_merged():
    sliver = [
        _candidate(0, 0, 100, 100, DiffType.COLOUR),
        _candidate(95, 0, 195, 100, DiffType.GRAPHICS),
    ]
    assert len(cluster_candidates(sliver, distance_px=30)) == 2


def test_a_small_difference_inside_a_big_one_is_a_finding_of_its_own():
    """A digit changed in a recoloured panel is the digit's finding, not part of the panel's."""
    nested = [
        _candidate(0, 0, 400, 300, DiffType.COLOUR),
        _candidate(40, 10, 70, 30, DiffType.GRAPHICS),  # fully inside, a small share of its area
    ]
    assert len(cluster_candidates(nested, distance_px=30)) == 2


def test_a_box_inside_one_only_a_little_bigger_is_that_box_again():
    same_dot = [_candidate(890, 981, 917, 1007, DiffType.GRAPHICS), _candidate(895, 985, 913, 1003, DiffType.COLOUR)]
    assert len(cluster_candidates(same_dot, distance_px=30)) == 1


def test_a_huge_outline_artefact_does_not_swallow_the_findings_it_touches():
    ring = _candidate(0, 0, 3000, 3000, DiffType.GRAPHICS)
    inside = [_candidate(100, 100, 140, 120, DiffType.COLOUR), _candidate(2000, 2000, 2040, 2020, DiffType.COLOUR)]
    assert len(cluster_candidates([ring, *inside], distance_px=30)) == 3


def test_a_difference_cut_in_two_along_a_tile_edge_is_one_finding():
    halves = [_candidate(0, 0, 60, 128, DiffType.GRAPHICS), _candidate(0, 128, 60, 260, DiffType.GRAPHICS)]
    assert len(cluster_candidates(halves, distance_px=30)) == 1
    corner = [_candidate(0, 0, 60, 60, DiffType.GRAPHICS), _candidate(60, 60, 120, 120, DiffType.GRAPHICS)]
    assert len(cluster_candidates(corner, distance_px=30)) == 2  # meeting at a corner is not one thing


# ---- moved ---------------------------------------------------------------------------------------------------------------------------


def test_words_that_moved_alike_are_one_move_and_words_that_moved_differently_are_not():
    alike = [_candidate(0, 0, 40, 20, subtype=DiffSubtype.MOVED, magnitude=50), _candidate(50, 0, 90, 20, subtype=DiffSubtype.MOVED, magnitude=52)]
    assert len(cluster_candidates(alike, distance_px=30)) == 1
    different = [_candidate(0, 0, 40, 20, subtype=DiffSubtype.MOVED, magnitude=50), _candidate(50, 0, 90, 20, subtype=DiffSubtype.MOVED, magnitude=200)]
    assert len(cluster_candidates(different, distance_px=30)) == 2


def test_a_moved_word_does_not_join_a_changed_word_beside_it():
    beside = [_candidate(0, 0, 40, 20, subtype=DiffSubtype.MOVED, magnitude=50), _candidate(50, 0, 90, 20, subtype=DiffSubtype.CHANGED)]
    assert len(cluster_candidates(beside, distance_px=30)) == 2


# ---- order and speed -----------------------------------------------------------------------------------------------------------


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
