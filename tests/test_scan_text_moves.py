"""Scan vs Digital text on a re-arranged carton: a line that moved is one finding boxed where it was and where it is,
readings that share nothing are not one 'changed' into the other, and no line runs across two panels."""

from __future__ import annotations

import numpy as np

from app.config import ClusteringConfig
from app.engine.aggregation.aggregator import build_clusters, merge_clusters
from app.engine.detectors.text_diff import OcrWord, _findings, _moved_findings, _moved_lines, _split_at_gaps
from app.models.diff import BoundingBox, DiffSubtype


def _line(text: str, x: float, y: float, height: float = 20, width_per_char: float = 11, conf: float = 90) -> list[OcrWord]:
    words, cursor = [], x
    for token in text.split():
        width = width_per_char * len(token)
        words.append(OcrWord(token, BoundingBox(cursor, y, cursor + width, y + height), conf))
        cursor += width + 8
    return words


def test_a_row_is_split_where_a_panels_worth_of_paper_runs_through_it():
    left = _line("Lipid Complex Injection", 100, 500)
    right = _line("Thanh phan: Moi ml chua", 700, 502)
    parts = _split_at_gaps(left + right)
    assert [[w.text for w in part] for part in parts] == [[w.text for w in left], [w.text for w in right]]
    assert len(_split_at_gaps(_line("Store below 25 C", 100, 500))) == 1  # the gaps between words are not panels


def test_readings_that_share_nothing_are_what_is_missing_and_what_is_there_not_a_change():
    gone = _line("Thanh phan:", 100, 500)
    junk = _line("eno", 110, 540, height=120)  # (a code on its side, read upright, where the line was)
    found = _findings(gone, junk)
    assert {c.subtype for c in found} == {DiffSubtype.REMOVED, DiffSubtype.ADDED} and len(found) == 2
    assert all(c.bbox.height < 130 for c in found)  # each boxed where it is: no box round both
    changed = _findings(_line("Store below 25 C", 100, 500), _line("Store below 30 C", 100, 500))
    assert [c.subtype for c in changed] == [DiffSubtype.CHANGED]  # alike: a change
    number = _findings(_line("1g", 100, 500), _line("28mg", 100, 500))
    assert [c.subtype for c in number] == [DiffSubtype.CHANGED]  # a number for a number: a change


def _page(lines):
    """A reading of a page, as `read_pages` gives it: (line key, word), one key per line."""
    return [((number,), word) for number, words in enumerate(lines) for word in words]


def test_a_line_moved_to_another_panel_is_found_there_and_nothing_else_is_mistaken_for_it():
    registration = _line("So dang ky: VN-19392-15", 1900, 1450)
    brand = _line("AMPHOLIP", 900, 1400)  # a word printed on every panel: never looked for elsewhere
    address_a, address_b = _line("K-27 Jambivili Village Anand Nagar", 470, 2090), _line("K-27 Jambivili Village Anand Nagar", 1480, 2090)
    anchors = [registration, brand, address_a, address_b]
    flat = [w for line in anchors for w in line]
    owner = [n for n, line in enumerate(anchors) for _w in line]

    moved_there = _line("So dang ky: VN-19392-15", 400, 2400)  # the scan: on the bottom flap
    brand_elsewhere = _line("AMPHOLIP", 1500, 1400)
    address_reflowed = _line("K-27 Jambivili Village Anand Nagar", 1490, 2092)  # one copy, re-flowed a little
    scan_lines = [moved_there, brand_elsewhere, address_reflowed]
    scan_words = [w for line in scan_lines for w in line]
    missing = set(range(len(flat)))  # nothing where it was (the worst case)

    moved, rows = _moved_lines(anchors, [0, 1, 2, 3], missing, owner, _page(scan_lines), scan_words, [])

    assert set(moved) == {0}  # the registration line: long, printed once
    assert [w.text for w in moved[0]] == [w.text for w in moved_there]
    # the brand is too short to tell apart; the address is printed twice, and the scan's copy is at one of them (home)


def test_a_moved_line_is_one_finding_shown_where_it_went_and_a_change_in_it_is_still_a_change():
    was = _line("Giay phep san xuat so: KD-360", 1900, 1480)
    now = _line("Giay phep san xuat so: KD-361", 400, 2420)
    found = _moved_findings(was, now)
    assert len(found) == 1 and found[0].subtype is DiffSubtype.CHANGED  # the change is what the finding says
    assert "KD-360" in found[0].detail and "KD-361" in found[0].detail and "in text moved" in found[0].detail

    clusters = build_clusters(found, np.eye(3), ClusteringConfig())
    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.bbox_in_original.y0 == 1480 and cluster.shown_in_suspect.y0 >= 2400  # where it was; where it is
    assert merge_clusters([cluster, clusters[0]]).moved_to is not None

    only_moved = _moved_findings(was, _line("Giay phep san xuat so: KD-360", 400, 2420))
    assert [c.subtype for c in only_moved] == [DiffSubtype.MOVED] and "Text moved" in only_moved[0].detail
    misread = _moved_findings(was, _line("Giay phep san xuat so; KD-360", 400, 2420))
    assert [c.subtype for c in misread] == [DiffSubtype.MOVED]  # (a scan's misreading in it is not a change)
