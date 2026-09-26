"""Automatic mode finds the artwork on a sheet without being told where it is (`app.engine.isolation.auto_area`).

These exercise `find_artwork` end to end on synthetic sheets (a die-cut border, some printing inside it,
sometimes a second closed shape beside it -- a table of details) rather than mocking the border-lifting or
feature-matching it is built from, since what matters is the decisions made from their combination.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.config import SubjectConfig
from app.engine.isolation.auto_area import _compactness, find_artwork

CONFIG = SubjectConfig()


def _bordered_page(width: int = 900, height: int = 700, border: tuple[int, int, int, int] = (60, 60, 840, 640)) -> np.ndarray:
    """A light page with a closed rectangular dieline and enough printing inside it for ORB to work on."""
    image = np.full((height, width, 3), 255, np.uint8)
    x0, y0, x1, y1 = border
    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 0), 4)
    cv2.putText(image, "PARACETAMOL 500mg", (x0 + 20, y0 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
    cv2.putText(image, "Tablets BP x16", (x0 + 20, y0 + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)
    cv2.rectangle(image, (x0 + 30, y0 + 130), (x0 + 220, y0 + 240), (200, 30, 40), -1)
    cv2.circle(image, (x1 - 120, y0 + 180), 60, (30, 90, 200), -1)
    for i in range(14):
        xx = x0 + 40 + i * 18
        cv2.line(image, (xx, y1 - 120), (xx, y1 - 50), (0, 0, 0), 3)
    return image


def _page_with_table(table_seed: int) -> np.ndarray:
    """`_bordered_page` beside a second, notched closed shape (a table of details) whose own content varies
    with `table_seed` -- big and closed enough to be a candidate in its own right, not the artwork."""
    image = np.full((700, 1300, 3), 255, np.uint8)
    image[:, :900] = _bordered_page(900, 700)
    notch = np.array([[980, 150], [1250, 150], [1250, 470], [1180, 470], [1180, 550], [980, 550]], np.int32)
    cv2.polylines(image, [notch], True, (0, 0, 0), 2)
    rng = np.random.default_rng(table_seed)
    for i in range(6):
        y = 175 + i * 40
        x_end = 1230 - int(rng.integers(0, 150))
        cv2.line(image, (1000, y), (x_end, y), (0, 0, 0), 2)
    return image


# ---- either file has no border at all: nothing is trimmed to -------------------------------------------------


def test_find_artwork_returns_neither_geometry_when_one_file_has_no_border():
    blank = np.full((700, 900, 3), 255, np.uint8)
    result = find_artwork(_bordered_page(), blank, CONFIG, None, None, None, None)
    assert result.original is None and result.suspect is None and result.note == ""


# ---- exactly one border per file -------------------------------------------------------------------------------


def test_find_artwork_accepts_a_single_pair_of_similar_shape_and_share():
    original = _bordered_page()
    suspect = _bordered_page(920, 720, (65, 65, 855, 655))  # a few pixels bigger, same proportions

    result = find_artwork(original, suspect, CONFIG, None, None, None, None)

    assert result.original is not None and result.suspect is not None
    assert "found automatically" in result.note


def test_find_artwork_gives_up_on_a_single_pair_of_different_shape_and_content():
    """Genuinely different artwork (a different aspect ratio, nothing in common to match on) is not forced
    together: the whole pages are compared instead."""
    original = _bordered_page()
    suspect = np.full((700, 900, 3), 255, np.uint8)
    cv2.rectangle(suspect, (350, 40), (860, 660), (0, 0, 0), 4)  # a tall, narrow, blank border elsewhere on the page

    result = find_artwork(original, suspect, CONFIG, None, None, None, None)

    assert result.original is None and result.suspect is None


def test_find_artwork_accepts_a_single_pair_whose_page_share_differs_but_printing_corresponds():
    """The same artwork, one file with far more blank margin round its border than the other (or printed a
    little smaller or larger on its own sheet altogether): their area shares disagree by more than the coarse
    gate alone allows, but the printing inside them still matches, which is stronger evidence."""
    tight = _bordered_page(980, 760, (60, 60, 920, 700))  # the border fills most of a snug page
    wide = np.full((1600, 2000, 3), 255, np.uint8)
    wide[100:800, 100:1000] = _bordered_page()  # the same artwork, a small share of a much bigger page

    result = find_artwork(tight, wide, CONFIG, None, None, None, None)

    assert result.original is not None and result.suspect is not None


def test_find_artwork_still_gives_up_when_the_page_share_differs_and_nothing_corresponds():
    """The fallback above is for when the printing matches despite the shares disagreeing -- it must not paper
    over two pages whose borders neither look alike nor share any printing."""
    tight = _bordered_page(980, 760, (60, 60, 920, 700))
    wide = np.full((1600, 2000, 3), 255, np.uint8)
    blank_border = wide.copy()
    cv2.rectangle(blank_border, (100, 100), (1000, 800), (0, 0, 0), 4)  # closed, but nothing printed inside it

    result = find_artwork(tight, blank_border, CONFIG, None, None, None, None)

    assert result.original is None and result.suspect is None


# ---- several closed borders on a sheet: the pair whose printing matches is the artwork --------------------------


def test_find_artwork_picks_the_carton_not_the_table_of_details():
    original = _page_with_table(table_seed=1)
    suspect = _page_with_table(table_seed=99)  # the table's own content differs; the carton's does not

    result = find_artwork(original, suspect, CONFIG, None, None, None, None)

    assert result.original is not None
    xs = [x for x, _y in result.original.boundary.points]
    assert max(xs) < 900 / 1300  # the carton's own share of the page, not the table's (beyond x=980/1300)


def test_find_artwork_gives_up_when_no_pair_of_several_borders_corresponds():
    """Two sheets that each have a carton and a table, but nothing in common between either pair: forcing the
    best-scoring (still weak) match together would compare the wrong thing."""
    original = _page_with_table(table_seed=1)
    suspect = np.full((700, 1300, 3), 255, np.uint8)
    cv2.rectangle(suspect, (40, 40), (860, 660), (0, 0, 0), 4)
    cv2.rectangle(suspect, (980, 150), (1250, 550), (0, 0, 0), 2)

    result = find_artwork(original, suspect, CONFIG, None, None, None, None)

    assert result.original is None and result.suspect is None


# ---- a scan's own dark-line detection can trace a paragraph of text as a false "border" -------------------------


def test_compactness_is_low_for_a_real_die_cut_outline():
    """A real outline's perimeter grows with its size whether or not it has tabs and notches: surveyed across
    ten real cartons' own outlines (a simple box to one with 172 corners), every one measured 17.7-32.6."""
    square = [(0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0)]
    assert _compactness(square) < 20.0  # a perfect square: the least jagged a closed shape can be


def test_compactness_is_high_for_a_paragraph_of_text_merged_into_one_shape():
    """A scan's dark-line detection, bridging small gaps between letters, can trace a dense paragraph of text
    as its own closed 'border' once enough of it happens to enclose -- a real false positive that measured
    87.9, well past any real outline surveyed."""
    rng = np.random.default_rng(0)
    jagged: list[tuple[float, float]] = []
    for i in range(200):
        angle = 2 * np.pi * i / 200
        radius = 80.0 + rng.uniform(-40.0, 40.0)  # a ragged, letter-by-letter boundary, not a smooth one
        jagged.append((100.0 + radius * np.cos(angle), 100.0 + radius * np.sin(angle)))
    assert _compactness(jagged) > 50.0


# ---- only one file has a border: it is laid over the other where the printing matches -------------------------


def _borderless(page: np.ndarray, border: tuple[int, int, int, int] = (60, 60, 840, 640), shift: tuple[int, int] = (25, 15)) -> np.ndarray:
    """`page` with its dieline painted out and the artwork moved a little -- a scan whose border is too faint to trace."""
    image = page.copy()
    x0, y0, x1, y1 = border
    cv2.rectangle(image, (x0, y0), (x1, y1), (255, 255, 255), 8)
    matrix = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderValue=(255, 255, 255))


def _box_of(geometry, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    height, width = shape
    xs = [x * width for x, _y in geometry.boundary.points]
    ys = [y * height for _x, y in geometry.boundary.points]
    return min(xs), min(ys), max(xs), max(ys)


def _printed_page() -> np.ndarray:
    """`_bordered_page` with more of what real artwork has: varied lines of text, so its printing can be matched."""
    image = _bordered_page()
    lines = ("Each tablet contains 500 mg", "Store below 25 C, dry place", "Keep out of reach of children",
             "Mfg. Lic. No. 16/UA/2006", "Batch B2291  Exp 07/2027")
    for i, line in enumerate(lines):
        cv2.putText(image, line, (330, 250 + i * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    return image


def test_a_border_found_in_only_one_file_is_placed_on_the_other_where_the_printing_matches():
    original = _printed_page()
    suspect = _borderless(original)

    result = find_artwork(original, suspect, CONFIG, None, None, None, None)

    assert result.original is not None and result.suspect is not None
    assert result.suspect.boundary_origin == "placed"
    x0, y0, x1, y1 = _box_of(result.suspect, suspect.shape[:2])
    assert abs(x0 - 85) < 8 and abs(y0 - 75) < 8 and abs(x1 - 865) < 8 and abs(y1 - 655) < 8  # the border, moved with the artwork
    assert "laid over the other" in result.note


def test_a_small_box_round_part_of_the_printing_is_not_taken_for_the_artwork():
    """A proof-reader's box drawn round one paragraph of a scan shares every feature inside it with the whole carton
    on the other file -- but it is a piece of the artwork, not the artwork, and trimming to it would compare only it."""
    original = _printed_page()
    suspect = _borderless(original, shift=(0, 0))
    cv2.rectangle(suspect, (70, 80), (420, 180), (30, 30, 200), 3)  # round the product name only

    result = find_artwork(original, suspect, CONFIG, None, None, None, None)

    assert result.suspect is not None
    x0, y0, x1, y1 = _box_of(result.suspect, suspect.shape[:2])
    assert (x1 - x0) > 600 and (y1 - y0) > 450  # the whole artwork, not the 350 x 100 box
