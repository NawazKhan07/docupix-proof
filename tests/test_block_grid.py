"""Dividing an artwork into its panels by its fold and cut lines (`block_grid`)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz

from app.engine.isolation.block_grid import divide_into_blocks, stroked_line_map
from tests.helpers import make_multi_panel_carton_pdf

DPI = 150


def _render(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """The page's pixels and its stroked-line map, at the same resolution."""
    with fitz.open(path) as doc:
        page = doc[0]
        pixmap = page.get_pixmap(dpi=DPI)
        image = np.frombuffer(pixmap.samples, np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)[:, :, :3].copy()
        return image, stroked_line_map(page, DPI, image.shape[:2])


def test_a_pdf_carton_is_divided_at_its_vector_fold_line(tmp_path: Path):
    image, lines = _render(make_multi_panel_carton_pdf(tmp_path / "carton.pdf"))
    assert lines is not None and lines.any()

    blocks = divide_into_blocks(image, line_map=lines)
    assert len(blocks) == 2
    fold = 200 * DPI / 72
    left, right = sorted(blocks, key=lambda block: block.bbox.x0)
    assert abs(left.bbox.x1 - fold) < 6 and abs(right.bbox.x0 - fold) < 6


def test_a_filled_bar_across_the_panels_is_not_a_fold(tmp_path: Path):
    """Only strokes are fold lines: a panel-wide colour bar (a fill) or a line of text never divides a block."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    page.draw_rect(fitz.Rect(20, 20, 380, 280), color=(0, 0, 0), width=1.5)
    page.draw_line(fitz.Point(200, 20), fitz.Point(200, 280), color=(0, 0, 0), width=1.5)
    page.draw_rect(fitz.Rect(20, 140, 380, 160), color=None, fill=(0.1, 0.3, 0.7))
    page.insert_text((35, 80), "PARACETAMOL 500 mg TABLETS AND MORE TEXT ACROSS", fontsize=10)
    page.insert_text((220, 220), "BATCH INFO", fontsize=10)
    path = tmp_path / "bar.pdf"
    doc.save(path)
    doc.close()

    image, lines = _render(path)
    blocks = divide_into_blocks(image, line_map=lines)
    assert len(blocks) == 2  # the fold alone: not split again along the bar


def _three_panel_raster() -> np.ndarray:
    image = np.full((700, 900, 3), 255, np.uint8)
    cv2.rectangle(image, (20, 20), (880, 680), (0, 0, 0), 3)
    for x in (300, 600):
        cv2.line(image, (x, 20), (x, 680), (0, 0, 0), 3)
    for x0 in (35, 315, 615):
        cv2.putText(image, "PANEL TEXT LINE", (x0, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)
        cv2.circle(image, (x0 + 100, 450), 25, (200, 30, 40), -1)
    return image


def test_a_raster_carton_is_divided_into_its_panels_without_the_empty_margin():
    """With no vector paths (a scan, an image), the panels closed by lines are the blocks -- and the blank
    strips between the page edge and the outer border line are not panels, having nothing printed in them."""
    blocks = divide_into_blocks(_three_panel_raster())
    assert len(blocks) == 3
    assert all(block.bbox.height > 600 for block in blocks)


def test_a_box_drawn_round_a_logo_does_not_divide_a_flat_label():
    image = np.full((500, 700, 3), 255, np.uint8)
    cv2.putText(image, "ONE PANEL LABEL", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 10, 10), 2)
    cv2.rectangle(image, (300, 250), (380, 320), (0, 0, 0), 2)
    assert divide_into_blocks(image) == []  # one block would be the whole artwork: nothing to divide


def test_blocks_stay_inside_the_artwork_region():
    image = _three_panel_raster()
    region = np.zeros(image.shape[:2], bool)
    region[20:681, 20:881] = True
    for block in divide_into_blocks(image, region=region):
        assert block.bbox.x0 >= 20 and block.bbox.x1 <= 881 and block.bbox.y0 >= 20 and block.bbox.y1 <= 681


def test_a_fold_and_a_box_just_below_it_are_not_averaged_into_a_cut_on_neither():
    """A full-width fold and the edge of a box drawn 58 px below it (a warning panel): two different lines, never
    merged into one cut half-way between them where there is no line at all -- which then failed its own check for
    a line along it and lost the fold (Met XL: the bottom panel started at a box edge, not at its fold)."""
    from app.engine.isolation.block_grid import _cluster

    fold = {3755: 1464, 3756: 1464, 3757: 1464}
    box = {3814: 695, 3815: 697, 3816: 697}
    cuts = _cluster([], tolerance=66, exact={**fold, **box})
    assert 3755 in cuts or 3756 in cuts or 3757 in cuts  # on the fold itself
    assert not any(3770 < cut < 3805 for cut in cuts)  # nothing in between


def test_a_box_drawn_across_part_of_a_panel_does_not_split_it(tmp_path: Path):
    """A PDF's folds are solid strokes along the whole edge between two panels; a box drawn inside one panel covers
    only part of any line through it, and is not a fold."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    page.draw_rect(fitz.Rect(20, 20, 380, 280), color=(0, 0, 0), width=1.5)
    page.draw_line(fitz.Point(200, 20), fitz.Point(200, 280), color=(0, 0, 0), width=1.5)
    page.draw_rect(fitz.Rect(40, 150, 140, 190), color=(0.8, 0.1, 0.1), width=1.2)  # a box across part of the left panel
    page.insert_text((35, 80), "PARACETAMOL 500 mg", fontsize=10)
    page.insert_text((220, 80), "BATCH INFO", fontsize=10)
    path = tmp_path / "box.pdf"
    doc.save(path)
    doc.close()

    image, lines = _render(path)
    assert len(divide_into_blocks(image, line_map=lines)) == 2
