from __future__ import annotations

import cv2
import pymupdf as fitz
import numpy as np
import pytest

from app.engine.isolation.polygon_tracer import (
    NoDielineFoundError,
    _rect_edge_coverage,
    find_blocks,
    trace_raster_dieline,
    trace_vector_dieline,
)


def _pdf_page_with_rect_dieline():
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    page.draw_rect(fitz.Rect(20, 20, 380, 280), color=(0, 0, 0), width=2)
    return doc, page


def test_trace_vector_dieline_returns_closed_rectangle_loop():
    doc, page = _pdf_page_with_rect_dieline()
    try:
        traced = trace_vector_dieline(page, click_point=(20, 150))  # on the left edge
        assert len(traced.points) == 4
        xs = [p[0] for p in traced.points]
        ys = [p[1] for p in traced.points]
        assert min(xs) == pytest.approx(20, abs=0.5)
        assert max(xs) == pytest.approx(380, abs=0.5)
        assert min(ys) == pytest.approx(20, abs=0.5)
        assert max(ys) == pytest.approx(280, abs=0.5)
    finally:
        doc.close()


def test_trace_vector_dieline_works_from_any_edge():
    """The click can land near any edge of the loop, not just the one it starts from."""
    doc, page = _pdf_page_with_rect_dieline()
    try:
        for click in [(200, 20), (380, 150), (200, 280)]:
            traced = trace_vector_dieline(page, click_point=click)
            assert len(traced.points) == 4
    finally:
        doc.close()


def test_trace_vector_dieline_raises_when_no_vector_paths():
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    try:
        with pytest.raises(NoDielineFoundError):
            trace_vector_dieline(page, click_point=(20, 150))
    finally:
        doc.close()


def test_trace_vector_dieline_open_path_is_rejected():
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    shape = page.new_shape()
    shape.draw_line(fitz.Point(20, 20), fitz.Point(200, 20))
    shape.finish(color=(0, 0, 0), width=1)
    shape.commit()
    try:
        with pytest.raises(NoDielineFoundError):
            trace_vector_dieline(page, click_point=(100, 20))
    finally:
        doc.close()


def test_trace_vector_dieline_handles_curved_corners():
    """A rounded-corner dieline (line segments + bezier curves) still closes."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    shape = page.new_shape()
    shape.draw_line(fitz.Point(40, 20), fitz.Point(360, 20))
    shape.draw_bezier(
        fitz.Point(360, 20), fitz.Point(380, 20), fitz.Point(380, 40), fitz.Point(380, 60)
    )
    shape.draw_line(fitz.Point(380, 60), fitz.Point(380, 280))
    shape.draw_line(fitz.Point(380, 280), fitz.Point(20, 280))
    shape.draw_line(fitz.Point(20, 280), fitz.Point(20, 40))
    shape.draw_bezier(
        fitz.Point(20, 40), fitz.Point(20, 20), fitz.Point(20, 20), fitz.Point(40, 20)
    )
    shape.finish(color=(0, 0, 0), width=1)
    shape.commit()
    try:
        traced = trace_vector_dieline(page, click_point=(200, 20))
        assert len(traced.points) >= 6  # straight edges + sampled curve points
    finally:
        doc.close()


def _raster_rect_image(width: int = 300, height: int = 300) -> np.ndarray:
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (60, 80), (240, 220), (0, 0, 0), 3)  # stroked, not filled
    return image


def test_trace_raster_dieline_finds_contour_near_click():
    image = _raster_rect_image()
    traced = trace_raster_dieline(image, click_point=(60, 150))  # on the left edge
    xs = [p[0] for p in traced.points]
    ys = [p[1] for p in traced.points]
    assert min(xs) <= 62 and max(xs) >= 238
    assert min(ys) <= 82 and max(ys) >= 218


def test_trace_raster_dieline_raises_when_click_far_from_any_contour():
    image = _raster_rect_image()
    with pytest.raises(NoDielineFoundError):
        trace_raster_dieline(image, click_point=(5, 5))


def test_trace_raster_dieline_raises_on_blank_image():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    with pytest.raises(NoDielineFoundError):
        trace_raster_dieline(image, click_point=(50, 50))


# ---------------------------------------------------------------------------
# find_blocks: every interior panel, not just one clicked loop
# ---------------------------------------------------------------------------


def _three_panel_carton(width: int = 900, height: int = 700) -> np.ndarray:
    """An outer boundary with two interior fold lines splitting it into three panels, plus a speck of noise
    (a mark far too small to be a panel) -- close to what a real carton's dieline looks like."""
    image = np.full((height, width, 3), 255, np.uint8)
    cv2.rectangle(image, (20, 20), (880, 680), (0, 0, 0), 3)
    for x in (300, 600):
        cv2.line(image, (x, 20), (x, 680), (0, 0, 0), 3)
    cv2.rectangle(image, (100, 100), (110, 110), (0, 0, 0), 1)
    return image


def test_find_blocks_finds_the_interior_panels_not_the_outer_boundary():
    blocks = find_blocks(_three_panel_carton())
    assert len(blocks) == 3
    widths = sorted(b.bbox.width for b in blocks)
    assert all(w == pytest.approx(widths[0], rel=0.1) for w in widths)  # three roughly equal panels
    assert all(b.bbox.height > 600 for b in blocks)  # each spans the full inner height


def test_find_blocks_ignores_a_speck_of_noise():
    image_area = _three_panel_carton().shape[0] * _three_panel_carton().shape[1]
    blocks = find_blocks(_three_panel_carton())
    assert all(b.bbox.area > 0.02 * image_area for b in blocks)


def test_find_blocks_returns_nothing_for_a_single_panel_artwork():
    """Just the outer boundary, no interior fold lines: there is no panel to find inside it."""
    image = np.full((300, 400, 3), 255, np.uint8)
    cv2.rectangle(image, (20, 20), (380, 280), (0, 0, 0), 3)
    assert find_blocks(image) == []


def _panel_with_artwork_across_its_edge(width: int = 900, height: int = 700) -> np.ndarray:
    """Like `_three_panel_carton`, but a filled illustration sits right across the fold line between the first
    two panels, the way a logo or a graphic is often printed right up to (or over) a panel's own edge -- large
    enough that the traced contour it corrupts is no longer close enough to a rectangle on its own."""
    image = _three_panel_carton(width, height)
    size, cx, cy = 160, 300, 400
    blob = np.array(
        [
            [cx - size, cy - size * 3 // 10],
            [cx, cy - size * 6 // 10],
            [cx + size, cy - size * 3 // 10],
            [cx + size // 2, cy + size],
            [cx - size // 2, cy + size * 9 // 10],
        ],
        np.int32,
    )
    cv2.fillPoly(image, [blob], (0, 0, 0))
    return image


def test_find_blocks_keeps_a_panel_whose_own_edge_has_artwork_drawn_across_it():
    """Artwork merged into a panel's fold line in the edge map corrupts the traced contour's shape (it is no
    longer close to a clean rectangle), but the panel is still there, by its straight sides -- it must not be
    lost the way a genuine decorative graphic correctly is (the next test)."""
    blocks = find_blocks(_panel_with_artwork_across_its_edge())
    assert len(blocks) == 3
    widths = sorted(b.bbox.width for b in blocks)
    assert all(w == pytest.approx(widths[0], rel=0.15) for w in widths)


def test_rect_edge_coverage_is_high_for_a_rectangle_traced_in_the_edge_map():
    edges = np.zeros((200, 300), np.uint8)
    cv2.rectangle(edges, (20, 20), (280, 180), 255, 2)
    rect = np.array([[20, 20], [280, 20], [280, 180], [20, 180]], dtype=np.float32)
    assert _rect_edge_coverage(rect, edges) > 0.95


def test_rect_edge_coverage_is_low_for_a_curve_that_does_not_hug_the_rectangle():
    """A decorative curve's own bounding rectangle must not be rescued by the edge-coverage fallback the way a
    real panel corrupted by artwork touching it is (the previous test): its edge does not run along the
    rectangle's straight sides at all."""
    edges = np.zeros((200, 300), np.uint8)
    cv2.ellipse(edges, (150, 100), (100, 70), 0, 0, 360, 255, 2)
    rect = np.array([[20, 20], [280, 20], [280, 180], [20, 180]], dtype=np.float32)
    assert _rect_edge_coverage(rect, edges) < 0.5


def test_find_blocks_rejects_a_non_rectangular_region():
    """A panel is bounded by straight fold lines: a big enclosed shape that is not rectangle-ish is not one."""
    image = np.full((700, 900, 3), 255, np.uint8)
    cv2.rectangle(image, (20, 20), (880, 680), (0, 0, 0), 3)
    triangle = np.array([[100, 650], [780, 650], [440, 60]], np.int32)
    cv2.polylines(image, [triangle], isClosed=True, color=(0, 0, 0), thickness=3)
    assert find_blocks(image) == []


def test_find_blocks_respects_a_mask():
    """An unrelated shape outside the mask must not be picked up as one of the artwork's own panels."""
    image = np.full((700, 1200, 3), 255, np.uint8)
    cv2.rectangle(image, (20, 20), (880, 680), (0, 0, 0), 3)
    for x in (300, 600):
        cv2.line(image, (x, 20), (x, 680), (0, 0, 0), 3)
    cv2.rectangle(image, (950, 100), (1150, 600), (0, 0, 0), 3)
    cv2.line(image, (1050, 100), (1050, 600), (0, 0, 0), 3)

    mask = np.zeros((700, 1200), np.uint8)
    mask[:, :900] = 255

    blocks = find_blocks(image, mask=mask)
    assert len(blocks) == 3
    assert all(b.bbox.x1 <= 900 for b in blocks)


def test_find_blocks_does_not_double_count_a_panel_from_its_own_dilated_edge():
    """The inner and outer edge of one dilated line must not be counted as two panels."""
    blocks = find_blocks(_three_panel_carton())
    for i, a in enumerate(blocks):
        for b in blocks[i + 1 :]:
            ix0, iy0 = max(a.bbox.x0, b.bbox.x0), max(a.bbox.y0, b.bbox.y0)
            ix1, iy1 = min(a.bbox.x1, b.bbox.x1), min(a.bbox.y1, b.bbox.y1)
            overlap = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            assert overlap / min(a.bbox.area, b.bbox.area) < 0.9
