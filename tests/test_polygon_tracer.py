from __future__ import annotations

import cv2
import pymupdf as fitz
import numpy as np
import pytest

from app.engine.isolation.polygon_tracer import (
    NoDielineFoundError,
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
