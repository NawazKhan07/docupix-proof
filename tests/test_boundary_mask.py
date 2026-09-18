from __future__ import annotations

import numpy as np
import pytest

from app.engine.isolation.boundary_mask import apply_mask, boundary_to_mask, resolve_keep_mask
from app.engine.isolation.erase_mask import rasterize_erase_strokes
from app.models.isolation import IsolationGeometry, NormalizedPolygon, NormalizedRect


def test_boundary_to_mask_rect_keeps_only_that_region():
    mask = boundary_to_mask((100, 200), NormalizedRect(x0=0.25, y0=0.2, x1=0.75, y1=0.8))
    assert mask.shape == (100, 200)
    assert mask[50, 100] == True  # noqa: E712 -- center is inside
    assert mask[0, 0] == False  # noqa: E712 -- corner is outside
    assert mask.sum() == pytest.approx(100 * 0.6 * 200 * 0.5, rel=0.05)


def test_boundary_to_mask_polygon_keeps_enclosed_area():
    triangle = NormalizedPolygon(points=[(0.5, 0.1), (0.9, 0.9), (0.1, 0.9)])
    mask = boundary_to_mask((100, 100), triangle)
    assert mask[80, 50] == True  # noqa: E712 -- inside the triangle
    assert mask[10, 10] == False  # noqa: E712 -- outside


def test_apply_mask_fills_outside_with_background():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    result = apply_mask(image, mask, background=(255, 255, 255))
    assert np.all(result[0, 0] == (255, 255, 255))
    assert np.all(result[5, 5] == (0, 0, 0))


def test_rasterize_erase_strokes_paints_along_the_path():
    stroke = NormalizedPolygon(points=[(0.1, 0.5), (0.9, 0.5)])
    mask = rasterize_erase_strokes((100, 100), [stroke], brush_radius_norm=0.05)
    assert mask[50, 50] == True  # noqa: E712 -- on the stroke line
    assert mask[5, 5] == False  # noqa: E712 -- far from it


def test_resolve_keep_mask_combines_boundary_and_erase():
    geometry = IsolationGeometry(
        boundary=NormalizedRect(x0=0.0, y0=0.0, x1=1.0, y1=1.0),
        erase_regions=[NormalizedPolygon(points=[(0.5, 0.5)])],
        erase_brush_radius_norm=0.1,
    )
    keep = resolve_keep_mask((100, 100), geometry)
    assert keep[50, 50] == False  # noqa: E712 -- erased
    assert keep[5, 5] == True  # noqa: E712 -- kept (inside boundary, not erased)


def test_resolve_keep_mask_with_no_boundary_keeps_everything_not_erased():
    geometry = IsolationGeometry(boundary=None)
    keep = resolve_keep_mask((50, 50), geometry)
    assert keep.all()
