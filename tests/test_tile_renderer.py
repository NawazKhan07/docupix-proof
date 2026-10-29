"""FR-9.1: full quality at every zoom level, including fully zoomed in."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from helpers import make_carton_pdf

from app.config import AlignmentConfig
from app.engine.alignment.aligner import align
from app.engine.file_loader import load_artwork
from app.engine.tile_renderer import (
    TileRect,
    is_vector,
    render_aligned_vector_tile,
    render_raster_tile,
    render_vector_tile,
)


@pytest.fixture
def original_pdf(tmp_path):
    artwork = load_artwork(make_carton_pdf(tmp_path / "original.pdf"))
    yield artwork
    artwork.close()


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(float)


# ---------------------------------------------------------------------------
# Vector tiles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rect",
    [TileRect(100, 80, 200, 150), TileRect(0, 0, 120, 90), TileRect(700, 450, 150, 100), TileRect(300, 200, 64, 64)],
)
def test_a_tile_at_100_percent_is_pixel_identical_to_the_full_render(original_pdf, rect):
    """The clip/translate maths must place every pixel exactly, wherever the tile is."""
    full = original_pdf.render(dpi=300)
    tile = render_vector_tile(original_pdf, 1.0, rect)
    assert np.array_equal(tile, full[rect.y : rect.y + rect.height, rect.x : rect.x + rect.width])


def test_a_tile_on_a_rotated_page_is_also_pixel_identical(tmp_path):
    artwork = load_artwork(make_carton_pdf(tmp_path / "rotated.pdf", page_rotation=90))
    try:
        full = artwork.render(dpi=300)
        rect = TileRect(100, 80, 200, 150)
        assert np.array_equal(render_vector_tile(artwork, 1.0, rect), full[80:230, 100:300])
    finally:
        artwork.close()


@pytest.mark.parametrize("zoom", [1.0, 2.5, 6.0])
def test_a_tile_is_always_exactly_the_requested_size(original_pdf, zoom):
    rect = TileRect(37, 51, 211, 133)
    assert render_vector_tile(original_pdf, zoom, rect).shape == (133, 211, 3)


def test_a_tile_past_the_page_edge_is_padded_with_white(original_pdf):
    """A viewport can extend beyond the page; that must not crash or wrap."""
    width, height = original_pdf.page_size_px(dpi=300)
    tile = render_vector_tile(original_pdf, 1.0, TileRect(width - 20, height - 20, 100, 100))
    assert tile.shape == (100, 100, 3)
    assert np.all(tile[30:, 30:] == 255)  # entirely off the page


def test_a_tile_far_off_the_page_is_blank(original_pdf):
    tile = render_vector_tile(original_pdf, 2.0, TileRect(50000, 50000, 80, 60))
    assert tile.shape == (60, 80, 3) and np.all(tile == 255)


def test_zooming_in_re_renders_from_vector_data_instead_of_magnifying_pixels(original_pdf):
    """FR-9.1, the point of this module: at 600% an edge must stay sharp."""
    zoom = 6.0
    magnified = cv2.resize(original_pdf.render(dpi=300), None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)

    edge_x = int(30 / 72 * 300 * zoom)  # the frame's left edge
    rect = TileRect(edge_x - 40, int(600 * zoom / 2) - 20, 80, 40)
    vector = render_vector_tile(original_pdf, zoom, rect)
    stretched = magnified[rect.y : rect.y + rect.height, rect.x : rect.x + rect.width]

    def intermediate_pixels(tile):
        row = _gray(tile)[tile.shape[0] // 2]
        return int(((row > 20) & (row < 235)).sum())

    assert intermediate_pixels(vector) <= 3  # a hard edge...
    assert intermediate_pixels(stretched) >= 8  # ...versus the smear from magnifying
    assert np.abs(np.gradient(_gray(vector), axis=1)).max() > 2 * np.abs(np.gradient(_gray(stretched), axis=1)).max()


def test_the_wrong_kind_of_source_is_refused(tmp_path):
    from PIL import Image

    path = tmp_path / "photo.png"
    Image.fromarray(np.full((50, 50, 3), 255, dtype=np.uint8)).save(path)
    image = load_artwork(path)
    try:
        assert not is_vector(image)
        with pytest.raises(ValueError):
            render_vector_tile(image, 2.0, TileRect(0, 0, 10, 10))
    finally:
        image.close()


def test_is_vector_only_for_pdfs(original_pdf):
    assert is_vector(original_pdf)
    assert not is_vector(None)


# ---------------------------------------------------------------------------
# Raster fallback
# ---------------------------------------------------------------------------


def test_a_raster_tile_at_100_percent_reproduces_the_source_exactly():
    rng = np.random.default_rng(7)
    image = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
    rect = TileRect(50, 40, 120, 90)
    assert np.array_equal(render_raster_tile(image, 1.0, rect), image[40:130, 50:170])


def test_a_raster_tile_magnifies_in_the_right_place():
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    image[80:120, 60:100] = (200, 30, 40)  # a red block

    tile = render_raster_tile(image, 4.0, TileRect(200, 280, 300, 250))  # device px at 4x

    red = np.all(np.abs(tile.astype(int) - (200, 30, 40)) < 30, axis=2)
    ys, xs = np.nonzero(red)
    # 60..100 at 4x is 240..400 in device px, i.e. 40..200 inside a tile starting at x=200.
    assert xs.min() == pytest.approx(240 - 200, abs=3) and xs.max() == pytest.approx(400 - 200, abs=3)
    assert ys.min() == pytest.approx(320 - 280, abs=3) and ys.max() == pytest.approx(480 - 280, abs=3)


def test_a_raster_tile_is_always_exactly_the_requested_size():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    assert render_raster_tile(image, 3.0, TileRect(280, 250, 90, 70)).shape == (70, 90, 3)


# ---------------------------------------------------------------------------
# Aligned vector tiles: the suspect, drawn through the alignment transform
# ---------------------------------------------------------------------------


@pytest.fixture
def aligned_pair(tmp_path):
    """An original and a suspect that is genuinely shifted and rotated in vector space."""
    original = load_artwork(make_carton_pdf(tmp_path / "o.pdf"))
    suspect = load_artwork(make_carton_pdf(tmp_path / "s.pdf", dx=9, dy=5, rotation=2.0))
    result = align(original.render(), suspect.render(), AlignmentConfig())
    yield original, suspect, result
    original.close()
    suspect.close()


@pytest.mark.parametrize(
    "zoom, rect",
    [(1.0, TileRect(100, 80, 300, 200)), (3.0, TileRect(420, 300, 500, 350)), (6.0, TileRect(900, 650, 600, 400))],
)
def test_the_aligned_suspect_tile_lands_on_the_originals_content(aligned_pair, zoom, rect):
    """The composed transform maths: the suspect must overlay the original at any zoom."""
    original, suspect, result = aligned_pair

    original_tile = render_vector_tile(original, zoom, rect)
    aligned = render_aligned_vector_tile(suspect, result.transform, zoom, rect)
    unaligned = render_aligned_vector_tile(suspect, np.eye(3), zoom, rect)

    error = np.abs(_gray(original_tile) - _gray(aligned)).mean()
    baseline = np.abs(_gray(original_tile) - _gray(unaligned)).mean()
    assert aligned.shape == original_tile.shape
    assert error < 5.0  # measured 1.4-2.7 across these zooms
    assert baseline > 5 * error  # and it is the transform that made the difference


def test_the_aligned_suspect_tile_is_sharp_not_a_magnified_warp(aligned_pair):
    """The suspect must be as crisp as the original, not capped at 300 DPI."""
    original, suspect, result = aligned_pair
    zoom, rect = 6.0, TileRect(900, 650, 600, 400)

    crisp = render_aligned_vector_tile(suspect, result.transform, zoom, rect)

    # What the viewer would otherwise show: the 300 DPI warped raster, magnified.
    from app.engine.alignment.homography import warp_to_reference

    warped = warp_to_reference(suspect.render(), result.transform, original.render().shape[:2])
    magnified = render_raster_tile(warped, zoom, rect)

    assert np.abs(np.gradient(_gray(crisp), axis=1)).max() > 1.5 * np.abs(np.gradient(_gray(magnified), axis=1)).max()


def test_a_pure_shift_is_undone_exactly(tmp_path):
    """The simplest case, where the answer is known analytically."""
    dx, dy = 12.0, 7.0
    original = load_artwork(make_carton_pdf(tmp_path / "o.pdf"))
    suspect = load_artwork(make_carton_pdf(tmp_path / "s.pdf", dx=dx, dy=dy))
    try:
        k = 300 / 72
        # suspect px -> original px: undo the (dx, dy) point shift.
        transform = np.array([[1, 0, -dx * k], [0, 1, -dy * k], [0, 0, 1.0]])
        rect = TileRect(300, 200, 400, 300)

        aligned = render_aligned_vector_tile(suspect, transform, 2.0, rect)
        expected = render_vector_tile(original, 2.0, rect)

        assert np.abs(_gray(aligned) - _gray(expected)).mean() < 1.0
    finally:
        original.close()
        suspect.close()


def test_an_aligned_tile_is_always_exactly_the_requested_size(aligned_pair):
    _original, suspect, result = aligned_pair
    assert render_aligned_vector_tile(suspect, result.transform, 2.5, TileRect(10, 20, 333, 177)).shape == (177, 333, 3)


def test_an_aligned_tile_past_the_suspects_edge_is_white(aligned_pair):
    _original, suspect, result = aligned_pair
    tile = render_aligned_vector_tile(suspect, result.transform, 2.0, TileRect(60000, 60000, 50, 50))
    assert np.all(tile == 255)


def test_a_tile_is_drawn_at_the_resolution_the_page_was_compared_at(original_pdf):
    """The window's own copy of a file may hold it at another resolution than the comparison did (less memory was free
    when the comparison ran): a zoomed-in re-render must land where the compared picture is, not elsewhere on the page."""
    compared = original_pdf.render(dpi=150)
    rect = TileRect(60, 40, 120, 90)
    tile = render_vector_tile(original_pdf, 1.0, rect, dpi=150)
    assert np.array_equal(tile, compared[rect.y : rect.y + rect.height, rect.x : rect.x + rect.width])
