from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from app.engine.file_loader import (
    CorruptFileError,
    UnsupportedFileFormatError,
    load_artwork,
)


def test_load_pdf_reports_kind_and_page_count(synthetic_pdf: Path):
    artwork = load_artwork(synthetic_pdf)
    try:
        assert artwork.kind == "pdf"
        assert artwork.page_count == 1
    finally:
        artwork.close()


def test_pdf_renders_at_working_dpi_with_correct_pixel_size(synthetic_pdf: Path):
    artwork = load_artwork(synthetic_pdf)
    try:
        expected_w, expected_h = artwork.page_size_px(dpi=300)
        rendered = artwork.render(dpi=300)
        assert rendered.shape == (expected_h, expected_w, 3)
        assert rendered.dtype == np.uint8
    finally:
        artwork.close()


def test_pdf_rerenders_at_higher_dpi_for_zoom_not_just_upscaled(synthetic_pdf: Path):
    """FR-9.1: zooming in re-rasterizes from vector data, not a blown-up bitmap."""
    artwork = load_artwork(synthetic_pdf)
    try:
        low = artwork.render(dpi=150)
        high = artwork.render(dpi=600)
        # Roughly 4x in each dimension (small integer drift is expected: PDF
        # pixmap sizing ceils a fractional point->pixel scale independently
        # at each DPI, so it isn't exactly proportional at every DPI pair).
        assert abs(high.shape[0] - low.shape[0] * 4) <= 4
        assert abs(high.shape[1] - low.shape[1] * 4) <= 4
        # A naive nearest-neighbour upscale of `low` would look blocky; a
        # real re-render at 4x DPI should differ from that (anti-aliased,
        # crisper edges) rather than being that same blown-up bitmap.
        upscaled_low = np.repeat(np.repeat(low, 4, axis=0), 4, axis=1)
        h, w = min(high.shape[0], upscaled_low.shape[0]), min(high.shape[1], upscaled_low.shape[1])
        assert not np.array_equal(high[:h, :w], upscaled_low[:h, :w])
    finally:
        artwork.close()


def test_load_png_preserves_native_resolution_exactly(synthetic_png: Path):
    artwork = load_artwork(synthetic_png)
    try:
        assert artwork.kind == "image"
        rendered = artwork.render()
        assert rendered.shape == (150, 200, 3)
        expected = np.array(PILImage.open(synthetic_png).convert("RGB"))
        assert np.array_equal(rendered, expected)  # no lossy re-encoding
    finally:
        artwork.close()


def test_image_render_ignores_zoom_never_downsamples(synthetic_png: Path):
    artwork = load_artwork(synthetic_png)
    try:
        first = artwork.render(dpi=72)
        second = artwork.render(dpi=600)
        assert np.array_equal(first, second)
        assert first.shape == (150, 200, 3)
    finally:
        artwork.close()


def test_unsupported_extension_raises_clean_error(tmp_path: Path):
    path = tmp_path / "artwork.txt"
    path.write_text("hello")
    with pytest.raises(UnsupportedFileFormatError):
        load_artwork(path)


def test_corrupt_pdf_raises_clean_error_not_a_crash(corrupt_pdf: Path):
    with pytest.raises(CorruptFileError):
        load_artwork(corrupt_pdf)


def test_corrupt_image_raises_clean_error_not_a_crash(corrupt_png: Path):
    with pytest.raises(CorruptFileError):
        load_artwork(corrupt_png)


def test_missing_file_raises_clean_error(tmp_path: Path):
    with pytest.raises(CorruptFileError):
        load_artwork(tmp_path / "does_not_exist.pdf")
