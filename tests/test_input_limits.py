"""Oversized input is refused with a clear message before anything large is allocated (Architecture §7)."""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz
import numpy as np
import pytest
from PIL import Image

from app.config import AppConfig, RuntimeConfig, load_config
from app.engine import file_loader
from app.engine.file_loader import ArtworkLoadError, InputTooLargeError, load_artwork


def _pdf(path: Path, pages: int = 1, size=(300, 400)) -> Path:
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page(width=size[0], height=size[1])
        page.draw_rect(fitz.Rect(20, 20, 100, 100), fill=(1, 0, 0))
    doc.save(path)
    doc.close()
    return path


def _png(path: Path, size=(200, 100)) -> Path:
    Image.fromarray(np.full((size[1], size[0], 3), 128, np.uint8)).save(path)
    return path


def test_a_file_over_the_size_limit_is_refused_and_says_how_to_proceed(tmp_path):
    path = _png(tmp_path / "a.png")
    tiny_limit = RuntimeConfig(max_file_size_mb=0)  # anything non-empty is too big

    with pytest.raises(InputTooLargeError) as caught:
        load_artwork(path, tiny_limit)

    message = str(caught.value)
    assert "a.png" in message and "MB" in message and "max_file_size_mb" in message


def test_a_file_within_the_limit_loads(tmp_path):
    with load_artwork(_png(tmp_path / "a.png"), RuntimeConfig(max_file_size_mb=1)) as artwork:
        assert artwork.kind == "image"


def test_too_many_pdf_pages_is_refused(tmp_path):
    path = _pdf(tmp_path / "catalogue.pdf", pages=7)
    with pytest.raises(InputTooLargeError) as caught:
        load_artwork(path, RuntimeConfig(max_pdf_pages=5))
    assert "7 pages" in str(caught.value) and "5-page limit" in str(caught.value)


def test_a_pdf_within_the_page_limit_loads_even_with_several_pages(tmp_path):
    with load_artwork(_pdf(tmp_path / "leaflet.pdf", pages=3), RuntimeConfig(max_pdf_pages=5)) as artwork:
        assert artwork.page_count == 3


def test_a_page_too_large_to_render_is_refused_before_rendering_it(tmp_path, monkeypatch):
    # 300 x 400 pt at 300 DPI is 1250 x 1667 px = 2.1 MP.
    path = _pdf(tmp_path / "sheet.pdf")
    renders = []
    monkeypatch.setattr(file_loader.LoadedArtwork, "render", lambda *a, **k: renders.append(1))

    with pytest.raises(InputTooLargeError) as caught:
        load_artwork(path, RuntimeConfig(max_render_megapixels=1))

    assert "megapixel" in str(caught.value) and "max_render_megapixels" in str(caught.value)
    assert renders == []


def test_an_image_too_large_is_refused_from_its_header_before_it_is_decoded(tmp_path, monkeypatch):
    path = _png(tmp_path / "big.png", size=(2000, 1500))  # 3 MP
    decoded = []
    original_load = Image.Image.load
    monkeypatch.setattr(Image.Image, "load", lambda self, *a, **k: decoded.append(1) or original_load(self, *a, **k))

    with pytest.raises(InputTooLargeError):
        load_artwork(path, RuntimeConfig(max_render_megapixels=1))

    assert decoded == []


def test_the_default_limits_admit_a_real_carton_and_refuse_absurd_input():
    limits = RuntimeConfig()
    carton_megapixels = 3660 * 4965 / 1e6  # 310 x 420 mm at 300 DPI
    metre_square_megapixels = 11811 * 11811 / 1e6
    assert carton_megapixels < limits.max_render_megapixels
    assert metre_square_megapixels < limits.max_render_megapixels  # big, but a real sheet
    assert 20_000 * 20_000 / 1e6 > limits.max_render_megapixels  # not a sheet: a mistake


def test_it_is_still_an_ordinary_load_error_for_the_ui_to_show(tmp_path):
    """The upload slot catches ArtworkLoadError and shows the message, so these must be one."""
    assert issubclass(InputTooLargeError, ArtworkLoadError)


def test_limits_come_from_config_json(tmp_path):
    (tmp_path / "config.json").write_text('{"runtime": {"max_pdf_pages": 3, "worker_recycle_after_jobs": 7}}')
    config = load_config(tmp_path / "config.json")
    assert config.runtime.max_pdf_pages == 3 and config.runtime.worker_recycle_after_jobs == 7
    assert config.runtime.max_file_size_mb == AppConfig().runtime.max_file_size_mb


def test_a_refused_file_leaves_no_open_handle_behind(tmp_path):
    path = _pdf(tmp_path / "catalogue.pdf", pages=7)
    with pytest.raises(InputTooLargeError):
        load_artwork(path, RuntimeConfig(max_pdf_pages=5))
    path.unlink()  # would fail on Windows if the document were still open
    assert not path.exists()
