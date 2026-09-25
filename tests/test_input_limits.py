"""No file is turned away for its size: a page too big to hold at 300 DPI is rendered at a lower DPI that fits, and the
comparison says so. The size and page-count limits exist (Architecture §7) but are off unless a person sets them."""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz
import numpy as np
import pytest
from PIL import Image

from app.config import AppConfig, RuntimeConfig, load_config
from app.engine.file_loader import WORKING_DPI, ArtworkLoadError, InputTooLargeError, load_artwork


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


# ---- nothing is refused by default -------------------------------------------------------------------------------------------------------


def test_the_default_limits_turn_nothing_away():
    limits = RuntimeConfig()
    assert limits.max_file_size_mb == 0 and limits.max_pdf_pages == 0  # (0 = no limit)


def test_a_huge_page_loads_and_is_rendered_at_the_dpi_that_fits(tmp_path):
    """A 2.2 m x 1.1 m shipper is 332 megapixels at 300 DPI, which used to be refused."""
    path = _pdf(tmp_path / "shipper.pdf", size=(6200, 3080))  # points: 2187 x 1087 mm
    with load_artwork(path, RuntimeConfig(max_render_megapixels=100)) as artwork:
        dpi = artwork.working_dpi()
        assert dpi < WORKING_DPI
        width, height = artwork.page_size_px(dpi)
        assert width * height <= 100e6 * 1.01
        assert width * height >= 100e6 * 0.9  # (as much resolution as the budget allows)
        image = artwork.render()
        assert image.shape[:2] == (height, width)
        assert "compared at" in artwork.reduced_note()


def test_a_page_that_fits_is_rendered_at_the_full_working_dpi_and_says_nothing(tmp_path):
    with load_artwork(_pdf(tmp_path / "carton.pdf"), RuntimeConfig()) as artwork:
        assert artwork.working_dpi() == WORKING_DPI
        assert artwork.reduced_note() is None


def test_an_image_bigger_than_the_budget_is_reduced_not_refused(tmp_path):
    path = _png(tmp_path / "big.png", size=(2000, 1500))  # 3 MP
    with load_artwork(path, RuntimeConfig(max_render_megapixels=1)) as artwork:
        image = artwork.render()
        assert image.shape[0] * image.shape[1] <= 1.05e6
        assert "more than fits" in artwork.reduced_note()


def test_an_image_within_the_budget_is_returned_at_its_own_pixels(tmp_path):
    with load_artwork(_png(tmp_path / "a.png", size=(200, 100)), RuntimeConfig()) as artwork:
        assert artwork.render().shape[:2] == (100, 200)
        assert artwork.reduced_note() is None


def test_a_pdf_of_any_number_of_pages_loads_by_default(tmp_path):
    with load_artwork(_pdf(tmp_path / "leaflet.pdf", pages=7), RuntimeConfig()) as artwork:
        assert artwork.page_count == 7


# ---- the limits, where a person has set them ------------------------------------------------------------------------------------------------


def test_a_file_over_a_limit_someone_set_is_refused_and_says_how_to_proceed(tmp_path):
    path = _png(tmp_path / "a.png")
    with pytest.raises(InputTooLargeError) as caught:
        load_artwork(path, RuntimeConfig(max_file_size_mb=-1))  # (nothing is that small: any non-empty file is over it)
    assert "a.png" in str(caught.value) and "max_file_size_mb" in str(caught.value)


def test_too_many_pdf_pages_is_refused_only_when_a_limit_is_set(tmp_path):
    path = _pdf(tmp_path / "catalogue.pdf", pages=7)
    with pytest.raises(InputTooLargeError) as caught:
        load_artwork(path, RuntimeConfig(max_pdf_pages=5))
    assert "7 pages" in str(caught.value) and "5-page limit" in str(caught.value)


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
