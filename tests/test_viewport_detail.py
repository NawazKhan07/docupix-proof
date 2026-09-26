"""FR-9.1 in the viewer: zoomed in, the visible region is re-rendered crisply."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import make_carton_pdf, make_multi_panel_carton_pdf

from app.config import AppConfig
from app.engine.file_loader import load_artwork
from app.engine.tile_renderer import TileRect, render_vector_tile
from app.engine.worker import run_comparison
from app.models.isolation import IsolationGeometry, NormalizedRect
from app.models.job import ComparisonMode
from app.ui.comparison_viewport import DETAIL_DEBOUNCE_MS, ComparisonViewport
from app.ui.composition import ViewMode
from app.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def pdf_case(tmp_path_factory):
    """Two real PDFs, run through the real pipeline (rotated, one block recoloured).

    Module-scoped: building and aligning them is the slow part, and no test
    here mutates them.
    """
    folder = tmp_path_factory.mktemp("pdfs")
    original = load_artwork(make_carton_pdf(folder / "original.pdf"))
    suspect = load_artwork(
        make_carton_pdf(folder / "suspect.pdf", dx=6, dy=4, rotation=1.2, recolour=True)
    )
    outcome = run_comparison(original.render(), suspect.render(), ComparisonMode.AUTOMATIC, AppConfig())
    yield original, suspect, outcome
    original.close()
    suspect.close()


@pytest.fixture
def viewport(qapp, pdf_case):
    original, suspect, outcome = pdf_case
    widget = ComparisonViewport()
    widget.resize(1100, 700)
    widget.show()
    widget.set_images(outcome.aligned_original, outcome.aligned_suspect)
    widget.set_sources(original, suspect, outcome.alignment.transform)
    widget.set_marker_clusters(outcome.clusters)
    qapp.processEvents()
    yield widget
    widget.close()


def _detail_size(pane):
    pixmap = pane._detail_pixmaps[pane.shown_key]
    return pixmap.width(), pixmap.height()


def test_zoomed_in_the_visible_region_is_re_rendered(viewport):
    viewport.set_zoom(4.0)
    viewport.update_detail_now()

    assert viewport.original_pane.has_detail
    assert viewport.suspect_pane.has_detail


def test_at_or_below_100_percent_no_tile_is_needed(viewport):
    """Below 100% the 300 DPI raster is being *reduced*, which loses nothing."""
    viewport.set_zoom(4.0)
    viewport.update_detail_now()
    viewport.set_zoom(1.0)
    viewport.update_detail_now()

    assert not viewport.has_detail


def test_changing_the_zoom_drops_a_tile_made_for_the_old_zoom_immediately(viewport):
    """A tile at the wrong scale would be drawn the wrong size; it must not linger."""
    viewport.set_zoom(4.0)
    viewport.update_detail_now()
    assert viewport.original_pane.has_detail

    viewport.set_zoom(5.0)

    assert not viewport.original_pane.has_detail  # hidden until the new one is ready
    viewport.update_detail_now()
    assert viewport.original_pane.has_detail


def test_the_tile_is_positioned_and_sized_for_pixel_exact_drawing(viewport):
    viewport.set_zoom(4.0)
    viewport.update_detail_now()
    pane = viewport.original_pane

    origin_x, origin_y = pane._detail_origin
    width, height = _detail_size(pane)
    item = pane._detail_item

    # Anchored so that its device-pixel origin is where it should be...
    assert item.pos().x() == pytest.approx(origin_x / 4.0)
    assert item.pos().y() == pytest.approx(origin_y / 4.0)
    # ...and it covers at least the whole visible region.
    x0, y0, x1, y1 = pane.visible_device_rect()
    assert origin_x <= x0 and origin_y <= y0
    assert origin_x + width >= x1 or origin_x + width >= viewport.original_pane.image_size[0] * 4
    # It ignores the view's zoom so each of its pixels is one screen pixel.
    assert item.flags() & item.GraphicsItemFlag.ItemIgnoresTransformations


def test_the_original_pane_shows_exactly_the_vector_render(viewport, pdf_case):
    original, _suspect, _outcome = pdf_case
    viewport.set_zoom(3.0)
    viewport.update_detail_now()
    pane = viewport.original_pane
    origin_x, origin_y = pane._detail_origin
    width, height = _detail_size(pane)

    expected = render_vector_tile(original, 3.0, TileRect(origin_x, origin_y, width, height))
    shown = pane._detail_pixmaps["original"].toImage().convertToFormat(
        pane._detail_pixmaps["original"].toImage().Format.Format_RGB888
    )
    buffer = np.frombuffer(shown.constBits(), dtype=np.uint8)
    # QImage rows are padded to 4 bytes; strip that before comparing.
    array = buffer.reshape(shown.height(), shown.bytesPerLine())[:, : shown.width() * 3]
    array = array.reshape(shown.height(), shown.width(), 3)

    assert np.array_equal(array, expected)


def test_each_mode_gets_the_tiles_it_needs(viewport):
    viewport.set_zoom(3.0)
    expected_keys = {
        ViewMode.FLICKER: {"flicker0", "flicker1"},
        ViewMode.OVERLAY: {"overlay"},
        ViewMode.GRAYSCALE: {"gray0", "gray1"},
    }
    for mode, keys in expected_keys.items():
        viewport.set_view_mode(mode)
        viewport.update_detail_now()
        assert set(viewport.original_pane._detail_pixmaps) == keys, mode


def test_side_by_side_gives_each_pane_its_own_file(viewport):
    viewport.set_zoom(3.0)
    viewport.update_detail_now()
    assert set(viewport.original_pane._detail_pixmaps) == {"original"}
    assert set(viewport.suspect_pane._detail_pixmaps) == {"suspect"}


def test_flicker_swaps_the_tile_along_with_the_image(viewport):
    """Both frames are pre-rendered, so alternating never waits on a render."""
    viewport.set_zoom(3.0)
    viewport.set_view_mode(ViewMode.FLICKER)
    viewport.update_detail_now()
    pane = viewport.original_pane
    first = pane._detail_item.pixmap().cacheKey()

    viewport._on_tick()

    assert pane.has_detail
    assert pane._detail_item.pixmap().cacheKey() != first


def test_the_viewer_still_works_for_image_files_without_vector_data(qapp, outcome):
    """No sources -> fall back to a bicubic resample of the 300 DPI pair."""
    widget = ComparisonViewport()
    widget.resize(1100, 700)
    widget.show()
    widget.set_images(outcome.aligned_original, outcome.aligned_suspect)
    widget.set_zoom(3.0)

    widget.update_detail_now()

    assert widget.original_pane.has_detail  # a resampled tile, not nothing
    widget.close()


def test_a_render_failure_does_not_take_the_viewer_down(viewport, monkeypatch, caplog):
    """NFR-2: a native-renderer error mid-review must degrade, not crash."""
    import app.ui.comparison_viewport as module

    def explode(*_args, **_kwargs):
        raise RuntimeError("renderer failed")

    monkeypatch.setattr(module, "render_vector_tile", explode)
    viewport.set_zoom(4.0)

    with caplog.at_level("ERROR", logger="app.ui.comparison_viewport"):
        viewport.update_detail_now()  # must not raise

    assert not viewport.original_pane.has_detail  # still showing the base image
    assert any("full-quality tile" in record.message for record in caplog.records)


def test_a_flurry_of_zooms_and_scrolls_is_one_render(viewport, monkeypatch, qapp):
    """Debounce: panning must not pay for a render per scroll event."""
    calls = []
    monkeypatch.setattr(viewport, "update_detail_now", lambda: calls.append(1))
    viewport._detail_timer.timeout.disconnect()
    viewport._detail_timer.timeout.connect(viewport.update_detail_now)

    for zoom in (2.0, 2.5, 3.0, 3.5, 4.0):
        viewport.set_zoom(zoom)
    bar = viewport.original_pane.horizontalScrollBar()
    for value in range(0, 200, 10):
        bar.setValue(value)

    assert viewport._detail_timer.isActive()
    assert calls == []  # nothing rendered yet
    viewport._detail_timer.timeout.emit()
    assert calls == [1]
    assert viewport._detail_timer.interval() == DETAIL_DEBOUNCE_MS


def test_a_hidden_viewer_renders_nothing(viewport):
    viewport.set_zoom(4.0)
    viewport.hide()
    viewport.update_detail_now()
    assert not viewport.has_detail
    assert not viewport._detail_timer.isActive()


def test_releasing_the_sources_clears_detail_and_stops_rendering(viewport):
    viewport.set_zoom(4.0)
    viewport.update_detail_now()
    assert viewport.has_detail

    viewport.release_sources()

    assert not viewport.has_detail
    assert viewport._original_artwork is None and viewport._suspect_artwork is None
    assert not viewport._detail_timer.isActive()


def test_loading_a_new_result_discards_the_old_tile(viewport, pdf_case):
    _original, _suspect, outcome = pdf_case
    viewport.set_zoom(4.0)
    viewport.update_detail_now()
    assert viewport.has_detail

    viewport.set_images(outcome.aligned_original, outcome.aligned_suspect)

    assert not viewport.has_detail


def test_a_locally_aligned_panel_is_crisp_zoomed_in_too(qapp, tmp_path):
    """FR-9.1's crisp zoomed-in vector re-render must also honour a panel's own local alignment
    (`ComparisonOutcome.block_alignments`), not just the whole artwork's transform -- otherwise a panel that
    needed local correction (a fold not quite flat, or files of different overall proportions) renders
    doubled at high zoom even though the very same panel is well registered in the non-zoomed composite."""
    original = load_artwork(make_multi_panel_carton_pdf(tmp_path / "o.pdf"))
    suspect = load_artwork(make_multi_panel_carton_pdf(tmp_path / "s.pdf", shift_second_panel=(15.0, -10.0)))
    carton = IsolationGeometry(boundary=NormalizedRect(0.03, 0.03, 0.97, 0.97))  # panels are found inside an artwork area
    outcome = run_comparison(
        original.render(), suspect.render(), ComparisonMode.AUTOMATIC, AppConfig(), original_geometry=carton, suspect_geometry=carton
    )
    assert outcome.block_alignments  # the shifted panel's own local fit was found and trusted

    widget = ComparisonViewport()
    widget.resize(1100, 700)
    widget.show()
    widget.set_images(outcome.aligned_original, outcome.aligned_suspect)
    widget.set_sources(original, suspect, outcome.alignment.transform, outcome.block_alignments)
    widget.set_zoom(3.0)
    qapp.processEvents()

    # The shifted (right) panel, in shared-frame pixels -- read from its own polygon, not a hardcoded box.
    polygon, _transform = next(item for item in outcome.block_alignments if min(x for x, _y in item[0]) > 400)
    xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
    zoom = widget._zoom
    rect = TileRect(
        int(min(xs) * zoom) + 10, int(min(ys) * zoom) + 10,
        int((max(xs) - min(xs)) * zoom) - 20, int((max(ys) - min(ys)) * zoom) - 20,
    )

    tile_with_block = widget._suspect_tile(rect)
    widget._block_alignments = []  # the bug: only the whole artwork's transform, as before this fix
    tile_global_only = widget._suspect_tile(rect)
    original_tile = widget._original_tile(rect)

    diff_with_block = np.abs(original_tile.astype(int) - tile_with_block.astype(int)).mean()
    diff_global_only = np.abs(original_tile.astype(int) - tile_global_only.astype(int)).mean()
    assert diff_with_block < diff_global_only / 2  # markedly closer with the panel's own correction
    widget.close()
    original.close()
    suspect.close()


# ---------------------------------------------------------------------------
# End to end, through the real window
# ---------------------------------------------------------------------------


def test_the_window_hands_the_pdfs_to_the_viewer(qapp, tmp_path, shared_jobs, run_comparison_in_window):
    original_path = make_carton_pdf(tmp_path / "o.pdf")
    suspect_path = make_carton_pdf(tmp_path / "s.pdf", dx=6, dy=4, rotation=1.2, recolour=True)

    window = MainWindow(jobs=shared_jobs)
    try:
        window.upload_panel.original_slot.load_path(original_path)
        window.upload_panel.suspect_slot.load_path(suspect_path)
        run_comparison_in_window(window)

        viewport = window.results_view.viewport
        assert viewport._original_artwork is window.upload_panel.original
        assert viewport._suspect_artwork is window.upload_panel.suspect
        assert viewport._alignment_transform is not None
    finally:
        window.close()


def test_replacing_a_file_releases_it_from_the_viewer(qapp, tmp_path, shared_jobs, run_comparison_in_window):
    """A replaced file is closed; the viewer must not keep rendering from it."""
    original_path = make_carton_pdf(tmp_path / "o.pdf")
    suspect_path = make_carton_pdf(tmp_path / "s.pdf", dx=6, dy=4, rotation=1.2, recolour=True)

    window = MainWindow(jobs=shared_jobs)
    try:
        window.upload_panel.original_slot.load_path(original_path)
        window.upload_panel.suspect_slot.load_path(suspect_path)
        run_comparison_in_window(window)
        assert window.results_view.viewport._original_artwork is not None

        window.upload_panel.original_slot.load_path(original_path)  # replaces (closes) the old one

        assert window.results_view.viewport._original_artwork is None
    finally:
        window.close()
