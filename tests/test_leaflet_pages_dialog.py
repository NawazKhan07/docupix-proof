"""Leaflet mode's Edit dialog: the whole file, every page, and what is set on each page carried into the comparison
(`app.ui.leaflet_pages_dialog`, `app.engine.leaflet`)."""

from __future__ import annotations

from pathlib import Path

import leaflet_scenes as scenes
import pytest

from app.config import AppConfig
from app.engine.file_loader import load_artwork
from app.engine.leaflet import compare_leaflet
from app.models.isolation import IsolationGeometry, NormalizedRect
from app.models.job import ComparisonMode
from app.ui.leaflet_pages_dialog import LeafletPagesDialog, stack_pages, to_pages, to_stack
from app.ui.main_window import MainWindow
from test_leaflet_options import _pair, _RecordingJobs


def _three_pages(tmp_path: Path) -> Path:
    return _pair(tmp_path)[0]


# ---- where a rectangle drawn on the stacked pages lands ------------------------------------------------------------------


def test_a_rectangle_on_the_stack_is_the_part_of_it_on_each_page_it_covers():
    import numpy as np

    stack, places = stack_pages([np.zeros((100, 200, 3), np.uint8), np.zeros((50, 100, 3), np.uint8)])
    size = (stack.shape[1], stack.shape[0])
    (x, y, w, h), (x2, y2, w2, h2) = places
    assert (w, h, w2, h2) == (200, 100, 100, 50) and y2 > y + h  # one under the other, a band between them
    # from the lower half of page 1 down into page 2
    drawn = NormalizedRect(0.0, (y + 50) / size[1], 50 / size[0], (y2 + 25) / size[1])
    parts = to_pages(drawn, size, places)
    assert set(parts) == {0, 1}
    assert parts[0].y0 == pytest.approx(0.5) and parts[0].y1 == pytest.approx(1.0) and parts[0].x1 == pytest.approx(0.25)
    assert parts[1].y0 == pytest.approx(0.0) and parts[1].y1 == pytest.approx(0.5) and parts[1].x1 == pytest.approx(0.5)
    # and back
    again = to_stack(1, parts[1], size, places)
    assert again.y1 == pytest.approx(drawn.y1)


# ---- the dialog --------------------------------------------------------------------------------------------------------


def test_the_dialog_shows_every_page_and_keeps_what_is_set_on_each(qapp, tmp_path):
    with load_artwork(_three_pages(tmp_path)) as artwork:
        dialog = LeafletPagesDialog(artwork, "Original")
        assert len(dialog._places) == 3  # the whole file, not one page
        size = dialog._size
        # erase an area on page 3, and give page 2 an area of its own
        dialog.canvas.add_erase_region(to_stack(2, NormalizedRect(0.1, 0.1, 0.5, 0.4), size, dialog._places))
        dialog.canvas.set_crop_rect(to_stack(1, NormalizedRect(0.2, 0.2, 0.8, 0.9), size, dialog._places))
        dialog._set_page_area()
        dialog._on_save()
        areas = dialog.result_areas()
        assert set(areas) == {1, 2}
        erased = areas[2].erase_regions[0]
        assert (erased.x0, erased.y0, erased.x1, erased.y1) == pytest.approx((0.1, 0.1, 0.5, 0.4), abs=0.01)
        assert areas[2].boundary is None
        own = areas[1].boundary
        assert (own.x0, own.y0, own.x1, own.y1) == pytest.approx((0.2, 0.2, 0.8, 0.9), abs=0.01)

        # opened again, what was set is shown again
        again = LeafletPagesDialog(artwork, "Original", areas=areas)
        assert len(again.canvas.erase_regions()) == 1 and set(again._page_areas) == {1}
        dialog.close()
        again.close()


def test_leaflet_mode_edits_the_whole_file_and_the_comparison_carries_it(qapp, tmp_path, monkeypatch):
    a, b = _pair(tmp_path)
    jobs = _RecordingJobs()
    window = MainWindow(jobs=jobs)
    window.upload_panel.original_slot.load_path(a)
    window.upload_panel.suspect_slot.load_path(b)
    window.mode_selector.set_mode(ComparisonMode.LEAFLET)
    set_on_page_3 = {2: IsolationGeometry(erase_regions=[NormalizedRect(0.1, 0.1, 0.5, 0.4)])}

    opened = []

    def accept(dialog):
        opened.append(len(dialog._places))
        dialog._result = set_on_page_3
        return LeafletPagesDialog.DialogCode.Accepted

    monkeypatch.setattr(LeafletPagesDialog, "exec", accept)
    slot = window.upload_panel.suspect_slot
    assert slot.edit_artwork_area()
    assert opened == [3]
    assert slot.page_areas == set_on_page_3
    assert "1 area erased" in slot.detail_label.text()
    request = window._setup_request()
    assert dict(request.suspect_page_areas) == set_on_page_3 and request.original_page_areas == ()
    window.close()


# ---- the comparison ------------------------------------------------------------------------------------------------------


class _FakeArtwork:
    def __init__(self, pages):
        self._pages = pages
        self.page_count = len(pages)

    def render(self, dpi=None, page=0):
        return self._pages[page]


def _both_ways():
    from dataclasses import replace

    config = AppConfig()
    return replace(config, leaflet=replace(config.leaflet, compare_text=True, compare_visual=True))


def test_a_change_in_an_area_erased_on_a_page_is_not_reported():
    original = scenes.page("indications")
    suspect = scenes.page("indications", edit=scenes.edit_first_line)
    assert compare_leaflet(_FakeArtwork([original]), _FakeArtwork([suspect]), _both_ways()).all_clusters

    height, width = original.shape[:2]
    first_line = NormalizedRect(40 / width, 130 / height, 640 / width, 195 / height)
    erased = {0: IsolationGeometry(erase_regions=[first_line])}
    outcome = compare_leaflet(_FakeArtwork([original]), _FakeArtwork([suspect]), _both_ways(), suspect_areas=erased)
    assert outcome.all_clusters == []


def test_a_change_outside_the_area_erased_is_still_reported():
    original = scenes.page("indications")
    suspect = scenes.page("indications", edit=scenes.edit_first_line)
    height, width = original.shape[:2]
    elsewhere = {0: IsolationGeometry(erase_regions=[NormalizedRect(40 / width, 900 / height, 640 / width, 990 / height)])}
    outcome = compare_leaflet(_FakeArtwork([original]), _FakeArtwork([suspect]), _both_ways(), original_areas=elsewhere)
    assert outcome.all_clusters


def test_a_pages_own_area_is_compared_instead_of_the_one_found():
    original = scenes.page("indications")
    suspect = scenes.page("indications", edit=scenes.edit_first_line)
    height, width = original.shape[:2]
    # only the lower part of the page is compared on both sides: the change in its first line is outside it
    lower = IsolationGeometry(boundary=NormalizedRect(0.0, 400 / height, 1.0, 1.0))
    outcome = compare_leaflet(
        _FakeArtwork([original]), _FakeArtwork([suspect]), _both_ways(), original_areas={0: lower}, suspect_areas={0: lower}
    )
    assert outcome.all_clusters == []
