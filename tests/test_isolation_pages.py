"""Every page of a file of several in the Edit Artwork Area dialog (every mode but Leaflet, which has its own
whole-file editor): shown as pictures to pick, the area drawn on the page on show, and that page the one compared."""

from __future__ import annotations

from app.engine.file_loader import load_artwork
from app.models.job import ComparisonMode
from app.ui.isolation_dialog import IsolationDialog
from app.ui.main_window import MainWindow
from test_leaflet_options import _pair, _RecordingJobs


def test_every_page_is_offered_and_picking_one_shows_it(qapp, tmp_path):
    a, _b = _pair(tmp_path)  # three pages
    with load_artwork(a) as artwork:
        dialog = IsolationDialog(artwork, "Original")
        assert len(dialog.page_buttons) == 3 and dialog.page_buttons[0].isChecked()
        first = dialog._rendered.copy()
        dialog.page_buttons[2].click()
        assert artwork.page == 2 and dialog.result_page() == 2
        assert (dialog._rendered != first).any()  # that page's picture is on show
        dialog.reject()
        assert artwork.page == 0  # cancelled: the file stays on the page it was on


def test_a_single_page_file_shows_no_pages_to_pick(qapp, tmp_path):
    import numpy as np
    from PIL import Image

    Image.fromarray(np.full((80, 120, 3), 200, np.uint8)).save(tmp_path / "one.png")
    with load_artwork(tmp_path / "one.png") as artwork:
        dialog = IsolationDialog(artwork, "Original")
        assert dialog.page_buttons == []
        dialog.close()


def test_the_page_the_area_was_drawn_on_is_the_one_compared(qapp, tmp_path, monkeypatch):
    a, b = _pair(tmp_path)
    window = MainWindow(jobs=_RecordingJobs())
    window.upload_panel.original_slot.load_path(a)
    window.upload_panel.suspect_slot.load_path(b)
    window.mode_selector.set_mode(ComparisonMode.AUTOMATIC)

    def pick_page_three_and_save(dialog):
        dialog.page_buttons[2].click()
        dialog._on_save()
        return IsolationDialog.DialogCode.Accepted

    monkeypatch.setattr(IsolationDialog, "exec", pick_page_three_and_save)
    slot = window.upload_panel.original_slot
    assert slot.edit_artwork_area()
    assert slot.page_choice == 2 and slot.loaded.page == 2
    assert window._setup_request().original_page == 2
    window.close()
