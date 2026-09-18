from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF

from app.ui.isolation_dialog import IsolationDialog
from app.ui.upload_panel import UploadSlot


def test_isolate_button_disabled_until_file_loaded(qapp):
    slot = UploadSlot("Original")
    assert slot.isolate_button.isEnabled() is False


def test_isolate_button_enables_after_load_and_disables_on_clear(qapp, synthetic_png: Path):
    slot = UploadSlot("Original")
    slot.load_path(synthetic_png)
    assert slot.isolate_button.isEnabled() is True

    slot.clear()
    assert slot.isolate_button.isEnabled() is False


def test_edit_artwork_area_stores_geometry_on_accept(qapp, synthetic_png: Path, monkeypatch):
    """Drives the isolation dialog without its real modal event loop: fakes
    `exec()` to draw a crop and accept, the way a user Save-clicking would."""

    def fake_exec(self: IsolationDialog):
        width, height = self._rendered.shape[1], self._rendered.shape[0]
        self.canvas.begin_crop_drag(QPointF(0, 0))
        self.canvas.end_crop_drag(QPointF(width, height))
        self._on_save()
        return IsolationDialog.DialogCode.Accepted

    monkeypatch.setattr(IsolationDialog, "exec", fake_exec)

    slot = UploadSlot("Original")
    slot.load_path(synthetic_png)
    changed_count = 0
    slot.changed.connect(lambda: None)  # sanity: signal is connectable

    slot._on_edit_artwork_area()

    assert slot.geometry is not None
    assert slot.geometry.boundary is not None
    assert "crop area set" in slot.detail_label.text()
    assert slot.status_pill.text() == "Loaded"


def test_edit_artwork_area_discards_geometry_on_cancel(qapp, synthetic_png: Path, monkeypatch):
    def fake_exec_cancel(self: IsolationDialog):
        return IsolationDialog.DialogCode.Rejected

    monkeypatch.setattr(IsolationDialog, "exec", fake_exec_cancel)

    slot = UploadSlot("Original")
    slot.load_path(synthetic_png)
    slot._on_edit_artwork_area()

    assert slot.geometry is None
