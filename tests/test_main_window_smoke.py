from __future__ import annotations

from pathlib import Path

from app.models.job import ComparisonMode
from app.ui.main_window import MainWindow


def test_main_window_constructs_with_compare_disabled(qapp):
    window = MainWindow()
    try:
        assert window.compare_button.isEnabled() is False
        assert window.mode_selector.mode == ComparisonMode.AUTOMATIC
    finally:
        window.close()


def test_compare_enables_only_once_both_files_loaded(
    qapp, synthetic_pdf: Path, synthetic_png: Path
):
    window = MainWindow()
    try:
        window.upload_panel.original_slot.load_path(synthetic_pdf)
        assert window.compare_button.isEnabled() is False
        window.upload_panel.suspect_slot.load_path(synthetic_png)
        assert window.compare_button.isEnabled() is True
    finally:
        window.close()
