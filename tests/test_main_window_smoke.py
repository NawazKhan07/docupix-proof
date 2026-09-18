from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

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


def test_compare_reports_alignment_result_in_the_status_strip(qapp, tmp_path: Path):
    """A registrable pair reports its measured residual to the user (NFR-3)."""
    original_path, suspect_path = _artwork_pair(tmp_path)

    window = MainWindow()
    try:
        window.upload_panel.original_slot.load_path(original_path)
        window.upload_panel.suspect_slot.load_path(suspect_path)
        window._on_compare()

        text = window.status_label.text()
        assert "alignment" in text and "px" in text  # NFR-3 residual is surfaced
        assert window.status_label.property("status") in {"ok", "warn"}
        assert window.compare_button.isEnabled() is True  # re-enabled afterwards
    finally:
        window.close()


def test_compare_surfaces_an_error_when_files_cannot_be_registered(qapp, tmp_path: Path):
    blank_a = tmp_path / "blank_a.png"
    blank_b = tmp_path / "blank_b.png"
    for path in (blank_a, blank_b):
        Image.fromarray(np.full((200, 200, 3), 255, dtype=np.uint8)).save(path)

    window = MainWindow()
    try:
        window.upload_panel.original_slot.load_path(blank_a)
        window.upload_panel.suspect_slot.load_path(blank_b)
        window._on_compare()

        assert window.status_label.property("status") == "error"
        assert "Could not register" in window.status_label.text()
        assert window.compare_button.isEnabled() is True
    finally:
        window.close()


def _artwork_pair(tmp_path: Path) -> tuple[Path, Path]:
    """An original and a slightly shifted suspect, saved as PNGs."""
    image = np.full((400, 600, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (40, 30), (560, 370), (0, 0, 0), 3)
    cv2.putText(image, "PARACETAMOL 500mg", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
    cv2.rectangle(image, (70, 200), (240, 290), (200, 30, 40), -1)
    cv2.circle(image, (430, 240), 50, (30, 90, 200), -1)

    matrix = cv2.getRotationMatrix2D((300, 200), 1.5, 1.0)
    matrix[0, 2] += 10
    shifted = cv2.warpAffine(image, matrix, (600, 400), borderValue=(255, 255, 255))

    original_path = tmp_path / "original.png"
    suspect_path = tmp_path / "suspect.png"
    Image.fromarray(image).save(original_path)
    Image.fromarray(shifted).save(suspect_path)
    return original_path, suspect_path
