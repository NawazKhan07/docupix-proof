from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QPointF

from app.engine.file_loader import load_artwork
from app.models.isolation import IsolationTool, NormalizedRect
from app.ui.isolation_dialog import IsolationDialog


def test_save_crop_applies_auto_rotate_and_returns_geometry(qapp, tmp_path: Path):
    width, height = 300, 300
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (60, 80), (240, 220), (0, 0, 0), -1)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 10, 1.0)
    image = cv2.warpAffine(image, matrix, (width, height), borderValue=(255, 255, 255))
    path = tmp_path / "rotated.png"
    Image.fromarray(image).save(path)

    artwork = load_artwork(path)
    try:
        dialog = IsolationDialog(artwork, "Original")
        dialog.canvas.set_tool(IsolationTool.CROP)
        dialog.canvas.begin_crop_drag(QPointF(0, 0))
        dialog.canvas.end_crop_drag(QPointF(width, height))

        dialog._on_save()

        geometry = dialog.result_geometry()
        assert geometry is not None
        assert isinstance(geometry.boundary, NormalizedRect)
        assert abs(geometry.rotation_deg) > 3  # a real correction was found
    finally:
        artwork.close()


def test_a_held_press_lifts_the_area_inside_a_pdfs_die_line(qapp, synthetic_pdf: Path):
    """The synthetic page is a vector rectangle (the die line) with text inside it."""
    artwork = load_artwork(synthetic_pdf)
    try:
        dialog = IsolationDialog(artwork, "Original")
        dialog.canvas.set_tool(IsolationTool.LIFT_SUBJECT)

        dialog._on_subject_requested(0.5, 0.5)

        polygon = dialog.canvas.polygon()
        assert polygon is not None
        xs = [x for x, _ in polygon.points]
        ys = [y for _, y in polygon.points]
        assert min(xs) == pytest.approx(20 / 400, abs=0.01) and max(xs) == pytest.approx(380 / 400, abs=0.01)
        assert min(ys) == pytest.approx(20 / 300, abs=0.01) and max(ys) == pytest.approx(280 / 300, abs=0.01)
    finally:
        artwork.close()


def test_nothing_to_lift_says_so_in_the_status_line(qapp, tmp_path: Path):
    blank = np.full((100, 100, 3), 255, dtype=np.uint8)
    path = tmp_path / "blank.png"
    Image.fromarray(blank).save(path)
    artwork = load_artwork(path)
    try:
        dialog = IsolationDialog(artwork, "Original")
        dialog.canvas.set_tool(IsolationTool.LIFT_SUBJECT)
        dialog._on_subject_requested(0.5, 0.5)
        assert dialog.canvas.polygon() is None
        assert "Press and hold on the artwork" in dialog.status_label.text() or "too small" in dialog.status_label.text()
    finally:
        artwork.close()


def test_save_with_no_selection_saves_no_boundary(qapp, synthetic_png: Path):
    artwork = load_artwork(synthetic_png)
    try:
        dialog = IsolationDialog(artwork, "Original")
        dialog._on_save()
        geometry = dialog.result_geometry()
        assert geometry is not None
        assert geometry.boundary is None
    finally:
        artwork.close()
