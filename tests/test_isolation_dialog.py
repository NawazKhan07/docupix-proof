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


def test_point_click_traces_dieline_on_vector_pdf(qapp, synthetic_pdf: Path):
    artwork = load_artwork(synthetic_pdf)
    try:
        dialog = IsolationDialog(artwork, "Original")
        dialog.canvas.set_tool(IsolationTool.POLYGON_TRACE)

        height, width = dialog._rendered.shape[:2]
        dialog.canvas.click_point(QPointF(5, height / 2))  # near left dieline edge

        polygon = dialog.canvas.polygon()
        assert polygon is not None
        assert len(polygon.points) == 4
    finally:
        artwork.close()


def test_point_click_shows_message_when_no_dieline_found(qapp, tmp_path: Path, monkeypatch):
    # QMessageBox.information() would otherwise block on a real (modal) event
    # loop waiting for a user click; there's no user under the offscreen
    # platform, so replace it with a no-op for this test.
    import app.ui.isolation_dialog as isolation_dialog_module

    monkeypatch.setattr(isolation_dialog_module.QMessageBox, "information", lambda *a, **k: None)

    blank = np.full((100, 100, 3), 255, dtype=np.uint8)
    path = tmp_path / "blank.png"
    Image.fromarray(blank).save(path)
    artwork = load_artwork(path)
    try:
        dialog = IsolationDialog(artwork, "Original")
        dialog.canvas.set_tool(IsolationTool.POLYGON_TRACE)
        dialog.canvas.click_point(QPointF(50, 50))
        assert dialog.canvas.polygon() is None
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
