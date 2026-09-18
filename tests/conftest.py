from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt for tests

from pathlib import Path

import fitz
import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="session")
def qapp():
    """A single QApplication instance shared by all UI tests in the session."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def synthetic_pdf(tmp_path: Path) -> Path:
    """A one-page vector PDF (rect + text) at a known page size in points."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)  # points -> 400x300 at 72 DPI
    page.draw_rect(fitz.Rect(20, 20, 380, 280), color=(0, 0, 0), width=2)
    page.insert_text((50, 150), "ORIGINAL ARTWORK", fontsize=24)
    path = tmp_path / "synthetic.pdf"
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def synthetic_png(tmp_path: Path) -> Path:
    """A small RGB PNG with a distinct, deterministic pixel pattern."""
    array = np.full((150, 200, 3), 255, dtype=np.uint8)
    array[40:110, 60:140] = (10, 20, 30)
    path = tmp_path / "synthetic.png"
    Image.fromarray(array).save(path)
    return path


@pytest.fixture
def corrupt_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"not actually a pdf")
    return path


@pytest.fixture
def corrupt_png(tmp_path: Path) -> Path:
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"not actually a png")
    return path
