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

    from app.ui.theme import apply_theme

    app = QApplication.instance() or QApplication([])
    apply_theme(app)
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


def _review_pair() -> tuple[np.ndarray, np.ndarray]:
    """An original and a suspect with three known changes, rotated and shifted.

    The red block is recoloured green and the blue roundel is removed, so the
    pipeline yields two colour findings and a graphics finding.
    """
    import cv2

    original = np.full((700, 500, 3), 255, dtype=np.uint8)
    cv2.rectangle(original, (30, 30), (470, 670), (0, 0, 0), 3)
    cv2.putText(original, "PARACETAMOL", (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (10, 10, 10), 3)
    cv2.putText(original, "500 mg Tablets BP", (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 10, 10), 2)
    cv2.rectangle(original, (60, 210), (260, 350), (200, 30, 40), -1)
    cv2.circle(original, (380, 280), 65, (30, 90, 200), -1)
    cv2.putText(original, "Store below 25 C", (60, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 2)
    for index in range(24):
        cv2.line(original, (70 + index * 12, 525), (70 + index * 12, 625), (0, 0, 0), 4)

    changed = original.copy()
    cv2.rectangle(changed, (60, 210), (260, 350), (40, 160, 60), -1)
    cv2.circle(changed, (380, 280), 65, (255, 255, 255), -1)
    matrix = cv2.getRotationMatrix2D((250.0, 350.0), 1.2, 1.0)
    matrix[0, 2] += 7
    matrix[1, 2] += 4
    suspect = cv2.warpAffine(changed, matrix, (500, 700), borderValue=(255, 255, 255))
    return original, suspect


@pytest.fixture
def review_pair() -> tuple[np.ndarray, np.ndarray]:
    return _review_pair()


@pytest.fixture(scope="session")
def _outcome_template():
    """The pipeline run once for the whole session; tests get copies of it."""
    from app.config import AppConfig
    from app.engine.worker import run_comparison
    from app.models.job import ComparisonMode

    original, suspect = _review_pair()
    result = run_comparison(original, suspect, ComparisonMode.AUTOMATIC, AppConfig())
    assert result.clusters, "the fixture pair is expected to produce findings"
    return result


@pytest.fixture
def outcome(_outcome_template):
    """A real ComparisonOutcome, fresh per test: tests ignore/restore findings,
    so they must not share one object, but they need not each pay for the pipeline."""
    import copy

    return copy.deepcopy(_outcome_template)
