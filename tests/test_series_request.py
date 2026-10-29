"""Series mode through the request pipeline (`app.engine.worker_process.execute_request`): loading files from
paths, finding the repeats, and reporting a friendly error when there are none."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import series_scenes as scenes
from PIL import Image

from app.config import AppConfig
from app.engine.series import NoRepeatsFoundError
from app.engine.worker_process import ComparisonRequest, execute_request
from app.models.isolation import IsolationGeometry, NormalizedRect
from app.models.job import ComparisonMode


def _files(tmp_path: Path) -> tuple[Path, Path, IsolationGeometry]:
    page, box = scenes.original_page()
    sheet = scenes.sheet([None, "colour", None, "graphic"])
    original, suspect = tmp_path / "original.png", tmp_path / "suspect.png"
    Image.fromarray(page).save(original)
    Image.fromarray(sheet).save(suspect)
    x0, y0, x1, y1 = box
    width, height = page.shape[1], page.shape[0]
    geometry = IsolationGeometry(boundary=NormalizedRect(x0 / width, y0 / height, x1 / width, y1 / height))
    return original, suspect, geometry


def test_series_mode_finds_and_compares_every_repeat_from_the_files_paths(tmp_path):
    original, suspect, geometry = _files(tmp_path)

    outcome = execute_request(
        ComparisonRequest(original, suspect, ComparisonMode.SERIES, AppConfig(), original_geometry=geometry)
    )

    assert outcome.mode is ComparisonMode.SERIES
    assert any("4 repeats" in note for note in outcome.notes)
    categories = {cluster.category for cluster in outcome.clusters}
    assert categories == {"colour", "graphics"}


def test_a_suspect_with_no_repeat_of_the_artwork_says_so_plainly(tmp_path):
    original, suspect, geometry = _files(tmp_path)
    unrelated = np.full((900, 700, 3), (40, 90, 150), np.uint8)
    Image.fromarray(unrelated).save(suspect)

    with pytest.raises(NoRepeatsFoundError):
        execute_request(ComparisonRequest(original, suspect, ComparisonMode.SERIES, AppConfig(), original_geometry=geometry))


def test_series_mode_does_not_auto_find_artwork_when_the_reviewer_already_cropped_or_lifted_one(tmp_path, monkeypatch):
    """A geometry the reviewer already set (a crop or a lift) is what settles the original's artwork: auto-finding
    must not run again and second-guess it."""
    import app.engine.worker_process as worker_process

    original, suspect, geometry = _files(tmp_path)
    called = []
    monkeypatch.setattr(worker_process, "find_artwork", lambda *a, **k: called.append(1) or (_ for _ in ()).throw(AssertionError))

    execute_request(ComparisonRequest(original, suspect, ComparisonMode.SERIES, AppConfig(), original_geometry=geometry))

    assert not called


def test_series_mode_auto_finds_the_original_artwork_when_nothing_was_lifted(tmp_path):
    """Like Automatic mode: if the reviewer drops in a whole sheet without cropping or lifting, the original's
    artwork is still found and trimmed to automatically, rather than comparing the whole, uncropped page."""
    page, box = scenes.original_page()
    sheet = scenes.sheet([None, "colour", None, "graphic"])
    original, suspect = tmp_path / "original.png", tmp_path / "suspect.png"
    Image.fromarray(page).save(original)
    Image.fromarray(sheet).save(suspect)

    outcome = execute_request(
        ComparisonRequest(original, suspect, ComparisonMode.SERIES, AppConfig())  # no original_geometry given
    )

    assert outcome.mode is ComparisonMode.SERIES
    assert any("4 repeats" in note for note in outcome.notes)


def test_series_mode_compares_the_page_the_sheets_copies_are_on(tmp_path):
    """An original of two pages -- a mock-up drawing first, the artwork second, the way a design house sends them: the
    page compared is the one holding what the sheet repeats, not the one that merely looks like a tall sheet."""
    import cv2
    import pymupdf as fitz

    drawing = np.full((900, 400, 3), 255, np.uint8)
    for y in range(60, 860, 120):  # a mock-up: a few boxes and lines, tall like a reel
        cv2.rectangle(drawing, (40, y), (360, y + 90), (0, 0, 0), 2)
    artwork_page, _box = scenes.original_page()
    reel = np.vstack([np.vstack([np.full((30, scenes.CELL_W, 3), scenes.SHEET_BG, np.uint8), scenes.artwork(None)]) for _ in range(5)])

    def pdf(path, pictures):
        document = fitz.open()
        for picture in pictures:
            height, width = picture.shape[:2]
            page = document.new_page(width=width * 72 / 300, height=height * 72 / 300)
            ok, encoded = cv2.imencode(".png", picture[:, :, ::-1])
            page.insert_image(page.rect, stream=encoded.tobytes())
        document.save(path)
        document.close()

    original, suspect = tmp_path / "original.pdf", tmp_path / "reel.pdf"
    pdf(original, [drawing, artwork_page])
    pdf(suspect, [reel])
    outcome = execute_request(ComparisonRequest(original, suspect, ComparisonMode.SERIES, AppConfig()))
    assert outcome.original_page == 1
    assert any("5 repeats" in note for note in outcome.notes)
