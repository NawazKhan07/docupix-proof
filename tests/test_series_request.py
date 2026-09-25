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


def test_series_mode_does_not_auto_find_artwork_in_the_original_the_way_automatic_mode_does(tmp_path, monkeypatch):
    """Series has its own way of settling what the original's artwork is (the reviewer's own crop/lift, or the
    whole page): `find_artwork` (Automatic mode's own-page auto-detection) must not also run for it."""
    import app.engine.worker_process as worker_process

    original, suspect, geometry = _files(tmp_path)
    called = []
    monkeypatch.setattr(worker_process, "find_artwork", lambda *a, **k: called.append(1) or (_ for _ in ()).throw(AssertionError))

    execute_request(ComparisonRequest(original, suspect, ComparisonMode.SERIES, AppConfig(), original_geometry=geometry))

    assert not called
