"""Leaflet mode through the request pipeline (`app.engine.worker_process.execute_request`): loading a multi-page
file from its path and comparing every page."""

from __future__ import annotations

from pathlib import Path

import leaflet_scenes as scenes
from PIL import Image

from app.config import AppConfig
from app.engine.worker_process import ComparisonRequest, execute_request
from app.models.job import ComparisonMode


def _pdf(path: Path, pages: list) -> None:
    images = [Image.fromarray(page) for page in pages]
    images[0].save(path, save_all=True, append_images=images[1:])


def test_leaflet_mode_compares_every_page_of_a_multi_page_pdf_from_its_path(tmp_path):
    original_pages = [scenes.page("indications"), scenes.page("dosage"), scenes.page("warnings")]
    suspect_pages = [scenes.page("indications"), scenes.page("dosage", scenes.edit_first_line), scenes.page("warnings")]
    original, suspect = tmp_path / "original.pdf", tmp_path / "suspect.pdf"
    _pdf(original, original_pages)
    _pdf(suspect, suspect_pages)

    outcome = execute_request(ComparisonRequest(original, suspect, ComparisonMode.LEAFLET, AppConfig()))

    assert outcome.mode is ComparisonMode.LEAFLET
    assert outcome.clusters
    assert all(cluster.detail.startswith("Page 2:") for cluster in outcome.clusters)


def test_leaflet_mode_ignores_any_geometry_the_request_happens_to_carry(tmp_path):
    """A crop/lift from a previous mode left on the request must not be applied to whole leaflet pages."""
    from app.models.isolation import IsolationGeometry, NormalizedRect

    pages = [scenes.page("indications"), scenes.page("dosage")]
    original, suspect = tmp_path / "original.pdf", tmp_path / "suspect.pdf"
    _pdf(original, pages)
    _pdf(suspect, pages)
    stray_geometry = IsolationGeometry(boundary=NormalizedRect(0.1, 0.1, 0.5, 0.5))

    outcome = execute_request(
        ComparisonRequest(
            original, suspect, ComparisonMode.LEAFLET, AppConfig(), original_geometry=stray_geometry, suspect_geometry=stray_geometry
        )
    )

    assert outcome.clusters == []  # identical pages, whole page compared: nothing outside a stray crop is missed
