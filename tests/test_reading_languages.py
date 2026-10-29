"""The reading languages: found on the files while they are compared (every mode, nothing to choose), and any given
languages carried into every reading of the comparison."""

from __future__ import annotations

from app.models.job import ComparisonMode
from app.models.languages import joined, languages_text

from test_leaflet_options import _loaded_window

INSTALLED = frozenset({"eng", "ara", "fra", "vie", "msa", "chi_sim", "tha", "osd", "khm"})


def test_a_choice_of_languages_reads_as_words_and_as_the_readers_code():
    assert languages_text(("eng",)) == "English"
    assert languages_text(("eng", "ara", "fra")) == "English, Arabic and French"
    assert joined(("ara", "eng", "fra")) == "eng+ara+fra"  # English first: the surest reader of numbers and Latin text
    assert joined(()) == "eng"


def test_every_mode_reads_in_the_languages_found_on_the_files(qapp, tmp_path):
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.AUTOMATIC)
    window.show()
    assert not hasattr(window, "language_row")  # nothing to choose
    for mode in (ComparisonMode.AUTOMATIC, ComparisonMode.SCAN_VS_DIGITAL, ComparisonMode.SERIES, ComparisonMode.LEAFLET):
        window.mode_selector.set_mode(mode)
        jobs.is_busy = False
        window._on_compare()
        config = jobs.requests[-1].config
        assert (config.leaflet.language if mode is ComparisonMode.LEAFLET else config.detection.ocr_language) == "auto"
    window.close()


def test_a_leaflet_read_from_its_pictures_is_read_in_every_language_chosen(tmp_path, monkeypatch):
    """Where a leaflet's text is not live (outlined, closed, scanned, a picture), every reading of its pages asks the
    reader for the languages chosen -- not English alone."""
    from dataclasses import replace

    import cv2
    import numpy as np

    from app.config import AppConfig
    from app.engine.detectors import text_diff
    from app.engine.ocr_locator import find_tesseract
    from app.engine.worker_process import ComparisonRequest, execute_request

    if find_tesseract() is None:
        import pytest

        pytest.skip("Tesseract is not installed")
    import pytesseract

    monkeypatch.setattr(pytesseract.pytesseract, "tesseract_cmd", str(find_tesseract()))  # (for this test only)
    asked = []
    run, data = text_diff._ocr_run, text_diff._ocr_data

    def spy_run(pictures, page_segmentation, lang):
        asked.append(lang)
        return run(pictures, page_segmentation, "eng")  # (read as English here: the test machine may lack Arabic)

    def spy_data(image, page_segmentation=None, lang=None):
        asked.append(lang)
        return data(image, page_segmentation, "eng")

    monkeypatch.setattr(text_diff, "_ocr_run", spy_run)
    monkeypatch.setattr(text_diff, "_ocr_data", spy_data)
    monkeypatch.setattr("app.engine.leaflet.usable_language", lambda requested: (requested, True))
    paths = []
    for name, dose in (("a.png", "500"), ("b.png", "250")):
        page = np.full((900, 700, 3), 255, np.uint8)
        for row in range(8):
            cv2.putText(page, f"Paracetamol {dose} mg tablet line {row}", (40, 80 + row * 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
        cv2.imwrite(str(tmp_path / name), page)
        paths.append(tmp_path / name)
    config = AppConfig()
    config = replace(config, leaflet=replace(config.leaflet, language="eng+ara"))

    execute_request(ComparisonRequest(paths[0], paths[1], ComparisonMode.LEAFLET, config))

    assert asked and all(lang == "eng+ara" for lang in asked), set(asked)
