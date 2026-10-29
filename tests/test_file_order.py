"""Which file goes in which slot.

Scan vs Digital compares a digital file (the Original) with a scan of it (the Suspect): the digital file's own words are
its text, a scanner's streaks are evened out of the scan, the Braille and barcodes are found on the digital file. Given
the other way round, the files are swapped before comparing -- on the setup page too -- and the results say so. Where
there is no telling (two scans, two digital files) nothing is swapped. Series mode, given the sheet as the Original,
says the files look the other way round.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import fitz
import numpy as np
import series_scenes
from PIL import Image
from test_leaflet_options import _RecordingJobs

from app.config import AppConfig
from app.engine.file_loader import load_artwork
from app.engine.series import _the_other_way_round
from app.models.job import ComparisonMode
from app.ui.main_window import MainWindow


def _artwork() -> np.ndarray:
    page = np.full((500, 700, 3), 255, np.uint8)
    cv2.rectangle(page, (30, 30), (670, 470), (0, 0, 0), 3)
    cv2.putText(page, "PARACETAMOL 500 mg", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 3, cv2.LINE_AA)
    return page


def _digital_pdf(path: Path) -> Path:
    """A PDF drawn digitally: its type and its outlines vector, a small photograph in it."""
    doc = fitz.open()
    page = doc.new_page(width=500, height=360)
    page.draw_rect(fitz.Rect(20, 20, 480, 340), color=(0, 0, 0), width=2)
    page.insert_text((50, 150), "PARACETAMOL 500 mg", fontsize=24)
    photo = path.with_suffix(".photo.png")
    Image.fromarray(np.full((60, 80, 3), 120, np.uint8)).save(photo)
    page.insert_image(fitz.Rect(380, 40, 460, 100), filename=str(photo))
    doc.save(path)
    doc.close()
    return path


def _scanned_pdf(path: Path) -> Path:
    """A scanner's PDF: the page one picture, its read-out text laid invisibly over it."""
    picture = path.with_suffix(".scan.png")
    Image.fromarray(_artwork()).save(picture)
    doc = fitz.open()
    page = doc.new_page(width=500, height=360)
    page.insert_image(page.rect, filename=str(picture))
    page.insert_text((50, 150), "PARACETAMOL 500 mg", fontsize=24, render_mode=3)  # (invisible, as a scanner's OCR)
    doc.save(path)
    doc.close()
    return path


def test_a_scan_is_told_from_a_digital_file(tmp_path):
    photo = tmp_path / "photo.png"
    Image.fromarray(_artwork()).save(photo)
    cases = {_digital_pdf(tmp_path / "digital.pdf"): False, _scanned_pdf(tmp_path / "scan.pdf"): True, photo: True}
    for path, scanned in cases.items():
        artwork = load_artwork(path)
        try:
            assert artwork.looks_scanned() is scanned, path.name
        finally:
            artwork.close()


def _window(tmp_path, original: Path, suspect: Path, mode: ComparisonMode):
    jobs = _RecordingJobs()
    window = MainWindow(jobs=jobs)
    window.upload_panel.original_slot.load_path(original)
    window.upload_panel.suspect_slot.load_path(suspect)
    window.mode_selector.set_mode(mode)
    return window, jobs


def test_a_scan_given_as_the_original_is_swapped_with_the_digital_file(qapp, tmp_path):
    digital, scan = _digital_pdf(tmp_path / "digital.pdf"), _scanned_pdf(tmp_path / "scan.pdf")
    window, jobs = _window(tmp_path, scan, digital, ComparisonMode.SCAN_VS_DIGITAL)

    window._on_compare()

    request = jobs.requests[-1]
    assert (request.original_path, request.suspect_path) == (digital, scan)
    assert window.upload_panel.original.path == digital and window.upload_panel.suspect.path == scan  # (shown so too)
    assert "other way round" in window._swap_note and "scan.pdf" in window._swap_note


def test_files_given_the_right_way_round_are_left_as_they_are(qapp, tmp_path):
    digital, scan = _digital_pdf(tmp_path / "digital.pdf"), _scanned_pdf(tmp_path / "scan.pdf")
    window, jobs = _window(tmp_path, digital, scan, ComparisonMode.SCAN_VS_DIGITAL)

    window._on_compare()

    assert (jobs.requests[-1].original_path, jobs.requests[-1].suspect_path) == (digital, scan)
    assert window._swap_note == ""


def test_nothing_is_swapped_where_there_is_no_telling_or_in_another_mode(qapp, tmp_path):
    digital, scan = _digital_pdf(tmp_path / "digital.pdf"), _scanned_pdf(tmp_path / "scan.pdf")
    other_scan = _scanned_pdf(tmp_path / "other.pdf")
    for original, suspect, mode in (
        (scan, other_scan, ComparisonMode.SCAN_VS_DIGITAL),  # two scans
        (scan, digital, ComparisonMode.AUTOMATIC),  # Automatic: which is approved is the reviewer's to say
    ):
        window, jobs = _window(tmp_path, original, suspect, mode)
        window._on_compare()
        assert (jobs.requests[-1].original_path, jobs.requests[-1].suspect_path) == (original, suspect)


def test_swapping_carries_each_files_page_choice_with_it(qapp, tmp_path):
    digital, scan = _digital_pdf(tmp_path / "digital.pdf"), _scanned_pdf(tmp_path / "scan.pdf")
    window, _jobs = _window(tmp_path, scan, digital, ComparisonMode.SCAN_VS_DIGITAL)
    panel = window.upload_panel

    panel.swap()

    assert panel.original.path == digital and panel.suspect.path == scan
    assert "digital.pdf" in panel.original_slot.detail_label.text() and "scan.pdf" in panel.suspect_slot.detail_label.text()
    assert panel.original_slot.isolate_button.isEnabled() and panel.suspect_slot.isolate_button.isEnabled()


def test_a_series_given_the_other_way_round_is_told_so():
    page, _box = series_scenes.original_page()  # the one artwork, on its page
    sheet = series_scenes.sheet([None] * 6)  # six repeats of it
    # the sheet given as the Original, the one artwork as the Suspect
    assert _the_other_way_round(sheet, page, AppConfig())
    assert not _the_other_way_round(page, series_scenes.artwork(), AppConfig())  # (nothing repeats either way)
