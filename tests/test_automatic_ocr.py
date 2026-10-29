"""Automatic mode's OCR findings: compared with every Automatic comparison, kept hidden, shown at once by the results
page's toggle (`app.engine.word_strategy`, `app.engine.worker_process`, `app.ui.results_view`); and the findings list
offering only the kinds a mode finds."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pymupdf as fitz

from app.config import AppConfig
from app.engine.worker_process import ComparisonRequest, execute_request
from app.models.diff import SOURCE_OCR
from app.models.job import ComparisonMode
from test_leaflet_options import _both_ways_outcome, _loaded_window, _outcome


def _pdfs(tmp_path: Path, changed: str) -> tuple[Path, Path]:
    """Two one-page PDFs with live text, the second with one line changed."""
    paths = []
    for name, dose in (("a.pdf", "Take one tablet twice daily"), ("b.pdf", changed)):
        document = fitz.open()
        page = document.new_page(width=420, height=300)
        page.draw_rect(fitz.Rect(20, 20, 400, 280), color=(0, 0, 0), width=1.5)
        page.insert_text((40, 70), "PARACETAMOL TABLETS 500 mg", fontsize=16)
        page.insert_text((40, 120), dose, fontsize=12)
        page.insert_text((40, 160), "Store below 25 C in a dry place", fontsize=12)
        page.insert_text((40, 200), "Keep out of the reach of children", fontsize=12)
        document.save(tmp_path / name)
        document.close()
        paths.append(tmp_path / name)
    return paths[0], paths[1]


def test_automatic_mode_compares_the_words_too_and_keeps_them_hidden(tmp_path):
    original, suspect = _pdfs(tmp_path, "Take one tablett twice daily")
    outcome = execute_request(ComparisonRequest(original, suspect, ComparisonMode.AUTOMATIC, AppConfig()))
    word_findings = [c for c in outcome.all_clusters if SOURCE_OCR in c.sources]
    assert word_findings, "the changed words are found by reading them"
    assert not any(SOURCE_OCR in c.sources for c in outcome.clusters)  # ...but not shown until asked for
    assert outcome.clusters, "the pixel findings are shown as ever"
    assert outcome.text_compared is False and outcome.graphics_compared is True
    assert outcome.compare_mask is None  # never sent back to the window


def test_the_words_can_be_left_uncompared(tmp_path):
    original, suspect = _pdfs(tmp_path, "Take one tablett twice daily")
    config = AppConfig()
    config = replace(config, detection=replace(config.detection, automatic_ocr=False))
    outcome = execute_request(ComparisonRequest(original, suspect, ComparisonMode.AUTOMATIC, config))
    assert not any(SOURCE_OCR in c.sources for c in outcome.all_clusters or outcome.clusters)


def _automatic_both_ways():
    both = _both_ways_outcome()  # one finding by reading, one in the pixels
    return replace(both, mode=ComparisonMode.AUTOMATIC, clusters=[c for c in both.all_clusters if "pixels" in c.sources],
                   text_compared=False, graphics_compared=True)


def test_automatic_results_open_on_pixel_findings_and_the_ocr_toggle_shows_the_words_at_once(qapp, tmp_path):
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.AUTOMATIC)
    window.show()
    window._on_compare()
    window._on_completed(_automatic_both_ways())
    results = window.results_view
    assert results.leaflet_text_check.isVisible() and not results.leaflet_text_check.isChecked()
    assert results.leaflet_visual_check.isVisible() and results.leaflet_visual_check.isChecked()
    assert [c.detail for c in results.panel.visible_clusters()] == ["by pixels"]

    results.leaflet_text_check.setChecked(True)  # OCR findings on
    assert sorted(c.detail for c in results.panel.visible_clusters()) == ["by ocr", "by pixels"]
    results.leaflet_visual_check.setChecked(False)  # the words alone
    assert [c.detail for c in results.panel.visible_clusters()] == ["by ocr"]
    assert len(jobs.requests) == 1  # instant: nothing compared again
    window.close()


def test_the_toggles_are_not_offered_in_modes_that_find_one_way(qapp, tmp_path):
    window, _jobs = _loaded_window(qapp, tmp_path, ComparisonMode.AUTOMATIC)
    window.show()
    results = window.results_view
    window.stack.setCurrentWidget(results)
    results.show_outcome(_outcome(ComparisonMode.BARCODE), "a vs b")
    assert not results.leaflet_text_check.isVisible()
    window.close()


def test_a_leaflets_findings_list_offers_no_barcode_or_braille(qapp, tmp_path):
    window, _jobs = _loaded_window(qapp, tmp_path, ComparisonMode.LEAFLET)
    results = window.results_view
    combo = results.panel.category_combo

    def labels():
        return [combo.itemText(i) for i in range(combo.count())]

    results.show_outcome(_outcome(ComparisonMode.LEAFLET), "a vs b")
    assert "Barcode" not in labels() and "Braille" not in labels()
    assert {"Text", "Colour", "Graphics", "OCR findings", "Pixel findings"} <= set(labels())

    combo.setCurrentIndex(labels().index("Colour"))
    results.show_outcome(_outcome(ComparisonMode.AUTOMATIC), "a vs b")
    assert "Barcode" in labels() and "Braille" in labels()
    assert combo.currentText() == "Colour"  # a pick still offered is kept
    window.close()


# ---- Automatic's OCR findings are spellings only -------------------------------------------------------------------------


def _word_change(was: str, now: str, subtype=None):
    from app.models.diff import BoundingBox, DiffCandidate, DiffSubtype, DiffType

    return DiffCandidate(
        BoundingBox(0, 0, 40, 12), DiffType.TEXT, subtype or DiffSubtype.CHANGED, 1.0, f"'{was}' -> '{now}'",
        source=SOURCE_OCR, was_text=was, now_text=now,
    )


def test_only_words_spelt_differently_are_kept():
    from app.engine.word_strategy import spelling_changes
    from app.models.diff import DiffSubtype

    kept = spelling_changes([
        _word_change("from", "fom"),  # a spelling
        _word_change("Keep out of reach", "Keep out of raech"),  # one misspelt word in a line
        _word_change("colour", "color"),
        _word_change("24", "36"),  # a number
        _word_change("25 mg", "50 mg"),  # a dose
        _word_change("SM4ATCAAINS952501", "SM4ASCAAINS952501"),  # a code
        _word_change("tablet", "capsule"),  # another word, not a spelling
        _word_change("Zimbabwe:", "Zimbabwe"),  # punctuation only
        _word_change("Read the leaflet", "", DiffSubtype.REMOVED),  # text taken away
        _word_change("", "RETAIL", DiffSubtype.ADDED),  # text added
        _word_change("one tablet", "one tablet daily"),  # a word added to a line
    ])
    assert [c.detail for c in kept] == [
        "Spelling: 'from' spelt 'fom'",
        "Spelling: 'reach' spelt 'raech'",
        "Spelling: 'colour' spelt 'color'",
    ]


def test_automatic_ocr_findings_are_the_misspelt_words_alone(tmp_path):
    original, suspect = _pdfs(tmp_path, "Take one tablett twice daily")  # 'tablet' misspelt
    outcome = execute_request(ComparisonRequest(original, suspect, ComparisonMode.AUTOMATIC, AppConfig()))
    spelt = [c.detail for c in outcome.all_clusters if SOURCE_OCR in c.sources]
    assert spelt == ["Spelling: 'tablet' spelt 'tablett'"]


def test_automatic_ocr_does_not_report_changed_numbers_or_new_words(tmp_path):
    original, suspect = _pdfs(tmp_path, "Take 2 tablets twice daily after food")  # a number and words, no spelling
    outcome = execute_request(ComparisonRequest(original, suspect, ComparisonMode.AUTOMATIC, AppConfig()))
    assert not [c for c in outcome.all_clusters if SOURCE_OCR in c.sources]
