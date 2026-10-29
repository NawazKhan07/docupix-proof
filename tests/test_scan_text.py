"""Text in a scan compared with the digital file it was made from.

A scan is read less surely than a digital file, so Scan vs Digital does not diff two whole-page readings (a
changed digit reshuffled the reading order and one identical layout gave 23 false changes on a real pair). It
anchors on the digital file's lines, reads the same window again in both files, and ranks a difference whose ink
matches as a likely misread. Tests that need the real engine are skipped where Tesseract is not installed.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytesseract
import pytest
from scan_simulator import LIGHT, TYPICAL, simulate_scan

from app.config import AppConfig, DetectionConfig
from app.engine.detectors.text_diff import LIKELY_MISREAD_CONFIDENCE
from app.engine.detectors import text_diff
from app.engine.detectors.text_diff import (
    OcrWord,
    TextScanResult,
    canonical,
    detect_scan,
    ink_agreement,
    _findings,
    _leftovers,
    _line_differences,
    _marked_if_misread,
    _same_word,
    _trustworthy,
)
from app.engine.ocr_locator import find_tesseract
from app.engine.worker import run_comparison
from app.models.diff import BoundingBox, DiffCandidate, DiffSubtype, DiffType
from app.models.job import ComparisonMode

CONFIG = DetectionConfig()

tesseract_required = pytest.mark.skipif(find_tesseract() is None, reason="Tesseract OCR binary is not installed")


@pytest.fixture
def real_ocr(monkeypatch):
    monkeypatch.setattr(pytesseract.pytesseract, "tesseract_cmd", str(find_tesseract()))


def _word(text, x=0, y=0, confidence=90.0, width=None) -> OcrWord:
    width = width or 12 * len(text)
    return OcrWord(text, BoundingBox(x, y, x + width, y + 20), confidence)


def _line(*texts, confidence=90.0) -> list[OcrWord]:
    """Words laid out one after another on a line, a few pixels apart."""
    words, x = [], 0
    for text in texts:
        word = _word(text, x=x, confidence=confidence)
        words.append(word)
        x = word.bbox.x1 + 18
    return words


# ---- what counts as the same word ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b",
    [
        ("intravenous", "Intravenous"),  # case
        ("children.", "children,"),  # punctuation
        ("ml", "mi"),  # an l read as an i
        ("K-27,", "K27"),
        ("100", "1OO"),  # o and 0
    ],
)
def test_readings_of_the_same_word_that_a_scan_muddles_are_the_same_word(a, b):
    assert _same_word(_word(a), _word(b))


@pytest.mark.parametrize(
    "a,b",
    [
        ("500mg", "250mg"),  # the difference this exists to find
        ("50", "60"),
        ("mg", "mcg"),  # a short word: one letter is a third of it
        ("Store", "Stone"),
        ("tablets", "capsules"),
        ("SERUMS", "SERUNS"),  # one letter: a smudge or a change -- the ink decides how it is ranked, not the matching
    ],
)
def test_a_real_difference_is_never_matched_loosely(a, b):
    assert not _same_word(_word(a), _word(b))


def test_a_number_must_match_exactly_however_close():
    assert not _same_word(_word("421501"), _word("421504"))
    assert not _same_word(_word("10.5"), _word("10.6"))


def test_the_canonical_form_ignores_case_marks_and_the_l_i_1_and_o_0_confusions():
    assert canonical("Il1") == "iii" and canonical("O0o") == "ooo" and canonical("U.S.P.,") == "usp" and canonical("|--") == ""


def test_what_a_scan_reads_as_marks_or_shakily_is_not_a_word():
    words = [_word("|"), _word("--"), _word("a", x=900, confidence=99), _word("of", confidence=60), _word("mg", confidence=90), _word("Store", confidence=60)]
    kept = [w.text for w in _trustworthy(words, CONFIG)]
    assert kept == ["mg", "Store"]  # a very short word must be read surely; "of" at 60 is not; a lone letter far off is a speck


def test_a_lone_digit_or_letter_read_surely_in_a_phrase_is_a_word():
    """'7 x 7 mm' -> '9 x 9 mm': the digits that changed are words of one character, kept where they stand in a phrase."""
    line = [_word("7", x=0, confidence=95), _word("x", x=20, confidence=95), _word("7", x=40, confidence=95), _word("mm)", x=60, confidence=95)]
    assert [w.text for w in _trustworthy(line, CONFIG)] == ["7", "x", "7", "mm)"]
    alone = [_word("0", x=0, confidence=95), _word("Store", x=900)]  # (a '0' read off a picture, far from any word)
    assert [w.text for w in _trustworthy(alone, CONFIG)] == ["Store"]


# ---- a line's differences --------------------------------------------------------------------------------------------


def test_identical_readings_have_no_differences():
    assert _line_differences(_line("Each", "tablet", "contains", "500mg"), _line("Each", "tablet", "contains", "500mg")) == []


def test_a_muddled_reading_of_the_same_line_has_no_differences():
    original = _line("intravenous", "infusion", "children.", "ml")
    suspect = _line("Intravenous", "infusion", "children,", "mi")
    assert _line_differences(original, suspect) == []


def test_a_changed_number_is_reported_with_both_values():
    (found,) = _line_differences(_line("Each", "contains", "500mg"), _line("Each", "contains", "250mg"))
    assert found.subtype is DiffSubtype.CHANGED and found.type is DiffType.TEXT
    assert "500mg" in found.detail and "250mg" in found.detail


def test_a_purely_numeric_change_is_a_number_change():
    (found,) = _line_differences(_line("Batch", "50"), _line("Batch", "60"))
    assert found.type is DiffType.NUMBER


def test_words_only_in_the_digital_reading_are_removed_and_only_in_the_scan_added():
    (removed,) = _line_differences(_line("Keep", "out", "of", "reach"), _line("Keep", "reach"))
    assert removed.subtype is DiffSubtype.REMOVED and "out" in removed.detail
    (added,) = _line_differences(_line("Keep", "reach"), _line("Keep", "well", "reach"))
    assert added.subtype is DiffSubtype.ADDED and "well" in added.detail


def test_leftovers_pair_each_word_with_the_first_unused_match():
    gone, new = _leftovers(_line("a1", "Sodium", "Sodium"), _line("sodium", "extra"))
    assert [w.text for w in gone] == ["a1", "Sodium"] and [w.text for w in new] == ["extra"]


def test_scattered_leftovers_are_separate_findings_not_one_huge_change():
    gone = [_word("alpha", x=0), _word("bravo", x=2000)]
    new = [_word("charlie", x=1000, y=900)]
    findings = _findings(gone, new)
    assert len(findings) == 3 and {f.subtype for f in findings} == {DiffSubtype.REMOVED, DiffSubtype.ADDED}


# ---- ink agreement: the same marks read differently ---------------------------------------------------------------------


def _ink_page(boxes, size=(200, 400)) -> np.ndarray:
    page = np.full(size, 255, np.uint8)
    for x0, y0, x1, y1 in boxes:
        page[y0:y1, x0:x1] = 0
    return page


def test_the_same_marks_agree_even_when_the_scans_are_fatter():
    thin, fat = _ink_page([(50, 50, 52, 90)]), _ink_page([(49, 49, 53, 91)])
    assert ink_agreement(thin, fat, BoundingBox(40, 40, 70, 100)) >= 0.96


def test_different_marks_do_not_agree():
    assert ink_agreement(_ink_page([(50, 50, 52, 90)]), _ink_page([(50, 50, 90, 52)]), BoundingBox(40, 40, 100, 100)) < 0.5


def test_where_one_file_has_no_ink_there_is_no_agreement():
    assert ink_agreement(_ink_page([(50, 50, 52, 90)]), _ink_page([]), BoundingBox(40, 40, 70, 100)) == 0.0


def test_a_difference_over_the_same_ink_is_marked_a_likely_misread_and_listed_apart():
    marks = _ink_page([(50, 50, 52, 90), (60, 50, 62, 90)])
    finding = DiffCandidate(BoundingBox(40, 40, 80, 100), DiffType.TEXT, DiffSubtype.CHANGED, 0.9, "'Plot' -> 'Phot'")

    marked = _marked_if_misread(finding, marks, marks.copy(), CONFIG)

    assert marked.confidence == LIKELY_MISREAD_CONFIDENCE and "likely a misread" in marked.detail
    assert marked.uncertain and not finding.uncertain  # the same finding, read surely, is not marked


def test_a_difference_over_different_ink_is_left_alone():
    finding = DiffCandidate(BoundingBox(40, 40, 100, 100), DiffType.NUMBER, DiffSubtype.CHANGED, 0.9, "'500' -> '250'")
    kept = _marked_if_misread(finding, _ink_page([(50, 50, 52, 90)]), _ink_page([(50, 50, 90, 52)]), CONFIG)
    assert kept is finding and not kept.uncertain


# ---- the worker ----------------------------------------------------------------------------------------------------------


def _leaflet(lines=("PARACETAMOL 500 mg TABLETS", "Each tablet contains 500 mg", "Store below 25 C in a dry place")) -> np.ndarray:
    page = np.full((900, 1300, 3), 255, np.uint8)
    cv2.rectangle(page, (60, 60), (1240, 840), (226, 238, 246), -1)
    for index, text in enumerate(lines):
        cv2.putText(page, text, (110, 200 + index * 130), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (20, 20, 20), 3, cv2.LINE_AA)
    return page


def _text_candidates(outcome) -> list[DiffCandidate]:
    return [c for c in outcome.candidates if c.type in (DiffType.TEXT, DiffType.NUMBER)]


def _serious(outcome) -> list[DiffCandidate]:
    return [c for c in _text_candidates(outcome) if not c.uncertain]


def test_unreadable_lines_are_reported_to_the_reviewer_not_as_changes(monkeypatch):
    page = _leaflet()
    monkeypatch.setattr(text_diff, "detect_scan", lambda *a, **k: TextScanResult([], unreadable_lines=3))
    outcome = run_comparison(page, simulate_scan(page, LIGHT), ComparisonMode.SCAN_VS_DIGITAL, AppConfig())
    assert any("3 line(s) of small print" in warning and "too soft to read" in warning for warning in outcome.warnings)


def test_text_comparison_is_unavailable_without_tesseract_and_the_reviewer_is_told(monkeypatch):
    monkeypatch.setattr(pytesseract.pytesseract, "tesseract_cmd", "no-such-tesseract-executable")
    page = _leaflet()
    outcome = run_comparison(page, simulate_scan(page, LIGHT), ComparisonMode.SCAN_VS_DIGITAL, AppConfig())
    assert any("Tesseract" in warning for warning in outcome.warnings) and _text_candidates(outcome) == []


@tesseract_required
@pytest.mark.parametrize("profile", [LIGHT, TYPICAL], ids=["light", "typical"])
def test_a_scan_of_identical_text_has_no_serious_text_findings(real_ocr, profile):
    page = _leaflet()
    outcome = run_comparison(page, simulate_scan(page, profile), ComparisonMode.SCAN_VS_DIGITAL, AppConfig())
    assert _serious(outcome) == [], [c.detail for c in _serious(outcome)]


@tesseract_required
@pytest.mark.parametrize("profile", [LIGHT, TYPICAL], ids=["light", "typical"])
def test_a_changed_number_in_the_scan_is_a_serious_finding_with_both_values(real_ocr, profile):
    original = _leaflet()
    changed = _leaflet(("PARACETAMOL 500 mg TABLETS", "Each tablet contains 250 mg", "Store below 25 C in a dry place"))
    outcome = run_comparison(original, simulate_scan(changed, profile), ComparisonMode.SCAN_VS_DIGITAL, AppConfig())

    found = [c for c in _serious(outcome) if "500" in c.detail and "250" in c.detail]
    assert found, [c.detail for c in _text_candidates(outcome)]
    assert 300 < found[0].bbox.center[1] < 460  # on the second line


@tesseract_required
def test_a_line_taken_out_of_the_scan_is_reported_removed(real_ocr):
    original = _leaflet()
    taken_out = _leaflet(("PARACETAMOL 500 mg TABLETS", "", "Store below 25 C in a dry place"))
    outcome = run_comparison(original, simulate_scan(taken_out, LIGHT), ComparisonMode.SCAN_VS_DIGITAL, AppConfig())

    removed = [c for c in _serious(outcome) if c.subtype is DiffSubtype.REMOVED]
    assert removed and any("contains" in c.detail for c in removed)


@tesseract_required
def test_a_line_added_in_the_scan_is_reported_added(real_ocr):
    original = _leaflet(("PARACETAMOL 500 mg TABLETS", "Each tablet contains 500 mg", ""))
    added = _leaflet(("PARACETAMOL 500 mg TABLETS", "Each tablet contains 500 mg", "Keep away from children"))
    outcome = run_comparison(original, simulate_scan(added, LIGHT), ComparisonMode.SCAN_VS_DIGITAL, AppConfig())

    found = [c for c in _serious(outcome) if c.subtype is DiffSubtype.ADDED and c.bbox.center[1] > 400]
    assert found and any("children" in c.detail for c in found), [c.detail for c in _text_candidates(outcome)]


@tesseract_required
def test_detect_scan_reads_the_digital_files_lines(real_ocr):
    from app.engine.preprocessing.scan_normalizer import ink_image

    ink = ink_image(_leaflet(), CONFIG)
    result = detect_scan(ink, ink.copy(), CONFIG)
    assert result.candidates == [] and result.unreadable_lines == 0
