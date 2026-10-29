"""Barcode mode: every code read (decoded) on both files and compared exactly, matched or not (FR-10.1/10.2).

Linear barcodes, Pharmacode and two-dimensional codes are all decoded; a code matches only if both files' decode to
the same value as the same symbology. The same code moved on the layout is still a match (said to have moved); a code
found but not decoded is listed so, never dropped; a value that differs is said how (which characters, or which GS1
field); a pattern that only looks like a code -- a job sheet's grid of tick boxes, the stems of 'mm' in a serif type --
is not taken for one.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import zxingcpp

from app.config import DetectionConfig
from app.engine.detectors import barcode_diff, pharmacode
from app.engine.detectors.barcode_diff import SymbolReading, _like_a_code, describe_difference
from app.models.diff import BoundingBox, DiffSubtype, DiffType
from app.models.result import SymbolComparison

CONFIG = DetectionConfig()


def _code(value: str, kind=zxingcpp.BarcodeFormat.QRCode, scale: int = 6) -> np.ndarray:
    symbol = np.array(zxingcpp.write_barcode_to_image(zxingcpp.create_barcode(value, kind)))
    symbol = cv2.resize(symbol, (symbol.shape[1] * scale, symbol.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(symbol, cv2.COLOR_GRAY2RGB) if symbol.ndim == 2 else symbol


def _page(*codes: tuple[np.ndarray, tuple[int, int]], size=(1200, 1600)) -> np.ndarray:
    page = np.full((size[0], size[1], 3), 255, np.uint8)
    for symbol, (x, y) in codes:
        page[y : y + symbol.shape[0], x : x + symbol.shape[1]] = symbol
    return page


# ---- decoded and compared exactly -----------------------------------------------------------------------------------------


def test_every_code_is_listed_and_the_same_ones_match():
    original = _page((_code("https://a.example/1"), (100, 100)), (_code("8901117282322", zxingcpp.BarcodeFormat.EAN13, 3), (700, 120)))
    candidates, comparisons = barcode_diff.compare(original, original.copy(), CONFIG)
    assert candidates == []
    assert sorted(c.group for c in comparisons) == ["Barcode", "QR code"] and all(c.passed for c in comparisons)


def test_the_same_code_moved_on_the_layout_still_matches_and_is_said_to_have_moved():
    original = _page((_code("https://a.example/1"), (100, 100)))
    suspect = _page((_code("https://a.example/1"), (900, 700)))

    candidates, (comparison,) = barcode_diff.compare(original, suspect, CONFIG)

    assert candidates == [] and comparison.passed
    assert comparison.moved_px > 500 and comparison.status.startswith("match (moved")


def test_a_value_that_differs_is_a_finding_saying_which_characters():
    original = _page((_code("8901117282322", zxingcpp.BarcodeFormat.EAN13, 3), (300, 200)))
    suspect = _page((_code("8901117282339", zxingcpp.BarcodeFormat.EAN13, 3), (300, 200)))

    candidates, (comparison,) = barcode_diff.compare(original, suspect, CONFIG)

    assert comparison.status == "value differs" and not comparison.passed
    (finding,) = candidates
    assert finding.type is DiffType.BARCODE and finding.subtype is DiffSubtype.CHANGED
    assert "'8901117282322' → '8901117282339'" in finding.detail and "character 12" in finding.detail


def test_the_same_digits_in_another_symbology_do_not_match():
    original = _page((_code("12345678", zxingcpp.BarcodeFormat.Code128, 3), (300, 200)))
    suspect = _page((_code("12345678", zxingcpp.BarcodeFormat.Code39, 3), (300, 200)))

    _candidates, (comparison,) = barcode_diff.compare(original, suspect, CONFIG)

    assert comparison.original_value == comparison.suspect_value and not comparison.passed
    assert comparison.status == "symbology differs"


def test_a_code_only_one_file_has_is_missing_or_extra():
    original = _page((_code("https://a.example/1"), (100, 100)))
    suspect = _page((_code("8901117282322", zxingcpp.BarcodeFormat.EAN13, 3), (900, 700)))

    candidates, comparisons = barcode_diff.compare(original, suspect, CONFIG)

    assert sorted(c.status for c in comparisons) == ["extra in suspect", "missing from suspect"]
    assert sorted(c.subtype.value for c in candidates) == ["added", "removed"]


def test_a_two_dimensional_code_replaced_by_another_kind_is_one_finding():
    """The only 2-D code of each file, a QR code in one and a Data Matrix in the other: paired, the symbology differs."""
    original = _page((_code("https://a.example/1"), (100, 100)))
    suspect = _page((_code("https://a.example/1", zxingcpp.BarcodeFormat.DataMatrix, 8), (900, 700)))

    candidates, (comparison,) = barcode_diff.compare(original, suspect, CONFIG)

    assert comparison.status == "symbology differs" and len(candidates) == 1
    assert "QRCode in the original, DataMatrix in the suspect" in candidates[0].detail


# ---- how a difference is said ---------------------------------------------------------------------------------------------


def test_a_gs1_code_is_compared_field_by_field():
    said = describe_difference("(01)08901117282322(17)261231(10)AB12", "(01)08901117282322(17)270131(10)AB12")
    assert said == "(17) expiry: 261231 → 270131"
    assert "(10) batch/lot" in describe_difference("(01)1(10)A", "(01)1(10)B")


def test_found_but_not_decoded_is_listed_so_not_dropped():
    found = SymbolComparison("DataMatrix", "8904103312775", None, BoundingBox(0, 0, 10, 10), suspect_type="DataMatrix", suspect_found=True)
    assert not found.passed and found.status == "could not be decoded in the suspect"
    gone = SymbolComparison("DataMatrix", "8904103312775", None, BoundingBox(0, 0, 10, 10), suspect_found=False)
    assert gone.status == "missing from suspect"


def test_codes_are_paired_by_value_first_then_by_place(monkeypatch):
    """Two QR codes swapped on the layout are two matches, not two values that differ."""
    first, second = BoundingBox(100, 100, 200, 200), BoundingBox(800, 100, 900, 200)
    readings = {
        "original": [SymbolReading("QRCode", "A", first), SymbolReading("QRCode", "B", second)],
        "suspect": [SymbolReading("QRCode", "B", first), SymbolReading("QRCode", "A", second)],
    }
    calls = iter(["original", "suspect"])
    monkeypatch.setattr(barcode_diff, "decode_symbols", lambda _image, _config=None: readings[next(calls)])

    candidates, comparisons = barcode_diff.compare(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10, 3), np.uint8), CONFIG)

    assert candidates == [] and all(c.passed for c in comparisons) and all(c.moved_px > 0 for c in comparisons)


# ---- not taken for a code ------------------------------------------------------------------------------------------------


def test_a_huge_found_but_undecoded_pattern_is_not_a_code():
    """A job sheet's grid of tick boxes was 'found' as a Data Matrix half the page wide."""
    page_shape = (3508, 2481)
    assert not _like_a_code(SymbolReading("DataMatrix", None, BoundingBox(243, 2169, 2232, 3123)), page_shape)
    assert _like_a_code(SymbolReading("DataMatrix", None, BoundingBox(300, 300, 380, 380)), page_shape)


def test_the_stems_of_lettering_are_not_a_pharmacode_but_a_code_beside_lettering_is_read():
    page = np.full((400, 1200, 3), 255, np.uint8)
    cv2.putText(page, "268 mm (+/-2 mm)", (40, 120), cv2.FONT_HERSHEY_COMPLEX, 1.4, (20, 20, 20), 2, cv2.LINE_AA)
    assert pharmacode.find_pharmacodes(page, CONFIG) == []

    bars, x = pharmacode.encode(28294)[::-1], 300
    for thick in bars:  # a code a little below the text, clear space either side of it
        width = 10 if thick else 4
        cv2.rectangle(page, (x, 200), (x + width - 1, 340), (30, 30, 30), -1)
        x += width + 8
    assert [code.value for code in pharmacode.find_pharmacodes(page, CONFIG)] == [28294]
