"""Pharmacode reading: neither bundled decoder supports it (SRS FR-10.1)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import zxingcpp

from app.config import DetectionConfig
from app.engine.detectors import barcode_diff, pharmacode
from app.models.diff import DiffType
from app.models.result import SymbolComparison, symbol_group

CONFIG = DetectionConfig()
WHITE = (255, 255, 255)


def _draw(
    value: int,
    *,
    sideways: bool = False,
    thin: int = 4,
    thick: int = 10,
    space: int = 8,
    length: int = 140,
    page=(420, 420),
    at=(120, 90),
    colour=(30, 30, 30),
    page_image: np.ndarray | None = None,
) -> np.ndarray:
    """A Pharmacode for `value`. Upright: bars side by side, the code read from the right.
    Sideways: bars stacked, read from the top."""
    page_image = np.full((*page, 3), 255, np.uint8) if page_image is None else page_image
    bars = pharmacode.encode(value)  # reading order
    order = bars if sideways else bars[::-1]  # laid out top-to-bottom / left-to-right
    cursor = 0
    x0, y0 = at
    for is_thick in order:
        size = thick if is_thick else thin
        if sideways:
            cv2.rectangle(page_image, (x0, y0 + cursor), (x0 + length - 1, y0 + cursor + size - 1), colour, -1)
        else:
            cv2.rectangle(page_image, (x0 + cursor, y0), (x0 + cursor + size - 1, y0 + length - 1), colour, -1)
        cursor += size + space
    return page_image


def _values(image: np.ndarray) -> list[int]:
    return sorted(code.value for code in pharmacode.find_pharmacodes(image, CONFIG))


# ---- the code itself -------------------------------------------------------------------


def test_encoding_and_decoding_are_inverse_over_the_whole_range():
    for value in list(range(3, 3000)) + [65535, 65536, 100000, 131069, 131070]:
        assert pharmacode.decode(pharmacode.encode(value)) == value


def test_the_known_extremes():
    assert pharmacode.encode(3) == [False, False]  # two thin bars
    assert pharmacode.encode(131070) == [True] * 16  # sixteen thick bars
    with pytest.raises(ValueError):
        pharmacode.encode(2)
    with pytest.raises(ValueError):
        pharmacode.encode(131071)


# ---- reading it from an image -----------------------------------------------------------


@pytest.mark.parametrize("sideways", [False, True], ids=["upright", "sideways"])
@pytest.mark.parametrize("value", [15, 21, 161, 324, 417, 1000, 4095, 65000, 131070])
def test_a_drawn_pharmacode_reads_back_in_either_orientation(value, sideways):
    image = _draw(value, sideways=sideways, page=(600, 900) if sideways else (420, 1000))
    assert _values(image) == [value]


@pytest.mark.parametrize(
    "thin,thick,space",
    [(3, 9, 6), (4, 10, 8), (2, 6, 4), (6, 18, 12), (5, 12, 10)],
    ids=["1:3", "carton-like", "tiny", "large", "1:2.4"],
)
def test_it_reads_whatever_size_and_proportions_it_is_printed_at(thin, thick, space):
    assert _values(_draw(2731, thin=thin, thick=thick, space=space, length=80)) == [2731]


def test_all_thin_and_all_thick_bars_are_told_apart_by_the_spaces():
    """With one thickness there is nothing to compare bar with bar; the space is the ruler."""
    assert _values(_draw(pharmacode.decode([False] * 7))) == [pharmacode.decode([False] * 7)]
    assert _values(_draw(pharmacode.decode([True] * 7))) == [pharmacode.decode([True] * 7)]


def test_it_reads_dark_colours_and_softened_prints():
    dark_blue = _draw(2731, colour=(20, 30, 90))
    assert _values(dark_blue) == [2731]

    soft = cv2.GaussianBlur(_draw(2731), (0, 0), 0.8)
    noisy = np.clip(soft + np.random.default_rng(3).normal(0, 4, soft.shape), 0, 255).astype(np.uint8)
    assert _values(noisy) == [2731]


def test_a_guide_line_crossing_the_symbol_does_not_change_the_reading():
    """Real artwork prints fold and cut lines over the symbol; on the sample carton a one
    pixel line through the gaps joined every bar into one blob."""
    for sideways in (False, True):
        image = _draw(161, sideways=sideways, page=(420, 420))
        if sideways:
            cv2.line(image, (190, 60), (190, 300), (200, 0, 30), 1)  # a thin line through the gaps
        else:
            cv2.line(image, (60, 160), (300, 160), (200, 0, 30), 1)
        assert _values(image) == [161], f"sideways={sideways}"


def test_a_line_that_cuts_every_bar_in_two_does_not_change_the_reading():
    image = _draw(161, sideways=True)
    cv2.line(image, (190, 60), (190, 300), (235, 70, 83), 3)  # lighter than a bar: leaves a notch in each
    assert _values(image) == [161]


def test_two_symbols_on_one_page_are_both_found():
    image = _draw(161, page=(420, 900), at=(80, 90))
    _draw(4004, page_image=image, at=(500, 90))
    assert _values(image) == [161, 4004]


def test_the_box_is_the_symbol_and_no_more():
    image = _draw(161, at=(120, 90), length=140)
    (code,) = pharmacode.find_pharmacodes(image, CONFIG)
    assert code.bbox.y0 == pytest.approx(90, abs=2) and code.bbox.y1 == pytest.approx(230, abs=2)
    assert code.bbox.x0 == pytest.approx(120, abs=2)


# ---- what it must not mistake for a Pharmacode ------------------------------------------


def _blank():
    return np.full((420, 900, 3), 255, np.uint8)


def test_an_ordinary_barcode_is_not_also_a_pharmacode():
    """An EAN symbol has several space widths; equal spaces are what marks a Pharmacode."""
    symbol = np.array(zxingcpp.write_barcode_to_image(zxingcpp.create_barcode("8901117282322", zxingcpp.BarcodeFormat.EAN13)))
    symbol = cv2.resize(symbol, (symbol.shape[1] * 3, symbol.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
    page = _blank()
    page[100 : 100 + symbol.shape[0], 100 : 100 + symbol.shape[1]] = symbol[..., None] if symbol.ndim == 2 else symbol
    assert _values(page) == []


def test_text_is_not_a_pharmacode():
    page = _blank()
    cv2.putText(page, "Illinois Ill Ill lIl 1lll", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
    assert _values(page) == []


def test_a_symbol_of_fewer_than_four_bars_is_left_alone():
    """Values below 15 have 2-3 bars, which cannot be told from ordinary marks; the
    price of not reading them is stated in the module, the alternative is false codes."""
    assert _values(_draw(3)) == [] and _values(_draw(6)) == []


def test_a_few_ticks_are_not_a_pharmacode():
    page = _blank()
    for x in (100, 120, 140):  # three equal bars: too few to be told from marks
        cv2.rectangle(page, (x, 100), (x + 3, 200), (0, 0, 0), -1)
    assert _values(page) == []


def test_bars_with_uneven_spaces_are_not_a_pharmacode():
    page = _blank()
    x = 100
    for gap in (8, 8, 20, 8, 31, 8):
        cv2.rectangle(page, (x, 100), (x + 3, 200), (0, 0, 0), -1)
        x += 4 + gap
    assert _values(page) == []


def test_bars_of_several_thicknesses_are_not_a_pharmacode():
    page = _blank()
    x = 100
    for size in (3, 6, 12, 3, 9, 3):
        cv2.rectangle(page, (x, 100), (x + size - 1, 200), (0, 0, 0), -1)
        x += size + 8
    assert _values(page) == []


def test_bars_of_unequal_length_are_not_a_pharmacode():
    page = _blank()
    for index, length in enumerate((100, 60, 100, 40, 100, 80)):
        x = 100 + index * 12
        cv2.rectangle(page, (x, 100), (x + 3, 100 + length), (0, 0, 0), -1)
    assert _values(page) == []


def test_a_run_taken_from_inside_a_longer_pattern_is_not_a_pharmacode():
    """Quiet space at both ends: eight equal bars flanked by more bars are part of something else."""
    page = _blank()
    for index in range(24):
        x = 100 + index * 12
        cv2.rectangle(page, (x, 100), (x + 3, 200), (0, 0, 0), -1)
    assert _values(page) == []


def test_a_light_colour_is_not_read_as_a_bar():
    assert _values(_draw(161, colour=(200, 200, 60))) == []


# ---- through the barcode comparison ---------------------------------------------------------


def test_a_pharmacode_that_changed_is_a_failed_symbol_and_a_finding():
    original, suspect = _draw(161), _draw(417)

    candidates, comparisons = barcode_diff.compare(original, suspect, CONFIG)

    (comparison,) = comparisons
    assert comparison.symbol_type == "Pharmacode" and not comparison.passed
    assert (comparison.original_value, comparison.suspect_value) == ("161", "417")
    (candidate,) = candidates
    assert candidate.type is DiffType.BARCODE and "Pharmacode" in candidate.detail


def test_a_matching_pharmacode_passes():
    _candidates, (comparison,) = barcode_diff.compare(_draw(161), _draw(161), CONFIG)
    assert comparison.passed and comparison.bbox is not None


def test_a_pharmacode_missing_from_the_suspect_is_reported():
    _candidates, (comparison,) = barcode_diff.compare(_draw(161), _blank(), CONFIG)
    assert comparison.status == "missing from suspect"


def test_symbols_are_grouped_by_type():
    """FR-10.2: Barcode / Pharmacode / QR code."""
    assert symbol_group("EAN13") == "Barcode"
    assert symbol_group("CODE128") == "Barcode"
    assert symbol_group("Code128") == "Barcode"
    assert symbol_group("Pharmacode") == "Pharmacode"
    assert symbol_group("QRCode") == "QR code"
    assert symbol_group("QRCODE") == "QR code"
    assert symbol_group("DataMatrix") == "QR code"
    assert SymbolComparison("EAN13", "1", "1").group == "Barcode"
