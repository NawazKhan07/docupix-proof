from __future__ import annotations


import cv2
import numpy as np
import pytesseract
import pytest
import zxingcpp

from app.config import DetectionConfig
from app.engine.detectors import barcode_diff, braille_diff, color_diff, graphics_diff, text_diff
from app.engine.ocr_locator import find_tesseract
from app.engine.preprocessing.scan_normalizer import normalize_for_scan
from app.models.diff import DiffSubtype, DiffType

CONFIG = DetectionConfig()

# The application finds Tesseract itself (its usual install folders, not only PATH), so the tests ask the same way.
tesseract_required = pytest.mark.skipif(
    find_tesseract() is None,
    reason="Tesseract OCR binary is not installed (pytesseract only wraps it)",
)


@pytest.fixture
def real_ocr(monkeypatch):
    """pytesseract pointed at the Tesseract the application would use (restored afterwards)."""
    monkeypatch.setattr(pytesseract.pytesseract, "tesseract_cmd", str(find_tesseract()))


@pytest.fixture
def no_ocr(monkeypatch):
    """pytesseract pointed at nothing, whether or not Tesseract is installed here."""
    monkeypatch.setattr(pytesseract.pytesseract, "tesseract_cmd", "no-such-tesseract-executable")


def _artwork(width: int = 600, height: int = 400) -> np.ndarray:
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (40, 30), (width - 40, height - 30), (0, 0, 0), 3)
    cv2.putText(image, "PARACETAMOL 500mg", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
    cv2.rectangle(image, (70, 200), (240, 290), (200, 30, 40), -1)
    cv2.circle(image, (430, 240), 50, (30, 90, 200), -1)
    return image


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------


def test_color_detector_finds_a_changed_fill():
    original = _artwork()
    suspect = original.copy()
    cv2.rectangle(suspect, (70, 200), (240, 290), (40, 160, 60), -1)  # red block -> green

    candidates = color_diff.detect(original, suspect, CONFIG)

    assert candidates
    assert all(candidate.type is DiffType.COLOUR for candidate in candidates)
    biggest = max(candidates, key=lambda candidate: candidate.bbox.area)
    assert biggest.bbox.x0 <= 90 and biggest.bbox.x1 >= 220
    assert biggest.bbox.y0 <= 220 and biggest.bbox.y1 >= 270


def test_color_detector_ignores_identical_images():
    original = _artwork()
    assert color_diff.detect(original, original.copy(), CONFIG) == []


def test_color_detector_respects_the_compare_mask():
    """Anything masked out is invisible to the detector (FR-2.1)."""
    original = _artwork()
    suspect = original.copy()
    cv2.rectangle(suspect, (70, 200), (240, 290), (40, 160, 60), -1)

    mask = np.zeros(original.shape[:2], dtype=bool)
    mask[0:100, 0:100] = True  # nowhere near the change

    assert color_diff.detect(original, suspect, CONFIG, compare_mask=mask) == []


# ---------------------------------------------------------------------------
# Graphics (SSIM)
# ---------------------------------------------------------------------------


def test_graphics_detector_finds_a_moved_shape():
    original = _artwork()
    suspect = _artwork()
    cv2.circle(suspect, (430, 240), 50, (255, 255, 255), -1)  # erase
    cv2.circle(suspect, (430, 180), 50, (30, 90, 200), -1)  # redraw higher

    candidates = graphics_diff.detect(original, suspect, CONFIG)

    assert candidates
    assert all(candidate.type is DiffType.GRAPHICS for candidate in candidates)
    covered = candidates[0].bbox
    for candidate in candidates[1:]:
        covered = covered.union(candidate.bbox)
    assert covered.y0 <= 140 and covered.y1 >= 280


def test_graphics_detector_ignores_identical_images():
    original = _artwork()
    assert graphics_diff.detect(original, original.copy(), CONFIG) == []


def test_registration_tolerance_drops_resampling_halos_but_keeps_real_changes():
    """Calibration regression (see DetectionConfig.edge_tolerance_px).

    Round-tripping through a rotation leaves a hairline mismatch along every
    edge; without the tolerance a single halo traces a whole shape and comes
    back as one page-sized region. The tolerance must remove that without
    also erasing a genuine thin-stroke change.
    """
    original = _artwork()
    height, width = original.shape[:2]
    rotation = cv2.getRotationMatrix2D((width / 2, height / 2), 4.0, 1.0)
    round_tripped = cv2.warpAffine(
        cv2.warpAffine(original, rotation, (width, height), borderValue=(255, 255, 255)),
        cv2.invertAffineTransform(rotation),
        (width, height),
        borderValue=(255, 255, 255),
    )

    genuinely_changed = np.full(original.shape, 255, dtype=np.uint8)
    cv2.rectangle(genuinely_changed, (40, 30), (560, 370), (0, 0, 0), 3)
    cv2.putText(
        genuinely_changed, "PARACETAMOL 250mg", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2
    )
    cv2.rectangle(genuinely_changed, (70, 200), (240, 290), (200, 30, 40), -1)
    cv2.circle(genuinely_changed, (430, 240), 50, (30, 90, 200), -1)

    assert graphics_diff.detect(original, round_tripped, CONFIG) == []
    assert graphics_diff.detect(original, genuinely_changed, CONFIG)


# ---------------------------------------------------------------------------
# Scan vs Digital preprocessing
# ---------------------------------------------------------------------------


def test_scan_normalizer_flattens_background_and_blackens_text():
    image = _artwork()
    # Simulate a scan: warm grey cast plus a vignette.
    scanned = (image * 0.82 + 30).astype(np.uint8)
    vignette = np.linspace(0.75, 1.0, scanned.shape[1])[None, :, None]
    scanned = np.clip(scanned * vignette, 0, 255).astype(np.uint8)

    normalized = normalize_for_scan(scanned)

    unique_values = np.unique(normalized)
    assert set(unique_values.tolist()) <= {0, 255}  # pure black and white only
    gray = cv2.cvtColor(normalized, cv2.COLOR_RGB2GRAY)
    assert (gray == 255).mean() > 0.5  # background really is white
    assert (gray == 0).any()  # and the text survived


def test_scan_normalizing_both_sides_removes_a_colour_cast_and_vignette():
    """A scan's lighting must not read as a difference once normalized."""
    original = _artwork()
    scanned = np.clip(original * 0.8 + 25, 0, 255).astype(np.uint8)
    vignette = np.linspace(0.70, 1.0, scanned.shape[1])[None, :, None]
    scanned = np.clip(scanned * vignette, 0, 255).astype(np.uint8)

    assert color_diff.detect(original, scanned, CONFIG)  # differs before

    normalized_original = normalize_for_scan(original)
    normalized_scan = normalize_for_scan(scanned)
    agreement = float((normalized_original == normalized_scan).mean())
    assert agreement > 0.995  # essentially identical after
    assert graphics_diff.detect(normalized_original, normalized_scan, CONFIG) == []


def test_scan_normalizer_keeps_solid_artwork_elements():
    """Regression: an early background estimate with a fixed 31 px window
    read a 100 px solid roundel as background and erased it outright."""
    image = _artwork()
    normalized = cv2.cvtColor(normalize_for_scan(image), cv2.COLOR_RGB2GRAY)

    assert normalized[240, 430] == 0  # the solid roundel survived
    assert normalized[245, 155] == 0  # so did the solid block
    assert normalized[390, 300] == 255  # and the page is still white


# ---------------------------------------------------------------------------
# Barcode / QR
# ---------------------------------------------------------------------------


_SYMBOL_SCALE = 4
_QUIET_ZONE_PX = 60


def _page_with_symbol(value: str, symbol_format=zxingcpp.BarcodeFormat.Code128) -> np.ndarray:
    """A symbol rendered at its natural module size, scaled up on a white page.

    Nearest-neighbour scaling keeps module edges crisp, and the quiet zone
    around it is what lets either decoder lock on at all.
    """
    symbol = np.array(zxingcpp.write_barcode_to_image(zxingcpp.create_barcode(value, symbol_format)))
    symbol = cv2.resize(
        symbol,
        (symbol.shape[1] * _SYMBOL_SCALE, symbol.shape[0] * _SYMBOL_SCALE),
        interpolation=cv2.INTER_NEAREST,
    )
    if symbol.ndim == 2:
        symbol = cv2.cvtColor(symbol, cv2.COLOR_GRAY2RGB)

    height, width = symbol.shape[:2]
    page = np.full(
        (height + 2 * _QUIET_ZONE_PX, width + 2 * _QUIET_ZONE_PX, 3), 255, dtype=np.uint8
    )
    page[_QUIET_ZONE_PX : _QUIET_ZONE_PX + height, _QUIET_ZONE_PX : _QUIET_ZONE_PX + width] = symbol
    return page


def test_barcode_detector_passes_matching_symbols():
    page = _page_with_symbol("LOT12345")

    candidates, comparisons = barcode_diff.compare(page, page.copy(), CONFIG)

    assert candidates == []
    assert len(comparisons) == 1
    assert comparisons[0].passed is True
    assert comparisons[0].original_value == "LOT12345"


def test_barcode_detector_flags_a_changed_value():
    original = _page_with_symbol("LOT12345")
    suspect = _page_with_symbol("LOT54321")

    candidates, comparisons = barcode_diff.compare(original, suspect, CONFIG)

    assert len(candidates) == 1
    assert candidates[0].type is DiffType.BARCODE
    assert candidates[0].subtype is DiffSubtype.CHANGED
    assert comparisons[0].passed is False
    assert comparisons[0].status == "value differs"


def test_barcode_detector_flags_a_missing_symbol():
    original = _page_with_symbol("LOT12345")
    blank = np.full(original.shape, 255, dtype=np.uint8)

    candidates, comparisons = barcode_diff.compare(original, blank, CONFIG)

    assert len(candidates) == 1
    assert candidates[0].subtype is DiffSubtype.REMOVED
    assert comparisons[0].status == "missing from suspect"


def test_barcode_detector_reads_qr_codes():
    page = _page_with_symbol("https://example.test/batch/1", zxingcpp.BarcodeFormat.QRCode)
    _candidates, comparisons = barcode_diff.compare(page, page.copy(), CONFIG)
    assert comparisons[0].symbol_type.upper().startswith("QR")
    assert comparisons[0].passed is True


# ---------------------------------------------------------------------------
# Braille
# ---------------------------------------------------------------------------


def _braille_page(dot_centers: list[tuple[int, int]]) -> np.ndarray:
    page = np.full((300, 400, 3), 255, dtype=np.uint8)
    for x, y in dot_centers:
        cv2.circle(page, (x, y), 4, (200, 20, 140), -1)  # magenta spot colour
    return page


def test_braille_detector_matches_identical_dot_patterns():
    dots = [(100, 100), (120, 100), (100, 120), (140, 160)]
    page = _braille_page(dots)

    candidates, comparison = braille_diff.compare(page, page.copy(), CONFIG)

    assert candidates == []
    assert comparison.passed is True
    assert comparison.original_dot_count == len(dots)
    assert comparison.matched == len(dots)


def test_braille_detector_reports_a_missing_dot():
    original = _braille_page([(100, 100), (120, 100), (100, 120)])
    suspect = _braille_page([(100, 100), (120, 100)])

    candidates, comparison = braille_diff.compare(original, suspect, CONFIG)

    assert comparison.missing == 1
    assert comparison.passed is False
    assert [c.subtype for c in candidates] == [DiffSubtype.REMOVED]
    assert candidates[0].type is DiffType.BRAILLE


def test_braille_detector_reports_a_moved_dot_rather_than_a_missing_plus_added_pair():
    original = _braille_page([(100, 100), (200, 200)])
    suspect = _braille_page([(100, 100), (218, 200)])  # 18 px away

    candidates, comparison = braille_diff.compare(original, suspect, CONFIG)

    assert comparison.moved == 1
    assert comparison.missing == 0 and comparison.added == 0
    assert candidates[0].subtype is DiffSubtype.MOVED


# ---------------------------------------------------------------------------
# Text / OCR (skipped without the Tesseract binary)
# ---------------------------------------------------------------------------


def test_text_detector_reports_ocr_unavailable_clearly(no_ocr):
    """Without Tesseract the detector must say so, not report 'no differences'."""
    with pytest.raises(text_diff.OcrUnavailableError, match="not installed"):
        text_diff.detect(_artwork(), _artwork(), CONFIG)


@tesseract_required
def test_text_detector_finds_a_changed_number(real_ocr):
    original = _artwork()
    suspect = np.full(original.shape, 255, dtype=np.uint8)
    cv2.rectangle(suspect, (40, 30), (560, 370), (0, 0, 0), 3)
    cv2.putText(suspect, "PARACETAMOL 250mg", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)

    candidates = text_diff.detect(original, suspect, CONFIG)

    assert any(candidate.subtype is DiffSubtype.CHANGED for candidate in candidates)
    assert any("500mg" in candidate.detail for candidate in candidates)


@tesseract_required
def test_text_detector_finds_no_differences_between_identical_pages(real_ocr):
    original = _artwork()
    assert text_diff.detect(original, original.copy(), CONFIG) == []
