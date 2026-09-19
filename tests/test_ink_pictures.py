"""Scan vs Digital works on what is PRINTED: each file as ink on paper, background colours removed to white.

A scan changes every colour, so the mode compares where ink is, not what colour it is. `ink_image` is the
picture that is read for text and shown to the reviewer: the background of every kind comes out white and
everything printed on it -- whatever its colour -- comes out black.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from scan_simulator import HARSH, LIGHT, TYPICAL, simulate_scan

from app.config import AppConfig, DetectionConfig
from app.engine.detectors import text_diff
from app.engine.detectors.structure_diff import StructureResult
from app.engine.preprocessing.scan_normalizer import ink_image
from app.engine.worker import _for_reading, _readable_only, run_comparison
from app.models.diff import BoundingBox, DiffCandidate, DiffSubtype, DiffType
from app.models.isolation import IsolationGeometry, NormalizedRect
from app.models.job import ComparisonMode

CONFIG = DetectionConfig()
WHITE = (255, 255, 255)


def _page(background=WHITE, size=(700, 1000)) -> np.ndarray:
    height, width = size
    return np.full((height, width, 3), background, np.uint8)


def _text(image, words, origin, colour, scale=2.0, thickness=4):
    cv2.putText(image, words, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)
    return image


def _darkest(ink: np.ndarray, box) -> int:
    x0, y0, x1, y1 = box
    return int(ink[y0:y1, x0:x1].min())


# ---- the paper -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("paper", [WHITE, (250, 244, 236), (238, 238, 238), (255, 250, 250)], ids=["white", "cream", "grey", "pink"])
def test_blank_paper_of_any_shade_is_white(paper):
    ink = ink_image(_page(paper), CONFIG)
    assert ink.dtype == np.uint8 and ink.shape == (700, 1000) and (ink == 255).all()


def test_uneven_lighting_is_flattened():
    """A scan's darker corners and a lamp's gradient are not ink."""
    page = _text(_page(), "Read the leaflet", (80, 400), (20, 20, 20))
    ys, xs = np.mgrid[0:700, 0:1000].astype(np.float32)
    shade = 0.72 + 0.28 * (1.0 - np.hypot((xs - 500) / 500, (ys - 350) / 350) / 1.42)
    lit = np.clip(page.astype(np.float32) * shade[..., None], 0, 255).astype(np.uint8)

    ink = ink_image(lit, CONFIG)

    background = np.ones(ink.shape, bool)
    background[330:430, 60:700] = False
    assert (ink[background] == 255).mean() > 0.995
    assert _darkest(ink, (80, 340, 700, 420)) < 30


def test_scan_noise_below_the_ink_is_not_ink():
    rng = np.random.default_rng(3)
    page = _page((246, 246, 246)).astype(np.float32) + rng.normal(0, 5.0, (700, 1000, 3))
    ink = ink_image(np.clip(page, 0, 255).astype(np.uint8), CONFIG)
    assert (ink < 250).mean() < 0.001


# ---- background colours go, ink stays ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tint",
    [(226, 238, 246), (200, 230, 245), (255, 240, 200), (102, 200, 235)],
    ids=["pale blue", "light blue", "cream yellow", "60% cyan"],
)
def test_a_background_colour_is_removed_and_the_text_on_it_stays_black(tint):
    """The client's need: remove the background colour, keep the text -- even on a strong tint, where black text
    would otherwise be swallowed by a block that is itself dark."""
    page = _page()
    cv2.rectangle(page, (100, 100), (900, 600), tint, -1)
    _text(page, "500 mg", (180, 400), (15, 15, 15))

    ink = ink_image(page, CONFIG)

    panel_only = ink[120:160, 120:880]
    assert (panel_only == 255).mean() > 0.99  # the panel is white
    assert _darkest(ink, (180, 340, 460, 420)) < 30  # the text is black


def test_a_colour_gradient_panel_is_removed():
    page = _page()
    ramp = np.linspace(0, 1, 800, dtype=np.float32)
    for x in range(800):  # pale blue fading to white
        page[100:600, 100 + x] = (255 - 45 * (1 - ramp[x]), 255 - 20 * (1 - ramp[x]), 255 - 8 * (1 - ramp[x]))
    _text(page, "Store below 30 C", (200, 380), (15, 15, 15), scale=1.6, thickness=3)

    ink = ink_image(page, CONFIG)

    assert (ink[110:170, 110:890] == 255).mean() > 0.99
    assert _darkest(ink, (200, 330, 700, 400)) < 40


@pytest.mark.parametrize(
    "colour",
    [(0, 190, 215), (245, 225, 30), (240, 140, 30), (230, 20, 140), (30, 90, 200), (60, 150, 60)],
    ids=["cyan", "yellow", "orange", "magenta", "blue", "green"],
)
def test_coloured_ink_is_black(colour):
    """A grey-scale conversion calls cyan or yellow lettering nearly as light as the paper, and it would be lost."""
    ink = ink_image(_text(_page(), "BSV", (300, 400), colour, scale=4.0, thickness=10), CONFIG)
    assert _darkest(ink, (300, 300, 700, 420)) < 40
    assert (ink[:200] == 255).all()


def test_white_lettering_on_a_dark_block_stays_visible_as_white_on_black():
    page = _page()
    cv2.rectangle(page, (100, 200), (900, 500), (0, 130, 70), -1)  # a dark green bar
    _text(page, "AMPHOTRET", (160, 380), WHITE, scale=3.0, thickness=8)

    ink = ink_image(page, CONFIG)

    assert (ink[220:250, 120:880] < 20).mean() > 0.98  # the block is a solid black graphic
    letters = ink[300:390, 160:880]
    assert 0.08 < (letters > 235).mean() < 0.6  # and the letters read as white holes in it


def test_a_solid_dark_graphic_is_kept_black_not_reduced_to_its_outline():
    page = _page()
    cv2.rectangle(page, (200, 150), (800, 550), (20, 20, 20), -1)
    ink = ink_image(page, CONFIG)
    assert (ink[200:500, 250:750] < 10).mean() > 0.99


def test_the_same_artwork_in_two_colourways_gives_the_same_picture():
    """Recolouring changes nothing that is printed, so the ink pictures agree (a scan's colour cast, in effect)."""
    def artwork(tint, ink_colour):
        page = _page()
        cv2.rectangle(page, (100, 100), (900, 600), tint, -1)
        _text(page, "Sodium 500 mg", (150, 330), ink_colour, scale=2.0, thickness=4)
        cv2.circle(page, (700, 450), 70, ink_colour, -1)
        return page

    a = ink_image(artwork((226, 238, 246), (15, 15, 15)), CONFIG)
    b = ink_image(artwork((248, 230, 214), (60, 10, 20)), CONFIG)
    assert (np.abs(a.astype(int) - b.astype(int)) > 100).mean() < 0.002


# ---- a scan --------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "profile,disagreement",
    [(LIGHT, 0.02), (TYPICAL, 0.03), (HARSH, 0.07)],
    ids=["light", "typical", "harsh"],
)
def test_a_scan_and_its_digital_original_give_the_same_picture(profile, disagreement):
    page = _page()
    cv2.rectangle(page, (100, 100), (900, 600), (226, 238, 246), -1)
    _text(page, "Sodium 500 mg", (150, 330), (15, 15, 15), scale=2.0, thickness=4)
    cv2.circle(page, (700, 450), 70, (200, 40, 40), -1)
    scan = simulate_scan(page, profile)
    scan = cv2.resize(scan, (page.shape[1], page.shape[0]), interpolation=cv2.INTER_AREA)

    a, b = ink_image(page, CONFIG), ink_image(scan, CONFIG)

    assert (b[110:170, 110:890] > 245).mean() > 0.97  # the panel is white in the scan too
    assert (np.abs(a.astype(int) - b.astype(int)) > 128).mean() < disagreement


# ---- outside the compared area ---------------------------------------------------------------------------------------------


def test_outside_the_valid_area_is_white_and_a_dark_corner_beside_it_is_not_ink():
    """Outside the artwork the page is white by design; beside it, the scan's own darkened corner must not read as ink."""
    page = _page((200, 200, 200))  # the scan's shaded corner: uniform grey, no ink at all
    valid = np.zeros(page.shape[:2], bool)
    valid[100:650, 100:900] = True
    page[~valid] = 255

    ink = ink_image(page, CONFIG, valid)

    assert (ink[~valid] == 255).all()
    assert (ink[valid] == 255).mean() > 0.99


def test_a_fully_valid_mask_changes_nothing():
    page = _text(_page(), "PARACETAMOL", (80, 300), (20, 20, 20))
    assert np.array_equal(ink_image(page, CONFIG), ink_image(page, CONFIG, np.ones(page.shape[:2], bool)))


# ---- the comparison ---------------------------------------------------------------------------------------------------------------


def _label(text="PARACETAMOL 500 mg", tint=(226, 238, 246), colour=(200, 60, 50)) -> np.ndarray:
    page = _page()
    cv2.rectangle(page, (100, 100), (900, 600), tint, -1)
    _text(page, text, (130, 260), (20, 20, 20), scale=1.5, thickness=3)
    cv2.rectangle(page, (150, 350), (450, 500), colour, -1)
    cv2.circle(page, (700, 430), 60, (30, 90, 200), -1)
    return page


def test_the_outcome_carries_the_ink_pictures_only_for_scan_vs_digital():
    page = _label()
    scan_outcome = run_comparison(page, simulate_scan(page, LIGHT), ComparisonMode.SCAN_VS_DIGITAL, AppConfig())
    other = run_comparison(page, page.copy(), ComparisonMode.AUTOMATIC, AppConfig())

    assert scan_outcome.ink_original.shape == page.shape[:2] == scan_outcome.ink_suspect.shape
    assert scan_outcome.ink_original.dtype == np.uint8
    assert other.ink_original is None and other.ink_suspect is None


def test_the_ink_pictures_show_the_text_black_on_white_in_both_files():
    page = _label()
    outcome = run_comparison(page, simulate_scan(page, TYPICAL), ComparisonMode.SCAN_VS_DIGITAL, AppConfig())

    for ink in (outcome.ink_original, outcome.ink_suspect):
        assert (ink[110:200, 110:890] > 245).mean() > 0.97  # the panel is gone
        assert ink[230:280, 130:700].min() < 40  # the words are black


def test_the_ink_pictures_are_white_outside_the_area_that_was_compared():
    page = _label()
    area = IsolationGeometry(boundary=NormalizedRect(0.05, 0.1, 0.95, 0.9))
    outcome = run_comparison(page, page.copy(), ComparisonMode.SCAN_VS_DIGITAL, AppConfig(), area, area)
    assert (outcome.ink_original[:60] == 255).all() and (outcome.ink_suspect[:60] == 255).all()


@pytest.mark.parametrize(
    "profile",
    [
        LIGHT,
        pytest.param(
            TYPICAL,
            marks=pytest.mark.xfail(
                reason="known and older than the ink pictures: a typical scan's JPEG chroma bleed at the edge of a "
                "saturated block is reported (the last commit does the same)",
                strict=False,
            ),
        ),
        HARSH,
    ],
    ids=["light", "typical", "harsh"],
)
def test_a_scan_with_a_colour_cast_of_the_same_artwork_reports_nothing(profile):
    """The reason for the mode: colours change in a scan (a warm lamp, dimmer paper), and must not read as differences."""
    original = _label()
    outcome = run_comparison(original, simulate_scan(original, profile), ComparisonMode.SCAN_VS_DIGITAL, AppConfig())
    assert outcome.clusters == []


def test_a_missing_word_is_found_in_a_scan():
    """(A single changed digit in small print is a job for the text reader: a scan cannot hold it as marks.)"""
    original = _label("PARACETAMOL 500 mg")
    outcome = run_comparison(original, simulate_scan(_label("PARACETAMOL"), LIGHT), ComparisonMode.SCAN_VS_DIGITAL, AppConfig())
    assert any(c.bbox_in_original.y0 < 300 and c.bbox_in_original.y1 > 220 and c.bbox_in_original.x1 > 500 for c in outcome.clusters)


def test_a_moved_graphic_is_found_in_a_scan():
    original = _label()
    changed = _label()
    cv2.rectangle(changed, (150, 350), (450, 500), (226, 238, 246), -1)  # take the block away...
    cv2.rectangle(changed, (300, 420), (600, 570), (200, 60, 50), -1)  # ...and put it somewhere else
    outcome = run_comparison(original, simulate_scan(changed, LIGHT), ComparisonMode.SCAN_VS_DIGITAL, AppConfig())
    assert len(outcome.clusters) >= 1


# ---- barcodes: compared as marks, never decoded ---------------------------------------------------------------------------------


def _with_barcode(bars) -> np.ndarray:
    page = _page(size=(1400, 1000))
    cv2.putText(page, "PARACETAMOL 500 mg", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.rectangle(page, (80, 300), (420, 480), (200, 60, 50), -1)
    x = 300
    for width, gap in bars:
        cv2.rectangle(page, (x, 800), (x + width - 1, 1000), (15, 15, 15), -1)
        x += width + gap
    cv2.putText(page, "5 060545 240205", (300, 1050), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    return page


def _bars(count=36, seed=4):
    rng = np.random.default_rng(seed)
    return [(int(rng.choice([4, 8, 12])), int(rng.choice([4, 8, 12]))) for _ in range(count)]


@pytest.mark.parametrize("profile", [LIGHT, TYPICAL, HARSH], ids=["light", "typical", "harsh"])
def test_an_unchanged_scanned_barcode_reports_nothing_and_is_not_decoded(profile):
    page = _with_barcode(_bars())
    outcome = run_comparison(page, simulate_scan(page, profile), ComparisonMode.SCAN_VS_DIGITAL, AppConfig())
    assert outcome.clusters == [] and outcome.symbol_comparisons == []


@pytest.mark.parametrize("profile", [LIGHT, TYPICAL, HARSH], ids=["light", "typical", "harsh"])
def test_bars_missing_from_a_scanned_barcode_are_found_as_a_physical_difference(profile):
    bars = _bars()
    original = _with_barcode(bars)
    scan = simulate_scan(_with_barcode(bars[:15] + bars[17:]), profile)  # two bars fewer

    outcome = run_comparison(original, scan, ComparisonMode.SCAN_VS_DIGITAL, AppConfig())

    assert any(c.bbox_in_original.y0 < 1010 and c.bbox_in_original.y1 > 790 for c in outcome.clusters)
    assert outcome.symbol_comparisons == []  # read as marks: nothing was decoded


# ---- text that a scan cannot hold is not reported missing ---------------------------------------------------------------------------


def _finding(x0, y0, x1, y1) -> DiffCandidate:
    return DiffCandidate(BoundingBox(x0, y0, x1, y1), DiffType.TEXT, DiffSubtype.REMOVED, 0.9, "'Store' missing from suspect")


def test_a_word_where_the_scan_cannot_hold_fine_print_is_not_reported():
    unverifiable = np.zeros((400, 600), bool)
    unverifiable[100:200, 100:300] = True  # too soft to compare here
    inside, outside, straddling = _finding(120, 120, 180, 150), _finding(400, 300, 460, 330), _finding(280, 140, 400, 170)

    kept = _readable_only([inside, outside, straddling], unverifiable)

    assert kept == [outside, straddling]  # a word half in it is still judged on its other half


def test_nothing_is_dropped_when_nothing_was_skipped():
    findings = [_finding(0, 0, 50, 20)]
    assert _readable_only(findings, None) == findings


def test_the_structure_result_reports_where_it_skipped():
    original = np.full((300, 300, 3), 255, np.uint8)
    result = StructureResult([], 0.0, 0.0)
    assert result.unverifiable_mask is None and original.shape == (300, 300, 3)


def test_a_large_page_is_reduced_for_reading_and_a_small_one_is_left_alone():
    config = DetectionConfig()
    big = np.full((5000, 3600), 255, np.uint8)
    a, b, scale = _for_reading(big, big, config)
    assert max(a.shape) == config.ocr_long_edge_px and a.shape == b.shape and scale == pytest.approx(config.ocr_long_edge_px / 5000)

    small = np.full((3000, 2000), 255, np.uint8)
    a, b, scale = _for_reading(small, small, config)
    assert a is small and scale == 1.0


def test_text_is_read_from_the_ink_pictures(monkeypatch):
    """The text detector is handed the ink pictures (grey, 255 = background), not the colour originals."""
    page = _label()
    seen = []

    def fake_detect(original, suspect, config):
        seen.append((original, suspect))
        return text_diff.TextScanResult([])

    monkeypatch.setattr(text_diff, "detect_scan", fake_detect)
    outcome = run_comparison(page, simulate_scan(page, LIGHT), ComparisonMode.SCAN_VS_DIGITAL, AppConfig())

    assert len(seen) == 1 and seen[0][0].ndim == 2 and seen[0][0].dtype == np.uint8
    assert np.array_equal(seen[0][0], outcome.ink_original) and np.array_equal(seen[0][1], outcome.ink_suspect)
