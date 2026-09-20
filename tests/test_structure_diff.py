"""Structural comparison for scans (SRS FR-6.3). Each test pins a failure that was measured."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest
from scan_simulator import HARSH, LIGHT, TYPICAL, ScanProfile, simulate_scan

from dataclasses import replace

from app.config import AppConfig, DetectionConfig
from app.engine.detectors import structure_diff
from app.engine.detectors.structure_diff import estimate_noise, slack_for, unresolvable_mask
from app.engine.worker import run_comparison
from app.models.diff import DiffSubtype
from app.models.job import ComparisonMode

CONFIG = DetectionConfig()
GRAPHICS = AppConfig(detection=replace(DetectionConfig(), scan_compare_graphics=True))  # (a scan's graphics are compared only when asked for)
ORANGE = (238, 106, 45)
GREY = (218, 218, 218)
SENTENCE = "Read the enclosed leaflet carefully before use."


_WORDS = ("Each", "capsule", "contains", "Indacaterol", "Glycopyrronium", "Store", "below", "protect", "from",
          "moisture", "Keep", "out", "reach", "children", "Mfd", "CIPLA", "Batch", "Expiry", "Rx", "only",
          "inhalation", "powder", "dosage", "physician", "not", "swallow", "MNB", "109", "2027", "500", "mcg")


def _varied_print(page: np.ndarray, rows: int, seed: int = 4) -> None:
    """Small print that differs line to line, like real copy (repeated lines confuse feature matching)."""
    rng = np.random.default_rng(seed)
    for row in range(rows):
        words = " ".join(rng.choice(_WORDS, size=int(rng.integers(6, 11))))
        x = int(rng.integers(60, 240))
        cv2.putText(page, words, (x, 60 + row * 27), cv2.FONT_HERSHEY_SIMPLEX, float(rng.uniform(0.42, 0.6)), (95, 95, 95), 1)


def _page(*, sentence: bool = True, box_text: str | None = None, box: bool = False) -> np.ndarray:
    """A grey carton-like page: varied small print, white elements, orange text, a red box."""
    page = np.full((1500, 2100, 3), GREY, np.uint8)
    _varied_print(page, 38)
    cv2.rectangle(page, (1400, 80), (1980, 480), (255, 255, 255), -1)  # a white element beside grey
    cv2.rectangle(page, (1450, 120), (1930, 440), (30, 60, 200), -1)
    cv2.circle(page, (1650, 900), 170, ORANGE, -1)
    cv2.rectangle(page, (200, 1150), (560, 1230), (60, 20, 200), -1)  # small red box...
    cv2.putText(page, "eco-pack" if box_text is None else box_text, (215, 1205), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)  # ...with white text
    if sentence:
        cv2.putText(page, SENTENCE, (250, 1350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ORANGE, 2)
    if box:
        cv2.rectangle(page, (700, 1100), (1150, 1280), (20, 20, 190), -1)
        cv2.putText(page, "SCHEDULE H PRESCRIPTION", (720, 1160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2)
    return page


def _boxes(result) -> list[tuple[float, float, float, float]]:
    return [(c.bbox.x0, c.bbox.y0, c.bbox.x1, c.bbox.y1) for c in result.candidates]


def _covers(candidates, x0, y0, x1, y1) -> bool:
    """Whether the candidates' union covers the given region."""
    grid = np.zeros((y1 + 5, x1 + 5), bool)
    for bx0, by0, bx1, by1 in _boxes_from(candidates):
        grid[int(by0) : int(by1), int(bx0) : int(bx1)] = True
    return bool(grid[y0:y1, x0:x1].mean() > 0.5)


def _boxes_from(candidates):
    return [(c.bbox.x0, c.bbox.y0, c.bbox.x1, c.bbox.y1) for c in candidates]


def _detect(original, suspect, slack=2):
    return structure_diff.detect(original, suspect, CONFIG, slack)


# ---------------------------------------------------------------------------


def test_identical_pages_have_no_differences():
    page = _page()
    assert _detect(page, page.copy()).candidates == []


def test_a_missing_thin_sentence_is_one_finding_covering_it():
    """The headline case. Thin orange print on grey vanished under a global black/white
    threshold; here it is ink like any other."""
    result = _detect(_page(sentence=True), _page(sentence=False))

    assert result.candidates and all(c.subtype is DiffSubtype.REMOVED for c in result.candidates)
    (text_w, _), _ = cv2.getTextSize(SENTENCE, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    assert _covers(result.candidates, 250, 1330, 250 + text_w, 1360)


def test_direction_is_kept_added_versus_removed():
    with_sentence, without = _page(sentence=True), _page(sentence=False)
    assert {c.subtype for c in _detect(with_sentence, without).candidates} == {DiffSubtype.REMOVED}
    assert {c.subtype for c in _detect(without, with_sentence).candidates} == {DiffSubtype.ADDED}


def test_a_grey_background_beside_white_elements_is_not_ink():
    """Regression: 'paper' was taken as the brightest nearby pixel, so grey beside white
    flipped in and out of 'ink' under noise and 4-34% of the page came back different."""
    page = _page()
    # Photometric damage only (no rotation/shift), so no alignment is needed to compare.
    photometric = ScanProfile(rotation_deg=0.0, scale=1.0, shift_px=(0.0, 0.0), blur_sigma=0.6,
                              noise_sigma=3.0, vignette=0.15, brightness=0.93,
                              colour_cast=(1.0, 0.98, 0.95), jpeg_quality=80)
    result = structure_diff.detect(page, simulate_scan(page, photometric), CONFIG, slack_px=2)
    flagged = sum(c.magnitude for c in result.candidates)
    assert flagged < 0.001 * page.shape[0] * page.shape[1]


def test_a_scan_corner_shadow_is_not_a_difference():
    """Regression: white fill outside the scan pulled the background estimate off a
    darkened corner, which then read as ink."""
    page = _page()
    height, width = page.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    radius = np.hypot((xs - width / 2) / (width / 2), (ys - height / 2) / (height / 2))
    shaded = np.clip(page * (1 - 0.25 * np.clip(radius, 0, 1.4) ** 2)[..., None], 0, 255).astype(np.uint8)
    valid = np.ones((height, width), bool)
    valid[:, width - 60 :] = False  # the scan ends here; beyond it is filler

    result = structure_diff.detect(page, shaded, CONFIG, slack_px=2, compare_mask=valid)

    assert sum(c.magnitude for c in result.candidates) < 0.001 * height * width


def test_faint_content_flickering_across_a_threshold_is_not_reported():
    """Hysteresis: a pale tint that is barely ink in one file and barely not in the
    other is not a difference; strong ink absent from the other still is."""
    base = np.full((600, 900, 3), GREY, np.uint8)
    faint_a, faint_b = base.copy(), base.copy()
    cv2.rectangle(faint_a, (100, 100), (400, 300), (208, 208, 208), -1)  # ~5% off the background
    cv2.rectangle(faint_b, (100, 100), (400, 300), (211, 211, 211), -1)
    assert _detect(faint_a, faint_b).candidates == []

    strong = base.copy()
    cv2.circle(strong, (600, 400), 60, (20, 20, 20), -1)
    assert _detect(base, strong).candidates


def test_a_change_inside_a_small_box_is_seen_at_the_small_scale():
    """Regression: with one large background window the whole small red box read as ink
    in both files, hiding a hyphen removed from white text inside it."""
    result = _detect(_page(box_text="eco-pack"), _page(box_text="ecopack"))
    assert result.candidates
    # The change is one hyphen: the finding must sit inside the small box, on the text,
    # rather than be lost in (or blown up to) the box around it.
    for candidate in result.candidates:
        box = candidate.bbox
        assert 200 <= box.x0 and box.x1 <= 575 and 1140 <= box.y0 and box.y1 <= 1240


def test_a_box_added_around_existing_text_is_seen_at_the_large_scale():
    """The mirror image: a small window sees only the added box's outline."""
    result = _detect(_page(box=False), _page(box=True))
    assert _covers(result.candidates, 700, 1100, 1150, 1280)


def test_registration_error_within_the_slack_is_not_a_difference_but_beyond_it_is():
    page = _page()
    for shift, should_flag in [(2, False), (6, True)]:
        moved = np.roll(page, shift, axis=1)
        assert bool(_detect(page, moved, slack=3).candidates) is should_flag


def test_slack_follows_the_alignment_residual_within_bounds():
    slacks = [slack_for(r, CONFIG) for r in (0.0, 0.4, 1.0, 2.0, 50.0)]
    assert slacks == sorted(slacks)
    assert min(slacks) >= CONFIG.structure_min_slack_px
    assert max(slacks) <= CONFIG.structure_max_slack_px


# ---- detail a scan cannot hold ----------------------------------------------------


def _dense_small_print() -> np.ndarray:
    page = np.full((900, 1400, 3), GREY, np.uint8)
    _varied_print(page, 31, seed=9)
    cv2.rectangle(page, (900, 60), (1300, 300), (40, 70, 190), -1)  # a landmark to register on
    cv2.circle(page, (1100, 620), 110, ORANGE, -1)
    return page


def test_a_softer_copy_of_the_same_print_is_compared_at_its_own_softness():
    """A scan is softer than its source. Comparing sharp against soft flags the small
    print as 'removed'; comparing at the softer image's softness flags nothing."""
    sharp = _dense_small_print()
    soft = cv2.GaussianBlur(sharp, (0, 0), 2.6)
    area = sharp.shape[0] * sharp.shape[1]

    for original, suspect in [(sharp, soft), (soft, sharp)]:  # either file may be the soft one
        result = _detect(original, suspect)
        assert 1.8 < result.matched_softness_px < 3.5
        assert sum(c.magnitude for c in result.candidates) < 0.0005 * area


def test_the_softness_measured_follows_the_softness_applied():
    sharp = _dense_small_print()
    measured = [_detect(sharp, cv2.GaussianBlur(sharp, (0, 0), sigma)).matched_softness_px for sigma in (1.0, 2.0, 3.0)]
    assert measured == sorted(measured)
    assert measured[0] == pytest.approx(1.0, abs=0.35)
    assert measured[2] == pytest.approx(3.0, abs=0.7)


def test_matching_softness_does_not_blur_a_missing_thin_sentence_out_of_existence():
    """Regression: blurring the sharp image to the scan's softness cut a 2 px stroke's
    contrast to a third, below the ink threshold, and a missing sentence was not seen."""
    original = _page(sentence=True)
    for sigma in (1.0, 2.0, 3.0):
        soft_suspect = cv2.GaussianBlur(_page(sentence=False), (0, 0), sigma)
        result = _detect(original, soft_suspect)
        assert _covers(result.candidates, 250, 1330, 250 + 480, 1360), f"missed at blur {sigma}"


def test_matching_softness_is_not_fooled_by_a_registration_error():
    """A shifted copy is not a blurred copy: the patches slide while they are scored."""
    page = _dense_small_print()
    result = _detect(page, np.roll(page, 3, axis=1))
    assert result.matched_softness_px == 0.0


def test_a_genuinely_missing_sentence_is_never_mistaken_for_blur():
    """Blur keeps the layout and loses detail; a missing sentence leaves nothing."""
    result = _detect(_page(sentence=True), _page(sentence=False))
    assert result.candidates
    assert _covers(result.candidates, 250, 1330, 250 + 480, 1360)


def test_blank_versus_content_is_not_masked():
    a = _dense_small_print()
    blank = np.full_like(a, GREY)
    mask = unresolvable_mask(a, blank, CONFIG)
    assert mask.mean() < 0.05  # nearly nothing is excused


# ---- noise ----------------------------------------------------------------------------


def test_a_clean_digital_render_measures_as_noiseless():
    assert estimate_noise(cv2.cvtColor(_page(), cv2.COLOR_RGB2GRAY)) == pytest.approx(0.0, abs=0.05)


def test_measured_noise_rises_with_actual_noise():
    gray = cv2.cvtColor(_page(), cv2.COLOR_RGB2GRAY)
    rng = np.random.default_rng(3)
    levels = [
        estimate_noise(np.clip(gray + rng.normal(0, sigma, gray.shape), 0, 255).astype(np.uint8))
        for sigma in (0, 2, 5, 10)
    ]
    assert levels == sorted(levels) and levels[-1] > 3 * levels[1]


# ---- end to end through the pipeline, on simulated scans -----------------------------


@pytest.mark.parametrize("profile", [LIGHT, TYPICAL], ids=["light", "typical"])
def test_a_scan_of_identical_content_reports_nothing(profile):
    """The generic sanity check for any scan path: same content in, no findings out."""
    page = _page()
    scan = simulate_scan(page, profile)

    outcome = run_comparison(page, scan, ComparisonMode.SCAN_VS_DIGITAL, GRAPHICS)

    assert outcome.clusters == []


@pytest.mark.parametrize("profile", [LIGHT, TYPICAL], ids=["light", "typical"])
def test_a_scan_with_a_missing_sentence_points_at_it(profile):
    original = _page(sentence=True)
    scan = simulate_scan(_page(sentence=False), profile)

    outcome = run_comparison(original, scan, ComparisonMode.SCAN_VS_DIGITAL, GRAPHICS)

    assert outcome.clusters
    (text_w, _), _ = cv2.getTextSize(SENTENCE, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    assert any(
        c.bbox_in_original.x0 <= 300 and c.bbox_in_original.x1 >= 250 + 0.9 * text_w
        and c.bbox_in_original.y0 > 1250
        for c in outcome.clusters
    )


def test_a_harsh_scan_degrades_gracefully_rather_than_flooding_the_reviewer():
    """A very soft, noisy, tilted scan of identical content: few findings, none huge."""
    page = _page()
    scan = simulate_scan(page, HARSH)

    outcome = run_comparison(page, scan, ComparisonMode.SCAN_VS_DIGITAL, GRAPHICS)

    page_area = page.shape[0] * page.shape[1]
    # An extreme scan (a 180 DPI, noisy, vignetted JPEG) may leave a few specks at
    # sharp corners, but it must not flood the reviewer: barely any of the page is
    # flagged and nothing is large. (Before the fixes this flagged 4-34% of the page.)
    assert sum(c.magnitude for c in outcome.candidates) < 0.0015 * page_area
    assert len(outcome.clusters) <= 10
    assert all(c.bbox_in_original.area < 0.02 * page_area for c in outcome.clusters)


def test_softness_the_scan_cannot_overcome_is_disclosed_to_the_reviewer():
    original = _dense_small_print()
    soft = cv2.GaussianBlur(original, (0, 0), 3.2)

    outcome = run_comparison(original, soft, ComparisonMode.SCAN_VS_DIGITAL, GRAPHICS)

    assert any("softer than the original" in warning for warning in outcome.warnings)


def test_a_sharp_scan_is_not_warned_about():
    original = _dense_small_print()
    scan = cv2.GaussianBlur(original, (0, 0), 0.7)

    outcome = run_comparison(original, scan, ComparisonMode.SCAN_VS_DIGITAL, GRAPHICS)

    assert not any("softer than the original" in warning for warning in outcome.warnings)


def test_the_structural_pass_is_fast_on_a_full_sheet():
    """An 18 MP sheet, the size of a real carton at 300 DPI (measured ~1.8 s)."""
    original = np.full((4965, 3660, 3), GREY, np.uint8)
    for row in range(150):
        cv2.putText(original, f"BATCH 2109{row:04d} EXP 2027-06 MNB/05/109", (300, 200 + row * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1)
    suspect = original.copy()
    cv2.putText(suspect, SENTENCE, (400, 4700), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ORANGE, 2)

    started = time.perf_counter()
    result = structure_diff.detect(original, suspect, CONFIG, slack_px=2)
    elapsed = time.perf_counter() - started

    assert result.candidates
    assert elapsed < 4.0
