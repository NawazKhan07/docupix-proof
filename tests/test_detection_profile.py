"""Detection adapts to how well the two files align -- not to which files they are."""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from app.config import AppConfig, DetectionConfig
from app.engine.worker import effective_detection_config, run_comparison
from app.models.diff import DiffType
from app.models.job import ComparisonMode
from app.models.result import AlignmentResult

ORANGE = (238, 106, 45)
GREY = (218, 218, 218)
SENTENCE = "Read the enclosed leaflet carefully before use."


def _alignment(residual: float) -> AlignmentResult:
    return AlignmentResult(
        transform=np.eye(3),
        residual_error_px=residual,
        residual_basis="keypoints",
        method="features",
        match_count=100,
        inlier_count=100,
        inlier_ratio=1.0,
        low_confidence=False,
        threshold_px=2.0,
    )


def test_an_exact_alignment_switches_every_noise_filter_off():
    config, profile = effective_detection_config(DetectionConfig(), _alignment(0.0))

    assert profile == "exact"
    assert config.registration_tolerance_px == 0
    assert config.color_registration_tolerance_px == 0
    assert config.edge_tolerance_px == 0  # the opening that erased thin strokes
    assert config.color_min_area_px == config.exact_min_area_px
    assert config.ssim_min_area_px == config.exact_min_area_px
    assert config.registration_intensity_tolerance == config.exact_intensity_tolerance


def test_a_poor_alignment_keeps_the_noise_filters():
    original = DetectionConfig()
    config, profile = effective_detection_config(original, _alignment(0.9))

    assert profile == "tolerant"
    assert config == original


def test_the_exactness_threshold_is_configurable():
    strict = DetectionConfig(exact_alignment_residual_px=0.05)
    assert effective_detection_config(strict, _alignment(0.1))[1] == "tolerant"
    lenient = DetectionConfig(exact_alignment_residual_px=0.5)
    assert effective_detection_config(lenient, _alignment(0.4))[1] == "exact"


def _big_page(with_sentence: bool) -> np.ndarray:
    """A large sheet (the size of a real carton at 300 DPI) with small orange print on grey."""
    page = np.full((4965, 3660, 3), GREY, np.uint8)
    for row in range(120):
        cv2.putText(page, f"BATCH 2109{row:04d} EXP 2027-06 MNB/05/109 Mfd by CIPLA", (300, 200 + row * 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1)
    cv2.rectangle(page, (2500, 300), (3300, 900), (0, 60, 200), -1)
    cv2.circle(page, (2900, 2600), 220, ORANGE, -1)
    if with_sentence:
        cv2.putText(page, SENTENCE, (700, 3000), cv2.FONT_HERSHEY_SIMPLEX, 0.65, ORANGE, 2)
    return page


def test_a_missing_sentence_on_a_large_sheet_is_one_finding_covering_it():
    """The reported bug: a whole missing sentence came back as tiny boxes on stray letters."""
    original, suspect = _big_page(True), _big_page(False)
    (text_width, _), _ = cv2.getTextSize(SENTENCE, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)

    outcome = run_comparison(original, suspect, ComparisonMode.AUTOMATIC, AppConfig())

    assert outcome.detection_profile == "exact"
    assert len(outcome.clusters) == 1
    box = outcome.clusters[0].bbox_in_original
    # One box, spanning the whole sentence rather than a few of its letters.
    assert box.x0 <= 700 + 10 and box.x1 >= 700 + text_width - 10
    assert box.width >= 0.95 * text_width


def test_the_old_behaviour_was_fragmentary_which_is_why_the_profile_exists():
    """Forcing the previous configuration reproduces the reported failure, so this
    test would notice if the fix were ever quietly undone."""
    original, suspect = _big_page(True), _big_page(False)
    (text_width, _), _ = cv2.getTextSize(SENTENCE, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    old_behaviour = dataclasses.replace(
        DetectionConfig(),
        exact_alignment_residual_px=-1.0,
        region_max_area_fraction=0.0,
        graphics_when_misregistered="ssim",  # the similarity map, which is what erased thin print
        local_enabled=False,  # (the full-size local comparison came later, and sees the whole sentence too)
    )

    outcome = run_comparison(
        original, suspect, ComparisonMode.AUTOMATIC, AppConfig(detection=old_behaviour)
    )

    assert not outcome.used_change_regions
    assert all(c.bbox_in_original.width < 0.2 * text_width for c in outcome.clusters)


def test_a_missing_sentence_is_one_finding_even_when_the_files_do_not_register_exactly():
    """The same complaint on the path used for rotated or shifted files: asking where ink is
    (rather than the similarity map's noise-filtered result) keeps the whole sentence."""
    original, suspect = _big_page(True), _big_page(False)
    (text_width, _), _ = cv2.getTextSize(SENTENCE, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    misregistered = dataclasses.replace(
        DetectionConfig(), exact_alignment_residual_px=-1.0, region_max_area_fraction=0.0
    )

    outcome = run_comparison(original, suspect, ComparisonMode.AUTOMATIC, AppConfig(detection=misregistered))

    assert outcome.detection_profile == "tolerant"
    assert len(outcome.clusters) == 1
    box = outcome.clusters[0].bbox_in_original
    assert box.x0 <= 700 + 10 and box.x1 >= 700 + text_width - 10


def test_a_tiny_difference_is_still_found_when_alignment_is_exact():
    """A missing decimal point is small and critical: exact alignment must catch it."""
    original = _big_page(True)
    suspect = original.copy()
    cv2.circle(suspect, (1500, 3500), 3, (0, 0, 0), -1)  # a ~30 px dot

    outcome = run_comparison(original, suspect, ComparisonMode.AUTOMATIC, AppConfig())

    assert len(outcome.clusters) == 1
    box = outcome.clusters[0].bbox_in_original
    assert box.x0 <= 1500 <= box.x1 and box.y0 <= 3500 <= box.y1


def test_an_identical_pair_reports_nothing_and_skips_the_expensive_passes(monkeypatch):
    import app.engine.worker as worker

    calls = []
    monkeypatch.setattr(worker.color_diff, "detect", lambda *a, **k: calls.append("colour") or [])
    monkeypatch.setattr(worker.graphics_diff, "detect", lambda *a, **k: calls.append("graphics") or [])

    page = _big_page(True)
    outcome = run_comparison(page, page.copy(), ComparisonMode.AUTOMATIC, AppConfig())

    assert outcome.clusters == []
    assert calls == []  # no changed region, so neither pass ran at all


def test_one_change_is_one_finding_not_a_colour_box_inside_a_graphics_box():
    """A new logo differs in colour and in structure; that is still one change."""
    original = _big_page(False)
    suspect = original.copy()
    cv2.putText(suspect, "Cipla", (900, 2000), cv2.FONT_HERSHEY_DUPLEX, 3.0, ORANGE, 6)

    outcome = run_comparison(original, suspect, ComparisonMode.AUTOMATIC, AppConfig())

    assert len(outcome.clusters) == 1  # (however many detectors saw it: one place, one finding)
    assert outcome.clusters[0].category in ("graphics", "text")


def test_findings_are_labelled_by_their_most_specific_kind():
    original = _big_page(False)
    suspect = original.copy()
    cv2.putText(suspect, "Cipla", (900, 2000), cv2.FONT_HERSHEY_DUPLEX, 3.0, ORANGE, 6)

    outcome = run_comparison(original, suspect, ComparisonMode.AUTOMATIC, AppConfig())

    assert outcome.clusters[0].type is DiffType.GRAPHICS  # not merely "colour"


def test_the_pipeline_is_much_faster_on_a_mostly_identical_sheet():
    """The measured motivation: colour + graphics over a whole 18 MP sheet was ~1.6 s."""
    import time

    original, suspect = _big_page(True), _big_page(False)

    started = time.perf_counter()
    run_comparison(original, suspect, ComparisonMode.AUTOMATIC, AppConfig())
    elapsed = time.perf_counter() - started

    assert elapsed < 3.0  # NFR-1's window, with margin (measured ~0.6 s)


# ---- exact is not zero: tolerances follow the residual that was measured ----------------------------------------------


def _alignment(residual: float):
    from app.models.result import AlignmentResult

    return AlignmentResult(
        transform=np.eye(3), residual_error_px=residual, residual_basis="keypoints", method="features",
        match_count=100, inlier_count=100, inlier_ratio=1.0, low_confidence=False, threshold_px=2.0,
    )


def test_a_cleaner_registration_keeps_the_sensitive_defaults():
    config, profile = effective_detection_config(DetectionConfig(), _alignment(0.004))
    assert profile == "exact"
    assert config.registration_intensity_tolerance == DetectionConfig().exact_intensity_tolerance
    assert config.color_delta_e_threshold == DetectionConfig().color_delta_e_threshold


def test_a_residual_of_a_few_hundredths_of_a_pixel_raises_the_tolerances_to_match():
    """Measured: 0.019 px moved a black-on-white edge by ~5 grey levels, which at full sensitivity
    came back as hairline differences along every strong edge."""
    low, _ = effective_detection_config(DetectionConfig(), _alignment(0.004))
    higher, profile = effective_detection_config(DetectionConfig(), _alignment(0.019))
    assert profile == "exact"
    assert higher.registration_intensity_tolerance > low.registration_intensity_tolerance
    assert higher.registration_intensity_tolerance >= 0.019 * 255  # at least what the shift explains
    assert higher.color_delta_e_threshold > low.color_delta_e_threshold


def test_the_tolerances_are_capped_so_a_real_change_is_never_explained_away():
    config, _ = effective_detection_config(DetectionConfig(), _alignment(0.24))
    assert config.registration_intensity_tolerance <= DetectionConfig().exact_max_intensity_tolerance
    assert config.color_delta_e_threshold <= DetectionConfig().exact_max_delta_e


def test_a_worse_registration_is_still_tolerant_not_exact():
    _config, profile = effective_detection_config(DetectionConfig(), _alignment(0.6))
    assert profile == "tolerant"
