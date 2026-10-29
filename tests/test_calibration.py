"""The pixel comparison calibrated to each pair's own noise (`local_diff.measure_noise`, `calibrated`, hysteresis)."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from app.config import DetectionConfig
from app.engine.detectors import local_diff

CONFIG = DetectionConfig()


def _page(text_rows: int = 12, seed: int = 0) -> np.ndarray:
    page = np.full((900, 1200, 3), 255, np.uint8)
    rng = np.random.default_rng(seed)
    for row in range(text_rows):
        words = " ".join("".join(rng.choice(list("abcdefghiklmnoprstuvw"), 5)) for _ in range(6))
        cv2.putText(page, words, (40, 60 + row * 65), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (30, 30, 30), 2, cv2.LINE_AA)
    return page


def _noisy(page: np.ndarray) -> np.ndarray:
    """The same page drawn a little differently all over: every stroke's edges a shade off (another renderer, a softer
    export) -- differences of a few tens of levels at the edges, and none of them a change."""
    return cv2.GaussianBlur(page, (0, 0), 1.0)


def test_two_identical_renders_have_no_noise():
    page = _page()
    assert local_diff.measure_noise(page, page.copy(), CONFIG) == 0.0


def test_a_pair_drawn_differently_all_over_has_measurable_noise():
    page = _page()
    assert local_diff.measure_noise(page, _noisy(page), CONFIG) > 0.0


def test_a_few_real_changes_do_not_count_as_the_pairs_noise():
    page = _page()
    changed = page.copy()
    cv2.rectangle(changed, (60, 30), (400, 80), (255, 255, 255), -1)  # one line's words gone
    assert local_diff.measure_noise(page, changed, CONFIG) == 0.0


def test_calibration_never_lowers_the_bar_below_the_configured_one():
    assert local_diff.calibrated(CONFIG, 0.0).local_proof_tolerance == 0
    assert local_diff.calibrated(CONFIG, 10.0).local_proof_tolerance == 0  # 2 x 10 is under the ordinary bar
    assert local_diff.calibrated(CONFIG, 20.0).local_proof_tolerance == 40
    assert local_diff.calibrated(CONFIG, 90.0).local_proof_tolerance == CONFIG.local_tolerance_max
    assert local_diff.calibrated(replace(CONFIG, local_calibrate=False), 90.0).local_proof_tolerance == 0


def test_a_noisy_pair_keeps_its_real_change_whole_and_loses_its_noise():
    page = _page()
    suspect = _noisy(page)
    cv2.putText(suspect, "NEW", (900, 860), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (200, 20, 20), 3, cv2.LINE_AA)  # the real change

    plain = local_diff.detect(page, suspect, CONFIG).candidates
    noise = local_diff.measure_noise(page, suspect, CONFIG)
    tuned = local_diff.detect(page, suspect, local_diff.calibrated(CONFIG, noise)).candidates

    def covers_new(candidates):
        return [c for c in candidates if c.bbox.x0 < 1000 and c.bbox.x1 > 880 and c.bbox.y1 > 820]

    assert covers_new(tuned), "the real change must still be found"
    assert len(plain) > 10  # the ordinary bar flags the pair's noise all over the page...
    assert len(tuned) == len(covers_new(tuned)) == 1  # ...calibrated, only the real change is left
    # its extent is not cut down by the higher bar: the same box as the ordinary comparison gives it
    (new_plain,), (new_tuned,) = covers_new(plain)[:1], covers_new(tuned)[:1]
    assert abs(new_plain.bbox.x0 - new_tuned.bbox.x0) <= 2 and abs(new_plain.bbox.x1 - new_tuned.bbox.x1) <= 2
