"""Checks against real artwork, when a developer has some.

Real packaging artwork is confidential, so it is never committed:
`sample test files/` is git-ignored, and every test here skips cleanly when the
folder or a listed file is absent.

Nothing in this module (or in the app) is specific to any sample file. Pairs
are listed in `sample test files/pairs.json`, and the expected answer is
*derived from the pixels* rather than written down:

* ``"truth": "pixel_diff"`` is for two digital files that are already
  registered (alignment exact). Their raw pixel difference IS the ground
  truth, so the test groups the differing pixels into objects and demands one
  finding per object, each pointing at it tightly, and nothing pointing at
  anything else.

The same pair is also put through the path for files that do not register exactly (the
suspect rotated, shifted and rescaled a little, as a re-exported file may be) and through
the Scan-vs-Digital path: the suspect is
degraded like a flatbed scan (tests/scan_simulator.py -- generic damage, nothing
tuned to a file) and the answer is still the digital pixel difference.

Other kinds of pair (a mode that ignores colour, ...) can be listed too; they
are skipped until a truth definition exists for them, rather than being checked
against a made-up one.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from scan_simulator import TYPICAL, simulate_scan

from app.config import AppConfig
from app.engine.file_loader import load_artwork
from app.engine.worker import run_comparison
from app.models.job import ComparisonMode

FOLDER = Path(__file__).resolve().parent.parent / "sample test files"
MANIFEST = FOLDER / "pairs.json"

# How far apart two differing pixels may be and still be one physical change,
# for the purpose of *measuring* the answer. Deliberately independent of the
# app's own clustering setting so the test is not marking its own homework.
TRUTH_GROUPING_PX = 41
TRUTH_THRESHOLD = 16


def _pairs():
    if not MANIFEST.is_file():
        return []
    return [entry for entry in json.loads(MANIFEST.read_text(encoding="utf-8"))]


PAIRS = _pairs()

pytestmark = pytest.mark.skipif(
    not PAIRS, reason="no real artwork: 'sample test files/pairs.json' is not present"
)


def _true_boxes(original: np.ndarray, suspect: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Tight boxes of the raw differing pixels, grouped into physical objects."""
    difference = np.abs(original.astype(np.int16) - suspect.astype(np.int16)).max(axis=2)
    mask = (difference > TRUTH_THRESHOLD).astype(np.uint8)
    grouping = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TRUTH_GROUPING_PX, TRUTH_GROUPING_PX))
    count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
        cv2.dilate(mask, grouping), connectivity=8
    )
    boxes = []
    for index in range(1, count):
        ys, xs = np.nonzero((labels == index) & (mask > 0))
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    return sorted(boxes, key=lambda box: (box[1], box[0]))


def _iou(a, b) -> float:
    inter_x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_x * inter_y
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def _coverage(truth, finding) -> float:
    inter_x = max(0, min(truth[2], finding[2]) - max(truth[0], finding[0]))
    inter_y = max(0, min(truth[3], finding[3]) - max(truth[1], finding[1]))
    return inter_x * inter_y / ((truth[2] - truth[0]) * (truth[3] - truth[1]))


@pytest.fixture(scope="module", params=PAIRS, ids=[entry["name"] for entry in PAIRS])
def images(request):
    entry = request.param
    original_path, suspect_path = FOLDER / entry["original"], FOLDER / entry["suspect"]
    if not (original_path.is_file() and suspect_path.is_file()):
        pytest.skip(f"files for '{entry['name']}' are not present")
    if entry.get("truth") != "pixel_diff":
        pytest.skip(f"no truth definition for '{entry['name']}' yet ({entry.get('truth')!r})")

    original, suspect = load_artwork(original_path), load_artwork(suspect_path)
    original_image, suspect_image = original.render(), suspect.render()
    original.close()
    suspect.close()
    return entry, original_image, suspect_image


@pytest.fixture(scope="module")
def compared(images):
    entry, original_image, suspect_image = images
    started = time.perf_counter()
    outcome = run_comparison(
        original_image, suspect_image, ComparisonMode(entry.get("mode", "automatic")), AppConfig()
    )
    elapsed = time.perf_counter() - started
    return outcome, _true_boxes(original_image, suspect_image), elapsed


def _placed_differently(image: np.ndarray) -> np.ndarray:
    """The same artwork rotated, shifted and rescaled slightly -- generic, not tuned to a file."""
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0.6, 1.005)
    matrix[0, 2] += 12
    matrix[1, 2] -= 7
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


@pytest.fixture(scope="module")
def misregistered(images):
    """(outcome on the moved changed file, outcome on the moved identical file, truth)."""
    _entry, original_image, suspect_image = images
    config = AppConfig()
    changed = run_comparison(original_image, _placed_differently(suspect_image), ComparisonMode.AUTOMATIC, config)
    identical = run_comparison(original_image, _placed_differently(original_image), ComparisonMode.AUTOMATIC, config)
    return changed, identical, _true_boxes(original_image, suspect_image)


@pytest.fixture(scope="module")
def scanned(images):
    """(outcome on a scan of the changed file, outcome on a scan of identical content, truth)."""
    _entry, original_image, suspect_image = images
    config = AppConfig()
    changed = run_comparison(
        original_image, simulate_scan(suspect_image, TYPICAL), ComparisonMode.SCAN_VS_DIGITAL, config
    )
    identical = run_comparison(
        original_image, simulate_scan(original_image, TYPICAL), ComparisonMode.SCAN_VS_DIGITAL, config
    )
    return changed, identical, _true_boxes(original_image, suspect_image)


def _finding_boxes(outcome):
    return [
        (c.bbox_in_original.x0, c.bbox_in_original.y0, c.bbox_in_original.x1, c.bbox_in_original.y1)
        for c in outcome.clusters
    ]


def test_alignment_is_recognised_as_exact(compared):
    outcome, _truth, _elapsed = compared
    assert outcome.detection_profile == "exact"
    assert outcome.used_change_regions


def test_there_is_one_finding_per_real_change(compared):
    """The reported complaint: one changed sentence must be one finding, not a
    scatter of boxes on stray letters, and nothing may be reported twice."""
    outcome, truth, _elapsed = compared
    findings = _finding_boxes(outcome)

    assert truth, "the pair was expected to differ"
    assert len(findings) == len(truth)


def test_every_real_change_is_pointed_at_precisely(compared):
    outcome, truth, _elapsed = compared
    findings = _finding_boxes(outcome)

    for box in truth:
        best = max(findings, key=lambda finding: _iou(box, finding))
        assert _coverage(box, best) >= 0.95, f"a real change at {box} is not fully covered"
        assert _iou(box, best) >= 0.85, f"the finding for {box} is too loose or too small"


def test_nothing_points_at_something_that_did_not_change(compared):
    outcome, truth, _elapsed = compared
    for finding in _finding_boxes(outcome):
        assert any(_iou(box, finding) > 0 for box in truth), f"spurious finding at {finding}"


def test_the_comparison_meets_the_time_budget(compared):
    """NFR-1: 3-5 s. Measured ~1.1 s on an 18 MP carton (it was 2.8 s)."""
    _outcome, _truth, elapsed = compared
    assert elapsed < 3.0


# ---- the same pair, scanned ------------------------------------------------------------


def test_a_scan_of_identical_artwork_reports_nothing(scanned):
    """The sanity check that matters most for a scan path: no differences in, none out."""
    _changed, identical, _truth = scanned
    assert identical.clusters == []


def test_a_scan_of_the_changed_artwork_points_at_real_changes_and_only_those(scanned):
    """A scan cannot show everything a digital file does (a faint 2 mm mark is below its
    noise), so this asks for most changes to be found, not all -- and no false alarms."""
    changed, _identical, truth = scanned
    findings = _finding_boxes(changed)

    found = sum(1 for box in truth if max((_iou(box, finding) for finding in findings), default=0) >= 0.3)
    assert found >= 0.5 * len(truth), f"only {found} of {len(truth)} real changes were pointed at"

    for finding in findings:
        assert any(_coverage(box, finding) > 0 for box in truth), f"spurious finding at {finding}"


# ---- the same pair, placed slightly differently ----------------------------------------


def test_identical_artwork_placed_differently_reports_nothing(misregistered):
    _changed, identical, _truth = misregistered
    assert identical.detection_profile == "tolerant"  # it really was not an exact registration
    assert identical.clusters == []


def test_changes_are_still_found_when_the_files_do_not_register_exactly(misregistered):
    """Not every change survives a misregistration (a 2 mm mark is at the noise floor), so most
    of them are asked for -- with no false alarms, and a missing line of print as one finding."""
    changed, _identical, truth = misregistered
    findings = _finding_boxes(changed)

    found = sum(1 for box in truth if max((_iou(box, finding) for finding in findings), default=0) >= 0.3)
    assert found >= 0.7 * len(truth), f"only {found} of {len(truth)} real changes were pointed at"
    for finding in findings:
        assert any(_coverage(box, finding) > 0 for box in truth), f"spurious finding at {finding}"
    assert len(findings) <= len(truth)  # one finding per change, not a scatter
