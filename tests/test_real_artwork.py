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

Other kinds of pair (a scan, a mode that ignores colour, ...) can be listed
too; they are skipped until a truth definition exists for them, rather than
being checked against a made-up one.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

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
def compared(request):
    entry = request.param
    original_path, suspect_path = FOLDER / entry["original"], FOLDER / entry["suspect"]
    if not (original_path.is_file() and suspect_path.is_file()):
        pytest.skip(f"files for '{entry['name']}' are not present")
    if entry.get("truth") != "pixel_diff":
        pytest.skip(f"no truth definition for '{entry['name']}' yet ({entry.get('truth')!r})")

    original, suspect = load_artwork(original_path), load_artwork(suspect_path)
    original_image, suspect_image = original.render(), suspect.render()

    started = time.perf_counter()
    outcome = run_comparison(
        original_image, suspect_image, ComparisonMode(entry.get("mode", "automatic")), AppConfig()
    )
    elapsed = time.perf_counter() - started
    truth = _true_boxes(original_image, suspect_image)
    original.close()
    suspect.close()
    return outcome, truth, elapsed


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
