"""FR-7.4: a difference must be marked at the corresponding place in both files.

The build prompt calls this "a hard correctness requirement" and asks for a
test that projected boxes round-trip. The algebraic round-trips are here, but
the test that actually proves the requirement is the last one: after a real
alignment of a rotated pair, each file's own box must land on the same
physical change in that file's own pixels.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.config import AppConfig
from app.engine.aggregation.coordinate_projection import (
    original_to_suspect,
    project_box,
)
from app.engine.alignment.homography import apply_transform
from app.engine.worker import run_comparison
from app.models.diff import BoundingBox, DiffType
from app.models.job import ComparisonMode


def _rotation(angle: float, tx: float = 0.0, ty: float = 0.0, centre=(300.0, 200.0)) -> np.ndarray:
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    return np.vstack([matrix, [0.0, 0.0, 1.0]])


def test_projecting_through_the_identity_changes_nothing():
    box = BoundingBox(10, 20, 110, 90)
    assert project_box(box, np.eye(3)) == box


def test_a_pure_translation_round_trips_exactly():
    translation = np.array([[1.0, 0.0, 25.0], [0.0, 1.0, -10.0], [0.0, 0.0, 1.0]])
    box = BoundingBox(100, 100, 300, 250)

    in_suspect = original_to_suspect(box, translation)
    assert in_suspect == BoundingBox(75, 110, 275, 260)

    back = project_box(in_suspect, translation)
    assert back == box


@pytest.mark.parametrize("angle", [0.5, 1.5, 4.0, -3.0, 12.0])
def test_box_corners_round_trip_exactly_through_the_transform(angle):
    transform = _rotation(angle, tx=15, ty=-8)
    corners = np.array([[100.0, 100.0], [300.0, 100.0], [300.0, 250.0], [100.0, 250.0]])

    out_and_back = apply_transform(transform, apply_transform(np.linalg.inv(transform), corners))

    assert np.allclose(out_and_back, corners, atol=1e-9)


@pytest.mark.parametrize("angle", [0.5, 1.5, 4.0, -3.0, 12.0])
def test_a_round_tripped_box_always_contains_the_original(angle):
    """Under rotation the axis-aligned box grows, but must never shrink."""
    transform = _rotation(angle, tx=15, ty=-8)
    box = BoundingBox(100, 100, 300, 250)

    round_trip = project_box(original_to_suspect(box, transform), transform)

    assert round_trip.x0 <= box.x0 + 1e-9 and round_trip.y0 <= box.y0 + 1e-9
    assert round_trip.x1 >= box.x1 - 1e-9 and round_trip.y1 >= box.y1 - 1e-9


def test_box_inflation_stays_small_for_realistic_alignments():
    """Real misalignments are a degree or two; the marker must not balloon."""
    box = BoundingBox(100, 100, 300, 250)
    round_trip = project_box(original_to_suspect(box, _rotation(1.5)), _rotation(1.5))
    assert round_trip.area < box.area * 1.15


def _artwork() -> np.ndarray:
    image = np.full((400, 600, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (40, 30), (560, 370), (0, 0, 0), 3)
    cv2.putText(image, "PARACETAMOL 500mg", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
    cv2.putText(image, "Tablets BP x16", (70, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)
    cv2.rectangle(image, (70, 200), (240, 290), (200, 30, 40), -1)  # red block
    cv2.circle(image, (430, 240), 50, (30, 90, 200), -1)
    for index in range(12):
        x = 300 + index * 18
        cv2.line(image, (x, 300), (x, 350), (0, 0, 0), 3)
    return image


def test_each_files_own_box_lands_on_the_same_physical_change():
    """The requirement itself, end to end (FR-7.4).

    The suspect is a rotated and shifted copy whose red block was recoloured.
    After a real alignment, the finding's box in the ORIGINAL's pixels must sit
    on the red block, and its box in the SUSPECT's own (unwarped) pixels must
    sit on the green block, at the position the block actually moved to.
    """
    original = _artwork()

    changed = original.copy()
    cv2.rectangle(changed, (70, 200), (240, 290), (40, 160, 60), -1)  # red -> green
    forward = _rotation(2.0, tx=14, ty=9)
    suspect = cv2.warpAffine(changed, forward[:2], (600, 400), borderValue=(255, 255, 255))

    outcome = run_comparison(original, suspect, ComparisonMode.AUTOMATIC, AppConfig())

    colour = [c for c in outcome.clusters if c.type is DiffType.COLOUR]
    assert colour, "the recoloured block should have been found"
    finding = max(colour, key=lambda c: c.bbox_in_original.area)

    # In the original's own pixels the box is over the red block.
    ox, oy = (int(v) for v in finding.bbox_in_original.center)
    assert tuple(original[oy, ox]) == (200, 30, 40)

    # In the suspect's own pixels it is over the green block -- not merely
    # somewhere plausible, but on the block's true position after the warp.
    sx, sy = (int(round(v)) for v in finding.bbox_in_suspect.center)
    assert tuple(suspect[sy, sx]) == (40, 160, 60)

    # Measured error across 0.5-5 degree rotations is at most ~2.6 px (the
    # detected region's edge halo plus the axis-aligned box's inflation), so
    # 5 px is tight enough that real drift would fail this, not slip under it.
    true_block = project_box(BoundingBox(70, 200, 240, 290), forward)
    tolerance = 5.0
    assert finding.bbox_in_suspect.x0 == pytest.approx(true_block.x0, abs=tolerance)
    assert finding.bbox_in_suspect.y0 == pytest.approx(true_block.y0, abs=tolerance)
    assert finding.bbox_in_suspect.x1 == pytest.approx(true_block.x1, abs=tolerance)
    assert finding.bbox_in_suspect.y1 == pytest.approx(true_block.y1, abs=tolerance)


def test_the_two_boxes_are_the_same_region_seen_through_one_transform():
    """Nothing re-aligns per view: suspect box -> original frame is the original box."""
    original = _artwork()
    changed = original.copy()
    cv2.rectangle(changed, (70, 200), (240, 290), (40, 160, 60), -1)
    suspect = cv2.warpAffine(changed, _rotation(1.5, tx=10, ty=6)[:2], (600, 400), borderValue=(255, 255, 255))

    outcome = run_comparison(original, suspect, ComparisonMode.AUTOMATIC, AppConfig())

    for cluster in outcome.clusters:
        back = project_box(cluster.bbox_in_suspect, outcome.alignment.transform)
        # The suspect-side box was inflated by the rotation, so what comes back
        # must contain the original box -- never miss part of what was found.
        assert back.x0 <= cluster.bbox_in_original.x0 + 1e-6
        assert back.y0 <= cluster.bbox_in_original.y0 + 1e-6
        assert back.x1 >= cluster.bbox_in_original.x1 - 1e-6
        assert back.y1 >= cluster.bbox_in_original.y1 - 1e-6
