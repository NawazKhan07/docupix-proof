"""The reported need: compare the artwork with the artwork, and nothing outside its border.

A packaging sheet carries the carton's die-cut outline and, around it, a title, a colour legend, crop
marks and notes. None of that is artwork. Lifting the artwork from each file and comparing inside the
lifted areas leaves everything outside them out -- whatever differs there.
"""

from __future__ import annotations

import pickle

import cv2
import numpy as np
import subject_scenes as scenes

from app.config import AppConfig, SubjectConfig
from app.engine.isolation.subject_lifter import lift_subject
from app.engine.worker import run_comparison
from app.models.isolation import IsolationGeometry
from app.models.job import ComparisonMode

CONFIG = AppConfig()


def _pair():
    """Two sheets that differ in three places: the title and the legend (outside the carton), and a word of
    body copy (inside it)."""
    original, truth, _ = scenes.dieline_sheet()
    suspect = original.copy()
    cv2.rectangle(suspect, (320, 100), (1000, 160), (255, 255, 255), -1)  # the title, reworded
    cv2.putText(suspect, "Carton Size : L x W x H = 40 x 40 x 80 mm", (330, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)
    cv2.rectangle(suspect, (120, 1370), (300, 1410), (200, 30, 40), -1)  # a legend swatch recoloured
    cv2.rectangle(suspect, (240, 675), (520, 715), (226, 238, 246), -1)  # a line of body copy removed, inside the carton
    return original, suspect, truth


def _lifted(image) -> IsolationGeometry:
    subject = lift_subject(image, (600, 800), SubjectConfig())
    return IsolationGeometry(boundary=subject.normalized_polygon(), boundary_origin="lifted")


def _boxes(outcome):
    return [(c.bbox_in_original.x0, c.bbox_in_original.y0, c.bbox_in_original.x1, c.bbox_in_original.y1) for c in outcome.clusters]


def _inside(box, region: np.ndarray) -> bool:
    x0, y0, x1, y1 = (int(v) for v in box)
    return bool(region[max(0, y0) : y1 + 1, max(0, x0) : x1 + 1].all())


def test_without_isolating_the_artwork_everything_on_the_sheet_is_compared():
    original, suspect, truth = _pair()
    boxes = _boxes(run_comparison(original, suspect, ComparisonMode.AUTOMATIC, CONFIG))
    region = cv2.dilate(truth.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)

    assert any(not _inside(box, region) for box in boxes)  # the title and legend show up as differences
    assert any(_inside(box, region) for box in boxes)


def test_with_the_artwork_lifted_only_what_differs_inside_the_border_is_reported():
    original, suspect, truth = _pair()
    outcome = run_comparison(
        original, suspect, ComparisonMode.AUTOMATIC, CONFIG, original_geometry=_lifted(original), suspect_geometry=_lifted(suspect)
    )
    region = cv2.dilate(truth.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)

    boxes = _boxes(outcome)
    assert boxes, "the change inside the carton must still be found"
    assert all(_inside(box, region) for box in boxes), boxes
    # ...and it is the body-copy change, not the title (y ~ 140) or the legend (y ~ 1390)
    assert any(box[1] < 720 and box[3] > 675 for box in boxes)
    assert not any(box[3] < 300 or box[1] > 1300 for box in boxes)


def test_lifting_only_one_file_still_confines_the_comparison():
    """The comparison area is what BOTH files' masks leave: lifting the original alone is enough to drop the
    legend and title from the comparison."""
    original, suspect, truth = _pair()
    outcome = run_comparison(original, suspect, ComparisonMode.AUTOMATIC, CONFIG, original_geometry=_lifted(original))
    region = cv2.dilate(truth.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    assert outcome.clusters and all(_inside(box, region) for box in _boxes(outcome))


def test_the_lifted_polygons_do_not_register_the_files_the_artworks_features_do():
    """Registering on outlines that were segmented would report a fit that is largely an artefact of how they
    were drawn: the residual reported must be the honest keypoint one."""
    original, suspect, _ = _pair()
    outcome = run_comparison(
        original, suspect, ComparisonMode.AUTOMATIC, CONFIG, original_geometry=_lifted(original), suspect_geometry=_lifted(suspect)
    )
    assert outcome.alignment.residual_basis == "keypoints" and "dieline_polygon" not in outcome.alignment.method


def test_a_lifted_geometry_survives_the_trip_to_the_worker_process():
    geometry = _lifted(scenes.dieline_sheet()[0])
    again = pickle.loads(pickle.dumps(geometry))
    assert again.boundary_origin == "lifted" and not again.anchors_alignment
    assert again.boundary.points == geometry.boundary.points
