"""The results show which part of the page was compared, so what was left out is visible rather than assumed."""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np
import pytest
import subject_scenes as scenes

from app.config import AppConfig, SubjectConfig
from app.engine.isolation.subject_lifter import lift_subject
from app.engine.worker import compared_area_outline, run_comparison
from app.models.isolation import IsolationGeometry, NormalizedRect
from app.models.job import ComparisonMode
from app.ui.pane_view import PaneView
from app.ui.results_view import ResultsView


def _mask(shape=(800, 1000), box=(200, 150, 700, 600)) -> np.ndarray:
    mask = np.zeros(shape, bool)
    mask[box[1] : box[3], box[0] : box[2]] = True
    return mask


# ---- the outline of what was compared ---------------------------------------------------------------------


def test_nothing_left_out_means_no_outline():
    assert compared_area_outline(np.ones((400, 500), bool)) is None


def test_a_region_gives_its_boundary_in_the_masks_own_pixels():
    (polygon,) = compared_area_outline(_mask())
    xs = [x for x, _ in polygon]
    ys = [y for _, y in polygon]
    assert (min(xs), min(ys)) == pytest.approx((200, 150), abs=2)
    assert (max(xs), max(ys)) == pytest.approx((699, 599), abs=2)


def test_the_outline_is_in_full_size_pixels_however_large_the_page():
    big = np.zeros((4965, 3660), bool)
    big[600:2800, 700:3000] = True
    (polygon,) = compared_area_outline(big)
    assert min(x for x, _ in polygon) == pytest.approx(700, abs=3)
    assert max(y for _, y in polygon) == pytest.approx(2800, abs=3)


def test_an_erased_part_is_reported_as_a_hole():
    mask = _mask()
    mask[300:400, 400:500] = False
    polygons = compared_area_outline(mask)
    assert len(polygons) == 2  # the outer boundary and the hole in it


def test_specks_of_the_mask_are_not_shown():
    mask = _mask()
    mask[10:13, 10:13] = True
    assert len(compared_area_outline(mask)) == 1


def test_a_comparison_reports_what_it_covered():
    sheet, truth, _ = scenes.dieline_sheet()
    subject = lift_subject(sheet, (600, 800), SubjectConfig())
    geometry = IsolationGeometry(boundary=subject.normalized_polygon(), boundary_origin="lifted")

    whole = run_comparison(sheet, sheet.copy(), ComparisonMode.AUTOMATIC, AppConfig())
    isolated = run_comparison(sheet, sheet.copy(), ComparisonMode.AUTOMATIC, AppConfig(), original_geometry=geometry)

    assert whole.compared_outline is None
    xs = [x for polygon in isolated.compared_outline for x, _ in polygon]
    ys = [y for polygon in isolated.compared_outline for _, y in polygon]
    x0, y0, x1, y1 = subject.bbox
    assert (min(xs), min(ys), max(xs), max(ys)) == pytest.approx((x0, y0, x1, y1), abs=4)


def test_a_crop_is_reported_too():
    sheet, _truth, _ = scenes.dieline_sheet()
    crop = IsolationGeometry(boundary=NormalizedRect(0.2, 0.3, 0.8, 0.7))
    outcome = run_comparison(sheet, sheet.copy(), ComparisonMode.AUTOMATIC, AppConfig(), original_geometry=crop)
    assert len(outcome.compared_outline) == 1


# ---- drawing it -----------------------------------------------------------------------------------------------------


def test_a_pane_dims_what_was_left_out_and_outlines_what_was_covered(qapp, outcome):
    pane = PaneView()
    from app.ui.pane_view import ImageStack

    pane.set_stacks({"original": ImageStack(outcome.aligned_original)})
    before = len(pane._scene.items())
    height, width = outcome.aligned_original.shape[:2]

    pane.set_compared_area([[(width * 0.2, height * 0.2), (width * 0.8, height * 0.2), (width * 0.8, height * 0.8), (width * 0.2, height * 0.8)]])

    assert len(pane._scene.items()) == before + 2 and pane.compared_area_visible
    pane.show_compared_area(False)
    assert len(pane._scene.items()) == before and not pane.compared_area_visible
    pane.show_compared_area(True)
    pane.set_compared_area(None)
    assert len(pane._scene.items()) == before


def test_the_dimming_lets_clicks_through_to_the_findings(qapp, outcome):
    from app.ui.pane_view import ImageStack

    pane = PaneView()
    pane.set_stacks({"original": ImageStack(outcome.aligned_original)})
    pane.set_compared_area([[(0, 0), (100, 0), (100, 100)]])
    from PySide6.QtCore import Qt

    assert pane._outside_item.acceptedMouseButtons() == Qt.MouseButton.NoButton


@pytest.fixture
def results(qapp, outcome):
    view = ResultsView()
    view.resize(1400, 800)
    view.show()
    yield view
    view.close()


def test_the_results_page_offers_the_toggle_only_when_part_of_the_page_was_left_out(results, outcome, qapp):
    results.show_outcome(dataclasses.replace(outcome, compared_outline=None), "a  vs  b")
    qapp.processEvents()
    assert results.viewport.compared_area_check.isHidden()

    height, width = outcome.aligned_original.shape[:2]
    area = [[(width * 0.1, height * 0.1), (width * 0.9, height * 0.1), (width * 0.9, height * 0.9), (width * 0.1, height * 0.9)]]
    results.show_outcome(dataclasses.replace(outcome, compared_outline=area), "a  vs  b")
    qapp.processEvents()
    check = results.viewport.compared_area_check
    assert not check.isHidden() and check.isChecked()
    assert results.viewport.original_pane.compared_area_visible and results.viewport.suspect_pane.compared_area_visible


def test_the_toggle_hides_and_shows_it_in_both_panes(results, outcome, qapp):
    height, width = outcome.aligned_original.shape[:2]
    area = [[(width * 0.1, height * 0.1), (width * 0.9, height * 0.1), (width * 0.9, height * 0.9)]]
    results.show_outcome(dataclasses.replace(outcome, compared_outline=area), "a  vs  b")

    results.viewport.compared_area_check.setChecked(False)
    assert not results.viewport.original_pane.compared_area_visible and not results.viewport.suspect_pane.compared_area_visible
    results.viewport.compared_area_check.setChecked(True)
    assert results.viewport.original_pane.compared_area_visible


def test_a_new_result_replaces_the_old_areas_drawing(results, outcome, qapp):
    height, width = outcome.aligned_original.shape[:2]
    area = [[(10, 10), (width - 10, 10), (width - 10, height - 10)]]
    results.show_outcome(dataclasses.replace(outcome, compared_outline=area), "x")
    results.show_outcome(dataclasses.replace(outcome, compared_outline=None), "y")
    assert not results.viewport.original_pane.compared_area_visible
    assert results.viewport.compared_area_check.isHidden()
