"""The viewer and the findings list must never disagree about what is shown."""

from __future__ import annotations

import pytest

from app.ui.composition import ViewMode
from app.ui.results_view import ResultsView


@pytest.fixture
def results(qapp, outcome):
    view = ResultsView()
    view.resize(1400, 800)
    view.show()
    view.show_outcome(outcome, "original.pdf  vs  suspect.pdf  ·  Automatic")
    qapp.processEvents()
    yield view
    view.close()


def test_the_result_is_shown_in_both_the_viewer_and_the_list(results, outcome):
    assert results.viewport.has_images
    assert results.panel.title_label.text() == f"Findings ({len(outcome.clusters)})"
    assert results.viewport.original_pane.marker_count == len(outcome.clusters)
    assert "suspect.pdf" in results.subtitle_label.text()


def test_choosing_a_finding_jumps_the_viewer_to_it(results, outcome):
    target = outcome.clusters[-1]

    results.panel._on_card_selected(target.id)

    assert results.viewport._selected_id == target.id


def test_clicking_a_marker_selects_its_finding_in_the_list(results, outcome):
    target = outcome.clusters[0]

    results.viewport.markerClicked.emit(target.id)

    assert results.panel._selected_id == target.id
    assert results.viewport._selected_id == target.id


def test_choosing_the_selected_finding_again_deselects_it_in_both(results, outcome):
    target = outcome.clusters[-1]
    results.panel._on_card_selected(target.id)

    results.panel._on_card_selected(target.id)  # click it again

    assert results.panel._selected_id is None
    assert results.viewport._selected_id is None


def test_ignoring_a_finding_removes_its_marker_from_every_pane(results, outcome):
    """FR-7.5 with FR-7.4: a dismissed finding disappears from both views."""
    target = outcome.clusters[0]
    total = len(outcome.clusters)

    results.panel._on_ignore_toggled(target.id)

    assert results.viewport.original_pane.marker_count == total - 1
    assert results.viewport.suspect_pane.marker_count == total - 1
    assert target.id not in results.viewport.original_pane._marker_items
    assert target.id not in results.viewport.suspect_pane._marker_items


def test_restoring_a_finding_brings_its_marker_back(results, outcome):
    target = outcome.clusters[0]
    results.panel._on_ignore_toggled(target.id)
    results.panel.show_ignored_check.setChecked(True)

    results.panel._on_ignore_toggled(target.id)  # restore

    assert target.id in results.viewport.original_pane._marker_items
    assert target.id in results.viewport.suspect_pane._marker_items


def test_filtering_the_list_filters_the_markers(results, outcome):
    results.panel.category_combo.setCurrentIndex(results.panel.category_combo.findData("graphics"))

    graphics = [c for c in outcome.clusters if c.category == "graphics"]
    assert results.viewport.original_pane.marker_count == len(graphics)
    assert results.viewport.suspect_pane.marker_count == len(graphics)


def test_ignored_findings_do_not_flicker_in_grayscale_mode(results, outcome):
    """Grayscale colours only the regions of active findings."""
    results.viewport.set_view_mode(ViewMode.GRAYSCALE)
    target = outcome.clusters[0]
    box = target.bbox_in_original
    cx, cy = int(box.center[0]), int(box.center[1])

    before = results.viewport._stacks["gray0"]._array(0)[cy, cx].tolist()
    results.panel._on_ignore_toggled(target.id)
    after = results.viewport._stacks["gray0"]._array(0)[cy, cx].tolist()

    assert before[0] != before[1] or before[1] != before[2]  # coloured while active
    assert after[0] == after[1] == after[2]  # gray once ignored


def test_the_back_button_requests_a_new_comparison(results):
    requested = []
    results.backRequested.connect(lambda: requested.append(True))
    results.back_button.click()
    assert requested


def test_showing_a_second_result_replaces_the_first(results, outcome, review_pair):
    from app.config import AppConfig
    from app.engine.worker import run_comparison
    from app.models.job import ComparisonMode

    original, _ = review_pair
    identical = run_comparison(original, original.copy(), ComparisonMode.AUTOMATIC, AppConfig())

    results.show_outcome(identical, "second")

    assert results.panel.title_label.text() == "Findings (0)"
    assert results.viewport.original_pane.marker_count == 0
