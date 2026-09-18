from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest

from app.ui.comparison_viewport import (
    MAX_ZOOM,
    MIN_ZOOM,
    ZOOM_STEP,
    ComparisonViewport,
)
from app.ui.composition import ViewMode


@pytest.fixture
def viewport(qapp, outcome):
    widget = ComparisonViewport()
    widget.resize(1100, 700)
    widget.show()
    widget.set_images(outcome.aligned_original, outcome.aligned_suspect)
    widget.set_marker_clusters(outcome.clusters)
    qapp.processEvents()
    yield widget
    widget.close()


# ---------------------------------------------------------------------------
# Zoom (FR-8.6)
# ---------------------------------------------------------------------------


def test_zoom_in_and_out_change_the_scale_by_a_fixed_step(viewport):
    viewport.set_zoom(1.0)
    viewport.zoom_in()
    assert viewport.zoom == pytest.approx(ZOOM_STEP)
    viewport.zoom_out()
    assert viewport.zoom == pytest.approx(1.0)


def test_the_current_zoom_percentage_is_displayed(viewport):
    viewport.set_zoom(1.0)
    assert viewport.zoom_label.text() == "100%"
    viewport.zoom_in()
    assert viewport.zoom_label.text() == "125%"


def test_zoom_applies_to_both_panes(viewport):
    """FR-8.6: "applied to both viewers"."""
    viewport.set_zoom(1.6)
    assert viewport.original_pane.zoom == pytest.approx(1.6)
    assert viewport.suspect_pane.zoom == pytest.approx(1.6)


def test_zoom_is_clamped_and_the_buttons_reflect_the_limits(viewport):
    viewport.set_zoom(1000.0)
    assert viewport.zoom == MAX_ZOOM
    assert not viewport.zoom_in_button.isEnabled()

    viewport.set_zoom(0.0001)
    assert viewport.zoom == MIN_ZOOM
    assert not viewport.zoom_out_button.isEnabled()

    viewport.set_zoom(1.0)
    assert viewport.zoom_in_button.isEnabled() and viewport.zoom_out_button.isEnabled()


def test_reset_fits_the_whole_page_in_view(viewport):
    viewport.set_zoom(3.0)
    viewport.fit_to_view()

    width, height = viewport.original_pane.image_size
    pane = viewport.original_pane.viewport()
    assert viewport.zoom * width <= pane.width()
    assert viewport.zoom * height <= pane.height()
    # ...and it is a *fit*: one dimension fills the pane, not a tiny thumbnail.
    fill = max(viewport.zoom * width / pane.width(), viewport.zoom * height / pane.height())
    assert fill > 0.9


def test_a_fitted_view_refits_when_the_window_resizes(viewport, qapp):
    viewport.fit_to_view()
    before = viewport.zoom
    viewport.resize(1500, 950)
    qapp.processEvents()
    assert viewport.zoom > before


def test_a_manual_zoom_is_not_undone_by_a_resize(viewport, qapp):
    viewport.set_zoom(2.0)
    viewport.resize(1500, 950)
    qapp.processEvents()
    assert viewport.zoom == 2.0


def test_ctrl_wheel_zooms(viewport):
    viewport.set_zoom(1.0)
    viewport.original_pane.zoomStepRequested.emit(1)
    assert viewport.zoom > 1.0
    viewport.original_pane.zoomStepRequested.emit(-1)
    assert viewport.zoom == pytest.approx(1.0)


def test_a_pinch_zooms_by_the_amount_the_fingers_moved(viewport):
    """Reported: pinching only scrolled. A pinch is many small factors, not fixed steps."""
    viewport.set_zoom(1.0)
    for _ in range(4):
        viewport.original_pane.zoomRequested.emit(1.1, QPointF(200, 150))
    assert viewport.zoom == pytest.approx(1.1**4)

    viewport.original_pane.zoomRequested.emit(1 / 1.1**4, QPointF(200, 150))
    assert viewport.zoom == pytest.approx(1.0)


def test_a_pinch_zooms_about_the_fingers_not_the_page_centre(viewport):
    viewport.set_zoom(3.0)  # scrollable, so there is somewhere to keep the point
    pane = viewport.original_pane
    anchor = QPointF(220, 160)
    before = pane.mapToScene(anchor.toPoint())

    pane.zoomRequested.emit(1.5, anchor)

    after = pane.mapToScene(anchor.toPoint())
    assert (after - before).manhattanLength() < 2.0


def test_a_pinch_moves_both_panes_together(viewport):
    viewport.set_zoom(1.0)
    viewport.original_pane.zoomRequested.emit(2.0, QPointF(100, 100))
    assert viewport.original_pane.zoom == pytest.approx(2.0)
    assert viewport.suspect_pane.zoom == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Scroll lock (FR-8.5)
# ---------------------------------------------------------------------------


def _scroll_bars(viewport):
    viewport.set_zoom(3.0)  # large enough that both panes can scroll
    return viewport.original_pane.horizontalScrollBar(), viewport.suspect_pane.horizontalScrollBar()


def test_scroll_lock_keeps_both_panes_in_step(viewport):
    original_bar, suspect_bar = _scroll_bars(viewport)
    viewport.set_scroll_lock(True)

    original_bar.setValue(original_bar.maximum() // 2)

    assert suspect_bar.value() == original_bar.value() != 0


def test_scroll_lock_works_from_either_pane(viewport):
    original_bar, suspect_bar = _scroll_bars(viewport)
    viewport.set_scroll_lock(True)

    suspect_bar.setValue(suspect_bar.maximum() // 3)

    assert original_bar.value() == suspect_bar.value() != 0


def test_without_scroll_lock_each_pane_scrolls_independently(viewport):
    original_bar, suspect_bar = _scroll_bars(viewport)
    viewport.set_scroll_lock(False)
    suspect_bar.setValue(0)

    original_bar.setValue(original_bar.maximum() // 2)

    assert original_bar.value() > 0
    assert suspect_bar.value() == 0


def test_turning_scroll_lock_back_on_recouples_the_panes(viewport):
    original_bar, suspect_bar = _scroll_bars(viewport)
    viewport.set_scroll_lock(False)
    original_bar.setValue(original_bar.maximum() // 2)
    assert suspect_bar.value() != original_bar.value()

    viewport.set_scroll_lock(True)

    assert suspect_bar.value() == original_bar.value()


def test_the_checkbox_drives_the_scroll_lock(viewport):
    viewport.scroll_lock_check.setChecked(False)
    assert viewport.scroll_lock is False
    viewport.scroll_lock_check.setChecked(True)
    assert viewport.scroll_lock is True


# ---------------------------------------------------------------------------
# The four modes (FR-8.1 - FR-8.4)
# ---------------------------------------------------------------------------


def test_side_by_side_shows_both_files(viewport):
    assert viewport.view_mode is ViewMode.SIDE_BY_SIDE
    assert not viewport._suspect_holder.isHidden()
    assert viewport.original_pane.shown_key == "original"
    assert viewport.suspect_pane.shown_key == "suspect"
    assert not viewport.is_animating


@pytest.mark.parametrize("mode", [ViewMode.FLICKER, ViewMode.OVERLAY, ViewMode.GRAYSCALE])
def test_the_other_modes_use_a_single_pane(viewport, mode):
    viewport.set_view_mode(mode)
    assert viewport._suspect_holder.isHidden()
    assert not viewport.scroll_lock_check.isEnabled()  # nothing to lock with one pane


def test_flicker_alternates_the_original_and_the_suspect(viewport):
    """FR-8.2: alternating display at the same screen position."""
    viewport.set_view_mode(ViewMode.FLICKER)
    assert viewport.is_animating
    assert viewport.original_pane.shown_key == "original"

    viewport._on_tick()
    assert viewport.original_pane.shown_key == "suspect"
    viewport._on_tick()
    assert viewport.original_pane.shown_key == "original"


def test_flicker_holds_the_view_still_between_frames(viewport):
    """Same screen position: swapping the image must not move the view."""
    viewport.set_view_mode(ViewMode.FLICKER)
    viewport.set_zoom(2.0)
    bar = viewport.original_pane.horizontalScrollBar()
    bar.setValue(bar.maximum() // 2)
    before = (bar.value(), viewport.original_pane.verticalScrollBar().value())

    viewport._on_tick()

    after = (bar.value(), viewport.original_pane.verticalScrollBar().value())
    assert before == after


def test_overlay_is_static(viewport):
    viewport.set_view_mode(ViewMode.OVERLAY)
    assert not viewport.is_animating
    assert viewport.original_pane.shown_key == "overlay"


def test_grayscale_animates_its_differences(viewport):
    """FR-8.4: areas containing a difference keep colour and flicker."""
    viewport.set_view_mode(ViewMode.GRAYSCALE)
    assert viewport.is_animating
    assert viewport.original_pane.shown_key == "gray0"
    viewport._on_tick()
    assert viewport.original_pane.shown_key == "gray1"


def test_leaving_an_animated_mode_stops_the_timer(viewport):
    viewport.set_view_mode(ViewMode.FLICKER)
    assert viewport.is_animating
    viewport.set_view_mode(ViewMode.SIDE_BY_SIDE)
    assert not viewport.is_animating


def test_a_hidden_viewer_does_not_animate(viewport):
    viewport.set_view_mode(ViewMode.FLICKER)
    viewport.hide()
    assert not viewport.is_animating
    viewport.show()
    assert viewport.is_animating


def test_switching_modes_keeps_the_zoom(viewport):
    viewport.set_zoom(2.0)
    viewport.set_view_mode(ViewMode.OVERLAY)
    assert viewport.zoom == 2.0


def test_the_mode_buttons_track_the_mode(viewport):
    viewport.set_view_mode(ViewMode.OVERLAY)
    assert viewport._mode_buttons[ViewMode.OVERLAY].isChecked()
    assert not viewport._mode_buttons[ViewMode.SIDE_BY_SIDE].isChecked()


def test_clicking_a_mode_button_switches_mode(viewport):
    viewport._mode_buttons[ViewMode.GRAYSCALE].click()
    assert viewport.view_mode is ViewMode.GRAYSCALE


# ---------------------------------------------------------------------------
# Markers (FR-7.2, FR-7.4)
# ---------------------------------------------------------------------------


def test_every_finding_is_marked_in_both_panes(viewport, outcome):
    assert viewport.original_pane.marker_count == len(outcome.clusters)
    assert viewport.suspect_pane.marker_count == len(outcome.clusters)


def test_a_finding_is_marked_at_the_same_place_in_both_panes(viewport, outcome):
    """FR-7.4: never at mismatched positions between the two views."""
    for cluster in outcome.clusters:
        in_original = viewport.original_pane._marker_items[cluster.id].rect()
        in_suspect = viewport.suspect_pane._marker_items[cluster.id].rect()
        assert in_original == in_suspect


def test_markers_stay_where_they_are_at_any_zoom_and_mode(viewport, outcome):
    """Markers live in image coordinates, so changing the view cannot shift them."""
    cluster = outcome.clusters[0]
    box = cluster.bbox_in_original
    for zoom in (0.3, 1.0, 4.0):
        viewport.set_zoom(zoom)
        rect = viewport.original_pane._marker_items[cluster.id].rect()
        assert (rect.x(), rect.y(), rect.width(), rect.height()) == pytest.approx(
            (box.x0, box.y0, box.width, box.height)
        )
    viewport.set_view_mode(ViewMode.OVERLAY)
    rect = viewport.original_pane._marker_items[cluster.id].rect()
    assert rect.x() == pytest.approx(box.x0)


def test_only_the_findings_passed_in_are_marked(viewport, outcome):
    remaining = outcome.clusters[1:]
    viewport.set_marker_clusters(remaining)
    assert viewport.original_pane.marker_count == len(remaining)
    assert viewport.suspect_pane.marker_count == len(remaining)


def test_every_marker_sits_on_a_halo_so_it_stays_visible_over_its_own_colour(viewport, outcome):
    """A red marker over a red block vanished; the halo keeps it legible."""
    for pane in (viewport.original_pane, viewport.suspect_pane):
        assert len(pane._halo_items) == len(outcome.clusters)
        for cluster in outcome.clusters:
            marker = pane._marker_items[cluster.id]
            halo = next(h for h in pane._halo_items if h.rect() == marker.rect())
            assert halo.zValue() < marker.zValue()  # underneath it
            assert halo.pen().color().red() == 255  # white


def test_halos_are_removed_along_with_their_markers(viewport, outcome):
    viewport.set_marker_clusters(outcome.clusters[:1])
    assert len(viewport.original_pane._halo_items) == 1
    viewport.set_marker_clusters([])
    assert viewport.original_pane._halo_items == []
    assert viewport.original_pane.marker_count == 0


def test_a_halo_never_steals_a_click_from_its_marker(viewport, outcome, qapp):
    cluster = outcome.clusters[0]
    pane = viewport.original_pane
    pane.center_on_box(cluster.bbox_in_original)
    qapp.processEvents()
    halo = next(h for h in pane._halo_items if h.rect() == pane._marker_items[cluster.id].rect())
    assert halo.data(0) is None  # only markers carry a finding id


def test_the_selected_finding_is_highlighted(viewport, outcome):
    target = outcome.clusters[0]
    viewport.set_selected(target.id)

    selected = viewport.original_pane._marker_items[target.id]
    other = viewport.original_pane._marker_items[outcome.clusters[1].id]
    assert selected.brush().style() != Qt.BrushStyle.NoBrush
    assert other.brush().style() == Qt.BrushStyle.NoBrush


def test_clicking_a_marker_reports_its_finding(viewport, outcome, qapp):
    viewport.set_zoom(1.0)
    cluster = outcome.clusters[0]
    pane = viewport.original_pane
    cx, cy = cluster.bbox_in_original.center
    pane.center_on_box(cluster.bbox_in_original)
    qapp.processEvents()

    clicked = []
    viewport.markerClicked.connect(clicked.append)
    device_point = pane.mapFromScene(cx, cy)
    QTest.mouseClick(
        pane.viewport(),
        Qt.MouseButton.LeftButton,
        pos=QPoint(device_point.x(), device_point.y()),
    )

    assert clicked and clicked[0] in {c.id for c in outcome.clusters}


def test_a_drag_pans_rather_than_selecting(viewport, qapp):
    viewport.set_zoom(2.0)
    pane = viewport.original_pane
    clicked = []
    viewport.markerClicked.connect(clicked.append)

    QTest.mousePress(pane.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(100, 100))
    QTest.mouseMove(pane.viewport(), QPoint(160, 140))
    QTest.mouseRelease(pane.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(160, 140))

    assert clicked == []


def test_focusing_a_finding_selects_it_and_brings_it_into_view(viewport, outcome, qapp):
    cluster = outcome.clusters[-1]
    viewport.fit_to_view()

    viewport.focus_cluster(cluster)
    qapp.processEvents()

    assert viewport._selected_id == cluster.id
    # The whole finding must be on screen. It can't always be *centred*: a
    # finding near the page edge is limited by how far the view can scroll.
    pane = viewport.original_pane
    visible = pane.mapToScene(pane.viewport().rect()).boundingRect()
    box = cluster.bbox_in_original
    assert visible.left() <= box.x0 and visible.right() >= box.x1
    assert visible.top() <= box.y0 and visible.bottom() >= box.y1


def test_focusing_never_zooms_out(viewport, outcome):
    viewport.set_zoom(4.0)
    viewport.focus_cluster(outcome.clusters[0])
    assert viewport.zoom >= 4.0 - 1e-9


def test_loading_new_images_resets_to_a_fitted_view(viewport, outcome):
    viewport.set_zoom(4.0)
    viewport.set_images(outcome.aligned_original, outcome.aligned_suspect)
    assert viewport._fit_mode
    assert viewport.zoom < 4.0
