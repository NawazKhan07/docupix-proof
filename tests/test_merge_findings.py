"""Merging findings: where several small differences are really one (the bars of a barcode), the reviewer draws a
rectangle round them and they become one finding -- in the list and on the page -- and can be split again."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt
from PySide6.QtGui import QMouseEvent

from app.engine.aggregation.aggregator import merge_clusters
from app.models.diff import BoundingBox, DiffCandidate, DiffCluster, DiffSubtype, DiffType
from app.ui.results_view import ResultsView, _mostly_inside


def _cluster(cluster_id, box, diff_type=DiffType.GRAPHICS, detail="Graphic changed", subtype=DiffSubtype.CHANGED):
    x0, y0, x1, y1 = box
    member = DiffCandidate(BoundingBox(x0, y0, x1, y1), diff_type, subtype, 0.9, detail, 1.0)
    return DiffCluster(
        id=cluster_id,
        bbox_in_original=BoundingBox(x0, y0, x1, y1),
        bbox_in_suspect=BoundingBox(x0 + 5, y0 + 5, x1 + 5, y1 + 5),
        type=diff_type,
        members=[member],
        detail=detail,
    )


# ---- the merged finding ---------------------------------------------------------------------------------------------------


def test_a_merged_finding_covers_all_of_them_in_both_files():
    a, b = _cluster(3, (100, 100, 120, 200)), _cluster(5, (140, 100, 160, 200))

    merged = merge_clusters([b, a])  # (in any order)

    assert (merged.bbox_in_original.x0, merged.bbox_in_original.x1, merged.bbox_in_original.y1) == (100, 160, 200)
    assert (merged.bbox_in_suspect.x0, merged.bbox_in_suspect.x1) == (105, 165)
    assert len(merged.members) == 2


def test_it_keeps_the_lowest_number_so_the_numbering_does_not_shift():
    assert merge_clusters([_cluster(7, (0, 0, 10, 10)), _cluster(4, (20, 0, 30, 10))]).id == 4


def test_it_is_as_specific_as_the_most_specific():
    graphics = _cluster(1, (0, 0, 10, 10), DiffType.GRAPHICS)
    text = _cluster(2, (20, 0, 30, 10), DiffType.TEXT, "'500' -> '250'")

    merged = merge_clusters([graphics, text])

    assert merged.type is DiffType.TEXT
    assert merged.types == {DiffType.GRAPHICS, DiffType.TEXT}
    assert "2 findings merged" in merged.detail and "'500' -> '250'" in merged.detail


def test_a_finding_cannot_be_merged_with_nothing():
    with pytest.raises(ValueError):
        merge_clusters([_cluster(1, (0, 0, 10, 10))])


def test_a_finding_is_in_the_rectangle_if_mostly_inside_it():
    region = BoundingBox(0, 0, 100, 100)
    assert _mostly_inside(BoundingBox(10, 10, 50, 50), region)
    assert _mostly_inside(BoundingBox(80, 10, 140, 50), region) is False  # a third inside
    assert _mostly_inside(BoundingBox(60, 10, 140, 50), region)  # half inside
    assert _mostly_inside(BoundingBox(20, 20, 110, 60), region)  # ~89% inside
    assert _mostly_inside(BoundingBox(200, 200, 260, 260), region) is False


# ---- on the results page ---------------------------------------------------------------------------------------------------------


@pytest.fixture
def results(qapp, outcome):
    view = ResultsView()
    view.resize(1400, 800)
    view.show()
    view.show_outcome(outcome, "original.pdf  vs  suspect.pdf  ·  Automatic")
    qapp.processEvents()
    yield view
    view.close()


def _everything(outcome) -> BoundingBox:
    box = outcome.clusters[0].bbox_in_original
    for cluster in outcome.clusters[1:]:
        box = box.union(cluster.bbox_in_original)
    return BoundingBox(box.x0 - 5, box.y0 - 5, box.x1 + 5, box.y1 + 5)


def test_a_rectangle_round_findings_merges_them_in_the_list_and_on_every_pane(results, outcome):
    total = len(outcome.clusters)
    assert total >= 2

    results._on_merge_region(_everything(outcome))

    assert results.panel.title_label.text() == "Findings (1)"
    assert results.viewport.original_pane.marker_count == 1 and results.viewport.suspect_pane.marker_count == 1
    assert "Merged" in results.viewport.merge_notice.text()
    assert results.viewport.undo_merge_button.isVisibleTo(results.viewport)


def test_only_the_findings_in_the_rectangle_are_merged(results, outcome):
    first, second = outcome.clusters[0], outcome.clusters[1]
    total = len(outcome.clusters)
    region = first.bbox_in_original.union(second.bbox_in_original)
    others_outside = [c for c in outcome.clusters[2:] if not _mostly_inside(c.bbox_in_original, region)]

    results._on_merge_region(region)

    assert results.panel.title_label.text() == f"Findings ({total - 1 - (total - 2 - len(others_outside))})"
    for cluster in others_outside:
        assert cluster.id in results.viewport.original_pane._marker_items  # untouched


def test_a_rectangle_round_one_finding_or_none_merges_nothing_and_says_so(results, outcome):
    total = len(outcome.clusters)
    results._on_merge_region(outcome.clusters[0].bbox_in_original)
    assert results.panel.title_label.text() == f"Findings ({total})" and "one finding" in results.viewport.merge_notice.text()

    results._on_merge_region(BoundingBox(-500, -500, -400, -400))
    assert "no findings" in results.viewport.merge_notice.text()


def test_ignored_findings_are_not_swept_into_a_merge(qapp, outcome):
    from app.ui.diff_list_panel import DiffListPanel

    panel = DiffListPanel()
    clusters = [_cluster(1, (0, 0, 10, 10)), _cluster(2, (20, 0, 30, 10)), _cluster(3, (40, 0, 50, 10))]
    panel.set_result(clusters, outcome.alignment, [])
    panel._on_ignore_toggled(2)

    view = ResultsView()
    view.panel = panel  # the page's own logic, over a list where the middle finding was dismissed
    view.viewport.set_marker_clusters(panel.marker_clusters())
    view._on_merge_region(BoundingBox(-5, -5, 60, 20))

    merged = panel._clusters[0]
    assert merged.id == 1 and {m.bbox.x0 for m in merged.members} == {0, 40}
    assert panel.cluster(2).is_ignored  # still there, still dismissed


def test_a_merge_can_be_undone_and_the_findings_come_back_in_their_places(results, outcome):
    ids = [c.id for c in outcome.clusters]
    results._on_merge_region(_everything(outcome))

    results.viewport.undo_merge_button.click()

    assert [c.id for c in results.panel._clusters] == ids
    assert results.panel.title_label.text() == f"Findings ({len(ids)})"
    assert results.viewport.original_pane.marker_count == len(ids)
    assert not results.viewport.undo_merge_button.isVisibleTo(results.viewport)


def test_a_merged_finding_can_be_ignored_like_any_other(results, outcome):
    results._on_merge_region(_everything(outcome))
    merged = results.panel._clusters[0]

    results.panel._on_ignore_toggled(merged.id)

    assert merged.is_ignored and results.viewport.original_pane.marker_count == 0


def test_merging_leaves_a_trace_in_the_log(results, outcome, caplog):
    with caplog.at_level("INFO", logger="app.ui.diff_list_panel"):
        results._on_merge_region(_everything(outcome))
    assert any("findings merged" in record.getMessage() for record in caplog.records)


def test_a_new_result_forgets_the_last_merge(results, outcome):
    results._on_merge_region(_everything(outcome))
    results.show_outcome(outcome, "again")
    assert not results.viewport.undo_merge_button.isVisibleTo(results.viewport) and not results.panel.can_undo_merge


# ---- drawing the rectangle ----------------------------------------------------------------------------------------------------------


def _mouse(kind, pane, scene_point, button=Qt.MouseButton.LeftButton, buttons=None):
    view = QPointF(pane.mapFromScene(scene_point))
    return QMouseEvent(kind, view, view, button, buttons if buttons is not None else button, Qt.KeyboardModifier.NoModifier)


def _drag(pane, start, end):
    pane.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, pane, start))
    pane.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, pane, end, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton))
    pane.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, pane, end, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton))


def test_dragging_on_a_pane_draws_a_rectangle_and_reports_it_only_while_merging(results, qapp):
    pane = results.viewport.original_pane
    got = []
    pane.regionSelected.connect(got.append)

    pane.set_region_selection(True)
    pane.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, pane, QPointF(60, 60)))
    pane.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, pane, QPointF(160, 140), Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton))
    assert pane._region_item is not None  # the rectangle is drawn as it is dragged
    pane.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, pane, QPointF(160, 140), Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton))

    assert pane._region_item is None
    (rect,) = got
    assert rect.left() == pytest.approx(60, abs=1.5) and rect.top() == pytest.approx(60, abs=1.5)
    assert rect.right() == pytest.approx(160, abs=1.5) and rect.bottom() == pytest.approx(140, abs=1.5)

    pane.set_region_selection(False)
    _drag(pane, QPointF(60, 60), QPointF(160, 140))
    assert len(got) == 1  # merging is off: a drag pans


def test_a_tiny_drag_is_a_click_and_selects_nothing(results):
    pane = results.viewport.original_pane
    got = []
    pane.regionSelected.connect(got.append)
    pane.set_region_selection(True)
    _drag(pane, QPointF(60, 60), QPointF(60.5, 60.5))
    assert got == []


def test_the_merge_button_turns_on_the_rectangle_and_one_rectangle_turns_it_off(results, outcome):
    viewport = results.viewport
    viewport.merge_button.click()
    assert viewport.merge_mode and results.viewport.original_pane.selecting_region and results.viewport.suspect_pane.selecting_region
    assert "Drag a rectangle" in viewport.merge_notice.text()

    asked = []
    viewport.mergeRegionSelected.connect(asked.append)
    viewport.original_pane.regionSelected.emit(QRectF(10, 20, 300, 400))

    assert not viewport.merge_mode and not viewport.original_pane.selecting_region
    (region,) = asked
    assert (region.x0, region.y0, region.x1, region.y1) == (10, 20, 310, 420)


def test_a_new_result_leaves_merge_mode(results, outcome):
    results.viewport.merge_button.click()
    results.show_outcome(outcome, "again")
    assert not results.viewport.merge_mode and not results.viewport.original_pane.selecting_region
