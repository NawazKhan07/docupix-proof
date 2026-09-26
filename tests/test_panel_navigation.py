"""Choosing something in the Barcode or Braille panel takes the viewer to it and marks it."""

from __future__ import annotations

import numpy as np

from app.config import DetectionConfig
from app.engine.detectors import braille_diff
from app.models.diff import BoundingBox, DiffCandidate, DiffCluster, DiffSubtype, DiffType
from app.models.result import BrailleComparison
from app.ui.braille_panel import BrailleResultsPanel
from app.ui.comparison_viewport import ComparisonViewport


def _viewport(qapp) -> ComparisonViewport:
    viewport = ComparisonViewport()
    viewport.resize(900, 600)
    viewport.show()
    page = np.full((3000, 2000, 3), 255, np.uint8)
    viewport.set_images(page, page.copy())
    viewport.fit_to_view()
    qapp.processEvents()
    return viewport


def _centre(pane) -> tuple[float, float]:
    point = pane.mapToScene(pane.viewport().rect().center())
    return point.x(), point.y()


def test_a_chosen_symbol_is_brought_into_view_and_marked_in_both_panes(qapp):
    viewport = _viewport(qapp)
    box = BoundingBox(1500, 2500, 1700, 2600)
    viewport.highlight_box(box)
    qapp.processEvents()
    assert viewport.original_pane.highlighted and viewport.suspect_pane.highlighted
    x, y = _centre(viewport.original_pane)
    assert abs(x - 1600) < 60 and abs(y - 2550) < 60

    member = DiffCandidate(BoundingBox(100, 100, 150, 150), DiffType.TEXT, DiffSubtype.CHANGED, 1.0, "x")
    viewport.focus_cluster(DiffCluster(1, member.bbox, member.bbox, DiffType.TEXT, [member], detail="x"))
    assert not viewport.original_pane.highlighted  # choosing a finding clears the panel's mark
    viewport.close()


def test_braille_knows_where_its_dots_are():
    dots = np.array([[100.0, 200.0], [130.0, 200.0], [100.0, 260.0]])
    region = braille_diff._region(dots, dots + 5)
    assert region.x0 < 100 and region.y0 < 200 and region.x1 > 135 and region.y1 > 265


def test_each_braille_difference_takes_the_viewer_to_it(qapp):
    panel = BrailleResultsPanel()
    chosen: list[BoundingBox] = []
    panel.regionSelected.connect(chosen.append)
    panel.set_comparison(
        BrailleComparison(original_dot_count=12, suspect_dot_count=11, matched=11, missing=1, added=0, moved=0,
                          region=BoundingBox(50, 50, 400, 200))
    )
    dot = BoundingBox(120, 80, 132, 92)
    member = DiffCandidate(dot, DiffType.BRAILLE, DiffSubtype.REMOVED, 1.0, "Braille dot missing from suspect")
    panel.set_differences([DiffCluster(3, dot, dot, DiffType.BRAILLE, [member], detail=member.detail)])

    (card,) = panel.difference_cards()
    card.clicked.emit(card.box)
    panel.verdict_card.clicked.emit(panel.verdict_card.box)
    assert chosen == [dot, BoundingBox(50, 50, 400, 200)]


def test_braille_comparison_reports_its_region_from_the_detector():
    config = DetectionConfig()
    image = np.full((400, 400, 3), 255, np.uint8)
    _candidates, comparison = braille_diff.compare(image, image.copy(), config)
    assert comparison.region is None  # no dots, nowhere to point
