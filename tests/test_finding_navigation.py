"""Stepping through the findings with the arrow keys, in every mode: with one selected (in the viewer or the list), Down
takes the view to the one below it in the list and Up to the one above (`ResultsView.step_finding`)."""

from __future__ import annotations

from dataclasses import replace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from app.models.diff import BoundingBox, DiffCandidate, DiffCluster, DiffSubtype, DiffType
from app.models.job import ComparisonMode
from test_leaflet_options import _loaded_window, _outcome


def _three(mode: ComparisonMode):
    found = []
    for number, y in enumerate((20, 120, 220), start=1):
        member = DiffCandidate(BoundingBox(30, y, 90, y + 30), DiffType.GRAPHICS, DiffSubtype.CHANGED, 1.0, f"finding {number}")
        found.append(DiffCluster(number, member.bbox, member.bbox, DiffType.GRAPHICS, [member], detail=member.detail))
    return replace(_outcome(mode), clusters=found, all_clusters=list(found), text_compared=True, graphics_compared=True)


def _results(qapp, tmp_path, mode):
    window, _jobs = _loaded_window(qapp, tmp_path, mode)
    window.resize(1400, 900)
    window.show()
    window._on_compare()
    window._on_completed(_three(mode))
    qapp.processEvents()
    return window, window.results_view


@pytest.mark.parametrize("mode", [ComparisonMode.AUTOMATIC, ComparisonMode.LEAFLET, ComparisonMode.SCAN_VS_DIGITAL])
def test_down_and_up_step_to_the_next_and_previous_finding(qapp, tmp_path, mode):
    window, results = _results(qapp, tmp_path, mode)
    results._on_marker_clicked(1)  # a finding clicked in the artwork
    pane = results.viewport.original_pane
    pane.setFocus()
    QTest.keyClick(pane, Qt.Key.Key_Down)
    assert results.panel._selected_id == 2
    assert results.viewport._selected_id == 2  # the viewer is on it too
    QTest.keyClick(pane, Qt.Key.Key_Down)
    assert results.panel._selected_id == 3
    QTest.keyClick(pane, Qt.Key.Key_Down)  # the last one: nothing further
    assert results.panel._selected_id == 3
    QTest.keyClick(results.panel, Qt.Key.Key_Up)  # from the list as well
    assert results.panel._selected_id == 2
    window.close()


def test_the_arrows_do_nothing_with_no_finding_selected(qapp, tmp_path):
    window, results = _results(qapp, tmp_path, ComparisonMode.AUTOMATIC)
    assert not results.step_finding(1)
    QTest.keyClick(results.viewport.original_pane, Qt.Key.Key_Down)
    assert results.panel._selected_id is None
    window.close()


def test_a_drop_down_keeps_its_own_arrows(qapp, tmp_path):
    window, results = _results(qapp, tmp_path, ComparisonMode.AUTOMATIC)
    results._on_marker_clicked(1)
    combo = results.panel.category_combo
    combo.setFocus()
    qapp.processEvents()
    QTest.keyClick(combo, Qt.Key.Key_Down)
    assert results.panel._selected_id in (1, None)  # not stepped (the kind picked may have changed what is listed)
    window.close()


# ---- Barcode and Braille modes: the list beside the viewer is stepped the same way --------------------------------------


def _side_results(qapp, mode, **changes):
    from app.ui.results_view import ResultsView

    view = ResultsView()
    view.resize(1400, 900)
    view.show()
    view.show_outcome(replace(_outcome(ComparisonMode.AUTOMATIC), mode=mode, **changes), "a.pdf  vs  b.pdf")
    qapp.processEvents()
    return view


def test_down_and_up_step_through_the_braille_lines_and_differences(qapp):
    from app.config import DetectionConfig
    from app.engine.detectors import braille_diff
    from test_braille_cells import _page

    candidates, comparison = braille_diff.compare(_page(["metformin", "500 mg"]), _page(["metformin", "800 mg"]), DetectionConfig())
    member = candidates[0]
    clusters = [DiffCluster(1, member.bbox, member.bbox, DiffType.BRAILLE, [member], detail=member.detail)]
    view = _side_results(qapp, ComparisonMode.BRAILLE, braille=comparison, clusters=clusters, all_clusters=list(clusters))
    panel = view.braille_panel
    shown = []
    panel.regionSelected.connect(shown.append)
    first, second = panel.line_cards()
    (difference,) = panel.difference_cards()

    first.clicked.emit(first.box)  # the first line chosen in the list
    pane = view.viewport.original_pane
    pane.setFocus()
    QTest.keyClick(pane, Qt.Key.Key_Down)
    QTest.keyClick(pane, Qt.Key.Key_Down)
    QTest.keyClick(pane, Qt.Key.Key_Down)  # the last one: nothing further
    QTest.keyClick(pane, Qt.Key.Key_Up)

    assert shown == [first.box, second.box, difference.box, second.box]
    assert second.property("selected") == "true" and first.property("selected") == "false"
    view.close()


def test_down_and_up_step_through_the_codes(qapp):
    from app.models.result import SymbolComparison

    boxes = [BoundingBox(30, y, 130, y + 60) for y in (20, 120, 220)]
    codes = [
        SymbolComparison("EAN13", "8901117282322", "8901117282322", boxes[0]),
        SymbolComparison("Pharmacode", "161", "417", boxes[1]),
        SymbolComparison("QRCode", "https://a.example", "https://a.example", boxes[2]),
    ]
    view = _side_results(qapp, ComparisonMode.BARCODE, symbol_comparisons=codes)
    panel = view.symbol_panel
    shown = []
    panel.symbolSelected.connect(shown.append)
    cards = panel.cards()

    cards[0].selectRequested.emit(cards[0].comparison.bbox)
    pane = view.viewport.original_pane
    pane.setFocus()
    QTest.keyClick(pane, Qt.Key.Key_Down)
    QTest.keyClick(pane, Qt.Key.Key_Down)
    QTest.keyClick(pane, Qt.Key.Key_Up)

    assert shown == [cards[0].comparison.bbox, cards[1].comparison.bbox, cards[2].comparison.bbox, cards[1].comparison.bbox]
    view.close()


def test_the_braille_lines_and_cells_are_put_on_the_page_with_everything_else():
    """Detection works in the artwork's own box; what it found is moved back onto the page -- each Braille line and cell
    as well as the region (a line chosen in the list was shown where it was in the box, not on the page)."""
    from app.engine.worker import _on_page
    from app.models.result import BrailleCellComparison, BrailleComparison, BrailleLineComparison

    cell = BrailleCellComparison(1, 1, "a", "a", BoundingBox(10, 10, 20, 30), "match")
    braille = BrailleComparison(1, 1, 1, 0, 0, 0, BoundingBox(10, 10, 20, 30), [BrailleLineComparison("a", "a", [cell], BoundingBox(10, 10, 20, 30))])

    *_rest, moved, _o, _s = _on_page((100, 200, 400, 600), (800, 800), [], [], [], [], braille, None, None)

    assert moved.region == BoundingBox(110, 210, 120, 230)
    assert moved.lines[0].bbox == BoundingBox(110, 210, 120, 230)
    assert moved.lines[0].cells[0].bbox == BoundingBox(110, 210, 120, 230)
