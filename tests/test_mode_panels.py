"""Barcode / Pharmacode / QR results (FR-10.2) and Braille results (FR-11.1)."""

from __future__ import annotations

import dataclasses
import re

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from app.models.diff import BoundingBox
from app.models.job import ComparisonMode
from app.models.result import BrailleComparison, SymbolComparison
from app.ui.braille_panel import BrailleResultsPanel
from app.ui.results_view import ResultsView
from app.ui.symbol_panel import CROSS, GROUP_ORDER, TICK, SymbolResultsPanel

BOX = BoundingBox(200, 300, 400, 380)


def _symbols():
    return [
        SymbolComparison("QRCode", "https://a.example/1", "https://a.example/1", BOX),
        SymbolComparison("EAN13", "8901117282322", "8901117282322", BOX),
        SymbolComparison("Pharmacode", "161", "417", BOX),
        SymbolComparison("CODE128", "LOT12345", None, BOX),
        SymbolComparison("Pharmacode", "324", "324", BOX),
    ]


@pytest.fixture
def symbols(qapp):
    panel = SymbolResultsPanel()
    panel.resize(400, 700)
    panel.show()
    panel.set_comparisons(_symbols())
    qapp.processEvents()
    yield panel
    panel.close()


def _click(widget) -> None:
    pos = QPointF(5, 5)
    widget.mousePressEvent(QMouseEvent(QEvent.Type.MouseButtonPress, pos, pos, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))


# ---- FR-10.2 ------------------------------------------------------------------------------


def test_results_are_grouped_by_symbol_type_in_a_fixed_order(symbols):
    groups = symbols.groups()
    assert list(groups) == ["Barcode", "Pharmacode", "QR code"]
    assert list(groups) == [g for g in GROUP_ORDER if g in groups]
    assert [c.symbol_type for c in groups["Barcode"]] == ["EAN13", "CODE128"]
    assert [c.symbol_type for c in groups["Pharmacode"]] == ["Pharmacode", "Pharmacode"]
    assert [c.symbol_type for c in groups["QR code"]] == ["QRCode"]


def test_each_symbol_has_a_green_tick_or_a_red_cross(symbols):
    marks = {card.comparison.original_value: card.badge for card in symbols.cards()}
    assert marks["8901117282322"].text() == TICK and marks["8901117282322"].accessibleName() == "pass"
    assert marks["161"].text() == CROSS and marks["161"].accessibleName() == "fail"
    assert "#15803d" in marks["324"].styleSheet()  # green
    assert "#b91c1c" in marks["161"].styleSheet()  # red


def test_a_symbol_missing_from_the_suspect_fails_and_says_not_found(symbols):
    (card,) = [c for c in symbols.cards() if c.comparison.symbol_type == "CODE128"]
    assert card.badge.accessibleName() == "fail"
    assert "not found" in card.suspect_label.text()
    assert "Missing from suspect" in card.status_label.text()


def test_both_values_are_shown_so_the_difference_can_be_read(symbols):
    (card,) = [c for c in symbols.cards() if c.comparison.original_value == "161"]
    shown = lambda label: re.sub("<[^>]+>", "", label.text())  # (the characters that differ are marked up)
    assert "161" in shown(card.original_label) and "417" in shown(card.suspect_label)
    assert "<b" in card.suspect_label.text()  # ...marked


def test_the_summary_counts_failures(symbols):
    assert symbols.title_label.text() == "Symbols (5)"
    assert symbols.summary_pill.text() == f"{CROSS} 2 of 5 differ"
    assert not symbols.empty_label.isVisible()


def test_all_matching_is_stated_as_such(qapp):
    panel = SymbolResultsPanel()
    panel.set_comparisons([SymbolComparison("EAN13", "1", "1", BOX)])
    assert panel.summary_pill.text() == f"{TICK} all match"


def test_nothing_read_is_a_warning_not_a_clean_bill_of_health(qapp):
    panel = SymbolResultsPanel()
    panel.show()
    panel.set_comparisons([])
    assert panel.summary_pill.text() == "nothing read"
    assert panel.empty_label.isVisible() and "could be read" in panel.empty_label.text()
    panel.close()


def test_choosing_a_symbol_reports_where_it_is(symbols):
    seen = []
    symbols.symbolSelected.connect(seen.append)

    _click(symbols.cards()[0])

    assert seen == [BOX]
    assert symbols.cards()[0].property("selected") == "true"
    assert all(card.property("selected") == "false" for card in symbols.cards()[1:])


def test_replacing_the_results_leaves_no_stale_cards(symbols):
    symbols.set_comparisons([SymbolComparison("EAN13", "1", "1", BOX)])
    assert len(symbols.cards()) == 1


# ---- FR-11.1 ------------------------------------------------------------------------------


def test_matching_dot_patterns_pass(qapp):
    panel = BrailleResultsPanel()
    panel.set_comparison(BrailleComparison(original_dot_count=24, suspect_dot_count=24, matched=24, missing=0, added=0, moved=0))
    assert panel.verdict_badge.accessibleName() == "pass"
    assert "match" in panel.verdict_label.text()
    assert panel.summary_pill.text() == f"{TICK} match"


def test_differing_dots_fail_with_the_counts_behind_it(qapp):
    panel = BrailleResultsPanel()
    panel.set_comparison(BrailleComparison(original_dot_count=24, suspect_dot_count=23, matched=20, missing=2, added=1, moved=1))
    assert panel.verdict_badge.accessibleName() == "fail"
    assert "4 dot differences" in panel.verdict_label.text()
    assert panel._count_labels["missing"].text() == "2"
    assert panel._count_labels["added"].text() == "1"
    assert panel._count_labels["moved"].text() == "1"
    assert panel._count_labels["matched"].text() == "20"


def test_no_dots_at_all_is_flagged_as_a_likely_colour_mismatch(qapp):
    panel = BrailleResultsPanel()
    panel.show()
    panel.set_comparison(BrailleComparison(0, 0, 0, 0, 0, 0))
    assert panel.verdict_badge is None  # neither a pass nor a fail
    assert "No Braille dots" in panel.verdict_label.text()
    assert panel.note_label.isVisible() and "colour" in panel.note_label.text()
    panel.close()


def test_the_colour_assumption_is_disclosed(qapp):
    panel = BrailleResultsPanel()
    texts = [label.text() for label in panel.findChildren(type(panel.title_label))]
    assert any("open item" in text for text in texts)


# ---- in the results page ----------------------------------------------------------------------


def _shown(qapp, outcome, **changes):
    view = ResultsView()
    view.resize(1400, 800)
    view.show()
    view.show_outcome(dataclasses.replace(outcome, **changes), "a.pdf  vs  b.pdf")
    qapp.processEvents()
    return view


def test_a_normal_comparison_shows_only_the_findings(qapp, outcome):
    view = _shown(qapp, outcome, mode=ComparisonMode.AUTOMATIC, symbol_comparisons=[], braille=None)
    assert view.side.count() == 1 and not view.side.tabBar().isVisible()
    view.close()


def test_the_unused_panels_are_not_left_drawn_over_the_page(qapp, outcome):
    """Regression: created as children of the page and never added to a tab, they showed up at its corner."""
    view = _shown(qapp, outcome, mode=ComparisonMode.AUTOMATIC, symbol_comparisons=[], braille=None)
    assert not view.symbol_panel.isVisible() and not view.braille_panel.isVisible()

    view.show_outcome(dataclasses.replace(outcome, mode=ComparisonMode.BARCODE, symbol_comparisons=_symbols()), "x")
    qapp.processEvents()
    assert view.symbol_panel.isVisible() and not view.braille_panel.isVisible()

    view.show_outcome(dataclasses.replace(outcome, mode=ComparisonMode.AUTOMATIC, symbol_comparisons=[]), "x")
    qapp.processEvents()
    assert not view.symbol_panel.isVisible()  # and gone again once the mode no longer calls for it
    view.close()


def test_barcode_mode_shows_the_symbols_tab_first(qapp, outcome):
    view = _shown(qapp, outcome, mode=ComparisonMode.BARCODE, symbol_comparisons=_symbols())
    assert view.side.currentWidget() is view.symbol_panel
    assert view.side.tabBar().isVisible()
    assert [view.side.tabText(i) for i in range(view.side.count())] == ["Findings", "Symbols"]
    assert len(view.symbol_panel.cards()) == 5
    view.close()


def test_barcode_mode_with_nothing_read_still_shows_the_tab_and_says_so(qapp, outcome):
    view = _shown(qapp, outcome, mode=ComparisonMode.BARCODE, symbol_comparisons=[])
    assert view.side.currentWidget() is view.symbol_panel
    assert view.symbol_panel.summary_pill.text() == "nothing read"
    view.close()


def test_braille_mode_shows_the_braille_tab(qapp, outcome):
    braille = BrailleComparison(10, 10, 10, 0, 0, 0)
    view = _shown(qapp, outcome, mode=ComparisonMode.BRAILLE, braille=braille)
    assert view.side.currentWidget() is view.braille_panel
    assert view.braille_panel.summary_pill.text() == f"{TICK} match"
    view.close()


def test_tabs_do_not_pile_up_across_comparisons(qapp, outcome):
    view = _shown(qapp, outcome, mode=ComparisonMode.BARCODE, symbol_comparisons=_symbols())
    view.show_outcome(dataclasses.replace(outcome, mode=ComparisonMode.AUTOMATIC, symbol_comparisons=[], braille=None), "x")
    assert view.side.count() == 1 and view.side.currentWidget() is view.panel
    view.close()


def test_choosing_a_symbol_brings_it_into_view_in_the_viewer(qapp, outcome):
    view = _shown(qapp, outcome, mode=ComparisonMode.BARCODE, symbol_comparisons=_symbols())
    view.viewport.set_zoom(3.0)
    horizontal = view.viewport.original_pane.horizontalScrollBar()
    before = horizontal.value()

    _click(view.symbol_panel.cards()[0])
    qapp.processEvents()

    pane = view.viewport.original_pane
    centre = pane.mapToScene(pane.viewport().rect().center())
    assert abs(centre.x() - BOX.center[0]) < 20 and abs(centre.y() - BOX.center[1]) < 20
    assert horizontal.value() != before or before == 0
    view.close()
