from __future__ import annotations

import numpy as np
import pytest

from app.models.categories import CATEGORY_LABELS, SOURCE_COLOURS, SUBTYPE_LABELS, category_colour
from app.models.diff import DiffCandidate, DiffCluster, DiffStatus, DiffSubtype, DiffType, BoundingBox
from app.models.result import AlignmentResult
from app.ui.diff_list_panel import DiffListPanel, FindingCard


def _alignment(residual: float = 0.74, low_confidence: bool = False) -> AlignmentResult:
    return AlignmentResult(
        transform=np.eye(3),
        residual_error_px=residual,
        residual_basis="keypoints",
        method="features",
        match_count=100,
        inlier_count=90,
        inlier_ratio=0.9,
        low_confidence=low_confidence,
        threshold_px=2.0,
    )


@pytest.fixture
def panel(qapp, outcome):
    widget = DiffListPanel()
    widget.resize(380, 700)
    widget.show()
    widget.set_result(outcome.clusters, outcome.alignment, outcome.warnings)
    qapp.processEvents()
    yield widget
    widget.close()


def _cards(panel: DiffListPanel) -> list[FindingCard]:
    return panel.findChildren(FindingCard)


def test_every_finding_is_listed_with_a_count(panel, outcome):
    assert panel.title_label.text() == f"Findings ({len(outcome.clusters)})"
    assert len(_cards(panel)) == len(outcome.clusters)


def test_each_finding_shows_what_it_is_and_what_happened_to_it(panel, outcome):
    """FR-7.1: its category and what happened (changed, moved, added, removed) -- there is no severity."""
    by_id = {c.id: c for c in outcome.clusters}
    for card in _cards(panel):
        cluster = by_id[card.cluster_id]
        assert CATEGORY_LABELS[cluster.category] in card.title_label.text()
        assert SUBTYPE_LABELS[cluster.subtype].capitalize() in card.title_label.text()
        assert "Critical" not in card.title_label.text() and "Major" not in card.title_label.text()
        assert card.detail_label.text() == cluster.detail


def test_each_finding_is_coloured_by_its_kind(panel, outcome):
    by_id = {c.id: c for c in outcome.clusters}
    for card in _cards(panel):
        assert category_colour(by_id[card.cluster_id].category) in card.colour_bar.styleSheet()


def test_the_summary_counts_findings_by_kind_in_their_colours(panel, outcome):
    summary = panel.summary_label.text()
    for category in {c.category for c in outcome.clusters}:
        count = sum(1 for c in outcome.clusters if c.category == category)
        assert f"{count} {CATEGORY_LABELS[category]}" in summary
        assert category_colour(category) in summary


def _found(cluster_id: int, source: str) -> DiffCluster:
    member = DiffCandidate(BoundingBox(0, 20 * cluster_id, 10, 20 * cluster_id + 10), DiffType.TEXT, DiffSubtype.CHANGED, 1.0, "x", source=source)
    return DiffCluster(
        id=cluster_id, bbox_in_original=member.bbox, bbox_in_suspect=member.bbox, type=DiffType.TEXT, members=[member], detail="x"
    )


def test_ocr_and_pixel_findings_take_their_own_colours_only_while_picked_as_such(qapp):
    from PySide6.QtGui import QColor

    from app.models.diff import SOURCE_OCR, SOURCE_PIXELS
    from app.ui.pane_view import PaneView

    ocr, pixels = _found(1, SOURCE_OCR), _found(2, SOURCE_PIXELS)  # both text
    assert len({SOURCE_COLOURS[SOURCE_OCR], SOURCE_COLOURS[SOURCE_PIXELS], category_colour("text")}) == 3
    widget = DiffListPanel()
    widget.set_source_kinds(True)
    widget.set_result([ocr, pixels], _alignment(), [])

    def bars():
        return {card.cluster_id: card.colour_bar.styleSheet() for card in _cards(widget)}

    # all kinds: coloured by kind -- both are text
    assert not widget.colour_by_source
    assert all(category_colour("text") in bar for bar in bars().values())
    # OCR findings picked: coloured by how they were found
    combo = widget.category_combo
    combo.setCurrentIndex(combo.findData("source:ocr"))
    assert widget.colour_by_source
    assert SOURCE_COLOURS[SOURCE_OCR] in bars()[1] and SOURCE_COLOURS[SOURCE_OCR] in widget.summary_label.text()
    combo.setCurrentIndex(combo.findData("source:pixels"))
    assert SOURCE_COLOURS[SOURCE_PIXELS] in bars()[2]
    # a kind picked again: by kind
    combo.setCurrentIndex(combo.findData("text"))
    assert not widget.colour_by_source

    titles = {card.cluster_id: card.title_label.text() for card in _cards(widget)}
    assert ">OCR<" in titles[1] and ">Pixel<" in titles[2]  # how each was found is always said

    pane = PaneView()
    pane.set_markers([ocr, pixels], None)
    assert pane._marker_items[1].pen().color() == QColor(category_colour("text"))
    pane.set_markers([ocr, pixels], None, by_source=True)
    assert pane._marker_items[1].pen().color() == QColor(SOURCE_COLOURS[SOURCE_OCR])
    assert pane._marker_items[2].pen().color() == QColor(SOURCE_COLOURS[SOURCE_PIXELS])


def _uncertain(cluster_id: int) -> DiffCluster:
    member = DiffCandidate(BoundingBox(0, 0, 10, 10), DiffType.TEXT, DiffSubtype.CHANGED, 0.25, "'a' -> 'b'", uncertain=True)
    return DiffCluster(
        id=cluster_id, bbox_in_original=member.bbox, bbox_in_suspect=member.bbox, type=DiffType.TEXT, members=[member], detail=member.detail
    )


def test_a_reading_the_pixels_do_not_bear_out_is_listed_apart_and_shown_on_request(qapp):
    widget = DiffListPanel()
    widget.show()
    widget.set_result([_uncertain(1)], _alignment(), [])
    qapp.processEvents()
    assert widget.visible_clusters() == [] and "1 possible misread" in widget.summary_label.text()
    assert widget.title_label.text() == "Findings (0)"

    widget.show_uncertain_check.setChecked(True)
    assert len(widget.visible_clusters()) == 1 and widget.title_label.text() == "Findings (1)"
    assert "possible misread" in _cards(widget)[0].title_label.text()
    widget.close()


def _finding(cluster_id: int, *, note: bool = False, y: float = 0) -> DiffCluster:
    member = DiffCandidate(
        BoundingBox(0, y, 10, y + 10), DiffType.TEXT, DiffSubtype.REMOVED, 1.0, "'UNVARNISHED AREA' missing" if note else "'a' -> 'b'",
        printers_note=note,
    )
    return DiffCluster(
        id=cluster_id, bbox_in_original=member.bbox, bbox_in_suspect=member.bbox, type=DiffType.TEXT, members=[member], detail=member.detail
    )


def test_notes_to_the_printer_are_hidden_until_picked_and_then_listed_alone(qapp):
    """A note the digital file draws for the printer ('unvarnished area') is never printed: it is not counted or drawn with
    the findings, but "Printer's notes" in the kinds lists them -- and only them."""
    widget = DiffListPanel()
    widget.show()
    change, note = _finding(1), _finding(2, note=True)
    widget.set_result([change, note], _alignment(), [])
    qapp.processEvents()
    assert widget.visible_clusters() == [change] and widget.marker_clusters() == [change]
    assert widget.title_label.text() == "Findings (1)" and "1 printer's note(s) hidden" in widget.summary_label.text()

    widget.category_combo.setCurrentIndex(widget.category_combo.findText("Printer's notes"))
    assert widget.visible_clusters() == [note] and widget.marker_clusters() == [note]
    assert widget.title_label.text() == "Findings (1)" and "printer's note" in _cards(widget)[0].title_label.text()
    assert "hidden" not in widget.summary_label.text()

    widget.set_result([change], _alignment(), [])  # none now: not offered, and the list is back to every kind
    assert widget.category_combo.findText("Printer's notes") == -1 and widget.visible_clusters() == [change]
    widget.close()


def test_notes_to_the_printer_are_numbered_after_the_findings():
    from app.models.diff import renumbered

    note, change = _finding(0, note=True, y=0), _finding(0, y=50)  # the note first in reading order
    assert renumbered([note, change]) == [change, note] and (change.id, note.id) == (1, 2)


# ---------------------------------------------------------------------------
# Ignore (FR-7.5)
# ---------------------------------------------------------------------------


def test_ignoring_a_finding_removes_it_from_the_list_but_not_from_the_result(panel, outcome):
    target = outcome.clusters[0]
    total = len(outcome.clusters)

    panel._on_ignore_toggled(target.id)

    assert target.status is DiffStatus.IGNORED
    assert len(outcome.clusters) == total  # soft state, never a deletion
    assert target not in panel.visible_clusters()
    assert panel.title_label.text() == f"Findings ({total - 1})"
    assert "1 ignored" in panel.summary_label.text()


def test_ignoring_does_not_touch_the_findings_data(panel, outcome):
    target = outcome.clusters[0]
    members = list(target.members)
    box = target.bbox_in_original

    panel._on_ignore_toggled(target.id)

    assert target.members == members
    assert target.bbox_in_original == box


def test_ignoring_tells_the_viewer_to_redraw(panel, outcome):
    fired = []
    panel.visibilityChanged.connect(lambda: fired.append(True))
    panel._on_ignore_toggled(outcome.clusters[0].id)
    assert fired


def test_ignored_findings_can_be_shown_and_restored(panel, outcome):
    target = outcome.clusters[0]
    panel._on_ignore_toggled(target.id)

    panel.show_ignored_check.setChecked(True)
    assert target in panel.visible_clusters()
    card = next(c for c in _cards(panel) if c.cluster_id == target.id)
    assert card.ignore_button.text() == "Restore"

    panel._on_ignore_toggled(target.id)  # restore
    assert target.status is DiffStatus.ACTIVE
    assert panel.title_label.text() == f"Findings ({len(outcome.clusters)})"


def test_an_ignored_finding_is_never_drawn_even_when_listed(panel, outcome):
    """Showing ignored findings in the list must not put their markers back."""
    target = outcome.clusters[0]
    panel._on_ignore_toggled(target.id)
    panel.show_ignored_check.setChecked(True)

    assert target in panel.visible_clusters()
    assert target not in panel.marker_clusters()


def test_ignoring_is_logged_for_traceability(panel, outcome, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="app.ui.diff_list_panel"):
        panel._on_ignore_toggled(outcome.clusters[0].id)

    logged = [r for r in caplog.records if r.message == "finding ignored"]
    assert len(logged) == 1
    assert logged[0].cluster_id == outcome.clusters[0].id


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def test_filtering_by_category(panel, outcome):
    panel.category_combo.setCurrentIndex(panel.category_combo.findData("graphics"))
    shown = panel.visible_clusters()
    assert shown and all(c.category == "graphics" for c in shown)
    assert len(_cards(panel)) == len(shown)


def test_filtering_by_a_kind_with_no_findings_says_so(panel, outcome):
    absent = next(key for key in CATEGORY_LABELS if all(c.category != key for c in outcome.clusters))
    panel.category_combo.setCurrentIndex(panel.category_combo.findData(absent))
    assert panel.visible_clusters() == []
    labels = [label.text() for label in panel.findChildren(type(panel.title_label))]
    assert "No findings match the current filters." in labels


def test_filters_change_what_the_viewer_draws(panel, outcome):
    fired = []
    panel.visibilityChanged.connect(lambda: fired.append(True))
    panel.category_combo.setCurrentIndex(panel.category_combo.findData("graphics"))
    assert fired
    assert all(c.category == "graphics" for c in panel.marker_clusters())


def test_filtering_does_not_change_the_headline_count(panel, outcome):
    """The count is findings in the result, not findings that pass the filter."""
    panel.category_combo.setCurrentIndex(panel.category_combo.findData("graphics"))
    assert panel.title_label.text() == f"Findings ({len(outcome.clusters)})"


def test_a_filter_that_matches_nothing_says_so(panel):
    panel.category_combo.setCurrentIndex(panel.category_combo.findData("text"))
    labels = [label.text() for label in panel.findChildren(type(panel.title_label))]
    assert "No findings match the current filters." in labels


# ---------------------------------------------------------------------------
# Selection, alignment status and warnings
# ---------------------------------------------------------------------------


def test_selecting_a_finding_highlights_its_card(panel, outcome):
    target = outcome.clusters[1]
    panel.select(target.id)

    selected = [c for c in _cards(panel) if c.property("selected") == "true"]
    assert [c.cluster_id for c in selected] == [target.id]


def test_choosing_a_card_reports_the_finding(panel, outcome):
    chosen = []
    panel.clusterSelected.connect(chosen.append)
    card = _cards(panel)[0]

    card.selectRequested.emit(card.cluster_id)

    assert chosen == [card.cluster_id]


def test_choosing_the_already_selected_card_again_deselects_it(panel, outcome):
    chosen = []
    panel.clusterSelected.connect(chosen.append)
    card = _cards(panel)[0]
    card.selectRequested.emit(card.cluster_id)  # select it first

    card.selectRequested.emit(card.cluster_id)  # click it again

    assert chosen == [card.cluster_id, None]
    assert panel._selected_id is None
    assert not any(c.property("selected") == "true" for c in _cards(panel))


def test_choosing_a_different_card_selects_it_rather_than_deselecting(panel, outcome):
    chosen = []
    panel.clusterSelected.connect(chosen.append)
    first, second = _cards(panel)[0], _cards(panel)[1]
    first.selectRequested.emit(first.cluster_id)

    second.selectRequested.emit(second.cluster_id)

    assert chosen == [first.cluster_id, second.cluster_id]


def test_the_alignment_error_is_shown_as_a_pill(panel, outcome):
    assert "px" in panel.alignment_pill.text()
    assert panel.alignment_pill.property("pill") == "loaded"


def test_low_confidence_alignment_is_flagged_prominently(qapp, outcome):
    """NFR-3: never proceed as though registration were perfect."""
    widget = DiffListPanel()
    widget.set_result(outcome.clusters, _alignment(3.4, low_confidence=True), [])
    assert "LOW CONFIDENCE" in widget.alignment_pill.text()
    assert widget.alignment_pill.property("pill") == "warn"


def test_warnings_about_unchecked_content_are_shown(qapp, outcome):
    widget = DiffListPanel()
    widget.show()
    widget.set_result(outcome.clusters, outcome.alignment, ["OCR is unavailable"])
    assert not widget.warnings_label.isHidden()
    assert "OCR is unavailable" in widget.warnings_label.text()

    widget.set_result(outcome.clusters, outcome.alignment, [])
    assert widget.warnings_label.isHidden()
    widget.close()


def test_no_differences_reports_that_plainly(qapp):
    widget = DiffListPanel()
    widget.set_result([], _alignment(), [])
    labels = [label.text() for label in widget.findChildren(type(widget.title_label))]
    assert widget.title_label.text() == "Findings (0)"
    assert "No differences found." in labels
