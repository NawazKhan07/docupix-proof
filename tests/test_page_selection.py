"""Choosing which page of a file of several to compare, and deselecting a finding by clicking away from it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pymupdf as fitz
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from app.config import AppConfig
from app.engine.file_loader import load_artwork
from app.engine.page_choice import choose_pages
from app.engine.worker_process import ComparisonRequest, execute_request
from app.models.diff import BoundingBox, DiffCandidate, DiffCluster, DiffSubtype, DiffType
from app.models.job import ComparisonMode
from app.ui.results_view import ResultsView
from app.ui.upload_panel import UploadSlot
from tests.helpers import make_carton_pdf


def _carton_page(page, label: str, colour) -> None:
    """A carton-like page with plenty of features of its own (a band of its colour, a label, a grid of marks)."""
    page.draw_rect(fitz.Rect(30, 30, 270, 170), color=(0, 0, 0), width=1.5)
    page.draw_rect(fitz.Rect(40, 40, 260, 70), color=None, fill=colour)
    page.insert_text((50, 100), label, fontsize=14)
    for row in range(3):
        for column in range(8):
            if (row * 8 + column) % 3 != hash(label) % 3:
                page.draw_circle(fitz.Point(60 + column * 25, 120 + row * 15), 4, color=None, fill=(0.1, 0.1, 0.1))


def _two_page_pdf(path: Path) -> Path:
    """Two sheets in one file: a different carton on each."""
    doc = fitz.open()
    _carton_page(doc.new_page(width=300, height=200), "OTHER PRODUCT 25", (0.8, 0.2, 0.2))
    _carton_page(doc.new_page(width=300, height=200), "RIGHT PRODUCT 50", (0.95, 0.75, 0.1))
    doc.save(path)
    doc.close()
    return path


def _one_page_pdf(path: Path) -> Path:
    doc = fitz.open()
    _carton_page(doc.new_page(width=300, height=200), "RIGHT PRODUCT 50", (0.95, 0.75, 0.1))
    doc.save(path)
    doc.close()
    return path


# ---- the engine ------------------------------------------------------------------------------------------------------


def test_the_page_the_other_files_artwork_is_on_is_chosen(tmp_path):
    single, double = load_artwork(_one_page_pdf(tmp_path / "one.pdf")), load_artwork(_two_page_pdf(tmp_path / "two.pdf"))
    first, second, first_guessed, second_guessed = choose_pages(single, double, None, None)
    assert (first, second) == (0, 1)
    assert not first_guessed and second_guessed


def test_a_page_the_reviewer_chose_is_kept(tmp_path):
    single, double = load_artwork(_one_page_pdf(tmp_path / "one.pdf")), load_artwork(_two_page_pdf(tmp_path / "two.pdf"))
    assert choose_pages(single, double, None, 0)[:2] == (0, 0)


def test_the_comparison_says_which_page_it_compared(tmp_path):
    outcome = execute_request(
        ComparisonRequest(_one_page_pdf(tmp_path / "one.pdf"), _two_page_pdf(tmp_path / "two.pdf"), ComparisonMode.AUTOMATIC, AppConfig())
    )
    assert outcome.suspect_page == 1
    assert any("has 2 pages: page 2 was compared" in note for note in outcome.notes)


def test_a_single_page_file_says_nothing_about_pages(tmp_path):
    a, b = make_carton_pdf(tmp_path / "a.pdf"), make_carton_pdf(tmp_path / "b.pdf")
    outcome = execute_request(ComparisonRequest(a, b, ComparisonMode.AUTOMATIC, AppConfig()))
    assert not any("pages:" in note for note in outcome.notes)


def test_a_file_renders_its_selected_page_by_default(tmp_path):
    artwork = load_artwork(_two_page_pdf(tmp_path / "two.pdf"))
    first = artwork.render(dpi=36)
    artwork.select_page(1)
    assert not np.array_equal(artwork.render(dpi=36), first)
    assert artwork.pdf_page().number == 1


# ---- the upload page ---------------------------------------------------------------------------------------------------


def test_a_file_of_several_pages_offers_its_pages_and_one_page_does_not(qapp, tmp_path):
    slot = UploadSlot("Suspect")
    slot.show()
    slot.load_path(_two_page_pdf(tmp_path / "two.pdf"))
    assert slot.page_selector.isVisibleTo(slot) and slot.page_choice is None  # best match until one is picked
    slot.page_selector.set_choice(None)
    slot.page_selector.combo.setCurrentIndex(2)  # page 2
    assert slot.page_choice == 1 and slot.loaded.page == 1
    assert "page 2 is compared" in slot.detail_label.text()

    slot.set_mode(ComparisonMode.LEAFLET)
    assert not slot.page_selector.isVisibleTo(slot)  # Leaflet compares every page
    slot.set_mode(ComparisonMode.AUTOMATIC)
    slot.load_path(_one_page_pdf(tmp_path / "one.pdf"))
    assert not slot.page_selector.isVisibleTo(slot) and slot.page_choice is None
    slot.close()


def test_picking_another_page_drops_an_area_drawn_on_the_old_one(qapp, tmp_path):
    from app.models.isolation import IsolationGeometry

    slot = UploadSlot("Suspect")
    slot.load_path(_two_page_pdf(tmp_path / "two.pdf"))
    slot.geometry = IsolationGeometry()
    slot.page_selector.combo.setCurrentIndex(2)
    assert slot.geometry is None


# ---- the results page ---------------------------------------------------------------------------------------------------


def _outcome(tmp_path):
    return execute_request(
        ComparisonRequest(_one_page_pdf(tmp_path / "one.pdf"), _two_page_pdf(tmp_path / "two.pdf"), ComparisonMode.AUTOMATIC, AppConfig())
    )


def test_the_results_page_offers_the_pages_of_a_file_of_several(qapp, tmp_path):
    view = ResultsView()
    outcome = _outcome(tmp_path)
    original, suspect = load_artwork(tmp_path / "one.pdf"), load_artwork(tmp_path / "two.pdf")
    view.show_outcome(outcome, "one vs two", original=original, suspect=suspect)
    assert not view.original_page_selector.isVisibleTo(view)  # one page: nothing to pick
    assert view.suspect_page_selector.isVisibleTo(view) and view.suspect_page_selector.choice == 1

    asked = []
    view.pageChangeRequested.connect(lambda which, page: asked.append((which, page)))
    view.suspect_page_selector.combo.setCurrentIndex(0)
    assert asked == [("suspect", 0)]


# ---- deselecting a finding ------------------------------------------------------------------------------------------------


def _view_with_findings(qapp):
    view = ResultsView()
    view.resize(1200, 800)
    view.show()
    page = np.full((1200, 1600, 3), 255, np.uint8)
    from app.engine.worker import ComparisonOutcome
    from app.models.result import AlignmentResult

    clusters = []
    for number, box in enumerate([(100, 100, 200, 160), (900, 700, 1000, 760)], start=1):
        member = DiffCandidate(BoundingBox(*box), DiffType.GRAPHICS, DiffSubtype.CHANGED, 1.0, "Graphic changed")
        clusters.append(DiffCluster(number, member.bbox, member.bbox, DiffType.GRAPHICS, [member], detail=member.detail))
    alignment = AlignmentResult(
        transform=np.eye(3), residual_error_px=0.0, residual_basis="keypoints", method="features",
        match_count=100, inlier_count=100, inlier_ratio=1.0, threshold_px=2.0, low_confidence=False,
    )
    outcome = ComparisonOutcome(alignment=alignment, candidates=[], clusters=clusters, aligned_original=page, aligned_suspect=page.copy(),
                                mode=ComparisonMode.AUTOMATIC)
    view.show_outcome(outcome, "a vs b")
    view.viewport.fit_to_view()
    qapp.processEvents()
    return view


def test_clicking_the_picture_where_there_is_no_finding_deselects(qapp):
    view = _view_with_findings(qapp)
    view._on_marker_clicked(1)
    assert view.panel._selected_id == 1
    pane = view.viewport.original_pane
    empty = pane.mapFromScene(1400, 400)
    QTest.mouseClick(pane.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(empty.x(), empty.y()))
    qapp.processEvents()
    assert view.panel._selected_id is None
    view.close()


def test_clicking_the_selected_finding_again_deselects_it(qapp):
    view = _view_with_findings(qapp)
    view._on_marker_clicked(2)
    view._on_marker_clicked(2)
    assert view.panel._selected_id is None
    view.close()


def test_clicking_elsewhere_on_the_results_page_deselects(qapp):
    view = _view_with_findings(qapp)
    view._on_marker_clicked(1)
    QTest.mouseClick(view.subtitle_label, Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert view.panel._selected_id is None
    view.close()


def test_choosing_a_finding_card_selects_it_and_does_not_deselect_it_at_once(qapp):
    view = _view_with_findings(qapp)
    card = view.panel._cards_layout.itemAt(0).widget()
    QTest.mouseClick(card, Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert view.panel._selected_id is not None
    view.close()
