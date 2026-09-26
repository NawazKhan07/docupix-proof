"""The app fits any reasonable window: rows of controls wrap,
and the viewers pan by dragging rather than with scroll bars. The findings list alone scrolls."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractScrollArea, QPushButton, QWidget

from app.models.diff import BoundingBox, DiffCandidate, DiffCluster, DiffSubtype, DiffType
from app.models.result import AlignmentResult
from app.ui.comparison_viewport import MIN_ZOOM, ComparisonViewport
from app.ui.diff_list_panel import DiffListPanel, FindingCard
from app.ui.flow_layout import flow_widget
from app.ui.main_window import NO_WARMUP_ENV, MainWindow

def _alignment() -> AlignmentResult:
    return AlignmentResult(
        transform=np.eye(3), residual_error_px=0.5, residual_basis="keypoints", method="features",
        match_count=100, inlier_count=90, inlier_ratio=0.9, low_confidence=False, threshold_px=2.0,
    )


def _finding(cluster_id: int, detail: str = "'10' -> '100'") -> DiffCluster:
    member = DiffCandidate(BoundingBox(0, 0, 10, 10), DiffType.TEXT, DiffSubtype.CHANGED, 0.9, detail)
    return DiffCluster(id=cluster_id, bbox_in_original=member.bbox, bbox_in_suspect=member.bbox, type=DiffType.TEXT,
                       members=[member], detail=detail)


# ---- a row of controls that wraps --------------------------------------------------------------------------------


def test_a_flow_row_needs_only_its_widest_control_and_wraps_the_rest(qapp):
    holder, flow = flow_widget()
    buttons = [QPushButton(f"Control number {i}") for i in range(8)]
    for button in buttons:
        flow.addWidget(button)
    widest = max(button.sizeHint().width() for button in buttons)
    assert flow.minimumSize().width() <= widest + 1
    assert flow.sizeHint().width() > 4 * widest  # one line, if it had the room

    one_line = flow.heightForWidth(flow.sizeHint().width())
    narrow = flow.heightForWidth(widest + 10)
    assert narrow >= 7 * one_line  # one control a line


# ---- no scroll bars --------------------------------------------------------------------------------------------


@pytest.fixture
def window(qapp, monkeypatch):
    monkeypatch.setenv(NO_WARMUP_ENV, "1")
    widget = MainWindow()
    widget.show()
    qapp.processEvents()
    yield widget
    widget.close()


def test_at_its_smallest_the_window_cuts_nothing_off(window, qapp):
    """Whatever the font, the window's minimum size is where every row has wrapped, not where controls get clipped."""
    for page in (window._setup_page, window.results_view):
        window.stack.setCurrentWidget(page)
        qapp.processEvents()
        window.resize(window.minimumSizeHint())
        qapp.processEvents()
        bounds = window.rect()
        for control in page.findChildren(QWidget):
            if control.isVisibleTo(page) and control.metaObject().className() in ("QPushButton", "QCheckBox", "QComboBox"):
                box = control.rect().translated(control.mapTo(window, control.rect().topLeft()))
                assert bounds.contains(box), (control.metaObject().className(), getattr(control, "text", lambda: "")())


def test_the_window_is_far_narrower_than_its_controls_laid_in_one_row(window, qapp):
    """The rows wrap: the window may be narrower than the viewer's toolbar and the findings panel side by side."""
    window.stack.setCurrentWidget(window.results_view)
    qapp.processEvents()
    results = window.results_view
    toolbar = results.viewport
    one_row = sum(button.sizeHint().width() for button in toolbar.findChildren(QPushButton) if button.isVisibleTo(toolbar))
    assert results.minimumSizeHint().width() < one_row + results.side.minimumWidth()


def test_the_viewers_have_no_scroll_bars(window):
    for view in window.findChildren(QAbstractScrollArea):
        if isinstance(view, QAbstractScrollArea) and view.metaObject().className() == "PaneView":
            assert view.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            assert view.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff


# ---- the findings list is the one part that scrolls ---------------------------------------------------------------


def test_the_findings_list_scrolls_and_brings_a_chosen_finding_into_view(qapp):
    panel = DiffListPanel()
    panel.resize(360, 520)
    panel.show()
    panel.set_result([_finding(i) for i in range(1, 31)], _alignment(), [])
    for _ in range(3):
        qapp.processEvents()
    assert len(panel.findChildren(FindingCard)) == 30
    bar = panel._scroll.verticalScrollBar()
    assert bar.maximum() > 0  # thirty cards do not fit: the list scrolls
    assert panel._scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff

    panel.select(29)
    for _ in range(3):
        qapp.processEvents()
    card = next(c for c in panel.findChildren(FindingCard) if c.cluster_id == 29)
    top = card.mapTo(panel._scroll.viewport(), card.rect().topLeft()).y()
    assert 0 <= top < panel._scroll.viewport().height()
    panel.close()


# ---- a very large document still fits ----------------------------------------------------------------------------


def test_fit_shows_the_whole_of_a_document_too_big_for_the_smallest_zoom(qapp):
    viewport = ComparisonViewport()
    viewport.resize(700, 500)
    viewport.show()
    tall = np.full((14000, 400, 3), 255, np.uint8)  # as tall as a stacked leaflet, but narrow: little memory
    viewport.set_images(tall, tall)
    viewport.fit_to_view()
    qapp.processEvents()
    assert viewport.zoom < MIN_ZOOM
    assert viewport.zoom * 14000 <= viewport.original_pane.viewport().height()
    viewport.close()
