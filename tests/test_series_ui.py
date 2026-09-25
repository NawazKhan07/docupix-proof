"""Series mode's UI touches: a mode card that says what it needs, a hint that the suspect is not isolated, and the
viewer not offering to blend two panes of different sizes together (`app.ui.mode_selector`, `app.ui.upload_panel`,
`app.ui.comparison_viewport`, `app.ui.results_view`)."""

from __future__ import annotations

import numpy as np

from app.engine.worker import ComparisonOutcome
from app.models.job import ComparisonMode
from app.models.result import AlignmentResult
from app.ui.comparison_viewport import ComparisonViewport
from app.ui.composition import ViewMode
from app.ui.mode_selector import ModeSelector
from app.ui.results_view import ResultsView
from app.ui.upload_panel import UploadPanel


def test_series_is_offered_as_a_mode_and_says_what_it_needs(qapp):
    selector = ModeSelector()
    assert ComparisonMode.SERIES in selector._cards
    text = selector._cards[ComparisonMode.SERIES].findChild(type(selector._cards[ComparisonMode.SERIES].radio)).text()
    assert "Series" in text
    selector.set_mode(ComparisonMode.SERIES)
    assert selector.mode is ComparisonMode.SERIES


def test_choosing_series_hints_that_the_suspect_is_not_isolated(qapp):
    panel = UploadPanel()
    panel.show()
    assert not panel.suspect_slot.mode_hint_label.isVisible()

    panel.set_mode(ComparisonMode.SERIES)
    assert panel.suspect_slot.mode_hint_label.isVisible()
    assert "whole sheet" in panel.suspect_slot.mode_hint_label.text()

    panel.set_mode(ComparisonMode.AUTOMATIC)
    assert not panel.suspect_slot.mode_hint_label.isVisible()
    panel.close()


def _alignment() -> AlignmentResult:
    return AlignmentResult(
        transform=np.eye(3), residual_error_px=0.5, residual_basis="keypoints", method="features",
        match_count=50, inlier_count=50, inlier_ratio=1.0, low_confidence=False, threshold_px=2.0,
    )


def test_flicker_overlay_and_grayscale_are_off_for_two_panes_of_different_sizes(qapp):
    viewport = ComparisonViewport()
    viewport.show()
    viewport.set_images(np.full((400, 300, 3), 255, np.uint8), np.full((1400, 1800, 3), 255, np.uint8))
    viewport.set_composite_modes_enabled(False)

    for mode in (ViewMode.FLICKER, ViewMode.OVERLAY, ViewMode.GRAYSCALE):
        assert not viewport._mode_buttons[mode].isEnabled()
    assert viewport._mode is ViewMode.SIDE_BY_SIDE
    viewport.close()


def test_re_enabling_composite_modes_restores_the_buttons(qapp):
    viewport = ComparisonViewport()
    viewport.show()
    viewport.set_composite_modes_enabled(False)
    viewport.set_composite_modes_enabled(True)
    for mode in (ViewMode.FLICKER, ViewMode.OVERLAY, ViewMode.GRAYSCALE):
        assert viewport._mode_buttons[mode].isEnabled()
    viewport.close()


def test_results_view_disables_composite_modes_only_when_the_two_panes_differ_in_size(qapp):
    view = ResultsView()
    view.show()

    small_original = np.full((400, 300, 3), 255, np.uint8)
    whole_sheet = np.full((1400, 1800, 3), 255, np.uint8)
    series_outcome = ComparisonOutcome(
        alignment=_alignment(), candidates=[], clusters=[], aligned_original=small_original,
        aligned_suspect=whole_sheet, mode=ComparisonMode.SERIES,
    )
    view.show_outcome(series_outcome, "Series")
    assert not view.viewport._mode_buttons[ViewMode.OVERLAY].isEnabled()

    same_size_outcome = ComparisonOutcome(
        alignment=_alignment(), candidates=[], clusters=[], aligned_original=small_original,
        aligned_suspect=small_original.copy(), mode=ComparisonMode.AUTOMATIC,
    )
    view.show_outcome(same_size_outcome, "Automatic")
    assert view.viewport._mode_buttons[ViewMode.OVERLAY].isEnabled()
    view.close()
