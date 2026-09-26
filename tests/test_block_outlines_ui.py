"""The results page outlines each panel a block-wise comparison found and aligned on its own, the same way
it already outlines the area it covered (`tests/test_compared_area.py`)."""

from __future__ import annotations

import dataclasses

import pytest

from app.ui.pane_view import ImageStack, PaneView
from app.ui.results_view import ResultsView


@pytest.fixture
def results(qapp, outcome):
    view = ResultsView()
    view.resize(1400, 800)
    view.show()
    yield view
    view.close()


def test_a_pane_outlines_what_block_wise_alignment_found(qapp, outcome):
    pane = PaneView()
    pane.set_stacks({"original": ImageStack(outcome.aligned_original)})
    before = len(pane._scene.items())
    height, width = outcome.aligned_original.shape[:2]

    pane.set_block_outlines([[(width * 0.1, height * 0.1), (width * 0.4, height * 0.1), (width * 0.4, height * 0.9), (width * 0.1, height * 0.9)]])

    assert len(pane._scene.items()) == before + 1 and pane.block_outlines_visible
    pane.show_block_outlines(False)
    assert len(pane._scene.items()) == before and not pane.block_outlines_visible
    pane.show_block_outlines(True)
    pane.set_block_outlines(None)
    assert len(pane._scene.items()) == before


def test_the_results_page_offers_the_toggle_only_when_blocks_were_found(results, outcome, qapp):
    results.show_outcome(dataclasses.replace(outcome, block_outlines=[]), "a  vs  b")
    qapp.processEvents()
    assert results.viewport.block_outline_check.isHidden()

    height, width = outcome.aligned_original.shape[:2]
    blocks = [[(width * 0.1, height * 0.1), (width * 0.4, height * 0.1), (width * 0.4, height * 0.9), (width * 0.1, height * 0.9)]]
    results.show_outcome(dataclasses.replace(outcome, block_outlines=blocks), "a  vs  b")
    qapp.processEvents()
    check = results.viewport.block_outline_check
    assert not check.isHidden() and check.isChecked()
    assert results.viewport.original_pane.block_outlines_visible and results.viewport.suspect_pane.block_outlines_visible


def test_the_block_outline_toggle_hides_and_shows_it_in_both_panes(results, outcome, qapp):
    height, width = outcome.aligned_original.shape[:2]
    blocks = [[(width * 0.1, height * 0.1), (width * 0.4, height * 0.1), (width * 0.4, height * 0.9), (width * 0.1, height * 0.9)]]
    results.show_outcome(dataclasses.replace(outcome, block_outlines=blocks), "a  vs  b")

    results.viewport.block_outline_check.setChecked(False)
    assert not results.viewport.original_pane.block_outlines_visible and not results.viewport.suspect_pane.block_outlines_visible
    results.viewport.block_outline_check.setChecked(True)
    assert results.viewport.original_pane.block_outlines_visible


def test_a_new_result_without_blocks_clears_the_old_outlines(results, outcome, qapp):
    height, width = outcome.aligned_original.shape[:2]
    blocks = [[(10, 10), (width - 10, 10), (width - 10, height - 10), (10, height - 10)]]
    results.show_outcome(dataclasses.replace(outcome, block_outlines=blocks), "x")
    results.show_outcome(dataclasses.replace(outcome, block_outlines=[]), "y")
    assert not results.viewport.original_pane.block_outlines_visible
    assert results.viewport.block_outline_check.isHidden()
