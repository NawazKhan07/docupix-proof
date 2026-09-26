"""Comparing a multi-page document page by page (`app.engine.leaflet`)."""

from __future__ import annotations

from dataclasses import replace

import leaflet_scenes as scenes
import numpy as np
import pytesseract
import pytest

from app.config import AppConfig
from app.engine.leaflet import (
    _PAGE_GAP_COLOUR,
    NoPagesComparedError,
    _pair_pages,
    _page_similarity,
    _parse_page_selection,
    _thumbnail,
    _word_set,
    compare_leaflet,
)
from app.engine.ocr_locator import find_tesseract
from app.models.diff import DiffSubtype
from app.models.job import ComparisonMode


@pytest.fixture(autouse=True)
def _ocr(monkeypatch):
    # Scoped to this test (like every other file's OCR fixture): it must not leave Tesseract configured for whatever
    # test runs next, or a test elsewhere that relies on it being UNconfigured (Scan vs Digital reads text only when
    # it is) would see different results depending on what ran before it.
    tesseract = find_tesseract()
    if tesseract is not None:
        monkeypatch.setattr(pytesseract.pytesseract, "tesseract_cmd", str(tesseract))


class _FakeArtwork:
    """Just enough of `LoadedArtwork` for `compare_leaflet`: a fixed list of already-rendered pages."""

    def __init__(self, pages):
        self._pages = pages
        self.page_count = len(pages)

    def render(self, dpi=None, page=0):
        return self._pages[page]


CONFIG = AppConfig().leaflet


# ---- pairing pages by what is on them, not their position ------------------------------------------------------


def test_identical_pages_pair_with_a_score_of_one():
    thumb = _thumbnail(scenes.page("dosage"), CONFIG.thumbnail_long_edge_px)
    assert _page_similarity(thumb, thumb) == pytest.approx(1.0, abs=1e-6)


def test_different_pages_score_lower_than_identical_ones():
    same = _thumbnail(scenes.page("dosage"), CONFIG.thumbnail_long_edge_px)
    other = _thumbnail(scenes.page("warnings"), CONFIG.thumbnail_long_edge_px)
    assert _page_similarity(other, same) < _page_similarity(same, same)


def test_a_pages_words_tell_it_from_a_page_that_only_looks_like_it():
    """Two pages can share a layout (the same banner, the same line spacing): it is the words that say they differ."""
    same_words = _word_set(scenes.page("dosage"), CONFIG.language)
    other_words = _word_set(scenes.page("warnings"), CONFIG.language)
    if same_words is None or other_words is None:
        pytest.skip("needs Tesseract")
    assert len(same_words & other_words) / len(same_words | other_words) < 0.3


def test_pages_in_the_same_order_pair_one_to_one():
    pages = [scenes.page(s) for s in ("indications", "dosage", "warnings")]
    pairs, unmatched_o, unmatched_s = _pair_pages(pages, list(pages), CONFIG, CONFIG.language)
    assert pairs == [(0, 0), (1, 1), (2, 2)]
    assert unmatched_o == [] and unmatched_s == []


def test_an_inserted_page_does_not_throw_the_pages_after_it_out_of_step():
    original = [scenes.page(s) for s in ("indications", "dosage", "warnings")]
    suspect = [scenes.page(s) for s in ("indications", "inserted", "dosage", "warnings")]
    pairs, unmatched_o, unmatched_s = _pair_pages(original, suspect, CONFIG, CONFIG.language)
    assert pairs == [(0, 0), (1, 2), (2, 3)]
    assert unmatched_o == [] and unmatched_s == [1]


def test_a_page_with_nothing_alike_in_the_other_file_is_left_unmatched():
    original = [scenes.page(s) for s in ("indications", "dosage", "warnings")]
    suspect = [scenes.page(s) for s in ("indications", "dosage")]
    pairs, unmatched_o, unmatched_s = _pair_pages(original, suspect, CONFIG, CONFIG.language)
    assert pairs == [(0, 0), (1, 1)]
    assert unmatched_o == [2] and unmatched_s == []


# ---- comparing every page ---------------------------------------------------------------------------------------


def test_a_change_on_one_page_is_found_and_the_others_are_silent():
    original = _FakeArtwork([scenes.page("indications"), scenes.page("dosage"), scenes.page("warnings")])
    suspect = _FakeArtwork([scenes.page("indications"), scenes.page("dosage", scenes.edit_first_line), scenes.page("warnings")])

    outcome = compare_leaflet(original, suspect, AppConfig())

    assert outcome.mode is ComparisonMode.LEAFLET
    assert outcome.clusters
    assert all(cluster.detail.startswith("Page 2:") for cluster in outcome.clusters)
    assert "3 pages" in outcome.notes[0]


def test_identical_documents_have_no_findings():
    pages = [scenes.page("indications"), scenes.page("dosage"), scenes.page("warnings")]
    outcome = compare_leaflet(_FakeArtwork(pages), _FakeArtwork(list(pages)), AppConfig())
    assert outcome.clusters == []


def test_pages_are_matched_by_content_across_an_inserted_page_and_the_change_still_found():
    original = _FakeArtwork([scenes.page("indications"), scenes.page("dosage"), scenes.page("warnings")])
    suspect = _FakeArtwork(
        [scenes.page("indications"), scenes.page("inserted"), scenes.page("dosage", scenes.edit_first_line), scenes.page("warnings")]
    )

    outcome = compare_leaflet(original, suspect, AppConfig())

    changed = [c for c in outcome.clusters if "Original p.2 / Suspect p.3" in c.detail]
    assert changed
    inserted = [c for c in outcome.clusters if "added in the suspect" in c.detail]
    assert len(inserted) == 1 and inserted[0].subtype is DiffSubtype.ADDED


def test_a_page_missing_from_the_suspect_is_its_own_finding():
    original = _FakeArtwork([scenes.page("indications"), scenes.page("dosage"), scenes.page("warnings")])
    suspect = _FakeArtwork([scenes.page("indications"), scenes.page("dosage")])

    outcome = compare_leaflet(original, suspect, AppConfig())

    missing = [c for c in outcome.clusters if "missing from the suspect" in c.detail]
    assert len(missing) == 1
    assert missing[0].subtype is DiffSubtype.REMOVED
    assert "warnings" not in [c.detail for c in outcome.clusters if c is missing[0]]  # (just the one finding, no per-line noise)


def test_the_findings_are_shown_on_their_own_page_in_the_stacked_view():
    original = _FakeArtwork([scenes.page("indications"), scenes.page("dosage")])
    suspect = _FakeArtwork([scenes.page("indications"), scenes.page("dosage", scenes.edit_first_line)])

    outcome = compare_leaflet(original, suspect, AppConfig())

    # A full rewritten sentence can read back as more than one word-level finding under the text strategy;
    # what this test checks is that each one is offset onto page 2's own position in the stack, not the count.
    assert outcome.clusters
    page_two = _gap_rows(outcome.aligned_original)[0]  # page 2 starts below the gap between the pages
    assert all(cluster.bbox_in_original.y0 > page_two for cluster in outcome.clusters)


def _gap_rows(stack: np.ndarray) -> list[int]:
    """The rows of the dark band between stacked pages."""
    gap = np.all(stack == np.array(_PAGE_GAP_COLOUR, np.uint8), axis=(1, 2))
    return [int(y) for y in np.flatnonzero(gap)]


def test_both_stacks_are_one_frame_page_for_page():
    """Each suspect page is shown warped onto the original page it was paired with, so the two stacks are the same
    size and every view that blends them (Flicker, Overlay, Grayscale) works on a leaflet too."""
    original = _FakeArtwork([scenes.page("indications"), scenes.page("dosage")])
    shifted = np.roll(scenes.page("dosage"), (40, 25), axis=(0, 1))  # the same page, printed a little off
    suspect = _FakeArtwork([scenes.page("indications"), shifted])

    outcome = compare_leaflet(original, suspect, AppConfig())

    assert outcome.aligned_original.shape == outcome.aligned_suspect.shape
    assert _gap_rows(outcome.aligned_original) == _gap_rows(outcome.aligned_suspect)


def test_a_wide_white_margin_round_a_page_is_trimmed():
    """The same page set on a sheet with far more blank paper round it: what is compared and shown is the page."""
    page = scenes.page("dosage")
    margined = np.full((page.shape[0] + 800, page.shape[1] + 1000, 3), 255, np.uint8)
    margined[300 : 300 + page.shape[0], 500 : 500 + page.shape[1]] = page
    outcome = compare_leaflet(_FakeArtwork([margined]), _FakeArtwork([page]), AppConfig())

    height, width = outcome.aligned_original.shape[:2]
    assert height < page.shape[0] + 100 and width < page.shape[1] + 100


def test_ids_are_renumbered_in_reading_order():
    original = _FakeArtwork([scenes.page("indications"), scenes.page("dosage"), scenes.page("warnings")])
    suspect = _FakeArtwork(
        [scenes.page("indications", scenes.edit_first_line), scenes.page("dosage"), scenes.page("warnings", scenes.edit_first_line)]
    )
    outcome = compare_leaflet(original, suspect, AppConfig())
    ids = [cluster.id for cluster in outcome.clusters]
    assert ids == list(range(1, len(ids) + 1))
    ys = [cluster.bbox_in_original.y0 for cluster in outcome.clusters]
    assert ys == sorted(ys)


def test_no_pages_at_all_is_reported_plainly():
    with pytest.raises(NoPagesComparedError):
        compare_leaflet(_FakeArtwork([]), _FakeArtwork([scenes.page("indications")]), AppConfig())


def test_an_unavailable_language_falls_back_and_says_so(monkeypatch):
    import app.engine.leaflet as leaflet_module

    monkeypatch.setattr(leaflet_module, "usable_language", lambda requested: ("eng", False))
    original = _FakeArtwork([scenes.page("indications")])
    suspect = _FakeArtwork([scenes.page("indications", scenes.edit_first_line)])

    outcome = compare_leaflet(original, suspect, AppConfig())

    assert any("was used instead" in warning for warning in outcome.warnings)


# ---- the two independent strategies: text (the default) and visual --------------------------------------------


def test_text_is_the_default_strategy_and_finds_a_text_change():
    original = _FakeArtwork([scenes.page("dosage")])
    suspect = _FakeArtwork([scenes.page("dosage", scenes.edit_first_line)])

    outcome = compare_leaflet(original, suspect, AppConfig())  # default: text on, visual off

    assert outcome.clusters
    assert outcome.text_compared is True
    assert outcome.graphics_compared is False


def test_visual_only_finds_a_pixel_change():
    config = replace(AppConfig(), leaflet=replace(AppConfig().leaflet, compare_text=False, compare_visual=True))
    original = _FakeArtwork([scenes.page("dosage")])
    suspect = _FakeArtwork([scenes.page("dosage", scenes.edit_first_line)])

    outcome = compare_leaflet(original, suspect, config)

    assert outcome.clusters
    assert outcome.text_compared is False
    assert outcome.graphics_compared is True


def test_both_strategies_together_merge_into_one_set_of_findings_with_unique_ids():
    config = replace(AppConfig(), leaflet=replace(AppConfig().leaflet, compare_text=True, compare_visual=True))
    original = _FakeArtwork([scenes.page("dosage")])
    suspect = _FakeArtwork([scenes.page("dosage", scenes.edit_first_line)])

    outcome = compare_leaflet(original, suspect, config)

    assert outcome.clusters
    ids = [cluster.id for cluster in outcome.clusters]
    assert len(ids) == len(set(ids))
    assert outcome.text_compared is True
    assert outcome.graphics_compared is True


def test_identical_pages_have_no_findings_under_either_strategy():
    pages = [scenes.page("dosage")]
    for compare_text, compare_visual in [(True, False), (False, True), (True, True)]:
        config = replace(AppConfig(), leaflet=replace(AppConfig().leaflet, compare_text=compare_text, compare_visual=compare_visual))
        outcome = compare_leaflet(_FakeArtwork(pages), _FakeArtwork(list(pages)), config)
        assert outcome.clusters == []


# ---- restricting which pages are reviewed -----------------------------------------------------------------------


def test_a_blank_selection_means_every_page():
    assert _parse_page_selection("", 5) is None
    assert _parse_page_selection("   ", 5) is None


def test_a_selection_parses_single_pages_and_ranges_to_zero_indexed():
    assert _parse_page_selection("1,3-5", 10) == {0, 2, 3, 4}


def test_a_selection_ignores_out_of_range_and_malformed_parts():
    assert _parse_page_selection("1, 99, two, x-y", 5) == {0}


def test_a_trailing_dash_is_read_as_that_one_page():
    assert _parse_page_selection("3-", 5) == {2}


def test_a_reversed_range_still_works():
    assert _parse_page_selection("5-3", 10) == {2, 3, 4}


def test_restricting_to_one_page_only_compares_that_page():
    config = replace(AppConfig(), leaflet=replace(AppConfig().leaflet, compare_pages="2"))
    original = _FakeArtwork([scenes.page("indications"), scenes.page("dosage"), scenes.page("warnings")])
    suspect = _FakeArtwork(
        [scenes.page("indications", scenes.edit_first_line), scenes.page("dosage", scenes.edit_first_line), scenes.page("warnings", scenes.edit_first_line)]
    )

    outcome = compare_leaflet(original, suspect, config)

    assert outcome.clusters
    assert all(cluster.detail.startswith("Page 2:") for cluster in outcome.clusters)
    assert "Restricted to original page(s) 2" in " ".join(outcome.notes)


def test_a_page_left_out_of_the_selection_is_not_reported_as_missing():
    config = replace(AppConfig(), leaflet=replace(AppConfig().leaflet, compare_pages="1"))
    original = _FakeArtwork([scenes.page("indications"), scenes.page("dosage")])
    suspect = _FakeArtwork([scenes.page("indications")])  # page 2 genuinely missing, but it is not selected

    outcome = compare_leaflet(original, suspect, config)

    assert outcome.clusters == []
    assert not any("missing" in note for note in outcome.notes)
    assert not any("missing" in warning for warning in outcome.warnings)
