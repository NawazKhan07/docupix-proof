"""Comparing a multi-page document page by page (`app.engine.leaflet`)."""

from __future__ import annotations

import leaflet_scenes as scenes
import pytesseract
import pytest

from app.config import AppConfig
from app.engine.leaflet import NoPagesComparedError, _pair_pages, _page_similarity, _thumbnail, _word_set, compare_leaflet
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

    (cluster,) = [c for c in outcome.clusters]
    assert cluster.bbox_in_original.y0 > scenes.PAGE_SIZE[0]  # below page 1's own height: on page 2
    assert outcome.aligned_original.shape[0] > scenes.PAGE_SIZE[0] * 2  # both pages, plus a gap, are in the stack


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
