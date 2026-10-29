"""Comparing two artworks by their words: the OCR strategy.

Two files are compared word by word where they are printed (`app.engine.detectors.document_text`): each file's words
are taken from its own text layer where it has one (a PDF with live text: exact, and instant) and read (Tesseract) only
where the two pictures differ where it has not. It runs on a pixel comparison's own alignment and compared area, so
both strategies judge the same two pictures, lined up the same way, and a word finding sits exactly where a pixel one
would. Leaflet mode shows both strategies' findings; Automatic mode keeps these ready, hidden, for its results page's
OCR toggle.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from difflib import SequenceMatcher

import numpy as np

from app.config import AppConfig
from app.engine.aggregation.aggregator import build_clusters
from app.engine.detectors import document_text
from app.engine.detectors.text_diff import OcrUnavailableError, OcrWord, ocr_available
from app.engine.isolation.boundary_mask import apply_mask
from app.engine.worker import ComparisonOutcome, in_compared_area, suspect_words_in_frame
from app.models.diff import DiffCandidate, DiffSubtype

logger = logging.getLogger(__name__)


def word_outcome(
    visual: ComparisonOutcome,
    original_layer: list[OcrWord] | None,
    suspect_layer: list[OcrWord] | None,
    config: AppConfig,
    language: str,
) -> ComparisonOutcome:
    """The words of one pair compared where they are printed (see the module's account), on the pixel comparison's own
    alignment of the pair (`visual`: the whole artwork by what is printed on it, then each panel on its own) and only
    inside the area it compared (`visual.compare_mask`). `original_layer`/`suspect_layer` are each file's own words
    (a PDF's text layer), None for a file that has none -- its words are then read where the two differ."""
    original_page, warped = visual.aligned_original, visual.aligned_suspect
    area = visual.compare_mask
    if area is not None and not area.all():
        # the leaflet alone: nothing printed outside it (marks, notes, a slug line) is read, or looked for
        original_page, warped = apply_mask(original_page, area), apply_mask(warped, area)
    transform = visual.alignment.transform
    suspect_in_frame = suspect_words_in_frame(suspect_layer, transform, visual.block_alignments) if suspect_layer else None
    if area is not None:
        original_layer = in_compared_area(original_layer, area)
        suspect_in_frame = in_compared_area(suspect_in_frame, area)
    heights = [w.bbox.height for w in (original_layer or []) + (suspect_in_frame or []) if w.upright is not False]
    type_px = float(np.median(heights)) if heights else document_text.type_height(original_page)
    warnings: list[str] = []
    candidates: list[DiffCandidate] = []
    try:
        if original_layer is not None and suspect_in_frame is not None:
            words_o, words_s = original_layer, suspect_in_frame  # both pages' own text: every word, exactly
        else:
            if not ocr_available():
                raise OcrUnavailableError(
                    "The text reader (Tesseract) is not available, so the text of a page with none of its own was not compared."
                )
            windows = document_text.difference_windows(original_page, warped, type_px)
            words_o, words_s = document_text.words_in_windows(
                windows, original_page, warped, original_layer, suspect_in_frame, config.detection, type_px
            )
        floor = config.detection.ocr_min_confidence
        candidates = document_text.compare_words(
            document_text.usable(words_o, floor),
            document_text.usable(words_s, floor),
            original_page,
            warped,
            config.detection,
            original_layer,
            suspect_in_frame,
            language,
        )
    except OcrUnavailableError as exc:
        logger.warning("leaflet page text not compared: %s", exc)
        warnings.append(str(exc))
    return ComparisonOutcome(
        alignment=visual.alignment,
        candidates=candidates,
        clusters=build_clusters(candidates, transform, config.clustering),
        aligned_original=visual.aligned_original,
        aligned_suspect=visual.aligned_suspect,
        warnings=warnings,
        detection_profile="exact",
        mode=visual.mode,
    )


_WORD = re.compile(r"^[^\W\d_]+(?:[-'’][^\W\d_]+)*$", re.UNICODE)  # letters only (a hyphen or apostrophe within)
_EDGE_MARKS = ".,;:!?()[]{}\"'‘’“”"
_SAME_WORD = 0.6  # two spellings this alike (letters in common, in order) are one word misspelt, not another word


def _misspelt(was: str, now: str) -> bool:
    """Whether a word is merely spelt another way: letters only on both sides (no number, dose or code), and still
    recognisably the same word -- 'from' as 'fom', 'MATRIX' as 'MATRX' -- not a different word put in its place."""
    a, b = was.strip(_EDGE_MARKS), now.strip(_EDGE_MARKS)
    if not a or not b or a == b or not _WORD.match(a) or not _WORD.match(b):
        return False
    return a.lower() == b.lower() or SequenceMatcher(None, a.lower(), b.lower()).ratio() >= _SAME_WORD


def spelling_changes(candidates: list[DiffCandidate]) -> list[DiffCandidate]:
    """Only the words spelt differently (Automatic mode's OCR findings): a word changed, in a line of the same number
    of words, where every word that differs is `_misspelt`. Text added or taken away, numbers, doses and codes changed,
    and one word replaced by another are left out. Each is described as the spelling it is."""
    kept = []
    for candidate in candidates:
        if candidate.subtype is not DiffSubtype.CHANGED or not candidate.was_text or not candidate.now_text:
            continue
        was, now = candidate.was_text.split(), candidate.now_text.split()
        if len(was) != len(now):
            continue
        pairs = [(a, b) for a, b in zip(was, now) if a != b]
        if pairs and all(_misspelt(a, b) for a, b in pairs):
            spelt = ", ".join(f"'{a}' spelt '{b}'" for a, b in pairs)
            kept.append(replace(candidate, detail=f"Spelling: {spelt}"))
    return kept
