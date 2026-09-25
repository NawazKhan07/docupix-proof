"""Comparing a multi-page document -- a leaflet, a pack insert -- page by page.

A leaflet's pages need no border or die-line: the whole page is the content, so each page is compared exactly as
Automatic mode compares a single pair (alignment, every detector, clustering), whatever made it -- a live PDF with
its own text, one outlined to pure vector paths with none, or a picture of it. Text is only ever READ to describe a
difference the pixels already found (never to decide one), so this works the same whether a page is open or closed,
and in any language: a script nobody has installed a Tesseract language for is still compared correctly, just not
captioned with its words.

Pages are paired by what is actually on them (a thumbnail correlation), not by position, so an inserted or a missing
page does not throw every page after it out of step; an unmatched page is its own finding, that it is missing or
added. All the findings come back as ONE list in ONE `ComparisonOutcome`: each file's pages are stacked into one tall
picture (a label at each page's own top), so a finding is shown on its own page, and the ordinary viewer -- zoom,
markers, merge -- works on it exactly as it does on a single page.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import cv2
import numpy as np

from app.config import AppConfig, LeafletConfig
from app.engine.detectors.text_diff import ocr_available, read_pages
from app.engine.file_loader import LoadedArtwork
from app.engine.ocr_locator import usable_language
from app.engine.worker import ComparisonOutcome, run_comparison
from app.logging_config import StageTimer
from app.models.diff import BoundingBox, DiffCandidate, DiffCluster, DiffSubtype, DiffType
from app.models.job import ComparisonMode

logger = logging.getLogger(__name__)

_PAGE_LABEL_COLOUR = (255, 255, 255)
_PAGE_GAP_COLOUR = (60, 64, 72)
_STACK_BG = (235, 235, 230)


class NoPagesComparedError(Exception):
    """Every page failed to compare (or one of the files has none)."""


_PAGE_THUMB_SIZE = (900, 660)  # every thumbnail is resized to this (width, height): same position means the same page
_BLANK_THUMB_STD = 4.0


def _thumbnail(image: np.ndarray, long_edge_px: int) -> np.ndarray:
    """A page's edges (what is printed, not its shading or colour), as a small, fixed-size picture: two pages of a
    document are laid out alike, so the SAME page of each -- resized to the same canvas -- has its content in about
    the same place, and a page with different content does not, however similar the two pages' layouts are. This is
    the fallback signal, for a page too closed or too graphical to be told apart by its words (`_word_set`)."""
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    fixed = cv2.resize(grey, _PAGE_THUMB_SIZE, interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(cv2.GaussianBlur(fixed, (0, 0), 1.0), 40, 120)
    return cv2.GaussianBlur(edges.astype(np.float32), (0, 0), 1.0)


def _page_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """How alike two pages' `_thumbnail`s are (roughly 0..1): normalised correlation, position for position (no
    search -- the same page of two versions of a document is framed the same way)."""
    a_blank, b_blank = float(a.std()) < _BLANK_THUMB_STD, float(b.std()) < _BLANK_THUMB_STD
    if a_blank or b_blank:
        return 1.0 if a_blank and b_blank else 0.0  # both blank says nothing either way; one blank is not the other
    return float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0, 0])


_MIN_WORDS_FOR_PAIRING = 8  # a page read as fewer distinct words than this is paired by its picture instead
_WORD_READ_MIN_CONFIDENCE = 20.0  # a word read this unsurely is not counted (pairing tolerates some noise either way)
_WORD_PATTERN = re.compile(r"[^\W\d_]{3,}", re.UNICODE)  # letters only, 3 or more: short OCR junk is not a word


def _word_set(image: np.ndarray, language: str) -> frozenset[str] | None:
    """The distinct words read on a page, lower-cased -- None where too few were read to mean anything (a page in a
    script this language cannot read, or one that is mostly pictures): its pairing falls back to `_page_similarity`.
    What a page's WORDS are is far more telling of which page it is than its layout (many of a leaflet's pages share
    the same frame and column widths), so this decides pairing wherever it can.

    Read at its own full size, in strips side by side (`text_diff.read_pages`): a page spread wide and short (an
    unfolded leaflet, many narrow columns) loses its print to a single reduced copy long before the copy is small
    enough to read quickly, so nothing here shrinks it first.
    """
    try:
        (page_words,) = read_pages([image], _WORD_READ_MIN_CONFIDENCE, language)
    except Exception as exc:  # noqa: BLE001 -- a reading that fails says nothing; the picture is still there to pair by
        logger.info("page reading for pairing failed: %s", exc)
        return None
    words = frozenset(
        match.group(0).lower() for _line, word in page_words for match in _WORD_PATTERN.finditer(word.text)
    )
    return words if len(words) >= _MIN_WORDS_FOR_PAIRING else None


def _word_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _pair_pages(
    original_pages: list[np.ndarray], suspect_pages: list[np.ndarray], config: LeafletConfig, language: str
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """(pairs, unmatched original page indices, unmatched suspect page indices), each page paired with whichever page
    of the other file it actually is -- by its words where enough were read, by its picture otherwise -- greedily,
    strongest pair first, rather than assumed to be at the same position: a page inserted or removed must not throw
    every page after it out of step."""
    original_thumbs = [_thumbnail(page, config.thumbnail_long_edge_px) for page in original_pages]
    suspect_thumbs = [_thumbnail(page, config.thumbnail_long_edge_px) for page in suspect_pages]
    if ocr_available():
        original_words = [_word_set(page, language) for page in original_pages]
        suspect_words = [_word_set(page, language) for page in suspect_pages]
    else:
        original_words = [None] * len(original_pages)
        suspect_words = [None] * len(suspect_pages)

    def score(i: int, j: int) -> float:
        a, b = original_words[i], suspect_words[j]
        if a is not None and b is not None:
            return _word_similarity(a, b)
        return _page_similarity(original_thumbs[i], suspect_thumbs[j])

    scores = [(score(i, j), i, j) for i in range(len(original_pages)) for j in range(len(suspect_pages))]
    scores.sort(key=lambda item: -item[0])
    paired_o: set[int] = set()
    paired_s: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for value, i, j in scores:
        if value < config.min_page_match_score or i in paired_o or j in paired_s:
            continue
        pairs.append((i, j))
        paired_o.add(i)
        paired_s.add(j)
    pairs.sort(key=lambda pair: pair[0])
    unmatched_o = [i for i in range(len(original_pages)) if i not in paired_o]
    unmatched_s = [j for j in range(len(suspect_pages)) if j not in paired_s]
    return pairs, unmatched_o, unmatched_s


def _stack_pages(pages: list[np.ndarray], gap_px: int, first_label: str) -> tuple[np.ndarray, list[int]]:
    """`pages` stacked one above another, each padded to the widest one's width, with a labelled gap between them.
    Returns the stacked picture and each page's own top (its y-offset in it)."""
    width = max(page.shape[1] for page in pages)
    offsets: list[int] = []
    rows: list[np.ndarray] = []
    y = 0
    for number, page in enumerate(pages, start=1):
        if number > 1:
            gap = np.full((gap_px, width, 3), _PAGE_GAP_COLOUR, np.uint8)
            cv2.putText(
                gap, f"{first_label} {number}", (10, gap_px - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _PAGE_LABEL_COLOUR, 1, cv2.LINE_AA
            )
            rows.append(gap)
            y += gap_px
        offsets.append(y)
        padded = page if page.shape[1] == width else np.pad(page, ((0, 0), (0, width - page.shape[1]), (0, 0)), constant_values=235)
        rows.append(padded)
        y += page.shape[0]
    return np.vstack(rows), offsets


_NOTHING_TO_POINT_AT_PX = 40  # a whole page missing has nowhere to point in the file that lacks it: a small marker, top left


def _whole_page_finding(kind: DiffSubtype, page_index: int, offset: int, size: tuple[int, int]) -> DiffCluster:
    """A whole page found in one file with nothing in the other that looks enough like it. `kind` is REMOVED for an
    original page with no match in the suspect, ADDED for a suspect page with no match in the original."""
    height, width = size
    box = BoundingBox(0.0, float(offset), float(width), float(offset + height))
    nowhere = BoundingBox(0.0, 0.0, float(_NOTHING_TO_POINT_AT_PX), float(_NOTHING_TO_POINT_AT_PX))
    word = "missing from" if kind is DiffSubtype.REMOVED else "added in"
    detail = f"Page {page_index + 1} {word} the suspect (no page of the other file looks enough like it)"
    member = DiffCandidate(bbox=box, type=DiffType.GRAPHICS, subtype=kind, confidence=1.0, detail=detail)
    return DiffCluster(
        id=0,
        bbox_in_original=box if kind is DiffSubtype.REMOVED else nowhere,
        bbox_in_suspect=nowhere if kind is DiffSubtype.REMOVED else box,
        type=DiffType.GRAPHICS,
        members=[member],
        detail=detail,
    )


def compare_leaflet(
    original: LoadedArtwork, suspect: LoadedArtwork, config: AppConfig, progress: Callable[[str], None] | None = None
) -> ComparisonOutcome:
    """Compare every page of `original` against every page of `suspect` it is paired with (see the module's account)."""
    say = progress or (lambda _text: None)
    leaflet_config = config.leaflet
    if original.page_count == 0 or suspect.page_count == 0:
        raise NoPagesComparedError("One of the files has no pages.")

    say("Rendering the pages…")
    with StageTimer(logger, "leaflet_render"):
        original_pages = [original.render(page=i) for i in range(original.page_count)]
        suspect_pages = [suspect.render(page=i) for i in range(suspect.page_count)]

    language, language_ok = usable_language(leaflet_config.language)
    page_config = config if language_ok else _with_language(config, language)
    warnings: list[str] = []
    if not language_ok:
        warnings.append(
            f"'{leaflet_config.language}' is not installed for the text reader, so '{language}' was used instead. "
            "Findings are still located and flagged correctly; only their words may not be read where this is the "
            "wrong language for them."
        )

    say("Matching the pages…")
    pairs, unmatched_o, unmatched_s = _pair_pages(original_pages, suspect_pages, leaflet_config, language)

    say("Comparing the pages…")
    results: dict[int, ComparisonOutcome] = {}

    def compare_one(pair: tuple[int, int]) -> tuple[int, ComparisonOutcome | Exception]:
        o, s = pair
        try:
            return o, run_comparison(original_pages[o], suspect_pages[s], ComparisonMode.LEAFLET, page_config)
        except Exception as exc:  # noqa: BLE001 -- one unreadable page must not sink the rest
            return o, exc

    with StageTimer(logger, "leaflet_pages"), ThreadPoolExecutor(max_workers=max(1, leaflet_config.max_page_workers)) as pool:
        for o, outcome in pool.map(compare_one, pairs):
            if isinstance(outcome, Exception):
                logger.warning("leaflet page %d could not be compared: %s", o + 1, outcome)
                warnings.append(f"Page {o + 1}: could not be compared ({outcome}).")
                continue
            results[o] = outcome

    if not results and not unmatched_o and not unmatched_s:
        raise NoPagesComparedError("No page of either file could be compared.")

    original_stack, original_offsets = _stack_pages(original_pages, leaflet_config.page_gap_px, "Original p.")
    suspect_stack, suspect_offsets = _stack_pages(suspect_pages, leaflet_config.page_gap_px, "Suspect p.")

    all_clusters: list[DiffCluster] = []
    reference_alignment = None
    profiles: list[str] = []
    for o, s in pairs:
        outcome = results.get(o)
        if outcome is None:
            continue
        if reference_alignment is None:
            reference_alignment = outcome.alignment
        profiles.append(outcome.detection_profile)
        oy, sy = original_offsets[o], suspect_offsets[s]
        page_label = f"Page {o + 1}" if o == s else f"Original p.{o + 1} / Suspect p.{s + 1}"
        for cluster in outcome.clusters:
            ob, sb = cluster.bbox_in_original, cluster.bbox_in_suspect
            cluster.bbox_in_original = BoundingBox(ob.x0, ob.y0 + oy, ob.x1, ob.y1 + oy)
            cluster.bbox_in_suspect = BoundingBox(sb.x0, sb.y0 + sy, sb.x1, sb.y1 + sy)
            cluster.detail = f"{page_label}: {cluster.detail}"
            all_clusters.append(cluster)
        warnings.extend(f"Page {o + 1}: {warning}" for warning in outcome.warnings)

    for o in unmatched_o:
        all_clusters.append(_whole_page_finding(DiffSubtype.REMOVED, o, original_offsets[o], original_pages[o].shape[:2]))
    for s in unmatched_s:
        all_clusters.append(_whole_page_finding(DiffSubtype.ADDED, s, suspect_offsets[s], suspect_pages[s].shape[:2]))

    if reference_alignment is None:
        raise NoPagesComparedError("No page could be compared (every pairing failed, or none paired at all).")

    all_clusters.sort(key=lambda cluster: (cluster.bbox_in_original.y0, cluster.bbox_in_original.x0))
    for number, cluster in enumerate(all_clusters, start=1):
        cluster.id = number

    notes = [
        f"Compared {len(pairs)} page{'s' if len(pairs) != 1 else ''} "
        f"({original.page_count} in the original, {suspect.page_count} in the suspect)."
    ]
    if unmatched_o or unmatched_s:
        notes.append(
            f"{len(unmatched_o)} original page(s) and {len(unmatched_s)} suspect page(s) had no match in the other "
            "file and are reported as whole pages added or missing."
        )

    return ComparisonOutcome(
        alignment=reference_alignment,
        candidates=[],
        clusters=all_clusters,
        aligned_original=original_stack,
        aligned_suspect=suspect_stack,
        warnings=warnings,
        detection_profile="exact" if profiles and all(p == "exact" for p in profiles) else "tolerant",
        used_change_regions=False,
        mode=ComparisonMode.LEAFLET,
        compared_outline=None,
        graphics_compared=True,
        notes=notes,
    )


def _with_language(config: AppConfig, language: str):
    from dataclasses import replace

    return replace(config, detection=replace(config.detection, ocr_language=language))
