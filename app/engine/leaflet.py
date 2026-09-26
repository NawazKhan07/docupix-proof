"""Comparing a multi-page document -- a leaflet, a pack insert -- page by page.

A leaflet's pages need no border or die-line: the whole page is the content. Two strategies, each on or off:

- TEXT (the default): what each page SAYS, word for word (`document_text`). A page's words are its own text where it has
  any (a live PDF: exact, instant) and are read from its picture otherwise (a closed leaflet, outlined to paths) -- read
  only where the two pages are printed differently, since the text cannot have changed where the print has not. A
  change of words is reported only where the ink differs too, so a misreading is never a finding; a paragraph that
  reflowed is the same words, not a change; and text in a script the reader has no language for is compared by its ink.
- VISUAL: Automatic mode's own pixel-by-pixel comparison of the page (every detector, clustering).

Pages are paired by what is actually on them -- their pictures (a thumbnail correlation), and their words wherever the
pictures leave it in doubt -- not by position, so an inserted or a missing page does not throw every page after it out
of step; an unmatched page is its own finding, that it is missing or added. All the findings come back as ONE list in ONE `ComparisonOutcome`: the pages are stacked into one tall picture
per file (a label at each page's own top) so a finding is shown on its own page, and the ordinary viewer -- zoom,
markers, merge -- works on it exactly as it does on a single page.

What is stacked is each pair AS ALIGNED: the suspect page warped onto the original page it was paired with, and both
trimmed to what is printed on them (a page set on a sheet with a wide white margin, beside one trimmed tight, is the
same page). So the two stacks are one picture in the same frame, page for page, and Flicker, Overlay and Grayscale
work on a leaflet exactly as on a carton. A page with no partner sits opposite a plain placeholder.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Callable

import cv2
import numpy as np

from app.config import AppConfig, LeafletConfig
from app.engine.aggregation.aggregator import build_clusters
from app.engine.alignment.aligner import align
from app.engine.alignment.homography import warp_to_reference
from app.engine.detectors import document_text
from app.engine.detectors.text_diff import OcrUnavailableError, OcrWord, ocr_available, read_pages, text_layer_words
from app.engine.file_loader import LoadedArtwork
from app.engine.ocr_locator import usable_language
from app.engine.worker import ComparisonOutcome, run_comparison
from app.logging_config import StageTimer
from app.models.diff import BoundingBox, DiffCandidate, DiffCluster, DiffSubtype, DiffType
from app.models.job import ComparisonMode, DetectorOptions

logger = logging.getLogger(__name__)

_PAGE_LABEL_COLOUR = (255, 255, 255)
_PAGE_GAP_COLOUR = (60, 64, 72)
_STACK_BG = (235, 235, 230)


class NoPagesComparedError(Exception):
    """Every page failed to compare (or one of the files has none)."""


# The visual strategy (`LeafletConfig.compare_visual`) is Automatic mode's own pixel-by-pixel comparison (which also
# reads and describes what a found difference says); the text strategy is `_text_outcome`.
_VISUAL_STRATEGY_OPTIONS = DetectorOptions(
    compare_color=True, compare_text=True, compare_graphics=True,
    decode_symbols=False, compare_braille_dots=False, normalize_scan=False,
)


def _merge_outcomes(primary: ComparisonOutcome, extra: ComparisonOutcome) -> ComparisonOutcome:
    """Both outcomes are of the SAME page pair, compared by a different strategy: their findings are combined
    into one (renumbered so ids stay unique), `primary`'s own aligned pictures and alignment stand for the pair."""
    clusters = list(primary.clusters)
    next_id = max((c.id for c in clusters), default=0) + 1
    for cluster in extra.clusters:
        cluster.id = next_id
        next_id += 1
        clusters.append(cluster)
    return replace(
        primary,
        clusters=clusters,
        candidates=primary.candidates + extra.candidates,
        warnings=primary.warnings + extra.warnings,
    )


_PAGE_THUMB_SIZE = (900, 660)  # every thumbnail is resized to this (width, height): same position means the same page
_BLANK_THUMB_STD = 4.0


_BULK_SHARE = 0.003  # a page's print is framed by where all but this share of its ink lies, either side, each way
_THUMB_BLUR_PX = 3.0
_THUMB_WORKING_PX = 2400  # a page is framed on a copy no bigger than this along its long edge


def _thumbnail(image: np.ndarray, long_edge_px: int) -> np.ndarray:
    """Where a page is printed, and how densely, as a small fixed-size picture: two versions of a page -- one with live
    text, one outlined; one on a bigger sheet -- have their print in the same places, and a different page does not,
    however alike the two pages' frames and columns (edges of small type at this size are only a texture: the same page
    of a pack insert correlated 0.24 by its edges, 0.94+ by its ink).

    Framed by the bulk of the print (`_bulk_box`), not its extremes: a crop mark in a sheet's corner is not the page."""
    if image.ndim == 3:
        channels = cv2.split(np.ascontiguousarray(image[:, :, :3]))
        darkest = cv2.min(cv2.min(channels[0], channels[1]), channels[2])
    else:
        darkest = image
    # (framed on a reduced copy first: a pack insert's page is 37 megapixels, and the frame needs nothing like that)
    scale = min(1.0, _THUMB_WORKING_PX / max(darkest.shape[:2]))
    if scale < 1.0:
        darkest = cv2.resize(darkest, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    x0, y0, x1, y1 = _bulk_box(darkest)
    fixed = cv2.resize(darkest[y0:y1, x0:x1], _PAGE_THUMB_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(255.0 - fixed.astype(np.float32), (0, 0), _THUMB_BLUR_PX)


def _bulk_box(darkest: np.ndarray) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) holding all but `_BULK_SHARE` of the page's ink at each side; the whole page if it has none."""
    ink = darkest < _CONTENT_LEVEL
    columns, rows = np.cumsum(ink.sum(axis=0)), np.cumsum(ink.sum(axis=1))
    total = int(columns[-1]) if len(columns) else 0
    if total == 0:
        return 0, 0, darkest.shape[1], darkest.shape[0]
    x0, x1 = int(np.searchsorted(columns, _BULK_SHARE * total)), int(np.searchsorted(columns, (1 - _BULK_SHARE) * total)) + 1
    y0, y1 = int(np.searchsorted(rows, _BULK_SHARE * total)), int(np.searchsorted(rows, (1 - _BULK_SHARE) * total)) + 1
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def _page_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """How alike two pages' `_thumbnail`s are (roughly 0..1): normalised correlation, position for position (no
    search -- the same page of two versions of a document is framed the same way)."""
    a_blank, b_blank = float(a.std()) < _BLANK_THUMB_STD, float(b.std()) < _BLANK_THUMB_STD
    if a_blank or b_blank:
        return 1.0 if a_blank and b_blank else 0.0  # both blank says nothing either way; one blank is not the other
    return float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0, 0])


_RELATIVE_MATCH_FLOOR = 0.12  # a leftover page pairs with its mutual best leftover page if it scores at least this...
_RELATIVE_MATCH_LEAD = 2.0  # ...and at least this many times anything else either page scores
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


def _parse_page_selection(spec: str, page_count: int) -> set[int] | None:
    """`spec` (1-indexed page numbers/ranges, e.g. "1,3-5"), as 0-indexed page indices -- None (every page) for
    a blank spec. A part that is not a number or range, or a page number out of range, is quietly skipped
    rather than failing the whole comparison over a typo in a field the reviewer may still be editing."""
    spec = spec.strip()
    if not spec:
        return None
    indices: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        start_text, _, end_text = part.partition("-")
        try:
            start = int(start_text)
            end = int(end_text) if end_text else start
        except ValueError:
            continue
        for page in range(min(start, end), max(start, end) + 1):
            if 1 <= page <= page_count:
                indices.add(page - 1)
    return indices


_CLEAR_PAGE_MATCH = 0.3  # two pages whose pictures correlate this well, each the other's best...
_CLEAR_PAGE_LEAD = 1.5  # ...by this many times anything else either scores, are the same page: their words are not needed


def _pair_pages(
    original_pages: list[np.ndarray],
    suspect_pages: list[np.ndarray],
    config: LeafletConfig,
    language: str,
    original_words: list[frozenset[str] | None] | None = None,
    suspect_words: list[frozenset[str] | None] | None = None,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """(pairs, unmatched original page indices, unmatched suspect page indices), each page paired with whichever page
    of the other file it actually is, greedily, strongest pair first, rather than assumed to be at the same position: a
    page inserted or removed must not throw every page after it out of step.

    A page whose picture is clearly one page of the other file's (each the other's best, well ahead of anything else) is
    paired so at once. The rest are paired by their words where enough were found -- `original_words`/`suspect_words`,
    a page's own text, where the caller has it; read from the page otherwise (only for the pages that need it: reading a
    whole page is slow) -- and by their pictures where not."""
    original_thumbs = [_thumbnail(page, config.thumbnail_long_edge_px) for page in original_pages]
    suspect_thumbs = [_thumbnail(page, config.thumbnail_long_edge_px) for page in suspect_pages]
    pictures = {
        (i, j): _page_similarity(original_thumbs[i], suspect_thumbs[j])
        for i in range(len(original_pages))
        for j in range(len(suspect_pages))
    }
    paired_o: set[int] = set()
    paired_s: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for i in range(len(original_pages)):
        if not suspect_pages:
            break
        j = max(range(len(suspect_pages)), key=lambda k: pictures[(i, k)])
        best = pictures[(i, j)]
        others = [pictures[(i, k)] for k in range(len(suspect_pages)) if k != j]
        others += [pictures[(m, j)] for m in range(len(original_pages)) if m != i]
        if best >= _CLEAR_PAGE_MATCH and all(best >= _CLEAR_PAGE_LEAD * max(other, 1e-6) for other in others):
            pairs.append((i, j))
            paired_o.add(i)
            paired_s.add(j)

    words: dict[tuple[int, int], frozenset[str] | None] = {}
    can_read: list[bool] = []

    def words_of(side: int, index: int) -> frozenset[str] | None:
        if (side, index) not in words:
            given = (original_words, suspect_words)[side]
            found = given[index] if given is not None else None
            if found is None:
                if not can_read:
                    can_read.append(ocr_available())
                found = _word_set((original_pages, suspect_pages)[side][index], language) if can_read[0] else None
            words[(side, index)] = found
        return words[(side, index)]

    def score(i: int, j: int) -> float:
        a, b = words_of(0, i), words_of(1, j)
        if a is not None and b is not None:
            return _word_similarity(a, b)
        return pictures[(i, j)]

    rest_o = [i for i in range(len(original_pages)) if i not in paired_o]
    rest_s = [j for j in range(len(suspect_pages)) if j not in paired_s]
    scores = [(score(i, j), i, j) for i in rest_o for j in rest_s]
    scores.sort(key=lambda item: -item[0])
    for value, i, j in scores:
        if value < config.min_page_match_score or i in paired_o or j in paired_s:
            continue
        pairs.append((i, j))
        paired_o.add(i)
        paired_s.add(j)
    # A page left over whose best match is a page left over in the other file -- and that page's best is it -- is that
    # page, even where neither reading nor picture is alike enough by the absolute bar: a page in a script the reader
    # cannot read, laid out a little differently, still matches itself far better than anything else.
    by_pair = {(i, j): value for value, i, j in scores}
    for i in rest_o:
        if i in paired_o:
            continue
        free = [j for j in rest_s if j not in paired_s]
        if not free:
            break
        j = max(free, key=lambda k: by_pair[(i, k)])
        best = by_pair[(i, j)]
        runner_up = max(
            [by_pair[(i, k)] for k in rest_s if k != j] + [by_pair[(m, j)] for m in rest_o if m != i] + [0.0]
        )
        mutual = all(by_pair[(m, j)] <= best for m in rest_o if m not in paired_o)
        if mutual and best >= _RELATIVE_MATCH_FLOOR and best >= _RELATIVE_MATCH_LEAD * max(runner_up, 1e-6):
            pairs.append((i, j))
            paired_o.add(i)
            paired_s.add(j)
    pairs.sort(key=lambda pair: pair[0])
    unmatched_o = [i for i in range(len(original_pages)) if i not in paired_o]
    unmatched_s = [j for j in range(len(suspect_pages)) if j not in paired_s]
    return pairs, unmatched_o, unmatched_s


def _layer_word_set(words: list[OcrWord] | None) -> frozenset[str] | None:
    """A page's own text as `_word_set` gives a reading of a page: its distinct words, None if too few."""
    if not words:
        return None
    found = frozenset(match.group(0).lower() for word in words for match in _WORD_PATTERN.finditer(word.text))
    return found if len(found) >= _MIN_WORDS_FOR_PAIRING else None


def _page_words(artwork, index: int, page: np.ndarray) -> list[OcrWord] | None:
    """A PDF page's own words in its picture's pixels (None for a page with none of its own, or a picture)."""
    if getattr(artwork, "kind", None) != "pdf":
        return None
    return text_layer_words(artwork.pdf_page(index), artwork.working_dpi(index), page.shape[:2])


def _text_outcome(
    original_page: np.ndarray,
    suspect_page: np.ndarray,
    original_layer: list[OcrWord] | None,
    suspect_layer: list[OcrWord] | None,
    config: AppConfig,
    language: str,
) -> ComparisonOutcome:
    """The text strategy on one pair of pages (see the module's account): the suspect page aligned onto the original,
    and their words compared where they are printed."""
    alignment = align(original_page, suspect_page, config.alignment)
    warped = warp_to_reference(suspect_page, alignment.transform, original_page.shape[:2])
    suspect_in_frame = document_text.in_frame(suspect_layer, alignment.transform) if suspect_layer else None
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
        alignment=alignment,
        candidates=candidates,
        clusters=build_clusters(candidates, alignment.transform, config.clustering),
        aligned_original=original_page,
        aligned_suspect=warped,
        warnings=warnings,
        detection_profile="exact",
        mode=ComparisonMode.LEAFLET,
    )


_CONTENT_LEVEL = 235  # a pixel darker than this in its darkest channel is printed (paper and a light margin tint are not)
_CONTENT_MARGIN_SHARE = 0.015  # a trimmed page keeps this share of its long edge as a margin round what is printed
_PLACEHOLDER_COLOUR = (225, 227, 232)


def _content_box(*pictures: np.ndarray) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1): what is printed on any of `pictures` (all the same size), with a small margin; the whole
    picture if nothing is."""
    height, width = pictures[0].shape[:2]
    printed = np.zeros((height, width), bool)
    for picture in pictures:
        channels = cv2.split(np.ascontiguousarray(picture[:, :, :3])) if picture.ndim == 3 else [picture]
        darkest = channels[0]
        for channel in channels[1:]:
            darkest = cv2.min(darkest, channel)
        printed |= darkest < _CONTENT_LEVEL
    if not printed.any():
        return 0, 0, width, height
    x, y, w, h = cv2.boundingRect(printed.astype(np.uint8))
    margin = round(_CONTENT_MARGIN_SHARE * max(height, width))
    return max(0, x - margin), max(0, y - margin), min(width, x + w + margin), min(height, y + h + margin)


def _placeholder(height: int, width: int, text: str) -> np.ndarray:
    picture = np.full((height, width, 3), _PLACEHOLDER_COLOUR, np.uint8)
    scale = max(0.6, min(height, width) / 900)
    cv2.putText(picture, text, (max(10, width // 20), height // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (90, 96, 108),
                max(1, round(scale * 2)), cv2.LINE_AA)
    return picture


def _stack_rows(rows: list[tuple[np.ndarray, np.ndarray, str]], gap_px: int) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Each row a (original, suspect, label) of the SAME size, stacked into one picture per file -- every row padded to
    the widest, a labelled gap between rows. Returns both stacks and each row's (x, y) offset (always x = 0)."""
    width = max(original.shape[1] for original, _suspect, _label in rows)
    parts: tuple[list[np.ndarray], list[np.ndarray]] = ([], [])
    offsets: list[tuple[int, int]] = []
    y = 0
    for number, (original, suspect, label) in enumerate(rows):
        if number:
            gap = np.full((gap_px, width, 3), _PAGE_GAP_COLOUR, np.uint8)
            cv2.putText(gap, label, (10, gap_px - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _PAGE_LABEL_COLOUR, 1, cv2.LINE_AA)
            parts[0].append(gap)
            parts[1].append(gap)
            y += gap_px
        offsets.append((0, y))
        for stack, picture in zip(parts, (original, suspect)):
            pad = width - picture.shape[1]
            stack.append(picture if pad == 0 else np.pad(picture, ((0, 0), (0, pad), (0, 0)), constant_values=_STACK_BG[0]))
        y += original.shape[0]
    return np.vstack(parts[0]), np.vstack(parts[1]), offsets


def _moved(box: BoundingBox, dx: float, dy: float) -> BoundingBox:
    return BoundingBox(box.x0 + dx, box.y0 + dy, box.x1 + dx, box.y1 + dy)




def _whole_page_finding(kind: DiffSubtype, page_index: int, offset: int, size: tuple[int, int]) -> DiffCluster:
    """A whole page found in one file with nothing in the other that looks enough like it. `kind` is REMOVED for an
    original page with no match in the suspect, ADDED for a suspect page with no match in the original."""
    height, width = size
    box = BoundingBox(0.0, float(offset), float(width), float(offset + height))
    word = "missing from" if kind is DiffSubtype.REMOVED else "added in"
    detail = f"Page {page_index + 1} {word} the suspect (no page of the other file looks enough like it)"
    member = DiffCandidate(bbox=box, type=DiffType.GRAPHICS, subtype=kind, confidence=1.0, detail=detail)
    return DiffCluster(
        id=0,
        bbox_in_original=box,
        bbox_in_suspect=box,
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

    # each PDF page's own words (none for an outlined page, or a picture): exact, and instant beside reading it
    original_layers = [_page_words(original, i, page) for i, page in enumerate(original_pages)]
    suspect_layers = [_page_words(suspect, i, page) for i, page in enumerate(suspect_pages)]

    say("Matching the pages…")
    with StageTimer(logger, "leaflet_pairing"):
        pairs, unmatched_o, unmatched_s = _pair_pages(
            original_pages,
            suspect_pages,
            leaflet_config,
            language,
            [_layer_word_set(words) for words in original_layers],
            [_layer_word_set(words) for words in suspect_layers],
        )

    selection = _parse_page_selection(leaflet_config.compare_pages, len(original_pages))
    if selection is not None:
        # Only the selected original pages are being reviewed: an unselected original, or a suspect page that
        # has nothing to do with the selection, is left out entirely -- not compared, and not reported as
        # missing or added either.
        pairs = [(o, s) for o, s in pairs if o in selection]
        unmatched_o = [o for o in unmatched_o if o in selection]
        unmatched_s = []

    say("Comparing the pages…")
    results: dict[int, ComparisonOutcome] = {}

    def compare_one(pair: tuple[int, int]) -> tuple[int, ComparisonOutcome | Exception]:
        o, s = pair
        original_page, suspect_page = original_pages[o], suspect_pages[s]
        try:
            outcome: ComparisonOutcome | None = None
            if leaflet_config.compare_text:
                outcome = _text_outcome(original_page, suspect_page, original_layers[o], suspect_layers[s], page_config, language)
            if leaflet_config.compare_visual:
                visual_outcome = run_comparison(
                    original_page,
                    suspect_page,
                    ComparisonMode.LEAFLET,
                    page_config,
                    options=_VISUAL_STRATEGY_OPTIONS,
                    original_words=original_layers[o],
                    suspect_words=suspect_layers[s],
                )
                outcome = _merge_outcomes(outcome, visual_outcome) if outcome is not None else visual_outcome
            if outcome is None:
                raise ValueError("Neither of Leaflet's comparison strategies is switched on.")
            return o, outcome
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

    # One row per page reviewed, in the original's page order (a suspect page with no partner at the end): each pair
    # as aligned (the suspect warped onto its original page), both trimmed to what is printed on them.
    rows: list[tuple[np.ndarray, np.ndarray, str]] = []
    row_clusters: list[list[DiffCluster]] = []
    row_crops: list[tuple[int, int]] = []
    all_clusters: list[DiffCluster] = []
    reference_alignment = None
    profiles: list[str] = []
    partner = dict(pairs)
    for o in range(len(original_pages)):
        if o in partner and o in results:
            s_index, outcome = partner[o], results[o]
            if reference_alignment is None:
                reference_alignment = outcome.alignment
            profiles.append(outcome.detection_profile)
            x0, y0, x1, y1 = _content_box(outcome.aligned_original, outcome.aligned_suspect)
            label = f"Page {o + 1}" if o == s_index else f"Original p.{o + 1} / Suspect p.{s_index + 1}"
            rows.append((outcome.aligned_original[y0:y1, x0:x1], outcome.aligned_suspect[y0:y1, x0:x1], label))
            row_crops.append((x0, y0))
            for cluster in outcome.clusters:
                cluster.detail = f"{label}: {cluster.detail}"
            row_clusters.append(list(outcome.clusters))
            warnings.extend(f"Page {o + 1}: {warning}" for warning in outcome.warnings)
        elif o in unmatched_o:
            page = original_pages[o]
            x0, y0, x1, y1 = _content_box(page)
            trimmed = page[y0:y1, x0:x1]
            rows.append((trimmed, _placeholder(*trimmed.shape[:2], "No page of the suspect matches this one"), f"Original p.{o + 1}"))
            row_crops.append((0, 0))
            row_clusters.append([_whole_page_finding(DiffSubtype.REMOVED, o, 0, trimmed.shape[:2])])
    for s_index in unmatched_s:
        page = suspect_pages[s_index]
        x0, y0, x1, y1 = _content_box(page)
        trimmed = page[y0:y1, x0:x1]
        rows.append((_placeholder(*trimmed.shape[:2], "No page of the original matches this one"), trimmed, f"Suspect p.{s_index + 1}"))
        row_crops.append((0, 0))
        row_clusters.append([_whole_page_finding(DiffSubtype.ADDED, s_index, 0, trimmed.shape[:2])])

    if not rows:
        raise NoPagesComparedError("No page could be compared (every pairing failed, or none paired at all).")
    original_stack, suspect_stack, offsets = _stack_rows(rows, leaflet_config.page_gap_px)
    for (crop_x, crop_y), (_x, row_y), clusters in zip(row_crops, offsets, row_clusters):
        for cluster in clusters:
            # the two stacks are one frame now: a finding is at the same place in both
            box = _moved(cluster.bbox_in_original, -crop_x, row_y - crop_y)
            cluster.bbox_in_original = box
            cluster.bbox_in_suspect = box
            all_clusters.append(cluster)

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
    if selection is not None:
        notes.append(f"Restricted to original page(s) {leaflet_config.compare_pages}: the rest were not reviewed.")

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
        graphics_compared=leaflet_config.compare_visual,
        text_compared=leaflet_config.compare_text,
        compared_pages=leaflet_config.compare_pages,
        notes=notes,
    )


def _with_language(config: AppConfig, language: str) -> AppConfig:
    return replace(config, detection=replace(config.detection, ocr_language=language))
