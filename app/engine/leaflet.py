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
from app.engine import prepared
from app.engine.detectors.text_diff import OcrWord, ocr_available, read_pages, shown_words, text_layer_words
from app.engine.file_loader import LoadedArtwork
from app.engine import language_detect
from app.engine.language_detect import detect_languages
from app.engine.ocr_locator import usable_language
from app.models.languages import AUTO
from app.engine.isolation.auto_area import find_artwork
from app.engine.isolation.block_grid import stroked_line_map
from app.engine.isolation.boundary_mask import apply_mask, boundary_mask
from app.engine.isolation.die_lines import vector_strokes
from app.engine.word_strategy import word_outcome
from app.engine.worker import ComparisonOutcome, run_comparison
from app.logging_config import StageTimer
from app.models.diff import BoundingBox, DiffCandidate, DiffCluster, DiffSubtype, DiffType, SOURCE_PAGE, renumbered, visible_by_source
from app.models.isolation import IsolationGeometry
from app.models.job import ComparisonMode, DetectorOptions

logger = logging.getLogger(__name__)

_PAGE_LABEL_COLOUR = (255, 255, 255)
_PAGE_GAP_COLOUR = (60, 64, 72)
_STACK_BG = (235, 235, 230)


class NoPagesComparedError(Exception):
    """Every page failed to compare (or one of the files has none)."""


# The visual strategy (`LeafletConfig.compare_visual`) is Automatic mode's own pixel-by-pixel comparison (which also
# reads and describes what a found difference says); the text strategy is `word_strategy.word_outcome`.
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


def _page(artwork, index: int) -> np.ndarray:
    """One page of a file as a picture."""
    return artwork.render(page=index)  # (not kept: a pack insert's pages are ~100 MB each; its alignments are)


def _page_words(artwork, index: int, page: np.ndarray) -> list[OcrWord] | None:
    """A PDF page's own words in its picture's pixels (None for a page with none of its own, or a picture)."""
    if getattr(artwork, "kind", None) != "pdf":
        return None
    return shown_words(text_layer_words(artwork.pdf_page(index), artwork.working_dpi(index), page.shape[:2]), page)


def _pdf_lines_and_strokes(artwork, index: int, page: np.ndarray) -> tuple:
    """A PDF page's own stroked lines (its folds, for dividing the leaflet into panels) and stroked paths with their
    colours (its outline and fold lines, drawn out of what is compared); (None, None) for a picture."""
    if getattr(artwork, "kind", None) != "pdf":
        return None, None
    page_pdf, dpi = artwork.pdf_page(index), artwork.working_dpi(index)
    return stroked_line_map(page_pdf, dpi, page.shape[:2]), vector_strokes(page_pdf, dpi)


_LEAFLET_PRINT_SHARE = 0.85  # an area found on a page is its leaflet only if at least this share of the page's print is in it


def _printed(picture: np.ndarray) -> np.ndarray:
    """Where `picture` is printed on: darker than paper (or a light margin tint) in its darkest channel."""
    if picture.ndim == 2:
        return picture < _CONTENT_LEVEL
    channels = cv2.split(np.ascontiguousarray(picture[:, :, :3]))
    darkest = channels[0]
    for channel in channels[1:]:
        darkest = cv2.min(darkest, channel)
    return darkest < _CONTENT_LEVEL


def _holds_the_print(page: np.ndarray, geometry: IsolationGeometry | None) -> bool:
    """Whether `geometry` is the leaflet on `page` rather than one box printed on it (a table, a panel, a logo's frame):
    what lies outside a leaflet is only its marks -- crop and fold marks, colour bars, a slug line -- a small share of
    the page's print."""
    if geometry is None:
        return True
    inside = boundary_mask(page.shape[:2], geometry)
    if inside is None:
        return True
    printed = _printed(page)
    total = int(np.count_nonzero(printed))
    return total == 0 or np.count_nonzero(printed & inside) >= _LEAFLET_PRINT_SHARE * total


_AREA_WORKING_PX = 3000  # a page's leaflet is looked for on a copy no bigger than this along its long edge


def _reduced(page: np.ndarray) -> tuple[np.ndarray, float]:
    """`page` no bigger than `_AREA_WORKING_PX` along its long edge, and the scale it was reduced by (1 if it was not)."""
    scale = min(1.0, _AREA_WORKING_PX / max(page.shape[:2]))
    if scale >= 1.0:
        return page, 1.0
    size = (max(1, round(page.shape[1] * scale)), max(1, round(page.shape[0] * scale)))
    return cv2.resize(page, size, interpolation=cv2.INTER_AREA), scale


def _leaflet_areas(
    original, suspect, o: int, s: int, original_page: np.ndarray, suspect_page: np.ndarray, config: AppConfig
) -> tuple[IsolationGeometry | None, IsolationGeometry | None]:
    """The leaflet on each page of a pair -- found the way a carton's artwork is found on its sheet (from a PDF's own
    lines where it has them), so its marks and whatever else is on the sheet are not compared. (None, None) where no
    area holding the leaflet's print is found: the whole page is compared, as it is."""
    if not config.detection.auto_artwork:
        return None, None
    is_pdf = (getattr(original, "kind", None) == "pdf", getattr(suspect, "kind", None) == "pdf")

    def found() -> tuple[IsolationGeometry | None, IsolationGeometry | None]:
        # Found on a reduced copy of each page: the area is kept in shares of the page (any size of it), and it only
        # says what is compared -- a leaflet page is 40-100 MP, where finding it at full size takes many seconds.
        (small_original, original_scale), (small_suspect, suspect_scale) = _reduced(original_page), _reduced(suspect_page)
        try:
            area = find_artwork(
                small_original,
                small_suspect,
                config.subject,
                original.pdf_page(o) if is_pdf[0] else None,
                suspect.pdf_page(s) if is_pdf[1] else None,
                original.working_dpi(o) * original_scale if is_pdf[0] else None,
                suspect.working_dpi(s) * suspect_scale if is_pdf[1] else None,
            )
        except Exception:  # noqa: BLE001 -- finding the leaflet is a help, never a reason for a page to fail
            logger.exception("leaflet area not found")
            return None, None
        return area.original, area.suspect

    paths = (getattr(original, "path", None), getattr(suspect, "path", None))
    try:
        key = (prepared.file_key(paths[0]), o, prepared.file_key(paths[1]), s, prepared.settings_key(config.subject))
    except (OSError, TypeError):
        key = (prepared.digest(original_page), prepared.digest(suspect_page), prepared.settings_key(config.subject))
    original_area, suspect_area = prepared.remembered("leaflet_area", key, found)
    if not (_holds_the_print(original_page, original_area) and _holds_the_print(suspect_page, suspect_area)):
        return None, None
    return original_area, suspect_area


_CONTENT_LEVEL = 235  # a pixel darker than this in its darkest channel is printed (paper and a light margin tint are not)
_CONTENT_MARGIN_SHARE = 0.015  # a trimmed page keeps this share of its long edge as a margin round what is printed
_PLACEHOLDER_COLOUR = (225, 227, 232)


def _content_box(*pictures: np.ndarray) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1): what is printed on any of `pictures` (all the same size), with a small margin; the whole
    picture if nothing is."""
    height, width = pictures[0].shape[:2]
    printed = np.zeros((height, width), bool)
    for picture in pictures:
        printed |= _printed(picture)
    if not printed.any():
        return 0, 0, width, height
    x, y, w, h = cv2.boundingRect(printed.astype(np.uint8))
    margin = round(_CONTENT_MARGIN_SHARE * max(height, width))
    return max(0, x - margin), max(0, y - margin), min(width, x + w + margin), min(height, y + h + margin)


def _leaflet_row(outcome: ComparisonOutcome) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """A compared pair of pages as its row shows them: the leaflet compared on them, everything else on the page cleared
    and trimmed away (a page compared whole: trimmed to what is printed). Returns both pictures and the crop's (x, y)."""
    original, suspect = outcome.aligned_original, outcome.aligned_suspect
    area = outcome.compare_mask
    if area is not None and area.any() and not area.all():
        original, suspect = apply_mask(original, area), apply_mask(suspect, area)
    x0, y0, x1, y1 = _content_box(original, suspect)
    return original[y0:y1, x0:x1], suspect[y0:y1, x0:x1], (x0, y0)


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
    member = DiffCandidate(bbox=box, type=DiffType.GRAPHICS, subtype=kind, confidence=1.0, detail=detail, source=SOURCE_PAGE)
    return DiffCluster(
        id=0,
        bbox_in_original=box,
        bbox_in_suspect=box,
        type=DiffType.GRAPHICS,
        members=[member],
        detail=detail,
    )


def _with_reviewers(found: IsolationGeometry | None, own: IsolationGeometry | None) -> IsolationGeometry | None:
    """A page's area as compared: the reviewer's own area for it where they set one, else the leaflet found on it; less
    whatever they erased on it either way."""
    if own is None:
        return found
    if own.boundary is not None:
        return own
    if found is None:
        return own
    return replace(found, erase_regions=list(found.erase_regions) + list(own.erase_regions), erase_brush_radius_norm=own.erase_brush_radius_norm)


def compare_leaflet(
    original: LoadedArtwork,
    suspect: LoadedArtwork,
    config: AppConfig,
    progress: Callable[[str], None] | None = None,
    original_areas: dict[int, IsolationGeometry] | None = None,
    suspect_areas: dict[int, IsolationGeometry] | None = None,
    align_only: bool = False,
) -> ComparisonOutcome:
    """Compare every page of `original` against every page of `suspect` it is paired with (see the module's account).
    `original_areas`/`suspect_areas`: what the reviewer set on each page of each file (by page, 0-based) -- a page's
    own area, compared instead of the leaflet found on it, and areas left out of it."""
    original_areas, suspect_areas = original_areas or {}, suspect_areas or {}
    say = progress or (lambda _text: None)
    leaflet_config = config.leaflet
    if original.page_count == 0 or suspect.page_count == 0:
        raise NoPagesComparedError("One of the files has no pages.")

    say("Rendering the pages…")
    with StageTimer(logger, "leaflet_render"):
        original_pages = [_page(original, i) for i in range(original.page_count)]
        suspect_pages = [_page(suspect, i) for i in range(suspect.page_count)]

    # each PDF page's own words (none for an outlined page, or a picture): exact, and instant beside reading it
    original_layers = [_page_words(original, i, page) for i, page in enumerate(original_pages)]
    suspect_layers = [_page_words(suspect, i, page) for i, page in enumerate(suspect_pages)]

    found_languages = None
    if leaflet_config.language == AUTO:
        # The languages printed are found from the first pages beside the pages being paired and compared; a reading
        # asked for meanwhile waits only for what is left of it (`language_detect`).
        sample = list(original_pages[:_LANGUAGE_PAGES]) + list(suspect_pages[:_LANGUAGE_PAGES])
        texts = [[w.text for w in (layer or [])] for layer in original_layers[:_LANGUAGE_PAGES] + suspect_layers[:_LANGUAGE_PAGES]]
        found_languages = language_detect.begin(lambda: detect_languages(sample, texts))
        language, language_ok = AUTO, True
    else:
        language, language_ok = usable_language(leaflet_config.language)
    page_config = _with_language(config, language)  # (every reading on the pages, not just the word comparison)
    warnings: list[str] = []
    if not language_ok:
        warnings.append(
            f"'{leaflet_config.language}' is not installed for the text reader, so '{language}' was used instead. "
            "Findings are still located and flagged correctly; only their words may not be read where this is the "
            "wrong language for them."
        )

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

    # each PDF page's own lines and paths (folds divide a leaflet into panels; its outline and fold lines are drawn out)
    original_extras = [_pdf_lines_and_strokes(original, i, page) for i, page in enumerate(original_pages)]
    suspect_extras = [_pdf_lines_and_strokes(suspect, i, page) for i, page in enumerate(suspect_pages)]

    def compare_one(pair: tuple[int, int]) -> tuple[int, ComparisonOutcome | Exception]:
        o, s = pair
        original_page, suspect_page = original_pages[o], suspect_pages[s]
        try:
            # 1. The leaflet on each page: what is outside it (marks, notes, the rest of the sheet) is not compared.
            original_area, suspect_area = _leaflet_areas(original, suspect, o, s, original_page, suspect_page, page_config)
            original_area = _with_reviewers(original_area, original_areas.get(o))
            suspect_area = _with_reviewers(suspect_area, suspect_areas.get(s))
            # 2. Its pixels: aligned by what is printed (never by the outlines -- see `run_comparison`), then panel by
            #    panel along its folds, with its outline and fold lines drawn out of both pages.
            visual_outcome = run_comparison(
                original_page,
                suspect_page,
                ComparisonMode.LEAFLET,
                page_config,
                original_geometry=original_area,
                suspect_geometry=suspect_area,
                options=_VISUAL_STRATEGY_OPTIONS,
                original_lines=original_extras[o][0],
                suspect_lines=suspect_extras[s][0],
                original_words=original_layers[o],
                suspect_words=suspect_layers[s],
                original_strokes=original_extras[o][1],
                suspect_strokes=suspect_extras[s][1],
                align_only=align_only,
            )
            if align_only:
                return o, visual_outcome
            # 3. Its words, on exactly that alignment. Both strategies always: the results page shows or hides each
            #    one's findings (its OCR / pixel toggles) at once, with nothing compared again.
            text_outcome = word_outcome(visual_outcome, original_layers[o], suspect_layers[s], page_config, language)
            return o, _merge_outcomes(visual_outcome, text_outcome)
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
            shown_original, shown_suspect, crop = _leaflet_row(outcome)
            label = f"Page {o + 1}" if o == s_index else f"Original p.{o + 1} / Suspect p.{s_index + 1}"
            rows.append((shown_original, shown_suspect, label))
            row_crops.append(crop)
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
            row_clusters.append([] if align_only else [_whole_page_finding(DiffSubtype.REMOVED, o, 0, trimmed.shape[:2])])
    for s_index in unmatched_s:
        page = suspect_pages[s_index]
        x0, y0, x1, y1 = _content_box(page)
        trimmed = page[y0:y1, x0:x1]
        rows.append((_placeholder(*trimmed.shape[:2], "No page of the original matches this one"), trimmed, f"Suspect p.{s_index + 1}"))
        row_crops.append((0, 0))
        row_clusters.append([] if align_only else [_whole_page_finding(DiffSubtype.ADDED, s_index, 0, trimmed.shape[:2])])

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
    said = language_detect.note(found_languages)
    if said:
        notes.append(said)

    return ComparisonOutcome(
        alignment=reference_alignment,
        candidates=[],
        clusters=[] if align_only else renumbered(
            visible_by_source(all_clusters, ocr_wanted=leaflet_config.compare_text, pixels_wanted=leaflet_config.compare_visual),
            all_clusters,
        ),
        all_clusters=[] if align_only else all_clusters,
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
        notes=["Files updated and aligned. Click Recompare to run comparison."] if align_only else notes,
    )


_LANGUAGE_PAGES = 2  # (a leaflet's languages are found on its first pages: a page or two holds every language it has)


def _with_language(config: AppConfig, language: str) -> AppConfig:
    return replace(config, detection=replace(config.detection, ocr_language=language))
