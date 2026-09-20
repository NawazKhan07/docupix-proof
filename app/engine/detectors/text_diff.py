"""Text and number difference detection via OCR (Architecture §3.3.5).

Both aligned artwork regions are OCR'd with Tesseract (offline), then diffed
at token level rather than pixel level, so "500mg -> 250mg" is reported as
one text change with the old and new values rather than as a scattering of
pixel regions. Because both images are already in the shared aligned space,
a token that OCRs identically but sits somewhere else is a *move* (FR-6.1),
which a pure string diff would never notice.

Tesseract is a separate native binary, not a Python package; when it is
absent this raises `OcrUnavailableError` so the pipeline can report that
text was not checked instead of quietly returning "no text differences".

A scan is read less surely than the file it was made from, so Scan vs Digital uses `detect_scan`,
which is anchored on the digital file's lines -- see `DetectionConfig.ocr_min_confidence_scan`.
"""

from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from difflib import SequenceMatcher

import cv2
import numpy as np
import pytesseract

from app.config import DetectionConfig
from app.engine.aggregation.severity import LIKELY_MISREAD_CONFIDENCE
from app.engine.detectors.verify import marks_agreement
from app.logging_config import StageTimer
from app.models.diff import BoundingBox, DiffCandidate, DiffSubtype, DiffType

logger = logging.getLogger(__name__)


class OcrUnavailableError(RuntimeError):
    """Tesseract is not installed or not on PATH."""


@dataclass
class OcrWord:
    text: str
    bbox: BoundingBox
    confidence: float


def _ocr_data(image: np.ndarray, page_segmentation: str | None = None) -> dict:
    try:
        return pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT, **({"config": page_segmentation} if page_segmentation else {})
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise OcrUnavailableError(
            "Tesseract OCR is not installed or not on PATH, so text and number "
            "differences could not be checked. Install Tesseract and restart."
        ) from exc


def _words_of(data: dict, floor: float) -> list[tuple[tuple[int, int, int], OcrWord]]:
    """(the line a word is on, the word), for every word read at least as surely as `floor`."""
    found = []
    for index, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        confidence = float(data["conf"][index])
        if not text or confidence < floor:
            continue
        left, top = float(data["left"][index]), float(data["top"][index])
        word = OcrWord(
            text=text,
            bbox=BoundingBox(left, top, left + float(data["width"][index]), top + float(data["height"][index])),
            confidence=confidence,
        )
        found.append(((data["block_num"][index], data["par_num"][index], data["line_num"][index]), word))
    return found


def _ocr_words(image: np.ndarray, config: DetectionConfig, floor: float | None = None) -> list[OcrWord]:
    data = _ocr_data(image)
    return [word for _line, word in _words_of(data, config.ocr_min_confidence if floor is None else floor)]


def _is_numeric(token: str) -> bool:
    return any(character.isdigit() for character in token) and not any(
        character.isalpha() for character in token
    )


def _span_type(tokens: list[str]) -> DiffType:
    """A span is a NUMBER change only if it is all numbers (FR-6.1)."""
    return DiffType.NUMBER if tokens and all(_is_numeric(token) for token in tokens) else DiffType.TEXT


def _span_bbox(words: list[OcrWord]) -> BoundingBox | None:
    if not words:
        return None
    box = words[0].bbox
    for word in words[1:]:
        box = box.union(word.bbox)
    return box


def _mean_confidence(words: list[OcrWord]) -> float:
    return float(np.mean([word.confidence for word in words])) / 100 if words else 0.5


def detect(
    original: np.ndarray, suspect: np.ndarray, config: DetectionConfig
) -> list[DiffCandidate]:
    """Text/number differences between two images already in the shared aligned space."""
    with StageTimer(logger, "detector_text"):
        # Rotated text, a barcode's digits and a dotted line are read as junk words; keep only what reads as words.
        original_words = _trustworthy(_ocr_words(original, config), config)
        suspect_words = _trustworthy(_ocr_words(suspect, config), config)

        matcher = SequenceMatcher(
            None,
            [word.text for word in original_words],
            [word.text for word in suspect_words],
            autojunk=False,
        )

        candidates: list[DiffCandidate] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            original_span = original_words[i1:i2]
            suspect_span = suspect_words[j1:j2]

            if tag == "equal":
                candidates.extend(_moved_candidates(original_span, suspect_span, config))
            else:
                # replace / delete / insert. Words far apart are separate findings (a long run of misaligned
                # readings is not one change the size of the page): see `_findings`.
                candidates.extend(_findings(original_span, suspect_span))

    candidates = [candidate for candidate in candidates if candidate is not None]
    logger.info(
        "text detector finished",
        extra={
            "original_words": len(original_words),
            "suspect_words": len(suspect_words),
            "candidates": len(candidates),
        },
    )
    return candidates


def _moved_candidates(
    original_span: list[OcrWord], suspect_span: list[OcrWord], config: DetectionConfig
) -> list[DiffCandidate]:
    """Identical tokens that sit somewhere else are moves, not matches."""
    candidates = []
    for original_word, suspect_word in zip(original_span, suspect_span):
        original_center, suspect_center = original_word.bbox.center, suspect_word.bbox.center
        displacement = math.dist(original_center, suspect_center)
        if displacement > config.text_move_tolerance_px:
            candidates.append(
                _candidate(
                    original_word.bbox.union(suspect_word.bbox),
                    DiffType.NUMBER if _is_numeric(original_word.text) else DiffType.TEXT,
                    DiffSubtype.MOVED,
                    _mean_confidence([original_word, suspect_word]),
                    f"'{original_word.text}' moved {displacement:.0f} px",
                    magnitude=displacement,  # how far it moved: words that moved alike are one move
                )
            )
    return candidates


def _join(words: list[OcrWord]) -> str:
    return " ".join(word.text for word in words)


def _candidate(
    bbox: BoundingBox | None,
    diff_type: DiffType,
    subtype: DiffSubtype,
    confidence: float,
    detail: str,
    magnitude: float = 0.0,
) -> DiffCandidate | None:
    if bbox is None:
        return None
    return DiffCandidate(
        bbox=bbox, type=diff_type, subtype=subtype, confidence=confidence, detail=detail, magnitude=magnitude
    )


# ---------------------------------------------------------------------------
# Comparing a scan with a digital file
# ---------------------------------------------------------------------------

# An l, a 1 and an i are one smudge apart in a soft scan, and so are an O and a 0.
_OCR_CONFUSABLE = str.maketrans({"l": "i", "1": "i", "0": "o"})
_MIN_ALNUM_SHARE = 0.5  # a "word" that is mostly marks is not one
_MAX_LEFTOVER_SPREAD = 8.0  # words this scattered (box area / area of the words) are separate findings
_SHAKY_DIGITAL_MAX = 75.0  # the digital file's words are read no more surely than this...
_SURE_SCAN_MIN = 80.0  # ...where the scan's are read at least this surely...
_SHAKY_MATCH_RATIO = 0.8  # ...and they are this alike
_WINDOW_UPSCALE_BELOW_PX = 80  # a line's window is enlarged to at least this tall before it is read
_DARK = 128  # an ink picture's pixels below this are ink
_LINE_PAGE_SEGMENTATION = "--psm 7"  # a window holds one line
_AGREEMENT_SLACK_PX = 2  # ink this close counts as the same ink when judging a misread
_AGREEMENT_PAD_PX = 6


@dataclass
class TextScanResult:
    candidates: list[DiffCandidate]
    # Lines of the digital file whose ink is in the scan but which the scan is too soft to read: not compared.
    unreadable_lines: int = 0


def canonical(text: str) -> str:
    """What a word is, for comparing across a scan: no case, no punctuation, no l/i/1 or o/0 confusion."""
    return "".join(ch for ch in text.casefold() if ch.isalnum()).translate(_OCR_CONFUSABLE)


def _has_digit(*texts: str) -> bool:
    return any(ch.isdigit() for text in texts for ch in text)


def _trustworthy(words: list[OcrWord], config: DetectionConfig) -> list[OcrWord]:
    """Drop what is read as marks rather than words: mostly punctuation, or a very short word read shakily."""
    kept = []
    for word in words:
        key = canonical(word.text)
        alnum = sum(ch.isalnum() for ch in word.text)
        if len(key) < 2 or alnum < _MIN_ALNUM_SHARE * len(word.text):
            continue
        if len(key) <= 2 and word.confidence < config.ocr_short_word_confidence:
            continue
        kept.append(word)
    return kept


def _same_word(a: OcrWord, b: OcrWord) -> bool:
    """Whether two readings are the same word once case, punctuation and the l/i/1 and o/0 confusions are set aside.

    Nothing looser: 'Store' against 'Stone' is one letter, and so is a real change to a word. A scan's smudged
    letters are instead ranked down when the ink shows the marks are the same (`_marked_if_misread`).
    """
    return canonical(a.text) == canonical(b.text)


def _leftovers(
    original_span: list[OcrWord], suspect_span: list[OcrWord]
) -> tuple[list[OcrWord], list[OcrWord]]:
    """The words of a replaced stretch that no word on the other side reads as: each word is matched, in order,
    to the first still-unused word it is the same as (a differently grouped line)."""
    unused = list(suspect_span)
    unmatched_original = []
    for word in original_span:
        match = next((other for other in unused if _same_word(word, other)), None)
        if match is None:
            unmatched_original.append(word)
        else:
            unused.remove(match)
    return unmatched_original, unused


_RUN_GAP_PX = 60  # words further apart than this (~5 mm at 300 DPI) are not one phrase


def _runs(words: list[OcrWord]) -> list[list[OcrWord]]:
    """The words grouped into phrases: each follows the one before it closely, on the same line."""
    runs: list[list[OcrWord]] = []
    for word in words:
        if runs:
            last = runs[-1][-1].bbox
            same_line = abs(word.bbox.center[1] - last.center[1]) <= max(last.height, word.bbox.height)
            if same_line and 0 <= word.bbox.x0 - last.x1 <= _RUN_GAP_PX:
                runs[-1].append(word)
                continue
        runs.append([word])
    return runs


def _digital_reading_unsure(gone: list[OcrWord], new: list[OcrWord]) -> bool:
    """Whether a difference is the DIGITAL file's reading being unsure: what it read is a shaky near-miss of a word the
    scan reads confidently ('Cilnidip' at 67% for 'Cilnidipine' at 87%: a hairline typeface is hard to read even in
    a crisp file). A real change to a word is read surely on the digital side."""
    if not gone or not new or max(word.confidence for word in gone) >= _SHAKY_DIGITAL_MAX or min(word.confidence for word in new) < _SURE_SCAN_MIN:
        return False
    ratio = SequenceMatcher(None, canonical(_join(gone)), canonical(_join(new)), autojunk=False).ratio()
    return ratio >= _SHAKY_MATCH_RATIO


def _findings(gone: list[OcrWord], new: list[OcrWord]) -> list[DiffCandidate]:
    """Findings for the words left over in one replaced stretch."""
    if gone and new:
        both = gone + new
        box = _span_bbox(both)
        spread = box.area / max(1.0, sum(word.bbox.area for word in both))
        if spread <= _MAX_LEFTOVER_SPREAD:
            unsure = _digital_reading_unsure(gone, new)
            return [
                _candidate(
                    box,
                    _span_type([word.text for word in both]),
                    DiffSubtype.CHANGED,
                    LIKELY_MISREAD_CONFIDENCE if unsure else _mean_confidence(both),
                    f"'{_join(gone)}' -> '{_join(new)}'" + (" (the digital file's reading is unsure: likely a misread)" if unsure else ""),
                )
            ]
    findings = []
    for words, subtype, wording in ((gone, DiffSubtype.REMOVED, "missing from suspect"), (new, DiffSubtype.ADDED, "added in suspect")):
        for span in _runs(words):
            findings.append(
                _candidate(_span_bbox(span), _span_type([word.text for word in span]), subtype, _mean_confidence(span), f"'{_join(span)}' {wording}")
            )
    return findings


def _line_differences(original_line: list[OcrWord], suspect_line: list[OcrWord]) -> list[DiffCandidate]:
    """The differences between two readings of one line."""
    matcher = SequenceMatcher(
        None, [canonical(w.text) for w in original_line], [canonical(w.text) for w in suspect_line], autojunk=False
    )
    found: list[DiffCandidate] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        gone, new = _leftovers(original_line[i1:i2], suspect_line[j1:j2])
        found += _findings(gone, new)
    return [candidate for candidate in found if candidate is not None]


def _padded(box: BoundingBox, pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    return max(0, int(box.x0) - pad), max(0, int(box.y0) - pad), min(width, math.ceil(box.x1) + pad), min(height, math.ceil(box.y1) + pad)


def _ink_in(image: np.ndarray, box: BoundingBox, pad: int = 0) -> int:
    """How much ink an ink picture has inside `box`."""
    x0, y0, x1, y1 = _padded(box, pad, image.shape)
    return int((image[y0:y1, x0:x1] < _DARK).sum())


# --- reading a whole page fast ---------------------------------------------------------------------------------------

_STRIP_MIN_HEIGHT_PX = 900  # a page is cut into strips at least this tall, as many as there are cores to read them
_STRIP_SEARCH_SHARE = 0.4  # a cut is looked for this share of a strip's height either side of the even split
_BLOCK_CLOSE_PX = 15  # lettering reversed out of a block is filled in, to see the block, with a square this big
_BLOCK_MIN_AREA_PX = 2500  # a solid block at least this big has its lettering read dark on light
_BLOCK_MIN_SIDE_PX = 24
_BLOCK_MIN_FILL = 0.55  # share of a block that is ink: a paragraph closed up is ~0.3, a bar with lettering ~0.75
_BLOCK_CORE_PX = 13  # ink that survives being shrunk by this much is SOLID: a stroke of lettering is thinner
_BLOCK_MIN_CORE = 0.15  # a block is at least this share solid ink (bold headings, whose closed lines fill as much, are not)
_MONTAGE_GAP_PX = 44  # blank between the windows stacked to be read at once
_MONTAGE_MAX_HEIGHT_PX = 6000
_MONTAGE_MAX_WIDTH_PX = 4200
_MATCH_MIN_REACH_PX = 40  # a word in the scan is 'the same word' if it is this close to it (two word-heights, if more):
_MATCH_REACH_HEIGHTS = 2.0  # a two-sheet scan of a carton was 45 px off in places
_MATCH_GRID_PX = 96


def reading_picture(ink: np.ndarray) -> np.ndarray:
    """An ink picture as it is READ: text reads best dark on light, so inside a solid block (white lettering
    reversed out of a dark bar) the picture is reversed and the lettering reads dark on light like any other.

    Left as it is, such lettering is read as junk in the file that has it, and the words 'missing' or 'changed'
    over every dark bar of the sheet were a large part of the false findings on a real pair.
    """
    dark = (ink < _DARK).astype(np.uint8)
    size = 2 * (_BLOCK_CLOSE_PX // 2) + 1
    closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (size, size)))
    solid = cv2.erode(dark, np.ones((_BLOCK_CORE_PX, _BLOCK_CORE_PX), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    result = None
    for label in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[label])
        if area < _BLOCK_MIN_AREA_PX or min(width, height) < _BLOCK_MIN_SIDE_PX:
            continue
        window = (slice(y, y + height), slice(x, x + width))
        inside = labels[window] == label
        if dark[window][inside].mean() < _BLOCK_MIN_FILL or solid[window][inside].mean() < _BLOCK_MIN_CORE:
            continue
        if result is None:
            result = ink.copy()
        view = result[window]
        view[inside] = 255 - view[inside]
    return _stroked(ink if result is None else result)


_THIN_STROKE_PX = 2.8  # ink whose strokes are on average thinner than this is thickened for reading


def _stroked(picture: np.ndarray) -> np.ndarray:
    """A picture whose ink is at least a few pixels thick, for reading. A light-weight typeface is a hairline in a
    crisp digital file (two pixels wide, on letters forty pixels tall) and Tesseract reads it much less surely than
    the scan of it, whose ink has spread: 'Cilnidioine' for 'Cilnidipine' at 57%. The average stroke is measured from
    the picture (twice the ink's area over its edge) and thickened by a pixel if it is thinner than
    `_THIN_STROKE_PX`; ordinary lettering, and a scan's fat strokes, are left as they are.
    """
    dark = picture < _DARK
    area = int(dark.sum())
    if area < 500:
        return picture
    edge = int((dark & ~cv2.erode(dark.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)).sum())
    if 2.0 * area / max(edge, 1) >= _THIN_STROKE_PX:
        return picture
    thick = cv2.dilate(dark.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return np.where(thick, 0, picture).astype(np.uint8)


def _cut_rows(picture: np.ndarray, parts: int) -> list[int]:
    """Rows to cut a page into `parts` strips at: near an even split, each in the emptiest row close by, so a
    line of text is not cut through."""
    height = picture.shape[0]
    if parts <= 1:
        return [0, height]
    ink = cv2.reduce((picture < _DARK).astype(np.uint8), 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S).ravel()
    reach = int(_STRIP_SEARCH_SHARE * height / parts)
    cuts = [0]
    for part in range(1, parts):
        even = round(part * height / parts)
        low, high = max(cuts[-1] + 1, even - reach), min(height - 1, even + reach)
        if high < low:
            cuts.append(even)
            continue
        window = ink[low : high + 1]
        emptiest = np.flatnonzero(window == window.min()) + low
        cuts.append(int(emptiest[np.argmin(np.abs(emptiest - even))]))  # the emptiest row, the one nearest the even split
    cuts.append(height)
    return cuts


def read_pages(pictures: list[np.ndarray], floor: float) -> list[list[tuple[tuple, OcrWord]]]:
    """Every word read on each picture, as (line, word): each page is cut into strips read side by side.

    Reading a page is by far the slowest step of Scan vs Digital (a text-dense sheet takes a few seconds on one
    core), so the strips of both pages are read together, one strip to a core.
    """
    workers = max(1, min(_MAX_READERS, os.cpu_count() or 2))
    jobs = []  # (picture number, strip number, first row, last row)
    for number, picture in enumerate(pictures):
        parts = max(1, min(workers, picture.shape[0] // _STRIP_MIN_HEIGHT_PX))
        cuts = _cut_rows(picture, parts)
        jobs += [(number, strip, cuts[strip], cuts[strip + 1]) for strip in range(len(cuts) - 1)]

    def read(job) -> list[tuple[tuple, OcrWord]]:
        number, strip, first, last = job
        found = []
        for line, word in _words_of(_ocr_data(pictures[number][first:last]), floor):
            b = word.bbox
            moved = OcrWord(word.text, BoundingBox(b.x0, b.y0 + first, b.x1, b.y1 + first), word.confidence)
            found.append(((strip, *line), moved))
        return found

    with StageTimer(logger, "read_pages"):
        if len(jobs) == 1:
            results = [read(jobs[0])]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(read, jobs))
    pages: list[list[tuple[tuple, OcrWord]]] = [[] for _ in pictures]
    for job, found in zip(jobs, results):
        pages[job[0]] += found
    return pages


def _stack(crops: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Pictures stacked one above the other with a blank band between, and the rows each one occupies."""
    width = max(crop.shape[1] for crop in crops)
    total = sum(crop.shape[0] for crop in crops) + _MONTAGE_GAP_PX * (len(crops) + 1)
    montage = np.full((total, width), 255, np.uint8)
    rows, y = [], _MONTAGE_GAP_PX
    for crop in crops:
        montage[y : y + crop.shape[0], : crop.shape[1]] = crop
        rows.append((y, y + crop.shape[0]))
        y += crop.shape[0] + _MONTAGE_GAP_PX
    return montage, rows


def read_windows(picture: np.ndarray, boxes: list[BoundingBox], config: DetectionConfig) -> list[list[OcrWord]]:
    """The words in a small window round each box (one line of text each), in the picture's own pixels.

    The windows are stacked into pictures and each picture is read once: reading them one at a time, twice for
    every line of a page, was most of the time Scan vs Digital took. Windows a little short of a line are
    enlarged first; the words are sorted along the line.
    """
    prepared = []  # (window's corner, scale, picture)
    for box in boxes:
        x0, y0, x1, y1 = _padded(box, config.text_line_pad_px, picture.shape)
        if x1 - x0 < 8 or y1 - y0 < 8:
            prepared.append(None)
            continue
        crop = picture[y0:y1, x0:x1]
        factor = min(max(1.0, _WINDOW_UPSCALE_BELOW_PX / crop.shape[0]), _MONTAGE_MAX_WIDTH_PX / crop.shape[1])
        if factor > 1.0:
            crop = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
        prepared.append((x0, y0, factor, crop))

    batches: list[list[int]] = [[]]
    height = 0
    for index, item in enumerate(prepared):
        if item is None:
            continue
        rows = item[3].shape[0] + _MONTAGE_GAP_PX
        if batches[-1] and height + rows > _MONTAGE_MAX_HEIGHT_PX:
            batches.append([])
            height = 0
        batches[-1].append(index)
        height += rows

    def read(batch: list[int]) -> list[tuple[int, list[OcrWord]]]:
        if not batch:
            return []
        montage, rows = _stack([prepared[i][3] for i in batch])
        placed = {i: [] for i in batch}
        for _line, word in _words_of(_ocr_data(montage, _BLOCK_PAGE_SEGMENTATION), config.ocr_min_confidence):
            b = word.bbox
            middle = (b.y0 + b.y1) / 2
            for i, (top, bottom) in zip(batch, rows):
                if top - _MONTAGE_GAP_PX / 2 <= middle < bottom + _MONTAGE_GAP_PX / 2:
                    x0, y0, factor, _crop = prepared[i]
                    placed[i].append(
                        OcrWord(word.text, BoundingBox(b.x0 / factor + x0, (b.y0 - top) / factor + y0, b.x1 / factor + x0, (b.y1 - top) / factor + y0), word.confidence)
                    )
                    break
        return list(placed.items())

    words: list[list[OcrWord]] = [[] for _ in boxes]
    with StageTimer(logger, "read_windows"):
        if len(batches) == 1:
            outcomes = [read(batches[0])]
        else:
            with ThreadPoolExecutor(max_workers=max(1, min(_MAX_READERS, os.cpu_count() or 2))) as pool:
                outcomes = list(pool.map(read, batches))
    for outcome in outcomes:
        for index, found in outcome:
            found.sort(key=lambda word: word.bbox.x0)
            words[index] = _trustworthy(found, config)
    return words


def _unmatched(anchor: list[OcrWord], scan: list[OcrWord]) -> tuple[set[int], set[int], list[tuple[int, int]]]:
    """(the digital words nothing in the scan matches, the scan words nothing in the digital file matches, and the
    (digital, scan) pairs that do match).

    A word matches a word of the scan that reads the same (`canonical`) and lies close to where it is, so where a
    page reads in a different order, or a line is grouped differently, nothing is out of step. Each scan word
    matches at most one digital word, the nearest first.
    """
    grid: dict[tuple[int, int], list[int]] = {}
    scan_keys = [canonical(word.text) for word in scan]
    for index, word in enumerate(scan):
        x, y = word.bbox.center
        grid.setdefault((int(x // _MATCH_GRID_PX), int(y // _MATCH_GRID_PX)), []).append(index)
    used: set[int] = set()
    missing: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for index, word in enumerate(anchor):
        key = canonical(word.text)
        x, y = word.bbox.center
        reach = max(float(_MATCH_MIN_REACH_PX), _MATCH_REACH_HEIGHTS * word.bbox.height)
        best, best_distance = None, reach
        for cell_x in range(int((x - reach) // _MATCH_GRID_PX), int((x + reach) // _MATCH_GRID_PX) + 1):
            for cell_y in range(int((y - reach) // _MATCH_GRID_PX), int((y + reach) // _MATCH_GRID_PX) + 1):
                for other in grid.get((cell_x, cell_y), ()):
                    if other in used or scan_keys[other] != key:
                        continue
                    ox, oy = scan[other].bbox.center
                    distance = math.hypot(ox - x, oy - y)
                    if distance < best_distance:
                        best, best_distance = other, distance
        if best is None:
            missing.add(index)
        else:
            used.add(best)
            pairs.append((index, best))
    return missing, set(range(len(scan))) - used, pairs


_OFFSET_RADIUS_PX = 600  # where the scan sits against the digital file is judged from the words matched this close
_OFFSET_MIN_PAIRS = 3


def _local_offset(
    pairs: list[tuple[int, int]], anchor: list[OcrWord], scan: list[OcrWord], box: BoundingBox
) -> tuple[float, float]:
    """How far the scan is from the digital file round `box`: the median displacement of the words that matched
    nearby. A scan of a carton is bent and scaled a little, so it is off by tens of pixels in places -- enough that
    a window cut from the digital file's line misses the first letter of the scan's."""
    cx, cy = box.center
    shifts = []
    for i, j in pairs:
        ax, ay = anchor[i].bbox.center
        if abs(ax - cx) <= _OFFSET_RADIUS_PX and abs(ay - cy) <= _OFFSET_RADIUS_PX:
            bx, by = scan[j].bbox.center
            shifts.append((bx - ax, by - ay))
    if len(shifts) < _OFFSET_MIN_PAIRS:
        return (0.0, 0.0)
    return (float(np.median([dx for dx, _ in shifts])), float(np.median([dy for _, dy in shifts])))


def _moved(words: list[OcrWord], offset: tuple[float, float]) -> list[OcrWord]:
    """Words moved by `offset` (the scan's words back into the digital file's frame use the negative of it)."""
    return [OcrWord(w.text, _shifted_box(w.bbox, offset), w.confidence) for w in words]


def _union_of_readings(first: list[OcrWord], second: list[OcrWord]) -> list[OcrWord]:
    """Two readings of one line as one: all of the first, and each word of the second that lies where none of the
    first's does. Whichever reading found a word, it is there."""
    merged = list(first)
    for word in second:
        if not any(word.bbox.intersection_area(other.bbox) > 0.3 * word.bbox.area for other in first):
            merged.append(word)
    return sorted(merged, key=lambda w: w.bbox.x0)


def _fewest_differences(digital: list[list[OcrWord]], scan: list[list[OcrWord]]) -> list[DiffCandidate]:
    """The differences between the readings that agree best: the same line read (or taken from the page) in the several
    ways, in either file. A word that one way of reading misses and another finds is not a difference, and a real
    change is different in every one of them."""
    best: list[DiffCandidate] | None = None
    best_cost: tuple[int, int] | None = None
    for reading in digital:
        for other in scan:
            if not other or not reading:
                continue
            found = _line_differences(reading, other)
            if not found:
                return []
            matcher = SequenceMatcher(None, [canonical(w.text) for w in reading], [canonical(w.text) for w in other], autojunk=False)
            unmatched = len(reading) + len(other) - 2 * sum(block.size for block in matcher.get_matching_blocks())
            cost = (len(found), unmatched)  # fewest findings, then fewest words that do not line up
            if best_cost is None or cost < best_cost:
                best, best_cost = found, cost
    return best or []


def _is_upright_unreadable(words: list[OcrWord]) -> bool:
    """Whether a line was read from text that is turned on its side: its words are taller than they are wide.
    Read upright it is junk ('HSINYVANN'), so it is not compared."""
    tall = sum(1 for word in words if word.bbox.height > 1.5 * word.bbox.width and word.bbox.height >= 40)
    return tall * 2 > len(words)


def detect_scan(original: np.ndarray, suspect: np.ndarray, config: DetectionConfig) -> TextScanResult:
    """Text differences between a digital file and a scan of it, both as ink pictures in the shared frame.

    Both files are read whole (in strips, side by side) and their words matched by what they say and where they
    are. Only where words do not match is anything read again, line by line: the line's own small window, in both
    files, which is what says whether a difference is real (a changed word cannot reshuffle the reading order the
    way a whole-page comparison lets it). A scan is never registered exactly, so each line is compared where the
    scan actually is (judged from the words that matched nearby), and every way the line was read -- its window,
    the whole-page reading -- is tried: only if none of them agree is it a difference. See
    `DetectionConfig.ocr_min_confidence_scan` for what is ignored and why.
    """
    with StageTimer(logger, "detector_text"):
        reading_original, reading_suspect = reading_picture(original), reading_picture(suspect)
        digital_page, scan_page = read_pages([reading_original, reading_suspect], config.ocr_min_confidence)

        lines: dict[tuple, list[OcrWord]] = {}
        for line, word in digital_page:
            if word.confidence >= config.ocr_min_confidence_scan:
                lines.setdefault(line, []).append(word)
        anchors = [_trustworthy(sorted(words, key=lambda w: w.bbox.x0), config) for words in lines.values()]
        anchors = [
            words
            for words in anchors
            if words and (len(words) > 1 or len(canonical(words[0].text)) >= 4) and not _is_upright_unreadable(words)
        ]
        scan_words = _trustworthy([word for _line, word in scan_page], config)

        flat = [word for words in anchors for word in words]
        owner = [number for number, words in enumerate(anchors) for _word in words]
        missing, extra, pairs = _unmatched(flat, scan_words)
        suspicious = sorted({owner[index] for index in missing})
        boxes = [_span_bbox(anchors[number]) for number in suspicious]
        offsets = [_local_offset(pairs, flat, scan_words, box) for box in boxes]
        scan_boxes = [_shifted_box(box, offset) for box, offset in zip(boxes, offsets)]

        candidates: list[DiffCandidate] = []
        unreadable = 0
        if boxes:
            digital_readings = read_windows(reading_original, boxes, config)
            scan_readings = read_windows(reading_suspect, scan_boxes, config)
            pad = config.text_line_pad_px
            for number, box, offset, scan_box, digital_reading, scan_reading in zip(
                suspicious, boxes, offsets, scan_boxes, digital_readings, scan_readings
            ):
                # everything in the digital file's frame: the scan's words are moved back by how far the scan is off here
                back = (-offset[0], -offset[1])
                on_page = [
                    word
                    for word in scan_words
                    if scan_box.x0 - pad <= word.bbox.center[0] <= scan_box.x1 + pad
                    and scan_box.y0 - pad <= word.bbox.center[1] <= scan_box.y1 + pad
                ]
                scan_variants = [_moved(scan_reading, back), _moved(on_page, back)]
                scan_variants.append(_union_of_readings(*scan_variants))
                if not any(scan_variants):
                    if ink_agreement(original, suspect, box, offset) < config.text_present_ink_agreement:
                        anchor = digital_reading or anchors[number]
                        candidates.append(
                            _candidate(
                                box,
                                _span_type([w.text for w in anchor]),
                                DiffSubtype.REMOVED,
                                _mean_confidence(anchor),
                                f"'{_join(anchor)}' missing from suspect",
                            )
                        )
                    else:
                        unreadable += 1  # the ink is there, but too soft to read: say so, do not call it a change
                    continue
                digital_variants = [digital_reading, anchors[number], _union_of_readings(digital_reading, anchors[number])]
                found = _fewest_differences(digital_variants, scan_variants)
                candidates += [_marked_if_misread(candidate, original, suspect, config, offset) for candidate in found]

        windows = [_span_bbox(words) for words in anchors]
        added = [scan_words[i] for i in sorted(extra)]
        candidates += _added_text(added, pairs, flat, scan_words, original, suspect, windows, config)

    logger.info(
        "text detector finished (scan)",
        extra={
            "lines": len(anchors),
            "reread_lines": len(boxes),
            "candidates": len(candidates),
            "unreadable_lines": unreadable,
        },
    )
    return TextScanResult(candidates, unreadable)


def _added_text(
    extra: list[OcrWord],
    pairs: list[tuple[int, int]],
    anchor: list[OcrWord],
    scan: list[OcrWord],
    original: np.ndarray,
    suspect: np.ndarray,
    windows: list[BoundingBox],
    config: DetectionConfig,
) -> list[DiffCandidate]:
    """Words the scan has, read surely, outside every line of the digital file, where the digital has no ink."""
    pad = config.text_line_pad_px
    found = []
    for word in extra:
        if word.confidence < config.text_added_confidence or len(canonical(word.text)) < 3:
            continue
        offset = _local_offset(pairs, anchor, scan, word.bbox)
        home = _shifted_box(word.bbox, (-offset[0], -offset[1]))  # where the word is in the digital file's frame
        cx, cy = home.center
        if any(w.x0 - pad <= cx <= w.x1 + pad and w.y0 - pad <= cy <= w.y1 + pad for w in windows):
            continue
        if _ink_in(original, home, config.text_ink_reach_px) < config.text_added_ink_share * _ink_in(suspect, word.bbox, 3):
            found.append(
                _candidate(home, _span_type([word.text]), DiffSubtype.ADDED, word.confidence / 100, f"'{word.text}' added in suspect")
            )
    return [candidate for candidate in found if candidate is not None]


def _shifted_box(box: BoundingBox, offset: tuple[float, float]) -> BoundingBox:
    return BoundingBox(box.x0 + offset[0], box.y0 + offset[1], box.x1 + offset[0], box.y1 + offset[1])


_MAX_STROKE_MATCH_PX = 3


def _stroke_width(ink: np.ndarray) -> float:
    """The average width of the strokes of an ink mask: twice its area over its edge."""
    edge = int((ink & ~cv2.erode(ink.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)).sum())
    return 2.0 * float(ink.sum()) / max(edge, 1)


def _equal_strokes(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two ink masks with strokes of the same width: the thinner is thickened to the other's.

    A scan's dot gain makes a hairline of lettering several times as thick, and the same word then has ink far from
    the digital file's centre-lines although it is the same word. (Whether the marks coincide is what is asked.)
    """
    width_a, width_b = _stroke_width(a), _stroke_width(b)
    if abs(width_a - width_b) < 1.0:
        return a, b
    thinner_is_a = width_a < width_b
    radius = min(_MAX_STROKE_MATCH_PX, int(round(abs(width_a - width_b) / 2)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    if thinner_is_a:
        return cv2.dilate(a.astype(np.uint8), kernel).astype(bool), b
    return a, cv2.dilate(b.astype(np.uint8), kernel).astype(bool)


def ink_agreement(
    original: np.ndarray,
    suspect: np.ndarray,
    box: BoundingBox,
    offset: tuple[float, float] = (0.0, 0.0),
    slack_px: int = _AGREEMENT_SLACK_PX,
) -> float:
    """How closely the two files' ink in `box` coincides: the smaller of the share of each one's ink lying within a
    couple of pixels of the other's. 1.0 is the same marks (a scan's fatter strokes still give ~1.0); 0.0 if either
    has none there. `offset` is where the scan sits relative to the digital file here (a scan is never registered
    exactly): its window is `box` moved by that much."""
    x0, y0, x1, y1 = _padded(box, _AGREEMENT_PAD_PX, original.shape)
    sx0, sy0, sx1, sy1 = (int(round(v)) for v in (x0 + offset[0], y0 + offset[1], x1 + offset[0], y1 + offset[1]))
    a = original[y0:y1, x0:x1] < _DARK
    height, width = suspect.shape[:2]
    window = np.full(a.shape, 255, np.uint8)  # (what lies beyond the scan's edge is blank)
    cx0, cy0, cx1, cy1 = max(sx0, 0), max(sy0, 0), min(sx1, width), min(sy1, height)
    if cx1 > cx0 and cy1 > cy0:
        window[cy0 - sy0 : cy1 - sy0, cx0 - sx0 : cx1 - sx0] = suspect[cy0:cy1, cx0:cx1]
    b = window < _DARK
    if not a.any() or not b.any():
        return 0.0
    a, b = _equal_strokes(a, b)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * slack_px + 1, 2 * slack_px + 1))
    a_in_b = (a & cv2.dilate(b.astype(np.uint8), kernel).astype(bool)).sum() / a.sum()
    b_in_a = (b & cv2.dilate(a.astype(np.uint8), kernel).astype(bool)).sum() / b.sum()
    return float(min(a_in_b, b_in_a))


def _ranked_down(candidate: DiffCandidate, why: str) -> DiffCandidate:
    return DiffCandidate(
        bbox=candidate.bbox,
        type=candidate.type,
        subtype=candidate.subtype,
        confidence=min(candidate.confidence, LIKELY_MISREAD_CONFIDENCE),
        detail=f"{candidate.detail} ({why})",
        magnitude=candidate.magnitude,
    )


def _marked_if_misread(
    candidate: DiffCandidate,
    original: np.ndarray,
    suspect: np.ndarray,
    config: DetectionConfig,
    offset: tuple[float, float] = (0.0, 0.0),
) -> DiffCandidate:
    """A difference that is probably the reading's and not the artwork's, ranked down (still listed).

    * A word the scan reads differently although its ink is the digital word's: a misread of the scan.
    * A word said to be new, or missing, where the other file has the ink all the same: read in one file and not the
      other, not added or removed. (The ink is looked for where the scan sits, a scan never being registered exactly.)
    """
    if candidate.subtype is DiffSubtype.ADDED:
        here = _shifted_box(candidate.bbox, offset)
        if _ink_in(original, candidate.bbox, config.text_ink_reach_px) >= config.text_added_ink_share * _ink_in(suspect, here, 3):
            return _ranked_down(candidate, "the digital file has ink here that was not read: likely a misread")
        return candidate
    if candidate.subtype is DiffSubtype.REMOVED:
        here = _shifted_box(candidate.bbox, offset)
        if _ink_in(suspect, here, config.text_ink_reach_px) >= config.text_added_ink_share * _ink_in(original, candidate.bbox, 3):
            return _ranked_down(candidate, "the scan has ink here that was not read: likely a misread")
        return candidate
    if ink_agreement(original, suspect, candidate.bbox, offset, config.text_misread_slack_px) < config.text_misread_ink_agreement:
        return candidate
    return _ranked_down(candidate, "the ink matches: likely a misread of the scan")


# ---------------------------------------------------------------------------
# Two digital files: a reading that the marks do not support
# ---------------------------------------------------------------------------


def corroborated(
    candidates: list[DiffCandidate], original: np.ndarray, suspect: np.ndarray, config: DetectionConfig
) -> list[DiffCandidate]:
    """Drop text findings over marks that are the same in both files.

    Reading a page groups its lines differently when anything else on it differs, so one file can yield a
    word the other's reading misses although the pixels are identical (two extra 'changes' on a real
    carton pair whose pixels agreed everywhere but six places). Where the marks agree that closely the
    reading, not the artwork, differs. A blank crop cannot be checked, so the finding stays.
    """
    kept = []
    for candidate in candidates:
        agreement = marks_agreement(original, suspect, candidate.bbox)
        if agreement is not None and agreement >= config.text_identical_marks_agreement:
            logger.info("text finding dropped: the marks agree", extra={"detail": candidate.detail, "agreement": round(agreement, 3)})
            continue
        kept.append(candidate)
    return kept


# ---------------------------------------------------------------------------
# Describing what the pixels found
# ---------------------------------------------------------------------------
#
# In Automatic mode a difference is FOUND in the pixels, never by reading: two files can be read differently
# although their pixels are the same (one draws its text as live type, the other as a picture of it; reading a
# page groups its lines differently when anything else on it differs), and a comparison of two readings then
# reports whole areas that did not change. So reading only DESCRIBES a difference the pixels already found: the
# words in the finding's own box are read in each file, alone, and if they differ the finding is a text
# change with the old and the new words.

# Windows a finding is read in, (across, up-and-down) padding in pixels, tried in turn. No one window reads everything: a
# small one reads the finding's own words most surely (over a gradient a wider one lost a '100 g'), and a wide, short one
# takes in a whole line (a long word cut by the small one, the words a dropped word let move along).
_DESCRIBE_WINDOWS = ((48, 48), (128, 24))
_DESCRIBE_UPSCALE = 2.0  # small type is read far more surely enlarged (a real pair: '100g' beside '10' 'g' became '100' 'g')
_DESCRIBE_MAX_AREA_PX = 3_500_000  # a finding this big is a panel, not a phrase: not read
_DESCRIBE_REACH_PX = 24  # the phrase a finding belongs to reaches this far along its line (a word's height and a half, if more)
_BLOCK_PAGE_SEGMENTATION = "--psm 6"  # a window holds a block of text
_MAX_DETAIL_CHARS = 70
_DARK_BACKGROUND = 110  # a crop whose median tone is darker than this is light text on dark, and is read reversed
_MAX_READERS = 8


def _clip(text: str) -> str:
    return text if len(text) <= _MAX_DETAIL_CHARS else text[: _MAX_DETAIL_CHARS - 1].rstrip() + "…"


def _readable(crop: np.ndarray) -> np.ndarray:
    """A crop as a grey picture of dark text on light, whichever way round it is drawn: where the marks are lighter
    than the crop's usual tone (white lettering on a red box), it is reversed."""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    median = float(np.median(gray))
    low, high = np.percentile(gray, (2, 98))
    if high - low >= 30:
        marks = np.abs(gray.astype(np.int16) - int(median)) > (high - low) / 3
        if marks.any() and float(gray[marks].mean()) > median:
            return 255 - gray
    return gray


def _read_around(image: np.ndarray, box: BoundingBox, config: DetectionConfig, pad_x: int, pad_y: int) -> list[OcrWord]:
    """The words in a window around `box`, in reading order and in the whole picture's pixels."""
    height, width = image.shape[:2]
    x0, x1 = max(0, int(box.x0) - pad_x), min(width, math.ceil(box.x1) + pad_x)
    y0, y1 = max(0, int(box.y0) - pad_y), min(height, math.ceil(box.y1) + pad_y)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return []
    factor = _DESCRIBE_UPSCALE
    crop = cv2.resize(_readable(image[y0:y1, x0:x1]), None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    crop_width = crop.shape[1]
    words = []
    for _line, word in _words_of(_ocr_data(crop, _BLOCK_PAGE_SEGMENTATION), config.ocr_min_confidence):
        b = word.bbox
        # A word touching the window's edge is cut short by it ('Take' read as 'ce'), unless that edge is the picture's own.
        if (b.x0 <= 2 and x0 > 0) or (b.x1 >= crop_width - 2 and x1 < width):
            continue
        words.append(
            OcrWord(word.text, BoundingBox(b.x0 / factor + x0, b.y0 / factor + y0, b.x1 / factor + x0, b.y1 / factor + y0), word.confidence)
        )
    return _trustworthy(words, config)


def _supported(
    box: BoundingBox,
    gone: list[OcrWord],
    new: list[OcrWord],
    other_read_something: bool,
    original: np.ndarray,
    suspect: np.ndarray,
    config: DetectionConfig,
) -> bool:
    """Whether the pixels back a claim that these words changed, are missing or are new.

    Two readings of one picture differ for reasons that are not in the picture (white lettering on red read in one
    file and not in the other), so a claim stands only if the words were read surely AND the marks where they sit
    are not the same in both files. A word 'changed' over the same marks is the reader's, not the artwork's; and a
    word 'missing' or 'new' when the OTHER file was read as nothing at all is unread there, not gone, unless its
    marks are mostly gone too. (Where the other file was read as something, the reading is to be trusted: the
    words that followed have moved along into the place.) A word the finding itself touches needs no such check
    for a change: the pixels there are what differ.
    """
    words = gone + new
    if any(word.confidence < config.text_describe_min_confidence for word in words):
        return False
    if gone and new and any(word.bbox.intersection_area(box) > 0 for word in words):
        return True
    agreement = marks_agreement(original, suspect, _span_bbox(words), 2, config.residue_slack_px)
    if agreement is None:
        return True  # one file has nothing there: the words really are gone, or new
    if gone and new:
        return agreement < config.residue_min_agreement
    return other_read_something or agreement < config.text_describe_present_agreement


def _described(
    box: BoundingBox,
    words_original: list[OcrWord],
    words_suspect: list[OcrWord],
    original: np.ndarray,
    suspect: np.ndarray,
    config: DetectionConfig,
) -> DiffCandidate | None:
    """The text change inside `box` that two readings of it show, if they show one and the pixels agree it is one.

    The words on the finding's line are lined up across the whole window, so a word that was dropped is the one
    that is missing rather than everything after it having 'changed' (the words beside it moved along). Only the
    words that differ AND touch the finding, or lie within a word's height and a half of it, are what it says.
    """
    heights = [min(word.bbox.height, 3 * box.height + 20) for word in words_original + words_suspect]
    height = float(np.median(heights)) if heights else box.height
    reach = max(float(_DESCRIBE_REACH_PX), 1.5 * height)
    slack = max(0.75 * box.height, 14.0)

    def on_line(word: OcrWord) -> bool:
        return box.y0 - slack <= word.bbox.center[1] <= box.y1 + slack

    def at_finding(word: OcrWord) -> bool:
        return on_line(word) and word.bbox.x1 >= box.x0 - reach and word.bbox.x0 <= box.x1 + reach

    original_line = [word for word in words_original if on_line(word)]
    suspect_line = [word for word in words_suspect if on_line(word)]
    matcher = SequenceMatcher(
        None, [word.text for word in original_line], [word.text for word in suspect_line], autojunk=False
    )
    gone: list[OcrWord] = []
    new: list[OcrWord] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            gone += [word for word in original_line[i1:i2] if at_finding(word)]
            new += [word for word in suspect_line[j1:j2] if at_finding(word)]
    other_read_something = bool(suspect_line if gone else original_line)
    if not (gone or new) or not _supported(box, gone, new, other_read_something, original, suspect, config):
        return None  # the words are the same, or the pixels do not bear out that they changed

    if gone and new:
        subtype, detail = DiffSubtype.CHANGED, f"'{_clip(_join(gone))}' -> '{_clip(_join(new))}'"
    elif gone:
        subtype, detail = DiffSubtype.REMOVED, f"'{_clip(_join(gone))}' missing from suspect"
    else:
        subtype, detail = DiffSubtype.ADDED, f"'{_clip(_join(new))}' added in suspect"
    return _candidate(box, _span_type([word.text for word in gone + new]), subtype, _mean_confidence(gone + new), detail)


def describe(
    boxes: list[BoundingBox], original: np.ndarray, suspect: np.ndarray, config: DetectionConfig
) -> list[DiffCandidate | None]:
    """For each box where the pixels differ: the text change the words there show (None if they read the same,
    or there are none). Each box is read in both files, alone; the reads run side by side."""

    def read(box: BoundingBox) -> DiffCandidate | None:
        if box.area > _DESCRIBE_MAX_AREA_PX:
            return None
        for pad_x, pad_y in _DESCRIBE_WINDOWS:
            found = _described(
                box,
                _read_around(original, box, config, pad_x, pad_y),
                _read_around(suspect, box, config, pad_x, pad_y),
                original,
                suspect,
                config,
            )
            if found is not None:
                return found
        return None

    with StageTimer(logger, "describe_text"):
        if len(boxes) <= 1:
            return [read(box) for box in boxes]
        with ThreadPoolExecutor(max_workers=max(1, min(_MAX_READERS, os.cpu_count() or 2))) as pool:
            return list(pool.map(read, boxes))
