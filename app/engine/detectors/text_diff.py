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
from dataclasses import dataclass
from difflib import SequenceMatcher

import cv2
import numpy as np
import pytesseract

from app.config import DetectionConfig
from app.engine.aggregation.severity import LIKELY_MISREAD_CONFIDENCE
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
        original_words = _ocr_words(original, config)
        suspect_words = _ocr_words(suspect, config)

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
            elif tag == "replace":
                candidates.append(
                    _candidate(
                        _span_bbox(original_span + suspect_span),
                        _span_type([w.text for w in original_span + suspect_span]),
                        DiffSubtype.CHANGED,
                        _mean_confidence(original_span + suspect_span),
                        f"'{_join(original_span)}' -> '{_join(suspect_span)}'",
                    )
                )
            elif tag == "delete":
                candidates.append(
                    _candidate(
                        _span_bbox(original_span),
                        _span_type([w.text for w in original_span]),
                        DiffSubtype.REMOVED,
                        _mean_confidence(original_span),
                        f"'{_join(original_span)}' missing from suspect",
                    )
                )
            elif tag == "insert":
                candidates.append(
                    _candidate(
                        _span_bbox(suspect_span),
                        _span_type([w.text for w in suspect_span]),
                        DiffSubtype.ADDED,
                        _mean_confidence(suspect_span),
                        f"'{_join(suspect_span)}' added in suspect",
                    )
                )

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
) -> DiffCandidate | None:
    if bbox is None:
        return None
    return DiffCandidate(
        bbox=bbox, type=diff_type, subtype=subtype, confidence=confidence, detail=detail
    )


# ---------------------------------------------------------------------------
# Comparing a scan with a digital file
# ---------------------------------------------------------------------------

# An l, a 1 and an i are one smudge apart in a soft scan, and so are an O and a 0.
_OCR_CONFUSABLE = str.maketrans({"l": "i", "1": "i", "0": "o"})
_MIN_ALNUM_SHARE = 0.5  # a "word" that is mostly marks is not one
_MAX_LEFTOVER_SPREAD = 8.0  # words this scattered (box area / area of the words) are separate findings
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


def _findings(gone: list[OcrWord], new: list[OcrWord]) -> list[DiffCandidate]:
    """Findings for the words left over in one replaced stretch."""
    if gone and new:
        both = gone + new
        box = _span_bbox(both)
        spread = box.area / max(1.0, sum(word.bbox.area for word in both))
        if spread <= _MAX_LEFTOVER_SPREAD:
            return [
                _candidate(
                    box,
                    _span_type([word.text for word in both]),
                    DiffSubtype.CHANGED,
                    _mean_confidence(both),
                    f"'{_join(gone)}' -> '{_join(new)}'",
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


def _read_window(image: np.ndarray, box: BoundingBox, config: DetectionConfig) -> list[OcrWord]:
    """The words a small picture of `box` alone reads as, in the whole picture's pixels."""
    x0, y0, x1, y1 = _padded(box, config.text_line_pad_px, image.shape)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return []
    crop = image[y0:y1, x0:x1]
    factor = max(1.0, _WINDOW_UPSCALE_BELOW_PX / crop.shape[0])
    if factor > 1.0:
        crop = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    words = []
    for _line, word in _words_of(_ocr_data(crop, _LINE_PAGE_SEGMENTATION), config.ocr_min_confidence):
        b = word.bbox
        words.append(
            OcrWord(word.text, BoundingBox(b.x0 / factor + x0, b.y0 / factor + y0, b.x1 / factor + x0, b.y1 / factor + y0), word.confidence)
        )
    words.sort(key=lambda w: w.bbox.x0)
    return _trustworthy(words, config)


def detect_scan(original: np.ndarray, suspect: np.ndarray, config: DetectionConfig) -> TextScanResult:
    """Text differences between a digital file and a scan of it, both as ink pictures in the shared frame.

    Anchored on the digital file: it is read as lines, and each line's window is read again, alone, in both
    files (so a changed word cannot reshuffle the reading order the way a whole-page comparison lets it).
    See `DetectionConfig.ocr_min_confidence_scan` for what is ignored and why.
    """
    with StageTimer(logger, "detector_text"):
        page = _ocr_data(original)
        lines: dict[tuple[int, int, int], list[OcrWord]] = {}
        for line, word in _words_of(page, config.ocr_min_confidence_scan):
            lines.setdefault(line, []).append(word)
        anchors = [_trustworthy(sorted(words, key=lambda w: w.bbox.x0), config) for words in lines.values()]
        anchors = [words for words in anchors if words and (len(words) > 1 or len(canonical(words[0].text)) >= 4)]

        candidates: list[DiffCandidate] = []
        unreadable = 0
        windows = []
        for anchor in anchors:
            box = _span_bbox(anchor)
            windows.append(box)
            digital_reading = _read_window(original, box, config) or anchor
            scan_reading = _read_window(suspect, box, config)
            if not scan_reading:
                if _ink_in(suspect, box) < config.text_removed_ink_share * _ink_in(original, box):
                    candidates.append(
                        _candidate(
                            box,
                            _span_type([w.text for w in digital_reading]),
                            DiffSubtype.REMOVED,
                            _mean_confidence(digital_reading),
                            f"'{_join(digital_reading)}' missing from suspect",
                        )
                    )
                else:
                    unreadable += 1  # the ink is there, but too soft to read: say so, do not call it a change
                continue
            candidates += _line_differences(digital_reading, scan_reading)

        candidates += _added_text(original, suspect, windows, config)
        candidates = [_marked_if_misread(candidate, original, suspect, config) for candidate in candidates]

    logger.info(
        "text detector finished (scan)",
        extra={"lines": len(anchors), "candidates": len(candidates), "unreadable_lines": unreadable},
    )
    return TextScanResult(candidates, unreadable)


def _added_text(original: np.ndarray, suspect: np.ndarray, windows: list[BoundingBox], config: DetectionConfig) -> list[DiffCandidate]:
    """Words the scan has, read surely, outside every line of the digital file, where the digital has no ink."""
    pad = config.text_line_pad_px
    found = []
    for _line, word in _words_of(_ocr_data(suspect), config.text_added_confidence):
        if not _trustworthy([word], config) or len(canonical(word.text)) < 3:
            continue
        cx, cy = word.bbox.center
        if any(w.x0 - pad <= cx <= w.x1 + pad and w.y0 - pad <= cy <= w.y1 + pad for w in windows):
            continue
        if _ink_in(original, word.bbox, 3) < config.text_added_ink_share * _ink_in(suspect, word.bbox, 3):
            found.append(
                _candidate(word.bbox, _span_type([word.text]), DiffSubtype.ADDED, word.confidence / 100, f"'{word.text}' added in suspect")
            )
    return [candidate for candidate in found if candidate is not None]


def ink_agreement(original: np.ndarray, suspect: np.ndarray, box: BoundingBox) -> float:
    """How closely the two files' ink in `box` coincides: the smaller of the share of each one's ink lying within a
    couple of pixels of the other's. 1.0 is the same marks (a scan's fatter strokes still give ~1.0); 0.0 if either
    has none there."""
    x0, y0, x1, y1 = _padded(box, _AGREEMENT_PAD_PX, original.shape)
    a = original[y0:y1, x0:x1] < _DARK
    b = suspect[y0:y1, x0:x1] < _DARK
    if not a.any() or not b.any():
        return 0.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * _AGREEMENT_SLACK_PX + 1, 2 * _AGREEMENT_SLACK_PX + 1))
    a_in_b = (a & cv2.dilate(b.astype(np.uint8), kernel).astype(bool)).sum() / a.sum()
    b_in_a = (b & cv2.dilate(a.astype(np.uint8), kernel).astype(bool)).sum() / b.sum()
    return float(min(a_in_b, b_in_a))


def _marked_if_misread(candidate: DiffCandidate, original: np.ndarray, suspect: np.ndarray, config: DetectionConfig) -> DiffCandidate:
    """A word the scan reads differently although its ink is the digital word's: most likely a misread."""
    if ink_agreement(original, suspect, candidate.bbox) < config.text_misread_ink_agreement:
        return candidate
    return DiffCandidate(
        bbox=candidate.bbox,
        type=candidate.type,
        subtype=candidate.subtype,
        confidence=min(candidate.confidence, LIKELY_MISREAD_CONFIDENCE),
        detail=f"{candidate.detail} (the ink matches: likely a misread of the scan)",
        magnitude=candidate.magnitude,
    )
