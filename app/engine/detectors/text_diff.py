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
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np
import pytesseract

from app.config import DetectionConfig
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


def _ocr_words(image: np.ndarray, config: DetectionConfig) -> list[OcrWord]:
    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError as exc:
        raise OcrUnavailableError(
            "Tesseract OCR is not installed or not on PATH, so text and number "
            "differences could not be checked. Install Tesseract and restart."
        ) from exc

    words = []
    for index, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        confidence = float(data["conf"][index])
        if not text or confidence < config.ocr_min_confidence:
            continue
        left, top = float(data["left"][index]), float(data["top"][index])
        words.append(
            OcrWord(
                text=text,
                bbox=BoundingBox(
                    left, top, left + float(data["width"][index]), top + float(data["height"][index])
                ),
                confidence=confidence,
            )
        )
    return words


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
