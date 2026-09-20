"""Barcode / QR symbol decoding and comparison (Architecture §3.3.5, FR-10.1/10.2).

Both ZBar (pyzbar) and ZXing (zxing-cpp) are run and their readings merged,
because each reads symbologies and print qualities the other misses.
Decoding only happens in Barcode mode; in every other mode these regions are
compared as pixels like any other artwork (FR-12.1).

Pharmacode (Laetus) is read by our own decoder (pharmacode.py). The Architecture
Document expected zxing-cpp to cover "some pharmacode variants", but neither
library implements it (zxing-cpp's `PZN` is a Pharmazentralnummer carried in
Code39, which is a different thing).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import zxingcpp
from pyzbar import pyzbar

from app.config import DetectionConfig
from app.logging_config import StageTimer
from app.models.diff import BoundingBox, DiffCandidate, DiffSubtype, DiffType
from app.engine.detectors import pharmacode, structure_diff
from app.engine.detectors.regions import padded_window
from app.models.result import PHARMACODE_TYPE, SymbolComparison

logger = logging.getLogger(__name__)


@dataclass
class SymbolReading:
    symbol_type: str
    value: str
    bbox: BoundingBox


def _zxing_readings(image: np.ndarray) -> list[SymbolReading]:
    readings = []
    for result in zxingcpp.read_barcodes(image):
        position = result.position
        corners = [position.top_left, position.top_right, position.bottom_right, position.bottom_left]
        xs = [float(corner.x) for corner in corners]
        ys = [float(corner.y) for corner in corners]
        readings.append(
            SymbolReading(
                symbol_type=result.format.name,
                value=result.text,
                bbox=BoundingBox(min(xs), min(ys), max(xs), max(ys)),
            )
        )
    return readings


def _zbar_readings(image: np.ndarray) -> list[SymbolReading]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    readings = []
    for result in pyzbar.decode(gray):
        rect = result.rect
        readings.append(
            SymbolReading(
                symbol_type=str(result.type),
                value=result.data.decode("utf-8", errors="replace"),
                bbox=BoundingBox(
                    float(rect.left),
                    float(rect.top),
                    float(rect.left + rect.width),
                    float(rect.top + rect.height),
                ),
            )
        )
    return readings


def decode_symbols(image: np.ndarray, config: DetectionConfig | None = None) -> list[SymbolReading]:
    """All symbols any decoder can read, de-duplicated."""
    readings = _zxing_readings(image)
    for reading in _zbar_readings(image):
        already_found = any(
            existing.value == reading.value and existing.bbox.iou(reading.bbox) > 0.3
            for existing in readings
        )
        if not already_found:
            readings.append(reading)

    # Pharmacode last: a bar symbol another decoder already read is not also a Pharmacode.
    for code in pharmacode.find_pharmacodes(image, config or DetectionConfig()):
        if not any(existing.bbox.iou(code.bbox) > 0.05 for existing in readings):
            readings.append(SymbolReading(PHARMACODE_TYPE, str(code.value), code.bbox))
    return readings


def _match(
    original_readings: list[SymbolReading],
    suspect_readings: list[SymbolReading],
    config: DetectionConfig,
) -> list[tuple[SymbolReading | None, SymbolReading | None]]:
    """Pair symbols by position in the shared aligned space, nearest first."""
    unmatched_suspects = list(suspect_readings)
    pairs: list[tuple[SymbolReading | None, SymbolReading | None]] = []

    for original in original_readings:
        original_center = original.bbox.center
        candidates = [
            (
                np.hypot(
                    suspect.bbox.center[0] - original_center[0],
                    suspect.bbox.center[1] - original_center[1],
                ),
                suspect,
            )
            for suspect in unmatched_suspects
        ]
        candidates = [
            (distance, suspect)
            for distance, suspect in candidates
            if distance <= config.symbol_match_tolerance_px
        ]
        if candidates:
            _distance, best = min(candidates, key=lambda item: item[0])
            unmatched_suspects.remove(best)
            pairs.append((original, best))
        else:
            pairs.append((original, None))

    pairs.extend((None, suspect) for suspect in unmatched_suspects)
    return pairs


def compare(
    original: np.ndarray, suspect: np.ndarray, config: DetectionConfig
) -> tuple[list[DiffCandidate], list[SymbolComparison]]:
    """Decode both files' symbols and compare their values (FR-10.1/10.2)."""
    with StageTimer(logger, "detector_barcode"):
        original_readings = decode_symbols(original, config)
        suspect_readings = decode_symbols(suspect, config)

        candidates: list[DiffCandidate] = []
        comparisons: list[SymbolComparison] = []
        for original_reading, suspect_reading in _match(original_readings, suspect_readings, config):
            comparison = SymbolComparison(
                symbol_type=(original_reading or suspect_reading).symbol_type,
                original_value=original_reading.value if original_reading else None,
                suspect_value=suspect_reading.value if suspect_reading else None,
                bbox=(
                    original_reading.bbox.union(suspect_reading.bbox)
                    if original_reading and suspect_reading
                    else (original_reading or suspect_reading).bbox
                ),
            )
            comparisons.append(comparison)
            if comparison.passed:
                continue

            if original_reading is None:
                bbox, subtype = suspect_reading.bbox, DiffSubtype.ADDED
            elif suspect_reading is None:
                bbox, subtype = original_reading.bbox, DiffSubtype.REMOVED
            else:
                bbox = original_reading.bbox.union(suspect_reading.bbox)
                subtype = DiffSubtype.CHANGED
            candidates.append(
                DiffCandidate(
                    bbox=bbox,
                    type=DiffType.BARCODE,
                    subtype=subtype,
                    confidence=1.0,  # a decoded value either matches or it doesn't
                    detail=f"{comparison.symbol_type}: {comparison.status}",
                )
            )

    logger.info(
        "barcode detector finished",
        extra={
            "original_symbols": len(original_readings),
            "suspect_symbols": len(suspect_readings),
            "mismatches": len(candidates),
        },
    )
    return candidates, comparisons


_MARKS_PAD_PX = 24  # what is compared round a barcode, so a bar at its very edge is inside
_MARKS_ECC_ITERATIONS = 60
_MARKS_ECC_EPSILON = 1e-5
_MARKS_MIN_CORRELATION = 0.5  # below this the scan does not have this barcode (it is blank, or another code): not moved onto it


def _registered(original: np.ndarray, suspect: np.ndarray) -> np.ndarray:
    """The scan's crop moved onto the digital's by the affine transform that best lines their marks up.

    A scan of a carton is bent by its fold lines and never has the sheet's scale exactly: over one barcode the bars
    drift apart by a few pixels, more than any tolerance that would still see a missing bar. So each barcode is
    registered on its own (the enhanced correlation coefficient, on blurred greys), unless the scan does not have it.
    """
    a = cv2.GaussianBlur(cv2.cvtColor(original, cv2.COLOR_RGB2GRAY), (0, 0), 1.0).astype(np.float32)
    b = cv2.GaussianBlur(cv2.cvtColor(suspect, cv2.COLOR_RGB2GRAY), (0, 0), 1.0).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, _MARKS_ECC_ITERATIONS, _MARKS_ECC_EPSILON)
    try:
        correlation, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_AFFINE, criteria, None, 5)
    except cv2.error:
        return suspect
    if correlation < _MARKS_MIN_CORRELATION:
        return suspect
    height, width = suspect.shape[:2]
    return cv2.warpAffine(
        suspect, warp, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE
    )


def compare_physical(
    original: np.ndarray, suspect: np.ndarray, config: DetectionConfig, slack_px: int
) -> list[DiffCandidate]:
    """Where the digital file has a barcode, whether the scan has the same bars: compared as marks, not decoded.

    A scan of a barcode rarely reads back the same (a soft, fat print), and Scan vs Digital does not decode. The
    digital file's barcodes are found (they are crisp), and the ink in each one's box is compared with the scan's
    where the files are registered, so a missing bar or a different one is a finding at the barcode.
    """
    found: list[DiffCandidate] = []
    with StageTimer(logger, "barcode_marks"):
        for reading in decode_symbols(original, config):
            x0, y0, x1, y1 = padded_window(reading.bbox, _MARKS_PAD_PX, original.shape)
            result = structure_diff.detect(
                original[y0:y1, x0:x1], _registered(original[y0:y1, x0:x1], suspect[y0:y1, x0:x1]), config, slack_px
            )
            if not result.candidates:
                continue
            box = result.candidates[0].bbox
            for candidate in result.candidates[1:]:
                box = box.union(candidate.bbox)
            found.append(
                DiffCandidate(
                    bbox=BoundingBox(box.x0 + x0, box.y0 + y0, box.x1 + x0, box.y1 + y0),
                    type=DiffType.BARCODE,
                    subtype=DiffSubtype.CHANGED,
                    confidence=0.9,
                    detail=f"{reading.symbol_type} bars differ from the digital file (compared as marks, not decoded)",
                    magnitude=float(sum(candidate.magnitude for candidate in result.candidates)),
                )
            )
    logger.info("barcode marks compared", extra={"differing": len(found)})
    return found
