"""Barcode / QR symbol decoding and comparison (Architecture §3.3.5, FR-10.1/10.2).

Barcode mode reads every code on both files -- linear barcodes (EAN, UPC, Code 128, Code 39, ITF, GS1 DataBar...),
Pharmacode, and the two-dimensional ones (QR, Data Matrix, PDF417, Aztec) -- by DECODING them, and compares them by
what they say, exactly: a code matches only if both files' codes decode to the same value, as the same symbology.

* **Reading.** ZXing (zxing-cpp) and ZBar (pyzbar) are both run and their readings merged, because each reads
  symbologies and print qualities the other misses; Pharmacode, which neither implements, is read by our own decoder
  (`pharmacode.py`; zxing-cpp's `PZN` is a Pharmazentralnummer carried in Code 39, a different thing). A code a decoder
  finds but cannot decode (too small, too soft) is looked at again, closer -- its own window enlarged, thresholded other
  ways -- and if it still cannot be read it is listed as found but not decoded, never silently left out.
* **Matching.** The same code (the same value, of the same kind) is matched wherever it is on the other file -- a code
  moved on the layout is still the same code, and said to have moved; what is left is paired by where it is (the value
  differs), then a code that is the only one of its kind in each file with the other's; what has no partner is missing
  from the suspect, or extra in it.
* **Comparing.** Value and symbology both. Where they differ, the finding says how: which characters, or -- a GS1 code
  (a GTIN with its batch, expiry and serial) -- which of its fields.

Decoding only happens in Barcode mode; in every other mode these regions are compared as pixels like any other artwork
(FR-12.1), and Scan vs Digital compares a scan's barcodes as marks (`compare_physical`).
"""

from __future__ import annotations

import logging
import re
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
from app.models.result import PHARMACODE_TYPE, SymbolComparison, symbol_group

logger = logging.getLogger(__name__)


@dataclass
class SymbolReading:
    symbol_type: str
    value: str | None  # None: the code was found but could not be decoded
    bbox: BoundingBox


def _gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _zxing_readings(gray: np.ndarray, binarizer=None, scale: float = 1.0, origin: tuple[int, int] = (0, 0)) -> list[SymbolReading]:
    """ZXing's readings of `gray` -- found-but-undecoded codes too (value None) -- in the full picture's pixels (`gray`
    may be a window of it at `origin`, enlarged by `scale`)."""
    options = {"return_errors": True}
    if binarizer is not None:
        options["binarizer"] = binarizer
    readings = []
    for result in zxingcpp.read_barcodes(gray, **options):
        position = result.position
        corners = [position.top_left, position.top_right, position.bottom_right, position.bottom_left]
        xs = [origin[0] + float(corner.x) / scale for corner in corners]
        ys = [origin[1] + float(corner.y) / scale for corner in corners]
        value = result.text if result.valid else None
        readings.append(SymbolReading(result.format.name, value, BoundingBox(min(xs), min(ys), max(xs), max(ys))))
    return readings


def _zbar_readings(gray: np.ndarray) -> list[SymbolReading]:
    readings = []
    for result in pyzbar.decode(gray):
        rect = result.rect
        readings.append(
            SymbolReading(
                symbol_type=str(result.type),
                value=result.data.decode("utf-8", errors="replace"),
                bbox=BoundingBox(float(rect.left), float(rect.top), float(rect.left + rect.width), float(rect.top + rect.height)),
            )
        )
    return readings


_CLOSER_SCALES = (2.0, 3.0)  # a code found but not decoded is read again enlarged this much...
_CLOSER_BINARIZERS = (None, "GlobalHistogram", "FixedThreshold")  # ...and thresholded these ways
_CLOSER_PAD = 0.3  # (with this share of its size round it, its quiet zone)


def _read_closer(gray: np.ndarray, reading: SymbolReading) -> SymbolReading | None:
    """A code found but not decoded, read again in its own window, enlarged and thresholded other ways."""
    box = reading.bbox
    pad = int(round(_CLOSER_PAD * max(box.width, box.height))) + 8
    x0, y0, x1, y1 = padded_window(box, pad, gray.shape)
    window = gray[y0:y1, x0:x1]
    if window.size == 0:
        return None
    for scale in _CLOSER_SCALES:
        enlarged = cv2.resize(window, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        for name in _CLOSER_BINARIZERS:
            binarizer = getattr(zxingcpp.Binarizer, name) if name else None
            for found in _zxing_readings(enlarged, binarizer, scale, (x0, y0)):
                if found.value is not None:
                    return found
    return None


_UNDECODED_MAX_SHARE = 0.15  # a code found but not decoded is listed only if no bigger than this share of the page...
_UNDECODED_MIN_PX = 15  # ...no smaller than this...
_UNDECODED_MAX_ASPECT = 3.0  # ...and not long and thin (a Data Matrix may be oblong, not a strip): a code, not a pattern


def _like_a_code(reading: SymbolReading, shape: tuple[int, ...]) -> bool:
    """Whether something a decoder found but could not decode looks like a code at all: a job sheet's grid of tick boxes
    was 'found' as a Data Matrix half the page wide."""
    box = reading.bbox
    longest, shortest = max(box.width, box.height), min(box.width, box.height)
    return (
        shortest >= _UNDECODED_MIN_PX
        and longest <= _UNDECODED_MAX_SHARE * max(shape[:2])
        and longest <= _UNDECODED_MAX_ASPECT * shortest
    )


def decode_symbols(image: np.ndarray, config: DetectionConfig | None = None) -> list[SymbolReading]:
    """Every code any decoder can read (and every one found that none could, where it looks like a code), each once."""
    gray = _gray(image)
    readings: list[SymbolReading] = []
    for reading in _zxing_readings(gray):
        if reading.value is None:
            if not _like_a_code(reading, gray.shape):
                continue
            reading = _read_closer(gray, reading) or reading
        readings.append(reading)
    for reading in _zbar_readings(gray):
        same = [existing for existing in readings if existing.bbox.iou(reading.bbox) > 0.3]
        if any(existing.value == reading.value for existing in same):
            continue
        undecoded = [existing for existing in same if existing.value is None]
        if undecoded:  # (ZBar read what ZXing could only find)
            readings[readings.index(undecoded[0])] = reading
        elif not same:
            readings.append(reading)
    # the same code read twice (ZXing's look at a downscaled copy, a found-and-read pair): once
    unique: list[SymbolReading] = []
    for reading in readings:
        twin = next((u for u in unique if u.bbox.iou(reading.bbox) > 0.5 and (u.value == reading.value or None in (u.value, reading.value))), None)
        if twin is None:
            unique.append(reading)
        elif twin.value is None and reading.value is not None:
            unique[unique.index(twin)] = reading
    # Pharmacode last: a bar symbol another decoder already read is not also a Pharmacode.
    for code in pharmacode.find_pharmacodes(image, config or DetectionConfig()):
        if not any(existing.bbox.iou(code.bbox) > 0.05 for existing in unique):
            unique.append(SymbolReading(PHARMACODE_TYPE, str(code.value), code.bbox))
    return unique


def _distance(a: SymbolReading, b: SymbolReading) -> float:
    return float(np.hypot(a.bbox.center[0] - b.bbox.center[0], a.bbox.center[1] - b.bbox.center[1]))


def _match(
    original_readings: list[SymbolReading],
    suspect_readings: list[SymbolReading],
    config: DetectionConfig,
) -> list[tuple[SymbolReading | None, SymbolReading | None]]:
    """Pair the two files' codes: the same code (value and kind) wherever it is, nearest first; then by where they are
    (within `symbol_match_tolerance_px`); then the only code of its kind in each file; the rest unpaired."""
    free_original, free_suspect = list(original_readings), list(suspect_readings)
    pairs: list[tuple[SymbolReading | None, SymbolReading | None]] = []

    def take(rule) -> None:
        candidates = [
            (_distance(o, s), i, j)
            for i, o in enumerate(free_original)
            for j, s in enumerate(free_suspect)
            if rule(o, s)
        ]
        used_o, used_s = set(), set()
        for _d, i, j in sorted(candidates):
            if i in used_o or j in used_s:
                continue
            used_o.add(i)
            used_s.add(j)
            pairs.append((free_original[i], free_suspect[j]))
        free_original[:] = [o for i, o in enumerate(free_original) if i not in used_o]
        free_suspect[:] = [s for j, s in enumerate(free_suspect) if j not in used_s]

    take(lambda o, s: o.value is not None and o.value == s.value and symbol_group(o.symbol_type) == symbol_group(s.symbol_type))
    take(lambda o, s: _distance(o, s) <= config.symbol_match_tolerance_px)
    groups_o = [symbol_group(o.symbol_type) for o in free_original]
    groups_s = [symbol_group(s.symbol_type) for s in free_suspect]
    take(lambda o, s: symbol_group(o.symbol_type) == symbol_group(s.symbol_type) and groups_o.count(symbol_group(o.symbol_type)) == 1 and groups_s.count(symbol_group(s.symbol_type)) == 1)
    pairs.extend((o, None) for o in free_original)
    pairs.extend((None, s) for s in free_suspect)
    return pairs


_GS1_FIELD = re.compile(r"\((\d{2,4})\)([^(]*)")
_GS1_NAMES = {
    "00": "SSCC", "01": "GTIN", "02": "GTIN of contents", "10": "batch/lot", "11": "production date", "15": "best before",
    "17": "expiry", "21": "serial number", "30": "count", "240": "additional ID", "710": "NHRN (DE)", "711": "NHRN (FR)",
    "712": "NHRN (ES)", "713": "NHRN (BR)", "714": "NHRN (PT)",
}


def _gs1_fields(value: str) -> dict[str, str] | None:
    """A GS1 element string's fields by Application Identifier ('(01)...(17)...'), or None if it is not one."""
    fields = {ai: data for ai, data in _GS1_FIELD.findall(value)}
    return fields if fields and value.startswith("(") else None


def describe_difference(original: str, suspect: str) -> str:
    """How two decoded values differ: a GS1 code by its fields, anything else by the characters that differ."""
    mine, theirs = _gs1_fields(original), _gs1_fields(suspect)
    if mine is not None and theirs is not None:
        changes = []
        for ai in list(mine) + [ai for ai in theirs if ai not in mine]:
            if mine.get(ai) != theirs.get(ai):
                name = _GS1_NAMES.get(ai, "field")
                changes.append(f"({ai}) {name}: {mine.get(ai, 'none')} → {theirs.get(ai, 'none')}")
        if changes:
            return "; ".join(changes)
    if len(original) == len(suspect):
        places = [i for i, (a, b) in enumerate(zip(original, suspect)) if a != b]
        if len(places) <= 4:
            return ", ".join(f"character {i + 1}: '{original[i]}' → '{suspect[i]}'" for i in places)
    return f"{len(original)} characters in the original, {len(suspect)} in the suspect"


def compare(
    original: np.ndarray, suspect: np.ndarray, config: DetectionConfig
) -> tuple[list[DiffCandidate], list[SymbolComparison]]:
    """Decode both files' codes and compare them exactly (FR-10.1/10.2): every code is listed, matched or not."""
    with StageTimer(logger, "detector_barcode"):
        original_readings = decode_symbols(original, config)
        suspect_readings = decode_symbols(suspect, config)

        candidates: list[DiffCandidate] = []
        comparisons: list[SymbolComparison] = []
        for original_reading, suspect_reading in _match(original_readings, suspect_readings, config):
            moved = (
                _distance(original_reading, suspect_reading)
                if original_reading is not None and suspect_reading is not None
                else 0.0
            )
            comparison = SymbolComparison(
                symbol_type=(original_reading or suspect_reading).symbol_type,
                original_value=original_reading.value if original_reading else None,
                suspect_value=suspect_reading.value if suspect_reading else None,
                bbox=(
                    original_reading.bbox.union(suspect_reading.bbox)
                    if original_reading and suspect_reading and moved <= config.symbol_match_tolerance_px
                    else (original_reading or suspect_reading).bbox
                ),
                suspect_type=suspect_reading.symbol_type if suspect_reading else None,
                original_found=original_reading is not None,
                suspect_found=suspect_reading is not None,
                moved_px=moved if moved > config.symbol_match_tolerance_px else 0.0,
                suspect_bbox=suspect_reading.bbox if suspect_reading else None,
            )
            comparisons.append(comparison)
            if comparison.passed:
                continue

            if original_reading is None:
                bbox, subtype = suspect_reading.bbox, DiffSubtype.ADDED
            elif suspect_reading is None:
                bbox, subtype = original_reading.bbox, DiffSubtype.REMOVED
            else:
                bbox, subtype = comparison.bbox, DiffSubtype.CHANGED
            candidates.append(
                DiffCandidate(
                    bbox=bbox,
                    type=DiffType.BARCODE,
                    subtype=subtype,
                    confidence=1.0,  # a decoded value either matches or it doesn't
                    detail=comparison.finding,
                    moved_to=suspect_reading.bbox if suspect_reading is not None and moved > config.symbol_match_tolerance_px else None,
                )
            )

    logger.info(
        "barcode detector finished",
        extra={
            "original_symbols": len(original_readings),
            "suspect_symbols": len(suspect_readings),
            "undecoded": sum(r.value is None for r in original_readings + suspect_readings),
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
