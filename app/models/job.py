"""Core job data model (Architecture Document §3.4).

`options` (Phase 5 clustering/severity settings) is filled in by a later
phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from app.models.isolation import IsolationGeometry


class ComparisonMode(Enum):
    AUTOMATIC = "automatic"
    SCAN_VS_DIGITAL = "scan_vs_digital"
    BARCODE = "barcode"
    BRAILLE = "braille"
    SERIES = "series"
    LEAFLET = "leaflet"


MODE_DISPLAY_NAMES: dict[ComparisonMode, str] = {
    ComparisonMode.AUTOMATIC: "Automatic",
    ComparisonMode.SCAN_VS_DIGITAL: "Scan vs Digital",
    ComparisonMode.BARCODE: "Barcode / Pharmacode / QR",
    ComparisonMode.BRAILLE: "Braille",
    ComparisonMode.SERIES: "Series (repeated artwork)",
    ComparisonMode.LEAFLET: "Leaflet (multi-page)",
}


@dataclass(frozen=True)
class DetectorOptions:
    """What a given mode actually compares (SRS §2.3, FR-6.3, FR-10.1-12.1).

    The mode is turned into this parameter object once, here, rather than
    being re-interpreted by each detector (Architecture §3.1).
    """

    compare_color: bool
    compare_text: bool
    compare_graphics: bool
    decode_symbols: bool
    compare_braille_dots: bool
    normalize_scan: bool


_MODE_OPTIONS = {
    # Everything, but barcode/Braille regions are only compared as pixels
    # here -- decoding happens solely in their own modes (FR-12.1).
    ComparisonMode.AUTOMATIC: DetectorOptions(
        compare_color=True,
        compare_text=True,
        compare_graphics=True,
        decode_symbols=False,
        compare_braille_dots=False,
        normalize_scan=False,
    ),
    # FR-6.3: colour is excluded outright (scanning shifts colour), and both
    # files are flattened to black-on-white first.
    ComparisonMode.SCAN_VS_DIGITAL: DetectorOptions(
        compare_color=False,
        compare_text=True,
        compare_graphics=True,
        decode_symbols=False,
        compare_braille_dots=False,
        normalize_scan=True,
    ),
    # FR-10.1: no other artwork content is compared in this mode.
    ComparisonMode.BARCODE: DetectorOptions(
        compare_color=False,
        compare_text=False,
        compare_graphics=False,
        decode_symbols=True,
        compare_braille_dots=False,
        normalize_scan=False,
    ),
    # FR-11.1: restricted to the coloured dot matrix.
    ComparisonMode.BRAILLE: DetectorOptions(
        compare_color=False,
        compare_text=False,
        compare_graphics=False,
        decode_symbols=False,
        compare_braille_dots=True,
        normalize_scan=False,
    ),
    # Every repeat is compared exactly as Automatic mode compares a single pair (`app/engine/series.py` is what finds
    # the repeats and runs one such comparison per repeat; this is only what each of those comparisons checks).
    ComparisonMode.SERIES: DetectorOptions(
        compare_color=True,
        compare_text=True,
        compare_graphics=True,
        decode_symbols=False,
        compare_braille_dots=False,
        normalize_scan=False,
    ),
    # Every page is compared exactly as Automatic mode compares a single pair (`app/engine/leaflet.py` is what pairs
    # the pages and runs one such comparison per pair; this is only what each of those comparisons checks).
    ComparisonMode.LEAFLET: DetectorOptions(
        compare_color=True,
        compare_text=True,
        compare_graphics=True,
        decode_symbols=False,
        compare_braille_dots=False,
        normalize_scan=False,
    ),
}


def options_for_mode(mode: ComparisonMode) -> DetectorOptions:
    return _MODE_OPTIONS[mode]


@dataclass
class ComparisonJob:
    """A single original-vs-suspect comparison request.

    `original_geometry`/`suspect_geometry` are independent (FR-3.1: the user
    selects the comparison area separately per file) and are None until the
    user (or, from Phase 3, automatic dieline detection) resolves one.
    """

    original_path: Path
    suspect_path: Path
    mode: ComparisonMode = ComparisonMode.AUTOMATIC
    original_geometry: Optional[IsolationGeometry] = None
    suspect_geometry: Optional[IsolationGeometry] = None
    options: dict[str, Any] = field(default_factory=dict)
