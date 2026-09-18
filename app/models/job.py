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


MODE_DISPLAY_NAMES: dict[ComparisonMode, str] = {
    ComparisonMode.AUTOMATIC: "Automatic",
    ComparisonMode.SCAN_VS_DIGITAL: "Scan vs Digital",
    ComparisonMode.BARCODE: "Barcode / Pharmacode / QR",
    ComparisonMode.BRAILLE: "Braille",
}


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
