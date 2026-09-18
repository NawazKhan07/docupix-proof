"""Core job data model (Architecture Document §3.4).

Only the pieces Phase 1 needs (mode + file identity) are fleshed out here;
`isolation_geometry` (Phase 2 crop/dieline-polygon/erase) and `options`
(Phase 5 clustering/severity settings) are filled in by later phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


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
    """A single original-vs-suspect comparison request."""

    original_path: Path
    suspect_path: Path
    mode: ComparisonMode = ComparisonMode.AUTOMATIC
    isolation_geometry: Optional[Any] = None
    options: dict[str, Any] = field(default_factory=dict)
