"""Result objects produced by the comparison pipeline (Architecture §3.4).

Phase 3 fills in the alignment half; `ComparisonResult` (diff clusters,
barcode/Braille results) joins it in Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.models.diff import BoundingBox

PHARMACODE_TYPE = "Pharmacode"
_QR_FAMILY = ("qr", "datamatrix", "aztec", "pdf417", "maxicode")


def symbol_group(symbol_type: str) -> str:
    """The group a symbol is listed under (FR-10.2): Barcode, Pharmacode or QR code.

    Decoders name formats their own way ("QRCode", "QRCODE", "EAN13", "CODE128");
    two-dimensional matrix codes are listed with QR codes, everything else read
    as bars is a barcode.
    """
    lowered = symbol_type.lower()
    if lowered == PHARMACODE_TYPE.lower():
        return "Pharmacode"
    if any(family in lowered for family in _QR_FAMILY):
        return "QR code"
    return "Barcode"


@dataclass
class SymbolComparison:
    """One decoded symbol matched original-to-suspect (FR-10.2)."""

    symbol_type: str
    original_value: str | None  # None when the symbol only exists in the suspect
    suspect_value: str | None  # None when the symbol is missing from the suspect
    # Where it is, in the shared aligned space, so a result can be shown in place.
    bbox: BoundingBox | None = None

    @property
    def group(self) -> str:
        return symbol_group(self.symbol_type)

    @property
    def passed(self) -> bool:
        return (
            self.original_value is not None
            and self.suspect_value is not None
            and self.original_value == self.suspect_value
        )

    @property
    def status(self) -> str:
        if self.passed:
            return "match"
        if self.suspect_value is None:
            return "missing from suspect"
        if self.original_value is None:
            return "extra in suspect"
        return "value differs"


@dataclass
class BrailleComparison:
    """Dot-pattern comparison summary (FR-11.1)."""

    original_dot_count: int
    suspect_dot_count: int
    matched: int
    missing: int
    added: int
    moved: int

    @property
    def passed(self) -> bool:
        return self.missing == 0 and self.added == 0 and self.moved == 0


@dataclass
class AlignmentResult:
    """How the suspect was registered onto the original, and how well.

    `transform` maps suspect image pixel coordinates -> original image pixel
    coordinates. That original-image space is the shared aligned space every
    later stage projects diffs back out of (Architecture §3.3.6), so there is
    only ever one transform in play (FR-7.4).
    """

    transform: np.ndarray  # 3x3 homogeneous
    residual_error_px: float
    residual_basis: str  # "keypoints" | "polygon" -- what the number measured
    method: str  # "dieline_polygon" | "dieline_polygon+features" | "features"
    match_count: int
    inlier_count: int
    inlier_ratio: float
    low_confidence: bool
    threshold_px: float

    def summary(self) -> str:
        """One-line human summary, used for the UI status strip and logs."""
        state = "LOW CONFIDENCE" if self.low_confidence else "OK"
        return (
            f"{state} — {self.residual_error_px:.2f} px residual "
            f"(threshold {self.threshold_px:.2f} px, basis: {self.residual_basis}, "
            f"method: {self.method}, {self.inlier_count}/{self.match_count} inliers)"
        )
