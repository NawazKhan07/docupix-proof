"""Result objects produced by the comparison pipeline (Architecture §3.4).

Phase 3 fills in the alignment half; `ComparisonResult` (diff clusters,
barcode/Braille results) joins it in Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


def _same_symbology(first: str, second: str) -> bool:
    """Whether two decoders' names are one symbology ('QRCode' and 'QRCODE', 'Code128' and 'CODE128')."""
    return "".join(c for c in first.lower() if c.isalnum()) == "".join(c for c in second.lower() if c.isalnum())


@dataclass
class SymbolComparison:
    """One code matched original-to-suspect (FR-10.2): what each file's decodes to, compared exactly."""

    symbol_type: str
    original_value: str | None  # None when the original has no such code, or has it but it could not be decoded
    suspect_value: str | None  # None when the suspect has no such code, or has it but it could not be decoded
    # Where it is, in the shared aligned space, so a result can be shown in place.
    bbox: BoundingBox | None = None
    suspect_type: str | None = None  # the suspect's symbology, where it has the code (None: as `symbol_type`)
    original_found: bool | None = None  # whether the file has the code at all (None: as its value says)
    suspect_found: bool | None = None
    moved_px: float = 0.0  # the same code, somewhere else on the suspect: how far
    suspect_bbox: BoundingBox | None = None

    @property
    def group(self) -> str:
        return symbol_group(self.symbol_type)

    @property
    def in_original(self) -> bool:
        return self.original_value is not None if self.original_found is None else self.original_found

    @property
    def in_suspect(self) -> bool:
        return self.suspect_value is not None if self.suspect_found is None else self.suspect_found

    @property
    def same_symbology(self) -> bool:
        return self.suspect_type is None or _same_symbology(self.symbol_type, self.suspect_type)

    @property
    def passed(self) -> bool:
        return (
            self.original_value is not None
            and self.suspect_value is not None
            and self.original_value == self.suspect_value
            and self.same_symbology
        )

    @property
    def status(self) -> str:
        if self.passed:
            return f"match (moved {self.moved_px:.0f} px)" if self.moved_px else "match"
        if not self.in_suspect:
            return "missing from suspect"
        if not self.in_original:
            return "extra in suspect"
        if self.original_value is None and self.suspect_value is None:
            return "could not be decoded in either file"
        if self.original_value is None:
            return "could not be decoded in the original"
        if self.suspect_value is None:
            return "could not be decoded in the suspect"
        if not self.same_symbology:
            return "symbology differs"
        return "value differs"

    @property
    def finding(self) -> str:
        """The finding's text: what it is, and how the two files' codes differ."""
        name = self.group if self.group.lower() == self.symbol_type.lower() else f"{self.group} ({self.symbol_type})"
        if not self.same_symbology and self.in_suspect:
            name = f"{self.group} ({self.symbol_type} in the original, {self.suspect_type} in the suspect)"
        text = f"{name}: {self.status}"
        if self.original_value is not None and self.suspect_value is not None and self.original_value != self.suspect_value:
            from app.engine.detectors.barcode_diff import describe_difference

            text += f": '{self.original_value}' → '{self.suspect_value}' ({describe_difference(self.original_value, self.suspect_value)})"
        elif self.original_value is not None and not self.in_suspect:
            text += f": '{self.original_value}'"
        elif self.suspect_value is not None and not self.in_original:
            text += f": '{self.suspect_value}'"
        if self.moved_px and not self.passed:
            text += f"; moved {self.moved_px:.0f} px"
        return text


@dataclass
class BrailleCellComparison:
    """One Braille cell, original against suspect: its dots in each (a pattern: dot n is bit n - 1, as Unicode numbers
    them; None where that file has no cell there) and what each says."""

    original_pattern: int | None
    suspect_pattern: int | None
    original_meaning: str
    suspect_meaning: str
    bbox: BoundingBox
    status: str  # "match" | "changed" | "missing" (from the suspect) | "added" (in the suspect) | "moved"

    @property
    def passed(self) -> bool:
        return self.status == "match"


@dataclass
class BrailleLineComparison:
    """One line of Braille, original against suspect, cell by cell (None: that file has no such line)."""

    original_text: str | None
    suspect_text: str | None
    cells: list[BrailleCellComparison]
    bbox: BoundingBox | None = None

    @property
    def passed(self) -> bool:
        return self.original_text is not None and self.suspect_text is not None and all(cell.passed for cell in self.cells)


@dataclass
class BrailleComparison:
    """Braille compared (FR-11.1): read cell by cell where the dots are set to Braille's grid (`lines`), and dot by dot
    for any that are not; the dot counts cover both."""

    original_dot_count: int
    suspect_dot_count: int
    matched: int
    missing: int
    added: int
    moved: int
    region: BoundingBox | None = None  # where the Braille is (every dot of both files, in the original's frame)
    lines: list[BrailleLineComparison] = field(default_factory=list)

    @property
    def cells_compared(self) -> int:
        return sum(len(line.cells) for line in self.lines)

    @property
    def cells_differing(self) -> int:
        return sum(1 for line in self.lines for cell in line.cells if not cell.passed)

    @property
    def passed(self) -> bool:
        return self.missing == 0 and self.added == 0 and self.moved == 0 and self.cells_differing == 0


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
    # Where on the original the features that registered the two lie ((N, 2) pixels), if they did: the part of the
    # original the suspect shows (a scan of one label of a proof sheet matches only on that label).
    matched_points: np.ndarray | None = None

    def summary(self) -> str:
        """One-line human summary, used for the UI status strip and logs."""
        state = "LOW CONFIDENCE" if self.low_confidence else "OK"
        return (
            f"{state} — {self.residual_error_px:.2f} px residual "
            f"(threshold {self.threshold_px:.2f} px, basis: {self.residual_basis}, "
            f"method: {self.method}, {self.inlier_count}/{self.match_count} inliers)"
        )
