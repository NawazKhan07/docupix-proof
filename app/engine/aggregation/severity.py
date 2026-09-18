"""Severity assignment (SRS FR-7.2, Architecture §3.3.6).

Everything a QA reviewer might want to change lives in this one file: the
tier colours, and `DEFAULT_RULES`, an ordered table where the FIRST rule that
matches a difference decides its severity. Adding, reordering or retuning a
tier is an edit to that table, not to any logic.

PENDING SIGN-OFF (SRS §7): the SRS says only "Critical and others" and
leaves the final tier list, definitions and colours open. Everything below
is a reasoned placeholder for QA to correct, not a validated policy. The
starting position is deliberately conservative for a regulated product:
anything that changes what the packaging *says* (text, numbers, barcodes,
Braille) is Critical, because a wrong dose or batch code is the failure this
tool exists to catch.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.models.diff import DiffCandidate, DiffSubtype, DiffType, Severity

# One colour per tier, for the findings list and the viewer's overlays (FR-7.2).
SEVERITY_COLOURS: dict[Severity, str] = {
    Severity.CRITICAL: "#d92d20",
    Severity.MAJOR: "#f79009",
    Severity.MINOR: "#2f6fed",
}

# Colour difference (CIEDE2000) at which a change reads as a different colour
# rather than a shade: ~2.3 is just noticeable, ~5 is plainly visible, and
# ~10 is red-versus-orange territory.
COLOUR_CRITICAL_DELTA_E = 10.0
COLOUR_MAJOR_DELTA_E = 5.0

# A graphics change smaller than this (in original-resolution pixels, ~4 mm
# square at 300 DPI) is reported as Minor rather than Major.
GRAPHICS_MAJOR_MIN_AREA_PX = 2500.0

_CONTENT_CHANGES = frozenset({DiffSubtype.CHANGED, DiffSubtype.ADDED, DiffSubtype.REMOVED})
_TEXT_TYPES = frozenset({DiffType.TEXT, DiffType.NUMBER})


@dataclass(frozen=True)
class SeverityRule:
    """One row of the table. A field left as None means "don't care"."""

    severity: Severity
    types: frozenset[DiffType] | None = None
    subtypes: frozenset[DiffSubtype] | None = None
    min_magnitude: float | None = None  # inclusive; colour: peak dE, graphics: 1 - SSIM
    min_area_px: float | None = None  # inclusive; box area in the shared space

    def matches(self, candidate: DiffCandidate) -> bool:
        if self.types is not None and candidate.type not in self.types:
            return False
        if self.subtypes is not None and candidate.subtype not in self.subtypes:
            return False
        if self.min_magnitude is not None and candidate.magnitude < self.min_magnitude:
            return False
        if self.min_area_px is not None and candidate.bbox.area < self.min_area_px:
            return False
        return True


DEFAULT_RULES: tuple[SeverityRule, ...] = (
    # Symbols and Braille: a wrong decoded value or a missing dot changes the
    # meaning of the pack, whatever the size of the marking.
    SeverityRule(Severity.CRITICAL, types=frozenset({DiffType.BARCODE})),
    SeverityRule(Severity.CRITICAL, types=frozenset({DiffType.BRAILLE})),
    # Text and numbers: content changes are Critical (a dose, batch or expiry
    # is exactly what must not slip through); a pure move is Major, since
    # regulated text often has to sit in a defined place.
    SeverityRule(Severity.CRITICAL, types=_TEXT_TYPES, subtypes=_CONTENT_CHANGES),
    SeverityRule(Severity.MAJOR, types=_TEXT_TYPES, subtypes=frozenset({DiffSubtype.MOVED})),
    # Colour, by how far it moved (Architecture: "colour dE above a threshold
    # = Critical").
    SeverityRule(
        Severity.CRITICAL,
        types=frozenset({DiffType.COLOUR}),
        min_magnitude=COLOUR_CRITICAL_DELTA_E,
    ),
    SeverityRule(
        Severity.MAJOR, types=frozenset({DiffType.COLOUR}), min_magnitude=COLOUR_MAJOR_DELTA_E
    ),
    SeverityRule(Severity.MINOR, types=frozenset({DiffType.COLOUR})),
    # Graphics: a small shift is Minor (Architecture: "minor sub-pixel graphic
    # shift = Low"); a logo or panel appearing/disappearing is Major.
    SeverityRule(Severity.MINOR, types=frozenset({DiffType.GRAPHICS}), subtypes=frozenset({DiffSubtype.MOVED})),
    SeverityRule(
        Severity.MAJOR,
        types=frozenset({DiffType.GRAPHICS}),
        subtypes=frozenset({DiffSubtype.ADDED, DiffSubtype.REMOVED}),
    ),
    SeverityRule(
        Severity.MAJOR, types=frozenset({DiffType.GRAPHICS}), min_area_px=GRAPHICS_MAJOR_MIN_AREA_PX
    ),
    SeverityRule(Severity.MINOR, types=frozenset({DiffType.GRAPHICS})),
    # Catch-all, so a difference type added later is never silently unranked.
    SeverityRule(Severity.MAJOR),
)


def severity_for(
    candidate: DiffCandidate, rules: tuple[SeverityRule, ...] = DEFAULT_RULES
) -> Severity:
    """The severity of one candidate: the first matching rule wins.

    If an edited table matches nothing, this fails safe to Critical rather
    than quietly downgrading a finding the rule author forgot to cover.
    """
    for rule in rules:
        if rule.matches(candidate):
            return rule.severity
    return Severity.CRITICAL


def worst_severity(
    candidates: list[DiffCandidate], rules: tuple[SeverityRule, ...] = DEFAULT_RULES
) -> Severity:
    """A cluster is as severe as its most severe member."""
    return min((severity_for(c, rules) for c in candidates), key=lambda severity: severity.rank)
