from __future__ import annotations

from app.engine.aggregation.severity import (
    DEFAULT_RULES,
    SEVERITY_COLOURS,
    SeverityRule,
    severity_for,
    worst_severity,
)
from app.models.diff import BoundingBox, DiffCandidate, DiffSubtype, DiffType, Severity


def _candidate(
    diff_type: DiffType,
    subtype: DiffSubtype = DiffSubtype.CHANGED,
    magnitude: float = 0.0,
    size: float = 20.0,
) -> DiffCandidate:
    return DiffCandidate(
        bbox=BoundingBox(0, 0, size, size),
        type=diff_type,
        subtype=subtype,
        confidence=1.0,
        magnitude=magnitude,
    )


def test_any_barcode_mismatch_is_critical():
    """Architecture §3.3.6: "any barcode/pharmacode mismatch = Critical"."""
    for subtype in DiffSubtype:
        assert severity_for(_candidate(DiffType.BARCODE, subtype)) is Severity.CRITICAL


def test_braille_differences_are_critical():
    assert severity_for(_candidate(DiffType.BRAILLE, DiffSubtype.REMOVED)) is Severity.CRITICAL


def test_a_changed_number_is_critical_but_a_moved_one_is_major():
    assert severity_for(_candidate(DiffType.NUMBER, DiffSubtype.CHANGED)) is Severity.CRITICAL
    assert severity_for(_candidate(DiffType.NUMBER, DiffSubtype.MOVED)) is Severity.MAJOR


def test_changed_added_and_removed_text_are_all_critical():
    for subtype in (DiffSubtype.CHANGED, DiffSubtype.ADDED, DiffSubtype.REMOVED):
        assert severity_for(_candidate(DiffType.TEXT, subtype)) is Severity.CRITICAL


def test_colour_severity_follows_delta_e():
    """Architecture: "colour dE above a threshold = Critical"."""
    assert severity_for(_candidate(DiffType.COLOUR, magnitude=25.0)) is Severity.CRITICAL
    assert severity_for(_candidate(DiffType.COLOUR, magnitude=10.0)) is Severity.CRITICAL  # inclusive
    assert severity_for(_candidate(DiffType.COLOUR, magnitude=6.0)) is Severity.MAJOR
    assert severity_for(_candidate(DiffType.COLOUR, magnitude=3.0)) is Severity.MINOR


def test_a_small_graphics_change_is_minor_and_a_large_one_is_major():
    """Architecture: "minor sub-pixel graphic shift = Low"."""
    assert severity_for(_candidate(DiffType.GRAPHICS, size=20)) is Severity.MINOR
    assert severity_for(_candidate(DiffType.GRAPHICS, size=200)) is Severity.MAJOR


def test_a_graphic_that_appears_or_disappears_is_major_whatever_its_size():
    assert severity_for(_candidate(DiffType.GRAPHICS, DiffSubtype.REMOVED, size=10)) is Severity.MAJOR


def test_first_matching_rule_wins():
    table = (
        SeverityRule(Severity.MINOR, types=frozenset({DiffType.TEXT})),
        SeverityRule(Severity.CRITICAL, types=frozenset({DiffType.TEXT})),
    )
    assert severity_for(_candidate(DiffType.TEXT), table) is Severity.MINOR


def test_the_rule_table_can_be_swapped_without_touching_logic():
    """The build prompt asks for a rule table that is easy to edit."""
    lenient = (SeverityRule(Severity.MINOR),)
    assert severity_for(_candidate(DiffType.BARCODE), lenient) is Severity.MINOR


def test_an_incomplete_table_fails_safe_to_critical():
    """A finding no rule covers must never be quietly downgraded."""
    table = (SeverityRule(Severity.MINOR, types=frozenset({DiffType.COLOUR})),)
    assert severity_for(_candidate(DiffType.BARCODE), table) is Severity.CRITICAL


def test_the_default_table_ends_in_a_catch_all():
    assert DEFAULT_RULES[-1].types is None and DEFAULT_RULES[-1].subtypes is None


def test_a_cluster_is_as_severe_as_its_worst_member():
    members = [
        _candidate(DiffType.COLOUR, magnitude=3.0),  # minor
        _candidate(DiffType.COLOUR, magnitude=6.0),  # major
        _candidate(DiffType.COLOUR, magnitude=3.0),  # minor
    ]
    assert worst_severity(members) is Severity.MAJOR


def test_every_tier_has_a_distinct_colour():
    """FR-7.2: each severity is rendered in a distinct colour."""
    assert set(SEVERITY_COLOURS) == set(Severity)
    assert len(set(SEVERITY_COLOURS.values())) == len(Severity)


def test_severity_rank_orders_most_severe_first():
    assert sorted(Severity, key=lambda s: s.rank) == [
        Severity.CRITICAL,
        Severity.MAJOR,
        Severity.MINOR,
    ]
