"""What each kind of finding is called and coloured (no Qt: the engine, the viewer and any report share it).

A finding is DESCRIBED, not ranked: it has a category -- text, colour, graphics, barcode, Braille -- and, within it, what
happened (changed, moved, added, removed). Its box in both files, and its card in the list, are coloured by its kind, so a
reviewer sees what kind of difference a box is before reading anything -- except while the list is showing OCR or pixel
findings, when they are coloured by how the finding was made (read word for word, or found in the pictures).
"""

from __future__ import annotations

from app.models.diff import SOURCE_OCR, SOURCE_PAGE, SOURCE_PIXELS, DiffSubtype

CATEGORY_ORDER = ("text", "colour", "graphics", "barcode", "braille")

CATEGORY_LABELS = {
    "text": "Text",
    "colour": "Colour",
    "graphics": "Graphics",
    "barcode": "Barcode",
    "braille": "Braille",
}

# Distinct from one another, and from the reds, blues and oranges that pack artwork is printed in; every box also sits on a
# white halo, so it shows over a colour of its own.
CATEGORY_COLOURS = {
    "text": "#7c3aed",  # violet
    "colour": "#e11d48",  # rose
    "graphics": "#f59e0b",  # amber
    "barcode": "#2563eb",  # blue
    "braille": "#0d9488",  # teal
}

SUBTYPE_LABELS = {
    DiffSubtype.CHANGED: "changed",
    DiffSubtype.MOVED: "moved",
    DiffSubtype.ADDED: "added",
    DiffSubtype.REMOVED: "removed",
}


# How a finding was made, which is what its box and card are coloured by in every mode: read and compared word for word
# (OCR), or found where the pictures differ (pixels). A whole page missing or added (Leaflet) is neither.
SOURCE_COLOURS = {  # (none of them a kind's colour: the two schemes are never confused)
    SOURCE_OCR: "#c026d3",  # fuchsia
    SOURCE_PIXELS: "#16a34a",  # green
    SOURCE_PAGE: "#6b7280",  # grey
}
SOURCE_SHORT_LABELS = {SOURCE_OCR: "OCR", SOURCE_PIXELS: "Pixel", SOURCE_PAGE: "Page"}


def source_of(cluster) -> str:
    """How a finding was made, as one of `SOURCE_COLOURS`: OCR only where it was found by reading alone (a merge of an
    OCR finding with a pixel one counts as pixels: its box is where the pictures differ)."""
    sources = cluster.sources
    if SOURCE_PAGE in sources:
        return SOURCE_PAGE
    if sources == {SOURCE_OCR}:
        return SOURCE_OCR
    return SOURCE_PIXELS


def finding_colour(cluster, by_source: bool = False) -> str:
    """The colour of a finding's box in the two files and of its card: its kind's (text, colour, graphics...), or --
    while the list is showing OCR or pixel findings (`by_source`) -- how it was made (`source_of`)."""
    return SOURCE_COLOURS[source_of(cluster)] if by_source else category_colour(cluster.category)


def category_colour(category: str) -> str:
    return CATEGORY_COLOURS.get(category, "#6b7280")
