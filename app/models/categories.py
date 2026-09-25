"""What each kind of finding is called and coloured (no Qt: the engine, the viewer and any report share it).

A finding is DESCRIBED, not ranked: it has a category -- text, colour, graphics, barcode, Braille -- and, within it, what
happened (changed, moved, added, removed). The category has one colour, used for its box in both files and its mark in the
list, so a reviewer sees what kind of difference a box is before reading anything.
"""

from __future__ import annotations

from app.models.diff import DiffSubtype

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


def category_label(category: str) -> str:
    return CATEGORY_LABELS.get(category, category.title())


def category_colour(category: str) -> str:
    return CATEGORY_COLOURS.get(category, "#6b7280")
