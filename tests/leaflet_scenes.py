"""Synthetic pages for the leaflet (multi-page) comparison: each page is its own section, with its own words, so
pages are genuinely different from one another (a leaflet's pages share a layout but not their vocabulary) -- drawn
with OpenCV only."""

from __future__ import annotations

import cv2
import numpy as np

PAGE_SIZE = (1200, 900)  # (height, width)

# Each section: (heading, banner colour, its own body lines -- no two sections share a sentence).
_SECTIONS = {
    "indications": (
        "INDICATIONS AND USAGE",
        (40, 90, 200),
        [
            "This medicine relieves fever and mild to moderate pain",
            "including headache toothache and muscular aches",
            "It is also used to reduce the discomfort of colds",
            "Suitable for adults and children over twelve years",
            "Always read the accompanying insert before taking it",
        ],
    ),
    "dosage": (
        "DOSAGE AND ADMINISTRATION",
        (90, 150, 40),
        [
            "Swallow one tablet whole with a full glass of water",
            "Do not exceed four doses within a twenty four hour period",
            "Leave at least four hours between successive doses",
            "Children under twelve should use the paediatric syrup",
            "Do not crush or chew the coated tablet before swallowing",
        ],
    ),
    "warnings": (
        "WARNINGS AND PRECAUTIONS",
        (200, 60, 40),
        [
            "Do not take this medicine if you are allergic to its ingredients",
            "Consult your physician before use if pregnant or breastfeeding",
            "Avoid alcohol while taking this medication regularly",
            "Stop use and seek medical advice if a rash develops",
            "Keep this and all medicines out of reach of children",
        ],
    ),
    "storage": (
        "STORAGE AND HANDLING",
        (150, 120, 30),
        [
            "Store below thirty degrees Celsius in a dry place",
            "Protect the blister strip from direct sunlight and moisture",
            "Keep the carton tightly closed when not in use",
            "Discard any tablet past its printed expiry date",
            "Return unused medicine to a pharmacy for safe disposal",
        ],
    ),
    "inserted": (
        "AN INSERTED PAGE",
        (150, 40, 150),
        [
            "This page was added between two existing sections",
            "Its wording has nothing in common with the others",
            "A completely different topic entirely appears here",
            "Nothing on this sheet echoes any neighbouring page",
            "Distinct vocabulary throughout, on purpose",
        ],
    ),
}


def page(section: str, edit=None) -> np.ndarray:
    """One page: a coloured heading band (`section`'s own colour) and that section's own body text."""
    height, width = PAGE_SIZE
    image = np.full((height, width, 3), 250, np.uint8)
    heading, colour, lines = _SECTIONS[section]
    cv2.rectangle(image, (0, 0), (width, 120), colour, -1)
    cv2.putText(image, heading, (60, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    for row in range(18):
        text = lines[row % len(lines)] + f" (paragraph {row})"
        cv2.putText(image, text, (60, 170 + row * 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    if edit is not None:
        edit(image)
    return image


def edit_first_line(image: np.ndarray) -> None:
    """Paint over and rewrite the first body line: a real, findable text change."""
    cv2.rectangle(image, (55, 145), (600, 185), (250, 250, 250), -1)
    cv2.putText(image, "This sentence was rewritten for the suspect", (60, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
