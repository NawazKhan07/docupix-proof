"""How each viewer mode turns the shared aligned image pair into pixels
(SRS §3.8, FR-8.1-8.4).

Every mode is a re-rendering of the *same* pair -- the original, and the
suspect already warped into the original's frame -- never an independent
re-alignment. That is what guarantees a difference sits at the same place in
every view (FR-7.4). This module is pure numpy with no Qt, so the modes can be
tested without a display.
"""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np

from app.models.diff import BoundingBox


class ViewMode(Enum):
    SIDE_BY_SIDE = "side_by_side"  # FR-8.1
    FLICKER = "flicker"  # FR-8.2
    OVERLAY = "overlay"  # FR-8.3
    GRAYSCALE = "grayscale"  # FR-8.4


VIEW_MODE_LABELS = {
    ViewMode.SIDE_BY_SIDE: "Side by Side",
    ViewMode.FLICKER: "Flicker",
    ViewMode.OVERLAY: "Overlay",
    ViewMode.GRAYSCALE: "Grayscale",
}

# The modes that draw a single image rather than a pair of panes.
SINGLE_PANE_MODES = frozenset({ViewMode.FLICKER, ViewMode.OVERLAY, ViewMode.GRAYSCALE})
# The modes whose picture changes over time.
ANIMATED_MODES = frozenset({ViewMode.FLICKER, ViewMode.GRAYSCALE})


def compose_flicker(original: np.ndarray, suspect: np.ndarray, phase: int) -> np.ndarray:
    """Alternate the two files at the same screen position (FR-8.2)."""
    return original if phase % 2 == 0 else suspect


def compose_overlay(original: np.ndarray, suspect: np.ndarray, opacity: float = 0.5) -> np.ndarray:
    """Superimpose the two files (FR-8.3).

    A straight alpha blend: content both files share stays crisp, while
    anything that differs shows as ghosting.
    """
    return cv2.addWeighted(original, 1.0 - opacity, suspect, opacity, 0.0)


def compose_grayscale(
    original: np.ndarray,
    suspect: np.ndarray,
    difference_boxes: list[BoundingBox],
    phase: int,
) -> np.ndarray:
    """Black-and-white everywhere except where a difference lives (FR-8.4).

    Unchanged areas are rendered in grayscale; each area containing a
    difference keeps its colour and alternates between the two files, so the
    difference flickers while everything around it stays still.
    """
    grayscale = cv2.cvtColor(cv2.cvtColor(original, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    source = original if phase % 2 == 0 else suspect

    height, width = grayscale.shape[:2]
    for box in difference_boxes:
        x0, y0 = max(0, int(box.x0)), max(0, int(box.y0))
        x1, y1 = min(width, int(np.ceil(box.x1))), min(height, int(np.ceil(box.y1)))
        if x1 > x0 and y1 > y0:
            grayscale[y0:y1, x0:x1] = source[y0:y1, x0:x1]
    return grayscale
