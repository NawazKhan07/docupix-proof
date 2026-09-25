"""Synthetic scenes for the series (multi-up) comparison: one small artwork, and a sheet printed with several
repeats of it, some altered. Nothing here is tuned to any real file -- generic packaging-like content, drawn with
OpenCV only (no font files needed)."""

from __future__ import annotations

import cv2
import numpy as np

PAPER = (250, 248, 244)
SHEET_BG = (230, 230, 225)
CELL_W, CELL_H = 360, 520


def artwork(defect: str | None = None) -> np.ndarray:
    """One carton face, ~360x520. `defect`: None (clean), 'text', 'colour' or 'graphic'."""
    image = np.full((CELL_H, CELL_W, 3), PAPER, np.uint8)
    cv2.rectangle(image, (4, 4), (CELL_W - 5, CELL_H - 5), (90, 90, 90), 3)
    cv2.putText(image, "PARACETAMOL", (24, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(image, "500 mg Tablets", (24, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.rectangle(image, (24, 130), (CELL_W - 24, 170), (20, 60, 150), -1)
    cv2.circle(image, (80, 250), 55, (200, 40, 40), -1)
    cv2.putText(image, "Batch AB1234", (24, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(image, "Exp 07/2028", (24, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
    if defect == "text":
        cv2.rectangle(image, (20, 385), (240, 420), PAPER, -1)
        cv2.putText(image, "Exp 09/2028", (24, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
    elif defect == "colour":
        cv2.rectangle(image, (24, 130), (CELL_W - 24, 170), (150, 20, 20), -1)
    elif defect == "graphic":
        cv2.circle(image, (80, 250), 55, PAPER, -1)
        cv2.rectangle(image, (40, 220), (120, 290), (40, 150, 60), -1)
    return image


def sheet(defects: list[str | None], cols: int = 3, gap: int = 40) -> np.ndarray:
    """A press sheet holding `len(defects)` repeats of the artwork, one per entry (None = clean)."""
    rows = (len(defects) + cols - 1) // cols
    width = cols * CELL_W + (cols + 1) * gap
    height = rows * CELL_H + (rows + 1) * gap
    canvas = np.full((height, width, 3), SHEET_BG, np.uint8)
    for i, defect in enumerate(defects):
        r, c = divmod(i, cols)
        y0, x0 = gap + r * (CELL_H + gap), gap + c * (CELL_W + gap)
        canvas[y0 : y0 + CELL_H, x0 : x0 + CELL_W] = artwork(defect)
    return canvas


def original_page(margin: int = 40) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """The original file: one clean artwork on a page with a margin round it. Returns (page, (x0, y0, x1, y1))."""
    art = artwork(None)
    page = np.full((CELL_H + 2 * margin, CELL_W + 2 * margin, 3), (235, 235, 230), np.uint8)
    page[margin : margin + CELL_H, margin : margin + CELL_W] = art
    return page, (margin, margin, margin + CELL_W, margin + CELL_H)
