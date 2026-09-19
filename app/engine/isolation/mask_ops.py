"""Small mask operations shared by the ways of lifting a subject."""

from __future__ import annotations

import cv2
import numpy as np


class NoSubjectFoundError(Exception):
    """Nothing could be lifted at the press point."""


def disc_kernel(radius: int) -> np.ndarray:
    size = 2 * max(0, int(radius)) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def disc(shape: tuple[int, int], centre: tuple[float, float], radius: float) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.circle(mask, (int(round(centre[0])), int(round(centre[1]))), max(1, int(round(radius))), 1, -1)
    return mask.astype(bool)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """The silhouette: an outline, unlike a segmentation, has no holes."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    return filled.astype(bool)


def component_at(mask: np.ndarray, seed: tuple[float, float], seed_radius: float) -> np.ndarray:
    """The connected part of `mask` at the press point (the thing pressed on, not its neighbours)."""
    _, components = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    x, y = int(round(seed[0])), int(round(seed[1]))
    height, width = mask.shape
    label = components[min(max(y, 0), height - 1), min(max(x, 0), width - 1)]
    if label == 0:  # the press landed on something judged background: take what is nearest under the disc
        under = components[disc(mask.shape, seed, seed_radius)]
        under = under[under > 0]
        if under.size == 0:
            raise NoSubjectFoundError("Nothing to lift at that spot. Press and hold on the artwork itself.")
        label = np.bincount(under).argmax()
    return components == label


def touches_edge(mask: np.ndarray) -> bool:
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
