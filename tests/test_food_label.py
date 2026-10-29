"""Food label comparison (`app.engine.food_label`): a label compared by the things printed on it -- one finding per
thing, said for what happened to it -- with every bar measured on the pair."""

from __future__ import annotations

import cv2
import numpy as np

from app.config import AppConfig
from app.engine.food_label import compare_label
from app.models.diff import DiffSubtype

CONFIG = AppConfig().detection
RED, GREY, WHITE = (200, 40, 50), (200, 200, 200), (255, 255, 255)


def _photo(rng: np.random.Generator, size: tuple[int, int]) -> np.ndarray:
    """A photograph-like patch: smooth, continuous tones (fries on a plate), not flat colour."""
    h, w = size
    noise = rng.normal(0, 1, (h // 3 + 2, w // 3 + 2, 3)).astype(np.float32)
    smooth = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    smooth = cv2.GaussianBlur(smooth, (0, 0), 1.5)
    smooth = (smooth - smooth.min()) / (np.ptp(smooth) + 1e-6)
    base = np.array([230, 180, 60], np.float32)
    return np.clip(base * (0.55 + 0.45 * smooth), 0, 255).astype(np.uint8)


def _label(logo_at=(120, 90), panel=GREY, sticker=False, seed=3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    page = np.full((900, 1400, 3), RED, np.uint8)
    cv2.rectangle(page, (900, 60), (1340, 840), panel, -1)  # a side panel
    for row in range(10):
        cv2.putText(page, f"Ingredients line {row}: potato and palm oil", (920, 120 + row * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2, cv2.LINE_AA)
    x, y = logo_at
    cv2.rectangle(page, (x, y), (x + 260, y + 140), WHITE, -1)  # a logo: white badge, red lettering
    cv2.putText(page, "ALLANA", (x + 20, y + 95), cv2.FONT_HERSHEY_SIMPLEX, 1.8, RED, 6, cv2.LINE_AA)
    page[450:800, 200:700] = _photo(rng, (350, 500))  # a photograph
    if sticker:
        cv2.ellipse(page, (560, 560), (110, 55), -15, 0, 360, WHITE, -1)  # a sticker laid on the photograph
        cv2.ellipse(page, (560, 560), (110, 55), -15, 0, 360, (20, 20, 20), 4)
        cv2.putText(page, "SHOESTRING", (470, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    return page


def _compare(original: np.ndarray, suspect: np.ndarray):
    compared = np.ones(original.shape[:2], bool)
    return compare_label(original, suspect, compared, CONFIG).clusters


def test_the_same_label_has_nothing_to_say():
    assert _compare(_label(), _label()) == []


def test_a_label_rendered_a_little_softer_has_nothing_to_say():
    """A proof prints softer: how this pair differs where nothing changed is what the bar is measured on."""
    soft = cv2.GaussianBlur(_label(), (0, 0), 0.7)
    assert _compare(_label(), soft) == []


def test_a_panel_recoloured_is_one_colour_finding_named():
    found = _compare(_label(), _label(panel=WHITE))
    colours = [c for c in found if c.category == "colour"]
    assert len(colours) == 1 and "light grey" in colours[0].detail and "white" in colours[0].detail
    box = colours[0].bbox_in_original
    assert box.x0 >= 880 and box.x1 <= 1360 and box.height > 500  # the panel, one box


def test_a_logo_moved_is_one_finding_shown_where_it_went():
    found = _compare(_label(), _label(logo_at=(160, 130)))
    moved = [c for c in found if c.subtype is DiffSubtype.MOVED]
    assert len(moved) == 1 and "40 px right" in moved[0].detail and "40 px down" in moved[0].detail
    assert moved[0].moved_to is not None and abs(moved[0].moved_to.x0 - moved[0].bbox_in_original.x0 - 40) <= 4
    assert len(found) == 1  # (not a finding at each of its edges)


def test_a_sticker_laid_on_the_photograph_is_a_finding_of_its_own():
    found = _compare(_label(), _label(sticker=True))
    added = [c for c in found if c.subtype is DiffSubtype.ADDED]
    assert added, [(c.detail, c.bbox_in_original) for c in found]
    sticker = min(added, key=lambda c: abs(c.bbox_in_original.center[0] - 560) + abs(c.bbox_in_original.center[1] - 560))
    assert abs(sticker.bbox_in_original.center[0] - 560) < 60 and abs(sticker.bbox_in_original.center[1] - 560) < 60
    assert sticker.bbox_in_original.width < 320  # the sticker, not the whole photograph
