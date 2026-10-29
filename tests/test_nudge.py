"""Letters only nudged (an open document against its outlined, "closed" copy) are not a change; letters changed are
(`app.engine.detectors.nudge`)."""

from __future__ import annotations

import random

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from app.engine.detectors.nudge import only_nudged
from app.models.diff import BoundingBox, DiffCandidate, DiffSubtype, DiffType

_SUPERSAMPLE = 4


def _line(text: str, size_px: int, stretch: float = 1.0, jitter: float = 0.0, shift=(0.0, 0.0), seed: int = 0, colour=30):
    """One line of type on paper, as an outlining step re-draws it: its letters' spacing stretched a little, each letter
    a fraction of a pixel off, the whole moved (all in pixels of the result)."""
    rng = random.Random(seed)
    font = ImageFont.load_default(size_px * _SUPERSAMPLE)
    width, height = int(font.getlength(text) / _SUPERSAMPLE * 1.1) + 60, int(size_px * 1.8)
    big = Image.new("RGB", (width * _SUPERSAMPLE, height * _SUPERSAMPLE), (255, 255, 255))
    draw = ImageDraw.Draw(big)
    x = 0.0
    for character in text:
        dx = (30 + x * stretch / _SUPERSAMPLE + rng.uniform(-jitter, jitter) + shift[0]) * _SUPERSAMPLE
        draw.text((dx, (size_px * 0.3 + shift[1]) * _SUPERSAMPLE), character, font=font, fill=(colour, colour, colour))
        x += font.getlength(character)
    return cv2.resize(np.array(big), (width, height), interpolation=cv2.INTER_AREA)


def _judged(original: np.ndarray, suspect: np.ndarray) -> bool:
    height = max(original.shape[0], suspect.shape[0])
    width = max(original.shape[1], suspect.shape[1])
    canvas = [np.full((height + 80, width + 80, 3), 255, np.uint8) for _ in range(2)]
    canvas[0][40 : 40 + original.shape[0], 40 : 40 + original.shape[1]] = original
    canvas[1][40 : 40 + suspect.shape[0], 40 : 40 + suspect.shape[1]] = suspect
    box = BoundingBox(60, 45, 40 + width - 20, 40 + height - 5)
    return only_nudged(canvas[0], canvas[1], DiffCandidate(box, DiffType.TEXT, DiffSubtype.CHANGED, 1.0), 3.0, 2.3)


_WORDS = ["dizziness", "treatment", "hyperkalaemia", "0.5 mg/kg", "1,250 mg", "Lisinopril 20", "Amlodipine besilate"]


@pytest.mark.parametrize("size_px", [26, 34])
def test_letters_only_nudged_and_respaced_are_not_a_change(size_px):
    for seed, word in enumerate(_WORDS):
        rng = random.Random(seed)
        original = _line(word, size_px)
        suspect = _line(word, size_px, stretch=1 + rng.uniform(-0.02, 0.02), jitter=0.3, shift=(rng.uniform(-1.5, 1.5), rng.uniform(-0.5, 0.5)), seed=seed)
        assert _judged(original, suspect), word


@pytest.mark.parametrize("size_px", [26, 34])
def test_a_letter_changed_is_still_a_change_however_the_rest_moved(size_px):
    changes = [
        ("treatment", "treatrnent"), ("recently", "recehtly"), ("recently", "reeently"), ("1,250 mg", "1.250 mg"),
        ("0.35 mg", "0.36 mg"), ("0.35 mg", "0.85 mg"), ("Lisinopril 10", "Lisinopril 70"), ("tablets", "tab1ets"),
        ("nightly", "nighfly"), ("20 mg", "20 rng"), ("each", "eaeh"), ("hyperkalaemia", "hyperkalaemla"),
    ]
    for seed, (word, changed) in enumerate(changes):
        rng = random.Random(seed)
        original = _line(word, size_px)
        suspect = _line(changed, size_px, stretch=1 + rng.uniform(-0.02, 0.02), jitter=0.3, shift=(rng.uniform(-1.5, 1.5), 0.0), seed=seed)
        assert not _judged(original, suspect), f"{word} -> {changed}"


def test_a_word_moved_further_than_a_nudge_is_a_change():
    original = _line("dizziness", 30)
    assert not _judged(original, _line("dizziness", 30, shift=(6.0, 0.0)))


def test_text_recoloured_is_a_change():
    original = _line("dizziness", 30)
    assert not _judged(original, _line("dizziness", 30, shift=(0.5, 0.0), colour=150))


def test_a_colour_rendered_a_shade_apart_is_still_only_nudged():
    # an outlined copy's ink is often converted a shade off (a spot magenta made process: dE ~2), invisibly
    original = _line("dizziness", 30, colour=30)
    assert _judged(original, _line("dizziness", 30, shift=(0.7, 0.3), colour=33))


def test_something_printed_in_one_file_only_is_a_change():
    original = _line("dizziness", 30)
    blank = np.full_like(original, 255)
    assert not _judged(original, blank)
