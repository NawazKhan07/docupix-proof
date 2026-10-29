"""Marks that only moved a hair: the same letters, nudged.

A document turned from live text into outlines (an "open" file made "closed" for print) keeps every letter, but re-draws
each where the conversion put it: a word comes out a pixel or so off, and its letters drift apart or together along the
line by up to a pixel or two (their spacing is re-computed, ~1% across a word). The whole page, and each panel, is
aligned far better than that on average, but no single shift lines up every word, so the pixel comparison sees a halo
round the letters and flags them, though nothing on them changed.

Each line of such a difference is fitted on its own -- each word, where the line's words drifted their own ways -- with
one slight stretch and shift (its middle moved no more than `tolerance_px`, plus `_RESPACING_SHARE` of its length, and a
stretch of no more than `_MAX_STRETCH`): drift like that is gone after it, down to edges a
fraction of a pixel apart, which differ by well under the line's own contrast (measured: at most 0.66 of it, at every
size from small print up). A letter changed, added or taken away -- even 'm' made 'rn', or a comma a full stop -- leaves
a pixel differing by at least 0.77 of it, as does anything re-coloured more than the colour comparison lets pass. The difference is dropped
only when every one of its lines fits like that.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.engine.detectors.colour_math import ciede2000, rgb_to_lab
from app.engine.detectors.regions import padded_window
from app.models.diff import DiffCandidate

_WINDOW_PAD_PX = 8  # each difference is judged with at least this much above and below it...
_WORD_REACH_PX = 240  # ...and this much either side, so a word it cuts through is seen whole
_BAND_PAD_PX = 3  # each line is fitted with this much paper round it
_BAND_GAP_PX = 1  # ink rows no further apart than this are one line
_MAX_STRETCH = 0.04  # a line may be fitted stretched or squeezed by up to this share (drift in its letter spacing)
_MAX_SHEAR = 0.02
_RESPACING_SHARE = 0.01  # a line's middle may move by the tolerance plus this share of its length (its spacing drifted)
_RESIDUE_SHARE = 0.72  # what is left after the fit must differ by less than this share of the line's own contrast
_MIN_CONTRAST = 40  # a line with less contrast than this (a faint tint) is not judged here
_EDGE_PX = 2  # a fitted line's own border (where the shift brings in what was outside it) is not judged
_ECC_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)


def _lines(ink: np.ndarray) -> list[tuple[int, int]]:
    """The (top, bottom) rows of each line of ink in a window, bottom exclusive."""
    rows = np.flatnonzero(ink.any(axis=1))
    if rows.size == 0:
        return []
    bands = [[int(rows[0]), int(rows[0]) + 1]]
    for row in rows[1:]:
        if row - bands[-1][1] <= _BAND_GAP_PX:
            bands[-1][1] = int(row) + 1
        else:
            bands.append([int(row), int(row) + 1])
    return [(top, bottom) for top, bottom in bands]


def _fits(first: np.ndarray, second: np.ndarray, tolerance_px: float, same_colour_delta_e: float) -> bool:
    """Whether `second` is `first` moved and stretched a little (see above): both are one line, same size, colour."""
    gray_first = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY).astype(np.float32) if first.ndim == 3 else first.astype(np.float32)
    gray_second = cv2.cvtColor(second, cv2.COLOR_RGB2GRAY).astype(np.float32) if second.ndim == 3 else second.astype(np.float32)
    height, width = gray_first.shape
    if height <= 2 * _EDGE_PX + 1 or width <= 2 * _EDGE_PX + 1:
        return False
    pooled = np.concatenate([first.reshape(-1, first.shape[-1] if first.ndim == 3 else 1), second.reshape(-1, second.shape[-1] if second.ndim == 3 else 1)])
    low, high = np.percentile(pooled, (1, 99), axis=0)
    contrast = float(np.max(high - low))
    if contrast < _MIN_CONTRAST:
        return False
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _score, warp = cv2.findTransformECC(gray_first, gray_second, warp, cv2.MOTION_AFFINE, _ECC_CRITERIA, None, 3)
    except cv2.error:
        return False  # did not settle: not a simple move
    (a, b, tx), (c, d, ty) = warp
    if abs(a - 1) > _MAX_STRETCH or abs(d - 1) > _MAX_STRETCH or abs(b) > _MAX_SHEAR or abs(c) > _MAX_SHEAR:
        return False
    centre = np.array([width / 2, height / 2, 1], np.float32)
    if np.abs(warp @ centre - centre[:2]).max() > tolerance_px + _RESPACING_SHARE * width:
        return False  # moved further than a nudge
    placed = cv2.warpAffine(
        second.astype(np.float32), warp, (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE
    )
    residue = np.abs(first.astype(np.float32) - placed)
    if residue.ndim == 3:
        residue = residue.max(axis=2)
    residue = residue[_EDGE_PX:-_EDGE_PX, _EDGE_PX:-_EDGE_PX]
    if float(residue.max()) >= _RESIDUE_SHARE * contrast:
        return False
    # and in the same colours: the ink at its darkest, and the paper, each no further from the other file's than the
    # colour comparison itself lets pass (the fit above only compares shapes to within the contrast; a word turned grey
    # or blue keeps its shape)
    if first.ndim == 3:
        for share in (2, 98):
            tones = [np.percentile(picture.reshape(-1, 3), share, axis=0) for picture in (first, second)]
            if _delta_e(tones[0], tones[1]) > same_colour_delta_e:
                return False
    return True


def _delta_e(first: np.ndarray, second: np.ndarray) -> float:
    """CIEDE2000 between two RGB colours (0..255)."""
    return ciede2000(*(rgb_to_lab(tuple(float(c) / 255.0 for c in colour)) for colour in (first, second)))


def _words(ink: np.ndarray, line_height: int) -> list[tuple[int, int]]:
    """The (left, right) columns of each word of one line of ink: runs of inked columns, split where the paper between
    them is at least a quarter of the line's height wide."""
    columns = np.flatnonzero(ink.any(axis=0))
    if columns.size == 0:
        return []
    gap = max(3, line_height // 4)
    words = [[int(columns[0]), int(columns[0]) + 1]]
    for column in columns[1:]:
        if column - words[-1][1] < gap:
            words[-1][1] = int(column) + 1
        else:
            words.append([int(column), int(column) + 1])
    return [(left, right) for left, right in words]


def only_nudged(
    original: np.ndarray, suspect: np.ndarray, candidate: DiffCandidate, tolerance_px: float, same_colour_delta_e: float
) -> bool:
    """Whether the difference `candidate` marks is nothing but what is printed there moved by up to `tolerance_px`, its
    letters' spacing drifting a little (see above)."""
    box = candidate.bbox
    # tall enough round the difference that every line passing through it is seen whole
    reach = max(_WINDOW_PAD_PX, int(np.ceil(box.height)))
    x0, y0, x1, y1 = padded_window(box, 0, original.shape)
    x0, x1 = max(0, x0 - _WORD_REACH_PX), min(original.shape[1], x1 + _WORD_REACH_PX)
    y0, y1 = max(0, y0 - reach), min(original.shape[0], y1 + reach)
    first, second = original[y0:y1, x0:x1], suspect[y0:y1, x0:x1]
    if first.size == 0 or first.shape != second.shape:
        return False
    flat_first = first.min(axis=2) if first.ndim == 3 else first
    flat_second = second.min(axis=2) if second.ndim == 3 else second
    brightest = np.maximum(first.max(axis=2) if first.ndim == 3 else first, second.max(axis=2) if second.ndim == 3 else second)
    paper = float(np.percentile(brightest, 90))
    ink = np.minimum(flat_first, flat_second).astype(np.int16) < paper - _MIN_CONTRAST
    height, width = ink.shape
    own_top, own_bottom = max(0, int(box.y0) - y0), min(height, int(np.ceil(box.y1)) - y0)
    own_left, own_right = max(0, int(box.x0) - x0), min(width, int(np.ceil(box.x1)) - x0)
    # the lines are told apart in the difference's own columns (a neighbouring column's lines sit at other heights)...
    near_left, near_right = max(0, own_left - _WINDOW_PAD_PX), min(width, own_right + _WINDOW_PAD_PX)
    judged = [(top, bottom) for top, bottom in _lines(ink[:, near_left:near_right]) if top < own_bottom and bottom > own_top]
    if not judged:
        return False
    for top, bottom in judged:
        if top == 0 or bottom == height:
            return False  # a line cut by the window: it cannot be judged whole
        band_top, band_bottom = max(0, top - _BAND_PAD_PX), min(height, bottom + _BAND_PAD_PX)
        if _fits(first[band_top:band_bottom, near_left:near_right], second[band_top:band_bottom, near_left:near_right], tolerance_px, same_colour_delta_e):
            continue
        # ...and a line that does not fit one stretch is judged word by word (its words may each have drifted their own
        # way), each word seen whole however far past the difference it runs
        words = [(left, right) for left, right in _words(ink[top:bottom], bottom - top) if left < own_right and right > own_left]
        if not words:
            return False
        for left, right in words:
            if left == 0 or right == width:
                return False
            word_left, word_right = max(0, left - _BAND_PAD_PX), min(width, right + _BAND_PAD_PX)
            if not _fits(first[band_top:band_bottom, word_left:word_right], second[band_top:band_bottom, word_left:word_right], tolerance_px, same_colour_delta_e):
                return False
    return True


def drop_nudged(
    candidates: list[DiffCandidate], original: np.ndarray, suspect: np.ndarray, tolerance_px: float, same_colour_delta_e: float
) -> tuple[list[DiffCandidate], int]:
    """`candidates` without the differences that are only what is printed nudged (see above), and how many were dropped.
    `same_colour_delta_e` is the colour difference the comparison lets pass (anything re-coloured more is kept)."""
    kept = [c for c in candidates if not only_nudged(original, suspect, c, tolerance_px, same_colour_delta_e)]
    return kept, len(candidates) - len(kept)
