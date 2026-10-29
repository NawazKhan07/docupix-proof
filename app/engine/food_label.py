"""Food label comparison: Automatic mode's own way with consumer labels -- a snack pack, a drinks wrapper, a 1 kg bag of
fries (`DetectionConfig.food_label`, the results page's Automatic settings).

A pharmaceutical carton is compared pixel by pixel: a letter changed is what matters, and the artwork holds still. A
food label is laid out more freely -- its background is a pattern re-laid, its photograph re-shot, its panels re-set,
its logo re-placed, a sticker laid over its picture -- and compared pixel by pixel it is a hundred small boxes, each a
sliver of something. So here the label is compared by the THINGS printed on it:

1. Each file is told apart into its BACKGROUND (the flat colours print sits on: a red ground, a grey panel, white) and
   its PRINT, and the print is joined into things -- a line of type, a logo and its frame, a sticker, a photograph --
   across the gaps inside a thing (a letter's height or so, measured on the label itself).
2. A thing is CHANGED where enough of its marks differ between the files. What differs is measured on the marks alone
   (each file's fine detail, its own colour taken away), against how much this pair differs where nothing changed
   (the proof prints softer, its edges a pixel off): never a preset bar.
3. What happened to each changed thing is said: MOVED (found near by in the other file), its TEXT changed (read in
   both, line set against line), ADDED or REMOVED (printed in one file only), its PICTURE changed (a photograph), or
   it changed otherwise. A flat graphic laid over a photograph (a sticker on the plate) is a thing of its own.
4. A flat AREA whose colour changed by more than the proof's colours shift overall (a panel gone from light grey to
   white) is one finding, named.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from app.config import DetectionConfig
from app.logging_config import StageTimer
from app.models.diff import BoundingBox, DiffCandidate, DiffCluster, DiffSubtype, DiffType

logger = logging.getLogger(__name__)

_WORK_LONG_EDGE_PX = 2400  # the label is compared at about this size (its long edge)...
_FORGIVE_PX = 1  # ...each mark against the other file's within this much of where it is (a proof's edges a pixel off)
_DETAIL_SIGMA = 6.0  # (a file's marks: its detail finer than this, its own colour taken away)
_NOISE_TILE_PX = 48  # the pair's own difference is measured tile by tile...
_NOISE_PERCENTILE = 99.0  # ...by how much the most differing pixels of a tile differ...
_DETAIL_PRINT = 20.0  # (a pixel of fine detail at least this strong, 8-bit Lab, is print)...
_NOISE_PRINTED = 0.05  # ...and a tile with this share of it is measured
_SPECK_PX = 4  # (a group of differing pixels smaller than this is a speck)
_NOISE_TIMES = 2.5  # ...and a mark differs where it differs this many times the typical tile's (the median tile)
_MARKS_FLOOR = 12.0  # (and by at least this much, 8-bit Lab: a flawless pair's noise is near nought)
_FLAT_GRADIENT = 18  # a pixel is flat where its grey changes by less than this across 5 px...
_GROUND_SHARE = 0.002  # ...and flat areas of one colour this big (a share of the label) are its background
_COLOUR_BINS = (24, 12, 12)  # (a flat area is one colour where its Lab falls in one bin of these sizes)
_MIN_THING_SHARE = 2e-5  # a thing smaller than this share of the label is a speck...
_THING_MAX_SHARE = 0.03  # ...and print joined into a box bigger than this share is not one thing (a panel's rules and all)
_CHANGED_SHARE = 0.04  # a thing has changed where at least this share of it differs (or a letter's worth of it)...
_LETTER_CHANGED = 0.06  # (a letter's worth: its ink, about this share of a letter height squared)
_CHANGED_MIN_PX = 12  # ...and at least this many of its pixels (at the working size)
_PICTURE_LETTERS = 4.0  # a photograph is at least this many letters tall and wide...
_PHOTO_WINDOW_LETTERS = 3.0  # (a photograph is told over windows this many letters wide)
_PATTERN_PIECE_LETTERS = 8.0  # graphics changed no bigger than this many letters...
_PATTERN_REACH_LETTERS = 6.0  # ...this close (in letters)...
_STRIP_WIDE_LETTERS = 12.0  # (a strip is at most this many letters across)
_STRIP_SHARE = 0.5  # (pieces of one strip share this much of the narrower one's width or height)
_STRIP_REACH_LETTERS = 60.0  # (...or lined up along one strip, a column or a row, at most this many letters apart)
_PATTERN_MIN_PIECES = 3  # ...and at least this many are a pattern changed
_PICTURE_JOIN_LETTERS = 3.0  # (the changed pieces of one photograph lie at most this many letters apart)
_PICTURE_PRINTED = 0.3  # ...a fair share printed (not ground)...
_GENTLE_STEP = (6, 40)  # ...its tones stepping gently (grey levels across 3 px)...
_PICTURE_GENTLE = 0.25  # ...over at least this share of it (a photograph: ~0.4; type, a logo, a table: ~0.05)...
_SHARP_STEP = 60  # ...and sharply...
_PICTURE_SHARP = 0.2  # ...over at most this share (a photograph: ~0.05; type and graphics: 0.5 and more)
_HAIRLINE_LETTERS = 0.3  # a thing thinner than this share of a letter is a hairline (a fold, a guide)
_MOVE_SEARCH_SHARE = 0.05  # a thing is looked for this far round it (a share of the label's long edge)...
_MOVE_FIT = 0.8  # ...and found where it fits at least this well...
_MOVE_GAIN = 0.15  # ...this much better than where it is...
_SPECK_LETTERS = 0.5  # a thing smaller than this many letters' squares is a speck
_MOVED_TOGETHER_SHARE = 0.1  # things moved the same way this close (a share of the label's long edge) moved together...
_SAME_MOVE_PX = 8.0  # ...by the same, give or take this
_MOVE_CLEAR = 0.05  # ...and better than anywhere else by this much (a stripe fits itself anywhere)
_STICKER_COLOUR = 40.0  # (a sticker is of another colour than the photograph under it by at least this, 8-bit Lab)
_STICKER_ITSELF = 2.0  # (the flat area a sticker is in is at most this many times what changed: it is the sticker)
_STICKER_MIN_SHARE = 1e-3  # a flat graphic on a photograph is at least this share of the label
_AREA_MIN_SHARE = 0.0015  # a flat area this big may be recoloured...
_AREA_TIMES = 2.0  # ...where its colour changed this many times as much as the label's typical area's...
_AREA_FLOOR = 15.0  # ...and by at least this much (8-bit Lab)
_AREA_JOIN_SHARE = 0.015  # pieces of one area this close (a share of the label's long edge)...
_AREA_SAME_COLOUR = 12.0  # ...that were and became the same colours (8-bit Lab) are one area
_AREA_EDGE_PX = 3  # (an area reaching the edge of what is compared is the sheet round the artwork, or its bleed)


@dataclass
class LabelComparison:
    clusters: list[DiffCluster] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class _Thing:
    box: tuple[int, int, int, int]  # (x0, y0, x1, y1) at the working size
    changed: int  # pixels of it that differ
    area: int  # pixels of it


def compare_label(
    original: np.ndarray,
    suspect: np.ndarray,
    compared: np.ndarray,
    config: DetectionConfig,
    say: Callable[[str], None] | None = None,
    original_words: list | None = None,
    suspect_words: list | None = None,
) -> LabelComparison:
    """Compare two aligned pictures of a label (`original`, `suspect`: RGB, the same size; `compared`: what of them is
    compared) as the module says. Findings are in the pictures' own pixels."""
    say = say or (lambda _text: None)
    result = LabelComparison()
    height, width = original.shape[:2]
    scale = min(1.0, _WORK_LONG_EDGE_PX / max(height, width))
    with StageTimer(logger, "label_prepare"):
        a = cv2.resize(original, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else original.copy()
        b = cv2.resize(suspect, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA) if scale < 1 else suspect.copy()
        inside = cv2.resize(compared.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        lab_a = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_b = cv2.cvtColor(b, cv2.COLOR_RGB2LAB).astype(np.float32)
    say("Comparing what is printed on the label…")
    with StageTimer(logger, "label_marks"):
        differs, bar = _marks_differ(lab_a, lab_b, inside)
    with StageTimer(logger, "label_things"):
        flat_a, ground_a = _ground(a, lab_a, inside)
        flat_b, ground_b = _ground(b, lab_b, inside)
        letter = _letter_height(~ground_a & inside)
        things = _things(~ground_a & inside, ~ground_b & inside, differs, letter, a.shape[:2])
    logger.info(
        "label compared",
        extra={"marks_bar": round(bar, 1), "letter_px": round(letter, 1), "things_changed": len(things), "scale": round(scale, 3)},
    )
    to_page = 1.0 / scale

    def on_page(box: tuple[int, int, int, int]) -> BoundingBox:
        x0, y0, x1, y1 = box
        return BoundingBox(x0 * to_page, y0 * to_page, min(width, x1 * to_page), min(height, y1 * to_page))

    findings: list[DiffCandidate] = []
    texts: list[tuple[DiffCandidate, BoundingBox]] = []
    gray_a, gray_b = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    claimed: list[tuple[int, int, int, int]] = []  # (where a thing that moved went: what differs there is its arrival)
    moved_things: list[DiffCandidate] = []
    with StageTimer(logger, "label_said"):
        kept = [
            thing for thing in things
            if (thing.box[2] - thing.box[0]) * (thing.box[3] - thing.box[1]) >= _SPECK_LETTERS * letter * letter
            and min(thing.box[2] - thing.box[0], thing.box[3] - thing.box[1]) >= _HAIRLINE_LETTERS * letter
        ]  # (not a speck, nor a hairline at a fold or a guide)
        # The pieces of a photograph that changed (a re-shot plate of fries changes in a hundred places, its tones stepping
        # gently everywhere): one picture, where they make one big enough to be a picture; else soft shading rendered a
        # little differently, which is no change. Never 'moved': a patch of fries fits another patch anywhere.
        photos = _photographs(a, ground_a, letter) + _photographs(b, ground_b, letter)
        photos = _joined_boxes(photos, letter)
        in_photo = {id(thing): next((photo for photo in photos if _inside(thing.box, photo, 0.5)), None) for thing in kept}
        soft = [
            thing for thing in kept
            if in_photo[id(thing)] is not None
            or (_photograph(a, ground_a, thing.box, any_size=True) and _photograph(b, ground_b, thing.box, any_size=True))
        ]
        pictures = {in_photo[id(thing)] for thing in soft if in_photo[id(thing)] is not None}
        loose = _joined_boxes([thing.box for thing in soft if in_photo[id(thing)] is None], _PICTURE_JOIN_LETTERS * letter)
        for picture in list(pictures) + loose:
            px0, py0, px1, py1 = picture
            claimed.append(picture)
            if min(px1 - px0, py1 - py0) <= _PICTURE_LETTERS * letter:
                continue
            findings.append(DiffCandidate(on_page(picture), DiffType.GRAPHICS, DiffSubtype.CHANGED, 0.8, "Picture changed", 0.0))
            others = [other.box for other in kept if other not in soft]
            findings += [
                DiffCandidate(on_page(sticker), DiffType.GRAPHICS, kind, 0.85, wording, 0.0)
                for sticker, kind, wording in _laid_on_picture(flat_a, flat_b, lab_a, lab_b, picture, a.shape[:2])
                if not any(_inside(sticker, other, 0.3) or _inside(other, sticker, 0.3) for other in others)
            ]
        for thing in sorted((t for t in kept if t not in soft), key=lambda t: -(t.box[2] - t.box[0]) * (t.box[3] - t.box[1])):
            x0, y0, x1, y1 = thing.box
            if any(_inside(thing.box, other) for other in claimed):
                continue  # (part of a thing already said: a photograph that changed, where something moved from or to)
            own = _own_box(~ground_a & inside, thing.box)  # (the thing as the original prints it: where it was)
            moved = _moved(gray_a, gray_b, own, max(a.shape[:2])) if own is not None else None
            if moved is not None:
                dx, dy = moved
                x0, y0, x1, y1 = own
                there = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
                claimed += [thing.box, own, there]
                how = _how_far(dx * to_page, dy * to_page)
                candidate = DiffCandidate(on_page(own), DiffType.GRAPHICS, DiffSubtype.MOVED, 0.9, f"Moved {how}", math.hypot(dx, dy) * to_page)
                candidate.moved_to = on_page(there)
                findings.append(candidate)
                if min(x1 - x0, y1 - y0) >= letter:
                    moved_things.append(candidate)  # (its words are read lined up, below: it may have changed as it moved)
                continue
            box = on_page(thing.box)
            ink_a = float((~ground_a[y0:y1, x0:x1] & inside[y0:y1, x0:x1]).mean())
            ink_b = float((~ground_b[y0:y1, x0:x1] & inside[y0:y1, x0:x1]).mean())
            if ink_a < 0.25 * ink_b:
                subtype, wording = DiffSubtype.ADDED, "Added in suspect"
            elif ink_b < 0.25 * ink_a:
                subtype, wording = DiffSubtype.REMOVED, "Missing from suspect"
            else:
                subtype, wording = DiffSubtype.CHANGED, "Changed"
            candidate = DiffCandidate(box, DiffType.GRAPHICS, subtype, 0.8, f"Graphic {wording.lower()}", thing.changed)
            findings.append(candidate)
            texts.append((candidate, box))  # (words may be what changed: they are read, below)
    with StageTimer(logger, "label_areas"):
        for area_box, was, now, change in _recoloured(lab_a, lab_b, flat_a, inside):
            if any(_inside(area_box, other) for other in claimed):
                continue  # (where something moved from or to, or a photograph: that is what changed there)
            name = _colour_change(was, now)
            findings.append(DiffCandidate(on_page(area_box), DiffType.COLOUR, DiffSubtype.CHANGED, 0.9, f"Colour changed: {name} (ΔE {change * 100 / 255:.1f})", change, region=True))
    if moved_things:
        say("Reading the words of what moved…")
        _read_moved(moved_things, original, suspect, config, result)
    findings = _moves_together(findings, _MOVED_TOGETHER_SHARE * max(width, height))
    if texts:
        say("Reading the words of what changed…")
        _read(texts, original, suspect, config, original_words, suspect_words, result)
    findings = _patterns(findings, letter * to_page)
    result.clusters = _as_clusters(findings)
    return result


# ---- what differs -----------------------------------------------------------------------------------------------


def _marks_differ(lab_a: np.ndarray, lab_b: np.ndarray, inside: np.ndarray) -> tuple[np.ndarray, float]:
    """Where the two files' marks differ (each file's fine detail, its own colour taken away; each pixel against the
    other file's nearest within `_FORGIVE_PX`), and the bar it was told by (the pair's own noise, `_NOISE_TIMES` over)."""
    detail_a = cv2.GaussianBlur(lab_a, (0, 0), 0.8) - cv2.GaussianBlur(lab_a, (0, 0), _DETAIL_SIGMA)
    detail_b = cv2.GaussianBlur(lab_b, (0, 0), 0.8) - cv2.GaussianBlur(lab_b, (0, 0), _DETAIL_SIGMA)
    nearest = None
    reach = range(-_FORGIVE_PX, _FORGIVE_PX + 1)
    for dy in reach:
        for dx in reach:
            shifted = np.roll(np.roll(detail_b, dy, axis=0), dx, axis=1)
            distance = np.linalg.norm(detail_a - shifted, axis=2)
            nearest = distance if nearest is None else np.minimum(nearest, distance)
    nearest[~inside] = 0
    height, width = nearest.shape
    tile = _NOISE_TILE_PX
    # (measured where there is print to differ: a tile of blank paper says nothing of how this pair renders an edge)
    detail = np.linalg.norm(detail_a, axis=2) > _DETAIL_PRINT
    levels = [
        float(np.percentile(nearest[y : y + tile, x : x + tile][inside[y : y + tile, x : x + tile]], _NOISE_PERCENTILE))
        for y in range(0, height - tile, tile)
        for x in range(0, width - tile, tile)
        if inside[y : y + tile, x : x + tile].mean() > 0.5 and detail[y : y + tile, x : x + tile].mean() >= _NOISE_PRINTED
    ]
    bar = max(_MARKS_FLOOR, _NOISE_TIMES * float(np.median(levels))) if levels else _MARKS_FLOOR
    # (isolated specks of a pixel or two left out -- not by opening the picture, which would take the thin strokes of
    # small type with them: '24' -> '36' months is a stroke's width wide)
    raw = (nearest > bar).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= _SPECK_PX
    return keep[labels], bar


def _ground(rgb: np.ndarray, lab: np.ndarray, inside: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(where the picture is flat, its background): flat pixels, and the flat areas of one colour big enough to be what
    print sits on."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    flat = (cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) < _FLAT_GRADIENT) & inside
    smooth = cv2.GaussianBlur(lab, (0, 0), 2)
    bins = (
        (smooth[..., 0] // _COLOUR_BINS[0]).astype(np.int32) * 10000
        + (smooth[..., 1] // _COLOUR_BINS[1]).astype(np.int32) * 100
        + (smooth[..., 2] // _COLOUR_BINS[2]).astype(np.int32)
    )
    ground = np.zeros(flat.shape, bool)
    least = _GROUND_SHARE * flat.size
    for value in np.unique(bins[flat]):
        count, labels, stats, _ = cv2.connectedComponentsWithStats((flat & (bins == value)).astype(np.uint8), connectivity=4)
        big = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= least) + 1
        if len(big):
            ground |= np.isin(labels, big)
    # (the rim of a ground area -- its anti-aliased edge against print, a pixel or two -- is ground too: counted as print,
    # the rims of every area would join every thing on the label into one)
    return flat, cv2.dilate(ground.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


def _letter_height(print_mask: np.ndarray) -> float:
    """The label's typical letter height (px): the median height of its small marks."""
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(print_mask.astype(np.uint8), connectivity=8)
    heights = stats[1:, cv2.CC_STAT_HEIGHT][(stats[1:, cv2.CC_STAT_AREA] >= 6) & (stats[1:, cv2.CC_STAT_HEIGHT] <= 60)]
    return float(np.clip(np.median(heights), 4, 30)) if len(heights) else 8.0


def _things(print_a: np.ndarray, print_b: np.ndarray, differs: np.ndarray, letter: float, shape: tuple[int, int]) -> list[_Thing]:
    """The printed things (in either file) that changed: print joined across a letter's width along a line and a third
    of one across lines (a line of type is one thing, the next line another), those with enough of them differing."""
    along, across = max(3, int(round(letter * 1.2))) | 1, max(1, int(round(letter * 0.35))) | 1
    printed = (print_a | print_b).astype(np.uint8)
    joined = cv2.dilate(printed, cv2.getStructuringElement(cv2.MORPH_RECT, (along, across)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    changed_per = np.bincount(labels[differs & printed.astype(bool)].ravel(), minlength=count)
    printed_per = np.bincount(labels[printed.astype(bool)].ravel(), minlength=count)
    least = _MIN_THING_SHARE * shape[0] * shape[1]
    largest = _THING_MAX_SHARE * shape[0] * shape[1]
    pad_x, pad_y = along // 2, across // 2  # (back to the print's own extent)
    things = []
    for index in range(1, count):
        area = int(printed_per[index])
        changed = int(changed_per[index])
        if area < least or changed < _CHANGED_MIN_PX:
            continue
        x, y, w, h = (int(v) for v in stats[index, :4])
        if w * h > largest:
            # Print joined into something as big as a panel (rules, borders, a pattern running into everything): what of
            # it differs is the thing that changed -- each part of it, joined the same way.
            own = (labels[y : y + h, x : x + w] == index) & differs[y : y + h, x : x + w]
            parts = cv2.dilate(own.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (along * 2 + 1, across * 2 + 1)))
            part_count, _part_labels, part_stats, _ = cv2.connectedComponentsWithStats(parts, connectivity=8)
            for part in range(1, part_count):
                px, py, pw, ph, pixels = (int(v) for v in part_stats[part])
                if pixels >= max(least, _CHANGED_MIN_PX * 4):
                    things.append(_Thing((x + px + along, y + py + across, x + px + pw - along, y + py + ph - across), pixels, pw * ph))
            continue
        if changed < _CHANGED_SHARE * area and changed < _LETTER_CHANGED * letter * letter:
            continue  # (a thing changed where a fair share of it differs -- or at least a letter's worth: '24' -> '36')
        things.append(_Thing((x + pad_x, y + pad_y, x + w - pad_x, y + h - pad_y), changed, area))
    return things


# ---- what happened to it ----------------------------------------------------------------------------------------


def _own_box(printed: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    """The box round what is printed inside `box` (in one file's print)."""
    x0, y0, x1, y1 = box
    ys, xs = np.nonzero(printed[y0:y1, x0:x1])
    if not len(xs):
        return None
    return x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1


def _moved(gray_a: np.ndarray, gray_b: np.ndarray, box: tuple[int, int, int, int], long_edge: int) -> tuple[int, int] | None:
    """(dx, dy) where the thing in `box` lies in the suspect, if it lies clearly elsewhere than where it is."""
    x0, y0, x1, y1 = box
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    template = gray_a[y0:y1, x0:x1].astype(np.float32)
    if float(template.std()) < 8:
        return None
    reach = int(_MOVE_SEARCH_SHARE * long_edge)
    height, width = gray_b.shape
    sx0, sy0, sx1, sy1 = max(0, x0 - reach), max(0, y0 - reach), min(width, x1 + reach), min(height, y1 + reach)
    area = gray_b[sy0:sy1, sx0:sx1].astype(np.float32)
    if area.shape[0] < template.shape[0] or area.shape[1] < template.shape[1]:
        return None
    scores = cv2.matchTemplate(area, template, cv2.TM_CCOEFF_NORMED)
    _low, best, _at_low, (bx, by) = cv2.minMaxLoc(scores)
    hx, hy = x0 - sx0, y0 - sy0
    home = float(scores[hy, hx]) if 0 <= hy < scores.shape[0] and 0 <= hx < scores.shape[1] else -1.0
    dx, dy = sx0 + bx - x0, sy0 + by - y0
    if best < _MOVE_FIT or best < home + _MOVE_GAIN or math.hypot(dx, dy) < 3:
        return None
    elsewhere = scores.copy()
    th, tw = template.shape
    elsewhere[max(0, by - th // 2) : by + th // 2 + 1, max(0, bx - tw // 2) : bx + tw // 2 + 1] = -1
    if float(elsewhere.max()) > best - _MOVE_CLEAR:
        return None
    return dx, dy


def _photograph(rgb: np.ndarray, ground: np.ndarray, box: tuple[int, int, int, int], any_size: bool = False) -> bool:
    """Whether what is printed in `box` is a photograph: its tones change gently, pixel to pixel, almost everywhere
    (fries, a plate's shading) -- where type, a logo or a table is flat colour meeting flat colour at sharp edges."""
    x0, y0, x1, y1 = box
    printed = ~ground[y0:y1, x0:x1]
    if not printed.any() or (printed.mean() < _PICTURE_PRINTED and not any_size):
        return False
    gray = cv2.cvtColor(np.ascontiguousarray(rgb[y0:y1, x0:x1]), cv2.COLOR_RGB2GRAY)
    steps = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))[printed]
    gentle = float(((steps > _GENTLE_STEP[0]) & (steps < _GENTLE_STEP[1])).mean())
    sharp = float((steps >= _SHARP_STEP).mean())
    return gentle >= _PICTURE_GENTLE and sharp <= _PICTURE_SHARP


def _photographs(rgb: np.ndarray, ground: np.ndarray, letter: float) -> list[tuple[int, int, int, int]]:
    """Where a picture holds photographs (boxes): print whose tones step gently almost everywhere, over a few letters'
    width -- fries on a plate, a face -- where type and graphics are flat colour meeting flat colour at sharp edges."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    steps = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    printed = (~ground).astype(np.float32)
    gentle = (((steps > _GENTLE_STEP[0]) & (steps < _GENTLE_STEP[1])) & ~ground).astype(np.float32)
    sharp = ((steps >= _SHARP_STEP) & ~ground).astype(np.float32)
    size = max(9, int(_PHOTO_WINDOW_LETTERS * letter)) | 1
    printed_d = cv2.blur(printed, (size, size))
    share = lambda part: cv2.blur(part, (size, size)) / np.maximum(printed_d, 1e-3)  # noqa: E731 -- share of what is printed
    photo = (printed_d >= _PICTURE_PRINTED) & (share(gentle) >= _PICTURE_GENTLE) & (share(sharp) <= _PICTURE_SHARP)
    photo = cv2.morphologyEx(photo.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(photo, connectivity=8)
    least = _PICTURE_LETTERS * letter
    return [
        (int(x), int(y), int(x + w), int(y + h)) for x, y, w, h, _area in stats[1:] if min(w, h) > least
    ]


def _laid_on_picture(
    flat_a: np.ndarray, flat_b: np.ndarray, lab_a: np.ndarray, lab_b: np.ndarray, box: tuple[int, int, int, int], shape: tuple[int, int]
):
    """Flat graphics laid over a photograph in one file only (a sticker on the plate): where one file is flat and of
    another colour altogether than the other there -- (box, subtype, wording)."""
    x0, y0, x1, y1 = box
    found = []
    least = _STICKER_MIN_SHARE * shape[0] * shape[1]
    recoloured = np.linalg.norm(
        cv2.GaussianBlur(lab_a[y0:y1, x0:x1], (0, 0), 2) - cv2.GaussianBlur(lab_b[y0:y1, x0:x1], (0, 0), 2), axis=2
    ) > _STICKER_COLOUR
    for flat, only, kind, wording in (
        (flat_b[y0:y1, x0:x1], flat_b[y0:y1, x0:x1] & recoloured, DiffSubtype.ADDED, "Graphic added on the picture in suspect"),
        (flat_a[y0:y1, x0:x1], flat_a[y0:y1, x0:x1] & recoloured, DiffSubtype.REMOVED, "Graphic missing from the picture in suspect"),
    ):
        whole_count, whole_labels, whole_stats, _ = cv2.connectedComponentsWithStats(flat.astype(np.uint8), connectivity=8)
        solid = cv2.morphologyEx(only.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        count, _labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
        for index in range(1, count):
            x, y, w, h, pixels = (int(v) for v in stats[index])
            if pixels < least or pixels < 0.5 * w * h:
                continue  # (not a solid shape: the scattered flat bits of a photograph)
            # A sticker is a shape of its own: the flat area it is in is (mostly) it. A flat area running on well
            # beyond what changed is the photograph's own (the plate, partly covered by fries in the other file).
            owners = np.unique(whole_labels[y : y + h, x : x + w][solid[y : y + h, x : x + w] > 0])
            owners = owners[owners > 0]
            extent = int(whole_stats[owners, cv2.CC_STAT_AREA].sum()) if len(owners) else pixels
            if extent > _STICKER_ITSELF * pixels:
                continue
            found.append(((x0 + x, y0 + y, x0 + x + w, y0 + y + h), kind, wording))
    return found


def _recoloured(lab_a: np.ndarray, lab_b: np.ndarray, flat_a: np.ndarray, inside: np.ndarray):
    """The original's flat areas whose colour changed by more than the label's areas typically do (`_AREA_TIMES` over the
    area-weighted median: the proof's colours printed a shade off throughout): (box, was, now, change)."""
    smooth = cv2.GaussianBlur(lab_a, (0, 0), 2)
    bins = (
        (smooth[..., 0] // _COLOUR_BINS[0]).astype(np.int32) * 10000
        + (smooth[..., 1] // _COLOUR_BINS[1]).astype(np.int32) * 100
        + (smooth[..., 2] // _COLOUR_BINS[2]).astype(np.int32)
    )
    core = cv2.erode(inside.astype(np.uint8), np.ones((2 * _AREA_EDGE_PX + 1, 2 * _AREA_EDGE_PX + 1), np.uint8)).astype(bool)
    least = _AREA_MIN_SHARE * flat_a.size
    areas = []
    for value in np.unique(bins[flat_a]):
        count, labels, stats, _ = cv2.connectedComponentsWithStats((flat_a & (bins == value)).astype(np.uint8), connectivity=4)
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] < least:
                continue
            region = labels == index
            if (region & ~core).any():
                continue  # (it reaches the edge of what is compared: the sheet round the artwork, its bleed)
            was, now = np.median(lab_a[region], axis=0), np.median(lab_b[region], axis=0)
            x, y, w, h = (int(v) for v in stats[index, :4])
            areas.append(((x, y, x + w, y + h), was, now, float(np.linalg.norm(now - was)), int(stats[index, cv2.CC_STAT_AREA])))
    if not areas:
        return []
    changes = np.array([area[3] for area in areas])
    weights = np.array([area[4] for area in areas], float)
    order = np.argsort(changes)
    typical = float(changes[order][np.searchsorted(np.cumsum(weights[order]) / weights.sum(), 0.5)])
    bar = max(_AREA_FLOOR, _AREA_TIMES * typical)
    recoloured = [[box, was, now, change] for box, was, now, change, _pixels in areas if change > bar]
    # one area cut in pieces by what is printed across it (a panel's lines of type): one finding, if each piece changed
    # from and to the same colours
    reach = _AREA_JOIN_SHARE * max(flat_a.shape)
    merged = True
    while merged:
        merged = False
        for i in range(len(recoloured)):
            for j in range(i + 1, len(recoloured)):
                (box_i, was_i, now_i, d_i), (box_j, was_j, now_j, d_j) = recoloured[i], recoloured[j]
                if _near(box_i, box_j, reach) and np.linalg.norm(was_i - was_j) < _AREA_SAME_COLOUR and np.linalg.norm(now_i - now_j) < _AREA_SAME_COLOUR:
                    recoloured[i] = [
                        (min(box_i[0], box_j[0]), min(box_i[1], box_j[1]), max(box_i[2], box_j[2]), max(box_i[3], box_j[3])),
                        was_i, now_i, max(d_i, d_j),
                    ]
                    del recoloured[j]
                    merged = True
                    break
            if merged:
                break
    return [tuple(area) for area in recoloured]


def _near(a: tuple[int, int, int, int], b: tuple[int, int, int, int], reach: float) -> bool:
    return a[0] - reach <= b[2] and b[0] - reach <= a[2] and a[1] - reach <= b[3] and b[1] - reach <= a[3]


# ---- what it says -----------------------------------------------------------------------------------------------


def _read(texts, original, suspect, config, original_words, suspect_words, result: LabelComparison) -> None:
    """Each changed thing that may be type, read in both files (in the languages found on them): where its words
    changed, that is what it says."""
    from app.engine.detectors import text_diff
    from app.engine.detectors.text_diff import OcrUnavailableError

    try:
        if not text_diff.ocr_available():
            raise OcrUnavailableError(text_diff._OCR_UNAVAILABLE)
        described = text_diff.describe([box for _c, box in texts], original, suspect, config, original_words, suspect_words)
    except OcrUnavailableError as exc:
        result.warnings.append(str(exc))
        return
    for (candidate, _box), said in zip(texts, described):
        if said is not None and said.detail:
            candidate.type = said.type
            candidate.subtype = said.subtype
            candidate.detail = said.detail


def _read_moved(moved: list[DiffCandidate], original, suspect, config, result: LabelComparison) -> None:
    """Each thing that moved, read where it was in the original and where it went in the suspect -- lined up, line set
    against line: where its words changed as well ('7 x 7 mm' -> '9 x 9 mm' in a block re-placed), it says so."""
    from app.engine.detectors import text_diff
    from app.engine.detectors.text_diff import OcrUnavailableError

    height, width = original.shape[:2]
    for candidate in moved:
        was, now = candidate.bbox, candidate.moved_to
        x0, y0 = int(max(0, was.x0)), int(max(0, was.y0))
        w = int(min(width - x0, was.width, width - max(0, now.x0)))
        h = int(min(height - y0, was.height, height - max(0, now.y0)))
        nx0, ny0 = int(max(0, now.x0)), int(max(0, now.y0))
        if w < 16 or h < 16:
            continue
        here = np.ascontiguousarray(original[y0 : y0 + h, x0 : x0 + w])
        there = np.ascontiguousarray(suspect[ny0 : ny0 + h, nx0 : nx0 + w])
        if here.shape != there.shape:
            continue
        try:
            (said,) = text_diff.describe([BoundingBox(0, 0, w, h)], here, there, config)
        except OcrUnavailableError as exc:
            result.warnings.append(str(exc))
            return
        if said is not None and said.subtype is not DiffSubtype.MOVED and "'" in said.detail:
            candidate.detail = f"{candidate.detail}, and its text changed: {said.detail}"


def _as_clusters(findings: list[DiffCandidate]) -> list[DiffCluster]:
    """One finding each, in reading order."""
    ordered = sorted(findings, key=lambda c: (round(c.bbox.y0 / 40), c.bbox.x0))
    clusters = []
    for number, candidate in enumerate(ordered, start=1):
        clusters.append(
            DiffCluster(
                id=number, bbox_in_original=candidate.bbox, bbox_in_suspect=candidate.bbox, type=candidate.type,
                members=[candidate], detail=candidate.detail, moved_to=candidate.moved_to,
            )
        )
    return clusters


def _moves_together(findings: list[DiffCandidate], reach: float) -> list[DiffCandidate]:
    """Things near each other that moved the same way (the scattered pieces of a background re-set as a whole, the
    lines of a block): one finding, round them all."""
    moved = [c for c in findings if c.subtype is DiffSubtype.MOVED and c.moved_to is not None]
    rest = [c for c in findings if not (c.subtype is DiffSubtype.MOVED and c.moved_to is not None)]
    groups: list[list[DiffCandidate]] = []
    for candidate in moved:
        shift = (candidate.moved_to.x0 - candidate.bbox.x0, candidate.moved_to.y0 - candidate.bbox.y0)
        for group in groups:
            first = group[0]
            other = (first.moved_to.x0 - first.bbox.x0, first.moved_to.y0 - first.bbox.y0)
            if math.hypot(shift[0] - other[0], shift[1] - other[1]) <= _SAME_MOVE_PX + 0.05 * math.hypot(*other) and any(
                _boxes_near(candidate.bbox, member.bbox, reach) for member in group
            ):
                group.append(candidate)
                break
        else:
            groups.append([candidate])
    out = list(rest)
    for group in groups:
        if len(group) == 1:
            out.append(group[0])
            continue
        was, now = group[0].bbox, group[0].moved_to
        for member in group[1:]:
            was, now = was.union(member.bbox), now.union(member.moved_to)
        one = DiffCandidate(was, DiffType.GRAPHICS, DiffSubtype.MOVED, 0.9, f"{group[0].detail} ({len(group)} pieces moved together)", group[0].magnitude)
        one.moved_to = now
        out.append(one)
    return out


def _patterns(findings: list[DiffCandidate], letter: float) -> list[DiffCandidate]:
    """Small graphics changed close together, none of them words (the fries of a background re-laid along a panel's
    edge, each one now a little elsewhere): one finding -- the pattern changed -- not one per piece."""
    small = [
        c for c in findings
        if c.type is DiffType.GRAPHICS and c.detail != "Picture changed" and not c.region
        and max(c.bbox.width, c.bbox.height) <= _PATTERN_PIECE_LETTERS * letter
        and "text changed" not in c.detail and "on the picture" not in c.detail
    ]
    rest = [c for c in findings if not any(c is piece for piece in small)]
    boxes = [(c.bbox.x0, c.bbox.y0, c.bbox.x1, c.bbox.y1) for c in small]
    groups: list[list[int]] = [[i] for i in range(len(small))]
    reach = _PATTERN_REACH_LETTERS * letter
    strip = _STRIP_REACH_LETTERS * letter
    joined = True
    while joined:
        joined = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                near = any(_near(boxes[a], boxes[b], reach) for a in groups[i] for b in groups[j])
                lined_up = not near and any(_in_one_strip(boxes[a], boxes[b], strip) for a in groups[i] for b in groups[j])
                if lined_up:
                    # (a strip stays a strip: joined along it, the group is still narrow across it)
                    xs = [boxes[k][0] for k in groups[i] + groups[j]] + [boxes[k][2] for k in groups[i] + groups[j]]
                    ys = [boxes[k][1] for k in groups[i] + groups[j]] + [boxes[k][3] for k in groups[i] + groups[j]]
                    lined_up = min(max(xs) - min(xs), max(ys) - min(ys)) <= _STRIP_WIDE_LETTERS * letter
                if near or lined_up:
                    groups[i] += groups[j]
                    del groups[j]
                    joined = True
                    break
            if joined:
                break
    out = list(rest)
    for group in groups:
        if len(group) < _PATTERN_MIN_PIECES:
            out += [small[i] for i in group]
            continue
        box = small[group[0]].bbox
        for i in group[1:]:
            box = box.union(small[i].bbox)
        out.append(DiffCandidate(box, DiffType.GRAPHICS, DiffSubtype.CHANGED, 0.8, f"Pattern changed ({len(group)} places)", 0.0))
    return out


def _in_one_strip(a: tuple, b: tuple, reach: float) -> bool:
    """Whether two pieces lie along one strip (a border, a band): sharing a column and at most `reach` apart down it,
    or sharing a row and at most `reach` apart along it."""
    # (sharing it: most of the narrower one's width -- or height -- in common, not a corner of it)
    share_x = min(a[2], b[2]) - max(a[0], b[0]) >= _STRIP_SHARE * min(a[2] - a[0], b[2] - b[0])
    share_y = min(a[3], b[3]) - max(a[1], b[1]) >= _STRIP_SHARE * min(a[3] - a[1], b[3] - b[1])
    gap_y = max(a[1] - b[3], b[1] - a[3], 0.0)
    gap_x = max(a[0] - b[2], b[0] - a[2], 0.0)
    return (share_x and gap_y <= reach) or (share_y and gap_x <= reach)


def _joined_boxes(boxes: list[tuple[int, int, int, int]], reach: float) -> list[tuple[int, int, int, int]]:
    """Boxes within `reach` of each other, joined (each group's box round them all)."""
    groups = [list(box) for box in boxes]
    joined = True
    while joined:
        joined = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = groups[i], groups[j]
                if a[0] - reach <= b[2] and b[0] - reach <= a[2] and a[1] - reach <= b[3] and b[1] - reach <= a[3]:
                    groups[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del groups[j]
                    joined = True
                    break
            if joined:
                break
    return [tuple(group) for group in groups]


def _boxes_near(a: BoundingBox, b: BoundingBox, reach: float) -> bool:
    return a.x0 - reach <= b.x1 and b.x0 - reach <= a.x1 and a.y0 - reach <= b.y1 and b.y0 - reach <= a.y1


def _inside(box: tuple[int, int, int, int], other: tuple[int, int, int, int], share: float = 0.6) -> bool:
    x0, y0, x1, y1 = box
    ox0, oy0, ox1, oy1 = other
    overlap = max(0, min(x1, ox1) - max(x0, ox0)) * max(0, min(y1, oy1) - max(y0, oy0))
    return overlap >= share * max(1, (x1 - x0) * (y1 - y0))


def _how_far(dx: float, dy: float) -> str:
    parts = []
    if abs(dx) >= 1:
        parts.append(f"{abs(dx):.0f} px {'right' if dx > 0 else 'left'}")
    if abs(dy) >= 1:
        parts.append(f"{abs(dy):.0f} px {'down' if dy > 0 else 'up'}")
    return ", ".join(parts)


def _colour_name(lab: np.ndarray) -> str:
    """A plain name for an (8-bit OpenCV) Lab colour: 'light grey', 'red', 'dark blue'..."""
    lightness = float(lab[0]) * 100.0 / 255.0
    a, b = float(lab[1]) - 128.0, float(lab[2]) - 128.0
    if math.hypot(a, b) < 10:
        if lightness < 20:
            return "black"
        if lightness > 92:
            return "white"
        return "dark grey" if lightness < 45 else "light grey" if lightness > 75 else "grey"
    hue = math.degrees(math.atan2(b, a)) % 360
    names = ((20, "pink"), (50, "red"), (75, "orange"), (100, "yellow"), (160, "green"), (220, "teal"), (290, "blue"), (330, "purple"), (360, "pink"))
    name = next(label for limit, label in names if hue < limit)
    if lightness < 35:
        return f"dark {name}"
    if lightness > 75 and name != "yellow":
        return f"light {name}"
    return name


def _colour_change(was: np.ndarray, now: np.ndarray) -> str:
    named_was, named_now = _colour_name(was), _colour_name(now)
    if named_was != named_now:
        return f"{named_was} → {named_now}"
    lighter = (float(now[0]) - float(was[0])) * 100.0 / 255.0
    return f"{named_was}, {'lighter' if lighter > 0 else 'darker'}" if abs(lighter) >= 3 else f"{named_was}, another shade"
