"""Braille dot-pattern comparison (Architecture §3.3.5, FR-11.1).

Braille on packaging artwork is marked as coloured dots, so detection is a
colour-blob problem rather than an OCR one: find dot centroids in both
files (already in the shared aligned space) and match them by nearest
neighbour.

PENDING SIGN-OFF (SRS §7): the dot colour convention used in our source
artwork is an open item, so the HSV ranges live in `DetectionConfig`. The default is red
(measured on the only real Braille files so far) and magenta/pink, not a signed-off value.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from app.config import DetectionConfig
from app.logging_config import StageTimer
from app.models.diff import BoundingBox, DiffCandidate, DiffSubtype, DiffType
from app.models.result import BrailleCellComparison, BrailleComparison, BrailleLineComparison

logger = logging.getLogger(__name__)


def detect_dots(image: np.ndarray, config: DetectionConfig) -> np.ndarray:
    """Centroids of the coloured Braille dots, as an (N, 2) array.

    A dot is a blob in the dot colour that is small and roundish; a letter, a logo or a panel in the same
    colour is not one.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lower, upper in config.braille_hsv_ranges:
        mask |= cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for index in range(1, count):  # 0 is the background
        area = stats[index, cv2.CC_STAT_AREA]
        width, height = stats[index, cv2.CC_STAT_WIDTH], stats[index, cv2.CC_STAT_HEIGHT]
        if not config.braille_dot_min_area_px <= area <= config.braille_dot_max_area_px:
            continue
        if max(width, height) > config.braille_dot_max_aspect * min(width, height):
            continue
        if area < config.braille_dot_min_fill * width * height:
            continue
        x, y = stats[index, cv2.CC_STAT_LEFT], stats[index, cv2.CC_STAT_TOP]
        blob = (_labels[y : y + height, x : x + width] == index).astype(np.uint8)
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = cv2.arcLength(max(contours, key=cv2.contourArea), True) if contours else 0.0
        if perimeter <= 0 or 4 * np.pi * area / perimeter**2 < config.braille_dot_min_circularity:
            continue
        blobs.append((area, centroids[index]))
    if len(blobs) >= config.braille_dot_min_count_for_size:
        median = float(np.median([area for area, _centre in blobs]))
        factor = config.braille_dot_size_factor
        blobs = [(area, centre) for area, centre in blobs if median / factor <= area <= median * factor]
    dots = [centre for _area, centre in blobs]
    return np.array(dots, dtype=np.float64) if dots else np.empty((0, 2))


def _pair_dots(
    original_dots: np.ndarray, suspect_dots: np.ndarray, tolerance: float
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedy nearest-neighbour pairing within a generous search radius.

    The radius is wider than the move tolerance so that a dot which shifted
    is reported as one moved dot rather than as a missing/added pair.
    """
    if len(original_dots) == 0 or len(suspect_dots) == 0:
        return [], list(range(len(original_dots))), list(range(len(suspect_dots)))

    distances = np.linalg.norm(original_dots[:, None, :] - suspect_dots[None, :, :], axis=2)
    search_radius = tolerance * 3

    pairs: list[tuple[int, int, float]] = []
    used_original: set[int] = set()
    used_suspect: set[int] = set()
    for original_index, suspect_index in zip(*np.unravel_index(np.argsort(distances, axis=None), distances.shape)):
        distance = distances[original_index, suspect_index]
        if distance > search_radius:
            break
        if original_index in used_original or suspect_index in used_suspect:
            continue
        used_original.add(int(original_index))
        used_suspect.add(int(suspect_index))
        pairs.append((int(original_index), int(suspect_index), float(distance)))

    missing = [index for index in range(len(original_dots)) if index not in used_original]
    added = [index for index in range(len(suspect_dots)) if index not in used_suspect]
    return pairs, missing, added


def _dot_box(center: np.ndarray, radius: float = 6.0) -> BoundingBox:
    x, y = float(center[0]), float(center[1])
    return BoundingBox(x - radius, y - radius, x + radius, y + radius)


def _region(*dot_sets: np.ndarray) -> BoundingBox | None:
    """The box round every dot found (None if there are none)."""
    dots = [d for d in dot_sets if len(d)]
    if not dots:
        return None
    every = np.vstack(dots)
    (x0, y0), (x1, y1) = every.min(axis=0), every.max(axis=0)
    return _dot_box(np.array([x0, y0])).union(_dot_box(np.array([x1, y1])))


def _dot_findings(
    original_dots: np.ndarray, suspect_dots: np.ndarray, config: DetectionConfig
) -> tuple[list[DiffCandidate], int, int, int, int]:
    """Dots compared one by one (those not set to a Braille grid): (findings, matched, missing, added, moved)."""
    pairs, missing, added = _pair_dots(original_dots, suspect_dots, config.braille_match_tolerance_px)
    candidates: list[DiffCandidate] = []
    moved = 0
    for original_index, suspect_index, distance in pairs:
        if distance > config.braille_match_tolerance_px:
            moved += 1
            candidates.append(
                DiffCandidate(
                    bbox=_dot_box(original_dots[original_index]).union(_dot_box(suspect_dots[suspect_index])),
                    type=DiffType.BRAILLE,
                    subtype=DiffSubtype.MOVED,
                    confidence=1.0,
                    detail=f"Braille dot moved {distance:.0f} px",
                )
            )
    for index in missing:
        candidates.append(
            DiffCandidate(_dot_box(original_dots[index]), DiffType.BRAILLE, DiffSubtype.REMOVED, 1.0, "Braille dot missing from suspect")
        )
    for index in added:
        candidates.append(
            DiffCandidate(_dot_box(suspect_dots[index]), DiffType.BRAILLE, DiffSubtype.ADDED, 1.0, "Braille dot added in suspect")
        )
    return candidates, len(pairs) - moved, len(missing), len(added), moved


def compare(
    original: np.ndarray, suspect: np.ndarray, config: DetectionConfig
) -> tuple[list[DiffCandidate], BrailleComparison]:
    """Braille compared (FR-11.1), cell by cell: each file's Braille read (`braille_reader`) -- its cells, the dots of
    each, what each says -- and the original's cells set against the suspect's, line by line, in order. A cell whose
    dots differ is a finding saying which letter each file has there and which dots were added or are missing; a cell
    or a line one file has and the other has not, likewise. Dots that are not set to a Braille grid (too few, or
    scattered) are compared dot by dot, as before."""
    from app.engine.detectors import braille_reader as reader  # (it reads with this module's dot finders)

    with StageTimer(logger, "detector_braille"):
        original_dots = reader.find_braille_dots(original, config)
        suspect_dots = reader.find_braille_dots(suspect, config)
        blocks, loose_original = reader.read_dots(original_dots)
        used = np.zeros(len(suspect_dots), bool)
        counts = {"matched": 0, "missing": 0, "added": 0, "moved": 0}
        lines: list[BrailleLineComparison] = []
        candidates: list[DiffCandidate] = []
        stray_original: list[np.ndarray] = []
        stray_suspect: list[np.ndarray] = []
        for block in blocks:
            # the suspect's dots where this block is, read the same way round (the files are aligned: it is not turned)
            theirs = None
            if len(suspect_dots):
                margin = 2.5 * block.spacing
                box = block.bbox
                inside = (
                    ~used
                    & (suspect_dots[:, 0] >= box.x0 - margin)
                    & (suspect_dots[:, 0] <= box.x1 + margin)
                    & (suspect_dots[:, 1] >= box.y0 - margin)
                    & (suspect_dots[:, 1] <= box.y1 + margin)
                )
                if inside.sum() >= 2:
                    theirs = reader.read_block(suspect_dots[inside], [(block.turn, block.mirrored)])
                if theirs is not None:
                    used |= inside
            _compare_block(block, theirs, config, lines, candidates, counts)
            # (dots off a block's grid -- a stray dot, one set between cells -- are compared one by one)
            stray_original.append(block.stray)
            if theirs is not None:
                stray_suspect.append(theirs.stray)
        suspect_blocks, loose_suspect = reader.read_dots(suspect_dots[~used]) if len(suspect_dots) else ([], suspect_dots)
        for block in suspect_blocks:
            _compare_block(None, block, config, lines, candidates, counts)
            stray_suspect.append(block.stray)
        loose_original = np.vstack([loose_original.reshape(-1, 2), *stray_original])
        loose_suspect = np.vstack([np.asarray(loose_suspect).reshape(-1, 2), *stray_suspect])
        dot_candidates, matched, missing, added, moved = _dot_findings(loose_original, loose_suspect, config)
        candidates += dot_candidates

        comparison = BrailleComparison(
            original_dot_count=len(original_dots),
            suspect_dot_count=len(suspect_dots),
            matched=counts["matched"] + matched,
            missing=counts["missing"] + missing,
            added=counts["added"] + added,
            moved=counts["moved"] + moved,
            region=_region(original_dots, suspect_dots),
            lines=lines,
        )

    logger.info(
        "braille detector finished",
        extra={
            "original_dots": comparison.original_dot_count,
            "suspect_dots": comparison.suspect_dot_count,
            "lines": len(lines),
            "cells": comparison.cells_compared,
            "cells_differing": comparison.cells_differing,
            "missing": comparison.missing,
            "added": comparison.added,
            "moved": comparison.moved,
        },
    )
    return candidates, comparison


def _said(meaning: str) -> str:
    """A cell's meaning as written in a finding: a letter quoted, a sign by name."""
    if meaning == " ":
        return "a space"
    return f"'{meaning}'" if len(meaning) <= 3 else f"the {meaning}"


def _cell_said(meaning: str, pattern: int) -> str:
    from app.engine.detectors.braille_reader import dots_text, glyph

    return f"{_said(meaning)} {glyph(pattern)} ({dots_text(pattern)})"


def _bits(pattern: int) -> int:
    return bin(pattern).count("1")


def _compare_block(mine, theirs, config: DetectionConfig, lines: list, candidates: list, counts: dict) -> None:
    """One block of the original (`mine`) against the suspect's there (`theirs`); either may be None. Lines are paired
    by where they are; a line only one file has is one finding."""
    mine_lines = list(mine.lines) if mine is not None else []
    their_lines = list(theirs.lines) if theirs is not None else []
    spacing = (mine or theirs).spacing
    pairs, left = [], list(range(len(their_lines)))
    for line in mine_lines:
        centre = line.bbox.center
        reach = 2.0 * spacing + 0.5 * max(line.bbox.width, line.bbox.height)
        near = [
            (float(np.hypot(their_lines[j].bbox.center[0] - centre[0], their_lines[j].bbox.center[1] - centre[1])), j)
            for j in left
        ]
        near = [(distance, j) for distance, j in near if distance <= reach]
        if near:
            _distance, j = min(near)
            left.remove(j)
            pairs.append((line, their_lines[j]))
        else:
            pairs.append((line, None))
    pairs += [(None, their_lines[j]) for j in left]  # (in the order the block is read, the suspect's extra lines last)
    for original_line, suspect_line in pairs:
        lines.append(_compare_line(original_line, suspect_line, len(lines) + 1, spacing, config, candidates, counts))


def _compare_line(
    original_line, suspect_line, number: int, spacing: float, config: DetectionConfig, candidates: list, counts: dict
) -> BrailleLineComparison:
    """One line, cell by cell: each cell of the original against the suspect's in its place, dot for dot."""
    from difflib import SequenceMatcher

    from app.engine.detectors.braille_reader import dots_of, text_of, translate

    mine = original_line.patterns() if original_line is not None else []
    theirs = suspect_line.patterns() if suspect_line is not None else []
    mine_said, their_said = translate(mine), translate(theirs)
    cells: list[BrailleCellComparison] = []

    if original_line is None or suspect_line is None:
        # a whole line one file has and the other has not: one finding for it
        line = original_line or suspect_line
        missing = suspect_line is None
        said = mine_said if missing else their_said
        first = line.cells[0].index
        for cell in line.cells:
            meaning = said[cell.index - first]
            cells.append(
                BrailleCellComparison(
                    cell.pattern if missing else None,
                    None if missing else cell.pattern,
                    meaning if missing else "",
                    "" if missing else meaning,
                    cell.bbox,
                    "missing" if missing else "added",
                )
            )
            counts["missing" if missing else "added"] += _bits(cell.pattern)
        text = text_of(mine if missing else theirs)
        candidates.append(
            DiffCandidate(
                bbox=line.bbox,
                type=DiffType.BRAILLE,
                subtype=DiffSubtype.REMOVED if missing else DiffSubtype.ADDED,
                confidence=1.0,
                detail=f"Braille line {number}: '{text}' ({len(line.cells)} cells) "
                + ("missing from suspect" if missing else "added in suspect"),
            )
        )
        return BrailleLineComparison(text if missing else None, None if missing else text, cells, line.bbox)

    mine_cells = {cell.index - original_line.cells[0].index: cell for cell in original_line.cells}
    their_cells = {cell.index - suspect_line.cells[0].index: cell for cell in suspect_line.cells}
    where = f"Braille line {number} ('{text_of(mine)}')"

    def add(i: int | None, j: int | None, status: str | None = None) -> None:
        mine_cell = mine_cells.get(i) if i is not None else None
        their_cell = their_cells.get(j) if j is not None else None
        if mine_cell is None and their_cell is None:
            return  # (a space against a space)
        mine_pattern = mine_cell.pattern if mine_cell else None
        their_pattern = their_cell.pattern if their_cell else None
        if status is None:
            if mine_cell is None:
                status = "added"
            elif their_cell is None:
                status = "missing"
            elif mine_pattern != their_pattern:
                status = "changed"
            else:
                status = "moved" if moved else "match"
        box = mine_cell.bbox if mine_cell else their_cell.bbox
        if mine_cell and their_cell:
            box = mine_cell.bbox.union(their_cell.bbox)
        mine_meaning = mine_said[i] if mine_cell else ""
        their_meaning = their_said[j] if their_cell else ""
        cells.append(BrailleCellComparison(mine_pattern, their_pattern, mine_meaning, their_meaning, box, status))
        place = f"{where}, cell {(i if mine_cell else j) + 1}"
        if status == "match":
            counts["matched"] += _bits(mine_pattern)
            return
        if status == "moved":
            counts["moved"] += _bits(mine_pattern)
            return  # (the line moved: one finding for it, below)
        if status == "changed":
            gone = [dot for dot in dots_of(mine_pattern) if dot not in dots_of(their_pattern)]
            new = [dot for dot in dots_of(their_pattern) if dot not in dots_of(mine_pattern)]
            counts["matched"] += _bits(mine_pattern & their_pattern)
            counts["missing"] += len(gone)
            counts["added"] += len(new)
            change = [f"dot{'s' if len(gone) > 1 else ''} {'-'.join(map(str, gone))} missing"] if gone else []
            change += [f"dot{'s' if len(new) > 1 else ''} {'-'.join(map(str, new))} added"] if new else []
            detail = (
                f"{place}: {_cell_said(mine_meaning, mine_pattern)} in the original, "
                f"{_cell_said(their_meaning, their_pattern)} in the suspect — {', '.join(change)}"
            )
            subtype = DiffSubtype.CHANGED
        elif status == "missing":
            counts["missing"] += _bits(mine_pattern)
            detail = f"{place}: {_cell_said(mine_meaning, mine_pattern)} missing from suspect"
            subtype = DiffSubtype.REMOVED
        else:
            counts["added"] += _bits(their_pattern)
            detail = f"{where}: {_cell_said(their_meaning, their_pattern)} added in suspect"
            subtype = DiffSubtype.ADDED
        candidates.append(DiffCandidate(bbox=box, type=DiffType.BRAILLE, subtype=subtype, confidence=1.0, detail=detail))

    # Cells are compared where they are -- Braille is set to fixed places on a pack, and the files are aligned -- once
    # the line's own shift is allowed for (the cells that read the same, in order, say how far the suspect's line sits
    # from the original's). A cell is paired with the suspect's cell in its place; one with none there is missing, or
    # added. A line the suspect has shifted further than the tolerance is one finding: moved.
    shifts = [
        (their_cells[j].centre[0] - mine_cells[i].centre[0], their_cells[j].centre[1] - mine_cells[i].centre[1])
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, mine, theirs, autojunk=False).get_opcodes()
        if tag == "equal"
        for i, j in zip(range(i1, i2), range(j1, j2))
        if i in mine_cells and j in their_cells
    ]
    dx, dy = (float(np.median([s[0] for s in shifts])), float(np.median([s[1] for s in shifts]))) if shifts else (0.0, 0.0)
    shift = float(np.hypot(dx, dy))
    moved = shift > config.braille_match_tolerance_px
    reach = 0.6 * spacing
    free = set(their_cells)
    for i in sorted(mine_cells):
        cx, cy = mine_cells[i].centre
        near = [(float(np.hypot(their_cells[j].centre[0] - dx - cx, their_cells[j].centre[1] - dy - cy)), j) for j in free]
        near = [(distance, j) for distance, j in near if distance <= reach]
        if near:
            j = min(near)[1]
            free.discard(j)
            add(i, j)
        else:
            add(i, None, "missing")
    for j in sorted(free):
        add(None, j, "added")
    if moved:
        candidates.append(
            DiffCandidate(
                bbox=original_line.bbox.union(suspect_line.bbox),
                type=DiffType.BRAILLE,
                subtype=DiffSubtype.MOVED,
                confidence=1.0,
                detail=f"{where} moved {shift:.0f} px in the suspect",
            )
        )
    return BrailleLineComparison(text_of(mine), text_of(theirs), cells, original_line.bbox.union(suspect_line.bbox))


# ---------------------------------------------------------------------------
# Braille found by its shape and arrangement, whatever its colour
# ---------------------------------------------------------------------------

_DOT_MIN_AREA_PX = 20
_DOT_MAX_ASPECT = 1.35  # a dot is round: about as wide as it is tall...
_DOT_MIN_FILL = 0.65  # ...fills its box as a disc does (~0.79)...
_DOT_MIN_CIRCULARITY = 0.75  # ...and its outline is a circle's
_DOT_SIZE_SPREAD = 1.5  # Braille's dots are one size: within this factor of the commonest round mark's area
_DOT_DARKER = 18  # (grey levels) a dot is at least this much darker than what is round it...
_DOT_REACH = 2.5  # ...judged against its surroundings this many dot widths across; or else
_DOT_CONTRAST = 24  # (levels, in any channel) it differs in colour from what is round it by at least this much...
_DOT_SURROUND_PX = 61  # ...judged against a window this wide (at 300 DPI: ~5 mm, a couple of Braille cells)
_DOT_PITCH = (1.3, 2.4)  # a Braille dot has another dot this many dot widths away (its cell's neighbour)
_BRAILLE_MIN_DOTS = 12  # fewer dots so arranged than this is not Braille (a few round bullets, full stops)
_DOT_RUN_REACH = 4.2  # a Braille word's dots are no further than this many dot widths from the next (a cell's width)
_DOT_RUN_MIN = 6  # ...and a word has at least this many of them
_DOT_MASK_GROW = 0.15  # each dot is left out this much (share of its width) wider than it is


def _round_marks(mask: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """(area, centre) of every solid round mark in `mask`."""
    count, labels, stats, centres = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    found = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < _DOT_MIN_AREA_PX or max(w, h) > _DOT_MAX_ASPECT * min(w, h) or area < _DOT_MIN_FILL * w * h:
            continue
        blob = (labels[y : y + h, x : x + w] == index).astype(np.uint8)
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = cv2.arcLength(max(contours, key=cv2.contourArea), True) if contours else 0.0
        if perimeter > 0 and 4 * np.pi * area / perimeter**2 >= _DOT_MIN_CIRCULARITY:
            found.append((int(area), centres[index]))
    return found


def _standing_out(image: np.ndarray) -> np.ndarray:
    """Where a picture differs from what is round it (a window a few Braille cells across), in any colour channel: a
    dot on paper, on a panel, a magenta dot on a green of the same lightness -- darker or lighter, as long as it is not
    the same colour as its surroundings."""
    channels = cv2.split(image) if image.ndim == 3 else [image]
    height, width = channels[0].shape
    small_size = (max(1, width // 2), max(1, height // 2))
    window = max(3, (_DOT_SURROUND_PX // 2) | 1)
    most = np.zeros((height, width), np.uint8)
    for channel in channels:
        around = cv2.resize(cv2.medianBlur(cv2.resize(channel, small_size, interpolation=cv2.INTER_AREA), window), (width, height), interpolation=cv2.INTER_LINEAR)
        cv2.max(most, cv2.absdiff(channel, around), dst=most)
    return most > _DOT_CONTRAST


def _dot_size(marks: np.ndarray) -> float | None:
    """The commonest area among the round marks of `marks` standing on their own (a Braille dot's, where it has any)."""
    alone = _round_marks(marks)
    if len(alone) < _BRAILLE_MIN_DOTS // 2:
        return None
    counts, edges = np.histogram(np.log([area for area, _centre in alone]), bins=30)
    peak = int(counts.argmax())
    return float(np.exp((edges[peak] + edges[peak + 1]) / 2))


def _dots_in(marks: np.ndarray, area: float) -> list[np.ndarray]:
    """The dot-sized round marks of `marks` -- dots touching type or each other too: what is dot-wide throughout."""
    width = 2.0 * np.sqrt(area / np.pi)
    core = max(1, int(0.4 * width))
    solid = cv2.morphologyEx(marks, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * core + 1, 2 * core + 1)))
    spread = _DOT_SIZE_SPREAD * 1.2
    return [centre for size, centre in _round_marks(solid) if area / spread <= size <= area * spread]


def _in_runs(centres: np.ndarray, candidates: list[int], reach: float) -> list[int]:
    """The candidate dots that belong to a run of at least `_DOT_RUN_MIN` dots, each within `reach` of the next."""
    parent = {index: index for index in candidates}

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    points = centres[candidates]
    for i, first in enumerate(candidates):
        near = np.flatnonzero(np.hypot(*(points - points[i]).T) <= reach)
        for j in near:
            parent[root(first)] = root(candidates[j])
    sizes: dict[int, int] = {}
    for index in candidates:
        sizes[root(index)] = sizes.get(root(index), 0) + 1
    return [index for index in candidates if sizes[root(index)] >= _DOT_RUN_MIN]


def find_dots(image: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Where a file draws Braille: its dots' centres (an (N, 2) array) and their width -- found by what they ARE,
    whatever their colour: solid round marks all of one size, each with another at Braille's spacing from it -- on paper
    or on a panel, and where they sit on lettering. None if it has none. (`detect_dots` finds dots by their colour.)

    Dots are looked for twice -- as marks DARKER than what is round them (a dark dot, on paper or a panel), and as marks
    of another COLOUR than what is round them (a magenta dot on a green as light as itself) -- and both sets taken."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    # the dots' size: from the dark marks standing on their own, else from the coloured ones
    darker_marks = (gray < np.percentile(gray, 50) * 0.75).astype(np.uint8)
    coloured_marks = _standing_out(image).astype(np.uint8)
    area = _dot_size(darker_marks) or _dot_size(coloured_marks)
    if area is None:
        return None
    width = 2.0 * np.sqrt(area / np.pi)
    reach = max(3, int(_DOT_REACH * width) | 1)
    darker = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach))) > _DOT_DARKER
    dots = _dots_in(darker.astype(np.uint8), area)
    for centre in _dots_in(coloured_marks, area):  # (the same dot found both ways is taken once)
        if not any(np.hypot(*(centre - other)) < 0.5 * width for other in dots):
            dots.append(centre)
    if len(dots) < _BRAILLE_MIN_DOTS:
        return None
    centres = np.array(dots, np.float64)
    low, high = _DOT_PITCH[0] * width, _DOT_PITCH[1] * width
    kept = []
    for start in range(0, len(centres), 256):  # (pairwise distances, a block at a time)
        block = centres[start : start + 256]
        distances = np.sqrt(((block[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2))
        kept.extend(start + np.flatnonzero(((distances >= low) & (distances <= high)).any(axis=1)))
    # ...and Braille runs in words: its cells a few dot widths apart, many dots together. Round parts of letters taken
    # for dots come in twos and threes, not so.
    kept = _in_runs(centres, kept, _DOT_RUN_REACH * width)
    if len(kept) < _BRAILLE_MIN_DOTS:
        return None
    return centres[kept], float(width)


def braille_mask(image: np.ndarray) -> tuple[np.ndarray, int] | None:
    """Where a file draws Braille, as a mask (each dot of `find_dots`, a little wider than it is), and how many dots; None
    if it has none."""
    found = find_dots(image)
    if found is None:
        return None
    centres, width = found
    mask = np.zeros(image.shape[:2], np.uint8)
    radius = int(round(width * (0.5 + _DOT_MASK_GROW)))
    for x, y in centres:
        cv2.circle(mask, (int(round(x)), int(round(y))), radius, 1, -1)
    return mask.astype(bool), len(centres)
