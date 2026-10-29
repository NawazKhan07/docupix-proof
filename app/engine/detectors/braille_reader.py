"""Braille read cell by cell: which dots make each cell, and what each cell says (Braille mode, SRS FR-11.1).

A Braille cell is two columns of three dot places; its dots are numbered 1-2-3 down the left column and 4-5-6 down the
right. On a pack it is set to a fixed grid -- the dots of a cell a fixed spacing apart (Marburg Medium: 2.5 mm), cells a
little over twice that (6 mm), lines four times it (10 mm) -- and that grid is what is found here:

* **The dots**, by their shape (`braille_diff.find_dots`), or, where none are found so, by their colour (`detect_dots`).
* **Blocks**: dots near one another (a few cells' reach) are one block -- the Braille on a panel, a legend beside it.
* **The grid of each block**, fitted to its dots: the dot spacing, the cell and line pitch, and where the grid starts,
  across (cells, two dot columns each) and down (lines, three dot rows each). A block printed turned (upside down on a
  carton's panel, on its side) is fitted every way round, and mirrored too -- the way round it reads as Braille (its
  cells letters, numbers, signs) is the way it is read.
* **The cells**: each dot is a place in its cell, so each cell is its dots -- the pattern compared -- and what they say
  (`translate`): uncontracted (grade 1) Braille, letters, numbers after the number sign, capitals, punctuation, and the
  commoner accented letters. What a pattern means depends on the language's Braille code; the PATTERN is what is
  compared, exactly, whatever it is read as.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from app.config import DetectionConfig
from app.engine.detectors import braille_diff
from app.models.diff import BoundingBox

logger = logging.getLogger(__name__)

# dot n of a cell is bit n - 1 (as Unicode's Braille patterns number them: U+2800 + the bits)
DOT_BITS = {1: 0x01, 2: 0x02, 3: 0x04, 4: 0x08, 5: 0x10, 6: 0x20}

_LETTERS = {
    0x01: "a", 0x03: "b", 0x09: "c", 0x19: "d", 0x11: "e", 0x0B: "f", 0x1B: "g", 0x13: "h", 0x0A: "i", 0x1A: "j",
    0x05: "k", 0x07: "l", 0x0D: "m", 0x1D: "n", 0x15: "o", 0x0F: "p", 0x1F: "q", 0x17: "r", 0x0E: "s", 0x1E: "t",
    0x25: "u", 0x27: "v", 0x3A: "w", 0x2D: "x", 0x3D: "y", 0x35: "z",
}
_DIGITS = {0x01: "1", 0x03: "2", 0x09: "3", 0x19: "4", 0x11: "5", 0x0B: "6", 0x1B: "7", 0x13: "8", 0x0A: "9", 0x1A: "0"}
_NUMBER_SIGN, _CAPITAL_SIGN, _LETTER_SIGN = 0x3C, 0x20, 0x30
_PUNCTUATION = {0x02: ",", 0x06: ";", 0x12: ":", 0x32: ".", 0x16: "!", 0x26: "?", 0x04: "'", 0x24: "-", 0x36: "( )"}
_IN_NUMBERS = {0x32: ".", 0x02: ","}  # (a decimal point and a thousands comma keep a number going)
# accented letters of the commoner European Braille codes (French, German, Spanish, Italian); a pattern two codes read
# differently is given both ways
_ACCENTED = {
    0x3F: "é", 0x2E: "è/ß", 0x37: "à", 0x2F: "ç", 0x21: "â", 0x23: "ê", 0x29: "î", 0x39: "ô", 0x31: "û", 0x2B: "ë",
    0x3B: "ï", 0x33: "ü", 0x3E: "ù", 0x2A: "œ/ö", 0x1C: "ä",
}
_SIGN_NAMES = {_NUMBER_SIGN: "number sign", _CAPITAL_SIGN: "capital sign", _LETTER_SIGN: "letter sign"}


def dots_of(pattern: int) -> list[int]:
    """The dots of a cell's pattern, by number."""
    return [dot for dot, bit in DOT_BITS.items() if pattern & bit]


def glyph(pattern: int) -> str:
    """The cell as Unicode's Braille pattern character (a blank cell is U+2800)."""
    return chr(0x2800 + pattern)


def dots_text(pattern: int) -> str:
    """'dots 1-3-4' (or 'no dots')."""
    numbers = dots_of(pattern)
    return f"dot{'s' if len(numbers) > 1 else ''} {'-'.join(map(str, numbers))}" if numbers else "no dots"


def translate(patterns: list[int]) -> list[str]:
    """What each cell of a line says (a blank cell, 0, is a space): a letter, a digit, a sign, or '?' for a pattern none
    of the codes known here gives a meaning. Uncontracted Braille: the number sign makes the letters a-j that follow it
    digits until a space or another cell; the capital sign makes the next letter a capital (twice: the whole word)."""
    meanings: list[str] = []
    number = capital = capital_word = False
    for pattern in patterns:
        if pattern == 0:
            meanings.append(" ")
            number = capital = capital_word = False
            continue
        if pattern in _SIGN_NAMES:
            meanings.append(_SIGN_NAMES[pattern])
            if pattern == _NUMBER_SIGN:
                number = True
            elif pattern == _LETTER_SIGN:
                number = False
            else:
                capital_word, capital = capital, True
            continue
        if number and pattern in _DIGITS:
            meanings.append(_DIGITS[pattern])
            continue
        if number and pattern in _IN_NUMBERS:
            meanings.append(_IN_NUMBERS[pattern])
            continue
        number = False
        if pattern in _LETTERS:
            letter = _LETTERS[pattern]
            meanings.append(letter.upper() if capital or capital_word else letter)
            capital = False
        else:
            meanings.append(_PUNCTUATION.get(pattern) or _ACCENTED.get(pattern) or "?")
    return meanings


def text_of(patterns: list[int]) -> str:
    """A line's cells as text: the signs say what follows rather than standing for anything themselves."""
    return "".join(meaning for meaning in translate(patterns) if meaning not in _SIGN_NAMES.values())


@dataclass(frozen=True)
class BrailleCell:
    """One cell: its dots, where it is (the image's pixels), and its place in its line (lattice index along it)."""

    pattern: int
    bbox: BoundingBox
    index: int
    centre: tuple[float, float]


@dataclass
class BrailleLine:
    cells: list[BrailleCell]  # in reading order, blank places left out (`index` says where each one is)

    def patterns(self) -> list[int]:
        """The line's cells in order, a blank place between two cells as 0 (a space)."""
        if not self.cells:
            return []
        by_place = {cell.index: cell.pattern for cell in self.cells}
        first, last = self.cells[0].index, self.cells[-1].index
        return [by_place.get(place, 0) for place in range(first, last + 1)]

    @property
    def text(self) -> str:
        return text_of(self.patterns())

    @property
    def bbox(self) -> BoundingBox:
        box = self.cells[0].bbox
        for cell in self.cells[1:]:
            box = box.union(cell.bbox)
        return box


@dataclass
class BrailleBlock:
    """A block of Braille read: its lines in reading order, which way round it was read, its dot spacing (pixels)."""

    lines: list[BrailleLine]
    turn: int  # degrees the Braille is turned on the page (0, 90, 180, 270), clockwise
    mirrored: bool
    spacing: float
    dots: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, 2)))
    stray: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, 2)))  # its dots off the grid

    @property
    def bbox(self) -> BoundingBox:
        box = self.lines[0].bbox
        for line in self.lines[1:]:
            box = box.union(line.bbox)
        return box

    @property
    def cells(self) -> list[BrailleCell]:
        return [cell for line in self.lines for cell in line.cells]


# --- the grid -----------------------------------------------------------------------------------------------------------

_CELL_PITCH = (2.2, 3.2)  # cells this many dot spacings apart (Marburg Medium: 6 / 2.5 = 2.4)
_LINE_PITCH = (3.0, 6.0)  # lines this many (10 / 2.5 = 4)
_CELL_PITCH_USUAL, _LINE_PITCH_USUAL = 2.4, 4.0  # (a block of one cell or one line cannot say: the usual is taken)
_CLUSTER_SHARE = 0.3  # dots this share of a spacing apart, across, are in one row (or one column)
_FIT_SHARE = 0.22  # a dot is in its grid place if within this share of a spacing of it
_MIN_FIT = 0.9  # a block is read if this share of its dots sit in their grid places
_BLOCK_REACH = 7.0  # dots within this many spacings of one another are one block (a word gap is ~4)
_MIN_BLOCK_DOTS = 6
_TURNS = [(turn, mirrored) for mirrored in (False, True) for turn in (0, 90, 180, 270)]


def _turned(points: np.ndarray, turn: int, mirrored: bool) -> np.ndarray:
    """`points` (image x, y) as the Braille is read when it is printed turned `turn` degrees clockwise (and mirrored):
    reading x runs along its lines, reading y down them."""
    x, y = points[:, 0], points[:, 1]
    if turn == 0:
        u, v = x, y
    elif turn == 90:
        u, v = y, -x
    elif turn == 180:
        u, v = -x, -y
    else:
        u, v = -y, x
    return np.column_stack([-u if mirrored else u, v])


def _unturned(points: np.ndarray, turn: int, mirrored: bool) -> np.ndarray:
    """The inverse of `_turned`: reading coordinates back to the image's."""
    u, v = points[:, 0], points[:, 1]
    if mirrored:
        u = -u
    if turn == 0:
        x, y = u, v
    elif turn == 90:
        x, y = -v, u
    elif turn == 180:
        x, y = -u, -v
    else:
        x, y = v, -u
    return np.column_stack([x, y])


def _rows(values: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """1-D clusters of `values` (no gap wider than `tolerance` inside one): (centres, how many in each)."""
    order = np.sort(values)
    groups = [[order[0]]]
    for value in order[1:]:
        if value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return np.array([np.mean(group) for group in groups]), np.array([len(group) for group in groups], float)


def _fit_axis(
    centres: np.ndarray, weights: np.ndarray, spacing: float, places: int, pitch_range: tuple[float, float], usual: float
) -> tuple[float, float, float] | None:
    """The grid along one axis: units (cells, or lines) `pitch` apart, each with `places` dot places `spacing` apart,
    from `origin`. Fitted to the row (or column) `centres`: (pitch, origin, cost); the cost is the share of the dots
    off their places (of two rows taking one place, the one of fewer dots is off)."""
    low, high = pitch_range[0] * spacing, pitch_range[1] * spacing
    candidates = {usual * spacing}
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            difference = abs(centres[j] - centres[i])
            for k in range(1, 7):
                if low <= difference / k <= high:
                    candidates.add(difference / k)
    offsets = np.arange(places) * spacing
    tolerance = _FIT_SHARE * spacing
    total = float(weights.sum())
    best = None
    for pitch in sorted({round(candidate * 4) / 4 for candidate in candidates}):
        # every start that puts some row on some place, once each (starts a pitch apart are one grid)
        origins = np.unique(np.round(((centres[:, None] - offsets[None, :]).ravel() % pitch) * 2) / 2)
        unit = np.round((centres[:, None, None] - origins[None, :, None] - offsets[None, None, :]) / pitch)
        error = np.abs(centres[:, None, None] - (origins[None, :, None] + unit * pitch + offsets[None, None, :]))
        chosen = error.argmin(axis=2)  # (centre, origin)
        least = np.take_along_axis(error, chosen[..., None], axis=2)[..., 0]
        units = np.take_along_axis(unit, chosen[..., None], axis=2)[..., 0].astype(int)
        off = least > tolerance
        # two rows on one place: the one of fewer dots is off (a stray round mark beside a row of Braille is not it)
        slot = units * places + chosen
        for o in range(len(origins)):
            values, counts = np.unique(slot[:, o], return_counts=True)
            for value in values[counts > 1]:
                sharing = np.flatnonzero((slot[:, o] == value) & ~off[:, o])
                if len(sharing) > 1:
                    off[sharing[sharing != sharing[np.argmax(weights[sharing])]], o] = True
        costs = (weights[:, None] * off).sum(axis=0) / total + 0.002 * abs(pitch / spacing - usual)
        o = int(costs.argmin())
        if best is None or costs[o] < best[2]:
            best = (pitch, float(origins[o]), float(costs[o]))
    return best


def _places(values: np.ndarray, pitch: float, origin: float, spacing: float, places: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each value's (unit, place in the unit) on the fitted grid, and how far off it it is."""
    offsets = np.arange(places) * spacing
    unit = np.round((values[:, None] - origin - offsets[None, :]) / pitch)
    error = np.abs(values[:, None] - (origin + unit * pitch + offsets[None, :]))
    chosen = error.argmin(axis=1)
    rows = np.arange(len(values))
    return unit[rows, chosen].astype(int), chosen, error[rows, chosen]


def _spacing(points: np.ndarray) -> float:
    """The dot spacing of a block: the commonest distance to a nearest neighbour (two dots of a cell, side by side or one
    above the other), refined to the steps between its rows and columns."""
    distances = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    first = float(np.percentile(nearest, 30))
    steps = []
    for axis in (0, 1):
        values = np.sort(points[:, axis])
        gaps = np.diff(np.unique(np.round(values, 1)))
        steps += [gap for gap in gaps if 0.75 * first <= gap <= 1.3 * first]
    return float(np.median(steps)) if steps else first


def _plausibility(lines: list[list[int]]) -> float:
    """How much a block read one way round reads as Braille: its cells letters, digits and signs, used as they are."""
    score = 0.0
    for patterns in lines:
        meanings = translate(patterns)
        for pattern, meaning in zip(patterns, meanings):
            if pattern == 0:
                continue
            if meaning == "?":
                score -= 1.5
            elif meaning.isalpha() and len(meaning) == 1 and meaning.lower() in "etaoinsrhlcdmu":
                score += 1.3
            elif meaning.isalnum() and len(meaning) == 1:
                score += 1.0
            elif meaning in _SIGN_NAMES.values():
                score += 0.6
            else:
                score += 0.2
        # (a sign that leads nowhere -- a number sign before no digit, a capital sign before no letter -- is a misreading)
        for here, after in zip(meanings, meanings[1:] + [" "]):
            if here == "number sign" and not after.isdigit():
                score -= 2.0
            if here == "capital sign" and not (after.isalpha() or after == "capital sign"):
                score -= 2.0
    return score


def read_block(points: np.ndarray, ways: list[tuple[int, bool]] = _TURNS) -> BrailleBlock | None:
    """One block of dots read: its grid fitted each way round of `ways` (every way, unless said), the way it reads as
    Braille kept (None if no way round fits a Braille grid)."""
    spacing = _spacing(points)
    tolerance = _CLUSTER_SHARE * spacing
    tried = []
    for turn, mirrored in ways:
        reading = _turned(points, turn, mirrored)
        row_centres, row_weights = _rows(reading[:, 1], tolerance)
        down = _fit_axis(row_centres, row_weights, spacing, 3, _LINE_PITCH, _LINE_PITCH_USUAL)
        if down is None or down[2] > 1 - _MIN_FIT:
            continue
        line_pitch, line_origin, _cost = down
        line, row, row_error = _places(reading[:, 1], line_pitch, line_origin, spacing, 3)
        # across, line by line: a pitch for the block, a start for each line (lines are not always set flush)
        line_numbers = sorted(set(line.tolist()))
        columns = {number: _rows(reading[line == number, 0], tolerance) for number in line_numbers}
        widest = max(line_numbers, key=lambda number: len(columns[number][0]))
        across = _fit_axis(*columns[widest], spacing, 2, _CELL_PITCH, _CELL_PITCH_USUAL)
        if across is None:
            continue
        cell_pitch = across[0]
        cell = np.zeros(len(points), int)
        column = np.zeros(len(points), int)
        column_error = np.zeros(len(points))
        origins = {}
        for number in line_numbers:
            fitted = _fit_axis(*columns[number], spacing, 2, (cell_pitch / spacing,) * 2, cell_pitch / spacing)
            origin = fitted[1] if fitted is not None else across[1]
            origins[number] = origin
            members = line == number
            cell[members], column[members], column_error[members] = _places(reading[members, 0], cell_pitch, origin, spacing, 2)
        placed = (row_error <= _FIT_SHARE * spacing) & (column_error <= _FIT_SHARE * spacing)
        if placed.mean() < _MIN_FIT:
            continue
        patterns: dict[tuple[int, int], int] = {}
        for index in np.flatnonzero(placed):
            dot = 1 + int(row[index]) + 3 * int(column[index])
            key = (int(line[index]), int(cell[index]))
            patterns[key] = patterns.get(key, 0) | DOT_BITS[dot]
        lines_patterns = []
        for number in line_numbers:
            places = sorted(key[1] for key in patterns if key[0] == number)
            if places:
                lines_patterns.append([patterns.get((number, place), 0) for place in range(places[0], places[-1] + 1)])
        score = _plausibility(lines_patterns) - (0.5 if mirrored else 0.0)
        tried.append((score, turn, mirrored, line_pitch, line_origin, cell_pitch, origins, patterns, placed.mean(), placed))
    if not tried:
        return None
    score, turn, mirrored, line_pitch, line_origin, cell_pitch, origins, patterns, _fit, placed = max(tried, key=lambda t: (t[0], t[8]))
    half = 0.45 * spacing
    lines = []
    for number in sorted({key[0] for key in patterns}):
        cells = []
        for (line_number, place), pattern in sorted(patterns.items()):
            if line_number != number:
                continue
            u0 = origins[number] + place * cell_pitch
            v0 = line_origin + number * line_pitch
            corners = _unturned(
                np.array([[u0 - half, v0 - half], [u0 + spacing + half, v0 - half], [u0 - half, v0 + 2 * spacing + half], [u0 + spacing + half, v0 + 2 * spacing + half]]),
                turn,
                mirrored,
            )
            (x0, y0), (x1, y1) = corners.min(axis=0), corners.max(axis=0)
            centre = _unturned(np.array([[u0 + spacing / 2, v0 + spacing]]), turn, mirrored)[0]
            cells.append(BrailleCell(pattern, BoundingBox(float(x0), float(y0), float(x1), float(y1)), place, (float(centre[0]), float(centre[1]))))
        lines.append(BrailleLine(cells))
    return BrailleBlock(lines, turn, mirrored, spacing, points[placed], points[~placed])


def blocks_of(points: np.ndarray, reach: float) -> list[np.ndarray]:
    """The dots split into blocks: each within `reach` of another of its block."""
    parent = list(range(len(points)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(points)):
        near = np.flatnonzero(np.hypot(*(points - points[i]).T) <= reach)
        for j in near:
            parent[root(int(j))] = root(i)
    groups: dict[int, list[int]] = {}
    for i in range(len(points)):
        groups.setdefault(root(i), []).append(i)
    return [points[indices] for indices in groups.values()]


def read_dots(points: np.ndarray) -> tuple[list[BrailleBlock], np.ndarray]:
    """Dots (an (N, 2) array of centres) read as Braille: the blocks read, and the dots that are in none (too few, or
    not set to a Braille grid)."""
    if len(points) < _MIN_BLOCK_DOTS:
        return [], points
    spacing = _spacing(points)
    blocks, loose = [], []
    for group in blocks_of(points, _BLOCK_REACH * spacing):
        block = read_block(group) if len(group) >= _MIN_BLOCK_DOTS else None
        if block is None:
            loose.append(group)
        else:
            blocks.append(block)
    blocks.sort(key=lambda block: (block.bbox.y0, block.bbox.x0))
    return blocks, (np.vstack(loose) if loose else np.empty((0, 2)))


def find_braille_dots(image: np.ndarray, config: DetectionConfig) -> np.ndarray:
    """A file's Braille dots: by their colour (`DetectionConfig.braille_hsv_ranges`, `detect_dots`) where that finds
    them, and by their shape, whatever their colour (`braille_diff.find_dots`), where it does not -- Braille of another
    colour somewhere else. (Not both in one place: by shape, round parts of the lettering a dot is set over are taken
    for dots too, and a dot set over lettering is not round on its own.)"""
    by_colour = braille_diff.detect_dots(image, config)
    found = braille_diff.find_dots(image)
    if found is None:
        return by_colour
    by_shape, width = found
    if not len(by_colour):
        return by_shape
    reach = _BLOCK_REACH * _spacing(by_colour) if len(by_colour) >= 2 else 10 * width
    far = np.sqrt(((by_shape[:, None, :] - by_colour[None, :, :]) ** 2).sum(axis=2)).min(axis=1) > reach
    return np.vstack([by_colour, by_shape[far]])
