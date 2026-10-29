"""Comparing the WORDS of two digital pages (a leaflet's), word for word, where they are printed.

Two versions of a digital document are both exact renders, but they are rarely the same picture: a sentence rewritten
reflows the rest of its paragraph, one file's text is live and the other's outlined to paths (drawn a hair
differently), and a page re-set a little wider or narrower lands every line a few pixels from where it was. A pixel
comparison flags all of that; what the reviewer needs is what the TEXT says differently. So here:

1. Each page's words are taken from its own text where it has any (exact, instant: `text_diff.text_layer_words`) and read
   from its picture otherwise (an outlined page), and the suspect's are brought into the original page's frame.
2. Every word is matched to the same word where it is printed in the other file -- first in place (the page aligned,
   every unchanged word lies within a few pixels of its partner), then within reach of it (a line that reflowed moved
   its words along, or down a line, but they are the same words).
3. What is left unmatched is the change: runs of words on a line, each paired with the run the other file prints in
   its place ('X' -> 'Y'), or missing, or added.
4. And a change stands only if the INK differs there too. The words of a picture are a reading, and a reading can be
   wrong ('Information' read as 'lnformation'); even a file's own text can say one thing and draw another (a font that
   maps its codes to other glyphs). Where the one file's letters are found, printed the same, where the other's are,
   nothing changed: the reading differed, not the page.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, replace
from difflib import SequenceMatcher

import cv2
import numpy as np

from app.config import DetectionConfig
from app.engine import language_detect
from app.engine.alignment.homography import apply_transform
from app.engine.detectors import text_diff
from app.engine.detectors.text_diff import OcrWord, canonical
from app.models.diff import SOURCE_OCR, BoundingBox, DiffCandidate, DiffSubtype, DiffType

logger = logging.getLogger(__name__)

_PLACE_REACH_HEIGHTS = 0.6  # a word within this many word-heights (or `_PLACE_REACH_MIN_PX`) of its partner is in place
_PLACE_REACH_MIN_PX = 6.0
_REFLOW_LINES = 3.5  # a word that reflowed moved at most this many line-heights up or down...
_REFLOW_ALONG_HEIGHTS = 80.0  # ...and at most this many word-heights along (a full line of a wide column)
_RUN_GAP_HEIGHTS = 1.6  # unmatched words closer than this (word-heights) along a line are one phrase
_PAIR_REACH_HEIGHTS = 2.0  # a phrase gone and one new this close (word-heights) are one change: 'X' -> 'Y'
_INK_MATCH = 0.85  # the letters are printed the same if one file's are found in the other's to this correlation (softened)...
_INK_BLUR_PX = 1.0  # (softened by this, or this share of the type's height, if more)
_INK_BLUR_SHARE = 0.025
_INK_PATCH_HEIGHTS = 0.3  # ...with no patch this many type-heights across differing by more than...
_INK_LOCAL_SHARE = 0.28  # ...this share of the type's contrast (identical words measured <= 0.24; 'o' -> 'a' 0.32)
_MIN_CONTRAST = 40.0
_MIN_JUDGED_PX = 14  # a line of words smaller than this either way is judged with the print round it
_INK_SEARCH_HEIGHTS = 0.6  # ...within this many word-heights (or `_INK_SEARCH_MIN_PX`) of the same place
_INK_SEARCH_MIN_PX = 5
_INK_PAD_PX = 2
_INK_LEVEL = 200  # darker than this in its darkest channel is ink
_MIN_INK_PX = 6
_MIN_ALNUM_SHARE = 0.5  # a reading that is mostly marks is not a word
_READ_CONFIDENCE = 99.0  # words from a file's own text are exact
_SURE_READ = 85.0  # a word read from a whole page less surely than this is read again, closely, where it is
_WORD = re.compile(r"\w", re.UNICODE)


@dataclass
class _Word:
    text: str
    box: BoundingBox
    read: bool  # read from the picture (a guess), not taken from the file's own text
    confidence: float = 100.0
    known_upright: bool | None = None

    @property
    def upright(self) -> bool:
        if self.known_upright is not None:
            return self.known_upright
        return not (len(self.text) > 1 and self.box.height > 1.5 * self.box.width)

    @property
    def height(self) -> float:
        return max(1.0, self.box.height if self.upright else self.box.width)  # (a word up the page: its width)

    @property
    def centre(self) -> tuple[float, float]:
        return (self.box.x0 + self.box.x1) / 2, (self.box.y0 + self.box.y1) / 2


def in_frame(words: list[OcrWord], transform: np.ndarray) -> list[OcrWord]:
    """Words of the suspect page, their boxes taken into the original page's frame by `transform` (suspect -> original)."""
    if not words:
        return []
    corners = np.array(
        [[(w.bbox.x0, w.bbox.y0), (w.bbox.x1, w.bbox.y0), (w.bbox.x1, w.bbox.y1), (w.bbox.x0, w.bbox.y1)] for w in words],
        np.float64,
    ).reshape(-1, 2)
    mapped = apply_transform(transform, corners).reshape(-1, 4, 2)
    return [
        OcrWord(word.text, BoundingBox(float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())), word.confidence, word.upright)
        for word, p in zip(words, mapped)
    ]


def usable(words: list[OcrWord], floor: float) -> list[OcrWord]:
    """The words worth comparing: read at least as surely as `floor`, and words (mostly letters or digits), not marks."""
    kept = []
    for word in words:
        text = word.text.strip()
        if not text or word.confidence < floor:
            continue
        if len(_WORD.findall(text)) < _MIN_ALNUM_SHARE * len(text):
            continue
        kept.append(word)
    return kept


def _words(words: list[OcrWord]) -> list[_Word]:
    return [_Word(w.text.strip(), w.bbox, w.confidence < _READ_CONFIDENCE, w.confidence, w.upright) for w in words if w.text.strip()]


def _key(text: str) -> str:
    """What two words must share to be the same word: their letters and digits (a reading's stray mark aside)."""
    return canonical(text)


def _match(original: list[_Word], suspect: list[_Word]) -> tuple[list[int], list[int]]:
    """(the original words, the suspect words) that have no partner: the same word printed in place, or within reach
    of its place (reflowed). Each word is partnered once, nearest first."""
    by_key: dict[str, list[int]] = {}
    for j, word in enumerate(suspect):
        by_key.setdefault(_key(word.text), []).append(j)
    free_o, free_s = set(range(len(original))), set(range(len(suspect)))

    def pass_(reach, accept=None) -> None:
        pairs = []
        for i in free_o:
            a = original[i]
            ax, ay = a.centre
            for j in by_key.get(_key(a.text), ()):
                if j not in free_s:
                    continue
                bx, by = suspect[j].centre
                height = max(a.height, suspect[j].height)
                cost = reach(abs(bx - ax), abs(by - ay), height)
                if cost is not None:
                    pairs.append((cost, i, j))
        for _cost, i, j in sorted(pairs):
            if i in free_o and j in free_s and (accept is None or accept(i, j)):
                free_o.discard(i)
                free_s.discard(j)

    def in_place(dx: float, dy: float, height: float) -> float | None:
        reach = max(_PLACE_REACH_MIN_PX, _PLACE_REACH_HEIGHTS * height)
        return dx * dx + dy * dy if dx <= reach and dy <= reach else None

    def reflowed(dx: float, dy: float, height: float) -> float | None:
        if dy > _REFLOW_LINES * 1.4 * height or dx > _REFLOW_ALONG_HEIGHTS * height:
            return None
        return (dy / height) * 1000.0 + dx / height  # the same line first, then the nearest line

    pass_(in_place)
    context_o, context_s = _neighbours(original), _neighbours(suspect)

    def reflowed_in_context(i: int, j: int) -> bool:
        """A word matched out of place is the same word only if its neighbours came along too: at least one of them the
        same either side -- both, for a word so short it is everywhere ('3', 'of'), unless it ends a line in one file."""
        (left_o, right_o), (left_s, right_s) = context_o[i], context_s[j]
        agree = [
            (a is None and b is None) or (a is not None and b is not None and _key(original[a].text) == _key(suspect[b].text))
            for a, b in ((left_o, left_s), (right_o, right_s))
        ]  # (a line's start or end in both is its neighbours agreeing)
        if len(_key(original[i].text)) > 2:
            return any(agree)
        broken = left_o is None or left_s is None or right_o is None or right_s is None  # a line broke next to it
        return all(agree) or (broken and any(agree))

    pass_(reflowed, reflowed_in_context)
    return sorted(free_o), sorted(free_s)


def _neighbours(words: list[_Word]) -> list[tuple[int | None, int | None]]:
    """(the word before, the word after) each word on its own line -- None where it starts or ends one."""
    order = sorted(range(len(words)), key=lambda k: (words[k].box.x0))
    found: list[list[int | None]] = [[None, None] for _ in words]
    best: list[list[float]] = [[float("inf"), float("inf")] for _ in words]
    rows: dict[int, list[int]] = {}
    for k in order:
        rows.setdefault(int(words[k].centre[1] // max(4.0, words[k].height)), []).append(k)
    for k in order:
        a = words[k]
        band = int(a.centre[1] // max(4.0, a.height))
        for row in (band - 1, band, band + 1):
            for m in rows.get(row, ()):
                if m == k:
                    continue
                b = words[m]
                if not _one_line(a.box, b.box):
                    continue
                gap = b.box.x0 - a.box.x1
                if -0.5 * a.height <= gap <= _RUN_GAP_HEIGHTS * max(a.height, b.height) and gap < best[k][1]:
                    found[k][1], best[k][1] = m, gap
                    if gap < best[m][0]:
                        found[m][0], best[m][0] = k, gap
    return [(left, right) for left, right in found]


def _bands(words: list[_Word]) -> list[list[_Word]]:
    """The words in bands down the page, each band the words whose middles lie within half a word-height of its first
    (so one line of type, across every column), each band left to right."""
    bands: list[list[_Word]] = []
    for word in sorted(words, key=lambda w: w.centre[1]):
        if bands and abs(word.centre[1] - bands[-1][0].centre[1]) <= 0.5 * max(word.height, bands[-1][0].height):
            bands[-1].append(word)
        else:
            bands.append([word])
    return [sorted(band, key=lambda w: w.box.x0) for band in bands]


def _runs(words: list[_Word], indices: list[int]) -> list[list[_Word]]:
    """The unmatched words as phrases: runs along one line, each word close after the last."""
    runs: list[list[_Word]] = []
    for band in _bands([words[i] for i in indices]):
        run = [band[0]]
        for word in band[1:]:
            last = run[-1]
            gap = word.box.x0 - last.box.x1
            if -0.5 * last.height <= gap <= _RUN_GAP_HEIGHTS * max(word.height, last.height):
                run.append(word)
            else:
                runs.append(run)
                run = [word]
        runs.append(run)
    return runs


def _in_order(words: list[_Word]) -> list[_Word]:
    """Words line by line, left to right along each."""
    return [word for band in _bands(words) for word in band]


def _box(run: list[_Word]) -> BoundingBox:
    return BoundingBox(min(w.box.x0 for w in run), min(w.box.y0 for w in run), max(w.box.x1 for w in run), max(w.box.y1 for w in run))


def _text(run: list[_Word]) -> str:
    return " ".join(word.text for word in run)


def _gap(a: BoundingBox, b: BoundingBox) -> tuple[float, float]:
    """(along, across) distance between two boxes (0 where they overlap)."""
    return max(0.0, max(a.x0, b.x0) - min(a.x1, b.x1)), max(0.0, max(a.y0, b.y0) - min(a.y1, b.y1))


def _one_line(a: BoundingBox, b: BoundingBox) -> bool:
    """Whether two boxes are on one line: they overlap by at least half the shorter one's height."""
    overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
    return overlap >= 0.5 * min(a.height, b.height)


def _darkest(picture: np.ndarray) -> np.ndarray:
    return picture.min(axis=2) if picture.ndim == 3 else picture


def _found_in(template_image: np.ndarray, box: BoundingBox, other_image: np.ndarray, height: float) -> bool | None:
    """Whether the ink of `box` in `template_image` is printed the same in `other_image`, at about the same place.
    None if there is no ink in the box to compare."""
    rows, cols = template_image.shape[:2]
    x0, y0 = max(0, int(box.x0) - _INK_PAD_PX), max(0, int(box.y0) - _INK_PAD_PX)
    x1, y1 = min(cols, int(np.ceil(box.x1)) + _INK_PAD_PX), min(rows, int(np.ceil(box.y1)) + _INK_PAD_PX)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    template = _darkest(template_image[y0:y1, x0:x1])
    if int((template < _INK_LEVEL).sum()) < _MIN_INK_PX:
        return None
    reach = max(_INK_SEARCH_MIN_PX, int(round(_INK_SEARCH_HEIGHTS * height)))
    sx0, sy0 = max(0, x0 - reach), max(0, y0 - reach)
    sx1, sy1 = min(other_image.shape[1], x1 + reach), min(other_image.shape[0], y1 + reach)
    if sx1 - sx0 < template.shape[1] or sy1 - sy0 < template.shape[0]:
        return False
    search = _darkest(other_image[sy0:sy1, sx0:sx1])
    if float(template.std()) < 1.0 or float(search.std()) < 1.0:
        return bool(abs(float(template.mean()) - float(search.mean())) < 8.0)
    # Softened a little first: live text and the same text outlined to paths are drawn a hair differently at the edges of
    # every stroke (identical words correlated at only ~0.9 sharp, ~0.95 softened); a different letter is not a hair.
    sigma = max(_INK_BLUR_PX, _INK_BLUR_SHARE * height)
    template = cv2.GaussianBlur(template.astype(np.float32), (0, 0), sigma)
    search = cv2.GaussianBlur(search.astype(np.float32), (0, 0), sigma)
    score = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _low, best, _where, (bx, by) = cv2.minMaxLoc(score)
    if best < _INK_MATCH:
        return False
    # ...and, where they line up best, no letter-sized patch differs by much of the type's own contrast: one letter of a
    # long word changed ('Paracetamol' -> 'Paracetamal') leaves the rest correlating well, but not that letter
    patch = search[by : by + template.shape[0], bx : bx + template.shape[1]]
    size = max(3, int(round(_INK_PATCH_HEIGHTS * height)))
    worst = float(cv2.blur(np.abs(template - patch), (size, size)).max())
    contrast = max(_MIN_CONTRAST, float(template.max() - template.min()))
    return worst <= _INK_LOCAL_SHARE * contrast


def _same_ink(original: np.ndarray, suspect: np.ndarray, gone: list[_Word], new: list[_Word]) -> bool:
    """Whether what differs in the words is printed the same in both files: every line of the words said to be gone is
    found printed in the suspect where it was, and every line of those said to be new in the original where it is (so a
    reading, not the page, differed). A line is judged whole, not word by word: a subscript ('SO3H') is too small to
    judge alone. A line with no ink to compare says nothing either way."""
    checked = False
    for words, source, other in ((gone, original, suspect), (new, suspect, original)):
        for line in _lines(words):
            height = float(np.median([word.height for word in line]))
            box = _box(line)
            if min(box.width, box.height) < _MIN_JUDGED_PX:
                # too small to judge alone (a subscript, a full stop): judged with the print round it, as it sits
                grow = max(_MIN_JUDGED_PX / 2, box.height)
                box = BoundingBox(box.x0 - grow, box.y0 - grow, box.x1 + grow, box.y1 + grow)
                height = max(height, box.height / 2)
            found = _found_in(source, box, other, height)
            if found is None:
                continue
            if not found:
                return False
            checked = True
    return checked


def _lines(words: list[_Word]) -> list[list[_Word]]:
    """The words split into their lines (upright words; a word up the page is a line of its own)."""
    lines: list[list[_Word]] = []
    for word in _in_order(words):
        if word.upright and lines and lines[-1][-1].upright and _one_line(_box(lines[-1]), word.box):
            lines[-1].append(word)
        else:
            lines.append([word])
    return lines


def _only_spacing(gone: list[_Word], new: list[_Word]) -> bool:
    """The same letters, split or joined differently (a word hyphenated across a line in one file, whole in the other)."""
    squash = lambda run: "".join(ch for word in run for ch in word.text if ch.isalnum())  # noqa: E731
    return bool(gone) and bool(new) and squash(gone) == squash(new)


_REREAD_TYPE_PX = 50.0  # a change's line is enlarged so its type is about this tall before it is read again
_REREAD_MAX_FACTOR = 4.0
_REREAD_MARGIN = 0.3  # word-heights of blank either side of the words read with a change (the words beside it, whole)
_REREAD_PAD_ACROSS = 0.12  # ...and above and below the line (a word's box already holds its ascenders and descenders)
_LINE_SEGMENTATION = "--psm 7"  # the window holds one line
_BORDER_PX = 24  # white set round a line before it is read
_REREAD_CONTEXT_WORDS = 2  # words either side of a change read with it
_REREAD_SCALES = (1.0, 1.5, 2.0)  # each line is read at these multiples of its enlargement, the surest reading kept
_REREAD_MIN_CONFIDENCE = 20.0  # a change's ink already differs: its reading only says what it says, so is kept if at all sure


def _reread(
    changes: list[tuple[list[_Word], list[_Word]]],
    everything: list[_Word],
    original: np.ndarray,
    suspect: np.ndarray,
    config: DetectionConfig,
    original_layer: list[OcrWord] | None,
    suspect_layer: list[OcrWord] | None,
) -> list[DiffCandidate | None]:
    """Each change whose words were read (from a whole page, less surely than a line is), read again: the line it is
    on, the same window in both files, enlarged to a size the reader reads surely -- and the words that differ there are
    the change. A file's own text, where it has it, is taken as it is. Where the words then read the same, how they are
    SET is what differs, if anything (`text_diff._set_differently`: typeface, size, weight, colour); None if nothing."""
    shape = original.shape[:2]
    windows: list[text_diff._Window] = []
    for gone, new in changes:
        run = gone + new
        box, height = _box(run), float(np.median([w.height for w in run]))
        # along the line to the words either side (in either file), whole -- two of them each way, if they are close: a
        # word is read cut short by the window's edge, and the reader reads a line far more surely with its context
        line = [w.box for w in everything if _one_line(w.box, box)]
        near = _RUN_GAP_HEIGHTS * height  # (a word further off is another column's)
        x0, x1 = box.x0, box.x1
        for _step in range(_REREAD_CONTEXT_WORDS):
            left = [b for b in line if x0 - near <= b.x1 < x0 + 1 and b.x0 < x0]
            right = [b for b in line if x1 - 1 < b.x0 <= x1 + near and b.x1 > x1]
            x0 = min([x0] + [b.x0 for b in left])
            x1 = max([x1] + [b.x1 for b in right])
        margin, pad_y = round(_REREAD_MARGIN * height), round(_REREAD_PAD_ACROSS * height)
        x0, y0 = max(0, int(min(x0, box.x0)) - margin), max(0, int(box.y0) - pad_y)
        x1, y1 = min(shape[1], int(np.ceil(max(x1, box.x1))) + margin), min(shape[0], int(np.ceil(box.y1)) + pad_y)
        factor = min(_REREAD_MAX_FACTOR, max(1.0, _REREAD_TYPE_PX / height))
        windows.append(text_diff._Window(x0, y0, x1, y1, factor, None))
    # a change over more than one line is read as a block of text
    one_line = [_box(gone + new).height < 1.6 * float(np.median([w.height for w in gone + new])) for gone, new in changes]
    jobs, slots = [], []
    for k, window in enumerate(windows):
        for side, (picture, layer) in enumerate(((original, original_layer), (suspect, suspect_layer))):
            if layer is None or (not text_diff._layer_words_in(layer, window, shape) and text_diff._printed(picture, window)):
                jobs.append((picture, window))
                slots.append((k, side))
    readings = {}
    for lines, segmentation in ((True, _LINE_SEGMENTATION), (False, _BLOCK_SEGMENTATION)):
        chosen = [(slot, job) for slot, job in zip(slots, jobs) if one_line[slot[0]] == lines]
        if chosen:
            readings.update(zip([slot for slot, _job in chosen], _read_lines([job for _slot, job in chosen], config, segmentation)))

    results: list[DiffCandidate | None] = []
    for k, ((gone, new), window) in enumerate(zip(changes, windows)):
        sides = []
        for side, layer in enumerate((original_layer, suspect_layer)):
            words = readings[(k, side)] if (k, side) in readings else text_diff._layer_words_in(layer or [], window, shape)
            sides.append(_reading_order(words))
        change_box = _box(gone + new)
        # the change's own line(s) only: the window reaches a little above and below, not to be read too
        words_o, words_s = ([w for w in words if change_box.y0 <= (w.bbox.y0 + w.bbox.y1) / 2 <= change_box.y1] for words in sides)
        if not words_o and not words_s:
            results.append(None)
            continue
        matcher = SequenceMatcher(None, [w.text for w in words_o], [w.text for w in words_s], autojunk=False)
        gone2: list[OcrWord] = []
        new2: list[OcrWord] = []
        alike: list[tuple[OcrWord, OcrWord]] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                alike += list(zip(words_o[i1:i2], words_s[j1:j2]))
            else:
                gone2 += words_o[i1:i2]
                new2 += words_s[j1:j2]
        # only what differs where the change is: a word at the window's end read differently is not it
        gone2 = [w for w in gone2 if _near(w.bbox, change_box)]
        new2 = [w for w in new2 if _near(w.bbox, change_box)]
        while gone2 and new2 and gone2[0].text == new2[0].text:
            gone2, new2 = gone2[1:], new2[1:]
        while gone2 and new2 and gone2[-1].text == new2[-1].text:
            gone2, new2 = gone2[:-1], new2[:-1]
        as_words = lambda found: [_Word(w.text, w.bbox, True) for w in found]  # noqa: E731
        same_words = sorted(w.text for w in gone2) == sorted(w.text for w in new2)  # the same words, only listed in another order
        if (gone2 or new2) and not same_words and not _only_spacing(as_words(gone2), as_words(new2)):
            text_gone, text_new = " ".join(w.text for w in gone2), " ".join(w.text for w in new2)
            if gone2 and new2:
                subtype, detail = DiffSubtype.CHANGED, f"'{text_gone}' -> '{text_new}'"
            elif gone2:
                subtype, detail = DiffSubtype.REMOVED, f"'{text_gone}' missing from suspect"
            else:
                subtype, detail = DiffSubtype.ADDED, f"'{text_new}' added in suspect"
            confidence = min(w.confidence for w in gone2 + new2) / 100.0
            results.append(
                DiffCandidate(
                    bbox=change_box, type=_span_type(text_gone + text_new), subtype=subtype, confidence=confidence, detail=detail,
                    was_text=text_gone, now_text=text_new,
                )
            )
        else:
            results.append(text_diff._set_differently(change_box, alike, original, suspect, config))
    return results


def _read_lines(
    jobs: list[tuple[np.ndarray, "text_diff._Window"]],
    config: DetectionConfig,
    segmentation: str = _LINE_SEGMENTATION,
    scales: tuple[float, ...] = _REREAD_SCALES,
    floor: float = _REREAD_MIN_CONFIDENCE,
) -> list[list[OcrWord]]:
    """The words on each (picture, window) holding one line: enlarged, set in a white border (the reader misses
    the first and last letters of a line cropped tight, and reads the rest unsurely), read as one line -- at a few sizes,
    the surest reading kept (a '6' read as '/7' at one size is a '6' at the next). Every word read at all surely is
    kept, however short ('c)' -> 'd)' is a change), and put back where it is in the picture."""
    pictures, where = [], []
    for number, (image, window) in enumerate(jobs):
        for scale in scales:
            sized = replace(window, factor=min(_REREAD_MAX_FACTOR, window.factor * scale))
            enlarged = text_diff._window_picture(image, sized)
            pictures.append(cv2.copyMakeBorder(enlarged, _BORDER_PX, _BORDER_PX, _BORDER_PX, _BORDER_PX, cv2.BORDER_CONSTANT, value=255))
            where.append((number, sized))
    best: list[tuple[float, list[OcrWord]]] = [(-1.0, []) for _ in jobs]
    for (number, window), data in zip(where, text_diff._ocr_many(pictures, segmentation, config.ocr_language)):
        words = []
        for index, raw in enumerate(data.get("text", [])):
            text, confidence = str(raw).strip(), float(data["conf"][index])
            if not text or confidence < floor or len(_WORD.findall(text)) < _MIN_ALNUM_SHARE * len(text):
                continue
            left, top = float(data["left"][index]) - _BORDER_PX, float(data["top"][index]) - _BORDER_PX
            f = window.factor
            words.append(OcrWord(text, BoundingBox(
                window.x0 + left / f, window.y0 + top / f,
                window.x0 + (left + float(data["width"][index])) / f, window.y0 + (top + float(data["height"][index])) / f,
            ), confidence))
        # the surest reading: the most sure letters (a reading that drops words is not surer for it)
        sureness = sum(word.confidence * len(word.text) for word in words)
        if sureness > best[number][0]:
            best[number] = (sureness, words)
    return [words for _sureness, words in best]


def _reading_order(words: list[OcrWord]) -> list[OcrWord]:
    """Words line by line (each line the words that overlap it by half their height), left to right along each."""
    lines: list[list[OcrWord]] = []
    for word in sorted(words, key=lambda w: (w.bbox.y0 + w.bbox.y1) / 2):
        for line in lines:
            if _one_line(line[0].bbox, word.bbox):
                line.append(word)
                break
        else:
            lines.append([word])
    return [word for line in lines for word in sorted(line, key=lambda w: w.bbox.x0)]


def _near(box: BoundingBox, change: BoundingBox) -> bool:
    along, across = _gap(box, change)
    return along <= 0.25 * max(1.0, change.height) and across <= 0.25 * max(1.0, change.height)


# The scripts each reader language reads (by the first word of a letter's Unicode name). A language not listed is taken
# to read whatever it is given: nothing is set aside that might be readable.
_LANGUAGE_SCRIPTS = {
    **{code: "LATIN" for code in (
        "eng", "fra", "deu", "spa", "ita", "por", "nld", "dan", "swe", "nor", "fin", "pol", "ces", "slk", "slv", "hrv",
        "hun", "ron", "tur", "est", "lav", "lit", "cat", "gle", "isl", "msa", "ind", "vie", "afr", "sqi", "eus", "glg",
    )},
    "ara": "ARABIC", "fas": "ARABIC", "urd": "ARABIC", "rus": "CYRILLIC", "ukr": "CYRILLIC", "bul": "CYRILLIC",
    "srp": "CYRILLIC", "bel": "CYRILLIC", "mkd": "CYRILLIC", "ell": "GREEK", "heb": "HEBREW", "hin": "DEVANAGARI",
    "mar": "DEVANAGARI", "nep": "DEVANAGARI", "ben": "BENGALI", "tam": "TAMIL", "tel": "TELUGU", "kan": "KANNADA",
    "mal": "MALAYALAM", "guj": "GUJARATI", "pan": "GURMUKHI", "tha": "THAI", "kor": "HANGUL",
    "chi_sim": "CJK", "chi_tra": "CJK", "jpn": "CJK",
}


def _scripts_of(language: str) -> set[str] | None:  # (`language` resolved: never "auto")
    """The scripts `language` (Tesseract's codes, '+'-joined) reads; None if any of it is a language not listed."""
    scripts: set[str] = set()
    for code in language.split("+"):
        if code == "osd" or not code:
            continue
        if code not in _LANGUAGE_SCRIPTS:
            return None
        scripts.add(_LANGUAGE_SCRIPTS[code])
    return scripts or None


def _script(ch: str) -> str | None:
    if not ch.isalpha():
        return None  # digits and marks are every script's
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return None
    first = name.split()[0]
    return "CJK" if first in ("CJK", "HIRAGANA", "KATAKANA") else first


def _in_scripts(text: str, scripts: set[str]) -> bool:
    """Whether a word is (mostly) in one of `scripts` -- a word with no letters at all is (a number reads in any)."""
    found = [_script(ch) for ch in text]
    letters = [script for script in found if script is not None]
    return not letters or sum(script in scripts for script in letters) >= 0.5 * len(letters)


def _outside(words: list[_Word], zones: list[_Word], shape: tuple[int, int]) -> list[_Word]:
    """`words` without those lying where `zones` are (a reading of text in another script is not words)."""
    scale = 0.25
    mask = np.zeros((max(1, int(shape[0] * scale) + 1), max(1, int(shape[1] * scale) + 1)), np.uint8)
    for zone in zones:
        pad = 0.3 * zone.height
        cv2.rectangle(
            mask,
            (int((zone.box.x0 - pad) * scale), int((zone.box.y0 - pad) * scale)),
            (int((zone.box.x1 + pad) * scale), int((zone.box.y1 + pad) * scale)),
            1,
            -1,
        )
    kept = []
    for word in words:
        x, y = word.centre
        row, col = int(y * scale), int(x * scale)
        if not (0 <= row < mask.shape[0] and 0 <= col < mask.shape[1] and mask[row, col]):
            kept.append(word)
    return kept


def _unreadable_changes(words: list[_Word], own: np.ndarray, other: np.ndarray, own_is_original: bool) -> list[DiffCandidate]:
    """Lines of text in a script the reader cannot read, compared by their ink: each line whose print is not found the
    same in the other file (both ways: the other file's print there, found in this one's) is a finding."""
    found = []
    for run in _runs(words, list(range(len(words)))):
        box = _box(run)
        height = float(np.median([w.height for w in run]))
        there = _found_in(own, box, other, height)
        back = _found_in(other, box, own, height)
        if there is None or (there and back is not False):
            continue
        text = _text(run)
        found.append(
            DiffCandidate(
                bbox=box,
                type=DiffType.TEXT,
                subtype=DiffSubtype.CHANGED,
                confidence=0.9,
                detail=f"Text differs where the {'original' if own_is_original else 'suspect'} has '{_clip(text)}' (compared by its "
                "print: this script is not one the text reader has installed)",
            )
        )
    return found


def _clip(text: str, limit: int = 60) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _span_type(text: str) -> DiffType:
    letters = [ch for ch in text if ch.isalnum()]
    return DiffType.NUMBER if letters and sum(ch.isdigit() for ch in letters) >= 0.5 * len(letters) else DiffType.TEXT


_REGION_SCALE = 0.5  # where two pages differ is found at this scale...
_REGION_BLUR_PX = 1.0  # ...softened this much (a stroke drawn a hair differently is not a difference)...
_REGION_SHIFT_PX = 1  # ...allowing each point this far out of place...
_REGION_LEVEL = 40  # ...a difference of this many grey levels in the darkest channel
_REGION_ALONG_HEIGHTS = 1.5  # a region is widened by this many type-heights along its line (its words whole)...
_REGION_ACROSS_HEIGHTS = 0.5  # ...and this many up and down
_WORD_CLOSE_HEIGHTS = 0.2  # letters closer than this (type-heights) along a line are one word's
_WORD_MAX_HEIGHTS = 3.0  # a blob of print taller than this many type-heights, or...
_WORD_MAX_WIDTH_HEIGHTS = 40.0  # ...wider, is not a word (a rule, a frame, a picture)
DEFAULT_TYPE_PX = 40.0  # the type's height where neither page has text of its own to measure it by (~9 pt at 300 DPI)
_BLOCK_SEGMENTATION = "--psm 6"


_LETTER_TO_WORD_BOX = 1.6  # a word's box (ascenders to descenders) is about this many times its letters' median height


def type_height(picture: np.ndarray) -> float:
    """About how tall a word of the page's text is, in pixels, measured from its printed letters (for a page with no
    text of its own to measure it by): the median height of the marks of letter size, times `_LETTER_TO_WORD_BOX`."""
    ink = (_darkest(picture) < _INK_LEVEL).astype(np.uint8)
    count, _labels, stats, _centres = cv2.connectedComponentsWithStats(ink, connectivity=8)
    heights = stats[1:count, 3]
    widths = stats[1:count, 2]
    letters = heights[(heights >= 5) & (heights <= 150) & (widths <= 2 * heights) & (widths >= 2)]
    if len(letters) < 20:
        return DEFAULT_TYPE_PX
    return float(np.median(letters)) * _LETTER_TO_WORD_BOX


def difference_windows(original: np.ndarray, suspect: np.ndarray, type_px: float) -> list["text_diff._Window"]:
    """The places two aligned pages are printed differently, as windows to read: a page is read only where it differs,
    since its text cannot have changed where its print has not (a pack insert read whole took half a minute; its few
    changes, a second). Each window reaches past its difference to the words round it, whole."""
    height, width = original.shape[:2]
    small = (max(1, round(width * _REGION_SCALE)), max(1, round(height * _REGION_SCALE)))
    a, b = (cv2.GaussianBlur(cv2.resize(_darkest(p), small, interpolation=cv2.INTER_AREA), (0, 0), _REGION_BLUR_PX) for p in (original, suspect))
    r = _REGION_SHIFT_PX
    padded = cv2.copyMakeBorder(b, r, r, r, r, cv2.BORDER_REPLICATE)
    least = None
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            diff = cv2.absdiff(a, padded[dy : dy + a.shape[0], dx : dx + a.shape[1]])
            least = diff if least is None else cv2.min(least, diff)
    changed = (least > _REGION_LEVEL).astype(np.uint8)
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if not changed.any():
        return []
    type_small = max(2.0, type_px * _REGION_SCALE)
    along, across = int(round(_REGION_ALONG_HEIGHTS * type_small)), int(round(_REGION_ACROSS_HEIGHTS * type_small))
    grown = cv2.dilate(changed, np.ones((2 * across + 1, 2 * along + 1), np.uint8))
    # ...and out to the whole of every word it touches, in either page: the print, closed up along its lines by less
    # than a word space, is words (a blob bigger than a few words -- a rule, a table's frame, a picture -- is not one)
    ink = cv2.min(a, b) < _INK_LEVEL
    gap = max(1, int(round(_WORD_CLOSE_HEIGHTS * type_small)))
    blobs = cv2.dilate(ink.astype(np.uint8), np.ones((1, 2 * gap + 1), np.uint8))
    count, labels, blob_stats, _c = cv2.connectedComponentsWithStats(blobs)
    wordlike = (blob_stats[:, 3] <= _WORD_MAX_HEIGHTS * type_small) & (blob_stats[:, 2] <= _WORD_MAX_WIDTH_HEIGHTS * type_small)
    wordlike[0] = False
    regions_count, regions, stats, _centres = cv2.connectedComponentsWithStats(grown)
    windows = []
    for label in range(1, regions_count):
        x, y, w, h, _area = stats[label]
        x0, y0, x1, y1 = x, y, x + w, y + h
        touched = np.unique(labels[y : y + h, x : x + w][regions[y : y + h, x : x + w] == label])
        touched = touched[wordlike[touched]]
        if len(touched):
            x0 = min(x0, int(blob_stats[touched, 0].min()))
            y0 = min(y0, int(blob_stats[touched, 1].min()))
            x1 = max(x1, int((blob_stats[touched, 0] + blob_stats[touched, 2]).max()))
            y1 = max(y1, int((blob_stats[touched, 1] + blob_stats[touched, 3]).max()))
        margin = max(1, int(round(0.3 * type_small)))
        x0, y0 = int(max(0, x0 - margin) / _REGION_SCALE), int(max(0, y0 - margin) / _REGION_SCALE)
        x1, y1 = min(width, int(np.ceil((x1 + margin) / _REGION_SCALE))), min(height, int(np.ceil((y1 + margin) / _REGION_SCALE)))
        if x1 - x0 >= 8 and y1 - y0 >= 8:
            windows.append(text_diff._Window(x0, y0, x1, y1, 1.0, None))
    return _merged(windows)


def _merged(windows: list["text_diff._Window"]) -> list["text_diff._Window"]:
    """Windows that overlap, as one (so no word is in two, or cut by one's edge inside another)."""
    merged = list(windows)
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                a, b = merged[i], merged[j]
                if a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1:
                    merged[i] = replace(a, x0=min(a.x0, b.x0), y0=min(a.y0, b.y0), x1=max(a.x1, b.x1), y1=max(a.y1, b.y1))
                    del merged[j]
                    changed = True
                    break
            if changed:
                break
    return merged


def words_in_windows(
    windows: list["text_diff._Window"],
    original: np.ndarray,
    suspect: np.ndarray,
    original_layer: list[OcrWord] | None,
    suspect_layer: list[OcrWord] | None,
    config: DetectionConfig,
    type_px: float,
) -> tuple[list[OcrWord], list[OcrWord]]:
    """Each page's words in `windows`: its own text where it has any (whole words inside a window only, as a reading
    would give them), read otherwise -- each window on its own, enlarged if the type is small, a window of one line read
    as one line and a bigger one as a block of text."""
    shape = original.shape[:2]
    factor = min(_REREAD_MAX_FACTOR, max(1.0, _REREAD_TYPE_PX / max(1.0, type_px)))
    found: tuple[list[OcrWord], list[OcrWord]] = ([], [])
    for side, (picture, layer) in enumerate(((original, original_layer), (suspect, suspect_layer))):
        if layer is not None:
            seen: set[int] = set()
            for window in windows:
                for word in text_diff._layer_words_in(layer, window, shape):
                    if id(word) not in seen:
                        seen.add(id(word))
                        found[side].append(word)
            continue
        sized = [replace(window, factor=factor) for window in windows]
        for lines, segmentation in ((True, _LINE_SEGMENTATION), (False, _BLOCK_SEGMENTATION)):
            chosen = [w for w in sized if ((w.y1 - w.y0) < 2.2 * type_px) == lines]
            if chosen:
                for words in _read_lines([(picture, w) for w in chosen], config, segmentation, (1.0,), config.ocr_min_confidence):
                    found[side].extend(words)
    return found


def compare_words(
    original_words: list[OcrWord],
    suspect_words: list[OcrWord],
    original: np.ndarray,
    suspect: np.ndarray,
    config: DetectionConfig,
    original_layer: list[OcrWord] | None = None,
    suspect_layer: list[OcrWord] | None = None,
    language: str = "eng",
) -> list[DiffCandidate]:
    """The text changes between two pages: `original_words` on `original`, `suspect_words` on `suspect` -- the suspect
    page ALREADY aligned onto the original (its words in the same frame, `in_frame`). See the module's account.

    `original_layer`/`suspect_layer` are the words that came from a file's own text (None for a page that was read): a
    change whose words were read is read again, closely, where it is (`text_diff.describe`, which also tells a change of
    typeface or size from a change of words), since a page read whole is read less surely than a window of it.

    Where one page was read and the other has its own text, the words of that text in a script the reader cannot read
    (`language`: Arabic on a page read for English) cannot be compared with a reading: their lines are compared by
    their ink alone (`_unreadable_changes`), and what was 'read' over them is set aside."""
    words_o, words_s = _words(original_words), _words(suspect_words)
    script_candidates: list[DiffCandidate] = []
    if (original_layer is None) != (suspect_layer is None):
        scripts = _scripts_of(language_detect.resolve(language))
        if scripts is not None:
            layer_first = original_layer is not None
            own, read = (words_o, words_s) if layer_first else (words_s, words_o)
            unreadable = [w for w in own if not _in_scripts(w.text, scripts)]
            if unreadable:
                own = [w for w in own if _in_scripts(w.text, scripts)]
                read = _outside(read, unreadable, original.shape[:2])
                words_o, words_s = (own, read) if layer_first else (read, own)
                script_candidates = _unreadable_changes(
                    unreadable, original if layer_first else suspect, suspect if layer_first else original, layer_first
                )
    free_o, free_s = _match(words_o, words_s)
    gone_runs, new_runs = _runs(words_o, free_o), _runs(words_s, free_s)

    # A phrase gone and the phrases new printed in its place are one change ('X' -> 'Y'): runs on the same line that
    # overlap or nearly touch are joined, however many there are on either side (a reading splits and joins words).
    runs = [(True, run) for run in gone_runs] + [(False, run) for run in new_runs]
    parent = list(range(len(runs)))

    def root(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    boxes = [_box(run) for _gone, run in runs]
    heights = [max(w.height for w in run) for _gone, run in runs]
    order = sorted(range(len(runs)), key=lambda k: boxes[k].y0)
    for position, i in enumerate(order):
        for j in order[position + 1 :]:
            if boxes[j].y0 > boxes[i].y1:
                break  # (sorted down the page: nothing further on can share this one's line)
            if runs[i][0] == runs[j][0]:
                continue
            height = max(heights[i], heights[j])
            if _one_line(boxes[i], boxes[j]) and _gap(boxes[i], boxes[j])[0] <= _PAIR_REACH_HEIGHTS * height:
                parent[root(i)] = root(j)
    grouped: dict[int, tuple[list[_Word], list[_Word]]] = {}
    for k, (gone, run) in enumerate(runs):
        group = grouped.setdefault(root(k), ([], []))
        (group[0] if gone else group[1]).extend(run)
    changes = [(_in_order(gone), _in_order(new)) for gone, new in grouped.values()]

    candidates: list[DiffCandidate] = []
    was_read: list[bool] = []
    changes_kept: list[tuple[list[_Word], list[_Word]]] = []
    dropped = 0
    for gone, new in changes:
        while gone and new and gone[0].text == new[0].text:  # the words both say, at either end, are not the change
            gone, new = gone[1:], new[1:]
        while gone and new and gone[-1].text == new[-1].text:
            gone, new = gone[:-1], new[:-1]
        if not gone and not new:
            continue
        reordered = bool(gone) and sorted(_key(w.text) for w in gone) == sorted(_key(w.text) for w in new)  # a line reflowed
        if reordered or _only_spacing(gone, new) or _same_ink(original, suspect, gone, new):
            dropped += 1
            continue
        box = _box(gone + new)
        read = any(word.read for word in gone + new)
        # read again, closely: a change a whole page's reading was unsure of, and a word gone or new with nothing in its
        # place (a page's own text can leave out words printed in a picture on it: they are read where they are)
        unsure = any(word.read and word.confidence < _SURE_READ for word in gone + new) or not (gone and new)
        unsure = unsure and all(word.upright for word in gone + new)  # (a line is read upright: text up the page is not)
        if gone and new:
            subtype, detail = DiffSubtype.CHANGED, f"'{_text(gone)}' -> '{_text(new)}'"
        elif gone:
            subtype, detail = DiffSubtype.REMOVED, f"'{_text(gone)}' missing from suspect"
        else:
            subtype, detail = DiffSubtype.ADDED, f"'{_text(new)}' added in suspect"
        candidates.append(
            DiffCandidate(
                bbox=box,
                type=_span_type(_text(gone + new)),
                subtype=subtype,
                confidence=0.8 if read else 1.0,
                detail=detail,
                was_text=_text(gone),
                now_text=_text(new),
            )
        )
        was_read.append(unsure)
        changes_kept.append((gone, new))
    reread = [k for k, read in enumerate(was_read) if read]
    if reread:
        try:
            closer = _reread([changes_kept[k] for k in reread], words_o + words_s, original, suspect, config, original_layer, suspect_layer)
        except text_diff.OcrUnavailableError:
            closer = [candidates[k] for k in reread]
        for k, candidate in zip(reread, closer):
            candidates[k] = candidate  # None: read closely, the words are the same and set the same
        dropped += sum(candidate is None for candidate in closer)
        candidates = [candidate for candidate in candidates if candidate is not None]
    candidates += script_candidates
    for candidate in candidates:
        candidate.source = SOURCE_OCR  # found by comparing the words (their print only confirms it)
    logger.info(
        "page words compared",
        extra={"original_words": len(words_o), "suspect_words": len(words_s), "changes": len(candidates), "same_ink": dropped},
    )
    return candidates
