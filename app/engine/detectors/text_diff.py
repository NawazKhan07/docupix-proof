"""Text and number difference detection via OCR (Architecture §3.3.5).

Both aligned artwork regions are OCR'd with Tesseract (offline), then diffed
at token level rather than pixel level, so "500mg -> 250mg" is reported as
one text change with the old and new values rather than as a scattering of
pixel regions. Because both images are already in the shared aligned space,
a token that OCRs identically but sits somewhere else is a *move* (FR-6.1),
which a pure string diff would never notice.

Tesseract is a separate native binary, not a Python package; when it is
absent this raises `OcrUnavailableError` so the pipeline can report that
text was not checked instead of quietly returning "no text differences".

A scan is read less surely than the file it was made from, so Scan vs Digital uses `detect_scan`,
which is anchored on the digital file's lines -- see `DetectionConfig.ocr_min_confidence_scan`.
"""

from __future__ import annotations

import logging
import math
import os
import re
import subprocess
import tempfile
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import cv2
import numpy as np
import pytesseract

from app.config import DetectionConfig
from app.engine import language_detect
from app.engine.detectors import movement
from app.engine.detectors.verify import marks_agreement
from app.engine.preprocessing import deskew
from app.logging_config import StageTimer
from app.models.diff import SOURCE_OCR, BoundingBox, DiffCandidate, DiffSubtype, DiffType

logger = logging.getLogger(__name__)

# A reading of a scan that the pixels do not bear out is marked with this confidence and `uncertain`: it is listed apart and
# shown only on request, so a real change in small print is not hidden but the reviewer is not shouted at.
LIKELY_MISREAD_CONFIDENCE = 0.25


class OcrUnavailableError(RuntimeError):
    """Tesseract is not installed or not on PATH."""


@dataclass
class OcrWord:
    text: str
    bbox: BoundingBox
    confidence: float
    upright: bool | None = None  # runs across the page (False: up or down it); None where not known (a reading)
    upside_down: bool = False  # (a file's own text) runs across the page right to left: printed upside down


_OCR_UNAVAILABLE = (
    "Tesseract OCR is not installed or not on PATH, so text and number "
    "differences could not be checked. Install Tesseract and restart."
)


def ocr_available() -> bool:
    """Whether the reader (Tesseract) can be run at all."""
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        return False
    except Exception:  # noqa: BLE001 -- an odd version string is still an engine that runs
        return True
    return True


def _ocr_data(image: np.ndarray, page_segmentation: str | None = None, lang: str | None = None) -> dict:
    lang = language_detect.resolve(lang)  # ("auto": the languages found on the files, waited for if still being found)
    try:
        return pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            **({"config": page_segmentation} if page_segmentation else {}),
            **({"lang": lang} if lang else {}),
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise OcrUnavailableError(_OCR_UNAVAILABLE) from exc


_TSV_COLUMNS = ("level", "page_num", "block_num", "par_num", "line_num", "word_num", "left", "top", "width", "height", "conf", "text")


def _tsv_pages(tsv: str, count: int) -> list[dict]:
    """Tesseract's TSV of a run over several pictures, as one `image_to_data` dict per picture (by its page number)."""
    pages = [{column: [] for column in _TSV_COLUMNS} for _ in range(count)]
    for row in tsv.splitlines()[1:]:
        cells = row.split("\t")
        if len(cells) < len(_TSV_COLUMNS) - 1:
            continue
        cells += [""] * (len(_TSV_COLUMNS) - len(cells))
        try:
            page = int(cells[1]) - 1
            values = [int(float(cell)) for cell in cells[:-1]]
        except ValueError:
            continue
        if 0 <= page < count:
            for column, value in zip(_TSV_COLUMNS, values + [cells[-1]]):
                pages[page][column].append(value)
    return pages


def _ocr_run(pictures: list[np.ndarray], page_segmentation: str, lang: str | None) -> list[dict]:
    """Every picture read by ONE run of the reader (Tesseract takes a list of pictures and reads each on its own, just as
    it reads one), so its start -- loading its models, which is most of what a reading of a small window costs -- is paid
    once, not once per picture."""
    if len(pictures) == 1:
        return [_ocr_data(pictures[0], page_segmentation, lang)]
    command = pytesseract.pytesseract.tesseract_cmd
    with tempfile.TemporaryDirectory(prefix="docpix_ocr_") as folder:
        # All the pictures as the pages of ONE multi-page TIFF: read the same, page for page, as one file each -- and a
        # file each (written, scanned by the antivirus, deleted) cost seconds a comparison where there are hundreds.
        source = os.path.join(folder, "pictures.tif")
        if not cv2.imwritemulti(source, pictures):
            names = []
            for number, picture in enumerate(pictures):
                name = os.path.join(folder, f"{number}.png")
                cv2.imwrite(name, picture, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                names.append(name)
            source = os.path.join(folder, "pictures.txt")
            with open(source, "w", encoding="utf-8") as handle:
                handle.write("\n".join(names) + "\n")
        lang = language_detect.resolve(lang)
        args = [command, source, "stdout", *page_segmentation.split(), *(["-l", lang] if lang else []), "tsv"]
        startup = None
        if hasattr(subprocess, "STARTUPINFO"):  # no console window flashing up for each run on Windows
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = subprocess.SW_HIDE
        try:
            run = subprocess.run(args, capture_output=True, stdin=subprocess.DEVNULL, startupinfo=startup, check=False)
        except FileNotFoundError as exc:
            raise OcrUnavailableError(_OCR_UNAVAILABLE) from exc
    if run.returncode != 0:
        logger.warning("the reader failed on a batch: %s", run.stderr.decode("utf-8", "replace")[-300:])
        return [_ocr_data(picture, page_segmentation, lang) for picture in pictures]
    return _tsv_pages(run.stdout.decode("utf-8", "replace"), len(pictures))


def _ocr_many(pictures: list[np.ndarray], page_segmentation: str, lang: str | None = None) -> list[dict]:
    """Each picture read on its own (as `_ocr_data` reads it), the lot shared among one run of the reader per core."""
    if not pictures:
        return []
    runs = max(1, min(os.cpu_count() or 2, len(pictures)))
    shares = [list(range(start, len(pictures), runs)) for start in range(runs)]  # dealt round, so each run gets big and small
    found: list[dict] = [{} for _ in range(len(pictures))]
    with ThreadPoolExecutor(max_workers=runs) as pool:
        for share, data in zip(shares, pool.map(lambda share: _ocr_run([pictures[i] for i in share], page_segmentation, lang), shares)):
            for index, page in zip(share, data):
                found[index] = page
    return found


def _words_of(data: dict, floor: float) -> list[tuple[tuple[int, int, int], OcrWord]]:
    """(the line a word is on, the word), for every word read at least as surely as `floor`."""
    found = []
    for index, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        confidence = float(data["conf"][index])
        if not text or confidence < floor:
            continue
        left, top = float(data["left"][index]), float(data["top"][index])
        word = OcrWord(
            text=text,
            bbox=BoundingBox(left, top, left + float(data["width"][index]), top + float(data["height"][index])),
            confidence=confidence,
        )
        found.append(((data["block_num"][index], data["par_num"][index], data["line_num"][index]), word))
    return found


def _ocr_words(image: np.ndarray, config: DetectionConfig, floor: float | None = None) -> list[OcrWord]:
    data = _ocr_data(image, lang=config.ocr_language)
    return [word for _line, word in _words_of(data, config.ocr_min_confidence if floor is None else floor)]


def _is_numeric(token: str) -> bool:
    return any(character.isdigit() for character in token) and not any(
        character.isalpha() for character in token
    )


def _span_type(tokens: list[str]) -> DiffType:
    """A span is a NUMBER change only if it is all numbers (FR-6.1)."""
    return DiffType.NUMBER if tokens and all(_is_numeric(token) for token in tokens) else DiffType.TEXT


def _span_bbox(words: list[OcrWord]) -> BoundingBox | None:
    if not words:
        return None
    box = words[0].bbox
    for word in words[1:]:
        box = box.union(word.bbox)
    return box


def _mean_confidence(words: list[OcrWord]) -> float:
    return float(np.mean([word.confidence for word in words])) / 100 if words else 0.5


def detect(
    original: np.ndarray, suspect: np.ndarray, config: DetectionConfig
) -> list[DiffCandidate]:
    """Text/number differences between two images already in the shared aligned space."""
    with StageTimer(logger, "detector_text"):
        # Rotated text, a barcode's digits and a dotted line are read as junk words; keep only what reads as words.
        original_words = _trustworthy(_ocr_words(original, config), config)
        suspect_words = _trustworthy(_ocr_words(suspect, config), config)

        matcher = SequenceMatcher(
            None,
            [word.text for word in original_words],
            [word.text for word in suspect_words],
            autojunk=False,
        )

        candidates: list[DiffCandidate] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            original_span = original_words[i1:i2]
            suspect_span = suspect_words[j1:j2]

            if tag == "equal":
                candidates.extend(_moved_candidates(original_span, suspect_span, config))
            else:
                # replace / delete / insert. Words far apart are separate findings (a long run of misaligned
                # readings is not one change the size of the page): see `_findings`.
                candidates.extend(_findings(original_span, suspect_span))

    candidates = [candidate for candidate in candidates if candidate is not None]
    logger.info(
        "text detector finished",
        extra={
            "original_words": len(original_words),
            "suspect_words": len(suspect_words),
            "candidates": len(candidates),
        },
    )
    return candidates


def _moved_candidates(
    original_span: list[OcrWord], suspect_span: list[OcrWord], config: DetectionConfig
) -> list[DiffCandidate]:
    """Identical tokens that sit somewhere else are moves, not matches."""
    candidates = []
    for original_word, suspect_word in zip(original_span, suspect_span):
        original_center, suspect_center = original_word.bbox.center, suspect_word.bbox.center
        displacement = math.dist(original_center, suspect_center)
        if displacement > config.text_move_tolerance_px:
            candidates.append(
                _candidate(
                    original_word.bbox.union(suspect_word.bbox),
                    DiffType.NUMBER if _is_numeric(original_word.text) else DiffType.TEXT,
                    DiffSubtype.MOVED,
                    _mean_confidence([original_word, suspect_word]),
                    f"'{original_word.text}' moved {displacement:.0f} px",
                    magnitude=displacement,  # how far it moved: words that moved alike are one move
                )
            )
    return candidates


def _join(words: list[OcrWord]) -> str:
    return " ".join(word.text for word in words)


def _candidate(
    bbox: BoundingBox | None,
    diff_type: DiffType,
    subtype: DiffSubtype,
    confidence: float,
    detail: str,
    magnitude: float = 0.0,
    uncertain: bool = False,
) -> DiffCandidate | None:
    if bbox is None:
        return None
    return DiffCandidate(
        bbox=bbox, type=diff_type, subtype=subtype, confidence=confidence, detail=detail, magnitude=magnitude, uncertain=uncertain
    )


# ---------------------------------------------------------------------------
# Comparing a scan with a digital file
# ---------------------------------------------------------------------------

# An l, a 1 and an i are one smudge apart in a soft scan, and so are an O and a 0.
_OCR_CONFUSABLE = str.maketrans({"l": "i", "1": "i", "0": "o"})
_MIN_ALNUM_SHARE = 0.5  # a "word" that is mostly marks is not one
_MAX_LEFTOVER_SPREAD = 8.0  # words this scattered (box area / area of the words) are separate findings
_UNSURE_DIGITAL_MAX = 55.0  # a difference whose digital words were all read less surely than this, even closely, is unsure
_UNSURE_WORD_CHARS = 4  # (an unsure word this long, on a line read surely, is compared: shorter ones are specks' guesses)
_SHAKY_DIGITAL_MAX = 75.0  # the digital file's words are read no more surely than this...
_SURE_SCAN_MIN = 80.0  # ...where the scan's are read at least this surely...
_SHAKY_MATCH_RATIO = 0.8  # ...and they are this alike
_CUT_SHORT_CHARS = 2  # (one reading the other with this many letters more at an end is a word cut short, not a misread)
_WINDOW_UPSCALE_BELOW_PX = 80  # a line's window is enlarged to at least this tall before it is read
_DARK = 128  # an ink picture's pixels below this are ink
_AGREEMENT_SLACK_PX = 2  # ink this close counts as the same ink when judging a misread
_AGREEMENT_PAD_PX = 6


@dataclass
class TextScanResult:
    candidates: list[DiffCandidate]
    # Lines of the digital file whose ink is in the scan but which the scan is too soft to read: not compared.
    unreadable_lines: int = 0
    # Where the words read the SAME in both files (each word's box in the digital file with its partner's in the scan):
    # text compared as text and found unchanged, so what its ink looks like in the scan is the scan's, not a difference.
    same_text: list[BoundingBox] = field(default_factory=list)


def canonical(text: str) -> str:
    """What a word is, for comparing across a scan: no case, no punctuation, no l/i/1 or o/0 confusion."""
    # (a degree sign is read as a nought as often as not: '175°C' is '1750C' to the reader)
    return "".join(ch for ch in text.casefold().replace("°", "0") if ch.isalnum()).translate(_OCR_CONFUSABLE)


_MIN_LONE_WORD_CHARS = 4  # a line of one word shorter than this is not compared...
_MIN_LONE_NUMBER_CHARS = 2  # ...unless it is a number at least this long


def _lone_number(text: str) -> bool:
    token = canonical(text)
    return len(token) >= _MIN_LONE_NUMBER_CHARS and sum(ch.isdigit() for ch in text) >= _MIN_LONE_NUMBER_CHARS


def _has_digit(*texts: str) -> bool:
    return any(ch.isdigit() for text in texts for ch in text)


_LONE_LETTER_CONFIDENCE = 90.0  # a letter on its own is a word only if read this surely (a speck reads as 'l' or 'i', unsurely)


def _trustworthy(words: list[OcrWord], config: DetectionConfig) -> list[OcrWord]:
    """Drop what is read as marks rather than words: mostly punctuation, or a very short word read shakily. A lone
    digit read surely (or a lone letter read very surely) is a word where it stands in a phrase, beside a word of its
    line ('7 x 7 mm' -> '9 x 9 mm', '1 tablet'): on a pack it is often the very thing that changed. Standing alone it is
    a speck or a glyph of a picture read as '0' or 'l'."""
    kept, lone = [], []
    for word in words:
        key = canonical(word.text)
        alnum = sum(ch.isalnum() for ch in word.text)
        if alnum < _MIN_ALNUM_SHARE * len(word.text):
            continue
        if len(key) < 2:
            if alnum == 1 and (any(ch.isdigit() for ch in word.text) and word.confidence >= config.ocr_short_word_confidence
                               or word.confidence >= _LONE_LETTER_CONFIDENCE):
                lone.append(word)
            continue
        if len(key) <= 2 and word.confidence < config.ocr_short_word_confidence:
            continue
        kept.append(word)
    # a lone character in a phrase: within a few of its heights along its line of a word kept, or of another such
    # character that is (the chain '7', 'x', '7' leading to 'mm)')
    phrase = list(kept)
    changed = True
    while changed and lone:
        changed = False
        for word in list(lone):
            if any(_beside(word, other) for other in phrase):
                phrase.append(word)
                lone.remove(word)
                changed = True
    return [word for word in words if any(word is w for w in phrase)]


_PHRASE_GAP_HEIGHTS = 2.0  # (a lone character this many of its heights from a word of its line is in its phrase)


def _beside(word: OcrWord, other: OcrWord) -> bool:
    a, b = word.bbox, other.bbox
    tall = max(a.height, b.height, 1.0)
    shared = min(a.y1, b.y1) - max(a.y0, b.y0)
    gap = max(a.x0 - b.x1, b.x0 - a.x1, 0.0)
    return shared >= 0.5 * min(a.height, b.height) and gap <= _PHRASE_GAP_HEIGHTS * tall


def _same_word(a: OcrWord, b: OcrWord) -> bool:
    """Whether two readings are the same word once case, punctuation and the l/i/1 and o/0 confusions are set aside.

    Nothing looser: 'Store' against 'Stone' is one letter, and so is a real change to a word. A scan's smudged
    letters are instead ranked down when the ink shows the marks are the same (`_marked_if_misread`).
    """
    return canonical(a.text) == canonical(b.text)


def _leftovers(
    original_span: list[OcrWord], suspect_span: list[OcrWord]
) -> tuple[list[OcrWord], list[OcrWord]]:
    """The words of a replaced stretch that no word on the other side reads as: each word is matched, in order,
    to the first still-unused word it is the same as (a differently grouped line)."""
    unused = list(suspect_span)
    unmatched_original = []
    for word in original_span:
        match = next((other for other in unused if _same_word(word, other)), None)
        if match is None:
            unmatched_original.append(word)
        else:
            unused.remove(match)
    return unmatched_original, unused


_RUN_GAP_PX = 60  # words further apart than this (~5 mm at 300 DPI) are not one phrase


def _runs(words: list[OcrWord]) -> list[list[OcrWord]]:
    """The words grouped into phrases: each follows the one before it closely, on the same line."""
    runs: list[list[OcrWord]] = []
    for word in words:
        if runs:
            last = runs[-1][-1].bbox
            same_line = abs(word.bbox.center[1] - last.center[1]) <= max(last.height, word.bbox.height)
            if same_line and 0 <= word.bbox.x0 - last.x1 <= _RUN_GAP_PX:
                runs[-1].append(word)
                continue
        runs.append([word])
    return runs


def _with_layer(page: list[tuple[tuple, OcrWord]], layer: list[OcrWord] | None) -> list[tuple[tuple, OcrWord]]:
    """A digital page's words, (line, word): its own text's words where it has any (exact; set out in lines of their
    own), and what was read only where it has none -- a word read over a word of the file's text is that word."""
    if not layer:
        return page
    own = [word for word in layer if any(ch.isalnum() for ch in word.text) and word.upright is not False and not word.upside_down]
    kept = [
        (line, word) for line, word in page
        if not any(word.bbox.intersection_area(mine.bbox) >= _LAYER_OVER * max(1.0, min(word.bbox.area, mine.bbox.area)) for mine in own)
    ]
    for number, line_words in enumerate(_lines_of(own)):
        kept += [(("layer", number), word) for word in line_words]
    return kept


def _layer_in(box: BoundingBox, layer: list[OcrWord] | None) -> list[OcrWord]:
    """The digital file's own words in a line's window (`box`, as the window is read: the words on its line), in order;
    [] where it has none there."""
    if not layer:
        return []
    inside = [
        word for word in layer
        if any(ch.isalnum() for ch in word.text) and word.upright is not False and not word.upside_down
        and box.y0 <= word.bbox.center[1] <= box.y1
        and word.bbox.x1 >= box.x0 and word.bbox.x0 <= box.x1
    ]
    return sorted(inside, key=lambda word: word.bbox.x0)


_LAYER_OVER = 0.5  # (a word of the file's own text overlapping a word read by this share of the smaller is that word)


def _digital_reading_unsure(gone: list[OcrWord], new: list[OcrWord]) -> bool:
    """Whether a difference is the DIGITAL file's reading being unsure: what it read is a shaky near-miss of a word the
    scan reads confidently ('Cilnidipene' at 67% for 'Cilnidipine' at 87%: a hairline typeface is hard to read even
    in a crisp file). A real change to a word is read surely on the digital side.

    A word CUT SHORT is not a near-miss, however shaky its reading ('Cilnidip' for 'Cilnidipine' -- the digital file
    really ends there): a reader misreads letters, it does not drop the end of a word it can see."""
    if gone and new and max(word.confidence for word in gone) < _UNSURE_DIGITAL_MAX:
        return True  # (read too unsurely to say anything at all, even closely: 'nyectatle' for 'Inyectable')
    if not gone or not new or max(word.confidence for word in gone) >= _SHAKY_DIGITAL_MAX or min(word.confidence for word in new) < _SURE_SCAN_MIN:
        return False
    old, fresh = canonical(_join(gone)), canonical(_join(new))
    shorter, longer = sorted((old, fresh), key=len)
    if len(longer) - len(shorter) >= _CUT_SHORT_CHARS and (longer.startswith(shorter) or longer.endswith(shorter)):
        return False
    ratio = SequenceMatcher(None, old, fresh, autojunk=False).ratio()
    return ratio >= _SHAKY_MATCH_RATIO


_UNRELATED_RATIO = 0.4  # two readings less alike than this (their letters) are not one changed into the other


def _related(gone: list[OcrWord], new: list[OcrWord]) -> bool:
    """Whether what the scan reads is plausibly what the digital file has, changed -- rather than other text that only
    happens to be where the digital file's was (text re-arranged: a line of another panel, a code turned on its side).
    A number for a number is always a change ('1g' -> '28mg'), however unlike their digits."""
    if _has_digit(_join(gone)) and _has_digit(_join(new)):  # (the readings as read: `canonical` turns 1 and 0 into letters)
        return True
    return SequenceMatcher(None, canonical(_join(gone)), canonical(_join(new)), autojunk=False).ratio() >= _UNRELATED_RATIO


def _findings(gone: list[OcrWord], new: list[OcrWord]) -> list[DiffCandidate]:
    """Findings for the words left over in one replaced stretch: one change where the two are alike, else what is missing
    and what is there instead, each boxed where it is (one box round both ran across the panel into the next)."""
    if gone and new and _related(gone, new):
        both = gone + new
        box = _span_bbox(both)
        spread = box.area / max(1.0, sum(word.bbox.area for word in both))
        if spread <= _MAX_LEFTOVER_SPREAD:
            unsure = _digital_reading_unsure(gone, new)
            return [
                _candidate(
                    box,
                    _span_type([word.text for word in both]),
                    DiffSubtype.CHANGED,
                    LIKELY_MISREAD_CONFIDENCE if unsure else _mean_confidence(both),
                    f"'{_join(gone)}' -> '{_join(new)}'" + (" (the digital file's reading is unsure: likely a misread)" if unsure else ""),
                    uncertain=unsure,
                )
            ]
    findings = []
    for words, subtype, wording in ((gone, DiffSubtype.REMOVED, "missing from suspect"), (new, DiffSubtype.ADDED, "added in suspect")):
        for span in _runs(words):
            unsure = subtype is DiffSubtype.REMOVED and max(word.confidence for word in span) < _UNSURE_DIGITAL_MAX
            findings.append(
                _candidate(
                    _span_bbox(span), _span_type([word.text for word in span]), subtype,
                    LIKELY_MISREAD_CONFIDENCE if unsure else _mean_confidence(span),
                    f"'{_join(span)}' {wording}" + (" (the digital file's reading is unsure: likely a misread)" if unsure else ""),
                    uncertain=unsure,
                )
            )
    return findings


def _line_differences(original_line: list[OcrWord], suspect_line: list[OcrWord]) -> list[DiffCandidate]:
    """The differences between two readings of one line."""
    matcher = SequenceMatcher(
        None, [canonical(w.text) for w in original_line], [canonical(w.text) for w in suspect_line], autojunk=False
    )
    found: list[DiffCandidate] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        gone, new = _leftovers(original_line[i1:i2], suspect_line[j1:j2])
        found += _findings(gone, new)
    return [candidate for candidate in found if candidate is not None]


def _padded(box: BoundingBox, pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    return max(0, int(box.x0) - pad), max(0, int(box.y0) - pad), min(width, math.ceil(box.x1) + pad), min(height, math.ceil(box.y1) + pad)


def _ink_in(image: np.ndarray, box: BoundingBox, pad: int = 0) -> int:
    """How much ink an ink picture has inside `box`."""
    x0, y0, x1, y1 = _padded(box, pad, image.shape)
    return int((image[y0:y1, x0:x1] < _DARK).sum())


# --- reading a whole page fast ---------------------------------------------------------------------------------------

_STRIP_MIN_HEIGHT_PX = 900  # a page is cut into strips at least this tall, as many as there are cores to read them
_STRIP_SEARCH_SHARE = 0.4  # a cut is looked for this share of a strip's height either side of the even split
_BLOCK_CLOSE_PX = 15  # lettering reversed out of a block is filled in, to see the block, with a square this big
_BLOCK_MIN_AREA_PX = 2500  # a solid block at least this big has its lettering read dark on light
_BLOCK_MIN_SIDE_PX = 24
_BLOCK_MIN_FILL = 0.55  # share of a block that is ink: a paragraph closed up is ~0.3, a bar with lettering ~0.75
_BLOCK_CORE_PX = 13  # ink that survives being shrunk by this much is SOLID: a stroke of lettering is thinner
_BLOCK_MIN_CORE = 0.15  # a block is at least this share solid ink (bold headings, whose closed lines fill as much, are not)
_BAR_MIN_LETTERS = 3  # a bar found on its own (`_blocks`) has at least this many letters reversed out of it...
_BAR_LETTER_MIN_PX = 20  # ...each a hole at least this big (a scan's speck in a bar is not a letter)
_BAR_MIN_SQUARENESS = 0.85  # ...and fills this much of its box: a bar, not a bold glyph or a notched edge
_ADDED_SAME_WORD_HEIGHTS = 5  # a word the digital file has this many word-heights off is not added: it moved
_WINDOW_BORDER_PX = 22  # white set round a window before it is read
_WINDOW_PAD_ACROSS = 0.15  # a line's window reaches this share of its height above and below it
_WINDOW_PAD_ALONG = 2.0  # ...and at least this many of its heights along it, either end
_WINDOW_MAX_WIDTH_PX = 4200  # a window is not enlarged past this width
_MATCH_MIN_REACH_PX = 40  # a word in the scan is 'the same word' if it is this close to it (two word-heights, if more):
_MATCH_REACH_HEIGHTS = 2.0  # a two-sheet scan of a carton was 45 px off in places
_MATCH_GRID_PX = 96


def reading_picture(ink: np.ndarray) -> np.ndarray:
    """An ink picture as it is READ: text reads best dark on light, so inside a solid block (white lettering
    reversed out of a dark bar) the picture is reversed and the lettering reads dark on light like any other.

    Left as it is, such lettering is read as junk in the file that has it, and the words 'missing' or 'changed'
    over every dark bar of the sheet were a large part of the false findings on a real pair.
    """
    blocks, closed_blocks = _blocks(ink)
    return _stroked(_reversed(ink, blocks), _reversed(ink, closed_blocks))


def _reversed(ink: np.ndarray, where: np.ndarray | None) -> np.ndarray:
    """`ink` reversed (dark for light) `where` that is set."""
    if where is None or not where.any():
        return ink
    result = ink.copy()
    result[where] = 255 - result[where]
    return result


def _blocks(ink: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(every solid block of an ink picture -- a bar its lettering is reversed out of -- and those of them found by
    closing their lettering over, `_BLOCK_CLOSE_PX`; None for none).

    A bar set close above or below a line of text is closed into that text as well, and the whole is no block (a
    heading bar a few pixels over its paragraph: its lettering read as nothing at all, and a word left out of it never
    found). So where the closing found no block, each piece of ink is looked at on its own, with the lettering inside
    it -- the holes in it, letters and their counters, no wider than half the piece -- filled in: a bar, if it fills
    most of its box and has a few letters reversed out of it (a bold glyph with a counter, a notched edge, is not). (Whether the strokes
    are thickened for reading is decided on the blocks the closing found, as before the others were: `reading_picture`.)"""
    dark = (ink < _DARK).astype(np.uint8)
    size = 2 * (_BLOCK_CLOSE_PX // 2) + 1
    closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (size, size)))
    solid = cv2.erode(dark, np.ones((_BLOCK_CORE_PX, _BLOCK_CORE_PX), np.uint8))

    def is_block(window: tuple[slice, slice], inside: np.ndarray) -> bool:
        return bool(dark[window][inside].mean() >= _BLOCK_MIN_FILL and solid[window][inside].mean() >= _BLOCK_MIN_CORE)

    by_closing = np.zeros(ink.shape, bool)
    not_blocks = np.zeros(ink.shape, np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    for label in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[label])
        if area < _BLOCK_MIN_AREA_PX or min(width, height) < _BLOCK_MIN_SIDE_PX:
            continue
        window = (slice(y, y + height), slice(x, x + width))
        inside = labels[window] == label
        if is_block(window, inside):
            by_closing[window] |= inside
        else:
            not_blocks[window][inside] = 1
    every = by_closing.copy()
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark & not_blocks, connectivity=8)
    for label in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[label])
        if area < _BLOCK_MIN_AREA_PX * _BLOCK_MIN_FILL or min(width, height) < _BLOCK_MIN_SIDE_PX:
            continue
        window = (slice(y, y + height), slice(x, x + width))
        piece = (labels[window] == label).astype(np.uint8)
        outside = cv2.copyMakeBorder(1 - piece, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
        cv2.floodFill(outside, None, (0, 0), 0)  # (what is left of it: the holes in the piece)
        holes = outside[1:-1, 1:-1]
        if not holes.any():
            continue  # (nothing reversed out of it)
        hole_count, hole_labels, hole_stats, _centres = cv2.connectedComponentsWithStats(holes, connectivity=4)
        widest = np.zeros(hole_count, np.float32)
        np.maximum.at(widest, hole_labels.ravel(), cv2.distanceTransform(holes, cv2.DIST_L2, 3).ravel())
        lettering = (widest <= min(width, height) / 2) & (hole_stats[:, cv2.CC_STAT_AREA] >= _BAR_LETTER_MIN_PX)
        lettering[0] = False
        if lettering.sum() < _BAR_MIN_LETTERS:
            continue  # a bold glyph (a counter or two in it), a shape: not a bar with its lettering
        piece |= lettering[hole_labels].astype(np.uint8)
        inside = piece.astype(bool)
        if inside.sum() >= max(_BLOCK_MIN_AREA_PX, _BAR_MIN_SQUARENESS * width * height) and is_block(window, inside):
            every[window] |= inside
    return (every if every.any() else None), (by_closing if by_closing.any() else None)


_THIN_STROKE_PX = 2.8  # ink whose strokes are on average thinner than this is thickened for reading


def _stroked(picture: np.ndarray, measured: np.ndarray | None = None) -> np.ndarray:
    """A picture whose ink is at least a few pixels thick, for reading. A light-weight typeface is a hairline in a
    crisp digital file (two pixels wide, on letters forty pixels tall) and Tesseract reads it much less surely than
    the scan of it, whose ink has spread: 'Cilnidioine' for 'Cilnidipine' at 57%. The average stroke is measured from
    the picture (twice the ink's area over its edge) and thickened by a pixel if it is thinner than
    `_THIN_STROKE_PX`; ordinary lettering, and a scan's fat strokes, are left as they are. (`measured`, if given, is the
    picture the strokes are measured on.)
    """
    dark = (picture if measured is None else measured) < _DARK
    area = int(dark.sum())
    if area < 500:
        return picture
    edge = int((dark & ~cv2.erode(dark.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)).sum())
    if 2.0 * area / max(edge, 1) >= _THIN_STROKE_PX:
        return picture
    thick = cv2.dilate((picture < _DARK).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return np.where(thick, 0, picture).astype(np.uint8)


def _cut_rows(picture: np.ndarray, parts: int) -> list[int]:
    """Rows to cut a page into `parts` strips at: near an even split, each in the emptiest row close by, so a
    line of text is not cut through."""
    height = picture.shape[0]
    if parts <= 1:
        return [0, height]
    ink = cv2.reduce((picture < _DARK).astype(np.uint8), 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S).ravel()
    reach = int(_STRIP_SEARCH_SHARE * height / parts)
    cuts = [0]
    for part in range(1, parts):
        even = round(part * height / parts)
        low, high = max(cuts[-1] + 1, even - reach), min(height - 1, even + reach)
        if high < low:
            cuts.append(even)
            continue
        window = ink[low : high + 1]
        emptiest = np.flatnonzero(window == window.min()) + low
        cuts.append(int(emptiest[np.argmin(np.abs(emptiest - even))]))  # the emptiest row, the one nearest the even split
    cuts.append(height)
    return cuts


def read_pages(pictures: list[np.ndarray], floor: float, lang: str | None = None) -> list[list[tuple[tuple, OcrWord]]]:
    """Every word read on each picture, as (line, word): each page is cut into strips read side by side.

    Reading a page is by far the slowest step of Scan vs Digital (a text-dense sheet takes a few seconds on one
    core), so the strips of both pages are read together, one strip to a core.
    """
    workers = max(1, min(_MAX_READERS, os.cpu_count() or 2))
    jobs = []  # (picture number, strip number, first row, last row)
    for number, picture in enumerate(pictures):
        parts = max(1, min(workers, picture.shape[0] // _STRIP_MIN_HEIGHT_PX))
        cuts = _cut_rows(picture, parts)
        jobs += [(number, strip, cuts[strip], cuts[strip + 1]) for strip in range(len(cuts) - 1)]

    def read(job) -> list[tuple[tuple, OcrWord]]:
        number, strip, first, last = job
        found = []
        for line, word in _words_of(_ocr_data(pictures[number][first:last], lang=lang), floor):
            b = word.bbox
            moved = OcrWord(word.text, BoundingBox(b.x0, b.y0 + first, b.x1, b.y1 + first), word.confidence)
            found.append(((strip, *line), moved))
        return found

    with StageTimer(logger, "read_pages"):
        if len(jobs) == 1:
            results = [read(jobs[0])]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(read, jobs))
    pages: list[list[tuple[tuple, OcrWord]]] = [[] for _ in pictures]
    for job, found in zip(jobs, results):
        pages[job[0]] += found
    return pages


def read_windows(picture: np.ndarray, boxes: list[BoundingBox], config: DetectionConfig) -> list[list[OcrWord]]:
    """The words in a small window round each box (one line of text each), in the picture's own pixels.

    Each window is read ON ITS OWN -- enlarged if a little short of a line, set in white -- but all of them by a few runs
    of the reader (`_ocr_many`): reading them one program start at a time was most of the time Scan vs Digital took.
    (They were once stacked into one picture and read together; how a line read then depended on which other lines
    happened to be stacked with it.) The words are sorted along the line.
    """
    prepared = []  # (window's corner, scale, picture), or None
    for box in boxes:
        # padded along the line, but hardly above and below it: the next line's ascenders and the last one's descenders
        # in the window were read as words of this line ('SWe' of 'swallowed' above 'chew.')
        # (and along it by two of its heights at least: a word that grew -- 'ch' to 'chew.' -- runs past the old one's end)
        x0, _y0, x1, _y1 = _padded(box, max(config.text_line_pad_px, round(_WINDOW_PAD_ALONG * box.height)), picture.shape)
        _x0, y0, _x1, y1 = _padded(box, min(config.text_line_pad_px, max(1, round(_WINDOW_PAD_ACROSS * box.height))), picture.shape)
        if x1 - x0 < 8 or y1 - y0 < 8:
            prepared.append(None)
            continue
        crop = picture[y0:y1, x0:x1]
        factor = min(max(1.0, _WINDOW_UPSCALE_BELOW_PX / crop.shape[0]), _WINDOW_MAX_WIDTH_PX / crop.shape[1])
        if factor > 1.0:
            crop = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
        border = _WINDOW_BORDER_PX
        crop = cv2.copyMakeBorder(crop, border, border, border, border, cv2.BORDER_CONSTANT, value=255)
        prepared.append((x0, y0, factor, crop))

    chosen = [i for i, item in enumerate(prepared) if item is not None]
    words: list[list[OcrWord]] = [[] for _ in boxes]
    with StageTimer(logger, "read_windows"):
        readings = _ocr_many([prepared[i][3] for i in chosen], _BLOCK_PAGE_SEGMENTATION, config.ocr_language)
    for index, data in zip(chosen, readings):
        x0, y0, factor, _crop = prepared[index]
        border = _WINDOW_BORDER_PX
        found = []
        for _line, word in _words_of(data, config.ocr_min_confidence):
            b = word.bbox
            found.append(
                OcrWord(
                    word.text,
                    BoundingBox((b.x0 - border) / factor + x0, (b.y0 - border) / factor + y0, (b.x1 - border) / factor + x0, (b.y1 - border) / factor + y0),
                    word.confidence,
                )
            )
        found.sort(key=lambda word: word.bbox.x0)
        words[index] = _trustworthy(found, config)
    return words


def _unmatched(anchor: list[OcrWord], scan: list[OcrWord]) -> tuple[set[int], set[int], list[tuple[int, int]]]:
    """(the digital words nothing in the scan matches, the scan words nothing in the digital file matches, and the
    (digital, scan) pairs that do match).

    A word matches a word of the scan that reads the same (`canonical`) and lies close to where it is, so where a
    page reads in a different order, or a line is grouped differently, nothing is out of step. Each scan word
    matches at most one digital word, the nearest first.
    """
    grid: dict[tuple[int, int], list[int]] = {}
    scan_keys = [canonical(word.text) for word in scan]
    for index, word in enumerate(scan):
        x, y = word.bbox.center
        grid.setdefault((int(x // _MATCH_GRID_PX), int(y // _MATCH_GRID_PX)), []).append(index)
    used: set[int] = set()
    missing: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for index, word in enumerate(anchor):
        key = canonical(word.text)
        x, y = word.bbox.center
        reach = max(float(_MATCH_MIN_REACH_PX), _MATCH_REACH_HEIGHTS * word.bbox.height)
        best, best_distance = None, reach
        for cell_x in range(int((x - reach) // _MATCH_GRID_PX), int((x + reach) // _MATCH_GRID_PX) + 1):
            for cell_y in range(int((y - reach) // _MATCH_GRID_PX), int((y + reach) // _MATCH_GRID_PX) + 1):
                for other in grid.get((cell_x, cell_y), ()):
                    if other in used or scan_keys[other] != key:
                        continue
                    ox, oy = scan[other].bbox.center
                    distance = math.hypot(ox - x, oy - y)
                    if distance < best_distance:
                        best, best_distance = other, distance
        if best is None:
            missing.add(index)
        else:
            used.add(best)
            pairs.append((index, best))
    return missing, set(range(len(scan))) - used, pairs


_OFFSET_RADIUS_PX = 600  # where the scan sits against the digital file is judged from the words matched this close
_OFFSET_MIN_PAIRS = 3


def _local_offset(
    pairs: list[tuple[int, int]], anchor: list[OcrWord], scan: list[OcrWord], box: BoundingBox
) -> tuple[float, float]:
    """How far the scan is from the digital file round `box`: the median displacement of the words that matched
    nearby. A scan of a carton is bent and scaled a little, so it is off by tens of pixels in places -- enough that
    a window cut from the digital file's line misses the first letter of the scan's."""
    cx, cy = box.center
    shifts = []
    for i, j in pairs:
        ax, ay = anchor[i].bbox.center
        if abs(ax - cx) <= _OFFSET_RADIUS_PX and abs(ay - cy) <= _OFFSET_RADIUS_PX:
            bx, by = scan[j].bbox.center
            shifts.append((bx - ax, by - ay))
    if len(shifts) < _OFFSET_MIN_PAIRS:
        return (0.0, 0.0)
    return (float(np.median([dx for dx, _ in shifts])), float(np.median([dy for _, dy in shifts])))


def _moved(words: list[OcrWord], offset: tuple[float, float]) -> list[OcrWord]:
    """Words moved by `offset` (the scan's words back into the digital file's frame use the negative of it)."""
    return [OcrWord(w.text, _shifted_box(w.bbox, offset), w.confidence) for w in words]


def _union_of_readings(first: list[OcrWord], second: list[OcrWord]) -> list[OcrWord]:
    """Two readings of one line as one: all of the first, and each word of the second that lies where none of the
    first's does. Whichever reading found a word, it is there."""
    merged = list(first)
    for word in second:
        if not any(word.bbox.intersection_area(other.bbox) > 0.3 * word.bbox.area for other in first):
            merged.append(word)
    return sorted(merged, key=lambda w: w.bbox.x0)


def _fewest_differences(digital: list[list[OcrWord]], scan: list[list[OcrWord]]) -> list[DiffCandidate]:
    """The differences between the readings that agree best: the same line read (or taken from the page) in the several
    ways, in either file. A word that one way of reading misses and another finds is not a difference, and a real
    change is different in every one of them."""
    best: list[DiffCandidate] | None = None
    best_cost: tuple[int, int, int] | None = None
    for reading in digital:
        for other in scan:
            if not other or not reading:
                continue
            found = _line_differences(reading, other)
            if not found:
                return []
            matcher = SequenceMatcher(None, [canonical(w.text) for w in reading], [canonical(w.text) for w in other], autojunk=False)
            unmatched = len(reading) + len(other) - 2 * sum(block.size for block in matcher.get_matching_blocks())
            # fewest findings; then fewest words said missing or added (a way of reading that missed a word another way
            # found -- 'Cilnidipine' lost from the scan's whole page, read in its line's window -- is not the evidence);
            # then fewest words that do not line up
            gaps = sum(1 for finding in found if finding.subtype in (DiffSubtype.ADDED, DiffSubtype.REMOVED))
            cost = (len(found), gaps, unmatched)
            if best_cost is None or cost < best_cost:
                best, best_cost = found, cost
    return best or []


def _is_upright_unreadable(words: list[OcrWord]) -> bool:
    """Whether a line was read from text that is turned on its side: its words are taller than they are wide.
    Read upright it is junk ('HSINYVANN'), so it is not compared."""
    tall = sum(1 for word in words if word.bbox.height > 1.5 * word.bbox.width and word.bbox.height >= 40)
    return tall * 2 > len(words)


def _unrotated(word: OcrWord, level: "deskew.Deskew") -> OcrWord:
    """`word`, read from a picture levelled by `level`, with its box turned back to the picture's own frame."""
    return OcrWord(word.text, level.unrotate_box(word.bbox), word.confidence) if not level.is_identity else word


def detect_scan(
    original: np.ndarray,
    suspect: np.ndarray,
    config: DetectionConfig,
    reading: tuple[np.ndarray, np.ndarray] | None = None,
    alternate: tuple[np.ndarray, np.ndarray] | None = None,
    digital_layer: list[OcrWord] | None = None,
) -> TextScanResult:
    """Text differences between a digital file and a scan of it, both as ink pictures in the shared frame.

    Both files are read whole (in strips, side by side) and their words matched by what they say and where they
    are. Only where words do not match is anything read again, line by line: the line's own small window, in both
    files, which is what says whether a difference is real (a changed word cannot reshuffle the reading order the
    way a whole-page comparison lets it). A scan is never registered exactly, so each line is compared where the
    scan actually is (judged from the words that matched nearby), and every way the line was read -- its window,
    the whole-page reading -- is tried: only if none of them agree is it a difference. See
    `DetectionConfig.ocr_min_confidence_scan` for what is ignored and why.

    `original` and `suspect` are the ink pictures (what the ink agreement of a difference is judged on); `reading` is
    the pair of pictures the words are read from, if not those (see `DetectionConfig.scan_reading_scan`), and `alternate`
    another pair the lines that differ are read from as well (see `DetectionConfig.scan_reading_alternate`).

    The WHOLE-PAGE reading below is what a residual tilt breaks (see `preprocessing.deskew`): each picture is
    levelled for that one reading only, and every word it returns is turned back to the shared, unrotated frame
    everything else in this function works in -- the re-reads further down (a line's own small window) see the
    picture exactly as `reading` gave it, untouched.
    """
    with StageTimer(logger, "detector_text"):
        blocks_original, blocks_suspect = _blocks(original), _blocks(suspect)
        if reading is None:
            reading_original, reading_suspect = (
                _stroked(_reversed(picture, blocks), _reversed(picture, closed_blocks))
                for picture, (blocks, closed_blocks) in ((original, blocks_original), (suspect, blocks_suspect))
            )
        else:
            reading_original, reading_suspect = (reading_picture(picture) for picture in reading)
        # Whether a word the scan has is in the digital file is judged by its lettering's ink, in a bar as anywhere (a bar
        # is in both files, whatever is reversed out of it): where either file has a bar, both are reversed -- a scan's
        # grainy bar is not always found a bar, and one file's lettering set against the other's solid bar is no measure.
        either = [mask for mask, _closed in (blocks_original, blocks_suspect) if mask is not None]
        lettering_original, lettering_suspect = original, suspect
        if either:
            both = either[0] if len(either) == 1 else either[0] | either[1]
            lettering_original, lettering_suspect = _reversed(original, both), _reversed(suspect, both)
        level_original, level_suspect = deskew.measured(reading_original, config), deskew.measured(reading_suspect, config)
        raw_digital_page, raw_scan_page = read_pages(
            [level_original.picture(reading_original), level_suspect.picture(reading_suspect)],
            config.ocr_min_confidence,
            config.ocr_language,
        )
        # The digital file's own text, where it has any, IS its words -- exactly, every one of them, whether or not the
        # reader happened to make them out (large light type it may miss altogether): what it reads is used only where
        # the file has no text of its own (type outlined to paths, a picture). See `_with_layer`.
        digital_page = _with_layer([(line, _unrotated(word, level_original)) for line, word in raw_digital_page], digital_layer)
        scan_page = [(line, _unrotated(word, level_suspect)) for line, word in raw_scan_page]

        lines: dict[tuple, list[OcrWord]] = {}
        for line, word in digital_page:
            lines.setdefault(line, []).append(word)
        # A word the whole-page reading was unsure of is compared where its line has words read surely (large light
        # type is read unsurely whole-page: 'Cilnidip', 41%, beside 'Release,' 93%): the line is then read again closely,
        # in both files, if it differs -- which is what decides. A line read unsurely throughout is not compared.
        lines = {
            line: [w for w in words if w.confidence >= config.ocr_min_confidence_scan or len(canonical(w.text)) >= _UNSURE_WORD_CHARS]
            for line, words in lines.items()
            if any(w.confidence >= config.ocr_min_confidence_scan for w in words)
        }
        # (a 'line' of the reader's layout is split where its words lie on different rows: two printed lines taken for
        # one made a window over both, and both were then misread)
        # (and where a wide gap runs through a row -- two panels' words on one baseline -- it is two lines)
        anchors = [
            part
            for words in lines.values()
            for row in _rows(words)
            for part in _split_at_gaps(_trustworthy(sorted(row, key=lambda w: w.bbox.x0), config))
        ]
        # A line of one short word is usually a speck read as letters, and is not compared -- unless it is a number:
        # a strength or a quantity ("10" of "10 g", its unit read too unsurely to count) is what matters most on a pack.
        anchors = [
            words
            for words in anchors
            if words
            and (len(words) > 1 or len(canonical(words[0].text)) >= _MIN_LONE_WORD_CHARS or _lone_number(words[0].text))
            and not _is_upright_unreadable(words)
        ]
        scan_words = [word for word in _trustworthy([word for _line, word in scan_page], config) if not _on_its_side(word)]

        flat = [word for words in anchors for word in words]
        owner = [number for number, words in enumerate(anchors) for _word in words]
        missing, extra, pairs = _unmatched(flat, scan_words)
        suspicious = sorted({owner[index] for index in missing})
        # A line not where it was may be somewhere else in the scan: text re-arranged (moved to another panel, a flap).
        # It is one finding, that it moved -- shown where it was and where it is -- and what differs in it, if anything;
        # not a line missing here and whatever now stands here read as the line 'changed' into it.
        moved, moved_rows = _moved_lines(anchors, suspicious, missing, owner, scan_page, scan_words, pairs)
        suspicious = [number for number in suspicious if number not in moved]
        boxes = [_span_bbox(anchors[number]) for number in suspicious]
        offsets = [_local_offset(pairs, flat, scan_words, box) for box in boxes]
        scan_boxes = [_shifted_box(box, offset) for box, offset in zip(boxes, offsets)]

        candidates: list[DiffCandidate] = []
        for number, row in moved.items():
            for found in _moved_findings(anchors[number], row):
                said = _CHANGE_SAID.match(found.detail)
                accents_only = bool(said) and _unaccented(said.group(1)) == _unaccented(said.group(2))
                if found.subtype is DiffSubtype.CHANGED and found.moved_to is not None and not accents_only:
                    # (judged by the ink where the text went: a '1' read '11' over the same ink is the reading's)
                    shift = (found.moved_to.center[0] - found.bbox.center[0], found.moved_to.center[1] - found.bbox.center[1])
                    found = _marked_if_misread(found, original, suspect, config, shift) or found
                candidates.append(found)
        unreadable = 0
        if boxes:
            digital_readings = [
                _layer_in(box, digital_layer) or reading for box, reading in zip(boxes, read_windows(reading_original, boxes, config))
            ]
            scan_readings = read_windows(reading_suspect, scan_boxes, config)
            digital_second = scan_second = [[] for _ in boxes]
            if alternate is not None:
                second_original, second_suspect = (reading_picture(picture) for picture in alternate)
                digital_second = [
                    _layer_in(box, digital_layer) or reading for box, reading in zip(boxes, read_windows(second_original, boxes, config))
                ]
                scan_second = read_windows(second_suspect, scan_boxes, config)
            pad = config.text_line_pad_px
            for number, box, offset, scan_box, digital_reading, scan_reading, digital_other, scan_other in zip(
                suspicious, boxes, offsets, scan_boxes, digital_readings, scan_readings, digital_second, scan_second
            ):
                # everything in the digital file's frame: the scan's words are moved back by how far the scan is off here
                back = (-offset[0], -offset[1])
                on_page = [
                    word
                    for word in scan_words
                    if scan_box.x0 - pad <= word.bbox.center[0] <= scan_box.x1 + pad
                    and scan_box.y0 - pad <= word.bbox.center[1] <= scan_box.y1 + pad
                ]
                scan_reading = [word for word in scan_reading if not _on_its_side(word)]
                scan_variants = [_moved(scan_reading, back), _moved(on_page, back)]
                if scan_other:
                    scan_variants.append(_moved(scan_other, back))
                scan_variants.append(_union_of_readings(*scan_variants[:2]))
                if scan_other:
                    scan_variants.append(_union_of_readings(scan_variants[-1], scan_variants[2]))
                if not any(scan_variants):
                    if ink_agreement(original, suspect, box, offset) < config.text_present_ink_agreement:
                        anchor = digital_reading or anchors[number]
                        candidates.append(
                            _candidate(
                                box,
                                _span_type([w.text for w in anchor]),
                                DiffSubtype.REMOVED,
                                _mean_confidence(anchor),
                                f"'{_join(anchor)}' missing from suspect",
                            )
                        )
                    else:
                        unreadable += 1  # the ink is there, but too soft to read: say so, do not call it a change
                    continue
                digital_variants = [digital_reading, anchors[number], _union_of_readings(digital_reading, anchors[number])]
                if digital_other:
                    digital_variants += [digital_other, _union_of_readings(digital_variants[2], digital_other)]
                found = _fewest_differences(digital_variants, scan_variants)
                candidates += [
                    marked
                    for marked in (_marked_if_misread(candidate, original, suspect, config, offset) for candidate in found)
                    if marked is not None
                ]

        windows = [_span_bbox(words) for words in anchors]
        taken = {id(word) for row in moved_rows for word in row}  # (a moved line's words are not text added)
        added = [scan_words[i] for i in sorted(extra) if id(scan_words[i]) not in taken]
        digital_words = [word for _line, word in digital_page]
        candidates += _added_text(
            added, pairs, flat, scan_words, lettering_original, lettering_suspect, windows, config, reading_suspect, digital_words
        )
        same_text = [flat[i].bbox.union(scan_words[j].bbox) for i, j in pairs]
        for number, row in moved.items():  # (moved: the same text, in both places)
            same_text += [_span_bbox(anchors[number]), _span_bbox(row)]
        # a line read again and found to say the same is the same text too
        changed_boxes = [candidate.bbox for candidate in candidates]
        same_text += [box for box in boxes if not any(box.intersection_area(other) > 0 for other in changed_boxes)]

    logger.info(
        "text detector finished (scan)",
        extra={
            "lines": len(anchors),
            "reread_lines": len(boxes),
            "candidates": len(candidates),
            "unreadable_lines": unreadable,
        },
    )
    for candidate in candidates:
        candidate.source = SOURCE_OCR  # found by reading the words, not by comparing the pictures
    return TextScanResult(candidates, unreadable, same_text)


_SPLIT_GAP_HEIGHTS = 2.5  # a gap in a line this many word-heights wide (and at least `_RUN_GAP_PX`) is two lines


def _split_at_gaps(words: list[OcrWord]) -> list[list[OcrWord]]:
    """A row of words split where a wide gap runs through it: the reader groups words of two panels (or two columns)
    that happen to share a baseline into one line, and a line compared or boxed as one then spans both."""
    if not words:
        return []
    ordered = sorted(words, key=lambda word: word.bbox.x0)
    height = float(np.median([word.bbox.height for word in ordered]))
    widest = max(float(_RUN_GAP_PX), _SPLIT_GAP_HEIGHTS * height)
    parts = [[ordered[0]]]
    for word in ordered[1:]:
        if word.bbox.x0 - parts[-1][-1].bbox.x1 > widest:
            parts.append([word])
        else:
            parts[-1].append(word)
    return parts


def _on_its_side(word: OcrWord) -> bool:
    """A word of two letters or more that is taller than it is wide: text turned on its side, read upright (junk)."""
    return len(canonical(word.text)) >= 2 and word.bbox.height > 1.5 * word.bbox.width and word.bbox.height >= 40


def _rows(words: list[OcrWord]) -> list[list[OcrWord]]:
    """Words split into the rows they lie on: a word joins a row whose middle is within half a word-height of its own."""
    rows: list[list[OcrWord]] = []
    for word in sorted(words, key=lambda w: (w.bbox.y0 + w.bbox.y1) / 2):
        middle = (word.bbox.y0 + word.bbox.y1) / 2
        if rows:
            last = rows[-1]
            row_middle = float(np.median([(w.bbox.y0 + w.bbox.y1) / 2 for w in last]))
            height = float(np.median([w.bbox.height for w in last + [word]]))
            if abs(middle - row_middle) <= 0.5 * height:
                last.append(word)
                continue
        rows.append([word])
    return rows


_MOVED_MIN_WORDS = 3  # a line of this many words at least is looked for elsewhere (or two, this many letters long):
_MOVED_MIN_LETTERS_OF_TWO = 12  # a word or two ('AMPHOLIP', 'Amphotericin B') is printed all over a carton: found anywhere
_MOVED_UNMATCHED_SHARE = 0.5  # ...if this much of it is not where it was
_MOVED_COVER = 0.75  # ...and found where this share of its words, in order, are one line of the scan
_MOVED_MIN_HEIGHTS = 3.0  # ...no nearer than this many of its word-heights (nearer, it is compared where it is)
_MOVED_ROW_USED_SHARE = 0.5  # a scan line with this much of it already matched to text where it is, is that text


def _word_cover(words: list[str], row: list[str]) -> float:
    """The share of `words` that are words of `row`, in the same order (whole words: letters scattered through other
    words are not the line)."""
    matcher = SequenceMatcher(None, words, row, autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks()) / max(1, len(words))


def _moved_lines(
    anchors: list[list[OcrWord]],
    suspicious: list[int],
    missing: set[int],
    owner: list[int],
    scan_page: list[tuple[tuple, OcrWord]],
    scan_words: list[OcrWord],
    pairs: list[tuple[int, int]],
) -> tuple[dict[int, list[OcrWord]], list[list[OcrWord]]]:
    """({digital line number: the scan's words it is at, elsewhere}, every scan line so taken) -- lines of the digital
    file not where they were that the scan has elsewhere: the same words, in order, in one line of the scan.

    Only where that cannot be mistaken: a line long enough to be itself (not a brand name printed on every panel), printed
    once in the digital file (an address on two panels could be either), and never to a place where the digital file has
    that text itself (it is that line there, re-flowed). Each scan line is taken by one digital line at most, the closest
    match first."""
    index = {id(word): number for number, word in enumerate(scan_words)}
    used = {j for _i, j in pairs}
    lines: dict[tuple, list[OcrWord]] = {}
    for line, word in scan_page:
        if id(word) in index:
            lines.setdefault(line, []).append(word)
    rows = [part for words in lines.values() for row in _rows(words) for part in _split_at_gaps(row)]
    free = [row for row in rows if sum(index[id(word)] in used for word in row) <= _MOVED_ROW_USED_SHARE * len(row)]
    row_words = [[canonical(word.text) for word in row] for row in free]
    row_boxes = [_span_bbox(row) for row in free]
    keys = [[canonical(word.text) for word in words] for words in anchors]
    printed = {}
    for key in keys:
        printed[" ".join(key)] = printed.get(" ".join(key), 0) + 1
    unmatched: dict[int, int] = {}
    for word_index in missing:
        unmatched[owner[word_index]] = unmatched.get(owner[word_index], 0) + 1

    offers, home = [], set()
    for number, words in enumerate(anchors):
        key = keys[number]
        long_enough = len(key) >= _MOVED_MIN_WORDS or (len(key) >= 2 and sum(map(len, key)) >= _MOVED_MIN_LETTERS_OF_TWO)
        if not long_enough:
            continue
        box = _span_bbox(words)
        height = float(np.median([word.bbox.height for word in words]))
        looking = (
            number in suspicious
            and unmatched.get(number, 0) >= _MOVED_UNMATCHED_SHARE * len(words)
            and printed[" ".join(key)] == 1
        )
        for row_number, (row_key, there) in enumerate(zip(row_words, row_boxes)):
            distance = math.hypot(there.center[0] - box.center[0], there.center[1] - box.center[1])
            near = distance < _MOVED_MIN_HEIGHTS * height
            if not (looking or near) or _word_cover(key, row_key) < _MOVED_COVER:
                continue
            if near:
                home.add(row_number)  # the digital file's own line, here: no other line moved to it
            elif looking:
                offers.append((distance, number, row_number))
    moved: dict[int, list[OcrWord]] = {}
    taken: set[int] = set()
    for _distance, number, row_number in sorted(offers):
        if number in moved or row_number in taken or row_number in home:
            continue
        moved[number] = _aligned_part(anchors[number], free[row_number])
        taken.add(row_number)
    return moved, [free[row_number] for row_number in taken]


def _aligned_part(words: list[OcrWord], row: list[OcrWord]) -> list[OcrWord]:
    """The words of a scan line that are the digital line (from the first word matched to the last): a line of the scan
    may run on into other text."""
    matcher = SequenceMatcher(None, [canonical(w.text) for w in words], [canonical(w.text) for w in row], autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size]
    if not blocks:
        return row
    return row[blocks[0].b : blocks[-1].b + blocks[-1].size]


_MOVED_MISREAD_RATIO = 0.75  # a word of a moved line read this alike in the scan is its misreading, not a change


def _moved_findings(words: list[OcrWord], row: list[OcrWord]) -> list[DiffCandidate]:
    """A line that moved: one finding, boxed where it was (the original) and where it is (the scan) -- that it moved, or,
    where something in it changed too, that change (a changed number in a moved line is still a change), one per change.
    A word read nearly the same in the scan is its misreading there, not a change."""
    box, there = _span_bbox(words), _span_bbox(row)
    how = movement.how_far(there.center[0] - box.center[0], there.center[1] - box.center[1])
    changes = []
    matcher = SequenceMatcher(None, [canonical(w.text) for w in words], [canonical(w.text) for w in row], autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        gone, new = words[i1:i2], row[j1:j2]
        digits_differ = [ch for ch in _join(gone) if ch.isdigit()] != [ch for ch in _join(new) if ch.isdigit()]  # (never 'misread')
        alike = bool(gone and new) and not digits_differ and (
            SequenceMatcher(None, canonical(_join(gone)), canonical(_join(new)), autojunk=False).ratio() >= _MOVED_MISREAD_RATIO
        )
        if alike:
            continue
        if gone and new:
            subtype, detail = DiffSubtype.CHANGED, f"'{_join(gone)}' -> '{_join(new)}' (in text moved {how})"
        elif gone:
            subtype, detail = DiffSubtype.REMOVED, f"'{_join(gone)}' missing from suspect (in text moved {how})"
        else:
            subtype, detail = DiffSubtype.ADDED, f"'{_join(new)}' added in suspect (in text moved {how})"
        changes.append(
            DiffCandidate(
                bbox=_span_bbox(gone) if gone else box,
                type=_span_type([word.text for word in gone + new]),
                subtype=subtype,
                confidence=_mean_confidence(gone + new),
                detail=detail,
                moved_to=_span_bbox(new) if new else there,
            )
        )
    if changes:
        return changes
    return [
        DiffCandidate(
            bbox=box,
            type=_span_type([word.text for word in words]),
            subtype=DiffSubtype.MOVED,
            confidence=_mean_confidence(words),
            detail=f"Text moved {how}: '{_join(words)}'",
            moved_to=there,
        )
    ]


def _added_text(
    extra: list[OcrWord],
    pairs: list[tuple[int, int]],
    anchor: list[OcrWord],
    scan: list[OcrWord],
    original: np.ndarray,
    suspect: np.ndarray,
    windows: list[BoundingBox],
    config: DetectionConfig,
    scan_picture: np.ndarray | None = None,
    digital_words: list[OcrWord] = (),
) -> list[DiffCandidate]:
    """Words the scan has, read surely, outside the digital file's lines (`windows`) -- or carrying one on, past its end
    or before its start ('STORE BELOW' and the scan's '30°C.' after it) -- where the digital file has no ink, or not
    this word's (`_marks_near`): a word added at the end of a line, the line above and below it a few pixels off, has
    their ink within reach of it, but not its own marks. Not a word the digital file has itself a few
    lines off (`digital_words`, every word read of it): text moved within its panel is compared as lines. A longer
    word the whole-page reading was less sure of is read again in its own window of `scan_picture` (what the scan is
    read from), and counts if read the same there, surely: a heading reversed out of a bar was read 'CATEGORY' at 67%
    on the page, the same at 95% on its own."""
    pad = config.text_line_pad_px
    found, unsure = [], []
    for word in extra:
        sure = word.confidence >= config.text_added_confidence
        if len(canonical(word.text)) < (3 if sure else _UNSURE_WORD_CHARS) or word.confidence < (
            config.text_added_confidence if sure or scan_picture is None else config.ocr_min_confidence_scan
        ):
            continue
        offset = _local_offset(pairs, anchor, scan, word.bbox)
        home = _shifted_box(word.bbox, (-offset[0], -offset[1]))  # where the word is in the digital file's frame
        cx, cy = home.center
        reach = _SPLIT_GAP_HEIGHTS * home.height  # (a word this close after a line's end is of that line)
        if any(w.x0 - pad <= cx <= w.x1 + pad and w.y0 - pad <= cy <= w.y1 + pad for w in windows) and not any(
            w.y0 <= cy <= w.y1 and (0 <= home.x0 - w.x1 <= reach or 0 <= w.x0 - home.x1 <= reach) for w in windows
        ):
            continue  # in a line of the digital file's (compared as lines) -- not one carried on past a line's end
        if any(
            _unaccented(canonical(other.text)) == _unaccented(canonical(word.text))
            and abs(other.bbox.center[0] - cx) <= _ADDED_SAME_WORD_HEIGHTS * home.height
            and abs(other.bbox.center[1] - cy) <= _ADDED_SAME_WORD_HEIGHTS * home.height
            for other in digital_words
        ):
            continue  # the digital file has the word a few lines off: text moved within its panel, compared as lines
        if (
            _ink_in(original, home, config.text_ink_reach_px) < config.text_added_ink_share * _ink_in(suspect, word.bbox, 3)
            or _marks_near(original, suspect, home, word.bbox, config.text_ink_reach_px) < config.text_present_ink_agreement
        ):
            (found if sure else unsure).append((word, home, word.confidence))
    if unsure:
        for (word, home, _confidence), reading in zip(unsure, read_windows(scan_picture, [word.bbox for word, *_ in unsure], config)):
            again = [w.confidence for w in reading if canonical(w.text) == canonical(word.text)]
            if again and max(again) >= config.text_added_confidence:
                found.append((word, home, max(again)))
    candidates = [
        _candidate(home, _span_type([word.text]), DiffSubtype.ADDED, confidence / 100, f"'{word.text}' added in suspect")
        for word, home, confidence in found
    ]
    return [candidate for candidate in candidates if candidate is not None]


def _marks_near(original: np.ndarray, suspect: np.ndarray, home: BoundingBox, box: BoundingBox, reach: int) -> float:
    """How well the digital file has the marks the scan has in `box` -- the scan's own ink, not just any ink -- where it
    has them best within `reach` of `home` (where the scan's `box` is in the digital file's frame): `ink_agreement` there.
    (Within reach, not at `home` only: the scan is never registered exactly, and a word of a block a few pixels further
    off than the words round it is still the digital file's word.)"""
    x0, y0, x1, y1 = (int(round(v)) for v in (box.x0, box.y0, box.x1, box.y1))
    word = 255 - suspect[max(0, y0) : y1, max(0, x0) : x1].astype(np.float32)
    rx0, ry0 = max(0, int(home.x0) - reach), max(0, int(home.y0) - reach)
    rx1, ry1 = min(original.shape[1], int(np.ceil(home.x1)) + reach), min(original.shape[0], int(np.ceil(home.y1)) + reach)
    around = 255 - original[ry0:ry1, rx0:rx1].astype(np.float32)
    if word.size == 0 or word.shape[0] > around.shape[0] or word.shape[1] > around.shape[1] or not word.any() or not around.any():
        return 0.0
    fit = cv2.matchTemplate(around, word, cv2.TM_CCORR_NORMED)
    _low, _high, _at_low, (fx, fy) = cv2.minMaxLoc(fit)
    best = BoundingBox(rx0 + fx, ry0 + fy, rx0 + fx + word.shape[1], ry0 + fy + word.shape[0])
    return ink_agreement(original, suspect, best, (max(0, x0) - best.x0, max(0, y0) - best.y0))


def _shifted_box(box: BoundingBox, offset: tuple[float, float]) -> BoundingBox:
    return BoundingBox(box.x0 + offset[0], box.y0 + offset[1], box.x1 + offset[0], box.y1 + offset[1])


_MAX_STROKE_MATCH_PX = 3


def _stroke_width(ink: np.ndarray) -> float:
    """The average width of the strokes of an ink mask: twice its area over its edge."""
    edge = int((ink & ~cv2.erode(ink.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)).sum())
    return 2.0 * float(ink.sum()) / max(edge, 1)


def _equal_strokes(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two ink masks with strokes of the same width: the thinner is thickened to the other's.

    A scan's dot gain makes a hairline of lettering several times as thick, and the same word then has ink far from
    the digital file's centre-lines although it is the same word. (Whether the marks coincide is what is asked.)
    """
    width_a, width_b = _stroke_width(a), _stroke_width(b)
    if abs(width_a - width_b) < 1.0:
        return a, b
    thinner_is_a = width_a < width_b
    radius = min(_MAX_STROKE_MATCH_PX, int(round(abs(width_a - width_b) / 2)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    if thinner_is_a:
        return cv2.dilate(a.astype(np.uint8), kernel).astype(bool), b
    return a, cv2.dilate(b.astype(np.uint8), kernel).astype(bool)


def ink_agreement(
    original: np.ndarray,
    suspect: np.ndarray,
    box: BoundingBox,
    offset: tuple[float, float] = (0.0, 0.0),
    slack_px: int = _AGREEMENT_SLACK_PX,
) -> float:
    """How closely the two files' ink in `box` coincides: the smaller of the share of each one's ink lying within a
    couple of pixels of the other's. 1.0 is the same marks (a scan's fatter strokes still give ~1.0); 0.0 if either
    has none there. `offset` is where the scan sits relative to the digital file here (a scan is never registered
    exactly): its window is `box` moved by that much."""
    x0, y0, x1, y1 = _padded(box, _AGREEMENT_PAD_PX, original.shape)
    sx0, sy0 = int(round(x0 + offset[0])), int(round(y0 + offset[1]))
    sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)  # (rounded together: the two windows must be the same size)
    a = original[y0:y1, x0:x1] < _DARK
    height, width = suspect.shape[:2]
    window = np.full(a.shape, 255, np.uint8)  # (what lies beyond the scan's edge is blank)
    cx0, cy0, cx1, cy1 = max(sx0, 0), max(sy0, 0), min(sx1, width), min(sy1, height)
    if cx1 > cx0 and cy1 > cy0:
        window[cy0 - sy0 : cy1 - sy0, cx0 - sx0 : cx1 - sx0] = suspect[cy0:cy1, cx0:cx1]
    b = window < _DARK
    if not a.any() or not b.any():
        return 0.0
    a, b = _equal_strokes(a, b)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * slack_px + 1, 2 * slack_px + 1))
    a_in_b = (a & cv2.dilate(b.astype(np.uint8), kernel).astype(bool)).sum() / a.sum()
    b_in_a = (b & cv2.dilate(a.astype(np.uint8), kernel).astype(bool)).sum() / b.sum()
    return float(min(a_in_b, b_in_a))


_CORRELATION_BLUR_PX = 0.8  # both pictures are softened a little before they are correlated: a scan is softer than the file
_CORRELATION_PAD_PX = 2
_CORRELATION_MIN_WINDOW_PX = 10
_CORRELATION_BLANK = 4.0  # a window whose ink strength varies by less than this (standard deviation) is blank
_CORRELATION_LOCAL_SLACK_PX = 6  # a window may re-settle this far from the box's own shift (a scan warps a little along a line)


def _ncc(a: np.ndarray, b: np.ndarray) -> float | None:
    """Normalised correlation of two equal crops; None where both are blank, 0 where only one has ink."""
    a_ink, b_ink = float(a.std()) >= _CORRELATION_BLANK, float(b.std()) >= _CORRELATION_BLANK
    if not a_ink and not b_ink:
        return None
    if a_ink != b_ink:
        return 0.0
    return float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0, 0])


def ink_correlation(
    original: np.ndarray,
    suspect: np.ndarray,
    box: BoundingBox,
    search_px: int = 10,
    window_share: float = 0.8,
    offset: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    """(as a whole, in its worst stretch) how closely the ink in `box` of the original is the ink of the suspect, wherever
    the suspect has it within `search_px` of where `offset` says the scan sits: the normalised correlation of the two ink pictures, whole and along windows about
    a letter wide. A word the reading got wrong over ink that is the same correlates well everywhere; a letter that was
    changed leaves one stretch that does not, however well the rest of the word does. (0, 0) where there is nothing to compare."""
    x0, y0, x1, y1 = _padded(box, _CORRELATION_PAD_PX, original.shape)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return 0.0, 0.0
    a = cv2.GaussianBlur(255.0 - original[y0:y1, x0:x1].astype(np.float32), (0, 0), _CORRELATION_BLUR_PX)
    sx0, sy0, sx1, sy1 = _padded(_shifted_box(BoundingBox(x0, y0, x1, y1), offset), search_px, suspect.shape)
    if sx1 <= sx0 or sy1 <= sy0:
        return 0.0, 0.0
    b = cv2.GaussianBlur(255.0 - suspect[sy0:sy1, sx0:sx1].astype(np.float32), (0, 0), _CORRELATION_BLUR_PX)
    height, width = a.shape
    if b.shape[0] < height or b.shape[1] < width or float(a.std()) < _CORRELATION_BLANK:
        return 0.0, 0.0
    fit = cv2.matchTemplate(b, a, cv2.TM_CCOEFF_NORMED)
    _low, best, _low_at, (left, top) = cv2.minMaxLoc(fit)
    # A long line of a scan can warp a little along its length (it is never quite flat under the platen), so the one
    # shift that fits the box as a whole can be a few pixels out partway along it. Each window is let re-settle on its
    # own, within a few pixels of that shift, rather than being judged at the whole box's shift alone.
    window = max(_CORRELATION_MIN_WINDOW_PX, int(window_share * height))
    local_radius = min(_CORRELATION_LOCAL_SLACK_PX, max(0, (b.shape[1] - width) // 2), max(0, (b.shape[0] - height) // 2))
    scores = []
    for start in range(0, max(1, width - window + 1), max(1, window // 2)):
        piece = a[:, start : start + window]
        aligned_at_start = b[top : top + height, left + start : left + start + window]
        if local_radius <= 0 or float(piece.std()) < _CORRELATION_BLANK:
            score = _ncc(piece, aligned_at_start)
        else:
            wleft, wtop = max(0, left + start - local_radius), max(0, top - local_radius)
            wright = min(b.shape[1], left + start + window + local_radius)
            wbottom = min(b.shape[0], top + height + local_radius)
            local = b[wtop:wbottom, wleft:wright]
            score = float(cv2.matchTemplate(local, piece, cv2.TM_CCOEFF_NORMED).max()) if local.shape[0] >= piece.shape[0] and local.shape[1] >= piece.shape[1] else _ncc(piece, aligned_at_start)
        if score is not None:
            scores.append(score)
    return float(best), (min(scores) if scores else float(best))


def _ranked_down(candidate: DiffCandidate, why: str) -> DiffCandidate:
    return DiffCandidate(
        bbox=candidate.bbox,
        type=candidate.type,
        subtype=candidate.subtype,
        confidence=min(candidate.confidence, LIKELY_MISREAD_CONFIDENCE),
        detail=f"{candidate.detail} ({why})",
        magnitude=candidate.magnitude,
        uncertain=True,
    )


def _same_ink(
    candidate: DiffCandidate, original: np.ndarray, suspect: np.ndarray, config: DetectionConfig, offset: tuple[float, float]
) -> bool:
    """Whether the two files' ink pictures are the same picture in the difference's box (see `text_same_ink_worst`).

    `worst` (every window re-settled on its own, so a line's warp along a scan does not count against it) is what
    decides it; `whole` (one shift for the box) only rules out an accident of many small windows each finding
    something to latch onto in an otherwise unrelated picture.
    """
    whole, worst = ink_correlation(original, suspect, candidate.bbox, offset=offset)
    return whole >= config.text_same_ink_loose_correlation and worst >= config.text_same_ink_worst


def _marked_if_misread(
    candidate: DiffCandidate,
    original: np.ndarray,
    suspect: np.ndarray,
    config: DetectionConfig,
    offset: tuple[float, float] = (0.0, 0.0),
) -> DiffCandidate | None:
    """A difference that is probably the reading's and not the artwork's, ranked down (still listed, hidden by
    default): the two files' ink pictures are the same picture in the difference's box (`_same_ink`).

    * A word the scan reads differently although its ink is the digital word's: a misread of the scan.
    * A word said to be new, or missing, where the other file has the ink all the same: read in one file and not the
      other, not added or removed. (The ink is looked for where the scan sits, a scan never being registered exactly.)
    """
    if candidate.subtype is DiffSubtype.ADDED:
        here = _shifted_box(candidate.bbox, offset)
        if _ink_in(original, candidate.bbox, config.text_ink_reach_px) >= config.text_added_ink_share * _ink_in(suspect, here, 3):
            return _ranked_down(candidate, "the digital file has ink here that was not read: likely a misread")
        return candidate
    if candidate.subtype is DiffSubtype.REMOVED:
        here = _shifted_box(candidate.bbox, offset)
        if _ink_in(suspect, here, config.text_ink_reach_px) >= config.text_added_ink_share * _ink_in(original, candidate.bbox, 3):
            return _ranked_down(candidate, "the scan has ink here that was not read: likely a misread")
        return candidate
    said = _CHANGE_SAID.match(candidate.detail)
    if said and said.group(1) != said.group(2) and _unaccented(said.group(1)) == _unaccented(said.group(2)) and _accents(
        said.group(2)
    ) < _accents(said.group(1)):
        # A scan's accents are specks a reader DROPS where it cannot see them ('administración' read 'administracion'):
        # the same letters with fewer accents are the same word, read less surely. (Accents the scan has and the digital
        # file has not are another matter -- a Vietnamese 'tinh' corrected to 'tĩnh' -- and stay a change.)
        return _ranked_down(candidate, "only its accents read differently: likely a misread of the scan")
    if not _same_ink(candidate, original, suspect, config, offset):
        return candidate
    return _ranked_down(candidate, "the ink matches: likely a misread of the scan")


_CHANGE_SAID = re.compile(r"'(.*?)' -> '(.*?)'")


def _accents(text: str) -> int:
    return sum(1 for c in unicodedata.normalize("NFKD", text) if unicodedata.combining(c))


def _unaccented(text: str) -> str:
    """'Médica' -> 'medica': the letters and digits alone, accents off, in one case."""
    stripped = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return "".join(c for c in stripped.casefold() if c.isalnum())


# ---------------------------------------------------------------------------
# Two digital files: a reading that the marks do not support
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Describing what the pixels found
# ---------------------------------------------------------------------------
#
# In Automatic mode a difference is FOUND in the pixels, never by reading: two files can be read differently
# although their pixels are the same (one draws its text as live type, the other as a picture of it; reading a
# page groups its lines differently when anything else on it differs), and a comparison of two readings then
# reports whole areas that did not change. So reading only DESCRIBES a difference the pixels already found: the
# words in the finding's own box are read in each file, alone, and if they differ the finding is a text
# change with the old and the new words.

# Windows a finding is read in, (across, up-and-down) padding in pixels, tried in turn. No one window reads everything: a
# small one reads the finding's own words most surely (over a gradient a wider one lost a '100 g'), and a wide, short one
# takes in a whole line (a long word cut by the small one, the words a dropped word let move along).
_DESCRIBE_WINDOWS = ((48, 48), (128, 24))
_DESCRIBE_UPSCALES = ((24, 4.0), (60, 3.0))  # (finding no bigger than this, enlarge by this): small type is read far more surely
_DESCRIBE_UPSCALE = 2.0  # enlarged (a real pair: '100g' beside '10' 'g' became '100' 'g'; a 5 px serial number needs 4x)
_DESCRIBE_ROTATE_MAX_PX = 100  # a finding this small, read as nothing upright, is tried as text on its side
_DESCRIBE_SIDEWAYS_WINDOW = (24, 128)  # (across, along) padding for text on its side: a serial number runs a long way
_DESCRIBE_WAYS = tuple((pad_x, pad_y, None, False) for pad_x, pad_y in _DESCRIBE_WINDOWS) + tuple(
    (*_DESCRIBE_SIDEWAYS_WINDOW, rotation, True) for rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)
)  # (x padding, y padding, turned, sideways): the ways a box is looked at, in turn
_DESCRIBE_MAX_AREA_PX = 3_500_000  # a finding this big is a panel, not a phrase: not read
_ON_LINE_OVERLAP = 0.3  # a word is on a finding's line if it overlaps it by this share of the smaller height
_DESCRIBE_REACH_PX = 24  # the phrase a finding belongs to reaches this far along its line (a word's height and a half, if more)
_BLOCK_PAGE_SEGMENTATION = "--psm 6"  # a window holds a block of text
_MAX_DETAIL_CHARS = 70
_SMALL_FINDING_PX = 24  # a finding no bigger than this either way is a speck (or a glyph), not a phrase
_MAX_READERS = 16  # readers started side by side (each is a program of its own: starting it is what costs)


def _clip(text: str) -> str:
    return text if len(text) <= _MAX_DETAIL_CHARS else text[: _MAX_DETAIL_CHARS - 1].rstrip() + "…"


def _readable(crop: np.ndarray) -> np.ndarray:
    """A crop as a grey picture of dark text on light, whichever way round it is drawn: where the marks are lighter
    than the crop's usual tone (white lettering on a red box), it is reversed."""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    median = float(np.median(gray))
    low, high = np.percentile(gray, (2, 98))
    if high - low >= 30:
        # Part of the crop only may be a dark block with its lettering light (a line of white type on a panel that ends
        # partway along it, the paper beside it in the crop): that block alone is reversed, the rest left as it is.
        block = _dark_block(gray, (low + high) / 2)
        if block is not None:
            readable = gray.copy()
            readable[block] = 255 - gray[block]
            return readable
        marks = np.abs(gray.astype(np.int16) - int(median)) > (high - low) / 3
        if marks.any() and float(gray[marks].mean()) > median:
            return 255 - gray
    return gray


_BLOCK_SOLID_SHARE = 0.6  # a dark region at least this dark throughout (lettering apart) is a block, not a line of dark type
_BLOCK_MIN_SHARE = 0.15  # ...covering at least this share of the crop, and leaving at least as much of it paper
_BLOCK_MIN_LETTERING = 0.02  # ...with at least this share of it light: lettering reversed out of it


def _dark_block(gray: np.ndarray, middle: float) -> np.ndarray | None:
    """Where a crop has a solid dark block with light lettering in it (a mask), or None. A block is mostly dark all
    through -- a line of dark type is not: filled in, it is mostly the paper between its letters."""
    dark = (gray < middle).astype(np.uint8)
    size = max(3, (min(gray.shape[:2]) // 3) | 1)  # (a letter's hole closes; the gap past a block's end does not)
    block = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))) > 0
    share = float(block.mean())
    if not _BLOCK_MIN_SHARE <= share < 1.0 - _BLOCK_MIN_SHARE:
        return None  # (all of it, or next to none: the crop as a whole is the case above)
    solid = float(dark[block].mean())
    if solid < _BLOCK_SOLID_SHARE or 1.0 - solid < _BLOCK_MIN_LETTERING:
        return None
    return block


def _enlargement(box: BoundingBox) -> float:
    """How much to enlarge a window before it is read: type that small is read only when enlarged a good deal."""
    size = max(box.width, box.height)
    for limit, factor in _DESCRIBE_UPSCALES:
        if size <= limit:
            return factor
    return _DESCRIBE_UPSCALE


def _window_picture(image: np.ndarray, window: "_Window") -> np.ndarray:
    """The window as it is given to the reader: dark on light, enlarged, and turned as the window says."""
    crop = image[window.y0 : window.y1, window.x0 : window.x1]
    factor = window.factor
    enlarged = cv2.resize(_readable(crop), None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    return enlarged if window.rotation is None else cv2.rotate(enlarged, window.rotation)


def _window_words(data: dict, window: "_Window", picture: np.ndarray, shape: tuple[int, int], config: DetectionConfig) -> list[OcrWord]:
    """The words a reading of a window's `picture` found, where they lie in the whole picture (of `shape`)."""
    height, width = shape
    x0, y0, x1, y1, factor, rotation = window.x0, window.y0, window.x1, window.y1, window.factor, window.rotation
    crop_width = picture.shape[1]
    enlarged_height, enlarged_width = picture.shape[:2] if rotation is None else picture.shape[1::-1]
    words = []
    for _line, word in _words_of(data, config.ocr_min_confidence):
        b = word.bbox
        if rotation is None:
            ex0, ey0, ex1, ey1 = b.x0, b.y0, b.x1, b.y1
            cut = (b.x0 <= 2 and x0 > 0) or (b.x1 >= crop_width - 2 and x1 < width)
        elif rotation == cv2.ROTATE_90_CLOCKWISE:  # the window's (x, y) went to (height - y, x): back again
            ex0, ey0, ex1, ey1 = b.y0, enlarged_height - b.x1, b.y1, enlarged_height - b.x0
            cut = (b.x0 <= 2 and y1 < height) or (b.x1 >= crop_width - 2 and y0 > 0)
        else:  # counter-clockwise: (x, y) went to (y, width - x)
            ex0, ey0, ex1, ey1 = enlarged_width - b.y1, b.x0, enlarged_width - b.y0, b.x1
            cut = (b.x0 <= 2 and y0 > 0) or (b.x1 >= crop_width - 2 and y1 < height)
        # A word touching the window's end is cut short by it ('Take' read as 'ce'), unless that edge is the picture's own.
        if cut:
            continue
        words.append(
            OcrWord(word.text, BoundingBox(ex0 / factor + x0, ey0 / factor + y0, ex1 / factor + x0, ey1 / factor + y0), word.confidence)
        )
    return _trustworthy(words, config)


def _supported(
    box: BoundingBox,
    gone: list[OcrWord],
    new: list[OcrWord],
    other_read_something: bool,
    original: np.ndarray,
    suspect: np.ndarray,
    config: DetectionConfig,
) -> bool:
    """Whether the pixels back a claim that these words changed, are missing or are new.

    Two readings of one picture differ for reasons that are not in the picture (white lettering on red read in one
    file and not in the other), so a claim stands only if the words were read surely AND the marks where they sit
    are not the same in both files. A word 'changed' over the same marks is the reader's, not the artwork's; and a
    word 'missing' or 'new' when the OTHER file was read as nothing at all is unread there, not gone, unless its
    marks are mostly gone too. (Where the other file was read as something, the reading is to be trusted: the
    words that followed have moved along into the place.) A word the finding itself touches needs no such check
    for a change: the pixels there are what differ.
    """
    words = gone + new
    if any(word.confidence < config.text_describe_min_confidence for word in words):
        return False
    touching = any(word.bbox.intersection_area(box) > 0 for word in words)
    if box.width <= _SMALL_FINDING_PX and box.height <= _SMALL_FINDING_PX:
        # A speck is not the words beside it: the enlarged reading of a corner of a line 'read' '38' beside it. A claim on
        # so small a finding has to be about a word the finding lies on, and a real one (three letters, or a number).
        if not touching or any(len(canonical(word.text)) < 3 and not canonical(word.text).isdigit() for word in words):
            return False
    if gone and new and touching:
        return True
    agreement = marks_agreement(original, suspect, _span_bbox(words), 2, config.residue_slack_px)
    if agreement is None:
        return True  # one file has nothing there: the words really are gone, or new
    if gone and new:
        return agreement < config.residue_min_agreement
    return other_read_something or agreement < config.text_describe_present_agreement


_NOT_PAIRED = object()  # (`_paired_lines`: the finding is one line, in place -- read it as such)
_LINE_SAME_HEIGHT = 0.5  # words overlapping this share of the smaller height are on one line
_FINDING_ON_LINE = 0.3  # a line is the finding's where the finding covers this share of its height
_LINES_ALIKE = 0.6  # two lines whose words are this alike (in order) are the same line, re-set or changed
_LINES_CLEARLY = 0.2  # ...and a line elsewhere is taken over the one in place only if it is this much more alike
_PAIRED_SURE = 60.0  # a word differing between paired lines is said only where it was read at least this surely


def _lines_of(words: list[OcrWord]) -> list[list[OcrWord]]:
    """Words as lines: those overlapping in height, left to right."""
    lines: list[list[OcrWord]] = []
    for word in sorted(words, key=lambda w: (w.bbox.y0 + w.bbox.y1) / 2):
        for line in lines:
            top, bottom = min(w.bbox.y0 for w in line), max(w.bbox.y1 for w in line)
            shared = min(bottom, word.bbox.y1) - max(top, word.bbox.y0)
            if shared >= _LINE_SAME_HEIGHT * min(bottom - top, word.bbox.height):
                line.append(word)
                break
        else:
            lines.append([word])
    return [sorted(line, key=lambda w: w.bbox.x0) for line in lines]


def _line_likeness(a: list[OcrWord], b: list[OcrWord]) -> float:
    return SequenceMatcher(None, [canonical(w.text) for w in a], [canonical(w.text) for w in b], autojunk=False).ratio()


def _paired_lines(
    box: BoundingBox,
    words_original: list[OcrWord],
    words_suspect: list[OcrWord],
    original: np.ndarray,
    suspect: np.ndarray,
    config: DetectionConfig,
):
    """The text change in a finding read line by line, each of its lines set against the suspect's line most like it --
    where reading it in place would not do: a finding several lines tall (a block of type re-set), or a line whose
    counterpart is not the line at the same height (the block it is in moved: '7 x 7 mm' would otherwise be set
    against the line above it). `_NOT_PAIRED` where it is one line, best set against the line in place."""
    lines_o = [line for line in _lines_of(words_original) if _covers(box, line)]
    lines_s = _lines_of(words_suspect)
    covered_s = [index for index, line in enumerate(lines_s) if _covers(box, line)]
    if not lines_o or not lines_s:
        return _NOT_PAIRED
    pairs: list[tuple[list[OcrWord], list[OcrWord] | None]] = []
    used: set[int] = set()
    moved = False
    for line in lines_o:
        likeness = [(_line_likeness(line, other), index) for index, other in enumerate(lines_s) if index not in used]
        best, at = max(likeness, default=(0.0, -1))
        in_place = max((score for score, index in likeness if _covers(_span_bbox(lines_s[index]), line)), default=0.0)
        if best < _LINES_ALIKE:
            pairs.append((line, None))
            continue
        if best >= in_place + _LINES_CLEARLY:
            moved = True
        used.add(at)
        pairs.append((line, lines_s[at]))
    every_o = _lines_of(words_original)
    added_lines = [
        lines_s[index] for index in covered_s
        if index not in used and not any(_line_likeness(lines_s[index], line) >= _LINES_ALIKE for line in every_o)
    ]  # (a line now where the finding is, that is another of the original's lines moved there, is no line added)
    if len(lines_o) < 2 and not moved and not (added_lines and len(covered_s) >= 2):
        return _NOT_PAIRED
    gone: list[OcrWord] = []
    new: list[OcrWord] = []
    spans: list[str] = []
    for line, other in pairs:
        if other is None:
            continue  # (a line with no counterpart is told as the line it is, below, if nothing else is)
        matcher = SequenceMatcher(None, [canonical(w.text) for w in line], [canonical(w.text) for w in other], autojunk=False)
        for i1, i2, j1, j2 in _joined_changes(matcher.get_opcodes()):
            was, now = line[i1:i2], other[j1:j2]
            if all(w.confidence < _PAIRED_SURE for w in was + now):
                continue  # (read too unsurely in both to say)
            read = [w for w in was + now if w.confidence < _TEXT_LAYER_CONFIDENCE]  # (a file's own text is exact: never misread)
            if was and now and read and _ink_alike(was, now, original, suspect, config):
                continue  # the two words' ink is the same picture: read differently, not printed differently
            gone += was
            new += now
            spans.append(f"'{_join(was)}' -> '{_join(now)}'" if was and now else f"'{_join(was)}' missing" if was else f"'{_join(now)}' added")
    # Lines of the finding with nothing like them in the other file: set against each other in order, top to bottom (a
    # line rewritten: 'ALLANA CONSUMER PRODUCTS LTD' -> 'PEPIZO FOODS LLP'); what is left over was added or is missing.
    lone_o = [line for line, other in pairs if other is None and any(w.confidence >= _PAIRED_SURE for w in line)]
    lone_s = [line for line in added_lines if any(w.confidence >= _PAIRED_SURE for w in line)]
    lone_o.sort(key=lambda line: _span_bbox(line).y0)
    lone_s.sort(key=lambda line: _span_bbox(line).y0)
    for index in range(max(len(lone_o), len(lone_s))):
        was = lone_o[index] if index < len(lone_o) else []
        now = lone_s[index] if index < len(lone_s) else []
        gone += was
        new += now
        spans.append(f"'{_join(was)}' -> '{_join(now)}'" if was and now else f"'{_join(was)}' missing" if was else f"'{_join(now)}' added")
    if spans and _glyphs_only(box, gone + new):
        spans = []
    if not spans:
        if not any(other is None for _line, other in pairs) and not added_lines:
            return None  # the same words, line for line: no change in the words (the block moved, or its look changed)
        return _NOT_PAIRED
    subtype = DiffSubtype.CHANGED if gone and new else DiffSubtype.REMOVED if gone else DiffSubtype.ADDED
    if len(spans) == 1 and gone and new:
        detail = f"'{_clip(_join(gone))}' -> '{_clip(_join(new))}'"
    elif len(spans) == 1:
        detail = f"'{_clip(_join(gone or new))}' {'missing from suspect' if gone else 'added in suspect'}"
    else:
        detail = _clip("; ".join(spans))
    return _candidate(box, _span_type([w.text for w in gone + new]), subtype, _mean_confidence(gone + new), detail)


_INK_AROUND_PX = 30  # (the ink of two words is compared in a stretch this much round them)
_JOIN_GAP_WORDS = 1  # changes this few unchanged words apart on one line are one change ('7 x 7' -> '9 x 9')


def _joined_changes(opcodes) -> list[tuple[int, int, int, int]]:
    """A line's changed stretches ((i1, i2, j1, j2) of `SequenceMatcher`), those a word or so apart joined into one."""
    joined: list[list[int]] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        if joined and i1 - joined[-1][1] <= _JOIN_GAP_WORDS and j1 - joined[-1][3] <= _JOIN_GAP_WORDS:
            joined[-1][1], joined[-1][3] = i2, j2
        else:
            joined.append([i1, i2, j1, j2])
    return [tuple(change) for change in joined]


def _ink_alike(was: list[OcrWord], now: list[OcrWord], original: np.ndarray, suspect: np.ndarray, config: DetectionConfig) -> bool:
    """Whether the ink of words read differently is the same picture, where each file has it (`ink_correlation`)."""
    a, b = _span_bbox(was), _span_bbox(now)
    if a is None or b is None or abs(a.width - b.width) > 0.25 * max(a.width, b.width) + 4:
        return False  # (words of other lengths are other words)
    offset = (b.center[0] - a.center[0], b.center[1] - a.center[1])
    # (as ink pictures: grey, of just the stretch round the two words)
    height, width = original.shape[:2]
    around = a.union(b)
    x0, y0 = max(0, int(around.x0) - _INK_AROUND_PX), max(0, int(around.y0) - _INK_AROUND_PX)
    x1, y1 = min(width, math.ceil(around.x1) + _INK_AROUND_PX), min(height, math.ceil(around.y1) + _INK_AROUND_PX)

    def grey(image: np.ndarray) -> np.ndarray:
        crop = image[y0:y1, x0:x1]
        return cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop

    local = BoundingBox(a.x0 - x0, a.y0 - y0, a.x1 - x0, a.y1 - y0)
    whole, worst = ink_correlation(grey(original), grey(suspect), local, offset=offset)
    return whole >= config.text_same_ink_loose_correlation and worst >= config.text_same_ink_worst


def _covers(box: BoundingBox | None, line: list[OcrWord]) -> bool:
    """Whether `box` covers a fair share of the line's height (and lies along it)."""
    if box is None:
        return False
    span = _span_bbox(line)
    shared = min(box.y1, span.y1) - max(box.y0, span.y0)
    along = min(box.x1, span.x1) - max(box.x0, span.x0)
    return shared >= _FINDING_ON_LINE * span.height and along > 0


def _degree_read_as_nought(was: str, now: str) -> bool:
    """Whether two readings differ only by a degree sign read as a nought (or an 'o'): '175°C' and '1750C'."""
    if "°" not in was + now:
        return False
    plain = lambda text: text.replace("°", "0").replace("º", "0").casefold().translate(_DEGREE_LOOKALIKES)  # noqa: E731
    return was != now and plain(was) == plain(now)


_DEGREE_LOOKALIKES = str.maketrans({"o": "0"})


_GLYPH_CHARS = 2  # words this short...
_GLYPH_FINDING = 25.0  # ...said of a finding more than this many times their size are glyphs misread, not its text


def _glyphs_only(box: BoundingBox, said: list[OcrWord]) -> bool:
    """Whether what a text change says is only a character or two, of a finding far bigger than they are: a glyph of a
    picture or of a line read one way in one file and another in the other ('炸', '١'), not the finding's text. Never a
    number ('24' -> '36' months) nor a file's own text (exact): those are what they say."""
    if not said or any(len(canonical(w.text)) > _GLYPH_CHARS for w in said):
        return False
    if any(w.confidence >= _TEXT_LAYER_CONFIDENCE or _is_numeric(w.text) for w in said):
        return False
    return box.area > _GLYPH_FINDING * max(1.0, sum(w.bbox.area for w in said))


def _described(
    box: BoundingBox,
    words_original: list[OcrWord],
    words_suspect: list[OcrWord],
    original: np.ndarray,
    suspect: np.ndarray,
    config: DetectionConfig,
    sideways: bool = False,
) -> DiffCandidate | None:
    """The text change inside `box` that two readings of it show, if they show one and the pixels agree it is one.
    (`sideways`: the text runs up or down the page, so a line is a column and 'along' is y.)

    The words on the finding's line are lined up across the whole window, so a word that was dropped is the one
    that is missing rather than everything after it having 'changed' (the words beside it moved along). Only the
    words that differ AND touch the finding, or lie within a word's height and a half of it, are what it says.
    """
    def across(b: BoundingBox) -> tuple[float, float]:  # the extent of a line's height (its width, if the text is on its side)
        return (b.x0, b.x1) if sideways else (b.y0, b.y1)

    def along(b: BoundingBox) -> tuple[float, float]:  # the extent along the line
        return (b.y0, b.y1) if sideways else (b.x0, b.x1)

    if not sideways:
        paired = _paired_lines(box, words_original, words_suspect, original, suspect, config)
        if paired is not _NOT_PAIRED:
            return paired  # (the finding's lines, each set against the suspect's line that is most like it)
    finding_across, finding_along = across(box), along(box)
    finding_thickness = finding_across[1] - finding_across[0]
    heights = [min(across(word.bbox)[1] - across(word.bbox)[0], 3 * finding_thickness + 20) for word in words_original + words_suspect]
    height = float(np.median(heights)) if heights else finding_thickness
    reach = max(float(_DESCRIBE_REACH_PX), 1.5 * height)
    slack = max(0.75 * finding_thickness, 14.0)

    def on_line(word: OcrWord) -> bool:
        first, last = across(word.bbox)
        if not finding_across[0] - slack <= (first + last) / 2 <= finding_across[1] + slack:
            return False
        # and it overlaps the finding's own line: the next line's words, whose middles a tall finding's slack can reach
        # ('Store below 30' under 'Dosage and Directions for use'), are not on it
        overlap = min(last, finding_across[1]) - max(first, finding_across[0])
        return overlap >= _ON_LINE_OVERLAP * min(last - first, finding_thickness)

    def at_finding(word: OcrWord) -> bool:
        first, last = along(word.bbox)
        return on_line(word) and last >= finding_along[0] - reach and first <= finding_along[1] + reach

    original_line = [word for word in words_original if on_line(word)]
    suspect_line = [word for word in words_suspect if on_line(word)]
    matcher = SequenceMatcher(
        None, [word.text for word in original_line], [word.text for word in suspect_line], autojunk=False
    )
    gone: list[OcrWord] = []
    new: list[OcrWord] = []
    alike: list[tuple[OcrWord, OcrWord]] = []  # the words that read the same, at the finding, each with its counterpart
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            gone += [word for word in original_line[i1:i2] if at_finding(word)]
            new += [word for word in suspect_line[j1:j2] if at_finding(word)]
        else:
            alike += [(a, b) for a, b in zip(original_line[i1:i2], suspect_line[j1:j2]) if at_finding(a) or at_finding(b)]
    other_read_something = bool(suspect_line if gone else original_line)
    if (gone or new) and _degree_read_as_nought(_join(gone), _join(new)):
        gone, new = [], []  # (read differently only where the reader always wavers: '175°C' as '1750C')
    if (gone or new) and _glyphs_only(box, gone + new):
        return None
    if not (gone or new):
        # The words are the same: what differs, if anything, is how they are set.
        return _set_differently(box, alike, original, suspect, config)
    if not _supported(box, gone, new, other_read_something, original, suspect, config):
        return None  # the pixels do not bear out that the words changed

    if gone and new:
        subtype, detail = DiffSubtype.CHANGED, f"'{_clip(_join(gone))}' -> '{_clip(_join(new))}'"
    elif gone:
        subtype, detail = DiffSubtype.REMOVED, f"'{_clip(_join(gone))}' missing from suspect"
    else:
        subtype, detail = DiffSubtype.ADDED, f"'{_clip(_join(new))}' added in suspect"
    return _candidate(box, _span_type([word.text for word in gone + new]), subtype, _mean_confidence(gone + new), detail)


_FONT_MIN_INK_PX = 30  # fewer ink pixels than this in a word's crop say nothing about how it is set
_FONT_COLOUR_DELTA = 9.0  # ink this many CIE76 units apart between the two files is a different colour of type
_FONT_WEIGHT_RATIO = 1.25  # strokes this many times as thick (for the type's size) in one file are a different weight
_FONT_INK_WEIGHT_RATIO = 1.15  # the same words, the same size, with this many times the ink are set bolder
_FONT_SIZE_RATIO = 1.07  # type whose ink is this many times as tall in one file is a different size
_FONT_BIG_SIZE_RATIO = 1.15  # ... and where the typeface changed too (its height differs of itself), this many
_FONT_HUE_ANGLE = 22.0  # thin type whose colour lies this many degrees round from the other file's (as seen from the paper) is another colour
_FONT_HUE_MIN_CONTRAST = 15.0  # ... if it stands out from the paper by at least this much
_FONT_MAX_WORDS = 12
_FONT_MOVED_PX = 6.0  # words this far from where they were have moved: the readings' boxes are good to a few px
_FONT_UNIFORM_SHARE = 0.25  # ... and every word by about the same (a wider typeface moves each word further than the last)
_FONT_TYPEFACE_AGREEMENT = 0.85  # below this the two files' letters do not coincide (each word cut out, made one size, one weight)
_FONT_WORD_MIN_INK_PX = 12


def _ink_of(crop: np.ndarray) -> np.ndarray:
    """A boolean mask of the type in an RGB crop: what departs from the paper by half of the crop's own range."""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    low, high = np.percentile(gray, (2, 98))
    if high - low < 30:
        return np.zeros(gray.shape, bool)
    rim = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    return np.abs(gray.astype(np.int16) - int(np.median(rim))) > (high - low) / 2


def _lab(pixels: np.ndarray) -> np.ndarray:
    """The average colour of RGB pixels (N x 3), as CIELAB."""
    average = np.uint8(np.round(pixels.mean(axis=0))).reshape(1, 1, 3)
    lab = cv2.cvtColor(average, cv2.COLOR_RGB2LAB).reshape(3).astype(np.float32)
    lab[0] *= 100.0 / 255.0
    lab[1:] -= 128.0
    return lab


_CORE_KERNEL = np.ones((3, 3), np.uint8)


def _type_colour(crop: np.ndarray, ink: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """(the type's colour, the paper's, whether the type has a solid core), as CIELAB. Where the strokes are thick enough
    to have a core, the type's colour is that of the core (the edge pixels are part paper: blur or a coarser scan lightens
    them); thinner strokes are all edge, and only their direction from the paper's colour can be trusted."""
    core = cv2.erode(ink.astype(np.uint8), _CORE_KERNEL).astype(bool)
    solid = int(core.sum()) >= _FONT_MIN_INK_PX
    rim = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    return _lab(crop[core] if solid else crop[ink]), _lab(rim), solid


def _font_colour_change(a_crop: np.ndarray, a_ink: np.ndarray, b_crop: np.ndarray, b_ink: np.ndarray) -> float | None:
    """How far apart (CIE76) the colours of the type in two crops are -- None if they are not different. Solid type is
    compared by colour. Thin type only by hue: its strokes are all edge, so blur or a softer render makes black type grey
    and says nothing about a change of colour, but black and red type lie in different directions from the paper."""
    (ink_a, paper_a, solid_a), (ink_b, paper_b, solid_b) = _type_colour(a_crop, a_ink), _type_colour(b_crop, b_ink)
    if solid_a and solid_b:
        delta = float(np.linalg.norm(ink_a - ink_b))
        return delta if delta >= _FONT_COLOUR_DELTA else None
    toward_a, toward_b = ink_a - paper_a, ink_b - paper_b
    size_a, size_b = float(np.linalg.norm(toward_a)), float(np.linalg.norm(toward_b))
    if min(size_a, size_b) < _FONT_HUE_MIN_CONTRAST:
        return None
    angle = math.degrees(math.acos(max(-1.0, min(1.0, float(toward_a @ toward_b) / (size_a * size_b)))))
    if angle < _FONT_HUE_ANGLE:
        return None
    return float(np.linalg.norm(toward_a / size_a - toward_b / size_b) * (size_a + size_b) / 2)


def _ink_extent(ink: np.ndarray) -> int:
    """How tall the ink is: from its top row to its bottom row."""
    rows = np.flatnonzero(ink.any(axis=1))
    return int(rows[-1] - rows[0] + 1) if len(rows) else 0


_LETTERS_UPSCALE = 3  # each word is enlarged this much before its letters are compared: the letters differ by less than a pixel


def _same_letters(original: np.ndarray, suspect: np.ndarray, pairs: list[tuple[OcrWord, OcrWord]]) -> float | None:
    """How closely the letters of the words coincide in the two files, whatever their size and weight: each word is cut
    out of both and enlarged, the suspect's is scaled to the original's box, the thinner strokes are thickened to the
    thicker, and the share of the ink that lies on the other's (within a third of a pixel) is measured. The same typeface
    gives 0.9 or more (a bolder cut, a bigger size); another one, 0.85 or less. (The median over the words, so one
    misread word does not decide it.) None where there is too little ink to say."""
    scores = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for a, b in pairs:
        ax0, ay0, ax1, ay1 = _padded(a.bbox, 1, original.shape)
        bx0, by0, bx1, by1 = _padded(b.bbox, 1, suspect.shape)
        if ax1 - ax0 < 6 or ay1 - ay0 < 6 or bx1 - bx0 < 6 or by1 - by0 < 6:
            continue
        a_big = cv2.resize(original[ay0:ay1, ax0:ax1], None, fx=_LETTERS_UPSCALE, fy=_LETTERS_UPSCALE, interpolation=cv2.INTER_CUBIC)
        b_big = cv2.resize(suspect[by0:by1, bx0:bx1], (a_big.shape[1], a_big.shape[0]), interpolation=cv2.INTER_CUBIC)
        a_ink, b_ink = _ink_of(a_big), _ink_of(b_big)
        if a_ink.sum() < 3 * _FONT_WORD_MIN_INK_PX or b_ink.sum() < 3 * _FONT_WORD_MIN_INK_PX:
            continue
        a_ink, b_ink = _equal_strokes(a_ink, b_ink)
        a_in_b = (a_ink & cv2.dilate(b_ink.astype(np.uint8), kernel).astype(bool)).sum() / a_ink.sum()
        b_in_a = (b_ink & cv2.dilate(a_ink.astype(np.uint8), kernel).astype(bool)).sum() / b_ink.sum()
        scores.append(float(min(a_in_b, b_in_a)))
    return float(np.median(scores)) if scores else None


def _set_differently(
    box: BoundingBox,
    same_words: list[tuple[OcrWord, OcrWord]],
    original: np.ndarray,
    suspect: np.ndarray,
    config: DetectionConfig,
) -> DiffCandidate | None:
    """How the words of a finding differ where they read the SAME in both files: where they are, and their colour,
    typeface, weight and size -- said only from what the pixels of the words themselves show.

    `same_words`: the words of the finding's line that read alike in the two files, each with its counterpart. They MOVED
    if every one is about the same distance from its counterpart (a wider typeface pushes each further than the last).
    The words are cut from both files: ink of another colour; letters that do not coincide once made one size and one
    weight (another typeface); strokes thicker or thinner for the size of the type (another weight); ink taller or
    shorter (another size). Where none of that shows, nothing is said and the finding stays what its pixels say it is.
    """
    pairs = [(a, b) for a, b in same_words if min(a.confidence, b.confidence) >= config.text_describe_min_confidence]
    if not pairs:
        return None
    pairs = pairs[:_FONT_MAX_WORDS]
    text = " ".join(a.text for a, _b in pairs)

    def crop_of(image: np.ndarray, boxes: list[BoundingBox]) -> np.ndarray | None:
        union = boxes[0]
        for other in boxes[1:]:
            union = union.union(other)
        x0, y0, x1, y1 = _padded(union, 2, image.shape)
        return image[y0:y1, x0:x1] if x1 - x0 > 4 and y1 - y0 > 4 else None

    a_crop, b_crop = crop_of(original, [a.bbox for a, _b in pairs]), crop_of(suspect, [b.bbox for _a, b in pairs])
    if a_crop is None or b_crop is None:
        return None

    said: list[str] = []
    subtype, magnitude = DiffSubtype.CHANGED, 0.0
    across = [b.bbox.x0 - a.bbox.x0 for a, b in pairs]
    down = [b.bbox.center[1] - a.bbox.center[1] for a, b in pairs]
    dx, dy = float(np.median(across)), float(np.median(down))
    uniform = (max(across) - min(across) <= max(3.0, _FONT_UNIFORM_SHARE * abs(dx))) and (
        max(down) - min(down) <= max(3.0, _FONT_UNIFORM_SHARE * abs(dy))
    )
    if uniform and math.hypot(dx, dy) >= _FONT_MOVED_PX:
        from app.engine.detectors.movement import how_far  # (imported here: movement needs no reader)

        subtype, magnitude = DiffSubtype.MOVED, math.hypot(dx, dy)
        said.append(f"moved {how_far(dx, dy)}")

    a_ink, b_ink = _ink_of(a_crop), _ink_of(b_crop)
    if a_ink.sum() >= _FONT_MIN_INK_PX and b_ink.sum() >= _FONT_MIN_INK_PX:
        colour = _font_colour_change(a_crop, a_ink, b_crop, b_ink)
        if colour is not None:
            said.append(f"font colour changed (dE {colour:.0f})")
        letters = _same_letters(original, suspect, pairs)
        height_a, height_b = _ink_extent(a_ink), _ink_extent(b_ink)
        taller = height_b / height_a if height_a and height_b else 1.0
        if letters is not None and letters < _FONT_TYPEFACE_AGREEMENT:
            said.append("font changed (a different typeface)")
            size_limit = _FONT_BIG_SIZE_RATIO
        else:
            size_limit = _FONT_SIZE_RATIO
            thick_a, thick_b = _stroke_width(a_ink), _stroke_width(b_ink)
            if min(thick_a, thick_b) > 0 and height_a and height_b:
                ratio = (thick_b / height_b) / (thick_a / height_a)  # (strokes grow with the type: what matters is per size)
                # the same words at the same size carry as much more ink as they are bolder: at small sizes, where a
                # stroke is two or three pixels and its width cannot be told to a fraction, that shows it plainly
                ink_ratio = float(b_ink.sum()) / max(1.0, float(a_ink.sum())) / max(1e-6, taller)
                same_size = 1 / _FONT_SIZE_RATIO < taller < _FONT_SIZE_RATIO
                if ratio >= _FONT_WEIGHT_RATIO or (same_size and ink_ratio >= _FONT_INK_WEIGHT_RATIO and ratio > 1.0):
                    said.append("font weight changed (bolder in suspect)")
                elif ratio <= 1 / _FONT_WEIGHT_RATIO or (same_size and ink_ratio <= 1 / _FONT_INK_WEIGHT_RATIO and ratio < 1.0):
                    said.append("font weight changed (lighter in suspect)")
        if taller >= size_limit:
            said.append(f"font size changed ({taller - 1:.0%} bigger in suspect)")
        elif taller <= 1 / size_limit:
            said.append(f"font size changed ({1 - taller:.0%} smaller in suspect)")
    if not said:
        return None
    detail = "Text " + "; ".join(said) + f": '{_clip(text)}'"
    return _candidate(box, DiffType.TEXT, subtype, 0.8, detail, magnitude=magnitude)


_TEXT_LAYER_CONFIDENCE = 99.0  # a word taken from a PDF's own text is not a guess
_INK_LEVEL = 200  # a window with nothing darker than this (in its darkest channel) has nothing printed to read


_SHOWN_INK = 40.0  # (CIE76-ish, 8-bit Lab) a pixel this far from a word's ground is its ink
_SHOWN_REACH = 0.85  # a word's ink reaches at least this far along its box where all of it is shown


def shown_words(words: list[OcrWord] | None, image: np.ndarray) -> list[OcrWord] | None:
    """A PDF's own words that are printed whole: a word whose ink stops well short of where its box ends is cut off on
    the page (set inside a clipping box: the live text 'Cilnidipine' shows as 'Cilnidip') -- what is printed must be read
    from the picture, not taken from the text, so it is left out here (it is read like any other)."""
    if not words:
        return words
    height, width = image.shape[:2]
    lab = None
    shown = []
    for word in words:
        b = word.bbox
        x0, y0, x1, y1 = max(0, int(b.x0)), max(0, int(b.y0)), min(width, int(math.ceil(b.x1))), min(height, int(math.ceil(b.y1)))
        if x1 - x0 < 6 or y1 - y0 < 4 or len(word.text) < 3:
            shown.append(word)
            continue
        if lab is None:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32) if image.ndim == 3 else None
        crop = lab[y0:y1, x0:x1] if lab is not None else image[y0:y1, x0:x1].astype(np.float32)[..., None]
        rim = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
        ground = np.median(rim, axis=0)
        ink = np.linalg.norm(crop - ground, axis=-1) > _SHOWN_INK
        columns = np.flatnonzero(ink.any(axis=0)) if (word.upright is None or word.upright) else np.flatnonzero(ink.any(axis=1))
        along = (x1 - x0) if (word.upright is None or word.upright) else (y1 - y0)
        if len(columns) == 0:
            continue  # (nothing of it printed at all: hidden text)
        reach = (columns.max() - columns.min() + 1) / max(1, along)
        if reach >= _SHOWN_REACH:
            shown.append(word)
    return shown


def text_layer_words(page, dpi: float, shape: tuple[int, int]) -> list[OcrWord] | None:
    """A PDF page's own words (its live text), in the rendered picture's pixels -- exact, and instant beside reading the
    picture. None if the page has no text of its own (it is outlined to paths, or a picture): it must be read."""
    try:
        textpage = page.get_textpage()
        raw = page.get_text("words", textpage=textpage)
        # which way each line runs: a short word's box ('to', 'c)') is as tall as it is wide either way
        # (each word's line found by where it lies, not by number: the two listings number blocks their own ways)
        line_boxes = [
            (tuple(line["bbox"]), line["dir"])
            for block in page.get_text("dict", textpage=textpage)["blocks"]
            if block.get("type") == 0
            for line in block["lines"]
        ]
    except Exception:  # noqa: BLE001 -- a page whose text cannot be extracted is read instead
        logger.exception("could not read the page's text")
        return None
    if not raw:
        return None
    turned = page.rotation % 180 == 90  # the page is shown turned a quarter: what ran across it runs up it
    to_pixels = dpi / 72.0
    rotation = page.rotation_matrix if page.rotation else None
    height, width = shape
    words = []
    for x0, y0, x1, y1, text, block, line, *_rest in raw:
        text = str(text).strip()
        if not text:
            continue
        rect = fitz_rect(x0, y0, x1, y1)
        if rotation is not None:
            rect = rect * rotation
        box = BoundingBox(
            max(0.0, rect.x0 * to_pixels), max(0.0, rect.y0 * to_pixels),
            min(float(width), rect.x1 * to_pixels), min(float(height), rect.y1 * to_pixels),
        )
        if box.width > 0 and box.height > 0:
            direction = _line_direction(line_boxes, x0, y0, x1, y1)
            across = None if direction is None else abs(direction[0]) >= abs(direction[1])
            upright = None if across is None else across != turned
            dx, dy = direction if direction is not None else (1.0, 0.0)
            flip = -1.0 if page.rotation % 360 == 180 else 1.0
            upside_down = page.rotation % 180 == 0 and flip * dx < -abs(dy)  # (runs right to left as shown)
            words.append(OcrWord(text, box, _TEXT_LAYER_CONFIDENCE, upright, upside_down))
    return words or None


def _line_direction(lines: list, x0: float, y0: float, x1: float, y1: float) -> tuple[float, float] | None:
    """The direction of the line a word (its box, in page points) lies in: the smallest line box holding its middle."""
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    holding = [
        (abs((bx1 - bx0) * (by1 - by0)), direction)
        for (bx0, by0, bx1, by1), direction in lines
        if bx0 - 0.5 <= cx <= bx1 + 0.5 and by0 - 0.5 <= cy <= by1 + 0.5
    ]
    return min(holding, key=lambda held: held[0])[1] if holding else None


def fitz_rect(x0: float, y0: float, x1: float, y1: float):
    import pymupdf as fitz

    return fitz.Rect(x0, y0, x1, y1)


@dataclass
class _Window:
    """One window of one file to find the words in: its corner and far corner in the picture, how much it is enlarged
    before it is read, and which way it is turned (None: upright)."""

    x0: int
    y0: int
    x1: int
    y1: int
    factor: float
    rotation: int | None


def _window(image: np.ndarray, box: BoundingBox, pad_x: int, pad_y: int, rotation: int | None) -> _Window | None:
    height, width = image.shape[:2]
    x0, x1 = max(0, int(box.x0) - pad_x), min(width, math.ceil(box.x1) + pad_x)
    y0, y1 = max(0, int(box.y0) - pad_y), min(height, math.ceil(box.y1) + pad_y)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return _Window(x0, y0, x1, y1, _enlargement(box), rotation)


def _layer_words_in(layer: list[OcrWord], window: _Window, shape: tuple[int, int]) -> list[OcrWord]:
    """The text-layer words that lie in a window, as a reading of that window would give them: whole words only (one
    running off the window's edge would be read cut short, so it is left out, as a reading leaves it out), and the ones
    that run the window's way (up the page for a window turned on its side)."""
    height, width = shape
    words = []
    for word in layer:
        if not any(ch.isalnum() for ch in word.text):
            continue  # a mark (':', a bullet) is not a word -- a reading drops it too (`_trustworthy`)
        b = word.bbox
        cx, cy = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
        if not (window.x0 <= cx <= window.x1 and window.y0 <= cy <= window.y1):
            continue
        upright = word.upright if word.upright is not None else not (len(word.text) > 1 and b.height > 1.5 * b.width)
        if window.rotation is None:
            if not upright:
                continue  # a word up the page is read with the window turned
            cut = (b.x0 < window.x0 and window.x0 > 0) or (b.x1 > window.x1 and window.x1 < width)
        else:
            if upright if word.upright is not None else (len(word.text) > 1 and b.width > 1.5 * b.height):
                continue
            cut = (b.y0 < window.y0 and window.y0 > 0) or (b.y1 > window.y1 and window.y1 < height)
        if not cut:
            words.append(word)
    return words


def _printed(image: np.ndarray, window: _Window) -> bool:
    crop = image[window.y0 : window.y1, window.x0 : window.x1]
    darkest = crop.min(axis=2) if crop.ndim == 3 else crop
    return bool((darkest < _INK_LEVEL).any())


def _read_windows_of(
    jobs: list[tuple[np.ndarray, _Window]], config: DetectionConfig, page_segmentation: str = _BLOCK_PAGE_SEGMENTATION
) -> list[list[OcrWord]]:
    """The words in each (picture, window): every window read on its own (small type enlarged, a window is read far more
    surely alone than stacked with others), all of them shared among a few runs of the reader."""
    pictures = [_window_picture(image, window) for image, window in jobs]
    readings = _ocr_many(pictures, page_segmentation, config.ocr_language)
    return [
        _window_words(data, window, picture, image.shape[:2], config)
        for (image, window), picture, data in zip(jobs, pictures, readings)
    ]


def describe(
    boxes: list[BoundingBox],
    original: np.ndarray,
    suspect: np.ndarray,
    config: DetectionConfig,
    original_layer: list[OcrWord] | None = None,
    suspect_layer: list[OcrWord] | None = None,
) -> list[DiffCandidate | None]:
    """For each box where the pixels differ: the text change the words there show (None if they read the same,
    or there are none).

    Every box is looked at the same ways, in turn -- a close window, a wider one, then (a small box read as nothing)
    the window turned on its side -- and the words in a window come from the file's own text where it has any there
    (`original_layer`, `suspect_layer`: `text_layer_words`, in these pictures' pixels), and are read otherwise. What
    has to be read, for every box and every way it might be looked at, is read at once, before any is judged: a
    reading costs little beside starting the reader, so reading a window that turns out not to be needed is cheaper
    than waiting for the ways one after the other."""
    with StageTimer(logger, "describe_text"):
        pending = [index for index, box in enumerate(boxes) if box.area <= _DESCRIBE_MAX_AREA_PX]
        pictures = (original, suspect)
        layers = (original_layer, suspect_layer)
        windows = {
            (way, index, side): _window(pictures[side], boxes[index], pad_x, pad_y, rotation)
            for way, (pad_x, pad_y, rotation, sideways) in enumerate(_DESCRIBE_WAYS)
            for index in pending
            if not sideways or max(boxes[index].width, boxes[index].height) <= _DESCRIBE_ROTATE_MAX_PX
            for side in (0, 1)
        }

        def given(key, layer_used: bool) -> list[OcrWord] | None:
            """A window's words from its file's own text, or None if they have to be read ([] if it is blank)."""
            window = windows[key]
            if window is None:
                return []
            layer = layers[key[2]] if layer_used else None
            if layer is not None:
                words = _layer_words_in(layer, window, pictures[key[2]].shape[:2])
                if words:
                    return words
            return None if _printed(pictures[key[2]], window) else []

        read: dict = {}

        def read_all(keys) -> None:
            keys = [key for key in dict.fromkeys(keys) if key not in read]
            jobs = [(pictures[key[2]], windows[key]) for key in keys]
            for key, words in zip(keys, _read_windows_of(jobs, config)):
                read[key] = words

        def run(indices: list[int], layer_used: bool) -> dict[int, DiffCandidate | None]:
            wanted = set(indices)
            known = {key: given(key, layer_used) for key in windows if key[1] in wanted}
            to_read = [key for key, words in known.items() if words is None]
            if layer_used:
                # Where the two files' own text differs, what it says is to be confirmed by reading (below): those
                # windows are read now, with the rest, rather than in a second round after.
                disputed = {
                    key[1] for key, words in known.items()
                    if key[2] == 0 and words and [w.text for w in words] != [w.text for w in (known[(key[0], key[1], 1)] or [])]
                }
                to_read += [key for key in windows if key[1] in disputed and given(key, False) is None]
            read_all(to_read)
            results: dict[int, DiffCandidate | None] = {}
            for index in indices:
                results[index] = None
                for way, (_pad_x, _pad_y, _rotation, sideways) in enumerate(_DESCRIBE_WAYS):
                    if (way, index, 0) not in windows:
                        continue
                    found = [known[(way, index, side)] for side in (0, 1)]
                    found = [read[(way, index, side)] if words is None else words for side, words in enumerate(found)]
                    results[index] = _described(boxes[index], *found, original, suspect, config, sideways=sideways)
                    if results[index] is not None:
                        break
            return results

        results: list[DiffCandidate | None] = [None] * len(boxes)
        with_layers = original_layer is not None or suspect_layer is not None
        for index, result in run(pending, with_layers).items():
            results[index] = result
        if with_layers:
            # A file's own text is what it SAYS it prints -- a barcode's digits or a logo's letters can be set in a font
            # whose text maps to other characters than it draws. So where it says the words CHANGED, the pictures are
            # read to say so (what is printed decides); where it says they are the same, that stands.
            changed = [i for i in pending if results[i] is not None and results[i].type in (DiffType.TEXT, DiffType.NUMBER)
                       and results[i].subtype is not DiffSubtype.MOVED and "font" not in results[i].detail.lower()]
            for index, result in run(changed, False).items():
                if result is None and _clean_text(results[index].detail):
                    continue  # the reader could not say either way: what the file's own text says, in plain words, stands
                results[index] = result
    return results


def _clean_text(detail: str) -> bool:
    """Whether a description is plain text -- no character a font maps to nothing printable (a barcode's digits taken
    from its text layer came out as '0�222')."""
    return all(ch.isprintable() and ch != "�" for ch in detail)
