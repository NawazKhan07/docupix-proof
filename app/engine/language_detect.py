"""The languages a file's text is printed in, found from the file itself -- so its words are read in them, in every mode,
without the reviewer having to say.

Two steps. First the WRITING SYSTEMS: a sample of the file's lines is read once with one language of each common
writing system together (Tesseract tries each on every word and keeps the best), and each word kept only where it was
read surely and its letters are all of one writing system -- a Latin line read as Thai comes out as Thai letters read
unsurely, never surely. A writing system is printed on the file where enough of what was read surely is in it (a share
of all of it, so the bar follows the file). Then, within the Latin writing system, the LANGUAGES: Latin letters are
shared by dozens of languages, which are told apart by their commonest small words ('dan', 'untuk' are Malay; 'và',
'của' Vietnamese), counted over the same reading. A file's own text (a PDF's live words) is added to what is read --
never instead of it: a PDF's English can be live text while its Hindi is outlined.

Several languages are read together (Tesseract's "eng+ara+tha"): every word is tried in each and read in the one it fits
best, so a pack printed in six languages is read in all six.

It is found WHILE the comparison runs, not before it (`begin`): the pictures are aligned and compared pixel by pixel
meanwhile, which needs no reading, and the first reading of words (`resolve`, through every reading of the comparison)
waits only for what is left of it.
"""

from __future__ import annotations

import logging
import threading
import time
import unicodedata
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

import cv2
import numpy as np

from app.models.languages import AUTO, DEFAULT_LANGUAGES

logger = logging.getLogger(__name__)

# One installed language for each writing system packs are printed in, read together to tell which are present (its
# Unicode name's first word -> the language its text is then read in). Latin's languages are told apart afterwards.
_SCRIPTS: dict[str, str] = {
    "LATIN": "eng",
    "ARABIC": "ara",
    "CJK": "chi_sim",
    "HIRAGANA": "jpn",
    "KATAKANA": "jpn",
    "HANGUL": "kor",
    "THAI": "tha",
    "DEVANAGARI": "hin",
    "CYRILLIC": "rus",
    "GREEK": "ell",
    "HEBREW": "heb",
    "BENGALI": "ben",
    "TAMIL": "tam",
}
_LETTER_WEIGHT = {"CJK": 3, "HANGUL": 3, "HIRAGANA": 2, "KATAKANA": 2}  # (one such character says a syllable or a word)
_SAMPLE_LINES = 64  # lines read from each file
_LINE_HEIGHT_PX = 36  # each read at this height
_SURE = 75.0  # a word is evidence only if read at least this surely...
_ONE_SCRIPT = 0.8  # ...with at least this share of its letters of one writing system
_SCRIPT_SHARE = 0.01  # a writing system is printed where it holds this share of the letters read surely...
_SCRIPT_MIN_LETTERS = 20  # ...and at least this many (a few letters misread as another script are not a language)
_MAX_LANGUAGES = 6  # (each language read makes reading slower and gives more ways to misread)

# Latin-script languages, each by its commonest small words (accents taken off: a reader set for English reads 'của' as
# 'cua'), chosen to be words the others do not have.
_LATIN_WORDS: dict[str, frozenset[str]] = {
    "eng": frozenset("the and of to with for is are not store keep use each contains only from this per below before".split()),
    "fra": frozenset("le les des du et est pour avec une dans conserver chaque contient sur au aux ne pas".split()),
    "deu": frozenset("der die das und ist mit fur nicht von zu eine bei aufbewahren enthalt".split()),
    "spa": frozenset("el los las del y con para cada contiene conservar por una".split()),
    "ita": frozenset("il gli della delle con ogni contiene conservare non una".split()),
    "por": frozenset("os dos das do da com para cada contem conservar nao uma".split()),
    "nld": frozenset("het een en van voor niet met bewaren bevat".split()),
    "msa": frozenset("dan yang untuk pada setiap dari tidak simpan sehingga selama jumlah oleh ini akan dengan atau minyak kira".split()),
    "vie": frozenset("va cua khong cho trong nhung duoc voi cac mot nay tren dau thuc chat tong phan nhiet ngay nhu bao quan san xuat tai".split()),
    "tur": frozenset("ve bir ile icin bu olarak saklayiniz".split()),
    "pol": frozenset("nie sie jest przechowywac lub oraz".split()),
    "ron": frozenset("si pentru nu este cu sau".split()),
    "fil": frozenset("ang ng sa mga".split()),
}
_LATIN_MIN_WORDS = 3  # a Latin language is printed where at least this many of its words are read...
_LATIN_SHARE = 0.02  # ...making at least this share of the Latin words


# ---- found while the comparison runs --------------------------------------------------------------------------------

_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="languages")
_lock = threading.Lock()
_pending: Future | None = None


def begin(find: Callable[[], tuple[str, ...]]) -> Future:
    """Start finding the languages (`find`, e.g. `detect_languages` on this comparison's pictures) beside the
    comparison; every reading asked for in "auto" waits for it (`resolve`). Ended by `end`."""
    global _pending

    def timed() -> tuple[str, ...]:
        started = time.monotonic()
        languages = find()
        logger.info("languages found", extra={"languages": "+".join(languages), "seconds": round(time.monotonic() - started, 2)})
        return languages

    future = _pool.submit(timed)
    with _lock:
        _pending = future
    return future


def adopt(future: Future | None) -> None:
    """Make `future` (a comparison's own finding, from `begin`) the one "auto" waits for again: for what is read of
    that comparison after it returned (Automatic's word comparison, sent after its pixel findings)."""
    global _pending
    with _lock:
        _pending = future


def note(future: Future | None) -> str | None:
    """What the results say about it: "Text read in English, Arabic and Thai (found on the files)."."""
    if future is None:
        return None
    from app.models.languages import languages_text

    try:
        languages = future.result()
    except Exception:  # noqa: BLE001
        return None
    return f"Text read in {languages_text(languages)} (the languages found on the files)."


def end() -> None:
    """The comparison is over: "auto" means English again until the next one `begin`s."""
    global _pending
    with _lock:
        _pending = None


def found() -> tuple[str, ...] | None:
    """The languages found for the comparison running (waiting for them if need be); None if none is being found."""
    with _lock:
        pending = _pending
    if pending is None:
        return None
    try:
        if not pending.done():
            started = time.monotonic()
            languages = pending.result()
            logger.info("reading waited for the languages", extra={"seconds": round(time.monotonic() - started, 2)})
            return languages
        return pending.result()
    except Exception:  # noqa: BLE001 -- finding them failed: the reader's own default, as without it
        logger.exception("language detection failed")
        return DEFAULT_LANGUAGES


def resolve(requested: str | None) -> str | None:
    """The languages to read in: `requested`, or -- for "auto" -- the ones found on the files (`begin`), English where
    none are being found."""
    if requested != AUTO:
        return requested
    return "+".join(found() or DEFAULT_LANGUAGES)


# ---- finding them ---------------------------------------------------------------------------------------------------


def _script(character: str) -> str | None:
    try:
        name = unicodedata.name(character)
    except ValueError:
        return None
    first = name.split(" ")[0]
    if first == "CJK" or "IDEOGRAPH" in name:
        return "CJK"
    return first if first in _SCRIPTS else None


def _plain(word: str) -> str:
    """'Của' -> 'cua': lower case, accents off, letters only."""
    stripped = "".join(c for c in unicodedata.normalize("NFKD", word.lower()) if not unicodedata.combining(c))
    return "".join(c for c in stripped.replace("đ", "d") if c.isalpha())


def _evidence(words: list[str], scripts: Counter, latin: list[str]) -> None:
    """Count each word's writing system (and keep Latin words for telling Latin languages apart)."""
    for word in words:
        kinds = Counter(kind for kind in map(_script, (c for c in word if c.isalpha())) if kind)
        if not kinds:
            continue
        kind, count = kinds.most_common(1)[0]
        if count < _ONE_SCRIPT * sum(kinds.values()):
            continue
        scripts[kind] += count * _LETTER_WEIGHT.get(kind, 1)
        if kind == "LATIN":
            latin.append(_plain(word))


def _line_crops(image: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """A spread of the picture's lines of type, each brought to one height for reading."""
    from app.engine.worker import _text_lines  # (the finder of lines of type the comparison itself uses)

    lines = [line for line in _text_lines((image,)) if line[2] - line[0] > 3 * (line[3] - line[1])]
    if len(lines) > _SAMPLE_LINES:
        lines = [lines[i] for i in sorted(rng.choice(len(lines), size=_SAMPLE_LINES, replace=False))]
    crops = []
    height, width = image.shape[:2]
    for x0, y0, x1, y1 in lines:
        tall = y1 - y0
        pad = int(tall * 0.3)
        crop = image[max(0, int(y0) - pad) : min(height, int(y1) + pad), max(0, int(x0) - pad) : min(width, int(x1) + pad)]
        if crop.size == 0:
            continue
        scale = _LINE_HEIGHT_PX / max(1.0, tall)
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        crops.append(cv2.copyMakeBorder(crop, 12, 12, 12, 12, cv2.BORDER_REPLICATE))
    return crops


def detect_languages(
    images: list[np.ndarray], own_texts: list[list[str] | None] | None = None, installed: frozenset[str] | None = None
) -> tuple[str, ...]:
    """The languages `images` are printed in (Tesseract codes, English first): each picture's own (from reading a sample
    of its lines, and its file's own text, `own_texts`, where it has any), all of them together -- the two files of a
    comparison may each carry a language the other dropped. English alone where nothing can be told (no text reader, no
    lines of type)."""
    from app.engine.ocr_locator import available_languages

    installed = available_languages() if installed is None else installed
    texts = list(own_texts or []) + [None] * (len(images) - len(own_texts or []))
    every: list[str] = []
    for image, text in zip(images, texts):
        every += [code for code in _detected(image, text, installed) if code not in every]
    usable = tuple(sorted(every, key=lambda code: code != "eng")[:_MAX_LANGUAGES]) or DEFAULT_LANGUAGES
    logger.info("languages detected", extra={"languages": "+".join(usable)})
    return usable


def _detected(image: np.ndarray, own_text: list[str] | None, installed: frozenset[str]) -> tuple[str, ...]:
    """One picture's languages (see `detect_languages`)."""
    from app.engine.detectors import text_diff

    scripts: Counter = Counter()
    latin: list[str] = []
    if own_text:
        _evidence(own_text, scripts, latin)
    if installed:
        crops = _line_crops(image, np.random.default_rng(0))
        readers = [code for code in dict.fromkeys(_SCRIPTS.values()) if code in installed]
        if crops and readers:
            try:
                readings = text_diff._ocr_many(crops, "--psm 7", "+".join(readers))
            except Exception:  # noqa: BLE001 -- no reader: what the file's own text says, else English
                logger.exception("language detection failed")
                readings = []
            sure = [
                text.strip()
                for data in readings
                for text, confidence in zip(data.get("text", []), data.get("conf", []))
                if text.strip() and float(confidence) >= _SURE
            ]
            _evidence(sure, scripts, latin)
    return tuple(code for code in _languages(scripts, latin) if not installed or code in installed)


def _languages(scripts: Counter, latin: list[str]) -> tuple[str, ...]:
    total = sum(scripts.values())
    if not total:
        return DEFAULT_LANGUAGES
    present = [kind for kind, count in scripts.most_common() if count >= max(_SCRIPT_MIN_LETTERS, _SCRIPT_SHARE * total)]
    found_here: list[tuple[str, float]] = []  # (language, how much of it was read)
    for kind in present:
        if kind == "LATIN":
            found_here += _latin_languages(latin, scripts[kind])
        elif kind in ("HIRAGANA", "KATAKANA"):
            found_here.append(("jpn", scripts[kind]))
        elif kind == "CJK" and scripts["HIRAGANA"] + scripts["KATAKANA"] >= _SCRIPT_SHARE * scripts[kind]:
            found_here.append(("jpn", scripts[kind]))  # (kanji among kana: Japanese, not Chinese)
        else:
            found_here.append((_SCRIPTS[kind], scripts[kind]))
    ranked: dict[str, float] = {}
    for code, weight in found_here:
        ranked[code] = ranked.get(code, 0.0) + weight
    chosen = sorted(ranked, key=lambda code: (code != "eng", -ranked[code]))[:_MAX_LANGUAGES]
    return tuple(chosen) or DEFAULT_LANGUAGES


def _latin_languages(words: list[str], letters: float) -> list[tuple[str, float]]:
    """The Latin-script languages among `words` (accents off), by their commonest small words; English if none is told."""
    found_here = []
    for code, common in _LATIN_WORDS.items():
        hits = [word for word in words if word in common]
        if len(hits) >= _LATIN_MIN_WORDS and len(set(hits)) >= 2 and len(hits) >= _LATIN_SHARE * len(words):
            found_here.append((code, letters * len(hits) / max(1, len(words))))
    return found_here or [("eng", letters)]
