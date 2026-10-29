"""The languages a file is printed in, found from the file itself (`app.engine.language_detect`)."""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np
import pytest

from app.engine.language_detect import _evidence, _languages, detect_languages


def _found(words: list[str]) -> tuple[str, ...]:
    scripts, latin = Counter(), []
    _evidence(words, scripts, latin)
    return _languages(scripts, latin)


def test_each_writing_system_printed_is_read_in_its_language_english_first():
    words = ["Each", "tablet", "contains", "the", "and", "of", "Store", "below"] * 6 + ["الحصص", "لكل", "عبوة"] * 4
    words += ["แสดงเปอร์เซ็นต์", "ของปริมาณ"] * 3 + ["饱和脂肪", "钠"] * 4
    assert _found(words) == ("eng", "chi_sim", "tha", "ara") or set(_found(words)) == {"eng", "ara", "tha", "chi_sim"}
    assert _found(words)[0] == "eng"


def test_latin_languages_are_told_apart_by_their_common_words_accents_or_not():
    malay = "Simpan pada suhu bilik dan jauhkan dari kanak kanak untuk setiap hidangan yang".split()
    vietnamese = "Thành phần khoai tây và dầu cọ thực vật của sản phẩm không".split()
    english = "Store below 30 and keep out of the reach of children for each use".split()
    found = _found(english * 3 + malay * 3 + vietnamese * 3)
    assert set(found) == {"eng", "msa", "vie"} and found[0] == "eng"


def test_a_stray_letter_of_another_writing_system_is_not_a_language():
    words = ["Each", "tablet", "contains", "the", "and", "of"] * 30 + ["Ж"]
    assert _found(words) == ("eng",)


def test_nothing_read_is_english():
    assert _found([]) == ("eng",)


def test_a_files_own_text_says_its_languages_without_reading():
    own = ("Chaque comprimé contient 500 mg de paracétamol. Conserver à une température ne dépassant pas 30 °C, dans "
           "son emballage d'origine. Tenir hors de la vue et de la portée des enfants pour une utilisation sur la peau.").split()
    assert detect_languages([np.full((50, 50, 3), 255, np.uint8)], [own], installed=frozenset({"eng", "fra"})) == ("fra",)


def test_an_english_page_read_from_its_picture_is_english(monkeypatch):
    import pytesseract

    from app.engine.ocr_locator import find_tesseract

    if find_tesseract() is None:
        pytest.skip("Tesseract is not installed")
    monkeypatch.setattr(pytesseract.pytesseract, "tesseract_cmd", str(find_tesseract()))  # (for this test only)
    page = np.full((900, 1400, 3), 255, np.uint8)
    for row in range(8):
        cv2.putText(page, "Store the tablets below 30 C and keep out of the reach of children", (40, 80 + row * 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    assert detect_languages([page]) == ("eng",)


def test_auto_waits_for_the_languages_being_found_beside_the_comparison():
    import threading
    import time

    from app.engine import language_detect

    started = threading.Event()

    def slow():
        started.set()
        time.sleep(0.3)  # (found while the comparison goes on: nothing waits for it until a word is read)
        return ("eng", "ara")

    begun = time.monotonic()
    future = language_detect.begin(slow)
    try:
        assert time.monotonic() - begun < 0.2  # begun, not waited for
        started.wait(2)
        assert language_detect.resolve("eng+fra") == "eng+fra"  # languages given are read as given
        assert language_detect.resolve("auto") == "eng+ara"  # a reading in "auto" waits for what was found
        assert "English and Arabic" in language_detect.note(future)
    finally:
        language_detect.end()
    assert language_detect.resolve("auto") == "eng"  # nothing being found: the reader's own default


def test_a_failed_finding_reads_in_english():
    from app.engine import language_detect

    def broken():
        raise RuntimeError("no reader")

    language_detect.begin(broken)
    try:
        assert language_detect.resolve("auto") == "eng"
    finally:
        language_detect.end()
