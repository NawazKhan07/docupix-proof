"""Findings that are pieces of one thing are one finding (a line of text re-set, a photograph re-edited), and an artwork
is found on a sheet however it lies there (turned, or beside an enlarged copy of itself)."""

from __future__ import annotations

import cv2
import numpy as np

from app.config import SeriesConfig
from app.engine.isolation.auto_area import located_by_print
from app.engine.worker import _by_picture, _by_text_line, _joined_words, _pictures, _text_lines
from app.models.diff import BoundingBox, DiffCandidate, DiffCluster, DiffSubtype, DiffType


def _cluster(box, kind=DiffType.GRAPHICS, subtype=DiffSubtype.CHANGED, number=1) -> DiffCluster:
    b = BoundingBox(*box)
    member = DiffCandidate(b, kind, subtype, 1.0, "Graphic changed")
    return DiffCluster(id=number, bbox_in_original=b, bbox_in_suspect=b, type=kind, members=[member], detail=member.detail)


def test_words_side_by_side_are_one_line_and_rows_apart_are_not():
    words = [[100, 50, 300, 90], [330, 52, 520, 92], [100, 150, 300, 190], [900, 50, 1000, 90]]
    lines = _joined_words(words)
    assert [100, 50, 520, 92] in lines  # a gap smaller than the type's height: one line
    assert any(line[1] == 150 for line in lines)  # the next row: its own line
    assert any(line[0] == 900 for line in lines)  # far along the row: its own


def _page() -> np.ndarray:
    page = np.full((900, 1600, 3), 255, np.uint8)
    cv2.putText(page, "Sugammadex Piramal 100 mg/mL", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (20, 20, 20), 4)
    cv2.putText(page, "soluzione iniettabile", (100, 330), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (20, 20, 20), 4)
    return page


def test_the_letters_of_one_re_set_line_are_one_finding_and_the_next_line_its_own():
    page = _page()
    pieces = [_cluster((110 + 90 * k, 160, 150 + 90 * k, 205), number=k + 1) for k in range(8)]  # letters of line one
    other = _cluster((120, 290, 200, 335), number=9)  # a letter of line two
    clusters = _by_text_line(pieces + [other], _text_lines((page, page.copy())))
    assert len(clusters) == 2
    line = max(clusters, key=lambda c: len(c.members))
    assert len(line.members) == 8 and line.bbox_in_original.x1 >= 110 + 90 * 7 + 40


def _photo_page() -> np.ndarray:
    rng = np.random.default_rng(3)
    page = np.full((1200, 1600, 3), 255, np.uint8)
    yy, xx = np.mgrid[0:600, 0:700]
    tone = (110 + 50 * np.sin(xx / 45.0) + 35 * np.cos(yy / 60.0) + rng.normal(0, 14, (600, 700))).clip(0, 255).astype(np.uint8)  # (a photograph's grain throughout)
    page[100:700, 100:800] = np.dstack([tone, tone, tone])  # a photograph: continuous tone
    for row in range(10):  # a block of small print beside it: not a photograph
        cv2.putText(page, "Store below 25 C, protect from light and moisture", (900, 150 + row * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1)
    return page


def test_a_photograph_is_found_and_small_print_is_not():
    page = _photo_page()
    labels = _pictures((page, page))
    scale = 0.25
    assert labels[int(400 * scale), int(450 * scale)] > 0
    assert labels[int(300 * scale), int(1200 * scale)] == 0


def test_the_pieces_of_a_re_edited_photograph_are_one_finding():
    page = _photo_page()
    pieces = [_cluster((150 + 120 * k, 200 + 60 * (k % 3), 200 + 120 * k, 260 + 60 * (k % 3)), DiffType.COLOUR, number=k + 1) for k in range(5)]
    text = _cluster((900, 130, 1300, 160), DiffType.TEXT, number=6)
    clusters = _by_picture(pieces + [text], page, page.copy())
    assert len(clusters) == 2
    picture = next(c for c in clusters if c.category != "text")
    assert "Picture changed" in picture.detail and len(picture.members) == 5


def _label() -> np.ndarray:
    label = np.full((240, 720, 3), 255, np.uint8)
    cv2.rectangle(label, (0, 0), (719, 239), (40, 40, 160), 6)
    cv2.putText(label, "ANTI-RABIES SERUM", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (150, 60, 20), 3)
    cv2.putText(label, "EQUIRAB 1000 IU", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 2)
    cv2.circle(label, (620, 120), 60, (40, 120, 200), -1)
    cv2.putText(label, "Batch 21A", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    return label


def test_a_label_laid_on_the_glass_turned_is_found_on_the_proof_that_also_shows_it_enlarged():
    label = _label()
    proof = np.full((2600, 2000, 3), 255, np.uint8)
    proof[200:440, 300:1020] = label  # actual size
    big = cv2.resize(label, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    proof[1200:1680, 300:1740] = big  # enlarged 200% (the largest print on the proof)
    scan = np.full((2400, 1800, 3), 250, np.uint8)
    turned = np.rot90(label).copy()
    scan[600:600 + turned.shape[0], 900:900 + turned.shape[1]] = turned

    area = located_by_print(proof, scan, SeriesConfig(), 1.0)

    assert area is not None
    found = area.suspect.boundary  # (normalised to the scan) round the turned label, at (900..1140, 600..1320)
    assert found.x0 * 1800 >= 840 and found.x1 * 1800 <= 1200  # on the label...
    assert found.y0 * 2400 >= 540 and found.y1 * 2400 <= 1380
    assert (found.y1 - found.y0) * 2400 >= 0.5 * 720  # ...and most of it (its blank ends and thin border aside)
    on_proof = area.original.boundary  # the label on the proof: its actual-size copy or the enlarged one
    assert (abs(on_proof.y0 * 2600 - 200) < 80) or (abs(on_proof.y0 * 2600 - 1200) < 120)
