"""Scan vs Digital: Braille left out (a print embosses it), residual ink left out of the graphics, a coloured panel shown
the same way in both files, lettering on part of a dark block read, and an artwork found by its print on a proof sheet.
Synthetic pictures only."""

from __future__ import annotations

import cv2
import numpy as np

from app.config import AppConfig
from app.engine.detectors.braille_diff import braille_mask
from app.engine.detectors.text_diff import _readable
from app.engine.isolation.auto_area import located_by_print, print_block
from app.engine.preprocessing.scan_normalizer import ink_image, ink_pair
from app.engine.worker import _drop_at_carton_edges, _drop_non_printing, _drop_residual_ink
from app.models.diff import BoundingBox, DiffCandidate, DiffSubtype, DiffType

_DOT = 9  # a Braille dot's radius at 300 DPI (~1.5 mm across)
_PITCH = 30  # between a cell's dots (~2.5 mm)


def _braille(image: np.ndarray, x0: int, y0: int, cells: int, colour=(20, 20, 20)) -> list[tuple[int, int]]:
    rng = np.random.default_rng(7)
    placed = []
    for cell in range(cells):
        for column in range(2):
            for row in range(3):
                if rng.random() < 0.55 or (column, row) == (0, 0):
                    centre = (x0 + cell * 3 * _PITCH + column * _PITCH, y0 + row * _PITCH)
                    cv2.circle(image, centre, _DOT, colour, -1, cv2.LINE_AA)
                    placed.append(centre)
    return placed


def _text(image: np.ndarray, x: int, y: int, lines: int = 4) -> None:
    for line in range(lines):
        cv2.putText(image, f"Store below 25 C. Keep dry. Tablets i.p. {line}", (x, y + 45 * line), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)


def test_braille_is_found_by_its_shape_and_arrangement_and_type_is_not():
    page = np.full((700, 1300, 3), 255, np.uint8)
    _text(page, 40, 60)
    dots = _braille(page, 60, 350, cells=8)
    found = braille_mask(page)
    assert found is not None
    mask, count = found
    assert count >= 0.9 * len(dots)
    assert all(mask[y, x] for x, y in dots)
    assert not mask[20:260].any()  # the type -- its full stops, the dots of its i's -- is not Braille


def test_braille_on_a_panel_and_under_type_is_found_too():
    page = np.full((500, 1300, 3), 255, np.uint8)
    page[200:480] = (0, 140, 70)  # a green panel
    dots = _braille(page, 60, 240, cells=8, colour=(200, 40, 160))  # magenta dots on it
    cv2.putText(page, "Solifenacin Tablets 10 mg", (60, 290), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (250, 230, 60), 3, cv2.LINE_AA)
    found = braille_mask(page)
    assert found is not None
    assert found[1] >= 0.7 * len(dots)


def test_a_page_without_braille_has_none():
    page = np.full((600, 1200, 3), 255, np.uint8)
    _text(page, 40, 60, lines=10)
    for x in range(100, 1100, 150):  # a few round bullets
        cv2.circle(page, (x, 520), _DOT, (20, 20, 20), -1)
    assert braille_mask(page) is None


def _graphic(box: BoundingBox) -> DiffCandidate:
    return DiffCandidate(box, DiffType.GRAPHICS, DiffSubtype.ADDED, 1.0, "Graphic added in suspect")


def test_residual_ink_is_left_out_of_the_graphics_but_a_printed_mark_is_not():
    digital = np.full((300, 600), 255, np.uint8)
    scan = digital.copy()
    for x, y in ((40, 40), (70, 55), (95, 45)):  # dust: specks of a few pixels
        cv2.circle(scan, (x, y), 2, 0, -1)
    cv2.circle(scan, (400, 150), 14, 0, -1)  # a mark really printed (a symbol, ~2.4 mm)
    specks = _graphic(BoundingBox(30, 30, 110, 70))
    mark = _graphic(BoundingBox(380, 130, 420, 170))
    text = DiffCandidate(BoundingBox(30, 30, 110, 70), DiffType.TEXT, DiffSubtype.REMOVED, 1.0, "'.' missing from suspect")
    kept, dropped = _drop_residual_ink([specks, mark, text], digital, scan, AppConfig().detection.scan_residual_ink_px)
    assert dropped == 1 and kept == [mark, text]


def _panel_pair() -> tuple[np.ndarray, np.ndarray]:
    """A digital label with a flat green panel and white type on it, and its 'scan': the same panel shaded (darker at
    one end, grainy), the type the same."""
    digital = np.full((400, 900, 3), 255, np.uint8)
    digital[60:340, 100:820] = (0, 150, 80)
    for picture in (digital,):
        cv2.putText(picture, "ORIGINAL", (160, 190), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (255, 255, 255), 8, cv2.LINE_AA)
        cv2.putText(picture, "LIQUID HANDWASH", (160, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4, cv2.LINE_AA)
    scan = digital.astype(np.float32)
    shade = np.linspace(1.0, 0.25, 720, dtype=np.float32)[None, :, None]  # (steep: not one flat colour in the scan)
    panel = scan[60:340, 100:820]
    green = (panel[..., 0] < 120)[..., None]
    scan[60:340, 100:820] = np.where(green, panel * shade, panel)
    scan += np.random.default_rng(3).normal(0, 6, scan.shape)
    return digital, np.clip(scan, 0, 255).astype(np.uint8)


def test_a_panel_comes_out_the_same_way_in_both_files():
    digital, scan = _panel_pair()
    alone = np.abs(ink_image(digital).astype(int) - ink_image(scan)) > 128
    first, second, _shown_first, _shown_second = ink_pair(digital, scan)
    together = np.abs(first.astype(int) - second) > 128
    assert together[60:340, 100:820].mean() <= alone[60:340, 100:820].mean()
    assert together[60:340, 100:820].mean() < 0.1


def test_light_type_on_part_of_a_dark_block_is_turned_the_right_way_to_read():
    crop = np.full((80, 600), 255, np.uint8)
    crop[:, 200:] = 0  # a black block from partway along
    cv2.putText(crop, "ORIGINAL", (230, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 255, 4)
    cv2.putText(crop, "BE SURE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    readable = _readable(crop)
    assert readable[:, 200:].mean() > 150  # the block is paper now, its type dark
    assert readable[:, :190].mean() > 150 and readable[:, :190].min() < 100  # the rest as it was: dark type on paper


def test_a_line_of_dark_type_is_not_taken_for_a_block():
    crop = np.full((70, 600), 255, np.uint8)
    cv2.putText(crop, "Keep out of the reach of children", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 3)
    assert (_readable(crop) == crop).all()


def _label(width: int = 600, height: int = 440) -> np.ndarray:
    label = np.full((height, width, 3), 255, np.uint8)
    label[40:300, 250:580] = (0, 140, 70)
    cv2.circle(label, (140, 130), 90, (0, 110, 50), -1)
    cv2.putText(label, "Dettol", (70, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    cv2.putText(label, "ORIGINAL", (280, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    cv2.putText(label, "BE 100% SURE", (30, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 40), 2)
    cv2.rectangle(label, (300, 330), (420, 420), (200, 30, 30), -1)
    return label


def test_one_artwork_is_found_on_a_proof_sheet_of_two_and_a_job_sheet():
    sheet = np.full((1500, 1500, 3), 255, np.uint8)
    sheet[200:640, 100:700] = _label()
    back = np.full((440, 600, 3), 250, np.uint8)
    for line in range(8):
        cv2.putText(back, f"Directions line {line} for use", (20, 40 + 45 * line), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
    sheet[200:640, 800:1400] = back
    for row in range(6):  # the job sheet
        cv2.rectangle(sheet, (100, 900 + 80 * row), (1400, 960 + 80 * row), (0, 0, 0), 2)
        cv2.putText(sheet, f"APPROVED BY  PERSON {row}   DATE 24-11-2022", (120, 940 + 80 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    scan = np.full((1400, 1000, 3), 245, np.uint8)
    scan[400:840, 200:800] = _label()
    scan[:, :4] = 30  # the scanner's edges
    scan[:4] = 30
    box = print_block(scan)
    assert box[0] >= 150 and box[2] <= 850 and box[1] >= 350 and box[3] <= 900
    found = located_by_print(sheet, scan, AppConfig().series, 1.0)
    assert found is not None
    b = found.original.boundary
    x0, y0, x1, y1 = b.x0 * 1500, b.y0 * 1500, b.x1 * 1500, b.y1 * 1500
    assert 90 <= x0 < 200 and 180 <= y0 < 280 and 640 < x1 < 760 and 560 < y1 < 700  # the front label: not the back, nor the job sheet


# ---- dark type on a panel the scan shades; a different ink; a scan's print beside a note to the printer ---------------------


def _red_box_pair() -> tuple[np.ndarray, np.ndarray]:
    """A digital warning box, flat red with black type, and its 'scan': the red uneven (darker along it, grainy)."""
    digital = np.full((300, 900, 3), 255, np.uint8)
    digital[60:240, 60:840] = (225, 20, 50)
    cv2.putText(digital, "Warning : To be sold by retail", (90, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(digital, "on the prescription only", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3, cv2.LINE_AA)
    scan = digital.astype(np.float32)
    shade = np.linspace(1.0, 0.3, 780, dtype=np.float32)[None, :, None]  # (steep: not one flat colour in the scan)
    box = scan[60:240, 60:840]
    red = (box[..., 0] > 150)[..., None]
    scan[60:240, 60:840] = np.where(red, box * shade, box)
    scan += np.random.default_rng(5).normal(0, 7, scan.shape)
    return digital, np.clip(scan, 0, 255).astype(np.uint8)


def test_dark_type_on_a_panel_is_readable_in_both_files():
    digital, scan = _red_box_pair()
    first, second, _shown_first, _shown_second = ink_pair(digital, scan)
    for ink in (first, second):
        box = ink[70:230, 70:830]
        assert np.median(box) > 200  # the red is paper...
        assert (box < 100).mean() > 0.03  # ...and its black type is ink, not lost in a black block


def test_an_area_printed_in_another_ink_is_a_colour_finding_and_a_scans_shift_is_not():
    from app.engine.worker import _colour_swaps

    digital = np.full((600, 900, 3), 255, np.uint8)
    digital[100:200, 100:200] = (220, 10, 70)  # red
    digital[100:200, 400:500] = (220, 10, 70)
    scan = digital.copy()
    scan[100:200, 100:200] = (0, 100, 120)  # teal: a different ink
    scan[100:200, 400:500] = (235, 60, 60)  # the same red, as a scan shifts it
    found = _colour_swaps(digital, scan, np.ones((600, 900), bool))
    assert len(found) == 1
    assert found[0].bbox.x0 < 150 and "red in the digital file, teal in the scan" in found[0].detail


def test_what_a_scan_prints_beside_a_note_to_the_printer_is_its_own():
    from app.engine.worker import _printed_beside

    note = np.zeros((100, 400), bool)
    note[30:60, 20:150] = True  # the digital file's non-printing note
    scan = np.full((100, 400, 3), 255, np.uint8)
    scan[30:60, 20:150] = (240, 120, 200)  # (the scan happens to show it faintly: it is the note's)
    assert _printed_beside(scan, note) is None
    scan[30:70, 250:380] = (40, 40, 160)  # a colour bar printed beside it
    box = _printed_beside(scan, note)
    assert box is not None and box[0] >= 240 and box[2] <= 385


def test_small_print_of_many_tiny_marks_is_not_residual_ink():
    """A date wheel (a ring of tiny digits the printer adds) is small pieces of ink, but many: it is kept."""
    digital = np.full((200, 300), 255, np.uint8)
    scan = digital.copy()
    for index in range(12):
        angle = 2 * np.pi * index / 12
        centre = (150 + int(50 * np.cos(angle)), 100 + int(50 * np.sin(angle)))
        cv2.putText(scan, str(index + 1)[-1], centre, cv2.FONT_HERSHEY_PLAIN, 0.8, 0, 1)
    wheel = _graphic(BoundingBox(90, 40, 215, 160))
    kept, dropped = _drop_residual_ink([wheel], digital, scan, AppConfig().detection.scan_residual_ink_px)
    assert dropped == 0 and kept == [wheel]


def test_a_note_to_the_printer_missing_from_a_scan_is_set_apart_not_dropped():
    """What the digital file draws in its die line's colour -- a note to the printer -- is never printed: missing from a
    scan it is kept as a printer's note (listed on request), not a finding and not thrown away."""
    digital = np.full((400, 600, 3), 255, np.uint8)
    magenta = (230, 0, 140)
    cv2.rectangle(digital, (40, 40), (560, 360), magenta, 2)  # the die line, in its spot colour
    cv2.putText(digital, "UNVARNISHED AREA", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, magenta, 2)
    cv2.putText(digital, "PARACETAMOL", (120, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    boundary = np.zeros((400, 600), bool)
    boundary[40:361, 40:561] = True
    scan = np.full((400, 600, 3), 250, np.uint8)
    note = DiffCandidate(BoundingBox(115, 170, 420, 210), DiffType.TEXT, DiffSubtype.REMOVED, 1.0, "'UNVARNISHED AREA' missing from suspect")
    word = DiffCandidate(BoundingBox(115, 270, 330, 310), DiffType.TEXT, DiffSubtype.REMOVED, 1.0, "'PARACETAMOL' missing from suspect")

    kept, dropped = _drop_non_printing([note, word], digital, boundary, scan)

    assert dropped == 0 and len(kept) == 2
    assert kept[0].printers_note and "note to the printer" in kept[0].detail
    assert not kept[1].printers_note  # black type is printed: a real finding


def test_residue_along_the_cartons_cut_and_fold_lines_is_left_out_but_print_near_them_is_not():
    """A scanned carton shows its cut edges, folds and creases, and lacks the file's cut lines and the bleed printed past
    them. A thin or small graphics finding lying along the digital file's line-work is that -- not something printed near
    an edge (thicker), not a line away from any edge, and never a text finding."""
    digital = np.full((600, 800, 3), 255, np.uint8)
    cv2.rectangle(digital, (100, 100), (700, 500), (30, 30, 30), 2)  # the carton's cut line
    cv2.line(digital, (400, 100), (400, 500), (30, 30, 30), 2)  # a fold between two panels
    compare = np.ones((600, 800), bool)

    def graphic(x0, y0, x1, y1, subtype=DiffSubtype.ADDED):
        return DiffCandidate(BoundingBox(x0, y0, x1, y1), DiffType.GRAPHICS, subtype, 1.0, "Graphic")

    shadow = graphic(395, 150, 403, 400)  # the fold's shadow in the scan
    cut = graphic(150, 99, 650, 102, DiffSubtype.REMOVED)  # the file's cut line, not printed
    bleed = graphic(300, 510, 316, 524, DiffSubtype.REMOVED)  # bleed past the cut, cut off the carton
    wheel = graphic(560, 120, 630, 175)  # a date wheel printed near the edge: not thin
    rule = graphic(200, 300, 350, 304)  # a rule missing in the middle of a panel: no edge near it
    text = DiffCandidate(BoundingBox(395, 150, 403, 400), DiffType.TEXT, DiffSubtype.ADDED, 1.0, "'l' added")

    kept, dropped = _drop_at_carton_edges([shadow, cut, bleed, wheel, rule, text], digital, compare)

    assert dropped == 3 and kept == [wheel, rule, text]


def test_a_finding_whose_difference_is_only_a_cartons_edge_is_left_out_however_big_its_box():
    """A scanned flap's edge runs diagonally, so its finding's box is wide, but what differs is one thin stroke (and the
    hairlines where two edges fall a pixel apart): residue. Print beside such an edge -- a code the printer adds on the
    flap -- is kept, as a finding of its own round just that."""
    digital = np.full((400, 500), 255, np.uint8)
    cv2.rectangle(digital, (50, 50), (450, 350), 0, -1)
    cv2.rectangle(digital, (52, 52), (448, 348), 255, -1)  # the flap's outline, in both
    scan = digital.copy()
    cv2.line(scan, (380, 60), (440, 330), 0, 4)  # the flap's own edge, only the scan sees
    scan[100:340, 120] = 0  # a hairline where an edge falls a pixel off
    compare = np.ones((400, 500), bool)
    colour = cv2.cvtColor(digital, cv2.COLOR_GRAY2RGB)
    flap = DiffCandidate(BoundingBox(100, 55, 445, 345), DiffType.GRAPHICS, DiffSubtype.CHANGED, 1.0, "Graphic changed")

    kept, dropped = _drop_at_carton_edges([flap], colour, compare, digital, scan)
    assert dropped == 1 and kept == []

    cv2.putText(scan, "06 ATFAH", (180, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)  # printed on the flap
    kept, dropped = _drop_at_carton_edges([flap], colour, compare, digital, scan)
    assert dropped == 0 and len(kept) == 1 and kept[0].subtype is DiffSubtype.ADDED
    box = kept[0].bbox
    assert 170 <= box.x0 and box.x1 <= 320 and 170 <= box.y0 and box.y1 <= 215  # round the code, not the flap
