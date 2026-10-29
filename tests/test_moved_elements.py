"""An element re-placed or resized in the suspect (a logo a vendor re-laid) is one finding, judged by how alike THIS
pair's unchanged print is -- never a preset bar (`app.engine.detectors.elements`)."""

from __future__ import annotations

import cv2
import numpy as np

from app.engine.detectors.elements import matched_elements
from app.models.diff import BoundingBox, DiffCandidate, DiffCluster, DiffSubtype, DiffType

_PAGE = (1600, 2200)  # (height, width)


def _page(rng: np.random.Generator) -> np.ndarray:
    """A page with print all over it (text-like bars and marks) for the pair's yardstick to be measured on."""
    page = np.full((*_PAGE, 3), 250, np.uint8)
    for _ in range(900):
        x, y = int(rng.integers(0, _PAGE[1] - 60)), int(rng.integers(0, _PAGE[0] - 20))
        colour = tuple(int(v) for v in rng.choice([(20, 20, 20), (30, 60, 150), (40, 120, 60)]))
        cv2.rectangle(page, (x, y), (x + int(rng.integers(8, 50)), y + int(rng.integers(4, 12))), colour, -1)
    return page


def _logo(size: float, colour=(215, 40, 50)) -> np.ndarray:
    """A logo -- a ring, a bar and a letter-like stroke, on white -- rendered as a PDF renders artwork: drawn finely, then
    brought to its size (so the same logo at another size is exactly it, scaled)."""
    fine = np.full((640, 640, 3), 255, np.uint8)
    cv2.circle(fine, (220, 240), 152, colour, 48)
    cv2.rectangle(fine, (400, 80), (500, 560), colour, -1)
    cv2.line(fine, (80, 560), (360, 440), colour, 40)
    side = int(round(160 * size))
    return cv2.resize(fine, (side, side), interpolation=cv2.INTER_AREA)


def _place(page: np.ndarray, logo: np.ndarray, x: int, y: int) -> None:
    page[y : y + logo.shape[0], x : x + logo.shape[1]] = logo


def _pieces(x: int, y: int, side: int) -> list[DiffCluster]:
    """What the pixel comparison reports at a moved logo: a finding at each of its edges."""
    boxes = [(x + 15, y + 20, x + 95, y + 100), (x + 95, y + 15, x + 130, y + 145), (x + 15, y + 105, x + 95, y + 145)]
    clusters = []
    for number, (x0, y0, x1, y1) in enumerate(boxes, start=1):
        box = BoundingBox(x0, y0, x1, y1)
        member = DiffCandidate(box, DiffType.COLOUR, DiffSubtype.CHANGED, 0.9, "Colour changed (dE 30.0)", 30.0, region=True)
        clusters.append(DiffCluster(number, box, box, DiffType.COLOUR, [member], detail=member.detail))
    return clusters


def _pair(shift=(36, 28), scale=0.95, recolour=None, suspect_tint=None):
    rng = np.random.default_rng(7)
    original = _page(rng)
    suspect = original.copy()
    x, y = 900, 700
    blank = np.full((260, 260, 3), 250, np.uint8)
    _place(original, blank, x - 50, y - 50)
    _place(suspect, blank, x - 50, y - 50)
    _place(original, _logo(1.0), x, y)
    _place(suspect, _logo(scale, recolour or (215, 40, 50)), x + shift[0], y + shift[1])
    if suspect_tint is not None:
        # the proof prints everything a shade warmer: how this pair differs where nothing changed
        suspect = np.clip(suspect.astype(np.int16) + np.array(suspect_tint, np.int16), 0, 255).astype(np.uint8)
    return original, suspect, _pieces(x, y, 160)


def test_a_moved_and_resized_logo_is_one_finding_shown_where_it_went():
    original, suspect, clusters = _pair()
    out = matched_elements(clusters, original, suspect)
    assert len(out) == 1
    moved = out[0]
    assert moved.subtype is DiffSubtype.MOVED and moved.category == "graphics"
    assert "moved" in moved.detail and "95% of its size" in moved.detail
    assert moved.moved_to is not None and abs(moved.moved_to.x0 - (900 + 36)) <= 6 and abs(moved.moved_to.y0 - (700 + 28)) <= 6


def test_a_logo_moved_and_recoloured_keeps_its_findings():
    original, suspect, clusters = _pair(recolour=(40, 60, 210))
    out = matched_elements(clusters, original, suspect)
    assert len(out) == 3 and all(c.subtype is not DiffSubtype.MOVED for c in out)


def test_the_bar_is_the_pairs_own_a_proof_printed_warmer_everywhere_still_finds_the_logo_moved():
    original, suspect, clusters = _pair(suspect_tint=(4, -3, -9))
    out = matched_elements(clusters, original, suspect)
    assert len(out) == 1 and out[0].subtype is DiffSubtype.MOVED


def test_findings_that_are_no_element_moved_stay():
    rng = np.random.default_rng(7)
    original = _page(rng)
    suspect = original.copy()
    blank = np.full((260, 260, 3), 250, np.uint8)
    _place(original, blank, 850, 650)
    _place(suspect, blank, 850, 650)
    _place(original, _logo(1.0), 900, 700)  # removed from the suspect altogether
    clusters = _pieces(900, 700, 160)
    assert len(matched_elements(clusters, original, suspect)) == 3


def _label(batch: str) -> np.ndarray:
    """A batch label -- five lines of small dark text in a box, as on a carton -- rendered finely and brought to size."""
    fine = np.full((480, 720, 3), 255, np.uint8)
    cv2.rectangle(fine, (8, 8), (712, 472), (60, 60, 60), 5)
    lines = ("Mfg. Lic. No.: 6/UA/2018", f"Batch No.: {batch}", "Mfg. Date: 2019-04", "Exp. Date: 2022-03", "MRP Rs.: 150.00")
    for row, text in enumerate(lines):
        cv2.putText(fine, text, (36, 90 + 82 * row), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (30, 30, 30), 4, cv2.LINE_AA)
    return cv2.resize(fine, (240, 160), interpolation=cv2.INTER_AREA)


def _label_pair(suspect_batch: str):
    rng = np.random.default_rng(11)
    original = _page(rng)
    blank = np.full((260, 400, 3), 250, np.uint8)
    _place(original, blank, 850, 650)
    suspect = original.copy()
    _place(original, _label("G5E1S001"), 900, 700)
    _place(suspect, _label(suspect_batch), 930, 724)  # re-placed by the vendor
    # ...in a proof rendered a little softer, with grain: how this pair differs where nothing changed
    suspect = cv2.GaussianBlur(suspect, (0, 0), 0.8)
    grain = np.random.default_rng(3).normal(0, 4, suspect.shape)
    suspect = np.clip(suspect + grain, 0, 255).astype(np.uint8)
    boxes = [(905, 705, 1015, 780), (1015, 705, 1135, 780), (905, 780, 1015, 855), (1015, 780, 1135, 855)]
    clusters = []
    for number, (x0, y0, x1, y1) in enumerate(boxes, start=1):
        box = BoundingBox(x0, y0, x1, y1)
        member = DiffCandidate(box, DiffType.GRAPHICS, DiffSubtype.CHANGED, 0.9, "Graphic changed", 0.3)
        clusters.append(DiffCluster(number, box, box, DiffType.GRAPHICS, [member], detail=member.detail))
    return original, suspect, clusters


def _recheck(original, suspect):
    """The comparison's settings calibrated to the pair, as the worker hands them over."""
    from app.config import AppConfig
    from app.engine.detectors import local_diff

    detection = AppConfig().detection
    return local_diff.calibrated(detection, local_diff.measure_noise(original, suspect, detection))


def test_a_label_re_placed_as_it_was_is_one_finding():
    original, suspect, clusters = _label_pair("G5E1S001")
    out = matched_elements(clusters, original, suspect, _recheck(original, suspect))
    assert len(out) == 1 and out[0].subtype is DiffSubtype.MOVED


def test_a_label_re_placed_with_a_figure_changed_keeps_its_findings():
    """Moved AND re-set: one changed figure barely moves the fit of the whole label, but compared again once lined up
    (the comparison's own settings) it is found -- so the label is not called 'moved' and its findings stay."""
    original, suspect, clusters = _label_pair("G5E1S002")
    out = matched_elements(clusters, original, suspect, _recheck(original, suspect))
    assert len(out) == 4 and all(c.subtype is not DiffSubtype.MOVED for c in out)
