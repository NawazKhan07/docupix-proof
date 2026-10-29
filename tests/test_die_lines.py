"""A carton's die lines are never differences (`app.engine.isolation.die_lines`, and the comparison leaving them out)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pymupdf as fitz
import pytest

from app.config import AppConfig
from app.engine.isolation import die_lines
from app.engine.worker_process import ComparisonRequest, execute_request
from app.models.job import ComparisonMode

DPI = 150
MAGENTA, GREEN, RED = (0.93, 0.2, 0.55), (0.1, 0.6, 0.3), (0.8, 0.1, 0.1)


def _carton(path: Path, die=MAGENTA, dashed: bool = False, with_die: bool = True, text: str = "PARACETAMOL 500 mg") -> Path:
    """A two-panel carton: its outline and fold drawn as a die line (or not), its printing in black."""
    doc = fitz.open()
    page = doc.new_page(width=500, height=360)
    page.draw_rect(fitz.Rect(60, 60, 440, 300), color=None, fill=(0.93, 0.95, 0.98))  # the printed panels
    page.insert_text((90, 140), text, fontsize=16)
    page.insert_text((256, 220), "Store below 25 C", fontsize=12)  # (set right up against the fold)
    page.insert_text((190, 180), "Keep dry", fontsize=12)
    page.draw_rect(fitz.Rect(90, 240, 200, 280), color=(0, 0, 0), fill=(0.1, 0.2, 0.6))  # a printed block
    page.draw_rect(fitz.Rect(220, 240, 290, 280), color=None, fill=(0.85, 0.2, 0.1))  # a block the fold runs across
    if with_die:
        dashes = "[6 4] 0" if dashed else None
        page.draw_rect(fitz.Rect(60, 60, 440, 300), color=die, width=1.2, dashes=dashes)
        page.draw_line(fitz.Point(250, 60), fitz.Point(250, 300), color=die, width=1.2, dashes=dashes)
        # short angled creases across the printing (a flap's notch): not long enough to be taken for a fold by shape
        page.draw_line(fitz.Point(215, 245), fitz.Point(285, 275), color=die, width=1.2, dashes=dashes)
        page.draw_line(fitz.Point(95, 245), fitz.Point(150, 275), color=die, width=1.2, dashes=dashes)
    doc.save(path)
    doc.close()
    return path


def _render(path: Path) -> tuple[np.ndarray, list, np.ndarray]:
    with fitz.open(path) as doc:
        page = doc[0]
        pixmap = page.get_pixmap(dpi=DPI)
        image = np.frombuffer(pixmap.samples, np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)[:, :, :3].copy()
        strokes = die_lines.vector_strokes(page, DPI)
    boundary = np.zeros(image.shape[:2], bool)
    scale = DPI / 72
    boundary[int(60 * scale) : int(300 * scale) + 1, int(60 * scale) : int(440 * scale) + 1] = True
    return image, strokes, boundary


def test_a_pdfs_die_lines_are_its_paths_along_the_outline_and_in_their_colour(tmp_path):
    image, strokes, boundary = _render(_carton(tmp_path / "carton.pdf"))
    mask = die_lines.die_line_mask(image, boundary, strokes)
    assert mask is not None
    scale = DPI / 72
    assert mask[int(180 * scale), int(250 * scale)]  # the fold, inside, found by its colour
    assert mask[int(60 * scale), int(150 * scale)]  # the outline
    assert not mask[int(260 * scale), int(145 * scale)]  # the printed block is not a die line
    text_row = mask[int(125 * scale) : int(142 * scale), int(95 * scale) : int(240 * scale)]
    assert not text_row.any()  # nor is the type


def test_a_design_printed_to_its_edge_has_no_die_line_to_take(tmp_path):
    image, strokes, boundary = _render(_carton(tmp_path / "carton.pdf", with_die=False))
    assert die_lines.die_line_mask(image, boundary, strokes) is None


def test_a_die_line_is_drawn_out_of_the_picture_leaving_what_it_crossed():
    image = np.full((60, 60, 3), 255, np.uint8)
    image[20:40, :] = (30, 60, 120)  # a band
    image[:, 29:32] = (235, 50, 140)  # a die line across it
    lines = np.zeros((60, 60), bool)
    lines[:, 29:32] = True
    filled = die_lines.drawn_out(image, lines)
    assert (filled[25, 30] == (30, 60, 120)).all()  # the band continues where the line was
    assert (filled[5, 30] == (255, 255, 255)).all()


@pytest.mark.parametrize(
    "suspect_die",
    [
        {"with_die": False},  # the other file has no die line at all
        {"die": GREEN},  # it draws it in another colour
        {"die": MAGENTA, "dashed": True},  # it dashes it
        {"die": RED, "dashed": True},  # both
    ],
    ids=["missing", "other-colour", "dashed", "other-colour-dashed"],
)
def test_die_lines_drawn_differently_are_never_findings(tmp_path, suspect_die):
    original = _carton(tmp_path / "original.pdf")
    suspect = _carton(tmp_path / "suspect.pdf", **suspect_die)
    outcome = execute_request(ComparisonRequest(original, suspect, ComparisonMode.AUTOMATIC, AppConfig()))
    assert outcome.clusters == []


def test_a_real_change_beside_the_die_lines_is_still_found(tmp_path):
    original = _carton(tmp_path / "original.pdf")
    suspect = _carton(tmp_path / "suspect.pdf", die=GREEN, text="PARACETAMOL 250 mg")
    outcome = execute_request(ComparisonRequest(original, suspect, ComparisonMode.AUTOMATIC, AppConfig()))
    assert len(outcome.clusters) == 1


def test_the_die_line_is_still_shown(tmp_path):
    """Left out of the COMPARISON only: the reviewer still sees it on both files."""
    original = _carton(tmp_path / "original.pdf")
    outcome = execute_request(ComparisonRequest(original, original, ComparisonMode.AUTOMATIC, AppConfig()))
    image = outcome.aligned_original
    magenta = (image[..., 0].astype(int) > 180) & (image[..., 1].astype(int) < 120) & (image[..., 2].astype(int) > 100)
    assert magenta.sum() > 200
