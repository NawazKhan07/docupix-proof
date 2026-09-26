"""Shared builders for tests that need real PDFs."""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz


def make_carton_pdf(
    path: Path,
    dx: float = 0,
    dy: float = 0,
    rotation: float | None = None,
    page_rotation: int = 0,
    recolour: bool = False,
) -> Path:
    """A small carton-like vector page.

    Optionally shifted, rotated in vector space (`rotation`, degrees about the
    page centre), rotated as a page (`page_rotation`), or with its red block
    recoloured green so a comparison against the plain one has a finding.
    """
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    if page_rotation:
        page.set_rotation(page_rotation)
    morph = (fitz.Point(150, 100), fitz.Matrix(rotation)) if rotation else None
    block = (0.15, 0.6, 0.25) if recolour else (0.8, 0.1, 0.15)

    page.draw_rect(fitz.Rect(30 + dx, 30 + dy, 270 + dx, 170 + dy), color=(0, 0, 0), width=1.5, morph=morph)
    page.insert_text((40 + dx, 100 + dy), "PARACETAMOL", fontsize=26, morph=morph)
    page.insert_text((40 + dx, 125 + dy), "500 mg Tablets BP", fontsize=11, morph=morph)
    page.draw_rect(
        fitz.Rect(40 + dx, 135 + dy, 120 + dx, 160 + dy), color=block, fill=block, morph=morph
    )
    page.draw_circle(
        fitz.Point(210 + dx, 120 + dy), 22, color=(0.1, 0.35, 0.8), fill=(0.1, 0.35, 0.8), morph=morph
    )
    for index in range(10):
        page.draw_line(
            fitz.Point(150 + dx + index * 6, 60 + dy),
            fitz.Point(150 + dx + index * 6, 85 + dy),
            color=(0, 0, 0),
            width=2,
            morph=morph,
        )
    doc.save(path)
    doc.close()
    return path


def make_multi_panel_carton_pdf(path: Path, shift_second_panel: tuple[float, float] = (0.0, 0.0)) -> Path:
    """A carton-like vector page of TWO panels side by side, each in its own frame (they meet at the fold), each
    with its own real content -- `find_blocks` finds both. `shift_second_panel`, if given, shifts the whole right
    panel, frame and content (a fold not quite flat when this file was made), independent of the left panel: the
    vector equivalent of `test_block_alignment.py`'s `_locally_misaligned_suspect`.
    """
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    # Each panel its own frame: a panel not quite flat when the file was made moves with its outline, as a real
    # out-of-plane panel does -- its printing does not slide about inside a frame that stays put.
    page.draw_rect(fitz.Rect(20, 20, 200, 280), color=(0, 0, 0), width=1.5)
    dx, dy = shift_second_panel
    page.draw_rect(fitz.Rect(200 + dx, 20 + dy, 380 + dx, 280 + dy), color=(0, 0, 0), width=1.5)

    page.insert_text((35, 80), "PARACETAMOL", fontsize=14)
    page.insert_text((35, 100), "500 mg Tablets", fontsize=9)
    page.draw_rect(fitz.Rect(35, 120, 100, 150), color=(0.8, 0.1, 0.15), fill=(0.8, 0.1, 0.15))
    for index in range(8):
        page.draw_line(fitz.Point(35 + index * 6, 200), fitz.Point(35 + index * 6, 230), color=(0, 0, 0), width=1.5)

    page.insert_text((220 + dx, 80 + dy), "BATCH INFO")
    page.insert_text((220 + dx, 100 + dy), "EXP DATE 2027", fontsize=9)
    page.draw_circle(fitz.Point(300 + dx, 180 + dy), 25, color=(0.1, 0.35, 0.8), fill=(0.1, 0.35, 0.8))
    for index in range(8):
        page.draw_line(
            fitz.Point(220 + dx + index * 6, 220 + dy), fitz.Point(220 + dx + index * 6, 250 + dy), color=(0, 0, 0), width=1.5
        )

    doc.save(path)
    doc.close()
    return path
