"""Shared builders for tests that need real PDFs."""

from __future__ import annotations

from pathlib import Path

import fitz


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
