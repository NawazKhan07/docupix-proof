"""Lifting the area inside a closed border line -- a die-cut outline.

Packaging artwork arrives as a *sheet*: the carton's die-cut outline (a thin closed
line), the printed panels inside it, and, around it, everything that is not the
artwork -- a title, a colour legend, crop marks, registration dots, a leader line
with a label. Only what is inside the outline is to be compared.

Colour cannot find that area: a carton's flaps are unprinted, the colour of the
paper, and only the outline separates them from the page. So this asks a different
question -- **what is sealed inside a closed line?**

* The lines (walls) are found: the dark pixels of the rendered page, or, for a PDF,
  every stroked vector path whatever its colour (a dieline is often a spot colour).
* Everything reachable from the edge of the page without crossing a wall is
  "outside". What is left -- the walls and the cells they seal in -- is split into
  connected pieces. The carton is one piece (its outline, its fold lines and every
  panel they enclose); the legend, the text and each crop mark are others.
* The piece under the press is the artwork. Nothing has to be traced, and the fold
  lines inside do not matter.

Refinements, each from a way real sheets differ from the ideal:

* **A small gap in the outline** (a dash, an anti-aliasing break) would let "outside"
  flow in and the carton would come back as a single panel, so walls are thickened
  by 0-5 px until the enclosure stops growing.
* **A leader line or dimension line touching the outline** is part of the same
  piece, and so is the label at its end. Thin protrusions are shaved off, then the
  outline's own corners are restored.
* **A frame around the whole sheet** would seal the whole page as one piece. When
  the piece is nearly the whole page, the frame is peeled away and the search
  repeats inside it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import SubjectConfig
from app.engine.isolation.mask_ops import component_at, disc_kernel, fill_holes, touches_edge

logger = logging.getLogger(__name__)

Point = tuple[float, float]

_BRIDGES_PX = (0, 1, 2, 3, 5)  # wall thickening tried, in working pixels
_GROWTH_TO_KEEP_GOING = 1.08  # a thicker wall must seal in this much more area to be preferred
_MAX_DRAWINGS = 300_000  # a page with more vector paths than this is not scanned for lines
_CURVE_SAMPLES = 8


@dataclass
class BorderRegion:
    mask: np.ndarray  # bool (H, W) at the image's own resolution
    source: str  # "vector lines" | "dark lines"


# ---------------------------------------------------------------------------
# Finding the lines
# ---------------------------------------------------------------------------


def _working_scale(height: int, width: int, long_edge: int) -> tuple[float, tuple[int, int]]:
    scale = min(1.0, long_edge / max(height, width))
    return scale, (max(2, round(width * scale)), max(2, round(height * scale)))


def raster_walls(image_rgb: np.ndarray, config: SubjectConfig) -> tuple[np.ndarray, float] | None:
    """(dark-line mask at working size, scale), or None if the page is not a light sheet.

    Lines are the pixels darker than most of the sheet is light. Reducing the page by
    averaging would fade a one-pixel line to nothing, so each working pixel takes the
    DARKEST pixel of the block it stands for.
    """
    height, width = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    band = max(2, int(round(0.03 * min(height, width))))
    edge = np.concatenate([gray[:band].ravel(), gray[-band:].ravel(), gray[:, :band].ravel(), gray[:, -band:].ravel()])
    page_level = float(np.median(edge))
    if page_level < config.border_min_page_brightness:
        return None  # a photograph or a dark sheet: lines cannot be told from the surface
    level = min(config.border_dark_level, 0.5 * page_level)

    scale, size = _working_scale(height, width, config.border_working_long_edge_px)
    if scale < 1.0:
        block = math.ceil(1.0 / scale)
        gray = cv2.resize(cv2.erode(gray, np.ones((block, block), np.uint8)), size, interpolation=cv2.INTER_NEAREST)
    return gray < level, scale


def _bezier(points: list[Point]) -> list[Point]:
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = points
    samples = []
    for i in range(_CURVE_SAMPLES + 1):
        t = i / _CURVE_SAMPLES
        u = 1 - t
        samples.append((
            u**3 * x0 + 3 * u * u * t * x1 + 3 * u * t * t * x2 + t**3 * x3,
            u**3 * y0 + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t**3 * y3,
        ))
    return samples


def vector_walls(page, dpi: float, size: tuple[int, int], scale: float) -> np.ndarray | None:
    """Every stroked vector path of a PDF page, in any colour, drawn as lines at working size.

    Only strokes count: a filled swatch is not a line, and text is not a path. A page
    that has no visible strokes gives None.
    """
    drawings = page.get_drawings()
    if not drawings or len(drawings) > _MAX_DRAWINGS:
        return None
    to_pixels = dpi / 72.0 * scale
    canvas = np.zeros((size[1], size[0]), np.uint8)
    stroked = 0

    def pixel(point) -> tuple[int, int]:
        return int(round(point[0] * to_pixels)), int(round(point[1] * to_pixels))

    for path in drawings:
        if path.get("type") not in ("s", "fs"):
            continue
        colour = path.get("color")
        if not colour or sum(colour) / len(colour) > 0.92:
            continue  # unstroked, or white on white
        thickness = max(1, int(round((path.get("width") or 0) * to_pixels)))
        for item in path["items"]:
            kind = item[0]
            if kind == "l":
                cv2.line(canvas, pixel(item[1]), pixel(item[2]), 1, thickness)
            elif kind == "re":
                rect = item[1]
                cv2.rectangle(canvas, pixel((rect.x0, rect.y0)), pixel((rect.x1, rect.y1)), 1, thickness)
            elif kind == "qu":
                quad = item[1]
                corners = np.array([pixel(p) for p in (quad.ul, quad.ur, quad.lr, quad.ll)], np.int32)
                cv2.polylines(canvas, [corners], True, 1, thickness)
            elif kind == "c":
                curve = np.array([pixel(p) for p in _bezier([(p.x, p.y) for p in item[1:5]])], np.int32)
                cv2.polylines(canvas, [curve], False, 1, thickness)
        stroked += 1
    return (canvas > 0) if stroked else None


# ---------------------------------------------------------------------------
# What a set of lines seals in
# ---------------------------------------------------------------------------


def _sealed_pieces(walls: np.ndarray, bridge: int) -> tuple[np.ndarray, int]:
    """Label the connected pieces of (walls + the cells they seal in).

    "Outside" is every empty cell joined, without crossing a wall, to the page edge; it is
    label 0 in the result.
    """
    solid = walls.astype(np.uint8)
    if bridge > 0:
        solid = cv2.dilate(solid, disc_kernel(bridge))
    _, cells = cv2.connectedComponents((solid == 0).astype(np.uint8), connectivity=4)

    reaches_outside = set(np.unique(np.concatenate([cells[0], cells[-1], cells[:, 0], cells[:, -1]])).tolist())
    reaches_outside.discard(0)
    is_outside = np.zeros(int(cells.max()) + 1, bool)
    is_outside[list(reaches_outside)] = True
    outside = is_outside[cells]

    count, labels = cv2.connectedComponents((~outside).astype(np.uint8), connectivity=8)
    return labels, count


def _pick(pieces: np.ndarray, count: int, seed: Point | None) -> np.ndarray | None:
    """The piece under `seed`, or, with no seed, the largest one that does not run off the page."""
    if seed is not None:
        x = min(max(int(round(seed[0])), 0), pieces.shape[1] - 1)
        y = min(max(int(round(seed[1])), 0), pieces.shape[0] - 1)
        label = pieces[y, x]
        return (pieces == label) if label > 0 else None

    areas = np.bincount(pieces.ravel(), minlength=count)
    areas[0] = 0
    on_edge = np.unique(np.concatenate([pieces[0], pieces[-1], pieces[:, 0], pieces[:, -1]]))
    areas[on_edge] = 0
    if not areas.any():
        return None
    return pieces == int(areas.argmax())


def _shave(region: np.ndarray, config: SubjectConfig, seed: Point | None) -> np.ndarray:
    """Remove thin protrusions (leader lines, dimension lines) but keep the outline's own corners."""
    radius = max(1, round(config.border_shave_fraction * max(region.shape)))
    solid = region.astype(np.uint8)
    opened = cv2.morphologyEx(solid, cv2.MORPH_OPEN, disc_kernel(radius))
    restored = solid.astype(bool) & cv2.dilate(opened, disc_kernel(radius + 1)).astype(bool)
    if seed is not None and not restored[min(max(int(seed[1]), 0), restored.shape[0] - 1), min(max(int(seed[0]), 0), restored.shape[1] - 1)]:
        return region  # shaving would lose the press point: leave it
    return restored


def _region(walls: np.ndarray, seed: Point | None, config: SubjectConfig, minimum: float) -> np.ndarray | None:
    """The artwork's region at working size, or None. Grows the wall thickness until the enclosure stops growing."""
    total = walls.size
    best: np.ndarray | None = None

    for bridge in _BRIDGES_PX:
        current = walls
        piece = None
        for _ in range(2):  # the second pass looks inside a sheet-wide frame
            pieces, count = _sealed_pieces(current, bridge)
            piece = _pick(pieces, count, seed)
            if piece is None or piece.sum() < config.border_frame_fraction * total:
                break
            # Nearly the whole sheet is sealed: a frame around it. Take the frame's own line
            # away, and the space between it and the page edge joins the outside.
            thickness = max(3, bridge + 4)
            ring = piece & ~cv2.erode(piece.astype(np.uint8), disc_kernel(thickness)).astype(bool)
            current = current & ~ring
        if piece is None or piece.sum() >= config.border_frame_fraction * total:
            continue

        if bridge > 0:
            piece = cv2.erode(piece.astype(np.uint8), disc_kernel(bridge)).astype(bool)
        piece = fill_holes(piece)
        if piece.sum() < minimum * total:
            continue
        if best is None or piece.sum() > _GROWTH_TO_KEEP_GOING * best.sum():
            best = piece
        else:
            break  # thicker walls seal in no more: the outline was already closed
    return _shave(best, config, seed) if best is not None else None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def border_region(
    image_rgb: np.ndarray,
    seed: Point | None,
    config: SubjectConfig,
    page=None,
    dpi: float | None = None,
    strict: bool = True,
) -> BorderRegion | None:
    """The region inside the closed border line under `seed`, or None if there is no such border.

    `strict` (automatic mode) only accepts a region that is a sizeable part of a light
    sheet, so a text box on a photograph is not taken for the artwork; otherwise any
    closed border of at least half a percent of the page will do. `page` and `dpi`, for a
    rendered PDF, let its vector strokes be used as the lines.
    """
    height, width = image_rgb.shape[:2]
    minimum = config.border_min_area_fraction if strict else config.border_min_area_fraction_forced
    scale, size = _working_scale(height, width, config.border_working_long_edge_px)
    seed_small = (seed[0] * scale, seed[1] * scale) if seed is not None else None

    candidates: list[tuple[np.ndarray, str]] = []
    if page is not None and dpi:
        vector = vector_walls(page, dpi, size, scale)
        if vector is not None:
            candidates.append((vector, "vector lines"))
    raster = raster_walls(image_rgb, config)
    if raster is not None:
        candidates.append((raster[0], "dark lines"))

    for walls, source in candidates:
        region = _region(walls, seed_small, config, minimum)
        if region is None:
            continue
        soft = cv2.GaussianBlur(region.astype(np.float32), (0, 0), 0.8) if scale < 1.0 else region.astype(np.float32)
        mask = cv2.resize(soft, (width, height), interpolation=cv2.INTER_LINEAR) > 0.5 if scale < 1.0 else region
        logger.info("lifted the area inside a closed border", extra={"source": source, "area_fraction": round(float(mask.sum()) / mask.size, 4)})
        return BorderRegion(mask, source)
    return None
