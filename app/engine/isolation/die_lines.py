"""A carton's die lines -- its cut, crease and perforation lines -- found so they are never compared.

The die line is not the artwork: it is drawn over it for the printer's cutter, in a spot colour of its own, and a print
never carries it. Two versions of a file differ in it all the time -- one has it and the other not, one draws it solid
and the other dashed, or in another colour -- and every such line would otherwise be a finding. So each file's die
lines are found (`die_line_mask`) and left out of what is compared, in both files, whatever the mode.

The die line is found from the file itself -- it is what draws the artwork's own outline:

* in a PDF, the stroked paths lying along the outline ARE the die line (whatever their colour); their colour, if it is a
  definite spot colour, finds the rest of it -- the folds and creases inside -- as every other path in that colour
  (`vector_strokes`: exact, dashed, curved or at an angle as drawn). An outline that is the edge of a filled design (a
  sachet printed to its edge) has no strokes along it, and no die line is taken from it;
* in a picture (a scan, an image export), the ink along the outline must be a thin line in a spot colour, and then only
  long line-work in that colour is a die line -- never anything the size of a letter.

A path of any colour that runs along a die line found (`strokes_along`) is part of it too: the same line drawn in
another colour in the other file, or a glue or safety line hugging the cut.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from app.engine.isolation.pdf_paths import stroked_paths

logger = logging.getLogger(__name__)

_RING_PX = 10  # the die line is looked for this close to either side of the artwork's outline
_MIN_CHROMA = 30.0  # a die line is drawn in a spot colour: a black or grey outline says nothing
_SAME_COLOUR = 25.0  # (CIE76) ink this close to the die colour is drawn in it
_MIN_SHARE = 0.5  # the outline is the die colour if this share of the ink along it is one colour
_PAPER_DISTANCE = 20.0  # (CIE76) a pixel this far from white is ink
_MIN_TINT = 0.15  # a tint of the die colour at least this strong is its ink (a thin line's anti-aliased edge)
_STROKE_MARGIN_PX = 3  # a die stroke is left out this far either side of its drawn width (the other file's may sit a hair off)
_RASTER_SOLID_PX = 7  # in a picture, die-coloured ink that survives an opening this wide is a solid area, not a line
_RASTER_MIN_EXTENT_PX = 150  # ...and line-work shorter than this either way is not a die line (a letter, a mark)
_RASTER_MAX_RING_INK = 0.45  # in a picture, an outline whose band is more ink than this is a filled edge, not a line
_ON_RING_SHARE = 0.6  # a stroked path with this much of its length along the outline is (part of) the die line
_RING_STROKES_MIN = 0.15  # ...and the die line is found if such paths run along this share of the outline at least
_MIN_STROKE_PX = 40  # (paths shorter than this are not looked at)
_FILL_MARGIN_PX = 2  # a die line is filled this much wider than it was found
_CURVE_SAMPLES = 16
_MAX_DRAWINGS = 300_000


def lab(pixels: np.ndarray) -> np.ndarray:
    """RGB pixels (any shape, last axis 3) as CIE Lab rows (L 0..100, a, b centred on 0)."""
    return cv2.cvtColor(np.asarray(pixels).reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32) * np.array(
        [100 / 255, 1, 1], np.float32
    ) - np.array([0, 128, 128], np.float32)


def _ring(boundary: np.ndarray) -> np.ndarray:
    kernel = np.ones((2 * _RING_PX + 1, 2 * _RING_PX + 1), np.uint8)
    solid = boundary.astype(np.uint8)
    return (cv2.dilate(solid, kernel) > 0) & (cv2.erode(solid, kernel) == 0)


def die_colour(image: np.ndarray, boundary: np.ndarray | None) -> np.ndarray | None:
    """The spot colour (CIE Lab) a file's outline is drawn in, read from its PICTURE: the ink along the outline, if it is
    a thin line (not a design filled to its edge) mostly of one definite colour -- None otherwise."""
    if boundary is None:
        return None
    ring = _ring(boundary)
    pixels = image[ring]
    if len(pixels) > 200_000:
        pixels = pixels[:: len(pixels) // 200_000]
    colours = lab(pixels)
    is_ink = np.linalg.norm(colours - np.array([100, 0, 0], np.float32), axis=1) > _PAPER_DISTANCE
    if is_ink.mean() > _RASTER_MAX_RING_INK:
        return None  # the band round the outline is mostly ink: the edge of a filled design, not a line drawn on paper
    inked = colours[is_ink]
    if len(inked) < 100:
        return None
    colour = np.median(inked, axis=0)
    if float(np.hypot(colour[1], colour[2])) < _MIN_CHROMA:
        return None
    if float((np.linalg.norm(inked - colour, axis=1) < _SAME_COLOUR).mean()) < _MIN_SHARE:
        return None
    return colour


def _drawn(stroke: "Stroke", shape: tuple[int, int], thickness: int = 1) -> tuple[np.ndarray, int, int] | None:
    """A stroke drawn on a canvas just its own size: (canvas, x0, y0), or None if it lies outside the picture."""
    height, width = shape
    points = np.concatenate(stroke.lines)
    x0, y0 = max(0, int(points[:, 0].min()) - thickness - 1), max(0, int(points[:, 1].min()) - thickness - 1)
    x1, y1 = min(width, int(points[:, 0].max()) + thickness + 2), min(height, int(points[:, 1].max()) + thickness + 2)
    if x1 <= x0 or y1 <= y0:
        return None
    canvas = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.polylines(canvas, [line - np.array([x0, y0], np.int32) for line in stroke.lines], False, 1, thickness)
    return canvas, x0, y0


def _outline_strokes(strokes: list["Stroke"], boundary: np.ndarray) -> list["Stroke"]:
    """The stroked paths that draw the artwork's outline (each mostly along it) -- the die line -- or none, if they do
    not run along enough of it to be what draws it."""
    ring = _ring(boundary)
    on_ring, along = [], 0
    for stroke in strokes:
        drawn = _drawn(stroke, ring.shape)
        if drawn is None:
            continue
        canvas, x0, y0 = drawn
        length = int(canvas.sum())
        if length < _MIN_STROKE_PX:
            continue
        inside = int((canvas.astype(bool) & ring[y0 : y0 + canvas.shape[0], x0 : x0 + canvas.shape[1]]).sum())
        if inside >= _ON_RING_SHARE * length:
            on_ring.append(stroke)
            along += inside
    perimeter = int(ring.sum()) / (2 * _RING_PX + 1)
    return on_ring if perimeter > 0 and along >= _RING_STROKES_MIN * perimeter else []


def die_tinted(colours: np.ndarray, colour: np.ndarray) -> np.ndarray:
    """Which of these (CIE Lab) colours are ink of the die colour -- it, or a tint of it with the paper: near the line
    from white to the die colour, and not too pale."""
    white = np.array([100, 0, 0], np.float32)
    along = colour - white
    t = np.clip(((colours - white) @ along) / float(along @ along), 0.0, 1.0)
    nearest = white + t[:, None] * along
    return (np.linalg.norm(colours - nearest, axis=1) < _SAME_COLOUR) & (t >= _MIN_TINT)


@dataclass(frozen=True)
class Stroke:
    """One stroked path of a PDF page, in the rendered picture's pixels."""

    colour: tuple[float, float, float]  # RGB 0..255
    width_px: float
    lines: tuple[np.ndarray, ...]  # polylines, each N x 2 int32 (closed ones repeat their first point)


def vector_strokes(page, dpi: float) -> list[Stroke] | None:
    """Every stroked path of a PDF page, with its colour and width, in pixels at `dpi` (None if it has none, or so many
    that reading them all is not worth it)."""
    try:
        total, drawings = stroked_paths(page)
    except Exception:  # noqa: BLE001 -- a page whose paths cannot be read has none to offer
        logger.exception("could not read the page's vector paths")
        return None
    if not drawings or total > _MAX_DRAWINGS:
        return None
    to_pixels = dpi / 72.0
    rotation = page.rotation_matrix if page.rotation else None

    def pixel(point) -> tuple[float, float]:
        if rotation is not None:
            point = point * rotation
        return point.x * to_pixels, point.y * to_pixels

    strokes = []
    for path in drawings:
        if path.get("type") not in ("s", "fs") or not path.get("color"):
            continue
        lines = []
        for item in path["items"]:
            kind = item[0]
            if kind == "l":
                points = [pixel(item[1]), pixel(item[2])]
            elif kind == "re":
                rect = item[1]
                points = [pixel(p) for p in (rect.tl, rect.tr, rect.br, rect.bl, rect.tl)]
            elif kind == "qu":
                quad = item[1]
                points = [pixel(p) for p in (quad.ul, quad.ur, quad.lr, quad.ll, quad.ul)]
            elif kind == "c":
                ends = [pixel(p) for p in item[1:5]]
                points = [_bezier(ends, i / _CURVE_SAMPLES) for i in range(_CURVE_SAMPLES + 1)]
            else:
                continue
            lines.append(np.round(np.array(points)).astype(np.int32))
        if lines:
            colour = tuple(float(c) * 255.0 for c in path["color"][:3])
            strokes.append(Stroke(colour, float(path.get("width") or 0.0) * to_pixels, tuple(lines)))
    return strokes or None


def _bezier(points: list[tuple[float, float]], t: float) -> tuple[float, float]:
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = points
    u = 1 - t
    return (
        u**3 * x0 + 3 * u * u * t * x1 + 3 * u * t * t * x2 + t**3 * x3,
        u**3 * y0 + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t**3 * y3,
    )


def die_line_mask(image: np.ndarray, boundary: np.ndarray | None, strokes: list[Stroke] | None = None) -> np.ndarray | None:
    """Where a file's die lines are (boolean, the picture's own pixels), to be left out of the comparison -- None if it
    has none that can be told (see the module's account)."""
    if boundary is None:
        return None
    height, width = image.shape[:2]
    if strokes is not None:  # a PDF: from its paths only
        outline = _outline_strokes(strokes, boundary)
        if not outline:
            return None
        mask = np.zeros((height, width), np.uint8)
        chosen = list(outline)
        # the folds and creases: every other path in the outline's colour, if that is a definite spot colour
        outline_colours = lab(np.array([stroke.colour for stroke in outline], np.float32))
        colour = np.median(outline_colours, axis=0)
        if float(np.hypot(colour[1], colour[2])) >= _MIN_CHROMA:
            colours = lab(np.array([stroke.colour for stroke in strokes], np.float32))
            same = np.linalg.norm(colours - colour, axis=1) < _SAME_COLOUR
            chosen += [stroke for stroke, match in zip(strokes, same) if match]
        for stroke in chosen:
            thickness = max(1, int(round(stroke.width_px)) + 2 * _STROKE_MARGIN_PX)
            cv2.polylines(mask, list(stroke.lines), False, 1, thickness)
        return mask.astype(bool)
    # a picture: the long, thin line-work in the outline's colour
    colour = die_colour(image, boundary)
    if colour is None:
        return None
    tinted = die_tinted(lab(image), colour).reshape(height, width).astype(np.uint8)
    solid = cv2.morphologyEx(tinted, cv2.MORPH_OPEN, np.ones((_RASTER_SOLID_PX, _RASTER_SOLID_PX), np.uint8))
    lines = tinted & (cv2.dilate(solid, np.ones((3, 3), np.uint8)) == 0)
    count, labels, stats, _centres = cv2.connectedComponentsWithStats(lines.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, bool)
    extent = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT])
    keep[1:] = extent >= _RASTER_MIN_EXTENT_PX
    lines = keep[labels]
    if not lines.any():
        return None
    return cv2.dilate(lines.astype(np.uint8), np.ones((2 * _STROKE_MARGIN_PX + 1,) * 2, np.uint8)).astype(bool)


def drawn_out(image: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """`image` with the pixels under `lines` filled from the nearest pixel beside them: a die line taken out of the
    picture, the band or panel it crossed continuing through where it was."""
    if not lines.any():
        return image
    # (a little wider than the line: the pixels just beside it still carry its anti-aliased halo, and a band filled
    # from them would keep a faint seam where the line was)
    size = 2 * _FILL_MARGIN_PX + 1
    lines = cv2.dilate(lines.astype(np.uint8), np.ones((size, size), np.uint8)).astype(bool)
    _distance, labels = cv2.distanceTransformWithLabels(
        lines.astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    # (the distance is to the nearest ZERO of the input -- a pixel outside the lines -- and the labels number those
    # pixels in raster order, from 1)
    outside_rows, outside_columns = np.nonzero(~lines)
    if len(outside_rows) == 0:
        return image
    rows, columns = np.nonzero(lines)
    nearest = labels[rows, columns] - 1
    filled = image.copy()
    filled[rows, columns] = image[outside_rows[nearest], outside_columns[nearest]]
    return filled


_ALONG_SHARE = 0.6  # a stroked path lying this much along the die lines found is a die line itself (drawn in another colour)
_ALONG_MIN_LENGTH_PX = 60  # ...if it is at least this long (a short mark beside a fold is artwork)
_ALONG_REACH_PX = 16  # (~1.3 mm at 300 DPI) a path this close to a die line runs along it


def strokes_along(strokes: list[Stroke] | None, die: np.ndarray) -> np.ndarray | None:
    """The stroked paths (of any colour) that run mostly along `die`, on it or hugging it -- where one file draws its die
    line in its die colour and the other the same line in another, or a glue or safety line beside it --
    drawn as a mask of the picture's size; None if there are none."""
    if not strokes:
        return None
    height, width = die.shape
    found = np.zeros((height, width), np.uint8)
    # (along it: on it, or hugging it -- a glue or safety line a millimetre inside the cut is the cutter's, not the
    # artwork's: printed artwork keeps well clear of the cut)
    size = 2 * _ALONG_REACH_PX + 1
    near = cv2.dilate(die.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))).astype(bool)
    for stroke in strokes:
        drawn = _drawn(stroke, (height, width))
        if drawn is None:
            continue
        canvas, x0, y0 = drawn
        length = int(canvas.sum())
        if length < _ALONG_MIN_LENGTH_PX:
            continue
        along = int((canvas.astype(bool) & near[y0 : y0 + canvas.shape[0], x0 : x0 + canvas.shape[1]]).sum())
        if along >= _ALONG_SHARE * length:
            thickness = max(1, int(round(stroke.width_px)) + 2 * _STROKE_MARGIN_PX)
            cv2.polylines(found, list(stroke.lines), False, 1, thickness)
    return found.astype(bool) if found.any() else None
