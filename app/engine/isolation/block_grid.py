"""Dividing an artwork into its panels by the lines that fold and cut it: the blocks block-wise alignment aligns.

A carton net is a grid. Its fold lines run edge to edge across it, and every panel, flap and tab is one cell of
that grid, trimmed to the die-cut outline. So rather than hoping each panel happens to trace as its own closed
contour (a panel open to the outer edge, or with a logo drawn across its fold, never does), the grid's lines are
found and every cell inside the artwork is a block -- the whole artwork covered, side panels and flaps included.

The lines come from:

* **the file's own stroked vector paths**, for a PDF (`stroked_line_map`): a die line or fold line is a stroke,
  and text and filled colour bars are not, so a paragraph or a panel-wide bar can never pass for a fold;
* **steps in the artwork's silhouette**: wherever a flap or tab meets a panel, the outline's width or height
  changes abruptly (the flaps of a carton are narrower than the panels they fold from);
* **closed panels found by tracing** (`polygon_tracer.find_blocks`), whose edges are real lines, for a scan or an
  image with no vector paths.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from app.engine.isolation.pdf_paths import stroked_paths
from app.engine.isolation.polygon_tracer import ArtworkBlock, find_blocks
from app.models.diff import BoundingBox

logger = logging.getLogger(__name__)

_LINE_SHARE = 0.2  # a fold line runs at least this share of the artwork's width (or height)
_STEP_SHARE = 0.06  # a silhouette step changes the outline's width (or height) by at least this share
_EDGE_MARGIN_SHARE = 0.025  # a line or step this close to the artwork's own edge is that edge, not a fold
_MERGE_SHARE = 0.02  # lines closer than this share of the artwork's shorter side are the same fold
_MIN_MERGE_PX = 8
_INK_LEVEL = 230  # a pixel darker than this in its darkest channel is printed
_MIN_FILL = 0.4  # a grid cell is a block if at least this share of it is artwork...
_MIN_AREA_SHARE = 0.015  # ...and it is at least this share of the artwork
_POLYGON_EPSILON_PX = 2.0
_WHITE_ON_WHITE = 0.92  # a stroke lighter than this (0-1 grey) is invisible on the page: not a line
_EDGE_LINE_SHARE = 0.45  # two cells stay apart only if a line runs along this much of their shared edge (a dashed fold ~0.6)
_VECTOR_EDGE_LINE_SHARE = 0.8  # ...where the lines are a PDF's own strokes, drawn solid
_STROKE_SHARE = 0.9  # rows carrying at least this share of a line's longest run are the same stroke
_STROKE_GAP_PX = 4  # lines further apart than this are separate strokes (a stroke is a few pixels wide)
_DOMINANT_LINE = 1.5  # a line this many times longer than the others it is grouped with is the cut on its own
_EDGE_SLACK_PX = 6  # a line counts for an edge if it lies within this many pixels of it
_SIDE_PROBE_PX = 6  # how far either side of an edge to look for artwork on both sides
_THIN_LINE_CONTRAST = 25  # grey levels a thin structure must be darker than its surroundings, in a raster


def stroked_line_map(page, dpi: float, shape: tuple[int, int]) -> np.ndarray | None:
    """Every stroked vector path of a PDF page, in any colour, drawn at the image's own resolution (`shape`).

    Only strokes: text is not a path and a filled swatch or bar is not a line. None if the page has no visible
    strokes, or so many vector paths that reading them all is not worth it.
    """
    try:
        total, drawings = stroked_paths(page)
    except Exception:  # noqa: BLE001 -- a page that cannot be read for lines just has no lines to offer
        logger.exception("could not read the page's vector paths")
        return None
    if not drawings or total > 300_000:
        return None
    height, width = shape
    to_pixels = dpi / 72.0
    rotation = page.rotation_matrix if page.rotation else None
    canvas = np.zeros((height, width), np.uint8)

    def pixel(point) -> tuple[int, int]:
        if rotation is not None:
            point = point * rotation
        return int(round(point.x * to_pixels)), int(round(point.y * to_pixels))

    stroked = 0
    for path in drawings:
        if path.get("type") not in ("s", "fs"):
            continue
        colour = path.get("color")
        if not colour or sum(colour) / len(colour) > _WHITE_ON_WHITE:
            continue
        for item in path["items"]:
            kind = item[0]
            if kind == "l":
                cv2.line(canvas, pixel(item[1]), pixel(item[2]), 1, 2)
            elif kind == "re":
                rect = item[1]
                corners = [rect.tl, rect.tr, rect.br, rect.bl]
                cv2.polylines(canvas, [np.array([pixel(p) for p in corners], np.int32)], True, 1, 2)
            elif kind == "qu":
                quad = item[1]
                cv2.polylines(canvas, [np.array([pixel(p) for p in (quad.ul, quad.ur, quad.lr, quad.ll)], np.int32)], True, 1, 2)
        stroked += 1
    return canvas.astype(bool) if stroked else None


def _cluster(values: list[int], tolerance: int, exact: dict[int, int] | None = None) -> list[int]:
    """Values within `tolerance` of the previous one are the same cut: each group becomes one position -- where
    an `exact` value (a real line, rather than an outline's step: position -> how long a run it carries) is in the
    group, on the lines, since the cut is later checked for a line running along it; otherwise the group's mean.

    Lines in a group are averaged (a stroke a few pixels wide, a fold drawn twice) -- unless one is clearly the
    LONGEST (a fold across the whole panel beside the edge of a box drawn just below it, in one panel only): then the
    cut is on that line, at the centre of its stroke. Averaged with the box's edge, it would sit between the two where
    there is no line at all, fail its own check for a line along it, and the fold would be lost."""
    exact = dict(exact or {})
    groups: list[list[int]] = []
    for value in sorted(set(values) | set(exact)):
        if groups and value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    positions = []
    for group in groups:
        on_line = [value for value in group if value in exact]
        if not on_line:
            positions.append(int(round(sum(group) / len(group))))
            continue
        longest = max(exact[value] for value in on_line)
        runner_up = max([exact[value] for value in on_line if exact[value] < _STROKE_SHARE * longest] or [0])
        if runner_up and longest >= _DOMINANT_LINE * runner_up:
            on_line = [value for value in on_line if exact[value] >= _STROKE_SHARE * longest]  # the long line's own stroke
        # Two separate strokes in one group -- the two edges of a narrow gutter between panels, each a fold -- are not
        # averaged either: that sits between them, on no line. The cut is on one of them (the longer; the first, tied).
        strokes: list[list[int]] = []
        for value in sorted(on_line):
            if strokes and value - strokes[-1][-1] <= _STROKE_GAP_PX:
                strokes[-1].append(value)
            else:
                strokes.append([value])
        if len(strokes) > 1:
            on_line = max(strokes, key=lambda stroke: (max(exact[value] for value in stroke), -stroke[0]))
        positions.append(int(round(sum(on_line) / len(on_line))))
    return positions


def _long_lines(lines: np.ndarray, horizontal: bool, length: int) -> dict[int, int]:
    """Rows (or columns) of `lines` that carry a straight run at least `length` long, with how much of it they carry."""
    kernel = np.ones((1, length) if horizontal else (length, 1), np.uint8)
    runs = cv2.morphologyEx(lines.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    counts = runs.sum(axis=1 if horizontal else 0)
    return {int(index): int(counts[index]) for index in np.flatnonzero(counts >= length)}


def _steps(profile: np.ndarray, threshold: float) -> list[int]:
    """Where a silhouette's width (or height) profile jumps by at least `threshold`."""
    return [int(index) for index in np.flatnonzero(np.abs(np.diff(profile.astype(np.float64))) >= threshold)]


def _darkest(image: np.ndarray) -> np.ndarray:
    """Each pixel's darkest channel: ink of any colour is dark in at least one (OpenCV's, many times numpy's speed)."""
    if image.ndim != 3:
        return image
    channels = cv2.split(np.ascontiguousarray(image[:, :, :3]))
    return cv2.min(cv2.min(channels[0], channels[1]), channels[2])


_RASTER_FOLD_SHARE = 0.6  # a picture's straight line is a fold only if it runs this share of the artwork
_STRAIGHT_SHARE = 0.05  # a straight run at least this share of the artwork's shorter side is line-work, not a letter
_LINE_MAX_THICKNESS_PX = 8  # ...if it is no thicker than this across
_LINE_BACKGROUND_PX = 15  # a pixel's background: the median this wide round it


def _straight_lines_any_colour(image: np.ndarray, length: int) -> np.ndarray:
    """The long straight thin lines of a picture -- a file with no vector paths (a scan, a flattened PDF) -- of ANY
    colour, darker or lighter than what is round them: a fold line printed blue over a red panel is darker than the red
    in no 'darkest channel' sense, but a thin line in the red and green channels all the same. Only straight runs at
    least `length` long are kept (along or across): type, a pattern and a picture make thin strokes too, but short."""
    # the pixels that ARE ink: off their own background (the median round them) in some channel -- not a filter's fill of
    # the gaps between letters, which made a line of small type read as one long line
    channels = cv2.split(np.ascontiguousarray(image[:, :, :3])) if image.ndim == 3 else [image]
    thin = np.zeros(image.shape[:2], bool)
    for channel in channels:
        thin |= cv2.absdiff(channel, cv2.medianBlur(channel, _LINE_BACKGROUND_PX)) >= _THIN_LINE_CONTRAST
    thin = thin.astype(np.uint8)
    # long one way, and thin the other: a line of small type also reads as a long band of thin marks (the gaps between
    # its letters fill in), but a band as tall as its letters -- a fold is a few pixels across
    across = _LINE_MAX_THICKNESS_PX
    along = cv2.morphologyEx(thin, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))
    along &= 1 - cv2.morphologyEx(along, cv2.MORPH_OPEN, np.ones((across, 1), np.uint8))
    down = cv2.morphologyEx(thin, cv2.MORPH_OPEN, np.ones((length, 1), np.uint8))
    down &= 1 - cv2.morphologyEx(down, cv2.MORPH_OPEN, np.ones((1, across), np.uint8))
    return (along | down).astype(bool)


def _edge_has_line(
    evidence: np.ndarray, region: np.ndarray, vertical: bool, at: int, start: int, stop: int, share: float = _EDGE_LINE_SHARE
) -> bool:
    """Whether a line runs along most of the shared edge `at` (a column if `vertical`, else a row) from `start`
    to `stop` -- only where there is artwork on both sides of it."""
    lo, hi = max(0, at - _EDGE_SLACK_PX), at + _EDGE_SLACK_PX + 1
    if vertical:
        band = evidence[start:stop, lo:hi].any(axis=1)
        both_sides = region[start:stop, max(0, at - _SIDE_PROBE_PX)] & region[start:stop, min(region.shape[1] - 1, at + _SIDE_PROBE_PX)]
    else:
        band = evidence[lo:hi, start:stop].any(axis=0)
        both_sides = region[max(0, at - _SIDE_PROBE_PX), start:stop] & region[min(region.shape[0] - 1, at + _SIDE_PROBE_PX), start:stop]
    if not both_sides.any():
        return True  # no artwork on one side: nothing to join across
    return float(band[both_sides].mean()) >= share


def divide_into_blocks(
    image: np.ndarray,
    region: np.ndarray | None = None,
    line_map: np.ndarray | None = None,
    min_area_share: float = _MIN_AREA_SHARE,
) -> list[ArtworkBlock]:
    """The panels of the artwork inside `region` (the whole picture if None), as the cells of its fold-line grid.

    `line_map`, if given, is the file's own stroked vector paths (`stroked_line_map`) at `image`'s resolution.
    Every candidate cut is taken right across the artwork, which slices more finely than the carton really is
    (a flap's step runs only as far as the flap; a box drawn round a logo is not a fold at all), so neighbouring
    cells are then joined wherever no line actually runs along the edge between them: only a real fold or cut
    separates two blocks. Empty when there are no interior folds to divide by (a flat label: its one block is
    the whole artwork, which the whole-artwork alignment already is).
    """
    height, width = image.shape[:2]
    region = np.ones((height, width), bool) if region is None else region.astype(bool)
    region_u8 = region.view(np.uint8)
    region_area = cv2.countNonZero(region_u8)
    if region_area == 0:
        return []
    x_min, y_min, box_w, box_h = cv2.boundingRect(region_u8)
    x_max, y_max = x_min + box_w, y_min + box_h
    if box_w < 32 or box_h < 32:
        return []
    margin_x, margin_y = max(8, int(box_w * _EDGE_MARGIN_SHARE)), max(8, int(box_h * _EDGE_MARGIN_SHARE))
    has_vectors = line_map is not None and line_map.shape == region.shape
    straight = max(24, int(_STRAIGHT_SHARE * min(box_w, box_h)))
    if has_vectors:
        found = line_map
    else:
        # (looked for only in the box round the artwork, a few pixels over: on a proof sheet it can be a third of the page)
        bx0, by0 = max(0, x_min - _LINE_BACKGROUND_PX), max(0, y_min - _LINE_BACKGROUND_PX)
        bx1, by1 = min(width, x_max + _LINE_BACKGROUND_PX), min(height, y_max + _LINE_BACKGROUND_PX)
        found = np.zeros((height, width), bool)
        found[by0:by1, bx0:bx1] = _straight_lines_any_colour(image[by0:by1, bx0:bx1], straight)
    evidence = found & cv2.dilate(region.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

    # Where a flap or tab meets a panel, the outline's width (or height) steps.
    cuts_y = _steps(region.sum(axis=1), _STEP_SHARE * box_w)
    cuts_x = _steps(region.sum(axis=0), _STEP_SHARE * box_h)
    lines_x: dict[int, int] = {}
    lines_y: dict[int, int] = {}
    if has_vectors:
        # The file's own fold and cut lines, as far as they run inside the artwork.
        lines_y = _long_lines(evidence, horizontal=True, length=max(24, int(_LINE_SHARE * box_w)))
        lines_x = _long_lines(evidence, horizontal=False, length=max(24, int(_LINE_SHARE * box_h)))
    else:
        # Panels closed by lines all round: the surest evidence of a fold a scan or a plain image can give...
        for block in find_blocks(image, region):
            for x in (int(block.bbox.x0), int(block.bbox.x1)):
                lines_x[x] = max(lines_x.get(x, 0), int(block.bbox.height))
            for y in (int(block.bbox.y0), int(block.bbox.y1)):
                lines_y[y] = max(lines_y.get(y, 0), int(block.bbox.width))
        # ...and, where they found none, the picture's own long straight lines -- folds of any colour (a blue fold over a
        # red panel closes no panel 'darker' than it), and folds where a panel is not closed all round
        near = max(_MIN_MERGE_PX, int(_MERGE_SHARE * min(box_w, box_h)))
        # (only lines across most of the artwork: a fold runs the carton's height, where a rule dividing a label's
        # columns, or a table's, runs a part of one panel -- taken for a fold, it split the panel and broke what is on it)
        for found, horizontal, span in ((lines_y, True, box_w), (lines_x, False, box_h)):
            for at, run in _long_lines(evidence, horizontal=horizontal, length=max(24, int(_RASTER_FOLD_SHARE * span))).items():
                if not any(abs(at - known) <= near for known in found):
                    found[at] = run

    tolerance = max(_MIN_MERGE_PX, int(_MERGE_SHARE * min(box_w, box_h)))
    # A PDF's fold lines are drawn solid (a dashed stroke is drawn whole here): a real fold runs along nearly all of
    # the edge between two panels, where a box drawn in a panel covers only part of it. A scan's lines are broken.
    share = _VECTOR_EDGE_LINE_SHARE if has_vectors else _EDGE_LINE_SHARE
    grid_x = [x_min] + [x for x in _cluster(cuts_x, tolerance, lines_x) if x_min + margin_x < x < x_max - margin_x] + [x_max]
    grid_y = [y_min] + [y for y in _cluster(cuts_y, tolerance, lines_y) if y_min + margin_y < y < y_max - margin_y] + [y_max]
    rows, columns = len(grid_y) - 1, len(grid_x) - 1

    # Every cell with any artwork in it, joined to its neighbour wherever no line separates them.
    occupied = {
        (i, j)
        for i in range(rows)
        for j in range(columns)
        if region[grid_y[i] : grid_y[i + 1], grid_x[j] : grid_x[j + 1]].any()
    }
    parent = {cell: cell for cell in occupied}

    def root(cell):
        while parent[cell] != cell:
            parent[cell] = parent[parent[cell]]
            cell = parent[cell]
        return cell

    for i, j in occupied:
        if (i, j + 1) in occupied and not _edge_has_line(evidence, region, True, grid_x[j + 1], grid_y[i], grid_y[i + 1], share):
            parent[root((i, j + 1))] = root((i, j))
        if (i + 1, j) in occupied and not _edge_has_line(evidence, region, False, grid_y[i + 1], grid_x[j], grid_x[j + 1], share):
            parent[root((i + 1, j))] = root((i, j))

    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in occupied:
        groups.setdefault(root(cell), []).append(cell)

    blocks: list[ArtworkBlock] = []
    min_area = min_area_share * region_area
    for cells in groups.values():
        # Each block's mask is made over just its own box (a page can be 100 megapixels).
        gx0 = min(grid_x[j] for _i, j in cells)
        gx1 = max(grid_x[j + 1] for _i, j in cells)
        gy0 = min(grid_y[i] for i, _j in cells)
        gy1 = max(grid_y[i + 1] for i, _j in cells)
        mask = np.zeros((gy1 - gy0, gx1 - gx0), np.uint8)
        for i, j in cells:
            mask[grid_y[i] - gy0 : grid_y[i + 1] - gy0, grid_x[j] - gx0 : grid_x[j + 1] - gx0] = 1
        mask &= region[gy0:gy1, gx0:gx1].astype(np.uint8)
        if mask.sum() < min_area:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < _MIN_FILL * w * h:
            continue  # mostly empty: the thin arms of a flap outline, not a panel
        ink = _darkest(image[gy0:gy1, gx0:gx1]) < _INK_LEVEL
        inside = cv2.erode(
            mask, np.ones((2 * _SIDE_PROBE_PX + 1,) * 2, np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0
        ).astype(bool)
        if not (ink & inside).any():
            continue  # nothing printed inside its edges (the margin outside a border line): nothing to align
        outline = cv2.approxPolyDP(contour, _POLYGON_EPSILON_PX, True)
        polygon = [(float(p[0][0] + gx0), float(p[0][1] + gy0)) for p in outline]
        if len(polygon) >= 3:
            blocks.append(ArtworkBlock(polygon=polygon, bbox=BoundingBox(float(x + gx0), float(y + gy0), float(x + gx0 + w), float(y + gy0 + h))))

    if len(blocks) < 2:
        return []
    blocks.sort(key=lambda block: (block.bbox.y0, block.bbox.x0))
    logger.info("divided the artwork into blocks", extra={"blocks": len(blocks), "columns": columns, "rows": rows})
    return blocks
