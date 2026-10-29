"""Comparing one artwork against every repeat of it on a sheet.

The reviewer crops or lifts the ONE artwork in the original file, exactly as for any other comparison, and leaves the
suspect file as it is: a sheet printed with the same artwork over and over (a press sheet, a multi-up proof). Every
repeat is found on it (`find_repeats`) and each is compared against the original on its own (`compare_series`) --
alignment, every detector, clustering, all of it -- so a repeat that is itself imperfectly printed is judged on its
own pixels, not blamed on its neighbours.

The findings all come back as ONE list, each saying which repeat it is in. The suspect pane shows the whole sheet, so
a finding sits where its repeat actually is; the original pane shows the original's own whole page (as every other
mode's does), so several repeats' findings can land on the same spot of it -- expected, since they are all about the
same master artwork.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Callable

import cv2
import numpy as np

from app.config import AppConfig, SeriesConfig
from app.engine.detectors.regions import padded_window
from app.engine.isolation.boundary_mask import apply_mask, resolve_keep_mask
from app.engine.worker import ComparisonOutcome, run_comparison
from app.logging_config import StageTimer
from app.models.diff import BoundingBox, DiffCluster
from app.models.isolation import IsolationGeometry, NormalizedPolygon, NormalizedRect
from app.models.job import ComparisonMode

logger = logging.getLogger(__name__)


class NoRepeatsFoundError(Exception):
    """No repeat of the original's artwork could be found on the suspect sheet."""


def _edge_picture(rgb: np.ndarray) -> np.ndarray:
    """The edges of a picture, softened, as a float image (what is printed, whatever its colours)."""
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    edges = cv2.Canny(cv2.GaussianBlur(grey, (0, 0), 1.0), 40, 120)
    return cv2.GaussianBlur(edges.astype(np.float32), (0, 0), 1.5)


_MIN_LOCAL_EDGE_STD = 3.0  # a window of the target whose edges vary by less than this is blank paper: matched nowhere


def _blank_windows(target: np.ndarray, kh: int, kw: int) -> np.ndarray:
    """Where a `kh` x `kw` window of `target`, at each of a `matchTemplate(target, template)`-shaped map's positions,
    is too flat to mean anything: normalised correlation over a nearly constant window is numerically unstable and can
    read as a near-perfect match anywhere blank (a wide margin, a gap between repeats) once the template is shrunk
    small enough -- exactly the accident a search across scales is prone to finding, so it must be ruled out, not just
    hoped past."""
    mean = cv2.boxFilter(target, -1, (kw, kh), borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(target * target, -1, (kw, kh), borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.clip(mean_sq - mean * mean, 0, None))
    # `matchTemplate`'s output at (y, x) is the window target[y:y+kh, x:x+kw]; the centred box filter's value at
    # (y + kh//2, x + kw//2) is that same window's, near enough for a threshold mask.
    out_h, out_w = target.shape[0] - kh + 1, target.shape[1] - kw + 1
    return std[kh // 2 : kh // 2 + out_h, kw // 2 : kw // 2 + out_w] < _MIN_LOCAL_EDGE_STD


_OVERHANG_SHARE = 0.15  # a copy may run off the sheet by up to this share of its size each way (a reel as wide as it)...
_MIN_INSIDE_SHARE = 0.85  # ...as long as at least this share of it is on the sheet (half a copy at a cut end is not one)


def _peak(target: np.ndarray, template: np.ndarray) -> tuple[float, float, tuple[int, int], np.ndarray, tuple[int, int]]:
    """(score, how far it stands above the correlation's own background, the best fit's top-left in `target`'s pixels
    (it may be a little outside), the correlation map with everything ruled out set to the background, and the map's
    (x, y) offset from `target`'s pixels).

    The background is the map's median and its spread the median absolute deviation: a sheet FULL of copies (a reel
    of ten) fills the map with equally strong peaks, which lift a plain mean and standard deviation until no one copy
    stands out from the others -- the median and its deviation are those of where there is no copy."""
    kh, kw = template.shape
    pad_y, pad_x = int(_OVERHANG_SHARE * kh), int(_OVERHANG_SHARE * kw)
    padded = cv2.copyMakeBorder(target, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)
    correlation = cv2.matchTemplate(padded, template, cv2.TM_CCOEFF_NORMED)
    ruled_out = _blank_windows(padded, kh, kw)
    # where too much of the template would lie off the sheet
    height, width = target.shape
    xs = np.arange(correlation.shape[1]) - pad_x
    ys = np.arange(correlation.shape[0]) - pad_y
    inside_x = (np.minimum(xs + kw, width) - np.maximum(xs, 0)) / kw
    inside_y = (np.minimum(ys + kh, height) - np.maximum(ys, 0)) / kh
    ruled_out |= np.outer(inside_y, inside_x) < _MIN_INSIDE_SHARE
    valid = correlation[~ruled_out] if (~ruled_out).any() else correlation.ravel()
    floor = float(np.median(valid))
    spread = 1.4826 * float(np.median(np.abs(valid - floor)))
    correlation[ruled_out] = floor
    _low, score, _low_at, at = cv2.minMaxLoc(correlation)
    return float(score), (float(score) - floor) / (spread + 1e-9), (at[0] - pad_x, at[1] - pad_y), correlation, (pad_x, pad_y)


def _settle_scale(target: np.ndarray, template: np.ndarray, config: SeriesConfig, expected_scale: float | None) -> float | None:
    """The one scale, of the template against the target, that the strongest match anywhere on the target picks out."""
    centre = expected_scale if expected_scale and expected_scale > 0 else config.default_expected_scale
    best: tuple[float, float] | None = None  # (peak, scale)
    for step in range(-config.scale_steps_each_way, config.scale_steps_each_way + 1):
        scale = centre * config.scale_step**step
        resized = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        if min(resized.shape) < 12 or resized.shape[0] * _MIN_INSIDE_SHARE > target.shape[0] or resized.shape[1] * _MIN_INSIDE_SHARE > target.shape[1]:
            continue
        if float(resized.std()) < 1e-3:
            continue
        score, peak, _at, _map, _offset = _peak(target, resized)
        if score < config.min_match_score:
            continue
        if best is None or peak > best[0]:
            best = (peak, scale)
    return best[1] if best is not None else None


def find_repeats(
    template_rgb: np.ndarray, sheet_rgb: np.ndarray, config: SeriesConfig, expected_scale: float | None = None
) -> list[BoundingBox]:
    """Every place on `sheet_rgb` that holds a copy of `template_rgb` (the original's one artwork), in reading order.

    The scale is settled once, by the strongest match anywhere on the sheet (near `expected_scale` first, if the two
    files' resolutions say what it is likely to be); every repeat is then found AT that one scale -- they are printed
    copies of each other, so it does not vary from one to the next -- by finding the strongest remaining match,
    blanking out where it was found, and looking again, until nothing left stands out enough or `max_repeats` is
    reached. No rotation is looked for: a multi-up sheet's copies are printed upright to each other.

    Boxes are in `sheet_rgb`'s own pixels, each padded by `config.instance_margin_px` beyond the template's size so
    the comparison's own registration (features, not this search) can settle its exact position.
    """
    with StageTimer(logger, "find_repeats"):
        height, width = sheet_rgb.shape[:2]
        search_scale = min(1.0, config.search_long_edge_px / max(height, width))
        target = _edge_picture(cv2.resize(sheet_rgb, None, fx=search_scale, fy=search_scale, interpolation=cv2.INTER_AREA))
        template_full = _edge_picture(template_rgb)
        template_search = cv2.resize(
            template_full, None, fx=search_scale, fy=search_scale, interpolation=cv2.INTER_AREA if search_scale < 1 else cv2.INTER_LINEAR
        )
        scale = _settle_scale(target, template_search, config, expected_scale)
        if scale is None:
            return []

        template = cv2.resize(template_search, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        th, tw = template.shape
        if th < 8 or tw < 8 or th * _MIN_INSIDE_SHARE > target.shape[0] or tw * _MIN_INSIDE_SHARE > target.shape[1]:
            return []

        # Every instance is found in ONE correlation map (blanking the picture and matching again is no good: a
        # template that is mostly blank paper matches its own just-blanked footprint about as well as the real thing,
        # and a window of blank paper is already ruled out above). Classic multi-peak non-max suppression instead:
        # take the strongest peak, rule out another top-left within a fraction of the template's size of it (a real
        # neighbour, printed beside this one with a gap between them, sits further off than that; this only stops the
        # same instance being picked a second time a few pixels away), and look again.
        score, peak, at, working, (pad_x, pad_y) = _peak(target, template)
        floor = float(np.median(working))
        spread = 1.4826 * float(np.median(np.abs(working - floor)))
        radius_x, radius_y = max(4, int(config.suppression_fraction * tw)), max(4, int(config.suppression_fraction * th))
        found: list[tuple[int, int]] = []  # (left, top) in the search copy's pixels (a copy may overhang a little)
        for _ in range(config.max_repeats):
            if score < config.min_match_score or peak < config.min_match_peak:
                break
            left, top = at
            found.append((left, top))
            mx, my = left + pad_x, top + pad_y  # where that is on the map
            x0, y0 = max(0, mx - radius_x), max(0, my - radius_y)
            x1, y1 = min(working.shape[1], mx + radius_x + 1), min(working.shape[0], my + radius_y + 1)
            working[y0:y1, x0:x1] = floor
            _low, score, _low_at, (ax, ay) = cv2.minMaxLoc(working)
            at = (ax - pad_x, ay - pad_y)
            peak = (score - floor) / (spread + 1e-9)

        if not found:
            return []

        margin = config.instance_margin_px
        boxes = [
            BoundingBox(
                max(0.0, left / search_scale - margin),
                max(0.0, top / search_scale - margin),
                min(float(width), (left + tw) / search_scale + margin),
                min(float(height), (top + th) / search_scale + margin),
            )
            for left, top in found
        ]
        # Reading order: top to bottom, then left to right -- rows are as tall as a repeat, so two in the same row of
        # a slightly uneven grid still round to the same one.
        row_height = max(1.0, (th / search_scale) * 0.6)
        boxes.sort(key=lambda box: (round(box.y0 / row_height), box.x0))
        logger.info("repeats found", extra={"count": len(boxes), "scale": scale})
        return boxes


_PERIOD_LONG_EDGE_PX = 800  # the sheet's repeat is measured on a copy no bigger than this
_MIN_PERIOD_SHARE = 0.05  # a repeat is at least this share of the sheet's extent that way...
_MIN_PERIOD_CORRELATION = 0.6  # ...and the sheet shifted by it must match itself at least this well
_MIN_PERIOD_PROMINENCE = 0.2  # ...standing this far above the least it matched at any shorter shift
_FUNDAMENTAL_SHARE = 0.95  # the shortest shift matching within this of the best one is the repeat (not two of it)
_LOCATE_SCALE_STEP = 1.03
_LOCATE_STEPS_EACH_WAY = 8


def _period(edges: np.ndarray, axis: int) -> int | None:
    """How far apart the sheet's copies are along `axis` (0 = down, 1 = across), in `edges`' pixels: the shortest shift
    of the sheet that matches it with itself about as well as any does. None if it does not repeat that way."""
    centred = edges - edges.mean()
    extent = centred.shape[axis]
    scores = []
    for lag in range(max(4, int(_MIN_PERIOD_SHARE * extent)), extent // 2 + 1):
        a = centred[lag:] if axis == 0 else centred[:, lag:]
        b = centred[:-lag] if axis == 0 else centred[:, :-lag]
        scores.append((float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9)), lag))
    # a real repeat is a PEAK the match climbs back to after falling away between copies; a picture shifted a little
    # always matches itself fairly well (its edges are soft), falling steadily -- that is no repeat
    peaks = []
    for index, (score, lag) in enumerate(scores):
        before = scores[index - 1][0] if index else score
        after = scores[index + 1][0] if index + 1 < len(scores) else score
        dip = min(value for value, _lag in scores[: index + 1])
        if index and score >= before and score >= after and score - dip >= _MIN_PERIOD_PROMINENCE:
            peaks.append((score, lag))
    if not peaks:
        return None
    best = max(score for score, _lag in peaks)
    if best < _MIN_PERIOD_CORRELATION:
        return None
    return min(lag for score, lag in peaks if score >= _FUNDAMENTAL_SHARE * best)


def _gap(edges: np.ndarray, axis: int, period: int) -> int:
    """Where along `axis` one copy ends and the next begins (0 <= it < `period`): where the sheet, folded onto one
    period, has the least printed on it."""
    profile = edges.sum(axis=1 - axis)
    folded = np.bincount(np.arange(len(profile)) % period, weights=profile, minlength=period)
    smooth = np.convolve(np.tile(folded, 3), np.ones(3) / 3, "same")[period : 2 * period]
    return int(np.argmin(smooth))


def _sheet_repeat(sheet_rgb: np.ndarray) -> tuple[list[float | None], list[float]] | None:
    """How the sheet repeats, in its own pixels: ([the copies' spacing down, across] -- None where it does not repeat
    that way --, [where a copy is guessed to begin, each way]); None if it repeats neither way."""
    height, width = sheet_rgb.shape[:2]
    scale = min(1.0, _PERIOD_LONG_EDGE_PX / max(height, width))
    edges = _edge_picture(cv2.resize(sheet_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    periods: list[float | None] = []
    starts: list[float] = []
    for axis in (0, 1):
        period = _period(edges, axis)
        periods.append(None if period is None else period / scale)
        starts.append(0.0 if period is None else _gap(edges, axis, period) / scale)
    return None if periods == [None, None] else (periods, starts)


def _unit_at(shape: tuple[int, int], periods: list[float | None], starts: list[float], offsets: tuple[float, float]) -> tuple[int, int, int, int] | None:
    """One whole copy, (x0, y0, x1, y1), with copies taken to begin at `starts` + `offsets` -- the one nearest the middle
    of the sheet; None where no whole copy fits."""
    spans = []
    for axis, extent in ((0, shape[0]), (1, shape[1])):
        period = periods[axis]
        if period is None:
            spans.append((0, extent))
            continue
        first = (starts[axis] + offsets[axis]) % period
        begins = [first + k * period for k in range(int(extent // period) + 1) if first + (k + 1) * period <= extent]
        if not begins:
            return None
        begin = begins[len(begins) // 2]
        spans.append((round(begin), min(extent, round(begin + period))))
    (y0, y1), (x0, x1) = spans
    return x0, y0, x1, y1


def sheet_unit(sheet_rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    """One whole copy of what the sheet repeats, as (x0, y0, x1, y1) in its pixels, cut where the sheet has least printed
    between copies and taken from the middle of the sheet; None if the sheet does not repeat either way. (Where a copy
    has blank bands of its own wider than the gaps, that cut can fall inside it: `find_on_original` settles the cut.)"""
    repeat = _sheet_repeat(sheet_rgb)
    return None if repeat is None else _unit_at(sheet_rgb.shape[:2], *repeat, (0.0, 0.0))


_PHASES = 12  # where a copy begins is tried at this many places along one spacing, each way it repeats


def find_on_original(
    original_rgb: np.ndarray, sheet_rgb: np.ndarray, config: SeriesConfig, expected_scale: float | None = None
) -> tuple[float, tuple[int, int, int, int]] | None:
    """The original's one artwork found from the copies printed on the sheet: (how well it matched, (x0, y0, x1, y1) in
    the original's pixels), or None. A copy is cut out of the sheet at its repeat and looked for on the original page
    -- which holds only the one, so only a cut that holds a WHOLE copy (not the end of one and the start of the next)
    matches it in full: the cut is tried at several places along the repeat and the best match taken."""
    repeat = _sheet_repeat(sheet_rgb)
    if repeat is None:
        return None
    periods, starts = repeat
    shape = sheet_rgb.shape[:2]

    looked_in = _search_picture(original_rgb, config)  # (made once: every cut is looked for in the same page)

    def tried(offsets, scales=None):
        unit = _unit_at(shape, periods, starts, offsets)
        if unit is None:
            return None
        x0, y0, x1, y1 = unit
        return _locate(original_rgb, sheet_rgb[y0:y1, x0:x1], config, expected_scale, scales, looked_in)

    best = tried((0.0, 0.0))
    if best is None:
        return None
    scale = best[2]
    offsets = [0.0, 0.0]
    for axis in (0, 1):
        if periods[axis] is None:
            continue
        for step in range(1, _PHASES):
            trial = list(offsets)
            trial[axis] = step * periods[axis] / _PHASES
            found = tried(tuple(trial), [scale])
            if found is not None and found[0] > best[0]:
                best, offsets = found, trial
    return best[0], best[1]


def _search_picture(original_rgb: np.ndarray, config: SeriesConfig) -> tuple[np.ndarray, float]:
    """The original page as it is searched: its edges, on a copy no bigger than the search size; and that copy's scale."""
    search = min(1.0, config.search_long_edge_px / max(original_rgb.shape[:2]))
    return _edge_picture(cv2.resize(original_rgb, None, fx=search, fy=search, interpolation=cv2.INTER_AREA)), search


def locate_on_original(
    original_rgb: np.ndarray, unit_rgb: np.ndarray, config: SeriesConfig, expected_scale: float | None = None
) -> tuple[int, int, int, int] | None:
    """Where on the original page the sheet's copy `unit_rgb` is -- its one artwork, however much else the page holds
    (a job sheet, dimensions, notes) -- as (x0, y0, x1, y1) in the original's pixels; None if it is not there."""
    found = _locate(original_rgb, unit_rgb, config, expected_scale)
    return None if found is None else found[1]


def _locate(
    original_rgb: np.ndarray,
    unit_rgb: np.ndarray,
    config: SeriesConfig,
    expected_scale: float | None = None,
    scales: list[float] | None = None,
    looked_in: tuple[np.ndarray, float] | None = None,
) -> tuple[float, tuple[int, int, int, int], float] | None:
    """`locate_on_original`, with how well it matched there and at what scale (only `scales` tried, where given).
    `looked_in` is `_search_picture(original_rgb)`, where the caller has it already."""
    height, width = original_rgb.shape[:2]
    target, search = looked_in if looked_in is not None else _search_picture(original_rgb, config)
    unit = _edge_picture(cv2.resize(unit_rgb, None, fx=search, fy=search, interpolation=cv2.INTER_AREA))
    # the unit is at the sheet's scale: on the original it is 1 / (sheet px per original px)
    centre = 1.0 / expected_scale if expected_scale and expected_scale > 0 else 1.0
    if scales is None:
        scales = [centre * _LOCATE_SCALE_STEP**step for step in range(-_LOCATE_STEPS_EACH_WAY, _LOCATE_STEPS_EACH_WAY + 1)]
    best = None
    for scale in scales:
        resized = cv2.resize(unit, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        if min(resized.shape) < 12 or resized.shape[0] > target.shape[0] or resized.shape[1] > target.shape[1]:
            continue
        score, peak, at, _map, _offset = _peak(target, resized)
        if score >= config.min_match_score and peak >= config.min_match_peak and (best is None or score > best[0]):
            best = (score, at, resized.shape, scale)
    if best is None:
        return None
    score, (left, top), (th, tw), scale = best
    x0, y0 = max(0, round(left / search)), max(0, round(top / search))
    return score, (x0, y0, min(width, round((left + tw) / search)), min(height, round((top + th) / search))), scale


_PAGE_DPI = 100.0  # pages are told apart this coarsely


def choose_series_pages(original, suspect, original_page: int | None, suspect_page: int | None, config: SeriesConfig) -> tuple[int, int] | None:
    """For files of several pages: the original page holding the artwork the suspect sheet repeats, and that sheet --
    whichever pair the sheet's own copy is found on best (see `sheet_unit`); None where no page repeats one, or its copy
    is on no page of the original. A page the reviewer chose is kept."""

    def pages(artwork, chosen: int | None) -> list[int]:
        if chosen is not None and 0 <= chosen < artwork.page_count:
            return [chosen]
        return list(range(artwork.page_count))

    def picture(artwork, page: int) -> np.ndarray:
        return artwork.render(dpi=_PAGE_DPI if artwork.kind == "pdf" else None, page=page)

    best = None
    for sheet_page in pages(suspect, suspect_page):
        sheet = picture(suspect, sheet_page)
        if _sheet_repeat(sheet) is None:
            continue
        for page in pages(original, original_page):
            # both PDFs are pictured at one resolution, so a copy is the same size on both; a picture's is not known
            scale = 1.0 if suspect.kind == "pdf" and original.kind == "pdf" else None
            found = find_on_original(picture(original, page), sheet, config, scale)
            if found is not None and (best is None or found[0] > best[0]):
                best = (found[0], page, sheet_page)
    if best is None:
        return None
    logger.info("series pages chosen", extra={"original_page": best[1] + 1, "suspect_page": best[2] + 1, "score": round(best[0], 2)})
    return best[1], best[2]


def _artwork_crop(image: np.ndarray, geometry: IsolationGeometry | None, margin_px: int = 8) -> np.ndarray:
    """The one artwork the reviewer chose in `image` (cropped/lifted), on white, cropped tight to its own box.

    The whole page if nothing was chosen -- the repeat search still works from a whole page, just more slowly.
    """
    if geometry is None or geometry.boundary is None:
        return image
    height, width = image.shape[:2]
    keep = resolve_keep_mask((height, width), geometry)
    ys, xs = np.nonzero(keep)
    if len(xs) == 0:
        return image
    x0, y0 = max(0, int(xs.min()) - margin_px), max(0, int(ys.min()) - margin_px)
    x1, y1 = min(width, int(xs.max()) + 1 + margin_px), min(height, int(ys.max()) + 1 + margin_px)
    return apply_mask(image, keep)[y0:y1, x0:x1]


def _located_artwork(
    original_image: np.ndarray, sheet_rgb: np.ndarray, config: SeriesConfig, expected_scale: float | None
) -> tuple[IsolationGeometry, list[BoundingBox]] | None:
    """The original's one artwork found from the sheet's own repeat (`find_on_original`), as an area of
    the original, with every repeat of it on the sheet; None where the sheet does not repeat or its copy is not on the
    original."""
    found = find_on_original(original_image, sheet_rgb, config, expected_scale)
    if found is None:
        return None
    box = found[1]
    height, width = original_image.shape[:2]
    bx0, by0, bx1, by1 = box
    geometry = IsolationGeometry(boundary=NormalizedRect(bx0 / width, by0 / height, bx1 / width, by1 / height), boundary_origin="manual")
    return geometry, find_repeats(original_image[by0:by1, bx0:bx1], sheet_rgb, config, expected_scale)


_FRAME_MARGIN_SHARE = 0.06  # the artwork is cut out of the original's page with this share of its size round it...
_FRAME_MARGIN_MIN_PX = 32
_FRAME_WORTH_SHARE = 0.7  # ...where it is less than this share of the page (otherwise the page is used as it is)


def _artwork_frame(shape: tuple[int, int], geometry: IsolationGeometry | None) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1): the part of the original's page the artwork is compared in -- its area and a margin -- or None
    where no area was chosen or it is most of the page anyway."""
    if geometry is None or geometry.boundary is None:
        return None
    keep = resolve_keep_mask(shape, geometry)
    ys, xs = np.nonzero(keep)
    if len(xs) == 0:
        return None
    height, width = shape
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    if (x1 - x0) * (y1 - y0) > _FRAME_WORTH_SHARE * width * height:
        return None
    margin_x = max(_FRAME_MARGIN_MIN_PX, round(_FRAME_MARGIN_SHARE * (x1 - x0)))
    margin_y = max(_FRAME_MARGIN_MIN_PX, round(_FRAME_MARGIN_SHARE * (y1 - y0)))
    return max(0, x0 - margin_x), max(0, y0 - margin_y), min(width, x1 + margin_x), min(height, y1 + margin_y)


def _in_frame(geometry: IsolationGeometry, shape: tuple[int, int], frame: tuple[int, int, int, int]) -> IsolationGeometry:
    """`geometry` (in shares of the whole page, `shape`) in shares of the part of it `frame` cuts out."""
    height, width = shape
    x0, y0, x1, y1 = frame
    fw, fh = x1 - x0, y1 - y0

    def point(x: float, y: float) -> tuple[float, float]:
        return (x * width - x0) / fw, (y * height - y0) / fh

    def moved(region):
        if isinstance(region, NormalizedRect):
            (ax, ay), (bx, by) = point(region.x0, region.y0), point(region.x1, region.y1)
            return NormalizedRect(ax, ay, bx, by)
        return NormalizedPolygon([point(x, y) for x, y in region.points])

    long_ratio = max(width, height) / max(fw, fh)  # lengths kept in shares of the page's long edge
    return replace(
        geometry,
        boundary=moved(geometry.boundary) if geometry.boundary is not None else None,
        erase_regions=[moved(region) for region in geometry.erase_regions],
        erase_brush_radius_norm=geometry.erase_brush_radius_norm * long_ratio,
        edge_inset_norm=geometry.edge_inset_norm * long_ratio,
    )


_SAME_PLACE_OVERLAP = 0.5  # two repeats' findings at the same place of the original (this much overlap) and of one kind...
_COMMON_SHARE = 0.8  # ...found in at least this share of the repeats compared: one difference of the sheet, listed once


def _overlap(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection over union of two boxes."""
    width = min(a.x1, b.x1) - max(a.x0, b.x0)
    height = min(a.y1, b.y1) - max(a.y0, b.y0)
    if width <= 0 or height <= 0:
        return 0.0
    shared = width * height
    return shared / (a.width * a.height + b.width * b.height - shared)


def _which_repeats(numbers: list[int], total: int) -> str:
    numbers = sorted(set(numbers))
    if len(numbers) == total:
        return f"All {total} repeats"
    if len(numbers) == 1:
        return f"Repeat {numbers[0]} of {total}"
    return f"Repeats {', '.join(str(n) for n in numbers)} of {total}"


def _gathered(found: list[tuple[int, DiffCluster]], total: int, compared: int) -> list[DiffCluster]:
    """The repeats' findings as the reviewer should read them. A difference at the same place of the original, of the
    same kind, in (nearly) every repeat is the sheet differing from the original there -- the artwork set differently,
    the stock another colour -- and is ONE finding, marked on each repeat it is in. A difference in only some repeats
    is a print going wrong: each is its own finding, saying which repeat."""
    groups: list[list[tuple[int, DiffCluster]]] = []
    for number, cluster in sorted(found, key=lambda item: item[0]):
        for group in groups:
            first = group[0][1]
            if (
                first.category == cluster.category
                and first.subtype is cluster.subtype
                and all(n != number for n, _c in group)
                and _overlap(first.bbox_in_original, cluster.bbox_in_original) >= _SAME_PLACE_OVERLAP
            ):
                group.append((number, cluster))
                break
        else:
            groups.append([(number, cluster)])
    needed = max(2, int(np.ceil(_COMMON_SHARE * compared)))
    gathered: list[DiffCluster] = []
    for group in groups:
        numbers = [number for number, _cluster in group]
        if len(group) >= needed:
            first = group[0][1]
            first.also_in_suspect = [cluster.bbox_in_suspect for _number, cluster in group[1:]]
            first.detail = f"{_which_repeats(numbers, total)}: {first.detail}"
            gathered.append(first)
        else:
            for number, cluster in group:
                cluster.detail = f"Repeat {number} of {total}: {cluster.detail}"
                gathered.append(cluster)
    return gathered


def _the_other_way_round(original_image: np.ndarray, sheet_rgb: np.ndarray, config: AppConfig) -> bool:
    """Whether the files were given the other way round: the original page holds repeats of the suspect's artwork (the
    sheet given as the Original, the one artwork as the Suspect). Asked only once no repeat was found the right way."""
    try:  # (the same search, the files' parts swapped: the original as the sheet, the suspect's page as the one artwork)
        located = _located_artwork(sheet_rgb, original_image, config.series, None)
    except Exception:  # noqa: BLE001 -- only a better message hangs on it
        return False
    return located is not None and len(located[1]) >= 2


def compare_series(
    original_image: np.ndarray,
    sheet_rgb: np.ndarray,
    mode: ComparisonMode,
    config: AppConfig,
    original_geometry: IsolationGeometry | None = None,
    expected_scale: float | None = None,
    find_area: Callable[[], IsolationGeometry | None] | None = None,
    align_only: bool = False,
) -> ComparisonOutcome:
    """Compare the one artwork the reviewer chose in `original_image` against every repeat of it found on the
    suspect sheet `sheet_rgb`. `find_area`, if given, finds the original's artwork some other way (its border, paired
    with the sheet's): asked only if the repeats cannot be found from the page as given or from the sheet's own copy.

    Each repeat is compared on its own by the ordinary single-pair pipeline (`run_comparison`, with the same
    `original_geometry` every time -- the original is the same file throughout), so a repeat's own misprint is its
    own finding, not blamed on its neighbours. The findings come back as one list in one `ComparisonOutcome`, each
    saying which repeat (of how many) it is in; the suspect pane is the whole sheet, the original pane the one
    artwork every repeat was measured against.
    """
    template = _artwork_crop(original_image, original_geometry)
    boxes = find_repeats(template, sheet_rgb, config.series, expected_scale)
    del template  # only needed to search for the repeats; `run_comparison` below re-derives its own crop from the geometry
    found_note = ""
    if len(boxes) < 2:
        # The original as given (a whole page with a job sheet, dimensions and notes round the artwork, or an area that
        # is not quite the one artwork) is not what the sheet repeats: the sheet's own copy is found on the original
        # instead, and that is the artwork compared.
        located = _located_artwork(original_image, sheet_rgb, config.series, expected_scale)
        if located is not None:
            geometry, located_boxes = located
            if len(located_boxes) > len(boxes):
                original_geometry, boxes = geometry, located_boxes
                found_note = (
                    "The original's artwork was found on its page by matching it to the copies printed on the suspect "
                    "sheet; only it was compared."
                )
    if len(boxes) < 2 and find_area is not None:
        geometry = find_area()
        if geometry is not None and geometry.boundary is not None:
            found_boxes = find_repeats(_artwork_crop(original_image, geometry), sheet_rgb, config.series, expected_scale)
            if len(found_boxes) > len(boxes):
                original_geometry, boxes, found_note = geometry, found_boxes, ""
    if not boxes:
        if _the_other_way_round(original_image, sheet_rgb, config):
            raise NoRepeatsFoundError(
                "The files look the other way round: the Original holds the repeats of the Suspect's artwork. In Series "
                "mode the Original is the one artwork and the Suspect the sheet of its repeats -- swap the two files."
            )
        raise NoRepeatsFoundError(
            "No repeat of the original's artwork could be found on the suspect sheet. Check that the original was "
            "cropped or lifted to just the one artwork, and that the suspect really holds copies of it."
        )
    total = len(boxes)
    height, width = sheet_rgb.shape[:2]
    all_clusters: list[DiffCluster] = []
    warnings: list[str] = []
    notes = [f"Found {total} repeat{'s' if total != 1 else ''} of the artwork on the suspect sheet; each was compared on its own."]
    if found_note:
        notes.append(found_note)
    profiles: list[str] = []
    graphics_compared = True
    reference_alignment = None
    # Each repeat is compared with the original's artwork cut out of its page (with a margin): a job sheet round it is
    # not compared anyway, and a page ten times the artwork's size made every repeat's comparison ten times the work.
    frame = _artwork_frame(original_image.shape[:2], original_geometry)
    fx0, fy0 = (frame[0], frame[1]) if frame is not None else (0, 0)
    working = original_image[frame[1] : frame[3], frame[0] : frame[2]] if frame is not None else original_image
    working_geometry = _in_frame(original_geometry, original_image.shape[:2], frame) if frame is not None else original_geometry

    def compare_one(box: BoundingBox) -> tuple[ComparisonOutcome | Exception, int, int]:
        x0, y0, x1, y1 = padded_window(box, 0, (height, width))
        try:
            return run_comparison(working, sheet_rgb[y0:y1, x0:x1], mode, config, working_geometry, None, align_only=align_only), x0, y0
        except Exception as exc:  # noqa: BLE001 -- one bad repeat (too little overlap to align) must not sink the rest
            return exc, x0, y0

    # the repeats are compared several at a time (each is its own comparison; the original's side of the work is
    # done once and shared -- see `app.engine.prepared`), their findings gathered in reading order
    with ThreadPoolExecutor(max_workers=max(1, min(config.series.max_repeat_workers, total))) as pool:
        compared = list(pool.map(compare_one, boxes))
    found: list[tuple[int, DiffCluster]] = []  # (repeat number, finding) -- boxes on the original's page and the sheet
    said: dict[str, list[int]] = {}  # each warning, and the repeats it was given for
    for number, (outcome, x0, y0) in enumerate(compared, start=1):
        if isinstance(outcome, Exception):
            logger.warning("repeat %d/%d could not be compared: %s", number, total, outcome)
            warnings.append(f"Repeat {number} of {total}: could not be aligned with the original, so it was skipped.")
            continue
        if reference_alignment is None:
            reference_alignment = outcome.alignment
        profiles.append(outcome.detection_profile)
        graphics_compared = graphics_compared and outcome.graphics_compared
        for cluster in outcome.clusters:
            shifted = cluster.bbox_in_suspect
            cluster.bbox_in_suspect = BoundingBox(shifted.x0 + x0, shifted.y0 + y0, shifted.x1 + x0, shifted.y1 + y0)
            own = cluster.bbox_in_original  # (in the artwork cut out of the original's page: back onto the page)
            cluster.bbox_in_original = BoundingBox(own.x0 + fx0, own.y0 + fy0, own.x1 + fx0, own.y1 + fy0)
            found.append((number, cluster))
        for warning in outcome.warnings:
            said.setdefault(warning, []).append(number)

    if reference_alignment is None:
        raise NoRepeatsFoundError("Every repeat that was found failed to align with the original; none could be compared.")

    compared_count = sum(1 for outcome, _x, _y in compared if not isinstance(outcome, Exception))
    all_clusters = _gathered(found, total, compared_count)
    warnings.extend(f"{_which_repeats(numbers, total)}: {warning}" for warning, numbers in said.items())
    common = sum(1 for cluster in all_clusters if cluster.also_in_suspect)
    if common:
        notes.append(
            f"{common} difference{'s are' if common != 1 else ' is'} the same in (nearly) every repeat -- the sheet differs "
            "from the original there, not one print from the others -- and listed once each."
        )

    all_clusters.sort(key=lambda cluster: (not cluster.also_in_suspect, cluster.bbox_in_suspect.y0, cluster.bbox_in_suspect.x0))
    for number, cluster in enumerate(all_clusters, start=1):
        cluster.id = number

    return ComparisonOutcome(
        alignment=reference_alignment,
        candidates=[],
        clusters=[] if align_only else all_clusters,
        all_clusters=[] if align_only else all_clusters,
        aligned_original=original_image,
        aligned_suspect=sheet_rgb,
        warnings=warnings,
        detection_profile="exact" if profiles and all(p == "exact" for p in profiles) else "tolerant",
        used_change_regions=False,
        mode=mode,
        compared_outline=None,
        graphics_compared=graphics_compared,
        notes=["Files updated and aligned. Click Recompare to run comparison."] if align_only else notes,
    )
