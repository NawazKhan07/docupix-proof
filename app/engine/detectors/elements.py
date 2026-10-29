"""Findings that are one ELEMENT moved or resized in the suspect -- a logo, a block of lettering, a badge re-placed and
rescaled when a vendor re-laid a panel -- made one finding, that it moved.

Such an element differs from the original at every edge, and each edge was a finding of its own ('Colour changed (dE 31)'
at a piece of a letter, dozens for one logo). Here the findings lying close together are taken as one element, and the
element is looked for in the suspect near where it was, at a size a little bigger or smaller. Only where it is found
plainly -- lined up, as like the original as this pair's unchanged print is -- is it one finding, shown where it was and
where it is now. Where it differs once lined up, something about it changed too, and its findings stay as they were.

"As like as this pair's unchanged print" is measured on the pair, never preset (`_Yardstick`): two files differ even
where nothing changed -- a vendor's proof prints its red a shade warmer everywhere, renders softer, registers a pixel
off -- and how much is the pair's own. A moved element is held to exactly that, no more and no less.
"""

from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import cv2
import numpy as np

from app.engine.aggregation.aggregator import combined
from app.config import DetectionConfig
from app.engine.detectors import local_diff, movement
from app.logging_config import StageTimer
from app.models.diff import BoundingBox, DiffCluster, DiffSubtype, DiffType

logger = logging.getLogger(__name__)

# What is a piece of an element, and what is looked for (shares of the page, so the same on any page at any resolution)
_GROUP_SHARE = 0.003  # findings this close (a share of the page's long edge) are pieces of one element...
_PIECE_MAX_SHARE = 0.06  # ...each no bigger than this (a finding across a panel is no piece of a logo, and would join all)
_ELEMENT_MAX_SHARE = 0.2  # ...and together no bigger than this: more is a re-laid panel, not an element
_MIN_PIECES = 2  # an element is at least this many findings (one finding alone is left to the other checks)
_PAD_SHARE = 0.08  # the element's picture is taken this much (a share of its size) round its findings
_SEARCH_SHARE = 0.06  # it is looked for this far (a share of the page's long edge) round where it was...
_SCALES = tuple(round(0.85 * 1.05**step, 4) for step in range(8))  # ...at these sizes (85% to ~120%), on a copy...
_WORK_SIDE_PX = 160  # ...with the element reduced to this long edge...
_REFINE_SIDE_PX = 600  # ...then refined round what that found on one this big (or full size, if smaller)...
_REFINE_SCALES = tuple(round(-0.025 + 0.005 * step, 3) for step in range(11))  # ...at these sizes round it
_MIN_CLEAR = 0.01  # (the least a best place must stand clear of the next, however steady the pair's fit)
_MIN_STRUCTURE = 12.0  # a flat crop fits anywhere: not looked for
_MIN_MOVE_PX = 4.0
_MIN_RESIZE = 0.02

# The pair's yardstick (`_Yardstick`): how it is measured
_SOFTNESS = (0.0, 0.6, 1.2, 2.0, 3.0, -0.6, -1.2, -2.0, -3.0)  # (blurs tried to tell how much softer one file renders)
_SAMPLE_SLACK_PX = 3  # (each lined up within this much before it is measured)
_SAMPLE_CROPS = 48  # unchanged crops the shape fit is measured on
_SAMPLE_SKIP_AREA = 0.05  # (crops are taken away from findings, bar those over this share of the page)
_SAMPLE_INK = 0.05  # (a crop with at least this share printed: paper against paper says nothing)
# (Crops away from every finding are not all alike: on a re-laid proof some differ in ways no finding marks. So what is
# typical of them is taken -- the median, sound while at least half the page's print is the same -- never the worst.)
_OFF_PERCENTILE = 75.0  # ...and miss its colours no more than three in four unchanged crops do...
_OFF_GRACE = 1.5  # ...with this much grace for the resized edges
_COLOUR_BIN = 16  # (8-bit Lab) how the pair's colours are told apart when learning how each one prints in the suspect
_COLOUR_SAMPLES = 1_000_000  # (the page is sampled evenly, about this many pixels, for it)
_COLOUR_MIN_SAMPLES = 40  # (a colour seen less than this is judged by the pair's spread overall)
_COLOUR_SPREAD_SHARE = 80.0  # how far a colour strays is how far all but the furthest fifth of its pixels do (a pair may
# print one colour two ways -- a logo's red and a panel's, alike in the original, apart in a proof -- and both are its own)
_COLOUR_SPREAD_TIMES = 1.5  # a pixel is off its expected colour when it strays this many times that...
_TRANSFORM_HOLD = 0.01  # (how firmly the overall colour change is held toward none, where the page says little)
_COLOUR_SPREAD_FLOOR = 4.0  # ...and at least this far (8-bit Lab: a spread of nought would make every pixel off)


class _Yardstick(NamedTuple):
    """How alike two pictures of the same thing are IN THIS PAIR (see the module), measured where nothing changed."""

    softness: float  # (Gaussian sigma, px) the original blurred this much is the suspect (minus: the suspect blurred)...
    noise: float  # ...and grain of this variance (grey levels squared) on top
    fit: float  # an element must correlate (grey, softened alike) at least this share of what that grain allows it
    off: float  # the largest share of its pixels that may miss their expected colour
    expected: np.ndarray  # (bins^3, 3) how each original colour shifts (Lab) printed in the suspect (NaN: not seen enough)...
    tolerance: np.ndarray  # (bins^3,) how far a pixel of that colour may stray from it
    shift: np.ndarray  # (3, 4) ...and how the pair's colours change overall, an RGB transform (for a colour seen too little)


def _bins(lab: np.ndarray) -> np.ndarray:
    levels = 256 // _COLOUR_BIN
    quantised = lab.reshape(-1, 3).astype(np.int32) // _COLOUR_BIN
    return (quantised[:, 0] * levels + quantised[:, 1]) * levels + quantised[:, 2]


def _colour_model(original: np.ndarray, suspect: np.ndarray, changed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """How each colour of the original shifts printed in the suspect, how far it strays, and how the colours change
    overall: learnt over the page bar `changed` -- a change is local, the way a press or a proof renders a colour is
    everywhere. (Never from where things moved: a logo moved off its place would teach that its colour prints as the
    paper it left.)"""
    step = max(1, int(math.sqrt(original.shape[0] * original.shape[1] / _COLOUR_SAMPLES)))
    keep = changed[::step, ::step].reshape(-1) == 0
    rgb_a = np.ascontiguousarray(original[::step, ::step]).reshape(-1, 3)[keep]
    rgb_b = np.ascontiguousarray(suspect[::step, ::step]).reshape(-1, 3)[keep]
    overall = _rgb_transform(rgb_a.astype(np.float32), rgb_b.astype(np.float32))
    lab_a = cv2.cvtColor(rgb_a.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    lab_b = cv2.cvtColor(rgb_b.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    differences = lab_b - lab_a.astype(np.float32)
    bins = _bins(lab_a)
    count = (256 // _COLOUR_BIN) ** 3
    expected = np.full((count, 3), np.nan, np.float32)
    tolerance = np.full(count, np.nan, np.float32)
    order = np.argsort(bins, kind="stable")
    sorted_bins = bins[order]
    starts = np.flatnonzero(np.r_[True, sorted_bins[1:] != sorted_bins[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for start, end in zip(starts, ends):
        if end - start < _COLOUR_MIN_SAMPLES:
            continue
        members = order[start:end]
        shifts = np.median(differences[members], axis=0)
        strays = float(np.percentile(np.linalg.norm(differences[members] - shifts, axis=1), _COLOUR_SPREAD_SHARE))
        expected[sorted_bins[start]] = shifts
        tolerance[sorted_bins[start]] = max(_COLOUR_SPREAD_FLOOR, _COLOUR_SPREAD_TIMES * strays)
    known = ~np.isnan(tolerance)
    typical = float(np.median(tolerance[known])) if known.any() else _COLOUR_SPREAD_FLOOR * _COLOUR_SPREAD_TIMES
    tolerance[~known] = typical
    return expected, tolerance, overall


def _rgb_transform(rgb_a: np.ndarray, rgb_b: np.ndarray) -> np.ndarray:
    """(3, 4) the affine RGB transform taking the original's colours to the suspect's: fitted as a change from leaving
    them as they are, by least squares held toward no change (a page of paper and black says nothing of how a red
    prints, and must not be read as saying it turns to anything), then again without the worst-fitting fifth (what
    changed, or sits on an edge). No change with too few samples."""
    identity = np.hstack([np.eye(3, dtype=np.float32), np.zeros((3, 1), np.float32)])
    if len(rgb_a) < 100:
        return identity
    design = np.hstack([rgb_a, np.full((len(rgb_a), 1), 255.0, np.float32)]).astype(np.float64)
    change = (rgb_b - rgb_a).astype(np.float64)
    chosen = np.arange(len(rgb_a))
    solution = np.zeros((4, 3))
    for _round in range(2):
        x = design[chosen]
        hold = _TRANSFORM_HOLD * len(chosen) * 255.0**2
        solution = np.linalg.solve(x.T @ x + hold * np.eye(4), x.T @ change[chosen])
        residual = np.linalg.norm(design @ solution - change, axis=1)
        chosen = np.flatnonzero(residual <= np.percentile(residual, 80))
    transform = identity.copy()
    transform[:, :3] += solution[:3].T.astype(np.float32)
    transform[:, 3] += 255.0 * solution[3].astype(np.float32)
    return transform


def _off_share(here: np.ndarray, there: np.ndarray, expected: np.ndarray, tolerance: np.ndarray, overall: np.ndarray) -> float:
    """The share of `there`'s pixels (the suspect's, lined up with `here`, the original's) off the colour this pair
    prints `here`'s colours as. (A colour the pair showed too little of is expected to change as its colours do overall.)"""
    rgb_here = np.ascontiguousarray(here).reshape(-1, 3)
    lab_here = cv2.cvtColor(rgb_here.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    lab_there = cv2.cvtColor(np.ascontiguousarray(there).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    bins = _bins(lab_here)
    predicted = lab_here.astype(np.float32) + expected[bins]
    unseen = np.isnan(predicted[:, 0])
    if unseen.any():
        changed = rgb_here[unseen].astype(np.float32) @ overall[:, :3].T + overall[:, 3]
        changed = np.clip(changed, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
        predicted[unseen] = cv2.cvtColor(changed, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return float(np.mean(np.linalg.norm(lab_there - predicted, axis=1) > tolerance[bins]))


def _grain_fit(contrast: float, noise: float) -> float:
    """How well a picture of `contrast` (grey variance) can correlate with itself under grain of variance `noise`."""
    return math.sqrt(contrast / (contrast + noise)) if contrast > 0 else 0.0


def _least_fit(crop_grey: np.ndarray, yardstick: _Yardstick) -> float:
    """The least the (softened) picture `crop_grey` must fit its counterpart to be the same, in this pair."""
    return yardstick.fit * _grain_fit(float(crop_grey.astype(np.float32).var()), yardstick.noise)


def _soften(crop: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(crop, (0, 0), sigma) if sigma > 0 else crop


def _grey(crop: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop


def _fit(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.matchTemplate(_grey(a), _grey(b), cv2.TM_CCOEFF_NORMED)[0, 0])


def _best_fit(area: np.ndarray, picture: np.ndarray) -> float:
    """How well `picture` fits `area` (a little bigger) where it fits best."""
    return float(cv2.matchTemplate(_grey(area), _grey(picture), cv2.TM_CCOEFF_NORMED).max())


def _yardstick(
    original: np.ndarray, suspect: np.ndarray, clusters: list[DiffCluster], size: tuple[int, int], elements: list[tuple[int, int, int, int]]
) -> _Yardstick | None:
    """The pair's yardstick (see `_Yardstick`): the shapes' fit and the colours' misses measured on crops of `size`
    (w, h) away from every finding; how each colour prints learnt from all the page but round the `elements` being
    judged (each as far again round it as it is big: where it may have gone) -- the small colour findings are this
    pair's own rendering of its colours, the very thing to learn. None where too little of the page is printed and
    unchanged to say."""
    height, width = original.shape[:2]
    w, h = min(size[0], width // 4), min(size[1], height // 4)
    if w < 16 or h < 16:
        return None
    changed = np.zeros((height, width), np.uint8)
    for cluster in clusters:
        box = cluster.bbox_in_original
        if (box.x1 - box.x0) * (box.y1 - box.y0) > _SAMPLE_SKIP_AREA * height * width:
            continue  # a finding across the page (the percentiles below allow for what changed in some crops)
        changed[max(0, int(box.y0)) : int(math.ceil(box.y1)), max(0, int(box.x0)) : int(math.ceil(box.x1))] = 1
    darkest = cv2.min(cv2.min(original[..., 0], original[..., 1]), original[..., 2]) if original.ndim == 3 else original
    printed = (darkest < 200).astype(np.uint8)
    integral_changed = cv2.integral(changed)
    integral_printed = cv2.integral(printed)

    def total(integral: np.ndarray, x: int, y: int) -> int:
        return int(integral[y + h, x + w] - integral[y, x + w] - integral[y + h, x] + integral[y, x])

    places = [
        (x, y)
        for y in range(0, height - h, max(1, h // 2))
        for x in range(0, width - w, max(1, w // 2))
        if total(integral_changed, x, y) == 0 and total(integral_printed, x, y) >= _SAMPLE_INK * w * h
    ]
    if len(places) < 8:
        return None
    places = places[:: max(1, len(places) // _SAMPLE_CROPS)][:_SAMPLE_CROPS]
    moving = np.zeros((height, width), np.uint8)
    for x0, y0, x1, y1 in elements:
        reach_x, reach_y = x1 - x0, y1 - y0
        moving[max(0, y0 - reach_y) : y1 + reach_y, max(0, x0 - reach_x) : x1 + reach_x] = 1
    expected, tolerance, shift = _colour_model(original, suspect, moving)
    slack = _SAMPLE_SLACK_PX
    crops = [
        (original[y : y + h, x : x + w], suspect[max(0, y - slack) : y + h + slack, max(0, x - slack) : x + w + slack])
        for x, y in places
        if float(_grey(original[y : y + h, x : x + w]).std()) >= _MIN_STRUCTURE
    ]
    if len(crops) < 8:
        return None
    # How much softer one file renders than the other (a proof, a scan): the blur that best makes the one the other --
    # each crop let find its own best place a pixel or two round (else blur would pass for being a pixel off, which it
    # hides). An element is lined up just as closely before it is judged (`_found`).
    def fits_at(sigma: float) -> list[float]:
        return [_best_fit(_soften(there, max(-sigma, 0.0)), _soften(here, max(sigma, 0.0))) for here, there in crops]

    best_fits, softness = None, 0.0
    for sigma in _SOFTNESS:
        fits = fits_at(sigma)
        if best_fits is None or np.median(fits) > np.median(best_fits):
            best_fits, softness = fits, sigma
    coarse_sigma = softness
    for step in range(-5, 6):  # (then finer round it: fine lettering tells a blur a tenth of a pixel off)
        sigma = round(coarse_sigma + 0.1 * step, 2)
        if step == 0 or (coarse_sigma > 0 > sigma) or (coarse_sigma < 0 < sigma):
            continue
        fits = fits_at(sigma)
        if np.median(fits) > np.median(best_fits):
            best_fits, softness = fits, sigma
    offs = []
    for here, there in crops:
        if here.ndim != 3:
            offs.append(0.0)
            continue
        soft_here, soft_there = _soften(here, max(softness, 0.0)), _soften(there, max(-softness, 0.0))
        _low, _fit_there, _at_low, (bx, by) = cv2.minMaxLoc(cv2.matchTemplate(_grey(soft_there), _grey(soft_here), cv2.TM_CCOEFF_NORMED))
        lined_up = soft_there[by : by + here.shape[0], bx : bx + here.shape[1]]
        offs.append(_off_share(soft_here, lined_up, expected, tolerance, shift))
    # Grain costs a faint, fine picture more of its fit than a bold one: a crop of contrast (variance) v under grain of
    # variance n correlates sqrt(v / (v + n)). The pair's grain is read from its crops that way...
    contrasts = [float(_grey(_soften(here, max(softness, 0.0))).astype(np.float32).var()) for here, _there in crops]
    noise = float(np.median([v * (1.0 / max(f, 1e-3) ** 2 - 1.0) for v, f in zip(contrasts, best_fits) if f > 0]))
    noise = max(0.0, noise)
    # ...and how far crops fall short of what it allows, typically -- the median, less the spread above it (the clean
    # side; the side below holds whatever on the page differs without a finding)
    shares = [f / _grain_fit(v, noise) for v, f in zip(contrasts, best_fits)]
    middle, upper = float(np.percentile(shares, 50)), float(np.percentile(shares, 75))
    return _Yardstick(
        softness=softness,
        noise=noise,
        fit=middle - (upper - middle),
        off=float(np.percentile(offs, _OFF_PERCENTILE)) * _OFF_GRACE,
        expected=expected,
        tolerance=tolerance,
        shift=shift,
    )


def _groups(clusters: list[DiffCluster], reach: float) -> list[list[DiffCluster]]:
    """Clusters whose boxes lie within `reach` of each other, joined (each a list)."""
    parent = list(range(len(clusters)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    boxes = [c.bbox_in_original for c in clusters]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if a.x0 - reach <= b.x1 and b.x0 - reach <= a.x1 and a.y0 - reach <= b.y1 and b.y0 - reach <= a.y1:
                parent[root(j)] = root(i)
    groups: dict[int, list[DiffCluster]] = {}
    for i, cluster in enumerate(clusters):
        groups.setdefault(root(i), []).append(cluster)
    return list(groups.values())


def _resize_cost(template: np.ndarray, scale: float) -> float:
    """How well `template` still fits itself once made `scale` its size and back (1: not at all softened) -- what
    re-sizing costs THIS picture (a fine hairline loses more than a solid block), allowed for when judging the fit of an
    element found resized."""
    if abs(scale - 1.0) < 1e-3:
        return 1.0
    h, w = template.shape[:2]
    sized = cv2.resize(template, (max(2, round(w * scale)), max(2, round(h * scale))), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    back = cv2.resize(sized, (w, h), interpolation=cv2.INTER_AREA if scale > 1 else cv2.INTER_LINEAR)
    return max(0.0, _fit(back, template))


def _edge_share(here: np.ndarray, scale: float, tolerance: np.ndarray) -> float:
    """The share of `here`'s pixels that drift off their colour (by more than the pair lets them stray) when it is made
    `scale` its size and back, half a pixel off: what re-sizing and placing it cost THIS picture's colours (its edges --
    many in fine lettering, few in a solid block), allowed for besides the pair's own."""
    h, w = here.shape[:2]
    sized = cv2.resize(here, (max(2, round(w * scale)), max(2, round(h * scale))), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    back = cv2.resize(sized, (w, h), interpolation=cv2.INTER_AREA if scale > 1 else cv2.INTER_LINEAR)
    back = cv2.warpAffine(back, np.float32([[1, 0, 0.5], [0, 1, 0.5]]), (w, h), borderMode=cv2.BORDER_REPLICATE)
    lab_here = cv2.cvtColor(np.ascontiguousarray(here), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    lab_back = cv2.cvtColor(back, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    return float(np.mean(np.linalg.norm(lab_back - lab_here, axis=1) > tolerance[_bins(lab_here)]))


def _found(
    original: np.ndarray, suspect: np.ndarray, box: tuple[int, int, int, int], reach: int, yardstick: _Yardstick
) -> tuple[int, int, float] | None:
    """(x, y, scale) where the original's picture in `box` lies in `suspect` (its top-left, full size) -- searched over a
    reduced copy, then refined at full size -- if it fits there better than where it is, and at least `least_fit`;
    None otherwise."""
    height, width = original.shape[:2]
    x0, y0, x1, y1 = box
    # (each side softened as this pair's softer file is: compared as alike as they print)
    softness = yardstick.softness
    template = _soften(_grey(original[y0:y1, x0:x1]), max(softness, 0.0))
    if float(template.std()) < _MIN_STRUCTURE:
        return None
    least_fit = _least_fit(template, yardstick)

    def suspect_grey(y_from: int, y_to: int, x_from: int, x_to: int) -> np.ndarray:
        return _soften(_grey(suspect[y_from:y_to, x_from:x_to]), max(-softness, 0.0))

    home = _fit(suspect_grey(y0, y1, x0, x1), template)  # where it lies now, at its own size: the fit to beat
    if home >= least_fit:
        return None  # it is where it was, as like as the pair's print is: what differs there is no move
    work = min(1.0, _WORK_SIDE_PX / max(x1 - x0, y1 - y0))
    sx0, sy0 = max(0, x0 - reach), max(0, y0 - reach)
    sx1, sy1 = min(width, x1 + reach), min(height, y1 + reach)
    search = cv2.resize(suspect_grey(sy0, sy1, sx0, sx1), None, fx=work, fy=work, interpolation=cv2.INTER_AREA)
    small = cv2.resize(template, None, fx=work, fy=work, interpolation=cv2.INTER_AREA)
    coarse, scores_at_best = None, None
    for scale in _SCALES:
        sized = cv2.resize(small, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        if sized.shape[0] < 8 or sized.shape[1] < 8 or sized.shape[0] > search.shape[0] or sized.shape[1] > search.shape[1]:
            continue
        scores = cv2.matchTemplate(search, sized, cv2.TM_CCOEFF_NORMED)
        _low, fit, _at_low, (bx, by) = cv2.minMaxLoc(scores)
        if coarse is None or fit > coarse[3]:
            coarse = (sx0 + bx / work, sy0 + by / work, scale, float(fit))
            scores_at_best, peak, sized_shape = scores, (bx, by), sized.shape
    if coarse is None:
        return None
    # Found plainly, or only somewhere along a pattern that fits itself anywhere (a rule, a stripe, a tint's grain)? The
    # best place must stand clear of the best anywhere else -- by more than the pair's own wobble in fit.
    elsewhere = scores_at_best.copy()
    reach_x, reach_y = max(1, sized_shape[1] // 2), max(1, sized_shape[0] // 2)
    elsewhere[max(0, peak[1] - reach_y) : peak[1] + reach_y + 1, max(0, peak[0] - reach_x) : peak[0] + reach_x + 1] = -1.0
    if float(elsewhere.max()) > coarse[3] - max(_MIN_CLEAR, (1.0 - least_fit) / 2):
        return None
    # refined round what the reduced search found, on a larger copy (its place then within a pixel or two at full size,
    # which the comparison once lined up allows for)
    fine = min(1.0, _REFINE_SIDE_PX / max(x1 - x0, y1 - y0))
    slack = int(math.ceil(fine / work)) + 2  # (a pixel of the reduced search, in the larger copy, and some)
    fine_template = cv2.resize(template, None, fx=fine, fy=fine, interpolation=cv2.INTER_AREA) if fine < 1 else template
    best = None
    for change in _REFINE_SCALES:
        scale = coarse[2] + change
        w, h = round(fine_template.shape[1] * scale), round(fine_template.shape[0] * scale)
        fx0, fy0 = int(coarse[0] * fine) - slack, int(coarse[1] * fine) - slack
        rx0, ry0 = max(0, fx0), max(0, fy0)
        rx1, ry1 = min(int(width * fine), fx0 + w + 2 * slack), min(int(height * fine), fy0 + h + 2 * slack)
        if w < 8 or h < 8 or rx1 - rx0 < w or ry1 - ry0 < h:
            continue
        window = suspect_grey(int(ry0 / fine), int(math.ceil(ry1 / fine)), int(rx0 / fine), int(math.ceil(rx1 / fine)))
        if fine < 1:
            window = cv2.resize(window, (rx1 - rx0, ry1 - ry0), interpolation=cv2.INTER_AREA)
        if window.shape[0] < h or window.shape[1] < w:
            continue
        sized = cv2.resize(fine_template, (w, h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        _low, fit, _at_low, (bx, by) = cv2.minMaxLoc(cv2.matchTemplate(window, sized, cv2.TM_CCOEFF_NORMED))
        if best is None or fit > best[3]:
            best = (int(round((rx0 + bx) / fine)), int(round((ry0 + by) / fine)), scale, float(fit))
    if best is None or best[3] < least_fit * _resize_cost(fine_template, best[2]) or best[3] <= home:
        return None
    return best[0], best[1], best[2]


def _displaced(original: np.ndarray, suspect: np.ndarray, box: tuple[int, int, int, int], reach: int) -> bool:
    """Whether the original's picture in `box` fits the suspect best somewhere other than where it is, or at another
    size -- looked for roughly (the reduced search `_found` begins with), as a first sieve."""
    height, width = original.shape[:2]
    x0, y0, x1, y1 = box
    template = _grey(original[y0:y1, x0:x1])
    if float(template.std()) < _MIN_STRUCTURE:
        return False
    work = min(1.0, _WORK_SIDE_PX / max(x1 - x0, y1 - y0))
    sx0, sy0 = max(0, x0 - reach), max(0, y0 - reach)
    sx1, sy1 = min(width, x1 + reach), min(height, y1 + reach)
    search = cv2.resize(_grey(suspect[sy0:sy1, sx0:sx1]), None, fx=work, fy=work, interpolation=cv2.INTER_AREA)
    small = cv2.resize(template, None, fx=work, fy=work, interpolation=cv2.INTER_AREA)
    best = None
    for scale in _SCALES:
        sized = cv2.resize(small, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        if sized.shape[0] < 8 or sized.shape[1] < 8 or sized.shape[0] > search.shape[0] or sized.shape[1] > search.shape[1]:
            continue
        _low, fit, _at_low, (bx, by) = cv2.minMaxLoc(cv2.matchTemplate(search, sized, cv2.TM_CCOEFF_NORMED))
        if best is None or fit > best[3]:
            best = (sx0 + bx / work, sy0 + by / work, scale, float(fit))
    if best is None:
        return False
    x, y, scale, _fit = best
    # where it would lie unmoved at that size (sized round its centre), and how far off the rough search can be
    cx, cy = x0 + (x1 - x0) * (1 - scale) / 2, y0 + (y1 - y0) * (1 - scale) / 2
    slack = max(_MIN_MOVE_PX, 2.0 / work)
    return math.hypot(x - cx, y - cy) > slack or abs(scale - 1.0) > _SCALE_SLACK


_SCALE_SLACK = 0.06  # (the rough search steps sizes 5% apart: more than a step off is another size)


def _same(
    original: np.ndarray,
    suspect: np.ndarray,
    box: tuple[int, int, int, int],
    at: tuple[int, int, float],
    yardstick: _Yardstick,
    recheck: DetectionConfig | None = None,
) -> bool:
    """Whether the original's picture in `box`, laid over the suspect where it was found (at that size), is as like it as
    this pair's unchanged print is: the same shapes (grey) AND the same colours (which grey does not see) -- and, with
    `recheck` (the comparison's own settings, calibrated to the pair), nothing found once the two are compared again
    lined up: a batch number re-set as well as re-placed is a few changed digits, too little to move the fit of the
    whole, and exactly what the comparison finds."""
    x0, y0, x1, y1 = box
    x, y, scale = at
    w, h = x1 - x0, y1 - y0
    tw, th = round(w * scale), round(h * scale)
    height, width = suspect.shape[:2]
    if x < 0 or y < 0 or x + tw > width or y + th > height:
        return False
    there = cv2.resize(suspect[y : y + th, x : x + tw], (w, h), interpolation=cv2.INTER_AREA if scale > 1 else cv2.INTER_LINEAR)
    here = original[y0:y1, x0:x1]
    soft_here, soft_there = _soften(here, max(yardstick.softness, 0.0)), _soften(there, max(-yardstick.softness, 0.0))
    if _fit(soft_there, soft_here) < _least_fit(_grey(soft_here), yardstick) * _resize_cost(_grey(soft_here), scale):
        return False
    if here.ndim != 3:
        return True
    allowed = yardstick.off + _edge_share(here, scale, yardstick.tolerance)
    if _off_share(soft_here, soft_there, yardstick.expected, yardstick.tolerance, yardstick.shift) > allowed:
        return False
    if recheck is None:
        return True
    # (compared softened alike, as everything here is: a proof's softer fine print is the pair's, not a change)
    again = local_diff.detect(np.ascontiguousarray(soft_here), np.ascontiguousarray(soft_there), recheck)
    # (what it finds as recoloured areas is the colours, judged above against how this pair prints them)
    found = again.candidates[: len(again.candidates) - again.colour_regions]
    if not found:
        return True
    own = _own_marks(here, yardstick.tolerance)
    return not any(
        own[max(0, int(c.bbox.y0)) : int(math.ceil(c.bbox.y1)), max(0, int(c.bbox.x0)) : int(math.ceil(c.bbox.x1))].any()
        for c in found
    )


def _own_marks(here: np.ndarray, tolerance: np.ndarray) -> np.ndarray:
    """The element's own print in the crop `here`: what stands off the crop's ground (its commonest colour round the
    edge, by more than the pair lets that colour stray) in marks wholly inside the crop -- a mark the crop's edge cuts
    through is something round the element (a panel's edge, a neighbour), which moving the element moves away from."""
    lab = cv2.cvtColor(np.ascontiguousarray(here), cv2.COLOR_RGB2LAB).astype(np.float32)
    rim = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    rim_bins = _bins(rim)
    ground_bin = int(np.bincount(rim_bins).argmax())
    ground = np.median(rim[rim_bins == ground_bin], axis=0)
    marks = (np.linalg.norm(lab - ground, axis=2) > tolerance[ground_bin]).astype(np.uint8)
    count, labels, stats, _centres = cv2.connectedComponentsWithStats(marks, connectivity=8)
    h, w = marks.shape
    inside = np.zeros(count, bool)
    for index in range(1, count):
        x, y, bw, bh = stats[index, :4]
        inside[index] = x > 0 and y > 0 and x + bw < w and y + bh < h
    return inside[labels]


def matched_elements(
    clusters: list[DiffCluster], original: np.ndarray, suspect: np.ndarray, recheck: DetectionConfig | None = None
) -> list[DiffCluster]:
    """`clusters` with every group of them that is one element moved or resized made one finding (see the module).
    `recheck`: the pixel comparison's settings (calibrated to this pair), to compare each element again once lined up."""
    height, width = original.shape[:2]
    long_edge = max(height, width)
    pieces = [
        c for c in clusters
        if c.category in ("graphics", "colour", "text") and c.subtype is not DiffSubtype.MOVED
        and not any(m.printers_note for m in c.members)
        and max(c.bbox_in_original.x1 - c.bbox_in_original.x0, c.bbox_in_original.y1 - c.bbox_in_original.y0)
        <= _PIECE_MAX_SHARE * long_edge
    ]
    if len(pieces) < _MIN_PIECES:
        return clusters
    def as_element(group: list[DiffCluster]) -> tuple[list[DiffCluster], tuple[int, int, int, int]] | None:
        if len(group) < _MIN_PIECES:
            return None
        gx0 = min(c.bbox_in_original.x0 for c in group)
        gy0 = min(c.bbox_in_original.y0 for c in group)
        gx1 = max(c.bbox_in_original.x1 for c in group)
        gy1 = max(c.bbox_in_original.y1 for c in group)
        if max(gx1 - gx0, gy1 - gy0) > _ELEMENT_MAX_SHARE * long_edge:
            return None
        pad = max(12.0, _PAD_SHARE * max(gx1 - gx0, gy1 - gy0))
        box = (max(0, int(gx0 - pad)), max(0, int(gy0 - pad)), min(width, math.ceil(gx1 + pad)), min(height, math.ceil(gy1 + pad)))
        return (group, box) if box[2] - box[0] >= 16 and box[3] - box[1] >= 16 else None

    elements = [e for e in (as_element(g) for g in _groups(pieces, _GROUP_SHARE * long_edge)) if e is not None]
    if not elements:
        return clusters
    replaced: dict[int, DiffCluster] = {}
    absorbed: set[int] = set()
    with StageTimer(logger, "elements_matched"):
        # First, cheaply: does any of them fit best anywhere but where it is? If none does, nothing moved, and the pair's
        # yardstick (the costly part) need not be measured at all.
        reach = int(_SEARCH_SHARE * long_edge)
        with ThreadPoolExecutor(max_workers=max(1, min(8, os.cpu_count() or 2))) as pool:
            displaced = list(pool.map(lambda element: _displaced(original, suspect, element[1], reach), elements))
        if not any(displaced):
            return clusters
        size = (
            int(np.median([box[2] - box[0] for _group, box in elements])),
            int(np.median([box[3] - box[1] for _group, box in elements])),
        )
        yardstick = _yardstick(original, suspect, clusters, size, [box for _group, box in elements])
        if yardstick is None:
            return clusters
        logger.info(
            "element yardstick measured",
            extra={"fit": round(yardstick.fit, 3), "off": round(yardstick.off, 3), "softness": yardstick.softness},
        )
        with ThreadPoolExecutor(max_workers=max(1, min(8, os.cpu_count() or 2))) as pool:
            founds = list(pool.map(lambda element: _found(original, suspect, element[1], reach, yardstick), elements))
            # A group found nowhere whole may be neighbours each moved its own way (a barcode and the label under it):
            # its parts that overlap one another are looked for each on its own.
            parts = [
                part
                for (group, _box), found in zip(elements, founds)
                if found is None
                for sub in _groups(group, 0.0)
                if len(sub) < len(group) and (part := as_element(sub)) is not None
            ]
            founds += list(pool.map(lambda element: _found(original, suspect, element[1], reach, yardstick), parts))
            elements += parts
        for (group, box), found in zip(elements, founds):
            if found is None:
                continue
            x, y, scale = found
            dx, dy = x - box[0], y - box[1]
            if math.hypot(dx, dy) < _MIN_MOVE_PX and abs(scale - 1.0) < _MIN_RESIZE:
                continue
            if not _same(original, suspect, box, found, yardstick, recheck):
                # Moved AND changed (a batch label re-placed and re-set): its findings stay as they are, but where it now
                # is is known -- the suspect's side of each is there, so its words are read where they are printed.
                for cluster in group:
                    b = cluster.bbox_in_original
                    cluster.moved_to = BoundingBox(
                        x + (b.x0 - box[0]) * scale, y + (b.y0 - box[1]) * scale, x + (b.x1 - box[0]) * scale, y + (b.y1 - box[1]) * scale
                    )
                continue
            one = combined(group)
            one.type = DiffType.GRAPHICS
            for member in one.members:
                member.subtype = DiffSubtype.MOVED
            w, h = box[2] - box[0], box[3] - box[1]
            one.moved_to = BoundingBox(x, y, x + w * scale, y + h * scale)
            resized = f", {100 * scale:.0f}% of its size" if abs(scale - 1.0) >= _MIN_RESIZE else ""
            how = movement.how_far(dx + (w * scale - w) / 2, dy + (h * scale - h) / 2)
            one.detail = f"Graphic moved{' ' + how if how else ''}{resized} (the same picture, re-placed)"
            first = min(group, key=lambda c: c.id)
            replaced[id(first)] = one
            absorbed.update(id(c) for c in group if c is not first)
    if not replaced:
        return clusters
    out = [replaced.get(id(c), c) for c in clusters if id(c) not in absorbed]
    for number, cluster in enumerate(out, start=1):
        cluster.id = number
    logger.info("elements found moved", extra={"elements": len(replaced), "findings_joined": len(absorbed) + len(replaced)})
    return out
