"""Scan-vs-Digital preprocessing (SRS FR-6.3, Architecture §3.3.4).

Flattens both files to the same black-on-white representation so a scan can
be compared against a digital original without every lighting and ink
variation reading as a difference. Colour comparison is disabled separately,
by the mode's `DetectorOptions`, so it is skipped rather than merely hidden.

`ink_image` is the picture that is read for text and compared: the file with its
background colours removed to white and everything printed on it black (and, each stroke
cut at its true edge, shown: `InkPictures`).
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import cv2
import numpy as np

from app.config import DetectionConfig

# The illumination field is estimated on a small copy of the page. Working
# at this size keeps the estimate smooth and makes a very wide neighbourhood
# affordable, which matters: any background estimator whose window is
# narrower than a solid artwork element will read that element as background
# and erase it. An early version used a fixed 31 px window and deleted a
# 100 px solid roundel outright.
_ILLUMINATION_LONG_EDGE_PX = 256
_ILLUMINATION_RADIUS_DIVISOR = 8


def illumination_field(gray: np.ndarray) -> np.ndarray:
    """A smooth estimate of the page's lighting, ignoring artwork detail."""
    height, width = gray.shape
    scale = _ILLUMINATION_LONG_EDGE_PX / max(height, width)
    small = (
        cv2.resize(gray, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        if scale < 1
        else gray.copy()
    )

    # Local maximum: on a page, the brightest value nearby is the paper.
    radius = max(2, max(small.shape) // _ILLUMINATION_RADIUS_DIVISOR)
    paper = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
    paper = cv2.GaussianBlur(paper, (0, 0), max(1.0, radius / 2))
    return cv2.resize(paper, (width, height), interpolation=cv2.INTER_LINEAR)


_FILL_LONG_EDGE_PX = 700  # what is outside the compared area is filled from its neighbours on a copy this size
_PAPER_LONG_EDGE_PX = 256  # the paper is estimated on a copy this size
_PAPER_BRIGHT_SHARE = 0.8  # pixels at least this share of the page's white count as paper
_PAPER_SIGMA_SHARE = 0.10  # how far (share of the long edge) each paper sample speaks for
_PAPER_FALLBACK_WEIGHT = 0.02  # far from any paper, the estimate relaxes to the page's white


def _paper_field(gray: np.ndarray) -> np.ndarray:
    """How bright blank paper is at every point, on a copy at most `_PAPER_LONG_EDGE_PX` long (float32).

    A scan is lit unevenly (dark corners, a lamp gradient), and that has to be divided out. A smooth surface is
    fitted through the BRIGHT parts of the page only, so a large dark graphic is not mistaken for shade: an
    estimate taken from the brightest pixel in a window (as `illumination_field` does) reads any solid block
    wider than its window as the paper itself, and the block vanishes.
    """
    height, width = gray.shape
    scale = min(1.0, _PAPER_LONG_EDGE_PX / max(height, width))
    small = cv2.resize(gray, (max(2, round(width * scale)), max(2, round(height * scale))), interpolation=cv2.INTER_AREA).astype(np.float32)
    white = float(np.percentile(small, 95))
    bright = (small >= _PAPER_BRIGHT_SHARE * white).astype(np.float32)
    sigma = max(2.0, _PAPER_SIGMA_SHARE * max(small.shape))
    weight = cv2.GaussianBlur(bright, (0, 0), sigma)
    total = cv2.GaussianBlur(small * bright, (0, 0), sigma)
    return (total + _PAPER_FALLBACK_WEIGHT * white) / (weight + _PAPER_FALLBACK_WEIGHT)


def _filled_outside(image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """`image` with everything outside `valid` replaced by the nearest pixel that is inside it.

    Outside the compared area the page is white by design, and white beside the darkened corner of a scan
    would make that corner look like ink and pull the estimate of the paper towards white. What is estimated
    from must be the picture itself, carried a little way out.
    """
    height, width = valid.shape
    scale = min(1.0, _FILL_LONG_EDGE_PX / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    inside = cv2.resize(valid.astype(np.uint8), size, interpolation=cv2.INTER_AREA) > 0  # any inside coverage counts
    if not inside.any():
        return image
    _, labels = cv2.distanceTransformWithLabels(
        (~inside).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    nearest_y, nearest_x = np.nonzero(inside)  # labels number the zero pixels in raster order, from 1
    filled = small[nearest_y[labels - 1].clip(0, size[1] - 1), nearest_x[labels - 1].clip(0, size[0] - 1)]
    filled = np.where(inside[..., None], small, filled)
    result = cv2.resize(filled, (width, height), interpolation=cv2.INTER_LINEAR)
    np.copyto(result, image, where=valid[..., None])  # (in place: no second full-size copy of the page)
    return result


_TINT_SCALE = 0.25  # the tint and the solid graphics are worked out on a copy this share of the size
_MIN_PAPER = 60  # the paper is never taken to be darker than this when it is divided out
_RAMP_SAMPLE_STEP = 8  # the strength of the ink is judged from every 8th pixel each way
_RAMP_MIN_INK = 40  # only pixels at least this much ink say how strong the ink is
_RAMP_MIN_SAMPLES = 200
_EDGE_REACH_PX = 4  # a stroke's strength at a pixel: the most ink within this far (half a bold stroke and its blur)...
_EDGE_SHARE = (0.4, 0.6)  # ...and a pixel is none of it below this share of that, all of it above (its edge: half)


def _smoothstep(values: np.ndarray, low: float | np.ndarray, high: float | np.ndarray) -> np.ndarray:
    """`low`/`high` may each be a scalar or a per-pixel array the same shape as `values`."""
    t = np.clip((values - low) / np.maximum(np.asarray(high) - np.asarray(low), 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _ink_contrast(
    image: np.ndarray, config: DetectionConfig, valid: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """(how much each pixel is ink, uint8: 0 = background .. 255 = full ink; the pixels outside `valid`, if any).

    * **The paper is divided out first**, so a page lit unevenly (a scan's darker corners) is flat. It is
      estimated from the bright parts of the page alone, so a big dark graphic is not taken for shade.
    * **Ink is darkness in ANY channel** (the darkest of red, green and blue). A saturated ink absorbs
      one channel: cyan, yellow, orange and magenta lettering are all dark in one of them, though a
      grey-scale conversion would call yellow or cyan text nearly as light as the paper. A pale tint is
      light in all three, so it stays background.
    * **A tint is estimated locally and subtracted.** The median darkness over a window wider than a
      stroke is the background there, so black text on a 40% panel is black on white, not black on a
      grey block that would swallow it.
    * **A solid graphic is told from a tint by how DARK it looks, and gradually.** A dark bar or a reversed-out
      block (a dark red, a navy, a green) stays black with its white lettering white, but a pale yellow or
      blue panel is light to the eye and is a tint. It used to be decided by a cut-off on the darkest channel,
      and a panel a few levels either side of it (a scan is a little lighter than the file it came from) came
      out black in one file and white in the other. Now how solid a region is (0 = tint, 1 = solid) is a smooth
      function of the median of LUMINANCE and the darkest channel over the window, and it makes the region that
      dark, so a panel a few levels lighter is a few levels less black, never the opposite colour. Luminance alone
      called a saturated orange or red light, being bright to the eye, and a reversed white banner on one came out
      as a smear: the darkest channel is included because a saturated colour is dark in it even when it reads
      bright overall, which is exactly why it is also how a pixel is judged to be ink in the first place. A
      colourless grey in that same ambiguous middle band is held to a stricter (darker) bar first: unlike a
      printed reversed panel, always some definite ink colour, a neutral grey there is at least as often a
      photograph's own backdrop, which must not render as a false solid black block.

    All of it is done in 8 bits, at a quarter of the size where a window is needed: at full size on an 18 MP
    carton the float version took over a second per file. Every full-size picture is let go as soon as it has
    been used, and the last steps write into pictures no longer needed: a 100 MP page (a 2 m shipper) otherwise
    held some twenty full-size copies at once, several GB for the two files.
    """
    height, width = image.shape[:2]
    outside = None
    if valid is not None and not valid.all():
        outside = ~valid
        image = _filled_outside(image, valid)
    small_size = (round(width * _TINT_SCALE), round(height * _TINT_SCALE))  # what `cv2.resize(fx=_TINT_SCALE)` gives
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    paper_small = _paper_field(gray)
    luminance = cv2.resize(gray, small_size, interpolation=cv2.INTER_AREA).astype(np.float32)
    del gray
    paper = cv2.resize(np.clip(paper_small, _MIN_PAPER, 255).astype(np.uint8), (width, height), interpolation=cv2.INTER_LINEAR)

    red, green, blue = cv2.split(image)
    del image  # (only this function's own filled copy, if one was made: the caller's picture is untouched)
    darkest = cv2.min(red, green)
    cv2.min(darkest, blue, dst=darkest)
    brightest = cv2.max(red, green)
    cv2.max(brightest, blue, dst=brightest)
    del red, green, blue
    chroma = cv2.resize(cv2.subtract(brightest, darkest, dst=brightest), small_size, interpolation=cv2.INTER_AREA)
    del brightest
    darkness = cv2.divide(darkest, paper, dst=darkest, scale=255.0)
    cv2.bitwise_not(darkness, dst=darkness)  # 0 for paper, 255 for black or a saturated ink
    del paper

    reduced = cv2.resize(darkness, None, fx=_TINT_SCALE, fy=_TINT_SCALE, interpolation=cv2.INTER_AREA)  # (by factor: not the same weights as by size)
    window = max(3, int(round(config.ink_window_fraction * max(height, width) * _TINT_SCALE)) | 1)
    tint_small = cv2.medianBlur(reduced, window)

    paper_here = cv2.resize(np.maximum(paper_small, _MIN_PAPER), small_size, interpolation=cv2.INTER_LINEAR)
    luminance_darkness = 255.0 - np.clip(luminance * (255.0 / paper_here), 0.0, 255.0)
    solid_darkness = (luminance_darkness + reduced.astype(np.float32)) / 2.0
    typical = cv2.medianBlur(solid_darkness.astype(np.uint8), window).astype(np.float32)

    chroma_typical = cv2.medianBlur(chroma, window).astype(np.float32)
    neutral_shift = config.ink_solid_neutral_margin * (1.0 - _smoothstep(chroma_typical, 0.0, config.ink_solid_chroma_px))
    solid_weight = _smoothstep(typical, config.ink_solid_low + neutral_shift, config.ink_solid_high + neutral_shift)
    solid_small = (255.0 * solid_weight).astype(np.uint8)

    tint = cv2.resize(tint_small, (width, height), interpolation=cv2.INTER_LINEAR)
    solid = cv2.resize(solid_small, (width, height), interpolation=cv2.INTER_LINEAR)
    room = cv2.bitwise_not(tint)
    cv2.max(room, 1, dst=room)  # what is left of the scale above the tint
    relative = cv2.subtract(darkness, tint, dst=tint)
    cv2.divide(relative, room, dst=relative, scale=255.0)
    del room
    ink_solid = cv2.multiply(darkness, solid, dst=solid, scale=1.0 / 255.0)
    contrast = cv2.max(relative, ink_solid, dst=relative)
    return contrast, outside


_STREAK_ROW_STEP = 2  # a streak runs the length of the page: every other row says how strong the ink is in a column
_STREAK_TILE_ROWS = 160  # ...judged in strips this many (sampled) rows tall, each on its own
_STREAK_MIN_INK = 25  # pixels at least this much ink say how strong a column's ink is (a washed-out stroke included)
_STREAK_MIN_INKED = 6  # ...if a strip has at least this many in that column
_STREAK_SHARE = 0.9  # a column's ink strength: the strongest tenth of its ink (the cores of its strokes)
_STREAK_NEIGHBOURS_PX = 151  # ...set against the columns round it this wide (a streak is a few to a few dozen wide)
_STREAK_MIN_STRIPS = 3  # a streak is in most strips: fewer strips cannot tell it from what is printed
_STREAK_SLACK_PX = 4  # the other file's column is looked for this near (the two are registered, never exactly)
_STREAK_NOISE_SIGMAS = 3.0  # a column weaker than the other file's by this many of the profiles' own spreads is a streak
_STREAK_MIN_DIP = 0.15  # ...and by at least this share (a streak a little lighter than that the ramp takes in its stride)
_STREAK_MAX_GAIN = 4.0  # a streak's ink is made at most this much stronger
_STREAK_BAND_ROWS = 1024  # (the gain is applied this many rows at a time: a float copy of a whole page is large)


def _column_strength(contrast: np.ndarray) -> np.ndarray | None:
    """How strong each column's ink is against the columns round it (1.0 = as strong), or None if the picture is too
    small or too blank to say. Judged in strips down the page and the median over the strips taken: what is printed
    differs from strip to strip (a heading in one, a photograph in another) and is outvoted; what runs the length of
    the page -- a scanner's streak, or a stripe printed down it -- is in every strip."""
    sampled = contrast[::_STREAK_ROW_STEP]
    width = sampled.shape[1]
    if width < _STREAK_NEIGHBOURS_PX or sampled.shape[0] < _STREAK_MIN_STRIPS * _STREAK_TILE_ROWS:
        return None
    columns = np.arange(width)
    ratios = []
    for top in range(0, sampled.shape[0] - _STREAK_TILE_ROWS // 2, _STREAK_TILE_ROWS):
        strip = sampled[top : top + _STREAK_TILE_ROWS]
        inked = (strip >= _STREAK_MIN_INK).sum(axis=0)
        # the strong tenth of each column's ink: sorted, the ink lies at the top of each column (the rest is below it)
        ordered = np.sort(np.where(strip >= _STREAK_MIN_INK, strip, 0), axis=0)
        rank = strip.shape[0] - 1 - np.floor((1.0 - _STREAK_SHARE) * np.maximum(inked - 1, 0)).astype(np.int64)
        strength = ordered[rank, columns].astype(np.float32)
        known = inked >= _STREAK_MIN_INKED
        if known.sum() < 0.2 * width:
            continue  # (a strip mostly blank: nothing to judge by)
        filled = np.interp(columns, columns[known], strength[known]).astype(np.float32)
        around = cv2.blur(filled.reshape(1, -1), (_STREAK_NEIGHBOURS_PX, 1), borderType=cv2.BORDER_REFLECT).ravel()
        ratio = filled / np.maximum(around, 1.0)
        ratio[~known] = np.nan
        ratios.append(ratio)
    if len(ratios) < _STREAK_MIN_STRIPS:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # (a column blank in every strip has no median: it is 1.0)
        ratio = np.nanmedian(np.array(ratios), axis=0)
    return np.where(np.isnan(ratio), 1.0, ratio).astype(np.float32)


def _streaks_evened(scan: np.ndarray, digital: np.ndarray) -> None:
    """A scanner's streaks evened out of `scan`, in place, against `digital` (the file it is a scan of, ALIGNED to it):
    narrow bands down the page where the lamp or the sensor gave less, and every stroke crossing one came out lighter
    (a band of washed-out letters, half of a letter missing: 'd montré' for 'démontré', read as a change). A column of
    the scan whose ink is weaker, against its neighbours, than the digital file's same column (`_column_strength`) is
    made as strong as that -- a grey stripe printed down the artwork is as weak in both, and is left as it is. Columns
    only, the bands down the page as the files are laid out: across a line of text, rows differ by themselves (a row
    through the letters' tops is weaker than one through their middles), so a row profile cannot tell a streak from
    the type."""
    own, other = _column_strength(scan), _column_strength(digital)
    if own is None or other is None or own.shape != other.shape:
        return
    nearby = cv2.erode(other.reshape(1, -1), np.ones((1, 2 * _STREAK_SLACK_PX + 1), np.uint8)).ravel()
    relative = own / np.maximum(nearby, 1e-3)  # (weaker even than the digital's weakest column close by)
    spread = 1.4826 * float(np.median(np.abs(relative - np.median(relative))))  # (the profiles' own noise)
    streak = relative < 1.0 - max(_STREAK_NOISE_SIGMAS * spread, _STREAK_MIN_DIP)
    if not streak.any():
        return
    gain = np.ones(relative.size, np.float32)
    gain[streak] = np.minimum(1.0 / np.maximum(relative[streak], 1e-3), _STREAK_MAX_GAIN)
    for top in range(0, scan.shape[0], _STREAK_BAND_ROWS):
        band = scan[top : top + _STREAK_BAND_ROWS]
        band[:] = np.clip(band.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def _ramp(contrast: np.ndarray, config: DetectionConfig) -> np.ndarray:
    """A 256-entry table turning how much a pixel is ink into how black it comes out (255 = white .. 0 = black).

    Ink is fully black from `ink_contrast_high` and speckle below `ink_contrast_low` is white, but a scan's
    strokes are lighter and softer than a digital file's, so the ramp follows how strong THIS picture's ink
    is (its 90th percentile): on a light scan a stroke of half the usual strength is still fully black
    instead of breaking up into grey speckle.
    """
    sample = contrast[::_RAMP_SAMPLE_STEP, ::_RAMP_SAMPLE_STEP]
    inked = sample[sample >= _RAMP_MIN_INK]
    high, low = config.ink_contrast_high, config.ink_contrast_low
    if inked.size >= _RAMP_MIN_SAMPLES:
        strength = float(np.percentile(inked, 90))
        high = float(np.clip(config.ink_ramp_share * strength, config.ink_ramp_min_high, config.ink_contrast_high))
        low = float(np.clip(config.ink_ramp_low_share * high, config.ink_ramp_min_low, config.ink_contrast_low))
    levels = np.arange(256, dtype=np.float32)
    ink = np.clip((levels - low) / max(high - low, 1.0), 0.0, 1.0)
    return (255.0 * (1.0 - ink)).round().astype(np.uint8)


def _beyond_edges(contrast: np.ndarray) -> np.ndarray:
    """Where each stroke's EDGE is (for the picture shown, `InkPictures`): 255 = beyond it (paper, however much ink the
    ramp makes of it), 0 = within it, graded between. A pixel is within a stroke if it is at least half as much ink as
    the stroke itself -- its strongest pixel close by. A scan blurs every edge, and the ramp -- black from a fifth or so of
    full strength, so faint thin type stays whole -- takes the blur's outer half for ink: a scan's type comes out bolder
    than the file's, its small print filled in. Half of a blurred edge is where the edge really is, however soft the scan,
    and each stroke is measured against itself, so a faint thin stroke stays as whole as a bold one. (A digital file's
    edges are sharp: its picture hardly changes.)"""
    peak = cv2.dilate(contrast, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * _EDGE_REACH_PX + 1,) * 2))
    cv2.max(peak, 1, dst=peak)
    share = cv2.divide(contrast, peak, dst=peak, scale=255.0)
    low, high = _EDGE_SHARE
    within = _smoothstep(np.arange(256, dtype=np.float32) / 255.0, low, high)
    return cv2.LUT(share, (255.0 * (1.0 - within)).round().astype(np.uint8), dst=share)


def ink_image(
    image: np.ndarray, config: DetectionConfig | None = None, valid: np.ndarray | None = None
) -> np.ndarray:
    """The file as ink on paper: a grey image, 255 where there is only background, 0 where there is ink.

    Every colour that is only a background -- a pale panel, a tint, a scan's cream paper or its uneven
    lighting -- comes out white, and whatever is printed on it comes out black, whatever its colour.
    That is what lets a scan be compared with its digital original by *what is printed*, since scanning
    changes every colour. It is the picture text is read from (the reviewer sees it cut at its edges: `ink_pair`); a ramp
    (not a cut-off) keeps edges soft and drops speckle below it. See `_ink_contrast` for how ink is told
    from background. `valid` marks where the picture really is (the compared area): the rest is
    background by construction, comes out white, and is not allowed to influence the estimates near it.

    Nothing is tuned to a file: the paper, the tint and the noise floor are all measured from its pixels.
    """
    config = config or DetectionConfig()
    contrast, outside = _ink_contrast(image, config, valid)
    result = cv2.LUT(contrast, _ramp(contrast, config))
    _panels_opened(image, result, valid)
    if outside is not None:
        result[outside] = 255
    return result


# (box, where in it, what the picture showed there before, whether its type is dark on it -- None: it was paper already)
_Opened = tuple[tuple[int, int, int, int], np.ndarray, np.ndarray, bool | None]


class InkPictures(NamedTuple):
    """`ink_pair`'s pictures. `first`/`second` are what text is read from and ink compared by. `shown_first`/`shown_second`
    are the same with each stroke cut at its true edge (`_beyond_edges`): what the reviewer is shown as black and white,
    a scan's type no bolder than the file's. (Not read from: Tesseract was measured reading the bolder picture better --
    on ten scans, more misreads from the edged one.)"""

    first: np.ndarray
    second: np.ndarray
    shown_first: np.ndarray
    shown_second: np.ndarray


def ink_pair(
    first: np.ndarray, second: np.ndarray, config: DetectionConfig | None = None, valid: np.ndarray | None = None
) -> InkPictures:
    """`ink_image` of two ALIGNED pictures of one artwork (a digital file and its scan, in one frame, in that order), the
    scan's streaks evened out against the digital (`_streaks_evened`), and made to agree on their coloured panels. Where one opens a panel (paper, its type ink: `_panels_opened`) and the other shows the same
    place as a solid block (a scan's shaded, grainy colour, not one flat colour to be opened), the two are made to show it
    the same way -- rather than paper in one and a block in the other, which read as whole areas missing:

    * dark type on it (black on a red box) -- the other is opened too, its type what is clearly darker than its panel;
    * light type reversed out of it (white on a green) -- the first is left a block as well: its type is still there to
      read, light on dark, in both (`text_diff._readable` reads it either way round).
    """
    config = config or DetectionConfig()
    made = []
    edges = []
    contrasts, outsides = zip(*(_ink_contrast(image, config, valid) for image in (first, second)))
    contrasts = list(contrasts)
    _streaks_evened(contrasts[1], contrasts[0])  # (the second is the scan: where it went through a streak)
    for image, outside in zip((first, second), outsides):
        contrast = contrasts.pop(0)
        ink = cv2.LUT(contrast, _ramp(contrast, config))
        edges.append(_beyond_edges(contrast))
        del contrast
        before = ink.copy()
        made.append((image, ink, before, _panels_opened(image, ink, valid), outside))
    for _image, _ink, _before, opened, _outside in made:
        for (x0, y0, x1, y1), where, _shown_before, dark_type in opened:
            # A panel opened: shown as `ink_pair` settles it, not cut at the strokes' edges -- inside a panel the edges
            # are the panel's, and its type reversed out of it the LEAST ink there: cut, white type on a green came out
            # paper. (Not only where the type is lighter than the panel than darker: a few dark marks on it as well --
            # Braille dots, a scan's grain -- and the white type was lost all the same.)
            if dark_type is not None:
                for edge in edges:
                    edge[y0:y1, x0:x1][where] = 0
    for (_image, ink, _before, opened, _outside), (_other, _other_ink, other_before, other_opened, _o) in (
        (made[0], made[1]),
        (made[1], made[0]),
    ):
        for box, where, shown_before, dark_type in opened:
            if dark_type is None or any(_boxes_overlap(box, other_box) for other_box, *_rest in other_opened):
                continue  # already paper here (nothing was undone), or the other opened it too
            x0, y0, x1, y1 = box
            there = other_before[y0:y1, x0:x1][where]
            if not there.size or float(np.median(there)) > _PANEL_SHOWN_BLACK:
                continue
            # The other shows this panel as a solid block. DARK type on it (black on a red warning box) is then lost in
            # both unless the other is opened too -- and dark type is what can be told from a panel reliably, however
            # shaded (a scan's grain, its uneven red): each pixel clearly darker than the panel round it is its type.
            # Taken only if it gives the type this picture shows there; otherwise, and for LIGHT type (reversed out,
            # still readable in a block), this one is left a block as well.
            if dark_type:
                opened_there = _dark_type_opened(_other, box, where)
                if _same_type(ink[y0:y1, x0:x1], opened_there, where):
                    _other_ink[y0:y1, x0:x1][where] = opened_there
                    continue
            ink[y0:y1, x0:x1][where] = shown_before  # a block here too, as it was
    results, shown = [], []
    for (_image, ink, _before, _opened, outside), edge in zip(made, edges):
        if outside is not None:
            ink[outside] = 255
        results.append(ink)
        shown.append(cv2.max(ink, edge, dst=edge))
    if config.shown_colour_panels_removed:
        # (both files the same way: one area, one tolerance -- else a band lighter than its panel was a mark in one file
        # and panel in the other)
        shown = _colour_panels_removed((first, second), shown)
        for picture, (_image, _ink, _before, _opened, outside) in zip(shown, made):
            if outside is not None:
                picture[outside] = 255
    return InkPictures(results[0], results[1], shown[0], shown[1])


_REMOVAL_SCALE = 0.25  # coloured panels are found, and their colour taken, on a copy this share of the size
_REMOVAL_MIN_CHROMA = 20.0  # (8-bit Lab) a pixel at least this colourful is a panel's (not paper, grey, black or white type)
_REMOVAL_JOIN_SHARE = 0.015  # a panel's pixels this close (a share of the long edge) are one panel, its type inside it
_REMOVAL_SOLID_SHARE = 0.55  # an area this much coloured (over that window) is a panel; type in colour on paper is less
_REMOVAL_CORE_PX = 20  # (full size) ...and solid this wide somewhere in it: a bar, not a bold letter's stroke
_REMOVAL_SAME_PANEL = 70.0  # (Lab) a panel grows over pixels this near its core's colour, not over type of another
_REMOVAL_FILL_PX = 6.0  # (on the copy) the panel's colour behind its type is filled in from this far round it
_REMOVAL_SMOOTH_PX = 1.5  # ...and smoothed this much (a scan's grain), little enough that a band or a box keeps its own
_REMOVAL_MIN_SPREAD = 0.08  # a panel's colour from its inner pixels where they are this share of its pixels round about
_REMOVAL_SHOWN_DARK = 128  # only what the picture shows as a block (darker than this) is changed
_REMOVAL_MIN_TOLERANCE = 35.0  # (8-bit Lab distance) a pixel at least this far from its panel's colour is a mark on it
_REMOVAL_RAMP = 1.5  # (fully a mark at this many times the tolerance)


def _holes_filled(mask: np.ndarray, largest: int) -> np.ndarray:
    """`mask` with its holes no bigger than `largest` pixels filled: the letters knocked out of a panel are the panel's."""
    count, labels, stats, _centres = cv2.connectedComponentsWithStats((1 - mask).astype(np.uint8), connectivity=4)
    filled = mask.copy()
    height, width = mask.shape
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area <= largest and x > 0 and y > 0 and x + w < width and y + h < height:
            filled[labels == label] = 1
    return filled


def _near_own_colour(lab: np.ndarray, solid: np.ndarray, cores: np.ndarray, join: int) -> np.ndarray:
    """Where a pixel is near (`_REMOVAL_SAME_PANEL`) the colour of the panel it lies by: each solid area's colour is its
    core's (the median of it), spoken for as far as it may grow (`join`)."""
    near = np.zeros(solid.shape, np.uint8)
    count, labels = cv2.connectedComponents(solid, connectivity=8)
    reach = np.ones((join, join), np.uint8)
    for label in range(1, count):
        panel = labels == label
        core = panel & (cores > 0)
        if not core.any():
            continue
        colour = np.median(lab[core].astype(np.float32), axis=0)
        around = cv2.dilate(panel.astype(np.uint8), reach) > 0
        distance = np.linalg.norm(lab[around].astype(np.float32) - colour, axis=1)
        near[around] |= (distance <= _REMOVAL_SAME_PANEL).astype(np.uint8)
    return near


def _colour_panels_removed(images: tuple[np.ndarray, np.ndarray], pictures: list[np.ndarray]) -> list[np.ndarray]:
    """The two pictures shown with every coloured panel removed like the paper: the panel white, what is printed on it --
    white type knocked out of a green, black type on a red -- black.

    The panel's colour at each pixel is taken from the panel's own (coloured) pixels only: type -- white, black, grey,
    colourless -- is left out and its place filled in from the panel round it, so the colour behind a big letter is the
    panel's however big the letter, and a band or a box of another colour inside the panel keeps its own. A mark is what
    stands well off that colour. Only what either picture showed as a coloured block is changed, and in both files alike
    (the same area); type on paper, tints and black are as they were."""
    height, width = pictures[0].shape[:2]
    size = (max(1, round(width * _REMOVAL_SCALE)), max(1, round(height * _REMOVAL_SCALE)))
    join = max(3, int(round(_REMOVAL_JOIN_SHARE * max(size)))) | 1
    fields, region_small = [], np.zeros((size[1], size[0]), np.uint8)
    for image, picture in zip(images, pictures):
        lab = cv2.cvtColor(cv2.resize(image, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2LAB)
        chroma = np.hypot(lab[..., 1].astype(np.float32) - 128.0, lab[..., 2].astype(np.float32) - 128.0)
        panel_px = (chroma >= _REMOVAL_MIN_CHROMA).astype(np.uint8)
        dark = (cv2.resize(picture, size, interpolation=cv2.INTER_AREA) < _REMOVAL_SHOWN_DARK).astype(np.uint8)
        # a coloured BLOCK shown dark: solid colour wider than any stroke of type (green type on white paper is not a
        # panel -- its paper is), grown back to its edges, with the type inside it closed over
        coloured_dark = panel_px & cv2.dilate(dark, np.ones((3, 3), np.uint8))
        # (solid by density: a box densely lettered has no square of pure colour, but is mostly colour all over)
        solid = (cv2.boxFilter(coloured_dark.astype(np.float32), -1, (join, join)) >= _REMOVAL_SOLID_SHARE).astype(np.uint8)
        # ...and somewhere solid through and through, wider than any stroke: bold type in colour on paper (a brand name, a
        # line of purple capitals) is as dense over the window, and was taken for a panel -- its letters made paper, a smear
        # (a panel is kept whole if it has such a core anywhere: its lettering breaks the colour up everywhere else)
        core = max(3, int(round(_REMOVAL_CORE_PX * _REMOVAL_SCALE)) | 1)
        cores = cv2.erode(coloured_dark, np.ones((core, core), np.uint8))
        count, labels = cv2.connectedComponents(solid, connectivity=8)
        with_core = np.zeros(count, bool)
        with_core[labels[cores > 0]] = True
        with_core[0] = False
        solid = with_core[labels].astype(np.uint8)
        if solid.any():
            # (closed over a stroke of its type or a scan's streak only -- the type inside it is a hole in it, filled -- never
            # over the paper between two panels: a line of text between two heading bars was made one panel with them, its
            # paper then 'a mark' on the panel, and shown as a black band)
            stroke = max(3, join // 3) | 1
            grown = cv2.dilate(solid, np.ones((join, join), np.uint8)) & cv2.morphologyEx(coloured_dark, cv2.MORPH_CLOSE, np.ones((stroke, stroke), np.uint8))
            # (grown over its own colour only: purple type set against an orange box was grown into the box, and made paper
            # with it -- a smear; a scan's panel faded by a streak is still near its colour)
            grown &= _near_own_colour(lab, solid, cores, join)
            region_small |= _holes_filled(grown, 30 * join * join)
        # the panel's colour: its own pixels (smoothed a little), and behind its type the colour of the panel round it
        # (a blur of the panel's pixels alone, over how much of the blur was panel: a normalised convolution)
        # (from pixels wholly panel: on this small copy a pixel at a letter's edge is part white type, part panel, and a
        # panel colour taken from it is paler there -- its white type then half a mark, shown hollow)
        # (and where a panel is too pale or too broken up for any pixel to be wholly panel -- a scan's bar faded by a streak
        # -- from all its pixels after all: from none, its colour was no colour, and the whole of it 'a mark', shown black)
        lab_f = lab.astype(np.float32)
        weight = cv2.erode(panel_px, np.ones((3, 3), np.uint8)).astype(np.float32)
        every = panel_px.astype(np.float32)
        spread = cv2.GaussianBlur(weight, (0, 0), _REMOVAL_FILL_PX)
        spread_every = cv2.GaussianBlur(every, (0, 0), _REMOVAL_FILL_PX)
        sparse = (spread < _REMOVAL_MIN_SPREAD * np.maximum(spread_every, 1e-3))[..., None]
        filled = np.where(
            sparse,
            cv2.GaussianBlur(lab_f * every[..., None], (0, 0), _REMOVAL_FILL_PX) / np.maximum(spread_every, 1e-3)[..., None],
            cv2.GaussianBlur(lab_f * weight[..., None], (0, 0), _REMOVAL_FILL_PX) / np.maximum(spread, 1e-3)[..., None],
        )
        own = cv2.GaussianBlur(lab_f * weight[..., None], (0, 0), _REMOVAL_SMOOTH_PX) / np.maximum(
            cv2.GaussianBlur(weight, (0, 0), _REMOVAL_SMOOTH_PX), 1e-3
        )[..., None]
        field = np.where((weight > 0)[..., None], own, filled)
        fields.append(np.clip(field, 0, 255).astype(np.uint8))
    if not region_small.any():
        return pictures
    region = cv2.resize(region_small, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    ys, xs = np.nonzero(region)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    inside = region[y0:y1, x0:x1]
    tolerance = _REMOVAL_MIN_TOLERANCE
    out = []
    for image, picture, field in zip(images, pictures, fields):
        lab = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_RGB2LAB).astype(np.float32)
        panel = cv2.resize(field, (width, height), interpolation=cv2.INTER_LINEAR)[y0:y1, x0:x1].astype(np.float32)
        distance = np.linalg.norm(lab - panel, axis=2)
        mark = np.clip((distance - tolerance) / (tolerance * (_REMOVAL_RAMP - 1.0)), 0.0, 1.0)
        cleared = (255.0 * (1.0 - mark)).astype(np.uint8)
        picture = picture.copy()
        picture[y0:y1, x0:x1][inside] = cleared[inside]
        out.append(picture)
    return out


_PANEL_SAME_OVERLAP = 0.5  # two panels' boxes overlapping this much (of the smaller) are the same panel
_PANEL_FORCED_AGREEMENT = 0.15  # a panel opened in the other picture too: at most this share of the two's type unmatched...
_PANEL_SHADE_WINDOW_PX = 120  # a shaded panel's own colour at a pixel: the median over this wide a window (full size)...
_PANEL_SHADE_SCALE = 0.25  # ...taken on a copy this share of the size


_PANEL_FORCED_SLACK_PX = 4  # ...where each one's type is looked for this near in the other (registration, a bolder print)


def _same_type(ours: np.ndarray, theirs_there: np.ndarray, where: np.ndarray) -> bool:
    """Whether the type found on a panel in the other picture (`theirs_there`: its pixels `where`) is the type this
    picture shows there (`ours`, the whole box): each one's ink lies within a few pixels of the other's."""
    theirs = np.full(where.shape, 255, np.uint8)
    theirs[where] = theirs_there
    ink_ours, ink_theirs = (ours < 128) & where, (theirs < 128) & where
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * _PANEL_FORCED_SLACK_PX + 1, 2 * _PANEL_FORCED_SLACK_PX + 1))
    near_ours = cv2.dilate(ink_ours.astype(np.uint8), kernel) > 0
    near_theirs = cv2.dilate(ink_theirs.astype(np.uint8), kernel) > 0
    stray = int((ink_ours & ~near_theirs).sum()) + int((ink_theirs & ~near_ours).sum())
    return stray <= _PANEL_FORCED_AGREEMENT * max(1, int(ink_ours.sum()) + int(ink_theirs.sum()))


def _local_background(region: np.ndarray) -> np.ndarray:
    """The colour (Lab, float) round each pixel of `region`: the median over a window far wider than a letter's stroke,
    so a panel's shading is kept and the type on it is not."""
    height, width = region.shape[:2]
    small_size = (max(1, round(width * _PANEL_SHADE_SCALE)), max(1, round(height * _PANEL_SHADE_SCALE)))
    lab_small = cv2.cvtColor(cv2.resize(np.ascontiguousarray(region[..., :3]), small_size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2LAB)
    window = max(3, int(round(_PANEL_SHADE_WINDOW_PX * _PANEL_SHADE_SCALE)) | 1)
    smooth = cv2.merge([cv2.medianBlur(channel, window) for channel in cv2.split(lab_small)])
    return cv2.resize(smooth, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def _dark_type_opened(image: np.ndarray, box: tuple[int, int, int, int], where: np.ndarray) -> np.ndarray:
    """A panel with dark type on it (the pixels `where` of `box`), shown as paper with its type as ink: what is clearly
    darker than the panel round it (its lightness, pixel by pixel, for a shaded one) is type; the rest is paper."""
    x0, y0, x1, y1 = box
    region = np.ascontiguousarray(image[y0:y1, x0:x1, :3])
    lightness = cv2.cvtColor(region, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    # the panel's shade from its own pixels only: what lies round it (paper) is taken for its colour first, or the
    # paper would count as the panel's colour near its edges and the panel itself as darker there
    own = region.copy()
    own[~where] = np.median(region[where], axis=0).astype(np.uint8)
    panel = _local_background(own)[..., 0]
    strength = np.clip((panel - lightness - _PANEL_DARKER_L) / _PANEL_INK_RAMP, 0.0, 1.0)
    return (255.0 * (1.0 - strength[where])).astype(np.uint8)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return False
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return width * height >= _PANEL_SAME_OVERLAP * smaller


_PANEL_CORE_PX = 15  # ink that survives being shrunk by this much (either way) is a solid panel, not strokes of type
_PANEL_MIN_AREA_PX = 6000  # ...if its solid core is at least this big (~7 x 7 mm at 300 DPI)
_PANEL_UNIFORM_SHARE = 0.6  # ...and this share of it is one colour (a flat panel, not a picture)
_PANEL_INK_FROM = 30.0  # pixels this far from it are fully its lettering; between, partly (a ramp, as elsewhere)
_PANEL_MIN_LETTERING = 0.01  # a panel with less than this share of other colour in it has nothing printed on it
_PANEL_SCALE = 0.25  # panels are found on a copy this share of the size


_PANEL_FROM_PAPER = 40.0  # (Lab, 8-bit units) a flat colour at least this far from the paper is a printed panel (a pale tint is not: type on it is already ink)
_PANEL_OWN_COLOUR = 27.0  # (Lab) pixels this close to the panel's colour are the panel (a scan's grain included)
_PANEL_DARKER_L = 12.0  # (8-bit Lab lightness) lettering at least this much darker (or lighter) than its panel is printed on it
_PANEL_GRAIN_PERCENTILE = 90  # a panel's own grain: how far its own pixels stray from its colour (all but the furthest tenth)
_PANEL_GRAIN_MARGIN = 4.0  # lettering starts this far beyond it...
_PANEL_INK_MIN_FROM = 12.0  # ...and never nearer the panel's colour than this (Lab)
_PANEL_INK_RAMP = 15.0  # it is fully ink this much further on
_PANEL_SHOWN_WHITE = 200  # a panel the ink picture shows as paper (a tint: its lettering lost with it)...
_PANEL_SHOWN_BLACK = 60  # ...or as a solid block (its lettering holes in it) is opened; one it shows in between is not
_LETTER_MIN_AREA_PX = 25  # a mark smaller than this (a speck, a scan's grain) is not lettering
_LETTER_MAX_HALF_STROKE_PX = 18  # a mark on it no thicker than twice this (at 300 DPI) is lettering, not a shape of its own


def _panels_opened(image: np.ndarray, ink: np.ndarray, valid: np.ndarray | None = None) -> list[_Opened]:
    """Every flat-coloured panel shown as paper, with what is printed on it as ink, in place -- whatever its colour. A
    dark panel otherwise comes out as a black block with its lettering white (or, dark type on it, all one block); a
    lighter one as paper with its lettering lost (white type reversed out of a green, yellow type on it). Either way the
    two files differ by how dark their colours came out, not by what is printed -- a scan's colours are never the
    digital file's -- and the words on it are compared as holes, or not at all. Here every one comes out the same way,
    as the rest of the page does: black type on white.

    Lettering is found by its SHAPE, not by reading it (a reader is at its worst on exactly this): a letter is a mark
    the panel surrounds on every side, and thin -- a stroke. Every such mark, lighter or darker than the panel, becomes
    ink, and the panel's own colour paper. What the panel does not surround (a stripe along its edge, the paper beside
    it) is not its lettering; a large shape it does surround (a white oval, a box) is a background of its own, left as the
    ink picture shows it -- with its own type on it. A panel with nothing printed on it (a plain block, a bar, a big bold
    shape) is a graphic and stays as it is; so does a picture (not one flat colour)."""
    opened_panels: list[_Opened] = []  # (returned: each panel opened, with what the picture showed there before)
    if image.ndim != 3:
        return opened_panels
    height, width = ink.shape
    small_size = (max(1, int(width * _PANEL_SCALE)), max(1, int(height * _PANEL_SCALE)))
    lab_small = cv2.cvtColor(cv2.resize(np.ascontiguousarray(image[..., :3]), small_size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2LAB).astype(np.float32)
    ink_small = cv2.resize(ink, small_size, interpolation=cv2.INTER_AREA)
    usable = np.ones(ink_small.shape, bool)
    if valid is not None and not valid.all():
        usable = cv2.resize(valid.astype(np.uint8), small_size, interpolation=cv2.INTER_NEAREST) > 0
    lightness = lab_small[..., 0][usable]
    if lightness.size < 100:
        return opened_panels
    paper_mask = usable & (lab_small[..., 0] >= np.percentile(lightness, 97) - 6)
    paper = np.median(lab_small[paper_mask], axis=0)
    coloured = (np.linalg.norm(lab_small - paper, axis=2) > _PANEL_FROM_PAPER) & usable
    core_px = max(1, round(_PANEL_CORE_PX * _PANEL_SCALE))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * core_px + 1, 2 * core_px + 1))
    closed = cv2.morphologyEx(coloured.astype(np.uint8), cv2.MORPH_CLOSE, kernel)  # (its lettering leaves gaps in it)
    cores = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _centres = cv2.connectedComponentsWithStats(cores)
    min_area = _PANEL_MIN_AREA_PX * _PANEL_SCALE * _PANEL_SCALE
    # (the full-size reach of the panel's core: the core was opened, so its outline sits a little inside the panel)
    kernel_full = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * _PANEL_CORE_PX + 1, 2 * _PANEL_CORE_PX + 1))
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        # the panel's box at full size, with a margin: what touches the box's edge is outside the panel
        x0, y0 = max(0, int(x / _PANEL_SCALE) - 8), max(0, int(y / _PANEL_SCALE) - 8)
        x1, y1 = min(width, int(np.ceil((x + w) / _PANEL_SCALE)) + 8), min(height, int(np.ceil((y + h) / _PANEL_SCALE)) + 8)
        core = cv2.resize((labels[y : y + h, x : x + w] == label).astype(np.uint8), (int(np.ceil((x + w) / _PANEL_SCALE)) - int(x / _PANEL_SCALE), int(np.ceil((y + h) / _PANEL_SCALE)) - int(y / _PANEL_SCALE)), interpolation=cv2.INTER_NEAREST)
        lab = cv2.cvtColor(np.ascontiguousarray(image[y0:y1, x0:x1, :3]), cv2.COLOR_RGB2LAB).astype(np.float32)
        in_core = np.zeros(lab.shape[:2], bool)
        cy0, cx0 = int(y / _PANEL_SCALE) - y0, int(x / _PANEL_SCALE) - x0
        in_core[cy0 : cy0 + core.shape[0], cx0 : cx0 + core.shape[1]] = core[: lab.shape[0] - cy0, : lab.shape[1] - cx0] > 0
        if in_core.sum() < 100:
            continue
        colour = np.median(lab[in_core], axis=0)
        if np.linalg.norm(colour - paper) <= _PANEL_FROM_PAPER:
            continue  # (mostly paper after all)
        shown = float(np.median(ink[y0:y1, x0:x1][in_core]))
        if _PANEL_SHOWN_BLACK < shown < _PANEL_SHOWN_WHITE:
            continue  # the ink picture already made something of it in between (a shaded background): left as it is
        distance = np.linalg.norm(lab - colour, axis=2)
        panel = distance < _PANEL_OWN_COLOUR
        if panel[in_core].mean() < _PANEL_UNIFORM_SHARE:
            continue  # not one flat colour: a picture, a gradient
        # the marks the panel surrounds: what is not the panel and does not reach the edge of the panel's box
        marks = (~panel).astype(np.uint8)
        found, mark_labels, mark_stats, _c = cv2.connectedComponentsWithStats(marks, connectivity=8)
        if found <= 1:
            continue
        box_h, box_w = marks.shape
        touching = (mark_stats[:, cv2.CC_STAT_LEFT] == 0) | (mark_stats[:, cv2.CC_STAT_TOP] == 0) | (
            mark_stats[:, cv2.CC_STAT_LEFT] + mark_stats[:, cv2.CC_STAT_WIDTH] >= box_w
        ) | (mark_stats[:, cv2.CC_STAT_TOP] + mark_stats[:, cv2.CC_STAT_HEIGHT] >= box_h)
        # ...and thin: no point of it further than a stroke's half-width from the panel
        half_stroke = cv2.distanceTransform(marks, cv2.DIST_L2, 3)
        thickest = np.zeros(found, np.float32)
        np.maximum.at(thickest, mark_labels.ravel(), half_stroke.ravel())
        big_enough = mark_stats[:, cv2.CC_STAT_AREA] >= _LETTER_MIN_AREA_PX  # (a scan's speckle is not type)
        lettering_marks = ~touching & (thickest <= _LETTER_MAX_HALF_STROKE_PX) & big_enough
        lettering_marks[0] = False
        is_lettering = lettering_marks[mark_labels]
        if is_lettering.sum() < _PANEL_MIN_LETTERING * panel.sum():
            continue  # nothing printed on it
        # how far from the panel's colour a pixel must be to be its lettering: measured on THIS panel -- beyond its own
        # grain (a scan's texture, its print's mottle) -- not fixed: a scan's dark type on a dark panel can be only a
        # little darker than it, and a fixed bar lost it
        grain = float(np.percentile(distance[panel & in_core], _PANEL_GRAIN_PERCENTILE)) if (panel & in_core).any() else 0.0
        ink_from = min(_PANEL_INK_FROM, max(_PANEL_INK_MIN_FROM, grain + _PANEL_GRAIN_MARGIN))
        strength = np.clip((distance - ink_from) / _PANEL_INK_RAMP, 0.0, 1.0)
        # the panel's own colour is paper (only this panel: the same colour elsewhere in its box is another element);
        # its lettering ink, softly at its edges, as elsewhere
        this_panel = panel & (cv2.dilate(in_core.astype(np.uint8), kernel_full) > 0)
        region = ink[y0:y1, x0:x1]
        if shown <= _PANEL_SHOWN_BLACK:
            # A solid block: the panel becomes paper and all its lettering ink. Type DARKER than the panel is taken pixel
            # by pixel, as close to its edge as it runs (it cannot be mistaken for the paper round the panel, which is
            # lighter); type LIGHTER than it -- reversed out of it, holes in the block until now -- by its shape (above),
            # which is what tells it from that paper.
            darker = lab[..., 0] < colour[0] - _PANEL_DARKER_L
            lighter = is_lettering & (lab[..., 0] > colour[0] + _PANEL_DARKER_L)
            ink_there = np.maximum(strength * darker, strength * lighter)
            # (only the panel and what is printed on it: the core is closed over gaps a stroke or two wide, and two bars
            # with a line of text between them came out one core -- the text, neither the panel's colour nor on it, wiped)
            opened = (in_core & (panel | is_lettering | darker)) | this_panel | lighter
            dark_type = int(darker.sum()) >= int(lighter.sum())
            opened_panels.append(((x0, y0, x1, y1), opened, region[opened].copy(), dark_type))
            region[opened] = (255.0 * (1.0 - ink_there[opened])).astype(np.uint8)
        else:
            # already paper: its lettering is added, and nothing already there is lightened (a letter's soft edge)
            opened_panels.append(((x0, y0, x1, y1), is_lettering, region[is_lettering].copy(), None))
            region[is_lettering] = np.minimum(region[is_lettering], (255.0 * (1.0 - strength[is_lettering])).astype(np.uint8))
    return opened_panels


READING_SOURCES = ("ink", "gray", "flat", "darkest", "clahe", "adaptive")
_ADAPTIVE_BLOCK_PX = 51  # the neighbourhood a local threshold is taken over: wider than a stroke, narrower than a line's height twice over
_ADAPTIVE_OFFSET = 12  # a pixel is ink if it is this much darker than its neighbourhood


def reading_source(
    image: np.ndarray, config: DetectionConfig | None = None, valid: np.ndarray | None = None, source: str = "ink"
) -> np.ndarray:
    """The file as it is READ for text: an 8-bit grey picture, 255 = paper, darker = ink (see `DetectionConfig.scan_reading_scan`).

    'ink' is `ink_image`. The others keep more of what the file is: 'gray' is its own grey, 'flat' divides the paper out
    (so a scan's darker corners and cream paper are white paper again), 'darkest' does the same on the darkest channel
    (a saturated ink is dark in one), 'clahe' stretches the local contrast of that, and 'adaptive' thresholds it locally.
    """
    config = config or DetectionConfig()
    if source == "ink":
        return ink_image(image, config, valid)
    if source not in READING_SOURCES:
        raise ValueError(f"unknown reading source {source!r}")
    outside = None
    if valid is not None and not valid.all():
        outside = ~valid
        image = _filled_outside(image, valid)
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if source == "gray":
        picture = gray
    else:
        base = cv2.min(cv2.min(image[..., 0], image[..., 1]), image[..., 2]) if source == "darkest" else gray
        paper = cv2.resize(np.clip(_paper_field(gray), _MIN_PAPER, 255).astype(np.uint8), (width, height), interpolation=cv2.INTER_LINEAR)
        picture = cv2.divide(base, paper, scale=255.0)
        if source == "clahe":
            picture = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(picture)
        elif source == "adaptive":
            picture = cv2.adaptiveThreshold(
                picture, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, _ADAPTIVE_BLOCK_PX, _ADAPTIVE_OFFSET
            )
    if outside is not None:
        picture = picture.copy()
        picture[outside] = 255
    return picture
