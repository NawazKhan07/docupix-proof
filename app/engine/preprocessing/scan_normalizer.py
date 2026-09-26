"""Scan-vs-Digital preprocessing (SRS FR-6.3, Architecture §3.3.4).

Flattens both files to the same black-on-white representation so a scan can
be compared against a digital original without every lighting and ink
variation reading as a difference. Colour comparison is disabled separately,
by the mode's `DetectorOptions`, so it is skipped rather than merely hidden.

`ink_image` is the picture that is shown, read for text and compared: the file with its
background colours removed to white and everything printed on it black.
"""

from __future__ import annotations

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


def normalize_for_scan(image: np.ndarray) -> np.ndarray:
    """Return an RGB image with a white background and pure-black foreground.

    The illumination is estimated and divided out first: a scan is rarely lit
    evenly, and thresholding a vignetted page directly would erase pale
    corners or keep grey ones.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    flattened = cv2.divide(gray, illumination_field(gray), scale=255)
    _, binary = cv2.threshold(flattened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


def is_effectively_blank(image: np.ndarray) -> bool:
    """Whether a normalized image carries essentially no foreground."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return bool(np.count_nonzero(gray < 128) < 0.0001 * gray.size)


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
    solid_small = (
        255.0 * _smoothstep(typical, config.ink_solid_low + neutral_shift, config.ink_solid_high + neutral_shift)
    ).astype(np.uint8)

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


def ink_image(
    image: np.ndarray, config: DetectionConfig | None = None, valid: np.ndarray | None = None
) -> np.ndarray:
    """The file as ink on paper: a grey image, 255 where there is only background, 0 where there is ink.

    Every colour that is only a background -- a pale panel, a tint, a scan's cream paper or its uneven
    lighting -- comes out white, and whatever is printed on it comes out black, whatever its colour.
    That is what lets a scan be compared with its digital original by *what is printed*, since scanning
    changes every colour. It is the picture the reviewer is shown and the one text is read from; a ramp
    (not a cut-off) keeps edges soft and drops speckle below it. See `_ink_contrast` for how ink is told
    from background. `valid` marks where the picture really is (the compared area): the rest is
    background by construction, comes out white, and is not allowed to influence the estimates near it.

    Nothing is tuned to a file: the paper, the tint and the noise floor are all measured from its pixels.
    """
    config = config or DetectionConfig()
    contrast, outside = _ink_contrast(image, config, valid)
    result = cv2.LUT(contrast, _ramp(contrast, config))
    _panels_opened(image, result)
    if outside is not None:
        result[outside] = 255
    return result


_PANEL_CORE_PX = 15  # ink that survives being shrunk by this much (either way) is a solid panel, not strokes of type
_PANEL_MIN_AREA_PX = 6000  # ...if its solid core is at least this big (~7 x 7 mm at 300 DPI)
_PANEL_UNIFORM_SHARE = 0.6  # ...and this share of it is one colour (a flat panel, not a picture)
_PANEL_SAME_COLOUR = 18.0  # pixels this close (Lab, 8-bit units) to the panel's colour are the panel
_PANEL_INK_FROM = 30.0  # pixels this far from it are fully its lettering; between, partly (a ramp, as elsewhere)
_PANEL_INK_TO = 60.0
_PANEL_MIN_LETTERING = 0.01  # a panel with less than this share of other colour in it has nothing printed on it
_PANEL_SCALE = 0.25  # panels are found on a copy this share of the size
_PANEL_MIN_SOLID = 0.6  # a panel is mostly ink in the ink picture (its lettering is the rest)


def _panels_opened(image: np.ndarray, ink: np.ndarray) -> None:
    """Lettering printed on a solid coloured panel (dark type on a red warning box, white type reversed out of it) shown
    as ink on paper, in place: every other colour counts as ink in `ink`, so a panel and its lettering come out as one
    black block and its words cannot be read, or compared. Inside each solid, flat-coloured panel, what is the panel's
    own colour becomes paper and what differs from it -- the lettering -- ink. A panel with nothing on it (a plain box,
    a big bold letter) stays solid; a picture (not one flat colour) is left as it is."""
    if image.ndim != 3:
        return
    height, width = ink.shape
    small = cv2.resize(ink, (max(1, int(width * _PANEL_SCALE)), max(1, int(height * _PANEL_SCALE))), interpolation=cv2.INTER_AREA)
    solid = (small < 128).astype(np.uint8)
    core_px = max(1, round(_PANEL_CORE_PX * _PANEL_SCALE))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * core_px + 1, 2 * core_px + 1))
    # closed first, so the gaps the lettering leaves in its panel do not break the panel up
    closed = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, kernel)
    cores = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _centres = cv2.connectedComponentsWithStats(cores)
    min_area = _PANEL_MIN_AREA_PX * _PANEL_SCALE * _PANEL_SCALE
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        x0, y0 = int(x / _PANEL_SCALE), int(y / _PANEL_SCALE)
        x1, y1 = min(width, int(np.ceil((x + w) / _PANEL_SCALE))), min(height, int(np.ceil((y + h) / _PANEL_SCALE)))
        inside = cv2.resize((labels[y : y + h, x : x + w] == label).astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0
        solid_there = ink[y0:y1, x0:x1][inside] < 128
        if solid_there.mean() < _PANEL_MIN_SOLID:
            continue  # mostly paper: lines of type closed up, not a panel
        lab = cv2.cvtColor(np.ascontiguousarray(image[y0:y1, x0:x1, :3]), cv2.COLOR_RGB2LAB).astype(np.float32)
        pixels = lab[inside][solid_there]
        if len(pixels) < 100:
            continue
        colour = np.median(pixels, axis=0)
        distance = np.linalg.norm(lab - colour, axis=2)
        near = distance[inside][solid_there] < _PANEL_SAME_COLOUR
        if near.mean() < _PANEL_UNIFORM_SHARE:
            continue  # not one flat colour: a picture
        # its lettering: DARKER than the panel (type printed on it). Lettering lighter than its panel -- reversed out of
        # it -- already shows as white in the black block, and is read so; paper caught in the panel's outline is not ink.
        darker = lab[..., 0] < colour[0] - _PANEL_DARKER_L
        opened = np.clip((distance - _PANEL_INK_FROM) / (_PANEL_INK_TO - _PANEL_INK_FROM), 0.0, 1.0) * darker
        if (opened[inside] > 0.5).mean() < _PANEL_MIN_LETTERING:
            continue  # nothing printed on it in a darker ink: a plain box, a big bold letter, a reversed-out panel
        region = ink[y0:y1, x0:x1]
        region[inside] = (255.0 * (1.0 - opened[inside])).astype(np.uint8)


_PANEL_DARKER_L = 12.0  # (8-bit Lab lightness) lettering at least this much darker than its panel is printed on it


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
