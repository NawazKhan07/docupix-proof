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
    return np.where(valid[..., None], image, cv2.resize(filled, (width, height), interpolation=cv2.INTER_LINEAR))


_TINT_SCALE = 0.25  # the tint and the solid graphics are worked out on a copy this share of the size
_MIN_PAPER = 60  # the paper is never taken to be darker than this when it is divided out
_RAMP_SAMPLE_STEP = 8  # the strength of the ink is judged from every 8th pixel each way
_RAMP_MIN_INK = 40  # only pixels at least this much ink say how strong the ink is
_RAMP_MIN_SAMPLES = 200


def _smoothstep(values: np.ndarray, low: float, high: float) -> np.ndarray:
    t = np.clip((values - low) / max(high - low, 1e-6), 0.0, 1.0)
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
      block (a dark red, a navy, a green) stays black with its white lettering white, but a saturated yellow or
      orange panel is light to the eye and is a tint. It used to be decided by a cut-off on the darkest channel,
      and a panel a few levels either side of it (a scan is a little lighter than the file it came from) came
      out black in one file and white in the other. Now how solid a region is (0 = tint, 1 = solid) is a smooth
      function of the median LUMINANCE over the window, and it makes the region that dark, so a panel a few
      levels lighter is a few levels less black, never the opposite colour.

    All of it is done in 8 bits, at a quarter of the size where a window is needed: at full size on an 18 MP
    carton the float version took over a second per file.
    """
    height, width = image.shape[:2]
    outside = None
    if valid is not None and not valid.all():
        outside = ~valid
        image = _filled_outside(image, valid)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    paper_small = _paper_field(gray)
    paper = cv2.resize(np.clip(paper_small, _MIN_PAPER, 255).astype(np.uint8), (width, height), interpolation=cv2.INTER_LINEAR)

    red, green, blue = cv2.split(image)
    darkest = cv2.min(cv2.min(red, green), blue)
    darkness = cv2.bitwise_not(cv2.divide(darkest, paper, scale=255.0))  # 0 for paper, 255 for black or a saturated ink

    reduced = cv2.resize(darkness, None, fx=_TINT_SCALE, fy=_TINT_SCALE, interpolation=cv2.INTER_AREA)
    window = max(3, int(round(config.ink_window_fraction * max(height, width) * _TINT_SCALE)) | 1)
    tint_small = cv2.medianBlur(reduced, window)

    small_size = (reduced.shape[1], reduced.shape[0])
    luminance = cv2.resize(gray, small_size, interpolation=cv2.INTER_AREA).astype(np.float32)
    paper_here = cv2.resize(np.maximum(paper_small, _MIN_PAPER), small_size, interpolation=cv2.INTER_LINEAR)
    luminance_darkness = 255.0 - np.clip(luminance * (255.0 / paper_here), 0.0, 255.0)
    typical = cv2.medianBlur(luminance_darkness.astype(np.uint8), window).astype(np.float32)
    solid_small = (255.0 * _smoothstep(typical, config.ink_solid_low, config.ink_solid_high)).astype(np.uint8)

    tint = cv2.resize(tint_small, (width, height), interpolation=cv2.INTER_LINEAR)
    solid = cv2.resize(solid_small, (width, height), interpolation=cv2.INTER_LINEAR)

    room = cv2.max(cv2.bitwise_not(tint), 1)  # what is left of the scale above the tint
    relative = cv2.divide(cv2.subtract(darkness, tint), room, scale=255.0)
    contrast = cv2.max(relative, cv2.multiply(darkness, solid, scale=1.0 / 255.0))
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
    if outside is not None:
        result[outside] = 255
    return result
