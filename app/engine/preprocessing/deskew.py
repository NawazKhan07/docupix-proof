"""Levelling a picture before it is read (Scan vs Digital).

A scan is never perfectly square on the glass, and the alignment step (a global homography fit
to matched features) does not always remove every trace of that: what is left, even a fraction of
a degree, is enough to throw off Tesseract's OWN line segmentation on packed, repeating lines --
words of one line merge into the next, or a whole line reads back as a scrambled run of words from
several real lines. A batch of near-identical lines (a repeated batch/expiry stamp, a dense table)
is the worst case, because there is nothing but the tilt itself to tell one line from the next.

`estimate_skew_deg` measures how far a picture's text lines are from level, by the classic
projection-profile method: rotate a small copy through a range of angles and pick the one whose
row-by-row ink profile is sharpest (a level line of type makes some rows solid ink and the rows
between lines empty; a tilted one blurs every row into a mix of its neighbours). `Deskew` then
turns the picture that far before it is handed to the reader, and turns each word's box back
afterwards -- the reader is the only thing that ever sees the turned picture; nothing else in the
comparison (matching, ink agreement, the boxes shown to the reviewer) knows it happened.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.config import DetectionConfig
from app.models.diff import BoundingBox

_DARK = 128  # an ink picture's pixels below this are ink (matches text_diff._DARK)


@dataclass(frozen=True)
class Deskew:
    """How far a picture was turned to level its text (degrees; positive as `cv2.getRotationMatrix2D` turns it),
    and the turning of it -- `picture` turns an image that far, `unrotate_box` turns a box in the turned picture's
    pixels back to the original, unrotated frame it was measured in."""

    angle_deg: float
    _center: tuple[float, float]
    _size: tuple[int, int]  # (width, height) of the picture the angle was measured on

    @property
    def is_identity(self) -> bool:
        return self.angle_deg == 0.0

    def picture(self, image: np.ndarray, fill: int = 255) -> np.ndarray:
        """`image` (the same size it was measured on) turned level. `fill` is what shows at the corners it
        uncovers (255 = paper, the background every reading picture in this pipeline has)."""
        if self.is_identity:
            return image
        matrix = cv2.getRotationMatrix2D(self._center, self.angle_deg, 1.0)
        height, width = self._size[1], self._size[0]
        return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=fill)

    def unrotate_box(self, box: BoundingBox) -> BoundingBox:
        """`box`, given in the turned picture's pixels, back in the frame it was measured in: its CENTRE turned
        back, keeping its own width and height.

        Turning the box's four corners back and taking the axis-aligned rectangle round them was tried first, but
        a box that is no longer axis-aligned once turned back grows one by roughly `size * sin(angle)` doing that
        -- tens of pixels for an ordinary word at only a few degrees -- which was enough to break the word
        matching downstream (a word placed 20 px from where its counterpart in the other file actually is no
        longer looks like the same word, close by). Keeping the box's own size and only moving its centre is a
        closer match to where a word painted at an angle actually sits, and every consumer of a word's box already
        tolerates a few pixels either way, not tens.
        """
        if self.is_identity:
            return box
        matrix = cv2.invertAffineTransform(cv2.getRotationMatrix2D(self._center, self.angle_deg, 1.0))
        cx, cy = box.center
        turned = np.array([cx, cy]) @ matrix[:, :2].T + matrix[:, 2]
        half_w, half_h = box.width / 2.0, box.height / 2.0
        return BoundingBox(float(turned[0] - half_w), float(turned[1] - half_h), float(turned[0] + half_w), float(turned[1] + half_h))


_IDENTITY = Deskew(0.0, (0.0, 0.0), (0, 0))
_COARSE_STEP_DEG = 0.5
_FINE_RANGE_DEG = 0.6  # refined this far either side of the coarse best
_MIN_INK_SHARE = 0.001  # a picture with less ink than this says nothing about its lines' angle


def _levelness(ink: np.ndarray, angle_deg: float) -> float:
    """How sharply `ink`'s rows alternate between full of ink and empty, turned by `angle_deg`: high for level
    text (a row is either inside a line of type or in the gap above or below it), low for tilted text (every
    row is a blur of the lines on either side of it)."""
    height, width = ink.shape
    if angle_deg != 0.0:
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_deg, 1.0)
        ink = cv2.warpAffine(ink, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
    profile = ink.sum(axis=1, dtype=np.float64)
    return float(profile.var())


def _best_angle(ink: np.ndarray, candidates: list[float]) -> float:
    return max(candidates, key=lambda angle: _levelness(ink, angle))


def estimate_skew_deg(picture: np.ndarray, config: DetectionConfig | None = None) -> float:
    """The angle that would make `picture`'s text lines most level (degrees; 0.0 if it is already close enough,
    or there is too little ink to tell). `picture` is a reading picture: grey, 255 = paper, darker = ink."""
    config = config or DetectionConfig()
    grey = cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY) if picture.ndim == 3 else picture
    scale = min(1.0, config.deskew_search_long_edge_px / max(grey.shape))
    small = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else grey
    ink = (small < _DARK).astype(np.uint8)
    if float(ink.mean()) < _MIN_INK_SHARE:
        return 0.0

    limit = config.deskew_max_angle_deg
    coarse = np.arange(-limit, limit + _COARSE_STEP_DEG / 2, _COARSE_STEP_DEG)
    best = _best_angle(ink, list(coarse))
    fine = np.arange(best - _FINE_RANGE_DEG, best + _FINE_RANGE_DEG + config.deskew_min_correction_deg / 2, config.deskew_min_correction_deg)
    best = _best_angle(ink, [max(-limit, min(limit, angle)) for angle in fine])
    return best if abs(best) >= config.deskew_min_correction_deg else 0.0


def measured(picture: np.ndarray, config: DetectionConfig | None = None) -> Deskew:
    """The `Deskew` that would level `picture` -- identity (nothing turned) where `deskew_enabled` is off, there
    is too little ink to measure, or the picture is already level enough to be worth leaving alone."""
    config = config or DetectionConfig()
    if not config.deskew_enabled:
        return _IDENTITY
    angle = estimate_skew_deg(picture, config)
    if angle == 0.0:
        return _IDENTITY
    height, width = picture.shape[:2]
    return Deskew(angle, (width / 2.0, height / 2.0), (width, height))
