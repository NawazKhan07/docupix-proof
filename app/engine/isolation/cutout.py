"""A lifted subject as a picture with a transparent background -- and back again.

`Cutout` is the artwork alone: the pixels inside the outline, the outline itself,
and nothing else. It is what "Save cutout" writes (a lossless PNG whose alpha
channel is the outline) and what can be laid over another file and dragged into
place. Reading a saved cutout recovers the outline from the alpha channel, so a
cutout saved today can trim a suspect next week.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.config import SubjectConfig
from app.engine.isolation.subject_lifter import LiftedSubject, NoSubjectFoundError, outline_from_mask

Point = tuple[float, float]

_ALPHA_EDGE_SOFTNESS_PX = 0.8  # keeps a saved cutout's edge from looking stair-stepped
_OPAQUE = 127


class CutoutError(Exception):
    """A cutout could not be made or read; the message says why, for a person."""


@dataclass
class Cutout:
    rgba: np.ndarray  # (h, w, 4) uint8, cropped to the subject's bounding box
    polygon: list[Point]  # the outline, in this picture's own pixels
    origin: tuple[int, int] | None = None  # where its top-left sat in the source image, if known
    edge_inset_px: float = 0.0  # how far inside its outline comparing should start (see LiftedSubject)

    @property
    def size(self) -> tuple[int, int]:
        """(width, height)."""
        return self.rgba.shape[1], self.rgba.shape[0]

    @property
    def mask(self) -> np.ndarray:
        return self.rgba[..., 3] > _OPAQUE

    def on_white(self) -> np.ndarray:
        """The artwork over plain white, as RGB -- for anything that cannot take transparency."""
        alpha = self.rgba[..., 3:4].astype(np.float32) / 255.0
        return (self.rgba[..., :3] * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)

    # -- making one ---------------------------------------------------------------
    @classmethod
    def from_subject(cls, image_rgb: np.ndarray, subject: LiftedSubject) -> "Cutout":
        x0, y0, x1, y1 = subject.bbox
        mask = subject.mask[y0:y1, x0:x1].astype(np.uint8) * 255
        alpha = cv2.GaussianBlur(mask, (0, 0), _ALPHA_EDGE_SOFTNESS_PX)
        alpha = np.where(mask > 0, np.maximum(alpha, 200), alpha).astype(np.uint8)  # soften outward, never eat inward
        rgba = np.dstack([image_rgb[y0:y1, x0:x1], alpha])
        polygon = [(x - x0, y - y0) for x, y in subject.polygon]
        return cls(rgba=np.ascontiguousarray(rgba), polygon=polygon, origin=(x0, y0), edge_inset_px=subject.edge_inset_px)

    # -- saving and loading ---------------------------------------------------------
    def save_png(self, path: Path | str) -> Path:
        """Write a lossless PNG with the outline as its transparency."""
        path = Path(path)
        if path.suffix.lower() != ".png":
            path = path.with_suffix(".png")
        Image.fromarray(self.rgba, "RGBA").save(path, format="PNG")
        return path

    @classmethod
    def load_png(cls, path: Path | str, config: SubjectConfig | None = None) -> "Cutout":
        """Read a saved cutout. The outline is recovered from the alpha channel."""
        path = Path(path)
        try:
            image = Image.open(path)
            has_alpha = image.mode in ("RGBA", "LA") or "transparency" in image.info
            rgba = np.array(image.convert("RGBA"))
        except (OSError, ValueError) as exc:
            raise CutoutError(f"Could not open '{path.name}' as an image: {exc}") from exc
        if not has_alpha:
            raise CutoutError(
                f"'{path.name}' has no transparent background, so there is no outline to use. "
                "Open a cutout saved with Save cutout, or a PNG with a transparent background."
            )

        mask = rgba[..., 3] > _OPAQUE
        if not mask.any():
            raise CutoutError(f"'{path.name}' is entirely transparent.")
        ys, xs = np.nonzero(mask)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        rgba = np.ascontiguousarray(rgba[y0:y1, x0:x1])
        try:
            polygon = outline_from_mask(rgba[..., 3] > _OPAQUE, config or SubjectConfig())
        except NoSubjectFoundError as exc:
            raise CutoutError(str(exc)) from exc
        return cls(rgba=rgba, polygon=polygon, origin=None)
