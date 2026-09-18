"""Loads PDF and raster artwork files (Architecture §3.3.1).

PDFs are rendered on demand from their vector content at whatever DPI is
requested — never cached at one fixed resolution — so the viewer can zoom in
without ever showing a pixelated re-scale of a smaller bitmap (FR-9.1).
Raster images have no vector source to re-rasterize from, so "full quality at
any zoom" means the opposite: `render()` always returns the untouched native
pixel data, and it is the caller's job (the Qt viewer, in a later phase) to
scale it for display rather than this loader silently downsampling it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, UnidentifiedImageError

from app.logging_config import StageTimer

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS

# Working resolution for detection (Architecture §6.1); display rendering
# uses whatever DPI the current zoom level calls for instead of this.
WORKING_DPI = 300
_PDF_BASE_DPI = 72.0  # PDF user-space units are defined at 72 DPI


class ArtworkLoadError(Exception):
    """Base class for artwork-loading failures the UI should show, not crash on."""


class UnsupportedFileFormatError(ArtworkLoadError):
    pass


class CorruptFileError(ArtworkLoadError):
    pass


@dataclass
class LoadedArtwork:
    """A loaded original/suspect file, able to render itself at any DPI/page."""

    path: Path
    kind: str  # "pdf" | "image"
    page_count: int
    color_profile: Optional[bytes]  # ICC profile bytes, if the source embeds one
    native_mode: str  # PIL/colour mode of the source (e.g. "RGB", "CMYK")
    _pdf_doc: Optional[fitz.Document] = None
    _pil_image: Optional[Image.Image] = None

    def page_size_px(self, dpi: float = WORKING_DPI, page: int = 0) -> tuple[int, int]:
        """(width, height) in pixels a render() at this DPI/page would produce."""
        if self.kind == "pdf":
            assert self._pdf_doc is not None
            rect = self._pdf_doc[page].rect
            scale = dpi / _PDF_BASE_DPI
            # PyMuPDF's pixmap sizing ceils the scaled extent (to guarantee
            # full page coverage), so mirror that here rather than round().
            return math.ceil(rect.width * scale), math.ceil(rect.height * scale)
        assert self._pil_image is not None
        return self._pil_image.size

    def render(self, dpi: float = WORKING_DPI, page: int = 0) -> np.ndarray:
        """Return an RGB uint8 numpy array (H, W, 3).

        For PDFs this re-rasterizes the vector page at `dpi` — full fidelity
        at every zoom level. For images `dpi` is ignored: the native pixels
        are always returned untouched (no invented detail, no silent
        downsampling).
        """
        with StageTimer(logger, f"render_{self.kind}"):
            if self.kind == "pdf":
                return self._render_pdf_page(dpi, page)
            return self._render_image()

    def _render_pdf_page(self, dpi: float, page: int) -> np.ndarray:
        assert self._pdf_doc is not None
        pdf_page = self._pdf_doc[page]
        scale = dpi / _PDF_BASE_DPI
        pixmap = pdf_page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )
        if pixmap.n == 1:  # grayscale PDF content
            array = np.repeat(array, 3, axis=2)
        return array.copy()  # detach from the pixmap's buffer before it's freed

    def _render_image(self) -> np.ndarray:
        assert self._pil_image is not None
        return np.array(self._pil_image.convert("RGB"))

    def close(self) -> None:
        if self._pdf_doc is not None:
            self._pdf_doc.close()
            self._pdf_doc = None
        if self._pil_image is not None:
            self._pil_image.close()
            self._pil_image = None

    def __enter__(self) -> "LoadedArtwork":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def load_artwork(path: Path | str) -> LoadedArtwork:
    """Load a PDF or raster image, validating format and readability.

    Raises `UnsupportedFileFormatError` for an unrecognized extension and
    `CorruptFileError` if the file matches a supported extension but can't
    actually be parsed (rather than letting PyMuPDF/Pillow's raw exception,
    or a native-library crash, take down the caller — NFR-2).
    """
    path = Path(path)
    if not path.is_file():
        raise CorruptFileError(f"File not found: {path}")
    if path.stat().st_size == 0:
        raise CorruptFileError(f"File is empty: {path}")

    ext = path.suffix.lower()
    with StageTimer(logger, "load_artwork"):
        if ext in SUPPORTED_PDF_EXTENSIONS:
            artwork = _load_pdf(path)
        elif ext in SUPPORTED_IMAGE_EXTENSIONS:
            artwork = _load_image(path)
        else:
            raise UnsupportedFileFormatError(
                f"Unsupported file format '{ext}'. Supported: "
                f"{sorted(SUPPORTED_EXTENSIONS)}"
            )
    logger.info(
        "loaded artwork file",
        extra={"path": str(path), "kind": artwork.kind, "page_count": artwork.page_count},
    )
    return artwork


def _load_pdf(path: Path) -> LoadedArtwork:
    try:
        doc = fitz.open(path)
        if doc.page_count < 1:
            raise CorruptFileError(f"PDF has no pages: {path}")
        # Touching page 0 forces PyMuPDF to parse content now, surfacing a
        # corrupt/truncated PDF here rather than deep inside a later render.
        _ = doc[0].rect
    except CorruptFileError:
        raise
    except Exception as exc:  # PyMuPDF raises a range of RuntimeError/ValueError types
        raise CorruptFileError(f"Could not parse PDF '{path}': {exc}") from exc
    return LoadedArtwork(
        path=path,
        kind="pdf",
        page_count=doc.page_count,
        color_profile=None,
        native_mode="RGB",
        _pdf_doc=doc,
    )


def _load_image(path: Path) -> LoadedArtwork:
    try:
        image = Image.open(path)
        image.load()  # force full decode now, not lazily on first use
    except (UnidentifiedImageError, OSError) as exc:
        raise CorruptFileError(f"Could not parse image '{path}': {exc}") from exc
    return LoadedArtwork(
        path=path,
        kind="image",
        page_count=1,
        color_profile=image.info.get("icc_profile"),
        native_mode=image.mode,
        _pil_image=image,
    )
