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

import ctypes
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pymupdf as fitz  # PyMuPDF (the "fitz" name is a deprecated alias)
import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import RuntimeConfig
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


class InputTooLargeError(ArtworkLoadError):
    """The file is over a limit that someone set (`RuntimeConfig.max_file_size_mb` / `max_pdf_pages`): off by default,
    so nothing is refused for its size unless a person asked for that."""


_BYTES_PER_MB = 1024 * 1024

# A comparison holds each file's page several times over while it aligns and detects: measured at ~70 bytes per pixel
# of page for the pair (a 93 MP shipper peaked at 6.4 GB). So a page is held at no more than memory can take:
#  * this machine's memory -- a share of what is installed, since the app's own window and the OS need the rest;
#  * and the memory actually free when the comparison starts -- a machine with an IDE, a browser and a database open
#    has far less than it has installed, and a page sized to the installed memory alone ran it out mid-comparison.
# What is free changes by the minute, so the resolution does not follow it smoothly: it steps down through a few fixed
# levels, and a page held below `WORKING_DPI` says so on the results page. A carton at full resolution (~18-25 MP) is
# never reduced.
_PEAK_BYTES_PER_PIXEL = 70
_MEMORY_SHARE = 0.45  # of the installed memory, the most one comparison may take
_FREE_MEMORY_SHARE = 0.8  # of the memory free now, the most one comparison may take (other programs keep running)
_BUDGET_LEVELS = (100.0, 80.0, 64.0, 50.0, 40.0, 32.0, 25.0, 18.0)  # megapixels; never below the last


class _MemoryStatus(ctypes.Structure):
    """Windows' MEMORYSTATUSEX."""

    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _memory_bytes() -> tuple[int, int] | None:
    """(installed, free now) memory in bytes, or None where it cannot be told. "Free" is what can still be committed
    (on Windows, allocations fail on the commit limit, not on physical memory), and no more than twice the physical
    memory free (beyond that a comparison would crawl, paging)."""
    try:
        if sys.platform == "win32":
            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys), int(min(status.ullAvailPageFile, status.ullAvailPhys * 2))
            return None
        page = os.sysconf("SC_PAGE_SIZE")
        return int(os.sysconf("SC_PHYS_PAGES") * page), int(os.sysconf("SC_AVPHYS_PAGES") * page)
    except (AttributeError, OSError, ValueError):
        return None


def affordable_megapixels(limit: float) -> float:
    """`limit` (`RuntimeConfig.max_render_megapixels`), lowered to the highest of `_BUDGET_LEVELS` that memory can hold."""
    memory = _memory_bytes()
    if memory is None:
        return limit
    installed, free = memory
    fits = min(_MEMORY_SHARE * installed, _FREE_MEMORY_SHARE * free) / _PEAK_BYTES_PER_PIXEL / 1e6
    level = next((level for level in _BUDGET_LEVELS if level <= fits), _BUDGET_LEVELS[-1])
    return float(min(limit, level))


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
    budget_megapixels: float = 100.0  # the most pixels a page is held at (see `RuntimeConfig.max_render_megapixels`)

    def working_dpi(self, page: int = 0) -> float:
        """The DPI a PDF page is rendered at: `WORKING_DPI`, or lower where the page would not fit `budget_megapixels`
        (a page 2 m across is not held at 300 DPI, it is held at what fits). Images have their own pixels: their nominal
        DPI is `WORKING_DPI`."""
        if self.kind != "pdf":
            return float(WORKING_DPI)
        assert self._pdf_doc is not None
        rect = self._pdf_doc[page].rect
        points = max(1.0, rect.width * rect.height)
        fitting = _PDF_BASE_DPI * math.sqrt(max(1.0, self.budget_megapixels) * 1e6 / points)
        return float(min(WORKING_DPI, fitting))

    def reduced_note(self, page: int = 0) -> str | None:
        """A sentence saying the page is held below `WORKING_DPI`, or None if it is not."""
        if self.kind == "pdf":
            dpi = self.working_dpi(page)
            if dpi < WORKING_DPI - 0.5:
                rect = self._pdf_doc[page].rect
                return (
                    f"'{self.path.name}' is {rect.width * 25.4 / 72:.0f} x {rect.height * 25.4 / 72:.0f} mm, more than fits in "
                    f"memory at {WORKING_DPI} DPI, so it was compared at {dpi:.0f} DPI."
                )
            return None
        if self._pil_image is not None:
            width, height = self._pil_image.size
            if width * height > self.budget_megapixels * 1e6:
                scale = math.sqrt(self.budget_megapixels * 1e6 / (width * height))
                return (
                    f"'{self.path.name}' is {width} x {height} px, more than fits in memory, so it was compared at "
                    f"{scale:.0%} of its size."
                )
        return None

    def pdf_page(self, page: int = 0) -> fitz.Page:
        if self.kind != "pdf":
            raise ValueError("pdf_page() is only valid for PDF artwork")
        assert self._pdf_doc is not None
        return self._pdf_doc[page]

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

    def render(self, dpi: float | None = None, page: int = 0) -> np.ndarray:
        """Return an RGB uint8 numpy array (H, W, 3).

        For PDFs this re-rasterizes the vector page at `dpi` (the working DPI, which is lower for a page too big to hold
        at 300) — full fidelity at every zoom level. An image is returned at its own pixels, reduced only if it is too big
        to hold (`budget_megapixels`), with a note in `reduced_note`.
        """
        if dpi is None:
            dpi = self.working_dpi(page)
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
        width, height = self._pil_image.size
        if width * height > self.budget_megapixels * 1e6:
            scale = math.sqrt(self.budget_megapixels * 1e6 / (width * height))
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            return np.array(self._pil_image.convert("RGB").resize(size, Image.Resampling.BOX))
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


def load_artwork(path: Path | str, limits: RuntimeConfig | None = None) -> LoadedArtwork:
    """Load a PDF or raster image, validating format, readability and size.

    Raises `UnsupportedFileFormatError` for an unrecognized extension and
    `CorruptFileError` if the file matches a supported extension but can't
    actually be parsed (rather than letting PyMuPDF/Pillow's raw exception,
    or a native-library crash, take down the caller — NFR-2). Raises
    `InputTooLargeError` for a file, page count or page size beyond `limits`,
    before anything large is allocated (Architecture §7).
    """
    limits = limits or RuntimeConfig()
    path = Path(path)
    if not path.is_file():
        raise CorruptFileError(f"File not found: {path}")
    size_bytes = path.stat().st_size
    if size_bytes == 0:
        raise CorruptFileError(f"File is empty: {path}")
    if limits.max_file_size_mb and size_bytes > limits.max_file_size_mb * _BYTES_PER_MB:
        raise InputTooLargeError(
            f"'{path.name}' is {size_bytes / _BYTES_PER_MB:.0f} MB, above the "
            f"{limits.max_file_size_mb} MB limit. Export a smaller version of the artwork, "
            "or raise \"runtime.max_file_size_mb\" in config.json if this file is genuine."
        )

    ext = path.suffix.lower()
    with StageTimer(logger, "load_artwork"):
        if ext in SUPPORTED_PDF_EXTENSIONS:
            artwork = _load_pdf(path, limits)
        elif ext in SUPPORTED_IMAGE_EXTENSIONS:
            artwork = _load_image(path, limits)
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


def _load_pdf(path: Path, limits: RuntimeConfig) -> LoadedArtwork:
    try:
        doc = fitz.open(path)
        if doc.page_count < 1:
            raise CorruptFileError(f"PDF has no pages: {path}")
        if limits.max_pdf_pages and doc.page_count > limits.max_pdf_pages:
            raise InputTooLargeError(
                f"'{path.name}' has {doc.page_count} pages, above the {limits.max_pdf_pages}-page limit. "
                "A comparison uses one page, so export just the artwork page."
            )
        # Touching page 0 forces PyMuPDF to parse content now, surfacing a
        # corrupt/truncated PDF here rather than deep inside a later render.
        _ = doc[0].rect
    except (CorruptFileError, InputTooLargeError):
        doc.close()
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
        budget_megapixels=affordable_megapixels(float(limits.max_render_megapixels)),
    )


def _load_image(path: Path, limits: RuntimeConfig) -> LoadedArtwork:
    try:
        Image.MAX_IMAGE_PIXELS = None  # a big scan is not a decompression bomb: it is held reduced, not refused
        image = Image.open(path)  # reads the header only; the pixels are decoded below
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
        budget_megapixels=affordable_megapixels(float(limits.max_render_megapixels)),
    )
