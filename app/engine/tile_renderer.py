"""On-demand, screen-resolution rendering of part of a page (SRS FR-9.1).

FR-9.1 asks for "full original quality at every zoom level; no visible
degradation or pixelation near text edges when fully zoomed in". The viewer
holds the aligned pair as 300 DPI rasters, which is right for detection and
for zoomed-out views but not for looking closely: at 600% a text edge spans
about twelve intermediate pixels when that raster is magnified, against two
when the PDF is rendered again from its vector data (measured).

So above 100% the viewer asks for a *tile*: just the visible region,
rendered at the on-screen scale straight from the PDF. Coordinates here are
"device pixels": the position in the shared aligned frame (the original's
300 DPI pixels) multiplied by the zoom, i.e. exactly what is on screen.

The suspect is rendered through the alignment transform rather than from its
warped raster, so it is as crisp as the original. Sources that are not
vector (a PNG, a scan) have nothing more to give, and fall back to a bicubic
resample of the raster, which is the best that can honestly be done.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import pymupdf as fitz
import numpy as np

from app.engine.file_loader import LoadedArtwork

_PDF_POINTS_DPI = 72.0
_EDGE_PAD_PX = 2  # over-render past the tile so edge pixels are never clipped
_MARGIN_PX = 3.0  # extra source margin around a warped tile, in working pixels
_WHITE = (255, 255, 255)


@dataclass(frozen=True)
class TileRect:
    """A region in device pixels: shared-frame pixels multiplied by the zoom."""

    x: int
    y: int
    width: int
    height: int


def is_vector(artwork: LoadedArtwork | None) -> bool:
    return artwork is not None and artwork.kind == "pdf"


def _place(pixmap: fitz.Pixmap, width: int, height: int) -> np.ndarray:
    """Copy a rendered pixmap onto a white canvas of exactly `width` x `height`.

    The tile origin is (0, 0); the pixmap reports where it actually landed
    (it may start a pixel or two before the tile, from the edge padding).
    """
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    samples = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )
    if pixmap.n == 1:
        samples = np.repeat(samples, 3, axis=2)

    dest_x0, dest_y0 = max(0, pixmap.x), max(0, pixmap.y)
    dest_x1 = min(width, pixmap.x + pixmap.width)
    dest_y1 = min(height, pixmap.y + pixmap.height)
    if dest_x1 > dest_x0 and dest_y1 > dest_y0:
        canvas[dest_y0:dest_y1, dest_x0:dest_x1] = samples[
            dest_y0 - pixmap.y : dest_y1 - pixmap.y, dest_x0 - pixmap.x : dest_x1 - pixmap.x, :3
        ]
    return canvas


def render_vector_tile(
    artwork: LoadedArtwork, zoom: float, rect: TileRect, page: int = 0
) -> np.ndarray:
    """Render `rect` of a PDF page straight from its vector content.

    Returns an (rect.height, rect.width, 3) RGB array, whatever the page size:
    anything beyond the page edge is white.
    """
    if not is_vector(artwork):
        raise ValueError("render_vector_tile needs a PDF; use render_raster_tile for images")

    scale = zoom * artwork.working_dpi(page) / _PDF_POINTS_DPI  # device pixels per PDF point
    # Shift so the tile's own origin lands on device (0, 0); the clip is then
    # the same region expressed back in page coordinates, padded slightly.
    matrix = fitz.Matrix(scale, 0, 0, scale, -rect.x, -rect.y)
    clip = fitz.Rect(
        (rect.x - _EDGE_PAD_PX) / scale,
        (rect.y - _EDGE_PAD_PX) / scale,
        (rect.x + rect.width + _EDGE_PAD_PX) / scale,
        (rect.y + rect.height + _EDGE_PAD_PX) / scale,
    )
    pixmap = artwork.pdf_page(page).get_pixmap(matrix=matrix, clip=clip, alpha=False)
    return _place(pixmap, rect.width, rect.height)


def render_raster_tile(image: np.ndarray, zoom: float, rect: TileRect) -> np.ndarray:
    """Resample a raster to `rect` at `zoom`, for sources with no vector data.

    Bicubic, and pixel-centre aware so a zoom of 1 reproduces the source
    exactly. This cannot add detail the raster does not have.
    """
    to_source = np.array(
        [
            [1 / zoom, 0.0, (rect.x + 0.5) / zoom - 0.5],
            [0.0, 1 / zoom, (rect.y + 0.5) / zoom - 0.5],
        ]
    )
    return cv2.warpAffine(
        image,
        to_source,
        (rect.width, rect.height),
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_WHITE,
    )


def render_aligned_vector_tile(
    artwork: LoadedArtwork,
    alignment_transform: np.ndarray,
    zoom: float,
    rect: TileRect,
    page: int = 0,
) -> np.ndarray:
    """A PDF rendered from vector data into the original's frame, crisp at `zoom`.

    `alignment_transform` maps this file's working-resolution pixels into the
    shared frame. Rather than warping an already-rasterized page (which would
    cap sharpness at 300 DPI), the region of the page that maps onto the
    tile is rendered at the on-screen scale and then warped by the
    transform's small residual -- so the suspect is as crisp as the original.
    """
    width, height = rect.width, rect.height

    # Output pixel index -> position in the shared frame (working pixels).
    to_shared = np.array(
        [
            [1 / zoom, 0.0, (rect.x + 0.5) / zoom - 0.5],
            [0.0, 1 / zoom, (rect.y + 0.5) / zoom - 0.5],
            [0.0, 0.0, 1.0],
        ]
    )
    to_suspect = np.linalg.inv(np.asarray(alignment_transform, dtype=np.float64))

    # Which part of the suspect page lands on this tile?
    corners = np.array(
        [[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]], dtype=np.float64
    ).T
    mapped = to_suspect @ to_shared @ corners
    mapped = mapped[:2] / mapped[2]
    low = mapped.min(axis=1) - _MARGIN_PX
    high = mapped.max(axis=1) + _MARGIN_PX

    source = TileRect(
        x=int(np.floor((low[0] + 0.5) * zoom)),
        y=int(np.floor((low[1] + 0.5) * zoom)),
        width=max(1, int(np.ceil((high[0] - low[0]) * zoom))),
        height=max(1, int(np.ceil((high[1] - low[1]) * zoom))),
    )
    rendered = render_vector_tile(artwork, zoom, source, page)

    # Output pixel -> source pixel: through the shared frame, back into the
    # suspect's own pixels, then into the rendered patch (whose origin sits
    # at `source.x, source.y` in device pixels).
    to_patch = np.array(
        [
            [zoom, 0.0, 0.5 * zoom - source.x - 0.5],
            [0.0, zoom, 0.5 * zoom - source.y - 0.5],
            [0.0, 0.0, 1.0],
        ]
    )
    output_to_patch = to_patch @ to_suspect @ to_shared
    return cv2.warpPerspective(
        rendered,
        output_to_patch,
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_WHITE,
    )
