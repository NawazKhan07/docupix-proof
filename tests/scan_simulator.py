"""Turn a clean digital render into something that looks like it was scanned.

Real scans are the hard case for a comparison tool, but real scans of
confidential artwork are not always to hand. So this applies the same kinds of
damage a flatbed scanner does -- generic ones, nothing tuned to any file --
letting the scan path be exercised against a *known* answer: the digital
file the scan was made from.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ScanProfile:
    rotation_deg: float = 0.9  # paper never sits perfectly square on the glass
    scale: float = 0.83  # e.g. a 250 DPI scan of a 300 DPI render
    shift_px: tuple[float, float] = (14.0, 9.0)
    blur_sigma: float = 0.8  # optics + resampling softness
    noise_sigma: float = 4.0  # sensor noise, in 0-255 levels
    vignette: float = 0.12  # lid shadow: edges darker than the centre
    brightness: float = 0.94  # white paper is never quite white
    colour_cast: tuple[float, float, float] = (1.0, 0.985, 0.955)  # warm lamp (R, G, B gain)
    jpeg_quality: int = 78  # scans are usually JPEG on the way in
    seed: int = 11


LIGHT = ScanProfile(rotation_deg=0.4, scale=1.0, shift_px=(4.0, 3.0), blur_sigma=0.5, noise_sigma=2.0, vignette=0.05, brightness=0.97, colour_cast=(1.0, 0.995, 0.98), jpeg_quality=90)
TYPICAL = ScanProfile()
HARSH = ScanProfile(rotation_deg=1.8, scale=0.6, shift_px=(30.0, -22.0), blur_sigma=1.4, noise_sigma=9.0, vignette=0.22, brightness=0.88, colour_cast=(1.0, 0.97, 0.92), jpeg_quality=60)


def simulate_scan(image: np.ndarray, profile: ScanProfile = TYPICAL) -> np.ndarray:
    """Return an RGB image degraded like a scan of `image`."""
    height, width = image.shape[:2]
    out_width, out_height = round(width * profile.scale), round(height * profile.scale)

    # Geometry: rotate about the page centre, rescale, and shift, in one warp. That
    # matrix keeps the centre where it was, so move the centre to the middle of the
    # (differently sized) output -- otherwise part of the page falls off the scan.
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), profile.rotation_deg, profile.scale)
    matrix[0, 2] += out_width / 2 - width / 2 + profile.shift_px[0]
    matrix[1, 2] += out_height / 2 - height / 2 + profile.shift_px[1]
    scanned = cv2.warpAffine(
        image, matrix, (out_width, out_height), flags=cv2.INTER_AREA if profile.scale < 1 else cv2.INTER_LINEAR,
        borderValue=(255, 255, 255),
    ).astype(np.float32)

    if profile.blur_sigma > 0:
        scanned = cv2.GaussianBlur(scanned, (0, 0), profile.blur_sigma)

    # Lighting: vignette (a lid shadow towards the edges), overall dimming, colour cast.
    ys, xs = np.mgrid[0:out_height, 0:out_width].astype(np.float32)
    radius = np.hypot((xs - out_width / 2) / (out_width / 2), (ys - out_height / 2) / (out_height / 2))
    scanned *= (1.0 - profile.vignette * np.clip(radius, 0, 1.4) ** 2)[..., None]
    scanned *= profile.brightness
    scanned *= np.array(profile.colour_cast, dtype=np.float32)[None, None, :]

    rng = np.random.default_rng(profile.seed)
    scanned += rng.normal(0.0, profile.noise_sigma, scanned.shape).astype(np.float32)
    scanned = np.clip(scanned, 0, 255).astype(np.uint8)

    # JPEG round trip (RGB <-> BGR for OpenCV's codec).
    ok, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(scanned, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, profile.jpeg_quality]
    )
    assert ok
    return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
