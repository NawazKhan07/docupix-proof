"""CIE Lab and CIEDE2000 for single colours, fast: the same formulas as scikit-image's `rgb2lab` (sRGB, D65, 2 degree
observer) and `deltaE_ciede2000` (kL = kC = kH = 1), without the cost of their array machinery for one colour at a time
-- measured ~2 ms a call there, called thousands of times a comparison."""

from __future__ import annotations

import math

_WHITE = (0.95047, 1.0, 1.08883)  # D65, 2 degree observer


def _linear(channel: float) -> float:
    return ((channel + 0.055) / 1.055) ** 2.4 if channel > 0.04045 else channel / 12.92


def _f(t: float) -> float:
    return t ** (1.0 / 3.0) if t > 0.008856 else 7.787 * t + 16.0 / 116.0


def rgb_to_lab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """An sRGB colour (each channel 0..1) as CIE Lab."""
    r, g, b = (_linear(float(c)) for c in rgb)
    x = (0.412453 * r + 0.357580 * g + 0.180423 * b) / _WHITE[0]
    y = (0.212671 * r + 0.715160 * g + 0.072169 * b) / _WHITE[1]
    z = (0.019334 * r + 0.119193 * g + 0.950227 * b) / _WHITE[2]
    fx, fy, fz = _f(x), _f(y), _f(z)
    lightness = 116.0 * fy - 16.0 if y > 0.008856 else 903.3 * y
    return lightness, 500.0 * (fx - fy), 200.0 * (fy - fz)


def ciede2000(lab1: tuple[float, float, float], lab2: tuple[float, float, float]) -> float:
    """CIEDE2000 colour difference between two Lab colours."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    c1, c2 = math.hypot(a1, b1), math.hypot(a2, b2)
    c_bar = (c1 + c2) / 2.0
    c7 = c_bar**7
    g = 0.5 * (1.0 - math.sqrt(c7 / (c7 + 25.0**7)))
    a1p, a2p = a1 * (1.0 + g), a2 * (1.0 + g)
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.atan2(b1, a1p) % (2 * math.pi)
    h2p = math.atan2(b2, a2p) % (2 * math.pi)
    dL = L2 - L1
    dC = c2p - c1p
    if c1p * c2p == 0:
        dh = 0.0
    else:
        dh = h2p - h1p
        if dh > math.pi:
            dh -= 2 * math.pi
        elif dh < -math.pi:
            dh += 2 * math.pi
    dH = 2.0 * math.sqrt(c1p * c2p) * math.sin(dh / 2.0)
    L_bar = (L1 + L2) / 2.0
    cp_bar = (c1p + c2p) / 2.0
    if c1p * c2p == 0:
        h_bar = h1p + h2p
    else:
        h_bar = (h1p + h2p) / 2.0
        if abs(h1p - h2p) > math.pi:
            h_bar += math.pi if h1p + h2p < 2 * math.pi else -math.pi
    t = (
        1.0
        - 0.17 * math.cos(h_bar - math.radians(30))
        + 0.24 * math.cos(2 * h_bar)
        + 0.32 * math.cos(3 * h_bar + math.radians(6))
        - 0.20 * math.cos(4 * h_bar - math.radians(63))
    )
    d_theta = math.radians(30) * math.exp(-(((math.degrees(h_bar) - 275.0) / 25.0) ** 2))
    cp7 = cp_bar**7
    r_c = 2.0 * math.sqrt(cp7 / (cp7 + 25.0**7))
    s_l = 1.0 + 0.015 * (L_bar - 50.0) ** 2 / math.sqrt(20.0 + (L_bar - 50.0) ** 2)
    s_c = 1.0 + 0.045 * cp_bar
    s_h = 1.0 + 0.015 * cp_bar * t
    r_t = -math.sin(2.0 * d_theta) * r_c
    return math.sqrt((dL / s_l) ** 2 + (dC / s_c) ** 2 + (dH / s_h) ** 2 + r_t * (dC / s_c) * (dH / s_h))


