"""Synthetic scenes for lifting a subject: artwork on a background, with the exact answer known.

Nothing here is tuned to any file: the artwork is generic packaging-like content
(panels, print, a roundel), the backgrounds are the kinds a carton is photographed
or printed on, and the true outline comes from how the scene was built.
"""

from __future__ import annotations

import cv2
import numpy as np

ORANGE = (238, 106, 45)
_WORDS = ("Each", "capsule", "contains", "Store", "below", "protect", "from", "moisture", "Keep", "out", "reach",
          "children", "Batch", "Expiry", "Rx", "only", "inhalation", "powder", "dosage", "physician")


def artwork(width: int = 700, height: int = 950, edge: str = "coloured", seed: int = 3) -> np.ndarray:
    """A carton-like face. `edge` says what the outermost margin is: coloured (contrasts with most
    backgrounds), white (does not contrast with a pale one), or dark."""
    rng = np.random.default_rng(seed)
    margin = {"coloured": (200, 60, 50), "white": (250, 250, 250), "dark": (30, 32, 40)}[edge]
    face = np.full((height, width, 3), margin, np.uint8)
    inner = (40, 40, width - 40, height - 40)
    cv2.rectangle(face, inner[:2], inner[2:], (236, 226, 214), -1)
    cv2.rectangle(face, (40, 40), (width - 40, 300), ORANGE, -1)
    cv2.circle(face, (width // 2, 520), 130, (60, 90, 190), -1)
    cv2.circle(face, (width // 2, 520), 70, (250, 250, 250), -1)
    cv2.rectangle(face, (70, 720), (width - 70, 860), (30, 120, 90), -1)
    for row in range(14):
        words = " ".join(rng.choice(_WORDS, size=int(rng.integers(3, 6))))
        cv2.putText(face, words, (70, 340 + row * 26 if row < 6 else 880 + (row - 6) * 0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1)
    cv2.putText(face, "NISTAMI", (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 5)
    return face


def background(kind: str, size: tuple[int, int], seed: int = 5) -> np.ndarray:
    """A surface to lay artwork on: (height, width) -> RGB."""
    height, width = size
    rng = np.random.default_rng(seed)
    if kind == "grey desk":
        base = np.full((height, width, 3), (150, 152, 156), np.float32)
        base += rng.normal(0, 3, base.shape)
    elif kind == "pale desk":
        base = np.full((height, width, 3), (228, 228, 226), np.float32)
        base += rng.normal(0, 2, base.shape)
    elif kind == "wood":
        xs = np.arange(width, dtype=np.float32)[None, :]
        grain = 18 * np.sin(xs / 7.0 + 3 * np.sin(np.arange(height, dtype=np.float32)[:, None] / 45.0))
        base = np.stack([150 + grain, 105 + 0.7 * grain, 65 + 0.4 * grain], axis=2)
        base += rng.normal(0, 4, base.shape)
    elif kind == "gradient":
        ramp = np.linspace(90, 200, width, dtype=np.float32)[None, :, None]
        base = np.repeat(np.repeat(ramp, height, axis=0), 3, axis=2) + np.array([0, 6, 12], np.float32)
        base += rng.normal(0, 3, base.shape)
    elif kind == "white page":
        base = np.full((height, width, 3), 255, np.float32)
    else:
        raise ValueError(kind)
    return np.clip(base, 0, 255).astype(np.uint8)


def compose(
    art: np.ndarray,
    bg_kind: str,
    canvas: tuple[int, int] = (1300, 1500),
    rotation_deg: float = 4.0,
    offset: tuple[int, int] = (0, 0),
    shadow: bool = True,
    scale: float = 1.0,
    seed: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """(image, truth mask): the artwork laid on the background, rotated, scaled and shifted."""
    height, width = canvas
    scene = background(bg_kind, canvas, seed)
    ah, aw = art.shape[:2]
    matrix = cv2.getRotationMatrix2D((aw / 2, ah / 2), rotation_deg, scale)
    matrix[0, 2] += width / 2 - aw / 2 + offset[0]
    matrix[1, 2] += height / 2 - ah / 2 + offset[1]

    layer = cv2.warpAffine(art, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    truth = cv2.warpAffine(np.full((ah, aw), 255, np.uint8), matrix, (width, height), flags=cv2.INTER_LINEAR) > 127

    if shadow:  # a soft drop shadow, as a photographed carton has
        cast = cv2.GaussianBlur(np.roll(np.roll(truth.astype(np.float32), 10, axis=0), 8, axis=1), (0, 0), 9)
        scene = (scene * (1 - 0.35 * cast[..., None])).astype(np.uint8)
    alpha = cv2.GaussianBlur(truth.astype(np.float32), (0, 0), 0.7)[..., None]
    scene = (layer * alpha + scene * (1 - alpha)).astype(np.uint8)
    return scene, truth


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def edge_error_px(found: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """(median, 95th percentile) distance in pixels from the found outline to the true one."""

    def outline(mask):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        canvas = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(canvas, contours, -1, 255, 1)
        return canvas > 0

    true_edge = outline(truth)
    distance_to_truth = cv2.distanceTransform((~true_edge).astype(np.uint8), cv2.DIST_L2, 3)
    errors = distance_to_truth[outline(found)]
    return float(np.median(errors)), float(np.percentile(errors, 95))


# ---------------------------------------------------------------------------------------------
# A packaging sheet: a carton's die-cut outline with printed panels inside it, and around it
# everything that is NOT artwork (title, colour legend, crop marks, dots, a leader line + label)
# ---------------------------------------------------------------------------------------------

SHEET_SIZE = (1560, 1300)  # (height, width)


def net_shapes() -> list[np.ndarray]:
    """The carton's flat net as polygons (int32 points): a body row, a glue flap, tuck and dust flaps."""

    def with_dome(x0, x1, y_join, y_end, up: bool) -> np.ndarray:
        radius = (x1 - x0) // 2
        centre = ((x0 + x1) // 2, y_end)
        arc = cv2.ellipse2Poly(centre, (radius, radius), 0, 180 if up else 0, 360 if up else 180, 6)
        if not up:
            arc = arc[::-1]  # keep the outline going round one way, not crossing itself
        return np.vstack([[(x0, y_join)], arc, [(x1, y_join)]]).astype(np.int32)

    return [
        np.array([(160, 620), (1140, 620), (1140, 980), (160, 980)], np.int32),  # the body row of panels
        np.array([(160, 640), (110, 675), (110, 925), (160, 960)], np.int32),  # glue flap
        with_dome(360, 620, 620, 470, up=True),  # tuck flap, rounded
        np.array([(640, 620), (662, 545), (858, 545), (880, 620)], np.int32),  # dust flap
        with_dome(620, 880, 980, 1110, up=False),  # bottom tuck, rounded
        np.array([(380, 980), (402, 1055), (598, 1055), (620, 980)], np.int32),  # bottom dust flap
    ]


def net_mask(shape=SHEET_SIZE) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    for polygon in net_shapes():
        cv2.fillPoly(mask, [polygon], 1)
    return mask.astype(bool)


def dieline_sheet(
    line_px: int = 2,
    line_colour=(20, 20, 20),
    frame: bool = False,
    gap_in_outline: int = 0,
    seed: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(sheet image, the artwork's true region, the pixels of everything that is not artwork).

    The flaps are unprinted paper (white) and only the outline separates them from the page.
    `gap_in_outline` cuts a break of that many pixels in the outline's straight top edge.
    """
    rng = np.random.default_rng(seed)
    height, width = SHEET_SIZE
    sheet = np.full((height, width, 3), 255, np.uint8)
    artwork_region = net_mask()

    # printed panels: a pale blue tint across the body row, a coloured band, text, a barcode-ish block
    cv2.rectangle(sheet, (160, 620), (1140, 980), (226, 238, 246), -1)
    cv2.rectangle(sheet, (160, 620), (200, 980), (60, 130, 60), -1)  # a dark green edge stripe
    cv2.rectangle(sheet, (200, 620), (216, 980), (245, 220, 60), -1)
    for row in range(6):
        words = " ".join(rng.choice(_WORDS, size=4))
        cv2.putText(sheet, words, (240, 700 + row * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
    cv2.rectangle(sheet, (900, 700), (1100, 900), (25, 25, 25), -1)
    for x in range(380, 600, 9):  # a barcode on the top flap
        cv2.line(sheet, (x, 520), (x, 590), (20, 20, 20), 2 + (x // 9) % 3)

    # everything below is drawn outside the carton
    cv2.putText(sheet, "Carton Size : L x W x H = 33 x 33 x 60 mm", (330, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)
    swatches = [((0, 130, 60), "Pantone 348 C_Green"), ((30, 190, 200), "Pantone 3115 C"), ((25, 25, 110), "Pantone 2756 C"),
                ((240, 225, 30), "Pantone Process Yellow C"), ((0, 120, 200), "Pantone 3005 C"), ((25, 25, 25), "Pantone Process Black C")]
    for index, (colour, label) in enumerate(swatches):
        x, y = 120 + (index % 2) * 560, 1370 + (index // 2) * 60
        cv2.rectangle(sheet, (x, y), (x + 40, y + 40), colour, -1)
        cv2.putText(sheet, label, (x + 60, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
    for y in (560, 1040):  # crop marks either side of the body row
        cv2.line(sheet, (30, y), (85, y), (20, 20, 20), 1)
        cv2.line(sheet, (1215, y), (1270, y), (20, 20, 20), 1)
    for centre in ((40, 250), (1260, 300), (60, 1200), (1230, 1180)):  # registration dots
        cv2.circle(sheet, centre, 4, (20, 20, 20), -1)

    # the artwork's outline: the boundary of the net, thin and closed
    contours, _ = cv2.findContours(artwork_region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = np.zeros((height, width), np.uint8)
    cv2.drawContours(outline, contours, -1, 255, line_px, cv2.LINE_8)
    if gap_in_outline:
        cv2.rectangle(outline, (700 - gap_in_outline // 2, 619 - line_px), (700 + gap_in_outline // 2, 622 + line_px), 0, -1)
    sheet[outline > 0] = line_colour
    for x in (360, 620, 880):  # fold lines between panels and where the flaps meet the body
        cv2.line(sheet, (x, 621), (x, 979), (60, 60, 60), 1)
    cv2.line(sheet, (360, 620), (620, 620), (60, 60, 60), 1)
    cv2.line(sheet, (620, 980), (880, 980), (60, 60, 60), 1)

    # a leader line touching the outline, with its label at the far end
    cv2.line(sheet, (750, 1240), (750, 1305), (20, 20, 20), 1)
    cv2.putText(sheet, "Silver Metallic Matt Board", (560, 1325), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 20, 120), 2)

    if frame:
        cv2.rectangle(sheet, (14, 14), (width - 15, height - 15), (20, 20, 20), 1)

    # the pixels that are NOT artwork: everything that is not white paper and lies outside the net (and its outline)
    near_artwork = cv2.dilate(artwork_region.astype(np.uint8), np.ones((2 * line_px + 5, 2 * line_px + 5), np.uint8)).astype(bool)
    not_paper = np.any(sheet < 235, axis=2)
    not_artwork = not_paper & ~near_artwork
    if frame:
        not_artwork[:30, :] = False
        not_artwork[-30:, :] = False
        not_artwork[:, :30] = False
        not_artwork[:, -30:] = False
    return sheet, artwork_region, not_artwork


def dieline_sheet_pdf(path, line_colour=(0.0, 0.0, 0.0), line_width_pt: float = 0.6, dashed_folds: bool = True) -> np.ndarray:
    """The same kind of sheet as a PDF with a VECTOR outline, in any stroke colour (a dieline is often a
    spot colour). Sized so that at 300 DPI a point is 25/6 pixels and coordinates match `dieline_sheet`.
    Returns the artwork's true region at that resolution."""
    import pymupdf as fitz

    height, width = SHEET_SIZE
    k = 72.0 / 300.0  # points per pixel
    doc = fitz.open()
    page = doc.new_page(width=width * k, height=height * k)

    def pt(x, y):
        return fitz.Point(x * k, y * k)

    # panels
    page.draw_rect(fitz.Rect(160 * k, 620 * k, 1140 * k, 980 * k), color=None, fill=(0.89, 0.93, 0.96))
    page.draw_rect(fitz.Rect(160 * k, 620 * k, 200 * k, 980 * k), color=None, fill=(0.24, 0.5, 0.24))
    page.insert_text(pt(240, 700), "AMPHOTERICIN B FOR INJECTION", fontsize=11)
    page.draw_rect(fitz.Rect(900 * k, 700 * k, 1100 * k, 900 * k), color=None, fill=(0.1, 0.1, 0.1))

    # not artwork: a title, a legend of filled swatches, crop marks, dots, a leader with a label
    page.insert_text(pt(330, 140), "Carton Size : L x W x H = 33 x 33 x 60 mm", fontsize=13)
    for index, colour in enumerate([(0, 0.5, 0.24), (0.1, 0.75, 0.8), (0.1, 0.1, 0.43), (0.94, 0.88, 0.1)]):
        x, y = 120 + (index % 2) * 560, 1370 + (index // 2) * 60
        page.draw_rect(fitz.Rect(x * k, y * k, (x + 40) * k, (y + 40) * k), color=None, fill=colour)
        page.insert_text(pt(x + 60, y + 30), f"Pantone {index}", fontsize=11)
    for y in (560, 1040):
        page.draw_line(pt(30, y), pt(85, y), color=(0, 0, 0), width=0.3)
        page.draw_line(pt(1215, y), pt(1270, y), color=(0, 0, 0), width=0.3)
    for centre in ((40, 250), (1260, 300)):
        page.draw_circle(pt(*centre), 4 * k, color=None, fill=(0, 0, 0))
    page.draw_line(pt(750, 1240), pt(750, 1305), color=(0, 0, 0), width=0.3)
    page.insert_text(pt(560, 1325), "Silver Metallic Matt Board", fontsize=11, color=(0.8, 0.08, 0.47))

    # fold lines, dashed like a real dieline's creases
    for x in (360, 620, 880):
        page.draw_line(pt(x, 621), pt(x, 979), color=(0.3, 0.3, 0.3), width=0.25, dashes="[3 2] 0" if dashed_folds else None)

    # the die-cut outline: one closed vector path per flap union boundary
    mask = net_mask()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(outline, 0.6, True)
    page.draw_polyline([pt(float(x), float(y)) for x, y in approx[:, 0, :]], color=line_colour, width=line_width_pt, closePath=True)

    doc.save(path)
    doc.close()
    return mask
