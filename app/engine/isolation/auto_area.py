"""Finding the artwork on a sheet without being told where it is (Automatic mode).

A packaging file is a sheet: the carton's die-cut outline with the printed panels inside it and, around it, a title,
a colour legend, crop marks, notes and often a table of details that is closed by a line too. Without a crop or a lift,
Automatic mode finds the artwork itself: the closed border on each file that CORRESPONDS to one on the other (where a
sheet has several, the pair whose printing matches best, whatever their sizes on their sheets), trimmed to that, and the
two trimmed files are aligned and compared. Where a file has no border to find (a flat label, a photograph), the whole
page is compared, as before.

Both files or neither: if the borders found do not correspond (they do not share enough printing, or one is far larger or
differently shaped than the other), one of them is a panel or a box and trimming to it would compare the wrong things.

Where only ONE file has a border to find (a scan or photograph, whose die line is too faint to trace, beside the digital
file it was printed from), that border is laid over the other file where its printing matches -- exactly what a reviewer
does by hand with Lift and Place -- and both are trimmed to it. Only a confident match is taken; otherwise the whole pages
are compared.
"""

from __future__ import annotations

from dataclasses import replace

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import AlignmentConfig, SeriesConfig, SubjectConfig
from app.engine.alignment.aligner import AlignmentError
from app.engine.isolation.border_lifter import border_regions
from app.engine.isolation.cutout import Cutout
from app.engine.isolation.placement import fit_placement
from app.engine.isolation.subject_lifter import LiftedSubject, outline_from_mask
from app.models.isolation import IsolationGeometry, NormalizedRect, NormalizedPolygon

logger = logging.getLogger(__name__)

_CANDIDATES = 3  # the biggest closed borders of each file that are tried against each other
_MIN_CORNERS = 6  # a die-cut outline has corners and notches (a carton: a dozen or more; a small foil: seven); a box is four...
_BOX_INK_SHARE = 0.6  # ...and a box is the artwork only if it holds this much of everything printed on the page (a label: 0.65-0.8; a box round a logo: 0.15-0.4)
# A scan's dark-line detection, bridging small gaps between letters, can trace a dense paragraph of text as
# its own "closed border" once enough of it happens to enclose -- corners in the hundreds, area a fraction of
# the real artwork. Perimeter^2 / area (how jagged a shape is for its size, scale-independent) tells this from
# a genuine die-cut outline cleanly where a corner count cannot: surveyed across ten real cartons' own outlines
# (a simple box to one with 172 corners of tabs and notches), every one measured 17.7-32.6; the real text-blob
# false positive that prompted this measured 87.9 -- a wide, safe margin either side.
_MAX_BORDER_COMPACTNESS = 50.0
_MATCH_LONG_EDGE_PX = 1200  # printing is matched on copies this size
_MATCH_FEATURES = 2500
_MIN_INLIERS = 25  # a pair of borders that share fewer consistent features than this do not correspond
_MAX_AREA_RATIO = 1.8  # where the choice is not made by matching: the shares of their pages may differ by this...
_MAX_ASPECT_RATIO = 1.5  # ...and their shapes (width to height) by this
_PLACED_MARGIN_PX = 1.5  # a placed outline is only as exact as the fit: comparing starts this much further in
_MIN_PLACED_INSIDE = 0.9  # a placed outline must lie at least this much on the other page...
_MIN_PLACED_SHARE = 0.02  # ...and cover at least this share of it
_MAX_EXTENT_RATIO = 1.6  # two borders that correspond cover the same artwork: one carried onto the other is about its size


@dataclass
class AutoArea:
    original: IsolationGeometry | None
    suspect: IsolationGeometry | None
    note: str = ""  # what was done, in words (empty if nothing was)


def _compactness(polygon: list) -> float:
    """How jagged a closed polygon is for its size (perimeter^2 / area, scale-independent): high for a
    paragraph of text traced as a false "border", low for any genuine die-cut outline -- see
    `_MAX_BORDER_COMPACTNESS`."""
    points = np.array([(round(x), round(y)) for x, y in polygon], dtype=np.int32)
    area = cv2.contourArea(points)
    if area <= 0:
        return float("inf")
    return float(cv2.arcLength(points, True) ** 2) / area


def _subjects(image: np.ndarray, config: SubjectConfig, page, dpi: float | None) -> list[LiftedSubject]:
    """The closed borders of a sheet, as lifted subjects, the biggest first."""
    try:
        regions = border_regions(image, config, page=page, dpi=dpi, limit=_CANDIDATES)
    except Exception:  # noqa: BLE001 -- finding the artwork is a help, never a reason for a comparison to fail
        logger.exception("automatic artwork detection failed")
        return []
    height, width = image.shape[:2]
    subjects = [
        LiftedSubject(region.mask, outline_from_mask(region.mask, config), region.source, (width, height), [], region.line_width_px + config.border_line_margin_px)
        for region in regions
    ]
    subjects = [subject for subject in subjects if _compactness(subject.polygon) <= _MAX_BORDER_COMPACTNESS]
    # A box drawn round a logo or a table of details is closed by a line too, but it is not the artwork: a die-cut outline
    # is not a rectangle. (A label that IS a box has nearly all the page's printing inside it.)
    return [subject for subject in subjects if len(subject.polygon) >= _MIN_CORNERS or _ink_inside(image, subject) >= _BOX_INK_SHARE]


def _ink_inside(image: np.ndarray, subject: LiftedSubject) -> float:
    """The share of everything printed on the page (all that is not paper-white) that lies inside the border."""
    height, width = image.shape[:2]
    scale = min(1.0, 800 / max(height, width))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    inked = gray < 215
    total = int(inked.sum())
    if total == 0:
        return 0.0
    inside = cv2.resize(subject.mask.astype(np.uint8), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return float((inked & inside).sum()) / total


def _aspect(subject: LiftedSubject) -> float:
    x0, y0, x1, y1 = subject.bbox
    return max(1.0, x1 - x0) / max(1.0, y1 - y0)


def _geometry(subject: LiftedSubject, shape: tuple[int, int]) -> IsolationGeometry:
    height, width = shape
    return IsolationGeometry(
        boundary=subject.normalized_polygon(),
        boundary_origin="lifted",
        edge_inset_norm=subject.edge_inset_px / max(height, width),
    )


@dataclass
class _Printing:
    """The printing inside a border, on a reduced copy: its features, and the border's own corners and area there."""

    keypoints: tuple
    descriptors: np.ndarray | None
    corners: np.ndarray  # (4, 1, 2) float32: the border's bounding box
    area: float  # the border's own area (pixels of the reduced copy)


def _features(image: np.ndarray, subject: LiftedSubject) -> _Printing:
    """Keypoints and descriptors of the printing inside a border, on a reduced copy."""
    height, width = image.shape[:2]
    scale = min(1.0, _MATCH_LONG_EDGE_PX / max(height, width))
    small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = cv2.resize(subject.mask.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST) * 255
    detector = cv2.ORB_create(nfeatures=_MATCH_FEATURES)
    keypoints, descriptors = detector.detectAndCompute(small, mask)
    x0, y0, x1, y1 = (value * scale for value in subject.bbox)
    corners = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).reshape(-1, 1, 2)
    return _Printing(keypoints, descriptors, corners, float(np.count_nonzero(mask)))


def _corresponding(first: _Printing, second: _Printing) -> int:
    """How many consistent features two borders' printing shares (0 if none can be told, or if they are not the same
    extent of artwork).

    Sharing printing is not enough on its own: a small box on one sheet (a proof-reader's marks round a paragraph, a
    panel) shares every feature inside it with the whole carton on the other. So the match must also carry the one
    border onto something the size of the other -- the whole artwork onto the whole artwork, not a piece onto it.
    """
    kp_a, desc_a, kp_b, desc_b = first.keypoints, first.descriptors, second.keypoints, second.descriptors
    if desc_a is None or desc_b is None or len(kp_a) < 8 or len(kp_b) < 8:
        return 0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_b, desc_a, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    if len(good) < 8:
        return 0
    source = np.float32([kp_b[m.queryIdx].pt for m in good])
    target = np.float32([kp_a[m.trainIdx].pt for m in good])
    transform, inliers = cv2.findHomography(source, target, cv2.RANSAC, 6.0)
    if transform is None or inliers is None:
        return 0
    carried = cv2.perspectiveTransform(second.corners, transform)
    carried_area = abs(cv2.contourArea(carried))
    own_area = abs(cv2.contourArea(first.corners))
    if own_area <= 0 or not (1.0 / _MAX_EXTENT_RATIO <= carried_area / own_area <= _MAX_EXTENT_RATIO):
        return 0
    return int(inliers.sum())


def _placed(source: np.ndarray, subject: LiftedSubject, target: np.ndarray, expected_scale: float | None) -> IsolationGeometry | None:
    """`subject` (a border on `source`) laid over `target` where its printing matches, or None without a confident fit."""
    try:
        cutout = Cutout.from_subject(source, subject)
        fit = fit_placement(cutout, target, AlignmentConfig(), expected_scale)
    except (AlignmentError, ValueError, cv2.error):
        return None
    # Only a match of the printing itself: matching bare edges (Place's fallback, for a reviewer steering it by eye)
    # would lay the artwork over any empty box of about its shape.
    if fit.alignment.low_confidence or fit.alignment.method == "template":
        return None
    height, width = target.shape[:2]
    points = np.array(fit.placement.map_points(cutout.polygon, cutout.size), np.float64)
    inside = np.clip(points, 0, [width - 1, height - 1])
    whole = abs(cv2.contourArea(points.astype(np.float32)))
    kept = abs(cv2.contourArea(inside.astype(np.float32)))
    if whole <= 0 or kept / whole < _MIN_PLACED_INSIDE or kept / (width * height) < _MIN_PLACED_SHARE:
        return None
    return IsolationGeometry(
        boundary=NormalizedPolygon(points=[(float(x) / width, float(y) / height) for x, y in inside]),
        boundary_origin="placed",
        edge_inset_norm=(subject.edge_inset_px * fit.placement.scale + _PLACED_MARGIN_PX) / max(height, width),
    )


def _place_across(
    original: np.ndarray,
    suspect: np.ndarray,
    firsts: list[LiftedSubject],
    seconds: list[LiftedSubject],
    original_dpi: float | None,
    suspect_dpi: float | None,
) -> AutoArea:
    """Where no pair of borders was found, one file's border laid over the other (the original's, first)."""
    scale = suspect_dpi / original_dpi if original_dpi and suspect_dpi else None
    if firsts:
        geometry = _placed(original, firsts[0], suspect, scale)
        if geometry is not None:
            return _placed_area(_geometry(firsts[0], original.shape[:2]), geometry, firsts[0], "original")
    if seconds:
        geometry = _placed(suspect, seconds[0], original, 1.0 / scale if scale else None)
        if geometry is not None:
            return _placed_area(geometry, _geometry(seconds[0], suspect.shape[:2]), seconds[0], "suspect")
    return AutoArea(None, None, "")


def _placed_area(original: IsolationGeometry, suspect: IsolationGeometry, subject: LiftedSubject, where: str) -> AutoArea:
    logger.info("artwork found in one file and placed on the other", extra={"found_in": where, "share": round(subject.area_fraction, 3)})
    return AutoArea(
        original,
        suspect,
        f"The artwork was found in the {where} file ({subject.area_fraction:.0%} of its page) and laid over the other "
        "where its printing matches; only it was compared.",
    )


def find_artwork(
    original: np.ndarray,
    suspect: np.ndarray,
    config: SubjectConfig,
    original_page=None,
    suspect_page=None,
    original_dpi: float | None = None,
    suspect_dpi: float | None = None,
) -> AutoArea:
    """The artwork area of each file, or neither (see the module's account). A PDF's page and DPI, where they are known,
    let its vector lines be the border."""
    with ThreadPoolExecutor(max_workers=2) as pool:  # each file's borders are found on their own, side by side
        firsts, seconds = pool.map(
            lambda job: _subjects(job[0], config, job[1], job[2]),
            [(original, original_page, original_dpi), (suspect, suspect_page, suspect_dpi)],
        )
    if not firsts or not seconds:
        return _place_across(original, suspect, firsts, seconds, original_dpi, suspect_dpi)

    if len(firsts) == 1 and len(seconds) == 1:
        first, second = firsts[0], seconds[0]
        ratio = max(first.area_fraction, second.area_fraction) / max(1e-9, min(first.area_fraction, second.area_fraction))
        shapes = max(_aspect(first), _aspect(second)) / max(1e-9, min(_aspect(first), _aspect(second)))
        if ratio > _MAX_AREA_RATIO or shapes > _MAX_ASPECT_RATIO:
            # A different margin or bleed on an otherwise identical sheet (one file's border sitting on a page
            # with a lot more blank space round it than the other's, or printed a little smaller or larger
            # altogether) disagrees on how much of their own pages the two borders occupy without the artwork
            # itself differing at all -- so this is not given up on until the printing itself is checked too,
            # the same way several borders on a sheet are told apart below: a strong feature correspondence is
            # much stronger evidence that these are the same artwork than their two area shares agreeing.
            score = _corresponding(_features(original, first), _features(suspect, second))
            if score < _MIN_INLIERS:
                logger.info(
                    "automatic artwork areas disagree and their printing does not correspond",
                    extra={"area_ratio": round(ratio, 2), "shape_ratio": round(shapes, 2), "inliers": score},
                )
                return _place_across(original, suspect, firsts, seconds, original_dpi, suspect_dpi)
    else:
        # several closed borders on a sheet (a carton, a table of details): the pair whose printing matches is the artwork
        first_features = [_features(original, subject) for subject in firsts]
        second_features = [_features(suspect, subject) for subject in seconds]
        best, best_score = None, 0
        for i, a in enumerate(first_features):
            for j, b in enumerate(second_features):
                score = _corresponding(a, b)
                if score > best_score:
                    best, best_score = (i, j), score
        if best is None or best_score < _MIN_INLIERS:
            logger.info("no pair of borders corresponds", extra={"best_inliers": best_score})
            return _place_across(original, suspect, firsts, seconds, original_dpi, suspect_dpi)
        first, second = firsts[best[0]], seconds[best[1]]

    logger.info(
        "artwork found automatically",
        extra={"original_share": round(first.area_fraction, 3), "suspect_share": round(second.area_fraction, 3)},
    )
    return AutoArea(
        _geometry(first, original.shape[:2]),
        _geometry(second, suspect.shape[:2]),
        f"The artwork was found automatically in both files ({first.area_fraction:.0%} and {second.area_fraction:.0%} of their pages) "
        "and only it was compared.",
    )


# ---------------------------------------------------------------------------
# Found by what is printed (the last resort: no border to go by)
# ---------------------------------------------------------------------------

_BLOB_LONG_EDGE_PX = 800  # a page's print is gathered on a copy this size
_BLOB_FROM_PAPER = 40  # (grey levels) darker than the page's paper by this much is print
_BLOB_THIN_SHARE = 0.004  # a line thinner than this share of the page (a scanner's edge, a rule) is not the artwork...
_BLOB_GATHER_SHARE = 0.03  # ...and print this close together (share of the page) is one piece of artwork
_BLOB_MARGIN_SHARE = 0.03  # the area compared is the piece with this much (share of its size) round it


def print_block(image: np.ndarray) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1): the largest piece of print on a page -- one artwork on a sheet, a label laid on a scanner's
    glass -- with thin lines (the scanner's edges, a rule) left out and nearby print gathered into it."""
    height, width = image.shape[:2]
    scale = min(1.0, _BLOB_LONG_EDGE_PX / max(height, width))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    darkest = small.min(axis=2) if small.ndim == 3 else small
    paper = float(np.percentile(darkest, 90))
    ink = (darkest.astype(np.int16) < paper - _BLOB_FROM_PAPER).astype(np.uint8)
    long_edge = max(small.shape[:2])
    thin = max(3, round(_BLOB_THIN_SHARE * long_edge)) | 1
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((thin, thin), np.uint8))
    gather = max(5, round(_BLOB_GATHER_SHARE * long_edge)) | 1
    pieces = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((gather, gather), np.uint8))
    count, _labels, stats, _centres = cv2.connectedComponentsWithStats(pieces)
    if count <= 1:
        return 0, 0, width, height
    best = max(range(1, count), key=lambda label: stats[label][cv2.CC_STAT_AREA])
    x, y, w, h, _area = stats[best]
    return round(x / scale), round(y / scale), min(width, round((x + w) / scale)), min(height, round((y + h) / scale))


_EXTENT_MIN_PIECE = 0.02  # (a piece of print at least this share of the largest one...
_EXTENT_NEAR_SHARE = 0.25  # ...and at most this share of the item's size from it belongs to the item)


def _print_pieces(image: np.ndarray) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """A page's print gathered into pieces, on a copy at most `_BLOB_LONG_EDGE_PX` long (thin lines left out): the copy,
    and the pieces as `cv2.connectedComponentsWithStats` gives them."""
    height, width = image.shape[:2]
    scale = min(1.0, _BLOB_LONG_EDGE_PX / max(height, width))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    darkest = small.min(axis=2) if small.ndim == 3 else small
    paper = float(np.percentile(darkest, 90))
    ink = (darkest.astype(np.int16) < paper - _BLOB_FROM_PAPER).astype(np.uint8)
    long_edge = max(small.shape[:2])
    thin = max(3, round(_BLOB_THIN_SHARE * long_edge)) | 1
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((thin, thin), np.uint8))
    gather = max(5, round(_BLOB_GATHER_SHARE * long_edge)) | 1
    pieces = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((gather, gather), np.uint8))
    count, labels, stats, _centres = cv2.connectedComponentsWithStats(pieces)
    return small, count, labels, stats


def item_in_artwork(artwork: np.ndarray, item: np.ndarray) -> np.ndarray:
    """`item` (a mask over `artwork`, e.g. a scanned label's extent laid over the digital file) widened by every piece
    of `artwork`'s own print lying mostly inside it, whole: what the item should hold, even where the item lacks it (a
    line missing at a label's foot is part of the label's print in the digital file, and is compared), while print of
    its own beside it (a proof sheet's note, another artwork) is not."""
    height, width = artwork.shape[:2]
    small, count, labels, _stats = _print_pieces(artwork)
    if count <= 1:
        return item
    inside = cv2.resize(item.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    totals = np.bincount(labels.ravel(), minlength=count)
    within = np.bincount(labels.ravel(), weights=inside.ravel(), minlength=count)
    wanted = np.flatnonzero(within[1:] >= _ITEM_PIECE_INSIDE * totals[1:]) + 1
    if not len(wanted):
        return item
    pieces = np.isin(labels, wanted).astype(np.uint8)
    return item | (cv2.resize(pieces, (width, height), interpolation=cv2.INTER_NEAREST) > 0)


_ITEM_PIECE_INSIDE = 0.5  # (a piece of the artwork's print at least this much inside the item belongs to it)


def scan_extent(image: np.ndarray) -> np.ndarray | None:
    """Where a scanned item lies on the scanner's page, as a mask the page's size: the hull of its print -- every
    sizable piece of it, the largest (the label) and those beside it (a barcode, a code block) -- with thin lines left
    out (the glass's edges, a lid's shadow, a fold) and its margin round it. None where the print is most of the page
    (the scan IS the item) or there is none."""
    height, width = image.shape[:2]
    small, count, labels, stats = _print_pieces(image)
    if count <= 1:
        return None
    # the largest piece is the item; a sizable piece beside it (within a share of the item's size) is part of it too,
    # and so on as the item grows -- specks out on the glass, or its far edges, are not
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    x0, y0 = stats[largest, cv2.CC_STAT_LEFT], stats[largest, cv2.CC_STAT_TOP]
    x1, y1 = x0 + stats[largest, cv2.CC_STAT_WIDTH], y0 + stats[largest, cv2.CC_STAT_HEIGHT]
    members, grown = {largest}, True
    while grown:
        grown = False
        near = _EXTENT_NEAR_SHARE * max(x1 - x0, y1 - y0)
        for label in range(1, count):
            if label in members or areas[label - 1] < _EXTENT_MIN_PIECE * areas.max():
                continue
            lx0, ly0 = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP]
            lx1, ly1 = lx0 + stats[label, cv2.CC_STAT_WIDTH], ly0 + stats[label, cv2.CC_STAT_HEIGHT]
            if lx0 - near <= x1 and x0 - near <= lx1 and ly0 - near <= y1 and y0 - near <= ly1:
                members.add(label)
                x0, y0, x1, y1 = min(x0, lx0), min(y0, ly0), max(x1, lx1), max(y1, ly1)
                grown = True
    kept = np.isin(labels, sorted(members))
    points = cv2.findNonZero(kept.astype(np.uint8))
    hull = np.zeros(small.shape[:2], np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(points), 1)
    x, y, w, h = cv2.boundingRect(points)
    margin = max(1, round(_BLOB_MARGIN_SHARE * max(w, h))) | 1
    hull = cv2.dilate(hull, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    if hull.mean() > _EXTENT_MOST_OF_PAGE:
        return None
    return cv2.resize(hull, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)


_EXTENT_MOST_OF_PAGE = 0.8  # (an item filling more of the page than this is the page: nothing to narrow)


def _rect(box: tuple[int, int, int, int], shape: tuple[int, int]) -> IsolationGeometry:
    height, width = shape
    x0, y0, x1, y1 = box
    mx, my = _BLOB_MARGIN_SHARE * (x1 - x0), _BLOB_MARGIN_SHARE * (y1 - y0)
    return IsolationGeometry(
        boundary=NormalizedRect(max(0.0, (x0 - mx) / width), max(0.0, (y0 - my) / height), min(1.0, (x1 + mx) / width), min(1.0, (y1 + my) / height)),
        boundary_origin="manual",
    )


_PRINT_TURNS = (0, 1, 3, 2)  # quarter turns the piece of print is looked for at (a label laid on the glass any way round)
_PRINT_SIZES = (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0)  # ...and sizes: a proof sheet also shows its artwork enlarged 200% or 300%
_PRINT_STEP = 1.03  # (each size give or take a few steps of this)
_PRINT_STEPS_EACH_WAY = 4


def _located_any_way(
    sheet: np.ndarray, piece: np.ndarray, config: SeriesConfig, scale: float | None
) -> tuple[float, tuple[int, int, int, int]] | None:
    """(how well, where on `sheet`) the piece of print is found, turned any quarter turn and at any of the sizes a proof
    sheet shows an artwork at; None if nowhere. The upright, same-size match is tried first and kept if it is sure."""
    from app.engine.series import _locate, _search_picture  # (a search the Series mode already makes)

    if min(piece.shape[:2]) < 8:
        return None
    centre = 1.0 / scale if scale and scale > 0 else 1.0

    def ways() -> list[tuple[int, float, np.ndarray]]:
        out = []
        for turns in _PRINT_TURNS:
            turned = np.ascontiguousarray(np.rot90(piece, turns)) if turns else piece
            out += [(turns, size, turned) for size in _PRINT_SIZES]
        return out

    def scales_of(size: float) -> list[float]:
        return [centre * size * _PRINT_STEP**step for step in range(-_PRINT_STEPS_EACH_WAY, _PRINT_STEPS_EACH_WAY + 1)]

    looked_in = _search_picture(sheet, config)
    # Upright first, at every size: the usual case, and if it is found surely so, done.
    upright = None
    for size in _PRINT_SIZES:
        located = _locate(sheet, piece, config, scale, scales=scales_of(size), looked_in=looked_in)
        if located is not None and (upright is None or located[0] > upright[0]):
            upright = located
    if upright is not None and upright[0] >= _SURE_PRINT_MATCH:
        return upright[0], upright[1]
    # Every other way round, at every size, is then looked for on a copy of the sheet half the size (a quarter of the work), only to
    # rank them -- a weak match (a foil's, a soft scan's) falls short of the bar on the small copy and is still the
    # likeliest -- and the likeliest few are looked for as closely as ever.
    coarse = _search_picture(sheet, replace(config, search_long_edge_px=max(200, round(config.search_long_edge_px * _COARSE_SHARE))))
    ranking = replace(config, min_match_score=-1.0, min_match_peak=float("-inf"))
    rough = []
    for turns, size, turned in ways():
        if not turns:
            continue  # (upright was looked for closely already)
        located = _locate(sheet, turned, ranking, scale, scales=scales_of(size), looked_in=coarse)
        rough.append((located[0] if located is not None else -1.0, turns, size, turned))
    rough.sort(key=lambda item: -item[0])
    best = (upright[0], upright[1]) if upright is not None else None
    for _score, turns, size, turned in rough[:_COARSE_KEEP]:
        located = _locate(sheet, turned, config, scale, scales=scales_of(size), looked_in=looked_in)
        if located is not None and (best is None or located[0] > best[0]):
            best = (located[0], located[1])
    return best


_COARSE_SHARE = 0.5  # (the rough search's copy, as a share of the close search's)
_COARSE_KEEP = 3  # (how many of the likeliest ways round and sizes are looked for closely)
_SURE_PRINT_MATCH = 0.6  # (edges correlating this well: the piece is surely found)


def located_by_print(
    original: np.ndarray, suspect: np.ndarray, config: SeriesConfig, scale: float | None = None
) -> AutoArea | None:
    """The artwork of two files that have no border to find it by, and do not register as whole pages -- a proof sheet
    holding the artwork beside another and a job sheet, against a scan of the one artwork on a scanner's glass: the
    largest piece of print on each page is looked for on the other (by its edges, at the scale the two files' resolutions
    give), and whichever is found more surely is the artwork on both. None if neither is found. `scale`: suspect pixels
    per original pixel."""
    found = []
    block = print_block(suspect)
    located = _located_any_way(original, suspect[block[1] : block[3], block[0] : block[2]], config, scale)
    if located is not None:
        found.append((located[0], located[1], block))  # (score, the original's box, the suspect's)
    block = print_block(original)
    located = _located_any_way(suspect, original[block[1] : block[3], block[0] : block[2]], config, 1.0 / scale if scale else None)
    if located is not None:
        found.append((located[0], block, located[1]))
    if not found:
        return None
    _score, original_box, suspect_box = max(found, key=lambda item: item[0])
    logger.info("artwork found by its print", extra={"score": round(_score, 2)})
    return AutoArea(
        _rect(original_box, original.shape[:2]),
        _rect(suspect_box, suspect.shape[:2]),
        "The two files did not line up as whole pages, so the artwork was found by what is printed: the largest piece of "
        "print on one was found on the other, and only it was compared.",
    )
