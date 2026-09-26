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

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import AlignmentConfig, SubjectConfig
from app.engine.alignment.aligner import AlignmentError
from app.engine.isolation.border_lifter import border_regions
from app.engine.isolation.cutout import Cutout
from app.engine.isolation.placement import fit_placement
from app.engine.isolation.subject_lifter import LiftedSubject, outline_from_mask
from app.models.isolation import IsolationGeometry, NormalizedPolygon

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
