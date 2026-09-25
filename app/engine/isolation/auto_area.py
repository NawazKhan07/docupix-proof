"""Finding the artwork on a sheet without being told where it is (Automatic mode).

A packaging file is a sheet: the carton's die-cut outline with the printed panels inside it and, around it, a title,
a colour legend, crop marks, notes and often a table of details that is closed by a line too. Without a crop or a lift,
Automatic mode finds the artwork itself: the closed border on each file that CORRESPONDS to one on the other (where a
sheet has several, the pair whose printing matches best, whatever their sizes on their sheets), trimmed to that, and the
two trimmed files are aligned and compared. Where a file has no border to find (a flat label, a photograph), the whole
page is compared, as before.

Both files or neither: if the borders found do not correspond (they do not share enough printing, or one is far larger or
differently shaped than the other), one of them is a panel or a box and trimming to it would compare the wrong things.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import SubjectConfig
from app.engine.isolation.border_lifter import border_regions
from app.engine.isolation.subject_lifter import LiftedSubject, outline_from_mask
from app.models.isolation import IsolationGeometry

logger = logging.getLogger(__name__)

_CANDIDATES = 3  # the biggest closed borders of each file that are tried against each other
_MIN_CORNERS = 6  # a die-cut outline has corners and notches (a carton: a dozen or more; a small foil: seven); a box is four...
_BOX_INK_SHARE = 0.6  # ...and a box is the artwork only if it holds this much of everything printed on the page (a label: 0.65-0.8; a box round a logo: 0.15-0.4)
_MATCH_LONG_EDGE_PX = 1200  # printing is matched on copies this size
_MATCH_FEATURES = 2500
_MIN_INLIERS = 25  # a pair of borders that share fewer consistent features than this do not correspond
_MAX_AREA_RATIO = 1.8  # where the choice is not made by matching: the shares of their pages may differ by this...
_MAX_ASPECT_RATIO = 1.5  # ...and their shapes (width to height) by this


@dataclass
class AutoArea:
    original: IsolationGeometry | None
    suspect: IsolationGeometry | None
    note: str = ""  # what was done, in words (empty if nothing was)


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


def _features(image: np.ndarray, subject: LiftedSubject):
    """Keypoints and descriptors of the printing inside a border, on a reduced copy."""
    height, width = image.shape[:2]
    scale = min(1.0, _MATCH_LONG_EDGE_PX / max(height, width))
    small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = cv2.resize(subject.mask.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST) * 255
    detector = cv2.ORB_create(nfeatures=_MATCH_FEATURES)
    keypoints, descriptors = detector.detectAndCompute(small, mask)
    return keypoints, descriptors


def _corresponding(first, second) -> int:
    """How many consistent features two borders' printing shares (0 if none can be told)."""
    (kp_a, desc_a), (kp_b, desc_b) = first, second
    if desc_a is None or desc_b is None or len(kp_a) < 8 or len(kp_b) < 8:
        return 0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_b, desc_a, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    if len(good) < 8:
        return 0
    source = np.float32([kp_b[m.queryIdx].pt for m in good])
    target = np.float32([kp_a[m.trainIdx].pt for m in good])
    transform, inliers = cv2.findHomography(source, target, cv2.RANSAC, 6.0)
    return int(inliers.sum()) if transform is not None and inliers is not None else 0


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
    firsts = _subjects(original, config, original_page, original_dpi)
    seconds = _subjects(suspect, config, suspect_page, suspect_dpi)
    if not firsts or not seconds:
        return AutoArea(None, None, "")

    if len(firsts) == 1 and len(seconds) == 1:
        first, second = firsts[0], seconds[0]
        ratio = max(first.area_fraction, second.area_fraction) / max(1e-9, min(first.area_fraction, second.area_fraction))
        shapes = max(_aspect(first), _aspect(second)) / max(1e-9, min(_aspect(first), _aspect(second)))
        if ratio > _MAX_AREA_RATIO or shapes > _MAX_ASPECT_RATIO:
            logger.info("automatic artwork areas disagree; the whole pages are compared", extra={"area_ratio": round(ratio, 2), "shape_ratio": round(shapes, 2)})
            return AutoArea(None, None, "")
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
            logger.info("no pair of borders corresponds; the whole pages are compared", extra={"best_inliers": best_score})
            return AutoArea(None, None, "")
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
