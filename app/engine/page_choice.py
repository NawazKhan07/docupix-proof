"""Which page of each file to compare, when a file has more than one (two scanned sheets in one PDF, say).

The reviewer can say which (`ComparisonRequest.original_page` / `suspect_page`). Where they have not, the page is the one
the other file's artwork is actually ON: each candidate page is matched against the other file's page the way the files
are aligned -- by the features they share, counting only the matches that agree on one placement -- and the page with
the most is taken. Two cartons of one family scanned on two sheets share a layout and colours in places, but only the
right sheet holds the same artwork, feature for feature; a whole-page likeness cannot tell them apart when the sheets
are framed differently (a scan on a board beside a digital sheet with its specification table). Where no page holds
enough of it, the page most like the other's as a whole (`leaflet._thumbnail`) is taken instead.
"""

from __future__ import annotations

import logging

import numpy as np

from app.config import AlignmentConfig
from app.engine.alignment import homography
from app.engine.alignment.feature_matcher import match_features
from app.engine.file_loader import LoadedArtwork
from app.engine.leaflet import _page_similarity, _thumbnail

logger = logging.getLogger(__name__)

_MATCH_DPI = 100.0  # pages are matched this coarsely: enough to hold a carton's features, quick to match
_THUMB_DPI = 40.0
_MIN_INLIERS = 12  # a page sharing fewer agreeing features than this with the other file is not shown to hold its artwork
_MAX_PAGES = 60  # a file with more pages than this is not searched: its first page is taken unless one is chosen


def _picture(artwork: LoadedArtwork, page: int, dpi: float) -> np.ndarray:
    return artwork.render(dpi=dpi if artwork.kind == "pdf" else None, page=page)


def _shared_features(a: np.ndarray, b: np.ndarray, config: AlignmentConfig) -> int:
    """How many features two pictures share that agree on one placement of the one on the other."""
    try:
        matches = match_features(a, b, config)
        estimate = homography.estimate_homography(matches, config)
    except Exception:  # noqa: BLE001 -- a pair that cannot be matched shares nothing
        return 0
    return 0 if estimate is None else int(estimate[1].sum())


def choose_pages(
    original: LoadedArtwork,
    suspect: LoadedArtwork,
    original_page: int | None,
    suspect_page: int | None,
    config: AlignmentConfig | None = None,
) -> tuple[int, int, bool, bool]:
    """(original page, suspect page, whether the original's was chosen here, whether the suspect's was) -- 0-based; a
    page the reviewer chose is kept (if the file has it)."""
    config = config or AlignmentConfig()

    def fixed(artwork: LoadedArtwork, page: int | None) -> int | None:
        if page is not None and 0 <= page < artwork.page_count:
            return page
        if artwork.page_count <= 1 or artwork.page_count > _MAX_PAGES:
            return 0
        return None

    first, second = fixed(original, original_page), fixed(suspect, suspect_page)
    if first is not None and second is not None:
        return first, second, False, False
    original_pages = [first] if first is not None else list(range(original.page_count))
    suspect_pages = [second] if second is not None else list(range(suspect.page_count))
    try:
        pictures_o = {page: _picture(original, page, _MATCH_DPI) for page in original_pages}
        pictures_s = {page: _picture(suspect, page, _MATCH_DPI) for page in suspect_pages}
        scores = {(i, j): _shared_features(pictures_o[i], pictures_s[j], config) for i in original_pages for j in suspect_pages}
        best = max(scores, key=scores.get)
        if scores[best] < _MIN_INLIERS:
            thumbs_o = {page: _thumbnail(_picture(original, page, _THUMB_DPI), 0) for page in original_pages}
            thumbs_s = {page: _thumbnail(_picture(suspect, page, _THUMB_DPI), 0) for page in suspect_pages}
            best = max(scores, key=lambda pair: _page_similarity(thumbs_o[pair[0]], thumbs_s[pair[1]]))
    except Exception:  # noqa: BLE001 -- pages that cannot be pictured cannot be matched: the first is taken
        logger.exception("could not match the pages to choose one")
        return first or 0, second or 0, first is None, second is None
    logger.info("pages chosen", extra={"original_page": best[0] + 1, "suspect_page": best[1] + 1, "shared_features": scores[best]})
    return best[0], best[1], first is None, second is None
