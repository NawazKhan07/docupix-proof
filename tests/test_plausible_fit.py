"""The whole sheet is fitted only as a flat print can be bent (`homography.estimate_homography(whole_sheet=True)`): a
touch of perspective (a scanner's) is kept as it is; a fit that folds the sheet (a re-laid proof's compromise, a few
matches on a label) is not -- the matches it agreed on are fitted again as an affine or a turn-and-scale."""

from __future__ import annotations

import numpy as np

from app.config import AlignmentConfig
from app.engine.alignment.feature_matcher import FeatureMatches
from app.engine.alignment.homography import apply_transform, estimate_homography
from app.engine.alignment.quality_gate import distortion


def _matches(transform: np.ndarray, count: int = 400, seed: int = 5) -> FeatureMatches:
    rng = np.random.default_rng(seed)
    src = rng.uniform([0, 0], [3000, 2000], (count, 2))
    dst = apply_transform(transform, src) + rng.normal(0, 0.3, (count, 2))
    return FeatureMatches(src.astype(np.float32), dst.astype(np.float32))


def _same_placement(fitted: np.ndarray, expected: np.ndarray) -> bool:
    grid = np.array([[x, y] for x in (0, 1500, 3000) for y in (0, 1000, 2000)], float)
    return float(np.abs(apply_transform(fitted, grid) - apply_transform(expected, grid)).max()) < 0.5


def test_a_scanners_touch_of_perspective_is_kept():
    scan = np.array([[1.002, 0.004, 12.0], [-0.003, 0.998, -7.0], [2e-6, -1e-6, 1.0]])
    transform, inliers = estimate_homography(_matches(scan), AlignmentConfig(), whole_sheet=True)
    assert inliers.sum() > 350
    assert _same_placement(transform, scan)


def test_a_fit_that_folds_the_sheet_is_never_returned():
    folded = np.array([[0.58, -0.47, 2540.0], [0.014, 0.265, 395.0], [1e-5, -1.5e-4, 1.0]])
    matches = _matches(folded)
    estimate = estimate_homography(matches, AlignmentConfig(), whole_sheet=True)
    if estimate is not None:
        transform, inliers = estimate
        spread, _stretch, shear = distortion(transform, matches.src_points[inliers])
        assert spread <= 1.15 and shear <= 0.08


def test_a_stretch_is_allowed():
    stretched = np.array([[1.0, 0.0, 30.0], [0.0, 1.15, -40.0], [0.0, 0.0, 1.0]])  # a carton of other proportions
    transform, inliers = estimate_homography(_matches(stretched), AlignmentConfig(), whole_sheet=True)
    assert inliers.sum() > 350
    assert _same_placement(transform, stretched)



def test_a_fit_mild_where_it_matched_but_folding_the_sheet_is_refused():
    """Matches in the middle of a big sheet can agree on a perspective that looks mild there and folds the sheet's far
    edges (a re-laid proof's compromise): the fit is judged over all it is applied to -- the compared area -- not only
    over its matches."""
    folding = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [6e-5, 2e-5, 1.0]])
    rng = np.random.default_rng(9)
    src = rng.uniform([2600, 1800], [3400, 2200], (300, 2))  # matches only in the middle
    dst = apply_transform(folding, src) + rng.normal(0, 0.3, (300, 2))
    matches = FeatureMatches(src.astype(np.float32), dst.astype(np.float32))
    sheet = np.array([[0, 0], [6000, 0], [0, 4000], [6000, 4000]], float)
    spread_middle, _stretch, _shear = distortion(folding, src)
    spread_sheet, _stretch, _shear = distortion(folding, sheet)
    assert spread_middle < 1.15 < spread_sheet  # (the case: mild in the middle, folding at the edges)
    transform, _inliers = estimate_homography(matches, AlignmentConfig(), whole_sheet=True, extent=sheet)
    spread, _stretch, shear = distortion(transform, sheet)
    assert spread <= 1.15 and shear <= 0.08


def test_blocks_that_share_a_side_are_neighbours():
    from app.engine.alignment.block_alignment import _bordering
    from app.models.diff import BoundingBox

    panel = BoundingBox(1000, 500, 3000, 4000)
    assert _bordering(panel, BoundingBox(1000, 240, 3000, 499))  # a strip above it
    assert _bordering(panel, BoundingBox(3001, 500, 4000, 4000))  # the panel beside it
    assert not _bordering(panel, BoundingBox(5000, 500, 6000, 4000))  # one panel further on
    assert not _bordering(panel, BoundingBox(3100, 100, 4000, 400))  # diagonally off its corner
