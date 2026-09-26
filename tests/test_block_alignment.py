"""Block-wise local alignment: each interior panel of an already-lifted artwork aligned on its own, the way
manually cropping and aligning just that panel would (`app.engine.alignment.block_alignment`)."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from app.config import AlignmentConfig, AppConfig
from app.engine.alignment.aligner import align
import app.engine.alignment.block_alignment as ba
from app.engine.alignment.block_alignment import (
    ArtworkBlockAlignment,
    align_blocks,
    composite_warp,
    seam_margin_mask,
)
from app.engine.alignment.feature_matcher import match_features
from app.engine.alignment.homography import warp_to_reference
from app.engine.isolation.polygon_tracer import ArtworkBlock, find_blocks
from app.engine.worker import run_comparison
from app.models.diff import BoundingBox
from app.models.isolation import IsolationGeometry, NormalizedRect
from app.models.job import ComparisonMode

CONFIG = AlignmentConfig()


def _three_panel_carton(width: int = 900, height: int = 700) -> np.ndarray:
    """An outer boundary with two interior fold lines, each panel with enough of its own text/shape detail
    for ORB to find keypoints on -- the same shape of scene `test_polygon_tracer.py` finds blocks in."""
    image = np.full((height, width, 3), 255, np.uint8)
    cv2.rectangle(image, (20, 20), (880, 680), (0, 0, 0), 3)
    for x in (300, 600):
        cv2.line(image, (x, 20), (x, 680), (0, 0, 0), 3)
    cv2.putText(image, "FRONT PANEL: LOREM IPSUM DOLOR", (35, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 2)
    cv2.putText(image, "SIT AMET CONSECTETUR ELIT", (35, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 2)
    cv2.putText(image, "STORE BELOW 25C DRY PLACE", (35, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 2)
    cv2.putText(image, "KEEP OUT OF REACH CHILDREN", (35, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 2)
    cv2.circle(image, (150, 500), 30, (200, 30, 40), -1)

    cv2.putText(image, "SIDE PANEL BATCH INFO", (315, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 2)
    cv2.putText(image, "EXP DATE MFG LICENCE NO", (315, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 2)
    cv2.circle(image, (450, 500), 20, (30, 160, 60), -1)

    cv2.putText(image, "BACK PANEL: MANUFACTURED BY", (615, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 2)
    cv2.putText(image, "COMPANY NAME PVT LTD INDIA", (615, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 2)
    cv2.putText(image, "DIRECTIONS FOR USE APPLY", (615, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 2)
    cv2.putText(image, "TWICE DAILY AFTER MEALS", (615, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 2)
    cv2.rectangle(image, (700, 480), (780, 550), (30, 90, 200), -1)
    return image


def _locally_misaligned_suspect(original: np.ndarray) -> np.ndarray:
    """A whole-page shift (an ordinary scan) plus an EXTRA local rotation+shift of just the back panel (a
    fold that was not quite flat when it was photographed or scanned). The panel moves with its own edge lines,
    as a real out-of-plane panel does, not just the content inside them."""
    suspect = original.copy()
    back = suspect[:, 601:].copy()
    matrix = cv2.getRotationMatrix2D((139, 350), 1.5, 1.0)
    matrix[:, 2] += (3, 2)
    suspect[:, 601:] = cv2.warpAffine(back, matrix, (299, 700), borderValue=(255, 255, 255))
    whole_page = np.float32([[1, 0, 6], [0, 1, 4]])
    return cv2.warpAffine(suspect, whole_page, (900, 700), borderValue=(255, 255, 255))


def test_a_locally_misaligned_panel_is_found_and_locally_corrected():
    original = _three_panel_carton()
    suspect = _locally_misaligned_suspect(original)
    global_alignment = align(original, suspect, CONFIG)
    blocks = find_blocks(original)
    assert len(blocks) == 3

    alignments = align_blocks(original, suspect, blocks, global_alignment.transform, CONFIG)
    back_panel = next(a for a in alignments if a.block.bbox.x0 > 500)
    assert back_panel.locally_aligned
    assert not np.allclose(back_panel.transform, global_alignment.transform, atol=0.02)


def test_a_panel_with_no_local_misalignment_still_warps_to_the_right_place():
    """The front and side panels were never given any extra local drift, so whichever transform each settles
    on -- its own independent fit, or the fallback -- must still put their content where it belongs: a
    panel's own homography, fit from a smaller and different set of keypoints than the whole artwork's, is
    not expected to match the global transform's coefficients, only what it does to that panel's own pixels."""
    original = _three_panel_carton()
    suspect = _locally_misaligned_suspect(original)
    global_alignment = align(original, suspect, CONFIG)
    blocks = find_blocks(original)

    alignments = align_blocks(original, suspect, blocks, global_alignment.transform, CONFIG)
    reference_shape = original.shape[:2]
    composite = composite_warp(suspect, alignments, global_alignment.transform, reference_shape)

    # the front panel's own line of text, well inside its block and away from any seam
    region = np.s_[185:215, 40:400]
    diff = np.abs(composite[region].astype(int) - original[region].astype(int)).mean()
    assert diff < 20  # close to a match; a wrongly-placed panel would be off by far more than this


def test_a_panel_containing_the_real_difference_does_not_get_a_wrong_transform():
    """Regression: a block with only a few weak keypoints near the actual change must not accept a
    plausible-looking but wrong local fit just because it clears the quality gate (it must beat the global
    transform's own residual on the very same points, not merely pass the gate on its own)."""
    original = _three_panel_carton()
    suspect = original.copy()
    cv2.rectangle(suspect, (700, 480), (780, 550), (200, 160, 30), -1)  # a real colour change, no misalignment
    global_alignment = align(original, suspect, CONFIG)
    blocks = find_blocks(original)

    alignments = align_blocks(original, suspect, blocks, global_alignment.transform, CONFIG)
    back_panel = next(a for a in alignments if a.block.bbox.x0 > 500)
    assert np.allclose(back_panel.transform, global_alignment.transform, atol=0.02)


def test_composite_warp_uses_each_locally_aligned_blocks_own_correction():
    original = _three_panel_carton()
    suspect = _locally_misaligned_suspect(original)
    global_alignment = align(original, suspect, CONFIG)
    blocks = find_blocks(original)
    alignments = align_blocks(original, suspect, blocks, global_alignment.transform, CONFIG)

    reference_shape = original.shape[:2]
    composite = composite_warp(suspect, alignments, global_alignment.transform, reference_shape)
    global_only = warp_to_reference(suspect, global_alignment.transform, reference_shape)

    # Well inside the back panel (away from any seam), on real content: the composite should match the
    # original much more closely than the single global warp, which never corrected this panel's own drift.
    region = np.s_[185:215, 615:860]
    composite_diff = np.abs(composite[region].astype(int) - original[region].astype(int)).mean()
    global_diff = np.abs(global_only[region].astype(int) - original[region].astype(int)).mean()
    assert composite_diff < global_diff


def test_composite_warp_falls_back_to_the_global_warp_with_no_blocks():
    original = _three_panel_carton()
    suspect = _locally_misaligned_suspect(original)
    global_alignment = align(original, suspect, CONFIG)
    reference_shape = original.shape[:2]

    composite = composite_warp(suspect, [], global_alignment.transform, reference_shape)
    global_only = warp_to_reference(suspect, global_alignment.transform, reference_shape)
    assert np.array_equal(composite, global_only)


def _flat_panel_carton(width: int = 700, height: int = 500) -> np.ndarray:
    """One panel with real, textured content (enough ORB keypoints for the whole-artwork alignment) beside a
    second, flat solid-colour panel with none of its own -- a blurred, low-contrast mark, not a sharp corner
    ORB/AKAZE could grab onto, the way a plain background colour panel is printed in practice."""
    image = np.full((height, width, 3), 255, np.uint8)
    cv2.putText(image, "PARACETAMOL 500mg", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
    cv2.putText(image, "Tablets BP x16", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)
    for i in range(10):
        cv2.line(image, (30 + i * 15, 130), (30 + i * 15, 170), (0, 0, 0), 2)

    x0, y0, x1, y1 = 400, 200, 650, 400
    cv2.rectangle(image, (x0, y0), (x1, y1), (30, 90, 180), -1)
    cv2.putText(image, "AMPHOTRET", (x0 + 30, y0 + 110), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (26, 80, 160), 14, cv2.LINE_AA)
    image[y0:y1, x0:x1] = cv2.GaussianBlur(image[y0:y1, x0:x1], (0, 0), 3.0)
    return image


_FLAT_PANEL_BOX = (400, 200, 650, 400)


def _flat_panel_shifted(original: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """`original` with only the flat panel's own content shifted -- a fold not quite flat when this one
    panel was photographed or scanned, same as `_locally_misaligned_suspect` above but on a texture-poor
    panel, and with no whole-page shift so the global transform is the identity."""
    x0, y0, x1, y1 = _FLAT_PANEL_BOX
    suspect = original.copy()
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    suspect[y0:y1, x0:x1] = cv2.warpAffine(original[y0:y1, x0:x1], matrix, (x1 - x0, y1 - y0), borderMode=cv2.BORDER_REPLICATE)
    return suspect


def test_a_flat_panel_with_no_orb_keypoints_is_still_locally_corrected():
    """The phase-correlation fallback: a panel with too little texture for ORB/AKAZE to find enough of its
    own keypoints on (a solid background colour) is not simply left on the global transform -- a shift read
    from the window's bulk content, rather than distinct corners, still finds and corrects a real local drift."""
    original = _flat_panel_carton()
    x0, y0, x1, y1 = _FLAT_PANEL_BOX
    suspect = _flat_panel_shifted(original, dx=6.0, dy=-4.0)
    global_transform = np.eye(3)

    matches = match_features(
        suspect, original, CONFIG,
        _mask(original.shape[:2], _FLAT_PANEL_BOX), _mask(original.shape[:2], _FLAT_PANEL_BOX),
    )
    assert len(matches) < CONFIG.block_min_match_count  # confirms this panel needs the fallback, not a lucky fit

    blocks = [ArtworkBlock(polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)], bbox=BoundingBox(x0, y0, x1, y1))]
    alignments = align_blocks(original, suspect, blocks, global_transform, CONFIG)
    assert alignments[0].locally_aligned

    composite = composite_warp(suspect, alignments, global_transform, original.shape[:2])
    seam = seam_margin_mask(original.shape[:2], alignments, margin_px=16)
    interior = np.zeros(original.shape[:2], bool)
    interior[y0:y1, x0:x1] = True
    compared = interior & ~seam  # what the pipeline actually looks at: the seam ring is excluded either way

    before = np.abs(original[compared].astype(int) - suspect[compared].astype(int)).mean()
    after = np.abs(original[compared].astype(int) - composite[compared].astype(int)).mean()
    assert after < before / 10  # corrected to close to a match, not merely somewhat better


def _mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_seam_margin_mask_only_rings_locally_aligned_blocks():
    shape = (200, 200)
    aligned = ArtworkBlockAlignment(
        block=ArtworkBlock(polygon=[(20, 20), (100, 20), (100, 100), (20, 100)], bbox=BoundingBox(20, 20, 100, 100)),
        transform=np.eye(3),
        locally_aligned=True,
    )
    not_aligned = ArtworkBlockAlignment(
        block=ArtworkBlock(polygon=[(120, 20), (180, 20), (180, 100), (120, 100)], bbox=BoundingBox(120, 20, 180, 100)),
        transform=np.eye(3),
        locally_aligned=False,
    )
    mask = seam_margin_mask(shape, [aligned, not_aligned], margin_px=5)
    assert mask[60, 20]  # right at the aligned block's own edge
    assert not mask[60, 60]  # well inside it, away from the edge
    assert not mask[60, 10]  # well outside it too
    assert not mask[60, 150]  # inside the block that fell back to global: no seam to hide there


# ---------------------------------------------------------------------------
# End to end, through run_comparison (Automatic mode: every mode shares this pipeline)
# ---------------------------------------------------------------------------


# Panels exist only inside an artwork outline (a whole page also holds its title, notes and blank paper): these compare
# the carton as a reviewer, or the automatic detection, would -- trimmed to it.
_CARTON = IsolationGeometry(boundary=NormalizedRect(0.01, 0.01, 0.99, 0.99))


def _compare(original: np.ndarray, suspect: np.ndarray, config: AppConfig):
    return run_comparison(
        original, suspect, ComparisonMode.AUTOMATIC, config, original_geometry=_CARTON, suspect_geometry=_CARTON
    )


def test_blocks_are_only_found_inside_an_artwork_area():
    """With no area found, cropped or lifted, the whole page is compared -- and a whole page is not divided into
    panels: a carton's fold lines drawn across it would cut the title, notes and blank paper into 'panels' too."""
    original = _three_panel_carton()
    outcome = run_comparison(original, _locally_misaligned_suspect(original), ComparisonMode.AUTOMATIC, AppConfig())
    assert outcome.block_outlines == [] and outcome.block_alignments == []


def test_block_alignment_removes_false_positives_from_a_non_coplanar_panel():
    """The whole point of the feature: a panel photographed or scanned slightly out of plane with the rest of
    the artwork, which one global transform cannot also satisfy, must not read as a wall of differences."""
    original = _three_panel_carton()
    suspect = _locally_misaligned_suspect(original)

    with_blocks = _compare(original, suspect, AppConfig())
    config_without = replace(AppConfig(), detection=replace(AppConfig().detection, block_alignment_enabled=False))
    without_blocks = _compare(original, suspect, config_without)

    assert len(with_blocks.block_outlines) == 3
    assert len(without_blocks.block_outlines) == 0
    assert len(with_blocks.clusters) < len(without_blocks.clusters)
    assert len(with_blocks.clusters) == 0


def test_block_alignment_still_finds_a_real_difference_with_a_tight_box():
    """Block-wise alignment must not blur or enlarge a genuine finding -- here, a panel with no local
    misalignment at all, just a real colour change, which should be reported exactly as before."""
    original = _three_panel_carton()
    suspect = original.copy()
    cv2.rectangle(suspect, (700, 480), (780, 550), (200, 160, 30), -1)

    outcome = _compare(original, suspect, AppConfig())
    assert len(outcome.clusters) == 1
    box = outcome.clusters[0].bbox_in_original
    assert box.x0 == pytest.approx(700, abs=3) and box.x1 == pytest.approx(780, abs=3)
    assert box.y0 == pytest.approx(480, abs=3) and box.y1 == pytest.approx(550, abs=3)


def test_block_alignment_can_be_switched_off():
    original = _three_panel_carton()
    suspect = _locally_misaligned_suspect(original)
    config = replace(AppConfig(), detection=replace(AppConfig().detection, block_alignment_enabled=False))

    outcome = _compare(original, suspect, config)
    assert outcome.block_outlines == []


def test_reflowed_lines_inside_a_same_size_panel_are_reported_not_aligned_away():
    """A panel the same size in both files whose lower lines moved down (a line of text added above them) has
    not changed shape: only its text changed, and that is a difference to report. Block-wise alignment may split
    a block into parts only where the counterpart panel itself grew or shrank enough to account for the move."""
    original = _three_panel_carton()
    suspect = original.copy()
    suspect[300:400, 605:875] = 255
    suspect[312:412, 605:875] = original[300:400, 605:875]  # the back panel's last two lines, 12 px lower

    outcome = _compare(original, suspect, AppConfig())
    moved = [c for c in outcome.clusters if c.bbox_in_original.x0 >= 600 and 290 <= c.bbox_in_original.y0 <= 420]
    assert moved


# ---------------------------------------------------------------------------
# Line-only blocks (flaps): aligned on where their lines land
# ---------------------------------------------------------------------------


def _flap_page(width_mm_scale: float = 1.0) -> np.ndarray:
    """A printed panel beside a flap that is nothing but a pale die line (a trapezoid, and a crease inside it) --
    `width_mm_scale` stretches the flap sideways, as a carton of another size would have it."""
    image = np.full((700, 1000, 3), 255, np.uint8)
    cv2.rectangle(image, (400, 100), (950, 600), (0, 0, 0), 2)
    for row, text in enumerate(("SUGAMMADEX 100 mg/mL", "soluzione iniettabile", "10 flaconcini", "Per uso endovenoso")):
        cv2.putText(image, text, (430, 180 + row * 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    right = 400
    left = int(right - 300 * width_mm_scale)
    flap = np.array([[left, 160], [right, 100], [right, 600], [left, 540]], np.int32)
    cv2.polylines(image, [flap], True, (120, 120, 255), 2)  # a pale red die line: nearly white in grey
    cv2.line(image, (left + 40, 350), (right - 30, 350), (255, 200, 120), 2)  # a pale blue crease
    return image


def test_a_flap_with_nothing_but_its_die_line_is_aligned_onto_the_line():
    """A flap has no printing to match keypoints on: its counterpart's outline is stretched onto it and refined until
    the die lines coincide -- judged by where the edges land, since a mostly blank block's pixel difference cannot
    tell a line 20 px off from one on target."""
    original = _flap_page(1.0)
    suspect = _flap_page(1.12)  # the other carton's flap is wider
    panel = ArtworkBlock(polygon=[(400, 100), (950, 100), (950, 600), (400, 600)], bbox=BoundingBox(400, 100, 950, 600))
    flap = ArtworkBlock(polygon=[(100, 100), (400, 100), (400, 600), (100, 600)], bbox=BoundingBox(100, 100, 400, 600))
    wide_flap = ArtworkBlock(polygon=[(64, 100), (400, 100), (400, 600), (64, 600)], bbox=BoundingBox(64, 100, 400, 600))

    alignments = align_blocks(original, suspect, [panel, flap], np.eye(3), CONFIG, [panel, wide_flap])
    flap_alignment = next(a for a in alignments if a.block is flap)
    assert flap_alignment.locally_aligned

    edges = ba._EdgeScorer(ba._darkest(original), ba._darkest(suspect), flap)
    assert edges(flap_alignment.transform) < 2.0 < edges(np.eye(3))


def test_a_fit_locked_on_one_logo_that_leaves_the_rest_off_is_not_taken():
    """Keypoints can agree on one bold mark and pull the rest of a block off it; a pixel difference swayed by the paper
    round it can even look better. Where the block's edges land clearly worse than with the whole-sheet fit, the fit
    is refused."""
    original = _three_panel_carton()
    suspect = original.copy()
    block = ArtworkBlock(polygon=[(20, 20), (300, 20), (300, 680), (20, 680)], bbox=BoundingBox(20, 20, 300, 680))
    edges = ba._EdgeScorer(ba._darkest(original), ba._darkest(suspect), block, interior=True)
    off = np.array([[1.0, 0.0, 9.0], [0.0, 1.0, -7.0], [0.0, 0.0, 1.0]])
    assert edges(off) > edges(np.eye(3)) + ba._EDGE_MIN_GAIN_PX  # what the veto sees
