"""Comparison pipeline: isolate -> align -> detect (Architecture §3.3).

This is the function Phase 7 runs inside a worker process. It owns the two things
individual detectors must not each decide for themselves:

* what is compared. Each file is first cleared to plain white everywhere outside its own
  artwork area (a lifted or placed outline, a crop, minus anything erased), so the two files
  that are aligned and compared are the two artworks and nothing else; then the single hard
  mask that keeps furniture out of every detector's view (FR-2.1) is built from both files'
  areas *and* the region the warped suspect actually covers;
* the working resolution detection runs at, with candidate boxes scaled back
  into the shared aligned space afterwards so everything downstream speaks
  one coordinate system (FR-7.4).
"""

from __future__ import annotations

import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from app.config import AppConfig, DetectionConfig
from app.engine.aggregation.coordinate_projection import original_to_suspect
from app.engine.alignment.aligner import AlignmentError, align
from app.engine.aggregation.aggregator import add_member, build_clusters
from app.engine.alignment.homography import warp_to_reference, warp_validity_mask
from app.engine.detectors import (
    barcode_diff,
    braille_diff,
    color_diff,
    graphics_diff,
    local_diff,
    movement,
    structure_diff,
    text_diff,
    verify,
)
from app.engine.detectors.change_regions import ChangeRegion, find_change_regions
from app.engine.detectors.regions import padded_window
from app.engine.detectors.text_diff import OcrUnavailableError
from app.engine.isolation.boundary_mask import apply_mask, boundary_mask, erased_mask
from app.engine.preprocessing.scan_normalizer import ink_image, reading_source
from app.logging_config import StageTimer
from app.models.diff import BoundingBox, DiffCandidate, DiffCluster, DiffSubtype, DiffType
from app.models.isolation import IsolationGeometry
from app.models.job import ComparisonMode, options_for_mode
from app.models.result import AlignmentResult, BrailleComparison, SymbolComparison

logger = logging.getLogger(__name__)


@dataclass
class ComparisonOutcome:
    alignment: AlignmentResult
    candidates: list[DiffCandidate]
    # The findings shown to the reviewer: `candidates` clustered, rated and
    # bound to both files' coordinates.
    clusters: list[DiffCluster]
    # The one aligned pair every viewer mode is drawn from (FR-7.4): the
    # original as-is and the suspect warped into the original's frame. Kept
    # unmasked so the reviewer sees the whole page, not just what was compared.
    aligned_original: np.ndarray
    aligned_suspect: np.ndarray
    symbol_comparisons: list[SymbolComparison] = field(default_factory=list)
    braille: BrailleComparison | None = None
    # Things the reviewer must know were *not* checked, e.g. OCR unavailable.
    warnings: list[str] = field(default_factory=list)
    # "exact" when alignment was exact and every noise filter was off, else
    # "tolerant"; and whether detection ran only inside the changed regions.
    detection_profile: str = "tolerant"
    used_change_regions: bool = False
    # Which mode produced this, so the results view can show the panels that mode calls for.
    mode: ComparisonMode | None = None
    # The area that was compared, as outlines in the original's pixels (holes included, for odd-even
    # filling), or None when the whole page was. Lets the viewer show what was left out.
    compared_outline: list[list[tuple[float, float]]] | None = None
    # Scan vs Digital only: each file as ink on paper (grey images, 255 = background, 0 = ink), aligned and
    # limited to the compared area -- the pictures that were read for text and compared. Shown by default.
    ink_original: np.ndarray | None = None
    ink_suspect: np.ndarray | None = None
    # Whether graphics were compared (Scan vs Digital leaves them out unless asked: the results page has the toggle).
    graphics_compared: bool = True
    # What was done on the reviewer's behalf that they should know (the artwork was found automatically, a page was
    # compared at a lower resolution): information, not something that was left unchecked (that is a warning).
    notes: list[str] = field(default_factory=list)


def run_comparison(
    original_image: np.ndarray,
    suspect_image: np.ndarray,
    mode: ComparisonMode,
    config: AppConfig,
    original_geometry: IsolationGeometry | None = None,
    suspect_geometry: IsolationGeometry | None = None,
    progress: Callable[[str], None] | None = None,
) -> ComparisonOutcome:
    """Run one comparison. `progress`, if given, is told what stage the job is at."""
    options = options_for_mode(mode)
    say = progress or (lambda _text: None)

    # Erase, in each file, everything outside its own artwork area: what is aligned and compared
    # is then the two artworks alone, on white, not two sheets with a title and marks around them.
    original_area = _Area.of(original_image, original_geometry)
    suspect_area = _Area.of(suspect_image, suspect_geometry)
    artwork_original = original_area.blanked(original_image)
    artwork_suspect = suspect_area.blanked(suspect_image)

    say("Aligning the files\u2026")
    try:
        alignment = align(
            artwork_original, artwork_suspect, config.alignment, original_geometry, suspect_geometry
        )
    except AlignmentError:
        if original_area.is_whole_page and suspect_area.is_whole_page:
            raise
        # Trimmed to their artwork the files share too little to register (the outlines cut their features): the whole
        # pages are registered instead, and the comparison is still only of the artwork.
        alignment = align(original_image, suspect_image, config.alignment)
    detection_config, profile = effective_detection_config(config.detection, alignment)
    say("Comparing the artwork\u2026")

    reference_shape = original_image.shape[:2]
    with StageTimer(logger, "warp_to_shared_space"):
        warped_artwork = warp_to_reference(artwork_suspect, alignment.transform, reference_shape)
        # The viewer shows the whole suspect page; only the comparison uses the cleared copy.
        warped_suspect = (
            warped_artwork
            if artwork_suspect is suspect_image
            else warp_to_reference(suspect_image, alignment.transform, reference_shape)
        )
        compare_mask = _build_compare_mask(
            suspect_image.shape[:2], reference_shape, alignment.transform, original_area, suspect_area
        )

    # FR-2.1, enforced once here rather than trusted to each detector.
    masked_original = apply_mask(artwork_original, compare_mask)
    masked_suspect = apply_mask(warped_artwork, compare_mask)

    candidates: list[DiffCandidate] = []
    warnings: list[str] = []
    symbol_comparisons: list[SymbolComparison] = []
    braille_comparison: BrailleComparison | None = None

    # A scan changes every colour, so Scan vs Digital compares what is PRINTED, not what colour it is: each
    # file is turned into ink on paper (background colours removed to white, everything else black). Text is
    # read from those pictures, and they are what the reviewer is shown. Graphics and barcodes are compared
    # by where the ink is, from the files' own grey levels rather than from the thresholded pictures: the
    # softness of a scan (blur, dot gain, lost fine print) is what tells it apart from a real change, and a
    # threshold applied first throws that away (a harsh scan lost its small print and the whole paragraph
    # read as removed). Barcodes are not decoded here, only compared as the marks they are.
    ink_original = ink_suspect = None
    read_original = read_suspect = None  # what the words are read from, where that is not the ink picture
    other_original = other_suspect = None  # what a line that differs is read from as well
    if options.normalize_scan:
        say("Removing the background colour…")
        with StageTimer(logger, "ink_pictures"), ThreadPoolExecutor(max_workers=2) as pool:
            # (OpenCV and NumPy release the interpreter lock, so the two files are done side by side)
            pending = pool.submit(ink_image, masked_suspect, detection_config, compare_mask)
            ink_original = ink_image(masked_original, detection_config, compare_mask)
            ink_suspect = pending.result()
        if (detection_config.scan_reading_digital, detection_config.scan_reading_scan) != ("ink", "ink"):
            with StageTimer(logger, "reading_pictures"):
                read_original = (
                    ink_original
                    if detection_config.scan_reading_digital == "ink"
                    else reading_source(masked_original, detection_config, compare_mask, detection_config.scan_reading_digital)
                )
                read_suspect = (
                    ink_suspect
                    if detection_config.scan_reading_scan == "ink"
                    else reading_source(masked_suspect, detection_config, compare_mask, detection_config.scan_reading_scan)
                )
        if detection_config.scan_reading_alternate:
            with StageTimer(logger, "alternate_pictures"):
                other_original = reading_source(masked_original, detection_config, compare_mask, detection_config.scan_reading_alternate)
                other_suspect = reading_source(masked_suspect, detection_config, compare_mask, detection_config.scan_reading_alternate)

    used_regions = False
    # Graphics of files that do not register exactly are compared by where INK is, within
    # the measured registration error: the similarity map's noise filters, needed there,
    # erase thin real changes (a missing line of small print scored 0.07 IoU).
    misregistered = options.normalize_scan or profile != "exact"
    graphics_mode = detection_config.graphics_when_misregistered if misregistered else "ssim"
    # What a scan compares is TEXT: its graphics are only compared if asked for (`scan_compare_graphics`): a scan
    # changes every colour, blurs every edge and never registers exactly, and comparing the pictures gave dozens
    # of false 'graphics' differences (fold lines, the scanner's own edge, every dark block) round the few real ones.
    # Barcodes are compared as marks either way (below).
    graphics_wanted = options.compare_graphics and (not options.normalize_scan or detection_config.scan_compare_graphics)
    local_wanted = detection_config.local_enabled and options.compare_color and options.compare_graphics and not options.normalize_scan
    local_only = local_wanted and not detection_config.legacy_pixel_detectors  # the full-size local comparison is the pixel detector
    use_structure = graphics_wanted and graphics_mode in ("structure", "both") and not local_only
    use_ssim = graphics_wanted and graphics_mode in ("ssim", "both") and not options.normalize_scan and not local_only
    unverifiable_area: np.ndarray | None = None
    structure_slack = structure_diff.slack_for(alignment.residual_error_px, detection_config)
    if options.normalize_scan and not use_structure:
        # The scan's softness still tells the reviewer how much small print can be trusted.
        _matched_original, _matched_suspect, softness = structure_diff.match_softness(
            masked_original, masked_suspect, detection_config, compare_mask
        )
        if softness >= detection_config.structure_soft_warning_sigma_px:
            warnings.append(
                f"The scan is much softer than the original (about {softness:.1f} px of blur at 300 DPI), so the "
                "finest print cannot be verified. Rescan at 300 DPI or higher."
            )
    # Compare where INK is, at full resolution, tolerating the registration error we measured (FR-6.3 for scans:
    # colour is ignored there). It is independent of the colour, similarity and text work below (and mostly NumPy and
    # OpenCV, which release the interpreter lock), so it runs beside them and is collected when its result is needed.
    structure_pool = ThreadPoolExecutor(max_workers=2)
    structure_job = (
        structure_pool.submit(structure_diff.detect, masked_original, masked_suspect, detection_config, structure_slack, compare_mask)
        if use_structure
        else None
    )
    # The deep local comparison (full size, tile by tile) runs beside it: it is what finds a small real change that the
    # reduced-size detectors' tolerance hides, and what the findings' borders are fitted to.
    local_job = (
        structure_pool.submit(local_diff.detect, masked_original, masked_suspect, detection_config, compare_mask)
        if local_wanted
        else None
    )
    residual_mask: np.ndarray | None = None
    structure_collected = False

    def collect_structure() -> np.ndarray | None:
        """Add what the structural comparison found (once) and return where it could not compare (or None)."""
        nonlocal structure_collected, unverifiable_area
        if structure_job is None or structure_collected:
            return unverifiable_area
        structure_collected = True
        structure = structure_job.result()
        candidates.extend(structure.candidates)
        unverifiable_area = structure.unverifiable_mask
        if structure.matched_softness_px >= detection_config.structure_soft_warning_sigma_px:
            warnings.append(
                f"The scan is much softer than the original (about {structure.matched_softness_px:.1f} px of "
                "blur at 300 DPI), so the finest print cannot be verified. Rescan at 300 DPI or higher."
            )
        if structure.unverifiable_fraction >= detection_config.unverifiable_warning_fraction:
            warnings.append(
                f"About {structure.unverifiable_fraction:.0%} of the artwork has fine detail (small "
                "text) that one file is too soft to show, so those areas could not be compared. "
                "Rescan at a higher resolution to check them."
            )
        return unverifiable_area

    pixel_detectors = options.compare_color or options.compare_graphics
    regions = None
    if pixel_detectors and not options.normalize_scan and not local_only:
        # Scan-vs-Digital normalises both files first, which only makes sense
        # over the whole page, so it never takes the region path.
        regions = find_change_regions(
            masked_original,
            masked_suspect,
            detection_config,
            compare_mask,
            threshold=_region_threshold(detection_config, profile),
        )

    if regions is not None:
        used_regions = True
        candidates += _detect_in_regions(
            masked_original,
            masked_suspect,
            compare_mask,
            regions,
            replace(options, compare_graphics=use_ssim),
            detection_config,
        )

    if options.compare_color or options.compare_graphics or options.compare_text:
        needs_pixels = regions is None and pixel_detectors and not options.normalize_scan and not local_only
        if needs_pixels or options.compare_text:
            if options.normalize_scan:
                # Text is read from the ink pictures at close to 300 DPI, not the coarser detection copy.
                small_original, small_suspect, scale = _for_reading(ink_original, ink_suspect, detection_config)
                small_read = (
                    _for_reading(read_original, read_suspect, detection_config)[:2] if read_original is not None else None
                )
                small_other = (
                    _for_reading(other_original, other_suspect, detection_config)[:2] if other_original is not None else None
                )
                small_mask = None
            else:
                small_original, small_suspect, small_mask, scale = _to_working_resolution(
                    masked_original, masked_suspect, compare_mask, detection_config
                )

            working_candidates: list[DiffCandidate] = []
            if needs_pixels and options.compare_color:
                working_candidates += color_diff.detect(
                    small_original, small_suspect, detection_config, small_mask
                )
            if needs_pixels and use_ssim:
                working_candidates += graphics_diff.detect(
                    small_original, small_suspect, detection_config, small_mask
                )
            if options.compare_text and options.normalize_scan:
                try:
                    scan_text = text_diff.detect_scan(small_original, small_suspect, detection_config, small_read, small_other)
                    working_candidates += scan_text.candidates
                    if scan_text.unreadable_lines:
                        warnings.append(
                            f"{scan_text.unreadable_lines} line(s) of small print are in the scan but too soft to "
                            "read, so their text was not compared. Rescan at a higher resolution to check them."
                        )
                except OcrUnavailableError as exc:
                    logger.warning("text detection skipped: %s", exc)
                    warnings.append(str(exc))

            # (a reduced picture's width and height are whole numbers: each axis has its own scale, not quite `scale`)
            full_size = (ink_original if options.normalize_scan else masked_original).shape
            factor_x, factor_y = small_original.shape[1] / full_size[1], small_original.shape[0] / full_size[0]
            working_candidates = [_rescaled(candidate, 1 / factor_x, 1 / factor_y) for candidate in working_candidates]
            if options.normalize_scan:
                # A scan too soft to hold small print cannot be read there either: not a difference.
                working_candidates = _readable_only(working_candidates, collect_structure())
            candidates += working_candidates

    collect_structure()
    if local_job is not None:
        local = local_job.result()
        candidates.extend(local.candidates)
        residual_mask = local.residual
    structure_pool.shutdown()

    if options.normalize_scan:
        # A barcode is not decoded here (a scan of one rarely reads the same), but where the digital file has one the
        # scan's bars are compared with its as marks.
        say("Comparing the barcodes…")
        candidates += barcode_diff.compare_physical(masked_original, masked_suspect, detection_config, structure_slack)

    # Symbols and Braille dots run at full resolution: a small barcode or a
    # 1 mm Braille dot does not survive the detection downscale.
    if options.decode_symbols:
        say("Reading barcodes and QR codes\u2026")
        symbol_candidates, symbol_comparisons = barcode_diff.compare(
            masked_original, masked_suspect, detection_config
        )
        candidates += symbol_candidates
        if not symbol_comparisons:
            warnings.append(
                "No barcode, Pharmacode or QR code could be read in either file, so nothing "
                "was compared. If the artwork has one, it may be too small or low-contrast "
                "to decode."
            )

    if options.compare_braille_dots:
        braille_candidates, braille_comparison = braille_diff.compare(
            masked_original, masked_suspect, detection_config
        )
        candidates += braille_candidates
        if braille_comparison.original_dot_count == 0 and braille_comparison.suspect_dot_count == 0:
            warnings.append(
                "No Braille dots were detected in either file. The dot colour range is a "
                "configurable assumption pending sign-off, so check it matches this artwork."
            )

    if pixel_detectors and not options.normalize_scan:
        candidates, ignored_lines = _drop_die_lines(candidates, masked_original, masked_suspect, compare_mask, detection_config)
        if ignored_lines:
            logger.info("die lines ignored", extra={"lines": ignored_lines})
        # A difference in the pixels is a change only if the two files really differ in its box, not merely along
        # an edge one of them draws a little differently (see `verify`).
        candidates = verify.drop_residue(candidates, original_image, warped_suspect, detection_config)

    say("Grouping the differences\u2026")
    clusters = build_clusters(candidates, alignment.transform, config.clustering)
    if residual_mask is not None:
        _fit_borders(clusters, residual_mask, alignment.transform, halo=1 + detection_config.local_forgive_px)

    if options.compare_text and not options.normalize_scan:
        # What a difference says is READ from its own box in each file, never found by comparing two readings of a page.
        say("Reading the words in each difference\u2026")
        try:
            if not text_diff.ocr_available():
                raise OcrUnavailableError(text_diff._OCR_UNAVAILABLE)
            _describe_clusters(clusters, masked_original, masked_suspect, detection_config, warnings)
        except OcrUnavailableError as exc:
            logger.warning("text description skipped: %s", exc)
            warnings.append(str(exc))

    if pixel_detectors and not options.normalize_scan:
        clusters = _mark_moves(clusters, masked_original, masked_suspect, detection_config)

    logger.info(
        "comparison finished",
        extra={
            "mode": mode.value,
            "candidates": len(candidates),
            "clusters": len(clusters),
            "warnings": len(warnings),
            "detection_profile": profile,
            "used_change_regions": used_regions,
        },
    )
    return ComparisonOutcome(
        alignment=alignment,
        candidates=candidates,
        clusters=clusters,
        aligned_original=original_image,
        aligned_suspect=warped_suspect,
        symbol_comparisons=symbol_comparisons,
        braille=braille_comparison,
        warnings=warnings,
        detection_profile=profile,
        used_change_regions=used_regions,
        mode=mode,
        compared_outline=compared_area_outline(compare_mask),
        ink_original=ink_original,
        ink_suspect=ink_suspect,
        graphics_compared=graphics_wanted,
    )


# The outline is traced on a copy of the mask no larger than this (a 300 DPI carton page is ~5000 px, so
# nearly at full size: traced at 1400 px it strayed several pixels from the true edge, which shows when
# zoomed in on a die line) and simplified to within this many pixels of that copy.
_OUTLINE_LONG_EDGE_PX = 4000
_OUTLINE_EPSILON_PX = 1.0
_OUTLINE_MIN_AREA_FRACTION = 1e-4  # specks of the mask smaller than this share of the page are not shown


def compared_area_outline(compare_mask: np.ndarray) -> list[list[tuple[float, float]]] | None:
    """The compared area as polygons in the mask's own pixels: outer boundaries and holes.

    None when nothing was left out, i.e. the whole page was compared.
    """
    if compare_mask.all():
        return None
    height, width = compare_mask.shape
    scale = min(1.0, _OUTLINE_LONG_EDGE_PX / max(height, width))
    small = cv2.resize(
        compare_mask.astype(np.uint8), (max(2, round(width * scale)), max(2, round(height * scale))), interpolation=cv2.INTER_NEAREST
    )
    contours, _ = cv2.findContours(small, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    minimum = _OUTLINE_MIN_AREA_FRACTION * small.size
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < minimum:
            continue
        simplified = cv2.approxPolyDP(contour, _OUTLINE_EPSILON_PX, True)
        polygons.append([(float(x) / scale, float(y) / scale) for x, y in simplified[:, 0, :]])
    return polygons or None


def effective_detection_config(
    config: DetectionConfig, alignment: AlignmentResult
) -> tuple[DetectionConfig, str]:
    """Pick how sensitive detection can afford to be, given how well the files align.

    Every noise filter (positional tolerance, the morphological opening, the
    minimum area) exists to absorb misregistration between the two files. When
    alignment is exact there is none to absorb, so those filters can only do
    harm: on a real carton pair they erased every stroke thinner than ~5 px,
    turning one missing sentence into four tiny boxes on stray letters. So an
    exact alignment switches them off and reports at full sensitivity.
    """
    if alignment.residual_error_px > config.exact_alignment_residual_px:
        return config, "tolerant"
    residual = alignment.residual_error_px
    intensity = min(
        config.exact_max_intensity_tolerance,
        max(config.exact_intensity_tolerance, math.ceil(config.exact_levels_per_px * residual)),
    )
    delta_e = min(config.exact_max_delta_e, max(config.color_delta_e_threshold, 1.0 + config.exact_delta_e_per_px * residual))
    exact = replace(
        config,
        registration_tolerance_px=0,
        color_registration_tolerance_px=0,
        edge_tolerance_px=0,
        registration_intensity_tolerance=intensity,
        color_delta_e_threshold=delta_e,
        color_min_area_px=config.exact_min_area_px,
        ssim_min_area_px=config.exact_min_area_px,
        residue_slack_px=min(config.residue_slack_px, 1),
        local_forgive_px=0,
        local_shift_radius_px=0,
        local_blur_px=0.0,
    )
    return exact, "exact"


def _region_threshold(config: DetectionConfig, profile: str) -> int:
    """How big a pixel change must be to open a region (see `find_change_regions`)."""
    if profile == "exact":
        return min(config.region_diff_threshold, config.registration_intensity_tolerance * 2)
    return config.region_diff_threshold


def _detect_in_regions(
    original: np.ndarray,
    suspect: np.ndarray,
    compare_mask: np.ndarray,
    regions: list[ChangeRegion],
    options,
    config: DetectionConfig,
) -> list[DiffCandidate]:
    """Run the pixel detectors on each changed region, at full resolution.

    Boxes come back in the region's own pixels and are shifted into the shared
    frame. Regions are cropped with a margin already, so a detector sees the
    context it needs (SSIM in particular wants a window's worth around a change).
    """
    candidates: list[DiffCandidate] = []
    for region in regions:
        window = (slice(region.y0, region.y1), slice(region.x0, region.x1))
        original_crop, suspect_crop, mask_crop = original[window], suspect[window], compare_mask[window]

        found: list[DiffCandidate] = []
        if options.compare_color:
            found += color_diff.detect(original_crop, suspect_crop, config, mask_crop)
        if options.compare_graphics:
            found += graphics_diff.detect(original_crop, suspect_crop, config, mask_crop)
        candidates += [_shifted(candidate, region.x0, region.y0) for candidate in found]
    return candidates


_LINE_CONTEXT_PX = 12  # a line has 'nothing printed beside it' if this much of the page on either side of it (outside its own box)...
_LINE_TEXTURE_SHARE = 0.03  # ...has less than this share of pixels at an edge: flat colour, or blank paper. Type and a table's rules do
_LINE_EDGE_LEVEL = 40  # (a pixel is at an edge if the picture changes by more than this across it)
_EDGE_BAND_PX = 40  # a line this close to the edge of the compared area is the die line round it
_EDGE_SCALE = 8  # (the band is worked out on a copy this many times smaller)


def _drop_die_lines(
    candidates: list[DiffCandidate], original: np.ndarray, suspect: np.ndarray, compare_mask: np.ndarray, config: DetectionConfig
) -> tuple[list[DiffCandidate], int]:
    """Leave out what is only a die line or fold line one file draws and the other does not (or draws in another colour).

    The die line is not the artwork: a carton's cut and crease lines are laid over it, in a spot colour that is often
    not there in the other file, and every line of them showed as a difference. A difference that is one long thin
    stroke (`local_diff` flags it) is ignored when nothing is printed beside it -- a fold line across a blank flap, or
    between two panels of flat colour, has flat colour or paper on both sides -- or when it lies on the edge of the
    compared area, where the outline is. A thin line beside type (a table's rule, an underline) is still a difference.
    """
    dashed = _dashed_runs(candidates)
    lines = [c for c in candidates if c.line_work] + [c for run in dashed for c in run if not c.line_work]
    if not lines:
        return candidates, 0
    height, width = compare_mask.shape
    small = cv2.resize(compare_mask.astype(np.uint8), (max(1, width // _EDGE_SCALE), max(1, height // _EDGE_SCALE)), interpolation=cv2.INTER_NEAREST)
    radius = max(1, _EDGE_BAND_PX // _EDGE_SCALE)
    inner = cv2.erode(small, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
    band = (small > 0) & (inner == 0)
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY), cv2.cvtColor(suspect, cv2.COLOR_RGB2GRAY)

    def on_edge(box: BoundingBox) -> bool:
        cx, cy = box.center
        column, row = min(band.shape[1] - 1, int(cx / _EDGE_SCALE)), min(band.shape[0] - 1, int(cy / _EDGE_SCALE))
        return bool(band[max(0, row - 1) : row + 2, max(0, column - 1) : column + 2].any())

    def nothing_beside(box: BoundingBox) -> bool:
        x0, y0, x1, y1 = padded_window(box, _LINE_CONTEXT_PX + 3, (height, width))
        inside = padded_window(box, 3, (height, width))
        for picture in gray:
            crop = picture[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            blurred = cv2.GaussianBlur(crop, (0, 0), 0.8)
            gradient = np.maximum(np.abs(cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)), np.abs(cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=3)))
            at_edge = gradient > _LINE_EDGE_LEVEL * 4  # (a 3x3 Sobel sums four times the step)
            at_edge[inside[1] - y0 : inside[3] - y0, inside[0] - x0 : inside[2] - x0] = False  # (the line itself is not 'beside' it)
            outside = crop.size - (inside[3] - inside[1]) * (inside[2] - inside[0])
            if outside > 0 and at_edge.sum() / outside > _LINE_TEXTURE_SHARE:
                return False
        return True

    long_ones = _long_lines(lines) | {id(c) for run in dashed if _run_length(run) >= _LONG_LINE_PX for c in run}
    dropped = [c for c in lines if id(c) in long_ones or on_edge(c.bbox) or nothing_beside(c.bbox)]
    dropped_ids = {id(c) for c in dropped}
    return [c for c in candidates if id(c) not in dropped_ids], len(dropped)


_LONG_LINE_PX = 400  # a straight line this long (in pieces that meet end to end) is a cut or crease line, whatever is beside it
_DASH_MAX_THICKNESS_PX = 12  # a dash of a dashed line is thin...
_DASH_MAX_LENGTH_PX = 80  # ...and short
_DASH_MAX_GAP_PX = 40  # dashes this close along a line are one dashed line
_DASH_LEVEL_PX = 3  # ...if they lie on it to within this
_DASH_MIN_PIECES = 4


def _dashed_runs(candidates: list[DiffCandidate]) -> list[list[DiffCandidate]]:
    """Runs of thin, short marks that lie in a row -- the dashes of a dashed cut or fold line, cut into findings one by one.

    A dashed line that one file draws a few pixels from where the other does (or in another colour) differs in every dash,
    and no dash is long enough to be 'line-work' on its own: the run of them is the line."""
    dashes = []
    for candidate in candidates:
        box = candidate.bbox
        thin, long = min(box.width, box.height), max(box.width, box.height)
        if (
            candidate.line_work
            or candidate.region
            or candidate.type not in (DiffType.GRAPHICS, DiffType.COLOUR)
            or thin > _DASH_MAX_THICKNESS_PX
            or not (8 <= long <= _DASH_MAX_LENGTH_PX)
            or long < 2 * thin
        ):
            continue
        dashes.append(candidate)
    runs: list[list[DiffCandidate]] = []
    for horizontal in (True, False):
        pieces = [c for c in dashes if (c.bbox.width >= c.bbox.height) == horizontal]
        # along the line, by where it lies across, so the dashes of one line come one after another
        pieces.sort(key=lambda c: (c.bbox.center[1] if horizontal else c.bbox.center[0]))
        rows: list[list[DiffCandidate]] = []
        for piece in pieces:
            level = piece.bbox.center[1] if horizontal else piece.bbox.center[0]
            if rows and abs(level - (rows[-1][-1].bbox.center[1] if horizontal else rows[-1][-1].bbox.center[0])) <= _DASH_LEVEL_PX:
                rows[-1].append(piece)
            else:
                rows.append([piece])
        for row in rows:
            row.sort(key=lambda c: c.bbox.x0 if horizontal else c.bbox.y0)
            run = [row[0]]
            for piece in row[1:]:
                previous = run[-1].bbox
                gap = piece.bbox.x0 - previous.x1 if horizontal else piece.bbox.y0 - previous.y1
                if gap <= _DASH_MAX_GAP_PX:
                    run.append(piece)
                else:
                    if len(run) >= _DASH_MIN_PIECES:
                        runs.append(run)
                    run = [piece]
            if len(run) >= _DASH_MIN_PIECES:
                runs.append(run)
    return runs


def _run_length(run: list[DiffCandidate]) -> float:
    first = run[0].bbox
    box = first
    for candidate in run[1:]:
        box = box.union(candidate.bbox)
    return max(box.width, box.height)



def _long_lines(lines: list[DiffCandidate]) -> set[int]:
    """The ids of the line-work pieces that lie end to end along one straight line at least `_LONG_LINE_PX` long: a fold or
    cut line running across the artwork is cut into pieces by the tiles the difference was found in."""
    parent = list(range(len(lines)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def vertical(box: BoundingBox) -> bool:
        return box.height >= 3 * box.width

    for i, a in enumerate(lines):
        for j in range(i + 1, len(lines)):
            b = lines[j]
            if vertical(a.bbox) != vertical(b.bbox):
                continue
            if vertical(a.bbox):
                across = abs(a.bbox.center[0] - b.bbox.center[0])
                gap = max(a.bbox.y0 - b.bbox.y1, b.bbox.y0 - a.bbox.y1, 0.0)
            else:
                across = abs(a.bbox.center[1] - b.bbox.center[1])
                gap = max(a.bbox.x0 - b.bbox.x1, b.bbox.x0 - a.bbox.x1, 0.0)
            if across <= 6 and gap <= 6:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for index in range(len(lines)):
        groups.setdefault(find(index), []).append(index)
    long_ones: set[int] = set()
    for members in groups.values():
        boxes = [lines[m].bbox for m in members]
        length = max(
            (max(b.y1 for b in boxes) - min(b.y0 for b in boxes)) if vertical(boxes[0]) else (max(b.x1 for b in boxes) - min(b.x0 for b in boxes)),
            0.0,
        )
        if length >= _LONG_LINE_PX:
            long_ones.update(id(lines[m]) for m in members)
    return long_ones


_FIT_PAD_PX = 4  # what lies this far outside a finding's box is looked at when its border is fitted
_FIT_MIN_PIXELS = 6  # fewest differing pixels to fit a border to
_FIT_MIN_SHARE = 0.35  # a border is fitted only if the differing pixels fill at least this much of the old box's area


def _fit_borders(clusters: list[DiffCluster], residual: np.ndarray, transform: np.ndarray, halo: int = 0) -> None:
    """Fit each finding's border to the pixels that really differ, at full size.

    A finding's box is the union of its candidates', and some of them come from reduced copies of the page or from a
    tolerance of a few pixels, so the border can lie a few pixels off what changed (a marker 'a little moved').
    The pixels the deep local comparison found different are exact; the border is drawn round the ones in and just
    round the box -- unless too little of it is backed by such pixels (a faint tint change, a barcode), when it stays.
    The border is drawn `halo` pixels outside them: the comparison forgives a pixel each way and needs a level of
    difference to count a pixel, so the pixels that differ are a little fewer than the ones that changed.
    """
    for cluster in clusters:
        if all(member.region for member in cluster.members):
            continue  # a recoloured area is found by its colour, at its own exact edge
        box = cluster.bbox_in_original
        x0, y0, x1, y1 = padded_window(box, _FIT_PAD_PX, residual.shape)
        ys, xs = np.nonzero(residual[y0:y1, x0:x1])
        if len(xs) < _FIT_MIN_PIXELS:
            continue
        height, width = residual.shape
        fitted = BoundingBox(
            float(max(0, x0 + xs.min() - halo)),
            float(max(0, y0 + ys.min() - halo)),
            float(min(width, x0 + xs.max() + 1 + halo)),
            float(min(height, y0 + ys.max() + 1 + halo)),
        )
        if fitted.area < _FIT_MIN_SHARE * box.area:
            continue
        cluster.bbox_in_original = fitted
        cluster.bbox_in_suspect = original_to_suspect(fitted, transform)


def _describe_clusters(
    clusters: list[DiffCluster], original: np.ndarray, suspect: np.ndarray, config: DetectionConfig, warnings: list[str]
) -> None:
    """Give each finding the text change its own box shows (its type and description follow)."""
    # (A recoloured area is a colour finding whatever is printed on it: the words on it are findings of their own.)
    describable = [cluster for cluster in clusters if not all(member.region for member in cluster.members)]
    chosen = describable[: config.text_describe_max_findings]
    for cluster, described in zip(chosen, text_diff.describe([c.bbox_in_original for c in chosen], original, suspect, config)):
        if described is not None:
            add_member(cluster, described)
    if len(describable) > len(chosen):
        warnings.append(
            f"There are {len(describable)} differences; the words were read in the first {len(chosen)} only, "
            "so the rest are described as artwork changes rather than text changes."
        )


_MOVE_MIN_PX = 6  # something that is this far from where it was, or further, has moved (nearer is the registration)
_MOVE_MAX_PX = 2400
_MOVE_SIZE_TOLERANCE = 0.12  # the two boxes' sides may differ by this much (the ink softens at an edge)
_MOVE_MIN_SIMILARITY = 0.8  # normalised correlation of what is in them
_MOVE_MIN_STRUCTURE = 10.0
_QUOTED_WORDS = re.compile(r"'(.*)' (?:missing from suspect|added in suspect)")


def _mark_moves(
    clusters: list[DiffCluster], original: np.ndarray, suspect: np.ndarray, config: DetectionConfig
) -> list[DiffCluster]:
    """Say which findings are things that MOVED: lettering or a graphic that lies elsewhere in the suspect.

    Two ways it shows. Far enough that it is a difference twice over -- missing where it was, added where it is -- the two
    findings are one (`_pair_moves`). Near enough that the two positions run together in one finding, the crop of the
    original fits the suspect a few pixels along (`_find_shifted`).
    """
    if config.pair_moves:
        clusters = _pair_moves(clusters, original, suspect)
    if config.find_moves:
        _find_shifted(clusters, original, suspect, config.find_moves_max_findings)
    return clusters


def _words_of_finding(cluster: DiffCluster) -> str | None:
    """The words a text finding's description quotes (for a word missing or added), lower-cased and alphanumeric only."""
    found = _QUOTED_WORDS.search(cluster.detail)
    if found is None:
        return None
    return text_diff.canonical(found.group(1)) or None


def _looks_alike(original: np.ndarray, suspect: np.ndarray, a: BoundingBox, b: BoundingBox) -> bool:
    """Whether the picture in `a` of the original is the picture in `b` of the suspect (about the same size, and correlated)."""
    if abs(b.width - a.width) > _MOVE_SIZE_TOLERANCE * max(a.width, b.width) + 2:
        return False
    if abs(b.height - a.height) > _MOVE_SIZE_TOLERANCE * max(a.height, b.height) + 2:
        return False
    height, width = original.shape[:2]
    ax0, ay0, ax1, ay1 = padded_window(a, 0, (height, width))
    bx0, by0, bx1, by1 = padded_window(b, 0, suspect.shape[:2])
    if ax1 - ax0 < 4 or ay1 - ay0 < 4 or bx1 - bx0 < 4 or by1 - by0 < 4:
        return False
    here = cv2.cvtColor(original[ay0:ay1, ax0:ax1], cv2.COLOR_RGB2GRAY)
    there = cv2.cvtColor(suspect[by0:by1, bx0:bx1], cv2.COLOR_RGB2GRAY)
    there = cv2.resize(there, (here.shape[1], here.shape[0]), interpolation=cv2.INTER_AREA)
    if float(here.std()) < _MOVE_MIN_STRUCTURE or float(there.std()) < _MOVE_MIN_STRUCTURE:
        return False
    return float(cv2.matchTemplate(here, there, cv2.TM_CCOEFF_NORMED)[0, 0]) >= _MOVE_MIN_SIMILARITY


def _pair_moves(clusters: list[DiffCluster], original: np.ndarray, suspect: np.ndarray) -> list[DiffCluster]:
    """Findings that are one thing moved: what the original has here and the suspect has there.

    Lettering or a graphic that sits somewhere else in the suspect is two differences to the pixels: it is missing where
    it was and added where it is. Where the two are alike -- the same words, or boxes of the same size holding the same
    picture -- they are one finding, that it MOVED, and it says how far. It is shown where it was in the original and where
    it is in the suspect.
    """
    removed = [c for c in clusters if c.subtype is DiffSubtype.REMOVED and c.category in ("graphics", "text") and not c.uncertain]
    added = [c for c in clusters if c.subtype is DiffSubtype.ADDED and c.category in ("graphics", "text") and not c.uncertain]
    if not removed or not added:
        return clusters
    used: set[int] = set()
    replaced: dict[int, DiffCluster] = {}
    for gone in removed:
        a = gone.bbox_in_original
        words = _words_of_finding(gone) if gone.category == "text" else None
        candidates = []
        for new in added:
            if id(new) in used or new.category != gone.category:
                continue
            b = new.bbox_in_original
            distance = math.hypot(b.center[0] - a.center[0], b.center[1] - a.center[1])
            if not (_MOVE_MIN_PX <= distance <= _MOVE_MAX_PX):
                continue
            if gone.category == "text":
                if words is None or _words_of_finding(new) != words:
                    continue
            elif not _looks_alike(original, suspect, a, b):
                continue
            candidates.append((distance, new))
        if not candidates:
            continue
        distance, new = min(candidates, key=lambda pair: pair[0])
        used.add(id(new))
        dx, dy = new.bbox_in_original.center[0] - a.center[0], new.bbox_in_original.center[1] - a.center[1]
        quoted = _QUOTED_WORDS.search(gone.detail)
        kind = "Text" if gone.category == "text" else "Graphic"
        detail = f"{kind} moved {movement.how_far(dx, dy)}" + (f": '{quoted.group(1)}'" if quoted and kind == "Text" else "")
        replaced[id(gone)] = DiffCluster(
            id=gone.id,
            bbox_in_original=gone.bbox_in_original,
            bbox_in_suspect=new.bbox_in_suspect,
            type=gone.type,
            members=[replace(m, subtype=DiffSubtype.MOVED, magnitude=distance) for m in gone.members + new.members],
            detail=detail,
        )
    if not replaced:
        return clusters
    out = [replaced.get(id(c), c) for c in clusters if id(c) not in used]
    for number, cluster in enumerate(out, start=1):  # (reading order is kept: a moved finding stands where it was)
        cluster.id = number
    return out


def _find_shifted(clusters: list[DiffCluster], original: np.ndarray, suspect: np.ndarray, most: int) -> None:
    """Findings whose crop of the original fits the suspect a few pixels along are that picture, moved."""
    shiftable = [
        c
        for c in clusters
        if c.subtype is DiffSubtype.CHANGED
        and c.category in ("graphics", "colour")
        and not any(m.region or m.line_work for m in c.members)
        and max(c.bbox_in_original.width, c.bbox_in_original.height) >= 8
    ][:most]
    explained: set[int] = set()
    for cluster, shift in zip(shiftable, movement.find_moves([c.bbox_in_original for c in shiftable], original, suspect)):
        if shift is None:
            continue
        distance = math.hypot(*shift)
        cluster.members = [replace(m, subtype=DiffSubtype.MOVED, magnitude=distance) for m in cluster.members]
        cluster.type = DiffType.GRAPHICS
        cluster.detail = f"Graphic moved {movement.how_far(*shift)}"
        # the strips where it was and where it is came out as findings of their own: they are this one
        box = cluster.bbox_in_original
        for other in clusters:
            piece = other.bbox_in_original
            if (
                other is cluster
                or other.subtype is DiffSubtype.MOVED
                or other.category not in ("graphics", "colour")
                or any(m.region for m in other.members)
                or piece.intersection_area(box) < 0.8 * piece.area
            ):
                continue
            if movement.explained_by_shift(original, suspect, piece, shift):
                explained.add(id(other))
    if explained:
        clusters[:] = [c for c in clusters if id(c) not in explained]
        for number, cluster in enumerate(clusters, start=1):
            cluster.id = number


def _shifted(candidate: DiffCandidate, dx: float, dy: float) -> DiffCandidate:
    box = candidate.bbox
    return replace(candidate, bbox=BoundingBox(box.x0 + dx, box.y0 + dy, box.x1 + dx, box.y1 + dy))


@dataclass
class _Area:
    """One file's artwork area: inside its boundary, less what was erased by hand. Nothing chosen = the page."""

    boundary: np.ndarray | None = None
    erased: np.ndarray | None = None

    @classmethod
    def of(cls, image: np.ndarray, geometry: IsolationGeometry | None) -> "_Area":
        shape = image.shape[:2]
        return cls(boundary_mask(shape, geometry), erased_mask(shape, geometry))

    @property
    def is_whole_page(self) -> bool:
        return self.boundary is None and self.erased is None

    def blanked(self, image: np.ndarray) -> np.ndarray:
        """The image with everything outside the area cleared to white (the image itself if nothing was chosen)."""
        if self.is_whole_page:
            return image
        keep = np.ones(image.shape[:2], bool)
        if self.boundary is not None:
            keep &= self.boundary
        if self.erased is not None:
            keep &= ~self.erased
        return apply_mask(image, keep)


def _warped(mask: np.ndarray, transform: np.ndarray, reference_shape: tuple[int, int]) -> np.ndarray:
    height, width = reference_shape
    warped = cv2.warpPerspective(
        mask.astype(np.uint8) * 255, transform, (width, height), flags=cv2.INTER_NEAREST, borderValue=0
    )
    return warped > 0


def _build_compare_mask(
    suspect_shape: tuple[int, int],
    reference_shape: tuple[int, int],
    transform: np.ndarray,
    original_area: _Area,
    suspect_area: _Area,
) -> np.ndarray:
    """What is compared, in the original's pixels.

    The artwork of either file: an area only the suspect has is compared too (where the original is
    blank, what the suspect has there is a difference), rather than only where the two outlines
    agree, which would hide a flap or a panel one of them lacks. A file with no area of its own is
    taken to have the other's. Whatever was erased by hand, in either file, is left out.
    """
    mask = warp_validity_mask(suspect_shape, transform, reference_shape)

    areas = []
    if original_area.boundary is not None:
        areas.append(original_area.boundary)
    if suspect_area.boundary is not None:
        areas.append(_warped(suspect_area.boundary, transform, reference_shape))
    if areas:
        mask &= np.logical_or.reduce(areas)

    if original_area.erased is not None:
        mask &= ~original_area.erased
    if suspect_area.erased is not None:
        mask &= ~_warped(suspect_area.erased, transform, reference_shape)
    return mask


_UNVERIFIABLE_SHARE = 0.5  # a word mostly inside an area the scan cannot hold is not reported


def _readable_only(candidates: list[DiffCandidate], unverifiable: np.ndarray | None) -> list[DiffCandidate]:
    """Drop the text findings that lie mostly where the two files cannot be compared at fine scale."""
    if unverifiable is None:
        return candidates
    height, width = unverifiable.shape
    kept = []
    for candidate in candidates:
        box = candidate.bbox
        x0, y0 = max(0, int(box.x0)), max(0, int(box.y0))
        x1, y1 = min(width, math.ceil(box.x1)), min(height, math.ceil(box.y1))
        window = unverifiable[y0:y1, x0:x1]
        if window.size and window.mean() > _UNVERIFIABLE_SHARE:
            logger.info("text finding skipped: the scan cannot hold detail there", extra={"detail": candidate.detail})
            continue
        kept.append(candidate)
    return kept


def _for_reading(
    original_ink: np.ndarray, suspect_ink: np.ndarray, config: DetectionConfig
) -> tuple[np.ndarray, np.ndarray, float]:
    """The two ink pictures reduced (never enlarged) to the size text is read at, and the scale used."""
    scale = min(1.0, config.ocr_long_edge_px / max(original_ink.shape[:2]))
    if scale >= 1.0:
        return original_ink, suspect_ink, 1.0
    size = (max(1, round(original_ink.shape[1] * scale)), max(1, round(original_ink.shape[0] * scale)))
    return (
        cv2.resize(original_ink, size, interpolation=cv2.INTER_AREA),
        cv2.resize(suspect_ink, size, interpolation=cv2.INTER_AREA),
        scale,
    )


def _to_working_resolution(
    original: np.ndarray, suspect: np.ndarray, mask: np.ndarray, config
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    long_edge = max(original.shape[:2])
    if long_edge <= config.detection_working_long_edge_px:
        return original, suspect, mask, 1.0

    scale = config.detection_working_long_edge_px / long_edge
    size = (int(original.shape[1] * scale), int(original.shape[0] * scale))
    return (
        cv2.resize(original, size, interpolation=cv2.INTER_AREA),
        cv2.resize(suspect, size, interpolation=cv2.INTER_AREA),
        cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool),
        scale,
    )


def _rescaled(candidate: DiffCandidate, factor_x: float, factor_y: float | None = None) -> DiffCandidate:
    """`candidate` with its box scaled: by `factor_x` across and `factor_y` down (the same, if only one is given)."""
    factor_y = factor_x if factor_y is None else factor_y
    if factor_x == 1.0 and factor_y == 1.0:
        return candidate
    return replace(candidate, bbox=candidate.bbox.scaled_xy(factor_x, factor_y))
