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
from typing import Callable
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from app.config import AppConfig, DetectionConfig
from app.engine.alignment.aligner import align
from app.engine.aggregation.aggregator import build_clusters
from app.engine.alignment.homography import warp_to_reference, warp_validity_mask
from app.engine.detectors import (
    barcode_diff,
    braille_diff,
    color_diff,
    graphics_diff,
    structure_diff,
    text_diff,
)
from app.engine.detectors.change_regions import ChangeRegion, find_change_regions
from app.engine.detectors.text_diff import OcrUnavailableError
from app.engine.isolation.boundary_mask import apply_mask, boundary_mask, erased_mask
from app.engine.preprocessing.scan_normalizer import ink_image
from app.logging_config import StageTimer
from app.models.diff import BoundingBox, DiffCandidate, DiffCluster
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
    alignment = align(
        artwork_original, artwork_suspect, config.alignment, original_geometry, suspect_geometry
    )
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
    if options.normalize_scan:
        say("Removing the background colour…")
        with StageTimer(logger, "ink_pictures"):
            ink_original = ink_image(masked_original, detection_config, compare_mask)
            ink_suspect = ink_image(masked_suspect, detection_config, compare_mask)

    used_regions = False
    # Graphics of files that do not register exactly are compared by where INK is, within
    # the measured registration error: the similarity map's noise filters, needed there,
    # erase thin real changes (a missing line of small print scored 0.07 IoU).
    misregistered = options.normalize_scan or profile != "exact"
    graphics_mode = detection_config.graphics_when_misregistered if misregistered else "ssim"
    use_structure = options.compare_graphics and graphics_mode in ("structure", "both")
    use_ssim = options.compare_graphics and graphics_mode in ("ssim", "both") and not options.normalize_scan
    unverifiable_area: np.ndarray | None = None
    if use_structure:
        # Compare where INK is, at full resolution, tolerating the registration error
        # we measured (FR-6.3 for scans: colour is ignored there).
        structure = structure_diff.detect(
            masked_original,
            masked_suspect,
            detection_config,
            structure_diff.slack_for(alignment.residual_error_px, detection_config),
            compare_mask,
        )
        candidates += structure.candidates
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

    pixel_detectors = options.compare_color or options.compare_graphics
    regions = None
    if pixel_detectors and not options.normalize_scan:
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
        needs_pixels = regions is None and pixel_detectors and not options.normalize_scan
        if needs_pixels or options.compare_text:
            if options.normalize_scan:
                # Text is read from the ink pictures at close to 300 DPI, not the coarser detection copy.
                small_original, small_suspect, scale = _for_reading(ink_original, ink_suspect, detection_config)
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
            if options.compare_text:
                try:
                    working_candidates += text_diff.detect(
                        small_original, small_suspect, detection_config
                    )
                except OcrUnavailableError as exc:
                    logger.warning("text detection skipped: %s", exc)
                    warnings.append(str(exc))

            working_candidates = [_rescaled(candidate, 1 / scale) for candidate in working_candidates]
            if options.normalize_scan:
                # A scan too soft to hold small print cannot be read there either: not a difference.
                working_candidates = _readable_only(working_candidates, unverifiable_area)
            candidates += working_candidates

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

    say("Grouping the differences\u2026")
    clusters = build_clusters(candidates, alignment.transform, config.clustering)

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


def _shifted(candidate: DiffCandidate, dx: float, dy: float) -> DiffCandidate:
    box = candidate.bbox
    return DiffCandidate(
        bbox=BoundingBox(box.x0 + dx, box.y0 + dy, box.x1 + dx, box.y1 + dy),
        type=candidate.type,
        subtype=candidate.subtype,
        confidence=candidate.confidence,
        detail=candidate.detail,
        magnitude=candidate.magnitude,
    )


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


def _rescaled(candidate: DiffCandidate, factor: float) -> DiffCandidate:
    if factor == 1.0:
        return candidate
    return DiffCandidate(
        bbox=candidate.bbox.scaled(factor),
        type=candidate.type,
        subtype=candidate.subtype,
        confidence=candidate.confidence,
        detail=candidate.detail,
        magnitude=candidate.magnitude,
    )
