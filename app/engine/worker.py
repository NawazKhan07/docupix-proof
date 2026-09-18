"""Comparison pipeline: isolate -> align -> detect (Architecture §3.3).

This is the function Phase 7 will run inside a worker process; for now the
UI calls it directly. It owns the two things individual detectors must not
each decide for themselves:

* the single hard mask that keeps furniture out of every detector's view
  (FR-2.1), built from both files' isolation geometry *and* the region the
  warped suspect actually covers;
* the working resolution detection runs at, with candidate boxes scaled back
  into the shared aligned space afterwards so everything downstream speaks
  one coordinate system (FR-7.4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from app.config import AppConfig, DetectionConfig
from app.engine.alignment.aligner import align
from app.engine.aggregation.aggregator import build_clusters
from app.engine.alignment.homography import warp_to_reference, warp_validity_mask
from app.engine.detectors import barcode_diff, braille_diff, color_diff, graphics_diff, text_diff
from app.engine.detectors.change_regions import ChangeRegion, find_change_regions
from app.engine.detectors.text_diff import OcrUnavailableError
from app.engine.isolation.boundary_mask import apply_mask, resolve_keep_mask
from app.engine.preprocessing.scan_normalizer import normalize_for_scan
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


def run_comparison(
    original_image: np.ndarray,
    suspect_image: np.ndarray,
    mode: ComparisonMode,
    config: AppConfig,
    original_geometry: IsolationGeometry | None = None,
    suspect_geometry: IsolationGeometry | None = None,
) -> ComparisonOutcome:
    options = options_for_mode(mode)

    alignment = align(
        original_image, suspect_image, config.alignment, original_geometry, suspect_geometry
    )
    detection_config, profile = effective_detection_config(config.detection, alignment)

    reference_shape = original_image.shape[:2]
    with StageTimer(logger, "warp_to_shared_space"):
        warped_suspect = warp_to_reference(suspect_image, alignment.transform, reference_shape)
        compare_mask = _build_compare_mask(
            original_image,
            suspect_image,
            alignment.transform,
            original_geometry,
            suspect_geometry,
        )

    # FR-2.1, enforced once here rather than trusted to each detector.
    masked_original = apply_mask(original_image, compare_mask)
    masked_suspect = apply_mask(warped_suspect, compare_mask)

    candidates: list[DiffCandidate] = []
    warnings: list[str] = []
    symbol_comparisons: list[SymbolComparison] = []
    braille_comparison: BrailleComparison | None = None

    used_regions = False
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
            masked_original, masked_suspect, compare_mask, regions, options, detection_config
        )

    if options.compare_color or options.compare_graphics or options.compare_text:
        needs_pixels = regions is None and pixel_detectors
        if needs_pixels or options.compare_text:
            small_original, small_suspect, small_mask, scale = _to_working_resolution(
                masked_original, masked_suspect, compare_mask, detection_config
            )
            if options.normalize_scan:
                small_original = normalize_for_scan(small_original)
                small_suspect = normalize_for_scan(small_suspect)

            working_candidates: list[DiffCandidate] = []
            if needs_pixels and options.compare_color:
                working_candidates += color_diff.detect(
                    small_original, small_suspect, detection_config, small_mask
                )
            if needs_pixels and options.compare_graphics:
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

            candidates += [_rescaled(candidate, 1 / scale) for candidate in working_candidates]

    # Symbols and Braille dots run at full resolution: a small barcode or a
    # 1 mm Braille dot does not survive the detection downscale.
    if options.decode_symbols:
        symbol_candidates, symbol_comparisons = barcode_diff.compare(
            masked_original, masked_suspect, detection_config
        )
        candidates += symbol_candidates
        if not symbol_comparisons:
            warnings.append(
                "No decodable symbols were found in either file. Note that Pharmacode "
                "is not supported by the bundled decoders."
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
    )


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
    exact = replace(
        config,
        registration_tolerance_px=0,
        color_registration_tolerance_px=0,
        edge_tolerance_px=0,
        registration_intensity_tolerance=config.exact_intensity_tolerance,
        color_min_area_px=config.exact_min_area_px,
        ssim_min_area_px=config.exact_min_area_px,
    )
    return exact, "exact"


def _region_threshold(config: DetectionConfig, profile: str) -> int:
    """How big a pixel change must be to open a region (see `find_change_regions`)."""
    if profile == "exact":
        return min(config.region_diff_threshold, config.exact_intensity_tolerance * 2)
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


def _build_compare_mask(
    original_image: np.ndarray,
    suspect_image: np.ndarray,
    transform: np.ndarray,
    original_geometry: IsolationGeometry | None,
    suspect_geometry: IsolationGeometry | None,
) -> np.ndarray:
    reference_shape = original_image.shape[:2]
    mask = warp_validity_mask(suspect_image.shape[:2], transform, reference_shape)

    if original_geometry is not None:
        mask &= resolve_keep_mask(reference_shape, original_geometry)

    if suspect_geometry is not None:
        suspect_keep = resolve_keep_mask(suspect_image.shape[:2], suspect_geometry)
        height, width = reference_shape
        warped_keep = cv2.warpPerspective(
            suspect_keep.astype(np.uint8) * 255,
            transform,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        mask &= warped_keep > 0

    return mask


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
