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
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.config import AppConfig
from app.engine.alignment.aligner import align
from app.engine.aggregation.aggregator import build_clusters
from app.engine.alignment.homography import warp_to_reference, warp_validity_mask
from app.engine.detectors import barcode_diff, braille_diff, color_diff, graphics_diff, text_diff
from app.engine.detectors.text_diff import OcrUnavailableError
from app.engine.isolation.boundary_mask import apply_mask, resolve_keep_mask
from app.engine.preprocessing.scan_normalizer import normalize_for_scan
from app.logging_config import StageTimer
from app.models.diff import DiffCandidate, DiffCluster
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


def run_comparison(
    original_image: np.ndarray,
    suspect_image: np.ndarray,
    mode: ComparisonMode,
    config: AppConfig,
    original_geometry: IsolationGeometry | None = None,
    suspect_geometry: IsolationGeometry | None = None,
) -> ComparisonOutcome:
    options = options_for_mode(mode)
    detection_config = config.detection

    alignment = align(
        original_image, suspect_image, config.alignment, original_geometry, suspect_geometry
    )

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

    if options.compare_color or options.compare_graphics or options.compare_text:
        small_original, small_suspect, small_mask, scale = _to_working_resolution(
            masked_original, masked_suspect, compare_mask, detection_config
        )
        if options.normalize_scan:
            small_original = normalize_for_scan(small_original)
            small_suspect = normalize_for_scan(small_suspect)

        working_candidates: list[DiffCandidate] = []
        if options.compare_color:
            working_candidates += color_diff.detect(
                small_original, small_suspect, detection_config, small_mask
            )
        if options.compare_graphics:
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
