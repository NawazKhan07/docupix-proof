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
import difflib
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
from app.engine import prepared
from app.engine.aggregation.aggregator import add_member, build_clusters, combined
from app.engine.alignment.block_alignment import align_blocks, composite_warp, seam_margin_mask
from app.engine.alignment.homography import apply_transform, warp_to_reference, warp_validity_mask
from app.engine.detectors import (
    barcode_diff,
    braille_diff,
    color_diff,
    elements,
    graphics_diff,
    local_diff,
    movement,
    structure_diff,
    text_diff,
    verify,
)
from app.engine.detectors import nudge
from app.engine.detectors.change_regions import ChangeRegion, find_change_regions
from app.engine.detectors.regions import padded_window
from app.engine.detectors.text_diff import OcrUnavailableError
from app.engine.isolation.boundary_mask import apply_mask, boundary_mask, erased_mask
from app.engine.isolation import auto_area, die_lines
from app.engine.isolation.block_grid import divide_into_blocks
from app.engine.preprocessing.scan_normalizer import ink_pair, reading_source
from app.logging_config import StageTimer
from app.models.diff import BoundingBox, DiffCandidate, DiffCluster, DiffSubtype, DiffType, renumbered, visible_clusters
from app.models.isolation import IsolationGeometry
from app.models.job import ComparisonMode, DetectorOptions, options_for_mode
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
    # Whether text was read and compared (Scan vs Digital does by default; the results page has the toggle).
    text_compared: bool = True
    # Leaflet only: which of the original's pages were reviewed ("" = every page; the results page has the field).
    compared_pages: str = ""
    # Every other mode: the page of each file that was compared (0-based; the results page has a selector for each file
    # of several pages).
    original_page: int = 0
    suspect_page: int = 0
    # The DPI each PDF was compared at (None for an image): the window's own copy of a file may work out another (less
    # memory was free when the comparison ran), and a zoomed-in re-render must be drawn at the comparison's.
    original_dpi: float | None = None
    suspect_dpi: float | None = None
    # Leaflet's pages and Automatic only (kept inside the worker, never shown or sent): what of the page was compared --
    # the leaflet or artwork found on it -- in the original's pixels, for the word comparison to read inside.
    compare_mask: np.ndarray | None = None
    # Block-wise segmentation (SRS block-alignment extension): each interior panel found and aligned on its
    # own, in the original's pixels -- what the results page outlines on the artwork. Empty when no blocks
    # were found, or `DetectionConfig.block_alignment_enabled` is off.
    block_outlines: list[list[tuple[float, float]]] = field(default_factory=list)
    # Each LOCALLY-aligned block's own transform (suspect pixels -> original pixels), with its polygon --
    # a block that fell back to the whole artwork's own transform is not included, since a viewer with no
    # entry for a point already uses that transform as its default. Lets a zoomed-in vector re-render
    # (`app.engine.tile_renderer.render_aligned_vector_tile`, crisp text at high zoom) apply each panel's own
    # correction too, not just the whole artwork's -- the raster views already do, via `composite_warp`.
    block_alignments: list[tuple[list[tuple[float, float]], np.ndarray]] = field(default_factory=list)
    # What was done on the reviewer's behalf that they should know (the artwork was found automatically, a page was
    # compared at a lower resolution): information, not something that was left unchecked (that is a warning).
    notes: list[str] = field(default_factory=list)
    # Scan vs Digital only: every finding `clusters` was filtered from by the graphics/text checkboxes (elsewhere,
    # the same list). Kept so the results page can flip a checkbox instantly with nothing recompared
    # (`app.models.diff.visible_clusters`).
    all_clusters: list[DiffCluster] = field(default_factory=list)
    # Automatic only: its word comparison (the results page's OCR findings) is still being made -- the pixel findings are
    # sent first, so the reviewer is not kept waiting for findings that start hidden; they follow in an update.
    ocr_pending: bool = False


def run_comparison(
    original_image: np.ndarray,
    suspect_image: np.ndarray,
    mode: ComparisonMode,
    config: AppConfig,
    original_geometry: IsolationGeometry | None = None,
    suspect_geometry: IsolationGeometry | None = None,
    progress: Callable[[str], None] | None = None,
    options: DetectorOptions | None = None,
    original_lines: np.ndarray | None = None,
    suspect_lines: np.ndarray | None = None,
    original_words: list | None = None,
    suspect_words: list | None = None,
    original_strokes: list | None = None,
    suspect_strokes: list | None = None,
    align_only: bool = False,
) -> ComparisonOutcome:
    """Run one comparison. `progress`, if given, is told what stage the job is at.

    `options` overrides the mode's own fixed ones (Leaflet's two independent strategies run the SAME page pair
    through this function twice, once with visual-style options and once with text-style ones, and merge the
    two outcomes -- see `app.engine.leaflet`); every other caller leaves it out and gets the mode's own.
    `original_lines`/`suspect_lines`, for a PDF, are its own stroked vector paths at the image's resolution
    (`block_grid.stroked_line_map`): the fold and cut lines the artwork is divided into blocks by. Without them
    the blocks are found from the pictures alone. `original_words`/`suspect_words`, for a PDF with live text, are its
    own words in its picture's pixels (`text_diff.text_layer_words`): what a finding says is taken from them rather
    than read from the pictures, where they are there. `original_strokes`/`suspect_strokes`, for a PDF, are its stroked
    paths with their colours (`die_lines.vector_strokes`): its die lines are found among them exactly.
    """
    options = options or options_for_mode(mode)
    say = progress or (lambda _text: None)

    # Erase, in each file, everything outside its own artwork area: what is aligned and compared
    # is then the two artworks alone, on white, not two sheets with a title and marks around them.
    if options.normalize_scan and suspect_geometry is not None and suspect_geometry.edge_inset_norm > config.detection.scan_edge_inset_max_norm:
        # A scan's own outline is traced on a soft, shadowed edge and set well inside it to leave that fringe out -- far
        # enough, on a label whose type runs close to its edge, to cut the first letters off every line. It is kept to a
        # few pixels: what of the fringe comes in is a thin scan-only mark along the outline, dropped as the carton edge.
        suspect_geometry = replace(suspect_geometry, edge_inset_norm=config.detection.scan_edge_inset_max_norm)
    original_area = _Area.of(original_image, original_geometry)
    suspect_area = _Area.of(suspect_image, suspect_geometry)
    artwork_original = original_area.blanked(original_image)
    artwork_suspect = suspect_area.blanked(suspect_image)

    say("Aligning the files\u2026")
    # (each slow stage is kept by what it is computed from, so a comparison prepared before Compare -- or one run
    # again in another mode -- does not do it twice: see `app.engine.prepared`)
    fingerprints: dict[int, bytes | None] = {}

    def fingerprint(array: np.ndarray | None) -> bytes | None:
        if array is None:
            return None
        if id(array) not in fingerprints:
            fingerprints[id(array)] = prepared.digest(array)
        return fingerprints[id(array)]

    def aligned(first: np.ndarray, second: np.ndarray, *geometries) -> AlignmentResult:
        key = (fingerprint(first), fingerprint(second), prepared.settings_key(config.alignment, *geometries))
        return prepared.remembered("align", key, lambda: align(first, second, config.alignment, *geometries))

    # A leaflet is aligned by what is printed on it, never by its outline: two versions' trim lines, bleed and marks
    # differ from each other, and fitting one outline onto the other would carry every line of text off by that much.
    # (Its area still decides what is compared: `artwork_original`/`artwork_suspect` are blanked outside it.)
    outlines = (None, None) if mode is ComparisonMode.LEAFLET else (original_geometry, suspect_geometry)
    try:
        alignment = aligned(artwork_original, artwork_suspect, *outlines)
    except AlignmentError:
        if original_area.is_whole_page and suspect_area.is_whole_page:
            raise
        # Trimmed to their artwork the files share too little to register (the outlines cut their features): the whole
        # pages are registered instead, and the comparison is still only of the artwork.
        alignment = aligned(original_image, suspect_image)
    detection_config, profile = effective_detection_config(config.detection, alignment)
    say("Comparing the artwork\u2026")

    reference_shape = original_image.shape[:2]
    blocks = []
    block_alignments = []
    # Panels only exist inside an artwork outline: a whole page (no area found, cropped or lifted) also holds the
    # title, notes, colour legend and blank paper, and a carton's fold lines drawn across ALL of that cut it into
    # "panels" that are nothing of the kind.
    # A leaflet page whose one page-wide fit is sure (most of its features agree on it, closely) has no panel moved
    # against the rest: aligning each panel on its own changes nothing there (measured on open-vs-closed pairs: no panel
    # ever taken) and costs seconds a page. A re-set page, where the fit leaves many features out, still gets it.
    sure_page_fit = (
        mode is ComparisonMode.LEAFLET
        and alignment.inlier_ratio >= detection_config.leaflet_sure_fit_inlier_share
        and alignment.residual_error_px <= detection_config.leaflet_sure_fit_residual_px
    )
    if (
        detection_config.block_alignment_enabled
        and not sure_page_fit
        and original_area.boundary is not None
        and suspect_area.boundary is not None
    ):
        with StageTimer(logger, "block_alignment"):
            # Every panel and flap of the artwork, in both files: each original block is aligned on its own
            # against its counterpart, the way cropping just that block in both files and aligning them would.
            def divided(job) -> list:
                image, boundary, lines = job
                key = (fingerprint(image), fingerprint(boundary), fingerprint(lines), detection_config.block_min_area_fraction)
                return prepared.remembered(
                    "blocks", key, lambda: divide_into_blocks(image, boundary, lines, detection_config.block_min_area_fraction)
                )

            with ThreadPoolExecutor(max_workers=2) as pool:
                blocks, suspect_blocks = list(
                    pool.map(divided, [(original_image, original_area.boundary, original_lines), (suspect_image, suspect_area.boundary, suspect_lines)])
                )
            if blocks:
                key = (
                    fingerprint(artwork_original), fingerprint(artwork_suspect),
                    prepared.settings_key(blocks, alignment.transform, config.alignment, suspect_blocks),
                )
                block_alignments = prepared.remembered(
                    "block_alignment",
                    key,
                    lambda: align_blocks(artwork_original, artwork_suspect, blocks, alignment.transform, config.alignment, suspect_blocks),
                )

    with StageTimer(logger, "warp_to_shared_space"):
        if block_alignments:
            warped_artwork = composite_warp(artwork_suspect, block_alignments, alignment.transform, reference_shape)
            # The viewer shows the whole suspect page; only the comparison uses the cleared copy.
            warped_suspect = (
                warped_artwork
                if artwork_suspect is suspect_image
                else composite_warp(suspect_image, block_alignments, alignment.transform, reference_shape)
            )
        else:
            warped_artwork = warp_to_reference(artwork_suspect, alignment.transform, reference_shape)
            warped_suspect = (
                warped_artwork
                if artwork_suspect is suspect_image
                else warp_to_reference(suspect_image, alignment.transform, reference_shape)
            )
        compare_mask = _build_compare_mask(
            suspect_image.shape[:2], reference_shape, alignment.transform, original_area, suspect_area
        )
        if options.normalize_scan and suspect_area.is_whole_page and _glass_beyond_page(suspect_image.shape[:2], alignment.transform, reference_shape):
            # A scan with no area set is an item laid on the glass -- a label, a carton panel -- and the scanner's page
            # round it: what of the digital file lies beyond the item (a proof sheet's other artworks, its dimension
            # lines and notes, 'FRONT', 'ACTUAL SIZE') is nothing the scan could show, missing or not. Only the item is
            # compared.
            # (Widened by the digital file's own print lying mostly inside it, whole: a line the scan is missing at the
            # item's edge is still compared, as missing.)
            extent = auto_area.scan_extent(suspect_image)
            if extent is not None:
                item = auto_area.item_in_artwork(original_image, _warped(extent, alignment.transform, reference_shape))
                compare_mask &= item
        # What the reviewer is shown as "what was covered" (FR: what was left out is visible, not assumed) --
        # the crop/lift/erase choices alone, before any seam margin is additionally hidden from detectors below.
        # A seam is this composite's own construction, not a choice the reviewer made or needs to see as a hole.
        reported_compare_mask = compare_mask
        if block_alignments:
            compare_mask = compare_mask & ~seam_margin_mask(
                reference_shape, block_alignments, detection_config.block_seam_margin_px
            )

    if align_only:
        return ComparisonOutcome(
            alignment=alignment,
            candidates=[],
            clusters=[],
            all_clusters=[],
            aligned_original=original_image,
            aligned_suspect=warped_suspect,
            symbol_comparisons=[],
            braille=None,
            warnings=[],
            detection_profile=profile,
            used_change_regions=False,
            mode=mode,
            compared_outline=compared_area_outline(reported_compare_mask),
            block_outlines=[block.polygon for block in blocks] if block_alignments else [],
            block_alignments=[(a.block.polygon, a.transform) for a in block_alignments if a.locally_aligned],
            notes=["Files updated and aligned. Click Recompare to run comparison."],
            graphics_compared=True,
            text_compared=True,
        )

    if detection_config.ignore_die_lines:
        # A carton's die lines -- cut, crease, perforation -- are not the artwork: one file drawing them and the other
        # not, or in another colour or dash, is never a difference. Both files' are left out of what is compared.
        with StageTimer(logger, "die_lines"):
            die = _die_lines_to_ignore(
                original_image, suspect_image, original_area, suspect_area, original_strokes, suspect_strokes,
                alignment.transform, reference_shape,
            )
        if die is not None:
            # each die line is drawn out of both pictures -- filled from what lies beside it, so a band or a panel it
            # crosses stays one band, one panel -- rather than cut out of the comparison (the pictures shown keep it)
            artwork_original = die_lines.drawn_out(artwork_original, die)
            warped_artwork = die_lines.drawn_out(warped_artwork, die)

    # FR-2.1, enforced once here rather than trusted to each detector.
    masked_original = apply_mask(artwork_original, compare_mask)
    masked_suspect = apply_mask(warped_artwork, compare_mask)
    # Everything from here on is worked out on the box round what is compared, not the whole page: a proof sheet's
    # artwork can be a third of it (a second plate below, a job sheet beside), and every detector's work grows with the
    # pixels. The findings are put back on the page at the end (`_on_page`).
    detect_frame = _detection_frame(compare_mask)
    page_original, page_warped = original_image, warped_suspect
    detect_transform = alignment.transform
    original_boundary = original_area.boundary
    if detect_frame is not None:
        fx0, fy0, fx1, fy1 = detect_frame
        compare_mask = compare_mask[fy0:fy1, fx0:fx1]
        masked_original = masked_original[fy0:fy1, fx0:fx1]
        masked_suspect = masked_suspect[fy0:fy1, fx0:fx1]
        original_image = original_image[fy0:fy1, fx0:fx1]
        warped_suspect = warped_suspect[fy0:fy1, fx0:fx1]
        if original_boundary is not None:
            original_boundary = original_boundary[fy0:fy1, fx0:fx1]
        detect_transform = np.array([[1.0, 0.0, -fx0], [0.0, 1.0, -fy0], [0.0, 0.0, 1.0]]) @ alignment.transform
        logger.info("detection limited to the compared area", extra={"box": list(detect_frame), "page": list(page_original.shape[:2])})
    if mode is ComparisonMode.AUTOMATIC and detection_config.food_label:
        # A food label is compared by what is printed on it, its own way (`app.engine.food_label`).
        from app.engine import food_label

        layers = (
            in_compared_area(_words_in_frame(original_words, detect_frame), compare_mask),
            in_compared_area(_words_in_frame(_suspect_words_in_frame(suspect_words, alignment.transform, block_alignments), detect_frame), compare_mask),
        )
        found = food_label.compare_label(masked_original, masked_suspect, compare_mask, detection_config, say, *layers)
        clusters = found.clusters
        candidates = [member for cluster in clusters for member in cluster.members]
        if detect_frame is not None:
            candidates, clusters, _all, _symbols, _braille, _shown_o, _shown_s = _on_page(
                detect_frame, page_original.shape[:2], candidates, clusters, [], [], None, None, None
            )
        for cluster in clusters:
            cluster.bbox_in_suspect = original_to_suspect(cluster.moved_to or cluster.bbox_in_original, alignment.transform)
        return ComparisonOutcome(
            alignment=alignment,
            candidates=candidates,
            clusters=clusters,
            all_clusters=list(clusters),
            aligned_original=page_original,
            aligned_suspect=page_warped,
            symbol_comparisons=[],
            braille=None,
            warnings=found.warnings,
            detection_profile=profile,
            used_change_regions=False,
            mode=mode,
            compared_outline=compared_area_outline(reported_compare_mask),
            graphics_compared=True,
            text_compared=True,
            block_outlines=[block.polygon for block in blocks] if block_alignments else [],
            block_alignments=[(a.block.polygon, a.transform) for a in block_alignments if a.locally_aligned],
            compare_mask=reported_compare_mask,
            notes=found.notes,
        )

    braille_note = ""
    braille_dots: np.ndarray | None = None
    if options.normalize_scan and detection_config.scan_ignore_braille:
        # A print carries Braille as embossing, not ink: a digital file's Braille dots (drawn over the artwork, often over
        # its type) are never in the scan, and are neither graphics missing from it nor letters to read. Where they are
        # is left out of the graphics compared, and filled in from round them in the pictures the words are read from --
        # only there: the pictures themselves are made as ever (their paper and ink are measured over the whole page,
        # and a page without its dots measured a little differently everywhere). The pictures shown keep them.
        with StageTimer(logger, "braille_left_out"):
            found = braille_diff.braille_mask(masked_original)
        if found is not None:
            braille_dots, count = found
            braille_note = (
                f"The digital file's Braille ({count} dots) was left out: a print carries Braille as embossing, not ink, "
                "so it is neither compared as graphics nor read as text."
            )

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
    shown_original = shown_suspect = None  # the ink pictures as the reviewer sees them (`scan_normalizer.InkPictures`)
    read_original = read_suspect = None  # what the words are read from, where that is not the ink picture
    other_original = other_suspect = None  # what a line that differs is read from as well
    if options.normalize_scan:
        say("Removing the background colour…")
        with StageTimer(logger, "ink_pictures"):
            # made together, so a coloured panel comes out the same way in both (see `ink_pair`)
            ink_original, ink_suspect, shown_original, shown_suspect = ink_pair(masked_original, masked_suspect, detection_config, compare_mask)
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
        if braille_dots is not None:
            compare_mask = compare_mask & ~braille_dots  # (from here on: the pictures above were made as ever)
            fill = cv2.dilate(braille_dots.astype(np.uint8), np.ones((3, 3), np.uint8))
            ink_original, ink_suspect, shown_original, shown_suspect = (
                cv2.inpaint(ink, fill, 3, cv2.INPAINT_TELEA) for ink in (ink_original, ink_suspect, shown_original, shown_suspect)
            )
            read_original, read_suspect, other_original, other_suspect = (
                None if picture is None else cv2.inpaint(picture, fill, 3, cv2.INPAINT_TELEA)
                for picture in (read_original, read_suspect, other_original, other_suspect)
            )

    used_regions = False
    # Graphics of files that do not register exactly are compared by where INK is, within
    # the measured registration error: the similarity map's noise filters, needed there,
    # erase thin real changes (a missing line of small print scored 0.07 IoU).
    misregistered = options.normalize_scan or profile != "exact"
    graphics_mode = detection_config.graphics_when_misregistered if misregistered else "ssim"
    # A scan is compared for BOTH graphics and text either way: `scan_compare_graphics`/`scan_compare_text` decide
    # only which are SHOWN by default (`graphics_wanted`/`text_wanted`, below). Detecting both always means the
    # results page's checkboxes are a display filter over `all_clusters` (`app.models.diff.visible_clusters`),
    # so ticking one back on is instant and nothing is compared again. A scan's graphics are noisy by default (it
    # changes every colour, blurs every edge and never registers exactly, and comparing the pictures gave dozens
    # of false 'graphics' differences -- fold lines, the scanner's own edge, every dark block -- round the few real
    # ones), which is why they start hidden; barcodes are compared as marks either way (below).
    graphics_detected = options.compare_graphics
    graphics_wanted = graphics_detected and (not options.normalize_scan or detection_config.scan_compare_graphics)
    text_detected = options.compare_text
    text_wanted = text_detected and (not options.normalize_scan or detection_config.scan_compare_text)
    local_wanted = detection_config.local_enabled and options.compare_color and options.compare_graphics and not options.normalize_scan
    local_only = local_wanted and not detection_config.legacy_pixel_detectors  # the full-size local comparison is the pixel detector
    use_structure = graphics_detected and graphics_mode in ("structure", "both") and not local_only
    use_ssim = graphics_detected and graphics_mode in ("ssim", "both") and not options.normalize_scan and not local_only
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
    structure_pool = ThreadPoolExecutor(max_workers=3)  # structure, local and (Scan vs Digital) barcode jobs, side by side
    structure_job = (
        structure_pool.submit(structure_diff.detect, masked_original, masked_suspect, detection_config, structure_slack, compare_mask)
        if use_structure
        else None
    )
    # The deep local comparison (full size, tile by tile) runs beside it: it is what finds a small real change that the
    # reduced-size detectors' tolerance hides, and what the findings' borders are fitted to.
    local_config = detection_config
    if local_wanted and detection_config.local_calibrate:
        # the tile comparison's bar, set from how much THIS pair differs where it prints the same (its noise)
        noise = local_diff.measure_noise(masked_original, masked_suspect, detection_config, compare_mask)
        local_config = local_diff.calibrated(detection_config, noise)
        logger.info("local comparison calibrated", extra={"noise": noise, "proof": local_config.local_proof_tolerance})
    local_job = (
        structure_pool.submit(local_diff.detect, masked_original, masked_suspect, local_config, compare_mask)
        if local_wanted
        else None
    )
    # (Scan vs Digital's barcodes, compared as marks, need nothing else found first: they are compared beside the rest)
    barcode_job = (
        structure_pool.submit(barcode_diff.compare_physical, masked_original, masked_suspect, detection_config, structure_slack)
        if options.normalize_scan
        else None
    )
    residual_mask: np.ndarray | None = None
    structure_collected = False
    same_text_boxes: list[BoundingBox] = []  # (Scan vs Digital) where the words read the same in both files

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
        if needs_pixels or text_detected:
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
            if text_detected and options.normalize_scan:
                try:
                    # (the digital file's own words, where it has any, at the reading's scale: exact where reading guesses)
                    full = ink_original.shape
                    sx, sy = small_original.shape[1] / full[1], small_original.shape[0] / full[0]
                    layer = [
                        replace(word, bbox=BoundingBox(word.bbox.x0 * sx, word.bbox.y0 * sy, word.bbox.x1 * sx, word.bbox.y1 * sy))
                        for word in (in_compared_area(_words_in_frame(original_words, detect_frame), compare_mask) or [])
                    ] or None
                    scan_text = text_diff.detect_scan(small_original, small_suspect, detection_config, small_read, small_other, layer)
                    working_candidates += scan_text.candidates
                    same_text_boxes = scan_text.same_text
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
            same_text_boxes = [
                BoundingBox(b.x0 / factor_x, b.y0 / factor_y, b.x1 / factor_x, b.y1 / factor_y) for b in same_text_boxes
            ]
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
        candidates += barcode_job.result()
        if detection_config.scan_colour_swaps:
            # Colour is not compared in a scan (scanning shifts every colour a little), but a solid area printed in a
            # different ink -- red in the file, teal in the print -- is not a shift: it is found by its hue.
            with StageTimer(logger, "colour_swaps"):
                candidates += _colour_swaps(masked_original, masked_suspect, compare_mask)
        # A print never carries the die line, nor what the digital file draws in its colour (cut and crease lines, notes
        # to the printer -- 'unvarnished area', a varnish-free box): missing from the scan, that is not a difference.
        candidates, non_printing = _drop_non_printing(candidates, original_image, original_boundary, warped_suspect)
        candidates, carton_edges = _drop_at_carton_edges(candidates, masked_original, compare_mask, ink_original, ink_suspect)
        candidates, over_text = _drop_over_same_text(candidates, same_text_boxes, original_image.shape[:2])
        candidates, residual = _drop_residual_ink(candidates, ink_original, ink_suspect, detection_config.scan_residual_ink_px)
        candidates = [_line_only(candidate) for candidate in candidates]
        if residual:
            logger.info("residual ink left out of the graphics", extra={"count": residual})
        if over_text:
            logger.info("graphics findings over unchanged text dropped", extra={"count": over_text})
        if carton_edges:
            logger.info("residue along the carton's edges dropped", extra={"count": carton_edges})
        if non_printing:
            logger.info("scan marks along the die line dropped", extra={"count": non_printing})

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
        if mode is ComparisonMode.LEAFLET and detection_config.nudge_tolerance_px > 0:
            # An open document compared with its closed (outlined) copy: every letter is re-drawn a pixel or so from
            # where it was, each by its own amount. Letters only moved like that are not a change.
            candidates, nudged = nudge.drop_nudged(
                candidates, masked_original, masked_suspect, detection_config.nudge_tolerance_px, detection_config.color_delta_e_threshold
            )
            if nudged:
                logger.info("moved-only differences dropped", extra={"count": nudged})

    say("Grouping the differences\u2026")
    clusters = build_clusters(candidates, detect_transform, config.clustering)
    if pixel_detectors and not options.normalize_scan:
        # the pieces of one line of text -- its letters, each a difference where the line was re-set -- are one finding,
        # read and described once
        # ...and the pieces of one photograph (a picture re-edited differs by the hundred, speck by speck) are one too.
        # (Photographs first: a photograph's own edges -- hair, a rim -- are line-shaped too, and are not text.)
        with StageTimer(logger, "lines_grouped"):
            pictures, text_lines = None, []
            if len(clusters) > 1 and (detection_config.group_text_lines or detection_config.group_pictures):
                every_line = _text_lines((masked_original, masked_suspect))
                pictures = _pictures((masked_original, masked_suspect), masked_original)
                text_lines = _outside_pictures(every_line, pictures)
            if detection_config.group_text_lines:
                clusters = _by_text_line(clusters, text_lines)
        if detection_config.group_pictures and pictures is not None:
            with StageTimer(logger, "pictures_grouped"):
                clusters = _by_picture(clusters, masked_original, masked_suspect, text_lines, pictures)
        if detection_config.match_moved_elements:
            # an element re-placed or resized (a logo a vendor re-laid) is one finding, not one at each of its edges
            clusters = elements.matched_elements(clusters, masked_original, masked_suspect, local_config)
    all_clusters = clusters
    if options.normalize_scan:
        # Both were just detected above regardless of the checkboxes: showing fewer of them is a filter,
        # not a smaller comparison, so a results-page toggle needs nothing recomputed (`visible_clusters`).
        clusters = renumbered(visible_clusters(clusters, text_wanted=text_wanted, graphics_wanted=graphics_wanted), all_clusters)
    if residual_mask is not None:
        _fit_borders(clusters, residual_mask, detect_transform, halo=1 + detection_config.local_forgive_px)

    if options.compare_text and not options.normalize_scan:
        # What a difference says is READ from its own box in each file, never found by comparing two readings of a page.
        say("Reading the words in each difference\u2026")
        try:
            if not text_diff.ocr_available():
                raise OcrUnavailableError(text_diff._OCR_UNAVAILABLE)
            layers = (
                in_compared_area(_words_in_frame(original_words, detect_frame), compare_mask),
                in_compared_area(_words_in_frame(_suspect_words_in_frame(suspect_words, alignment.transform, block_alignments), detect_frame), compare_mask),
            )
            _describe_clusters(clusters, masked_original, masked_suspect, detection_config, warnings, *layers)
        except OcrUnavailableError as exc:
            logger.warning("text description skipped: %s", exc)
            warnings.append(str(exc))

    if pixel_detectors and not options.normalize_scan:
        clusters = _mark_moves(clusters, masked_original, masked_suspect, detection_config)
        clusters = _consolidated(clusters, masked_original, masked_suspect)
        all_clusters = clusters

    if detect_frame is not None:
        candidates, clusters, all_clusters, symbol_comparisons, braille_comparison, shown_original, shown_suspect = _on_page(
            detect_frame, page_original.shape[:2], candidates, clusters, all_clusters, symbol_comparisons, braille_comparison,
            shown_original, shown_suspect,
        )
        original_image, warped_suspect = page_original, page_warped
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
        all_clusters=all_clusters,
        aligned_original=original_image,
        aligned_suspect=warped_suspect,
        symbol_comparisons=symbol_comparisons,
        braille=braille_comparison,
        warnings=warnings,
        detection_profile=profile,
        used_change_regions=used_regions,
        mode=mode,
        compared_outline=compared_area_outline(reported_compare_mask),
        # (each stroke cut where it is half its strength (`scan_normalizer.InkPictures`), so a scan's blurred type is shown
        # no bolder than the file's; once a whole picture was thinned instead, and thin strokes broke)
        ink_original=shown_original,
        ink_suspect=shown_suspect,
        graphics_compared=graphics_wanted,
        text_compared=text_wanted,
        block_outlines=[block.polygon for block in blocks] if block_alignments else [],
        block_alignments=[
            (alignment.block.polygon, alignment.transform) for alignment in block_alignments if alignment.locally_aligned
        ],
        compare_mask=reported_compare_mask if mode in (ComparisonMode.LEAFLET, ComparisonMode.AUTOMATIC) else None,
        notes=[braille_note] if braille_note else [],
    )


_FRAME_MARGIN_SHARE = 0.01  # the box detection works in is what is compared with this share of the page round it...
_FRAME_MARGIN_MIN_PX = 32
_FRAME_WORTH = 0.85  # ...and is used where it is less than this share of the page (otherwise the page is, as it is)


def _detection_frame(compare_mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) the box round what is compared, with a margin, that detection works in -- None where it is
    nearly the whole page (or nothing is compared)."""
    if not compare_mask.any():
        return None
    height, width = compare_mask.shape
    rows = np.flatnonzero(compare_mask.any(axis=1))
    columns = np.flatnonzero(compare_mask.any(axis=0))
    margin = max(_FRAME_MARGIN_MIN_PX, int(_FRAME_MARGIN_SHARE * max(height, width)))
    x0, x1 = max(0, int(columns[0]) - margin), min(width, int(columns[-1]) + 1 + margin)
    y0, y1 = max(0, int(rows[0]) - margin), min(height, int(rows[-1]) + 1 + margin)
    if (x1 - x0) * (y1 - y0) >= _FRAME_WORTH * height * width:
        return None
    return x0, y0, x1, y1


def _words_in_frame(words: list | None, frame: tuple[int, int, int, int] | None) -> list | None:
    """A file's own words (in the original's page pixels) in the detection box's pixels."""
    if words is None or frame is None:
        return words
    x0, y0 = frame[0], frame[1]
    return [replace(word, bbox=_box_moved(word.bbox, -x0, -y0)) for word in words]


def _box_moved(box: BoundingBox | None, dx: float, dy: float) -> BoundingBox | None:
    return None if box is None else BoundingBox(box.x0 + dx, box.y0 + dy, box.x1 + dx, box.y1 + dy)


def _on_page(frame, page_shape, candidates, clusters, all_clusters, symbols, braille, shown_original, shown_suspect):
    """What detection found in its box (`_detection_frame`), put back on the original's page: every box in the
    original's frame moved by the box's corner (a box in the suspect's own pixels is already there), the ink pictures
    laid on a blank page."""
    x0, y0 = frame[0], frame[1]
    moved: set[int] = set()

    def candidate(member: DiffCandidate) -> None:
        if id(member) in moved:
            return
        moved.add(id(member))
        member.bbox = _box_moved(member.bbox, x0, y0)
        member.moved_to = _box_moved(member.moved_to, x0, y0)

    for member in candidates:
        candidate(member)
    for cluster in [*clusters, *all_clusters]:
        if id(cluster) in moved:
            continue
        moved.add(id(cluster))
        cluster.bbox_in_original = _box_moved(cluster.bbox_in_original, x0, y0)
        cluster.moved_to = _box_moved(cluster.moved_to, x0, y0)
        for member in cluster.members:
            candidate(member)
    symbols = [
        replace(symbol, bbox=_box_moved(symbol.bbox, x0, y0), suspect_bbox=_box_moved(symbol.suspect_bbox, x0, y0))
        for symbol in symbols
    ]
    if braille is not None:
        # (the region, and each line and cell read: the Braille tab shows each of them where it is)
        lines = [
            replace(
                line,
                bbox=_box_moved(line.bbox, x0, y0),
                cells=[replace(cell, bbox=_box_moved(cell.bbox, x0, y0)) for cell in line.cells],
            )
            for line in braille.lines
        ]
        braille = replace(braille, region=_box_moved(braille.region, x0, y0), lines=lines)

    def on_blank(picture):
        if picture is None:
            return None
        page = np.full((*page_shape, *picture.shape[2:]), 255, picture.dtype)
        page[y0 : y0 + picture.shape[0], x0 : x0 + picture.shape[1]] = picture
        return page

    return candidates, clusters, all_clusters, symbols, braille, on_blank(shown_original), on_blank(shown_suspect)


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


_DIE_FINDING_SHARE = 0.8  # a finding is the die colour's if this share of the digital file's ink in it is
_PAPER_DISTANCE = 20.0  # a pixel this far (CIE76) from white is ink
_DIE_EDGE_MAX_THICKNESS_PX = 40  # a mark only the scan has, this thin at most (a cut edge's shadow, round a corner)...
_DIE_EDGE_REACH_PX = 8  # ...with the digital file's die line within this of it...
_DIE_EDGE_COVER = 0.7  # ...along this share of its length, is the carton's own edge


_lab = die_lines.lab


def _die_colour(image: np.ndarray, boundary: np.ndarray | None) -> np.ndarray | None:
    return die_lines.die_colour(image, boundary)


_SCAN_LINE_MAX_THICKNESS_PX = 28  # a scan's graphics finding this thin at most...
_SCAN_LINE_MIN_ASPECT = 6.0  # ...and this many times as long is a line


def _line_only(candidate: DiffCandidate) -> DiffCandidate:
    """A line gone from the scan, or only in it, is set apart as likely not a change: a cut or crease line the file
    draws is never printed, and a scan shows the carton's own cut edges, folds and creases. It is kept (a rule really
    missing from a print is possible), but listed apart and shown only on request, like a likely misreading."""
    box = candidate.bbox
    thin, long = min(box.width, box.height), max(box.width, box.height)
    if (
        candidate.type is not DiffType.GRAPHICS
        or candidate.subtype not in (DiffSubtype.REMOVED, DiffSubtype.ADDED)
        or thin > _SCAN_LINE_MAX_THICKNESS_PX
        or long < _SCAN_LINE_MIN_ASPECT * max(1.0, thin)
    ):
        return candidate
    why = (
        "likely a cut or fold line, which is not printed"
        if candidate.subtype is DiffSubtype.REMOVED
        else "likely the carton's own edge, fold or crease, seen by the scanner"
    )
    return replace(candidate, uncertain=True, confidence=min(candidate.confidence, text_diff.LIKELY_MISREAD_CONFIDENCE), detail=f"{candidate.detail} ({why})")


_EDGE_WORK_MIN_PX = 150  # the digital file's line-work: thin ink (no solid part) running at least this far either way...
_EDGE_WORK_SOLID_PX = 9  # ..."solid" being what survives an opening this wide (a panel, a bar, bold type)
_EDGE_WORK_INK = 40.0  # (CIE76) a pixel this far from white is ink here (a pale tint is not line-work)
_EDGE_MAX_THICKNESS_PX = 28  # a graphics finding this thin at most...
_EDGE_NEAREST_PX = 24  # ...coming within this of a cut or fold line or the artwork's outline (~2 mm at 300 DPI)...
_EDGE_ALONG_PX = 40  # ...and lying this close to it on average is residue of the carton's edge
# A finding of any size whose differing ink is only strokes -- a cut or fold line, a carton edge, a crease, a hairline where
# two edges fall a pixel apart -- is line-work too: thin and long (in the finding's box: an edge crossing a small box is a
# short piece of it), or a sliver (thin, however short)...
_EDGE_STROKE_MIN_PX = 40
_EDGE_STROKE_THICK_PX = 6.0
_EDGE_LONG_PX = 60  # (and one stroke at least this long is an edge's)
_EDGE_SLIVER_THICK_PX = 3.0
_EDGE_SPECK_PX = 30  # (a piece smaller than this is a speck: neither)
_EDGE_STROKES_SHARE = 0.7  # ...if strokes are this much of it, and what else there is is no more than
_EDGE_FEW_MARKS = 2  # ...this many pieces of its own (letters, blocks)...
_EDGE_MARKS_PX = 150  # ...with less ink than this in all. More marks than that are print beside the edge: kept, alone
_EDGE_MARK_PAD_PX = 4
_EDGE_ONE_SIDE = 0.9  # marks this much in the scan's ink alone are added (in the file's alone, missing)
# (measured: a carton's flap edge 88% strokes and one 58 px blob; a flap edge with '06 ATFAH' printed on it, 75% strokes
# and seven letters, 524 px; real print elsewhere 0-38% strokes)


def _apart_from_strokes(differs: np.ndarray) -> tuple[bool, tuple[int, int, int, int] | None]:
    """The ink two files differ by in a finding (`differs`, a mask) told apart from line-work (`_EDGE_*`): (it is line-work
    only, the box (x0, y0, x1, y1) of what else is there when that is print beside line-work -- a long stroke, a carton's
    edge, and several marks of their own: the finding is those -- or None)."""
    count, _labels, stats, _centres = cv2.connectedComponentsWithStats(differs.astype(np.uint8), connectivity=8)
    strokes, edge, marks = 0, False, []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if area < _EDGE_SPECK_PX:
            continue
        length = max(width, height)
        thickness = area / max(1, length)
        if (length >= _EDGE_STROKE_MIN_PX and thickness <= _EDGE_STROKE_THICK_PX) or thickness <= _EDGE_SLIVER_THICK_PX:
            strokes += area
            edge = edge or length >= _EDGE_LONG_PX
        else:
            marks.append((x, y, x + width, y + height, area))
    marked = sum(mark[4] for mark in marks)
    if not edge:
        return False, None
    if strokes >= _EDGE_STROKES_SHARE * (strokes + marked) and len(marks) <= _EDGE_FEW_MARKS and marked < _EDGE_MARKS_PX:
        return True, None
    if len(marks) <= _EDGE_FEW_MARKS:
        return False, None  # (a mark or two of real size beside it: left as it was found)
    pad = _EDGE_MARK_PAD_PX
    return False, (
        max(0, min(m[0] for m in marks) - pad),
        max(0, min(m[1] for m in marks) - pad),
        max(m[2] for m in marks) + pad,
        max(m[3] for m in marks) + pad,
    )


def _drop_at_carton_edges(
    candidates: list[DiffCandidate],
    original: np.ndarray,
    compare_mask: np.ndarray,
    ink_original: np.ndarray | None = None,
    ink_suspect: np.ndarray | None = None,
) -> tuple[list[DiffCandidate], int]:
    """Leave out what a scanned carton shows along its cut and fold lines and a digital file does not, or the other way
    round: the shadow of a cut edge or a fold, a crease, the file's own cut line (never printed), the bleed printed past a
    cut that the cut carton no longer has. A graphics finding no thicker than a line or a sliver, lying along the digital
    file's line-work (its cut and fold lines, whatever their colour: long thin ink) or the artwork's outline, is that.
    Real print near an edge -- a colour bar, a date wheel, a code -- is thicker, or not along it (measured: residue within
    0-21 px of the lines, real small print 30+). And a finding of any size -- a whole flap boxed for its edge -- where the
    ink the two ink pictures differ by is only line-work (`_apart_from_strokes`: a carton edge running diagonally has a wide box,
    though it is one thin stroke) is that residue too; real print differs by letters, blocks and symbols."""
    residue = set()
    if ink_original is not None and ink_suspect is not None:
        height, width = ink_original.shape[:2]
        reframed = []
        for candidate in candidates:
            if candidate.type is DiffType.GRAPHICS and not candidate.printers_note:
                b = candidate.bbox
                x0, y0 = max(0, int(b.x0)), max(0, int(b.y0))
                x1, y1 = min(width, int(np.ceil(b.x1)) + 1), min(height, int(np.ceil(b.y1)) + 1)
                if x1 > x0 and y1 > y0:
                    before, after = ink_original[y0:y1, x0:x1], ink_suspect[y0:y1, x0:x1]
                    only_lines, marks = _apart_from_strokes(cv2.absdiff(before, after) >= _RESIDUE_INK_LEVELS)
                    if only_lines:
                        residue.add(id(candidate))
                    elif marks is not None:  # print beside the edge (a code on a flap): it alone is the finding
                        mx0, my0, mx1, my1 = marks
                        was, now = before[my0:my1, mx0:mx1].astype(np.int16), after[my0:my1, mx0:mx1].astype(np.int16)
                        differs = np.abs(was - now) >= _RESIDUE_INK_LEVELS
                        darker = float((now < was)[differs].mean()) if differs.any() else 0.5
                        subtype, detail = candidate.subtype, candidate.detail
                        if darker >= _EDGE_ONE_SIDE:
                            subtype, detail = DiffSubtype.ADDED, "Graphic added in suspect"
                        elif darker <= 1.0 - _EDGE_ONE_SIDE:
                            subtype, detail = DiffSubtype.REMOVED, "Graphic missing from suspect"
                        candidate = replace(
                            candidate,
                            bbox=BoundingBox(x0 + mx0, y0 + my0, min(width, x0 + mx1), min(height, y0 + my1)),
                            subtype=subtype,
                            detail=detail,
                        )
            reframed.append(candidate)
        candidates = reframed
    small = [
        c for c in candidates
        if c.type is DiffType.GRAPHICS and min(c.bbox.width, c.bbox.height) <= _EDGE_MAX_THICKNESS_PX and not c.printers_note
        and id(c) not in residue
    ]
    if not small:
        return [c for c in candidates if id(c) not in residue], len(residue)
    lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).astype(np.float32)
    ink = (np.linalg.norm(lab - np.array([255.0, 128.0, 128.0], np.float32), axis=2) > _EDGE_WORK_INK).astype(np.uint8)
    del lab
    solid = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (_EDGE_WORK_SOLID_PX,) * 2))
    thin = ink & (1 - cv2.dilate(solid, np.ones((3, 3), np.uint8)))
    count, labels, stats, _centres = cv2.connectedComponentsWithStats(thin, connectivity=8)
    long = np.zeros(count, bool)
    long[1:] = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]) >= _EDGE_WORK_MIN_PX
    edges = long[labels].astype(np.uint8)
    del labels, thin, solid, ink
    outline, _hierarchy = cv2.findContours(compare_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(edges, outline, -1, 1, 1)
    distance = cv2.distanceTransform(1 - edges, cv2.DIST_L2, 5)
    height, width = distance.shape
    for candidate in small:
        b = candidate.bbox
        x0, y0 = max(0, int(b.x0)), max(0, int(b.y0))
        x1, y1 = min(width, int(np.ceil(b.x1)) + 1), min(height, int(np.ceil(b.y1)) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        near = distance[y0:y1, x0:x1]
        if float(near.min()) <= _EDGE_NEAREST_PX and float(near.mean()) <= _EDGE_ALONG_PX:
            residue.add(id(candidate))
    return [c for c in candidates if id(c) not in residue], len(residue)


_SAME_TEXT_SCALE = 4  # where the text reads the same is marked on a copy this many times smaller
_SAME_TEXT_REACH_PX = 6  # ...reaching this far round each word (a scan's letters spread a little)
_SAME_TEXT_COVER = 0.6  # a graphics finding lying this much over text that reads the same is that text, as scanned


_SWAP_SCALE = 0.25  # hues are compared on a copy this share of the size
_SWAP_MIN_CHROMA = 15.0  # (Lab) both files' colour at least this saturated: not paper (a scan's is <= ~10), grey or black
_SWAP_MIN_HUE_DEG = 60.0  # hues further apart than this are different inks (a scan shifts a hue by far less)
_SWAP_EDGE_PX = 8  # (full size) the edges of an area are not judged: misregistration, anti-aliasing, a scan's fringes
_SWAP_MIN_AREA_PX = 900  # (full size, ~2.5 x 2.5 mm at 300 DPI) a solid area at least this big: not a letter's stroke
_HUE_NAMES = ((15, "red"), (40, "orange"), (70, "yellow"), (160, "green"), (200, "teal"), (250, "blue"), (290, "purple"), (330, "magenta"), (360, "red"))


def _hue_name(rgb: np.ndarray) -> str:
    """A plain name for a colour (its hue; a pale red or magenta is pink), for a reviewer to read."""
    hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV_FULL)[0, 0]
    hue = float(hsv[0]) * 360.0 / 256.0
    name = next(name for limit, name in _HUE_NAMES if hue < limit)
    if name in ("red", "magenta") and hsv[1] < 110 and hsv[2] > 170:
        return "pink"
    return name


def _colour_swaps(original: np.ndarray, suspect: np.ndarray, compare_mask: np.ndarray) -> list[DiffCandidate]:
    """Solid areas printed in a different colour in the two files -- where both are clearly coloured (not paper, grey or
    a pale tint) and their hues are far apart -- as colour findings, saying which colours. Scan vs Digital only: every
    other colour difference there is how scanning reproduced it."""
    height, width = original.shape[:2]
    size = (max(1, round(width * _SWAP_SCALE)), max(1, round(height * _SWAP_SCALE)))
    labs = [cv2.cvtColor(cv2.resize(image, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2LAB).astype(np.float32) for image in (original, suspect)]
    chromas, hues = [], []
    for lab in labs:
        a, b = lab[..., 1] - 128.0, lab[..., 2] - 128.0
        chromas.append(np.hypot(a, b))
        hues.append(np.degrees(np.arctan2(b, a)))
    apart = np.abs(hues[0] - hues[1])
    apart = np.minimum(apart, 360.0 - apart)
    valid = cv2.resize(compare_mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    swapped = (chromas[0] >= _SWAP_MIN_CHROMA) & (chromas[1] >= _SWAP_MIN_CHROMA) & (apart >= _SWAP_MIN_HUE_DEG) & valid
    edge = max(1, round(_SWAP_EDGE_PX * _SWAP_SCALE))
    swapped = cv2.erode(swapped.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (2 * edge + 1, 2 * edge + 1)))
    count, labels, stats, _centres = cv2.connectedComponentsWithStats(swapped, connectivity=8)
    found = []
    small_images = [cv2.resize(image, size, interpolation=cv2.INTER_AREA) for image in (original, suspect)]
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area * (1.0 / _SWAP_SCALE) ** 2 < _SWAP_MIN_AREA_PX:
            continue
        pixels = labels == label
        was, now = (_hue_name(np.median(image[pixels], axis=0).astype(np.uint8)) for image in small_images)
        scale = 1.0 / _SWAP_SCALE
        box = BoundingBox((x - edge) * scale, (y - edge) * scale, (x + w + edge) * scale, (y + h + edge) * scale)
        detail = f"Colour changed: {was} in the digital file, {now} in the scan" if was != now else "Colour changed (a different ink)"
        found.append(DiffCandidate(box, DiffType.COLOUR, DiffSubtype.CHANGED, 1.0, detail, magnitude=float(np.median(apart[labels == label]))))
    return found


_RESIDUE_INK_LEVELS = 128  # the two ink pictures differ at a pixel by at least this much: one has ink there, one not
_RESIDUE_TOTAL_SHARE = 2.5  # residue is little ink in all too: less than this many times the largest-piece limit
# (measured: specks and slivers total <= 73 px; a date wheel of 19 tiny digits, 266)


def _drop_residual_ink(
    candidates: list[DiffCandidate], ink_original: np.ndarray | None, ink_suspect: np.ndarray | None, largest_px: int
) -> tuple[list[DiffCandidate], int]:
    """Leave out graphics findings that are only residual ink -- a scan's dust, a speck of toner, a hair of bleed along a
    stroke, a sliver where the two pictures' edges fall a pixel apart: the ink the two files differ by there comes in no
    piece bigger than `largest_px` (at the pictures' resolution), and is little in all. A mark that is really printed differently -- a symbol
    added, part of a logo missing -- is a solid piece of ink well above it. Text findings are never left out so: a full
    stop or a decimal point missing is small and still a change."""
    if ink_original is None or ink_suspect is None or largest_px <= 0:
        return candidates, 0
    height, width = ink_original.shape[:2]
    kept, dropped = [], 0
    for candidate in candidates:
        if candidate.type is DiffType.GRAPHICS:
            b = candidate.bbox
            x0, y0 = max(0, int(b.x0)), max(0, int(b.y0))
            x1, y1 = min(width, int(np.ceil(b.x1)) + 1), min(height, int(np.ceil(b.y1)) + 1)
            if x1 > x0 and y1 > y0:
                differs = cv2.absdiff(ink_original[y0:y1, x0:x1], ink_suspect[y0:y1, x0:x1]) >= _RESIDUE_INK_LEVELS
                count, _labels, stats, _centres = cv2.connectedComponentsWithStats(differs.astype(np.uint8), connectivity=8)
                largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
                total = int(stats[1:, cv2.CC_STAT_AREA].sum()) if count > 1 else 0
                # (small print -- a date wheel's digits, a batch code -- is small pieces too, but many: ink enough in all)
                if largest < largest_px and total < _RESIDUE_TOTAL_SHARE * largest_px:
                    dropped += 1
                    continue
        kept.append(candidate)
    return kept, dropped


def _drop_over_same_text(
    candidates: list[DiffCandidate], same_text: list[BoundingBox], shape: tuple[int, int]
) -> tuple[list[DiffCandidate], int]:
    """Leave out graphics findings that are text read the SAME in both files (Scan vs Digital compares text as text):
    a word printed a little bolder, softer or wider in the scan than drawn in the file is how it scanned, not a change."""
    graphics = [c for c in candidates if c.type is DiffType.GRAPHICS]
    if not same_text or not graphics:
        return candidates, 0
    scale = _SAME_TEXT_SCALE
    mask = np.zeros((shape[0] // scale + 1, shape[1] // scale + 1), np.uint8)
    reach = _SAME_TEXT_REACH_PX
    for box in same_text:
        cv2.rectangle(mask, (int((box.x0 - reach) / scale), int((box.y0 - reach) / scale)), (int((box.x1 + reach) / scale), int((box.y1 + reach) / scale)), 1, -1)
    integral = cv2.integral(mask)
    kept, dropped = [], 0
    for candidate in candidates:
        if candidate.type is DiffType.GRAPHICS:
            b = candidate.bbox
            x0, y0 = max(0, int(b.x0 / scale)), max(0, int(b.y0 / scale))
            x1, y1 = min(mask.shape[1], max(x0 + 1, int(np.ceil(b.x1 / scale)))), min(mask.shape[0], max(y0 + 1, int(np.ceil(b.y1 / scale))))
            covered = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
            if covered >= _SAME_TEXT_COVER * max(1, (x1 - x0) * (y1 - y0)):
                dropped += 1
                continue
        kept.append(candidate)
    return kept, dropped


_die_tinted = die_lines.die_tinted


def _drop_non_printing(
    candidates: list[DiffCandidate], original: np.ndarray, boundary: np.ndarray | None, suspect: np.ndarray | None = None
) -> tuple[list[DiffCandidate], int]:
    """Set apart what the digital file has and a print cannot (`_die_colour`): something missing from (or changed in) the
    scan where nearly all the digital file's ink is the die line's colour -- a note to the printer, kept as one
    (`DiffCandidate.printers_note`: listed only on request) -- and leave out a thin mark only the scan has, lying along
    the die line (the carton's own cut or creased edge, which the scanner sees; the count returned). But where the scan
    (`suspect`) prints something there of its own -- a colour bar beside a note to the printer, in one finding with it --
    that is kept as a finding, framed round what the scan prints."""
    colour = _die_colour(original, boundary)
    if colour is None:
        return candidates, 0
    kept, dropped = [], 0
    height, width = original.shape[:2]
    for candidate in candidates:
        if candidate.type in (DiffType.BARCODE, DiffType.BRAILLE) or candidate.subtype is DiffSubtype.MOVED:
            kept.append(candidate)
            continue
        box = candidate.bbox
        if candidate.subtype is DiffSubtype.ADDED:
            if candidate.type is not DiffType.GRAPHICS or min(box.width, box.height) > _DIE_EDGE_MAX_THICKNESS_PX:
                kept.append(candidate)
                continue
            pad = _DIE_EDGE_REACH_PX
        else:
            pad = 0
        x0, y0 = max(0, int(box.x0) - pad), max(0, int(box.y0) - pad)
        x1, y1 = min(width, int(np.ceil(box.x1)) + 1 + pad), min(height, int(np.ceil(box.y1)) + 1 + pad)
        crop = original[y0:y1, x0:x1]
        if crop.size == 0:
            kept.append(candidate)  # (a finding off the picture's edge: nothing there to tell it by)
            continue
        lab = _lab(crop)
        inked = np.linalg.norm(lab - np.array([100, 0, 0], np.float32), axis=1) > _PAPER_DISTANCE / 2
        die = _die_tinted(lab, colour) & inked
        if candidate.subtype is DiffSubtype.ADDED:
            # the scan's mark lies along the digital file's die line: the die line runs the length of the mark's box
            die_mask = die.reshape(crop.shape[:2])
            along_x = die_mask.any(axis=0).mean() if box.width >= box.height else die_mask.any(axis=1).mean()
            non_printing = along_x >= _DIE_EDGE_COVER
        else:
            non_printing = bool(inked.any()) and float(die.sum()) >= _DIE_FINDING_SHARE * float(inked.sum())
            if non_printing and suspect is not None:
                own = _printed_beside(suspect[y0:y1, x0:x1], die.reshape(crop.shape[:2]))
                if own is not None:
                    ox0, oy0, ox1, oy1 = own
                    kept.append(replace(candidate, bbox=BoundingBox(x0 + ox0, y0 + oy0, x0 + ox1, y0 + oy1), subtype=DiffSubtype.ADDED, detail="Graphic added in suspect"))
                    continue
            if non_printing:
                # a note to the printer, missing from the print as it always is: listed apart, on request
                kept.append(replace(candidate, printers_note=True, detail=f"{candidate.detail} {_PRINTERS_NOTE_DETAIL}".strip()))
                continue
        if non_printing:
            dropped += 1
        else:
            kept.append(candidate)
    return kept, dropped


_PRINTERS_NOTE_DETAIL = "(a note to the printer, in the die line's colour: never printed)"
_BESIDE_REACH_PX = 6  # the scan's ink this near the digital file's non-printing marks is theirs (registration)
_BESIDE_MIN_PX = 200  # ...and at least this much of it elsewhere in the finding is the scan's own print


_BESIDE_EDGE_PX = 6  # (the band round a finding's box where print is its neighbours')


def _printed_beside(scan_crop: np.ndarray, non_printing: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) in the crop of what the scan prints there away from the digital file's non-printing marks, or
    None if it prints nothing more than its paper there."""
    lab = _lab(scan_crop).reshape(scan_crop.shape[0], scan_crop.shape[1], 3)
    inked = np.linalg.norm(lab - np.array([100, 0, 0], np.float32), axis=2) > _PAPER_DISTANCE
    near = cv2.dilate(non_printing.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * _BESIDE_REACH_PX + 1,) * 2)) > 0
    own = inked & ~near
    # (print at the very edge of the box is a neighbour's -- a colour bar just below a note -- not something printed here)
    edge = _BESIDE_EDGE_PX
    own[:edge, :] = own[-edge:, :] = False
    own[:, :edge] = own[:, -edge:] = False
    if int(own.sum()) < _BESIDE_MIN_PX:
        return None
    ys, xs = np.nonzero(own)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


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


def _suspect_words_in_frame(words: list | None, transform: np.ndarray, block_alignments) -> list | None:
    """The suspect's own words (its pixels) carried into the original's frame the way its picture was: by the block's
    own transform where the block they lie in was aligned on its own, by the whole sheet's elsewhere."""
    pieces = [(a.block.polygon, a.transform) for a in (block_alignments or []) if a.locally_aligned]
    return suspect_words_in_frame(words, transform, pieces)


def suspect_words_in_frame(words: list | None, transform: np.ndarray, pieces) -> list | None:
    """`_suspect_words_in_frame` with the locally aligned blocks as `ComparisonOutcome.block_alignments` keeps them:
    (polygon in the original's pixels, suspect -> original transform) pairs."""
    if words is None:
        return None
    local = []
    for polygon, block_transform in pieces or []:
        outline = np.array(polygon, np.float32)
        local.append((outline, cv2.boundingRect(outline), block_transform))
    carried = []
    for word in words:
        b = word.bbox
        corners = np.array([[b.x0, b.y0], [b.x1, b.y0], [b.x1, b.y1], [b.x0, b.y1]], np.float64)
        moved = apply_transform(transform, corners)
        if local:
            cx, cy = moved.mean(axis=0)
            for outline, (bx, by, bw, bh), block_transform in local:
                if bx <= cx <= bx + bw and by <= cy <= by + bh and cv2.pointPolygonTest(outline, (float(cx), float(cy)), False) >= 0:
                    moved = apply_transform(block_transform, corners)
                    break
        (x0, y0), (x1, y1) = moved.min(axis=0), moved.max(axis=0)
        carried.append(text_diff.OcrWord(word.text, BoundingBox(float(x0), float(y0), float(x1), float(y1)), word.confidence, word.upright))
    return carried


def in_compared_area(words: list | None, compare_mask: np.ndarray) -> list | None:
    """The words whose middle lies in what is compared (the rest are blanked out of the pictures, so not there to read)."""
    if words is None:
        return None
    height, width = compare_mask.shape
    kept = []
    for word in words:
        cx, cy = int((word.bbox.x0 + word.bbox.x1) / 2), int((word.bbox.y0 + word.bbox.y1) / 2)
        if 0 <= cx < width and 0 <= cy < height and compare_mask[cy, cx]:
            kept.append(word)
    return kept


def _describe_clusters(
    clusters: list[DiffCluster],
    original: np.ndarray,
    suspect: np.ndarray,
    config: DetectionConfig,
    warnings: list[str],
    original_layer: list | None = None,
    suspect_layer: list | None = None,
) -> None:
    """Give each finding the text change its own box shows (its type and description follow)."""
    # (A recoloured area is a colour finding whatever is printed on it: the words on it are findings of their own.)
    # (Nor is an element found moved whole -- the same picture re-placed, `elements.matched_elements`: read at its old
    # place, the suspect there shows what moved in or out, which would make it a text change it is not.)
    describable = [
        cluster
        for cluster in clusters
        if not all(member.region for member in cluster.members)
        and not (cluster.moved_to is not None and all(member.subtype is DiffSubtype.MOVED for member in cluster.members))
    ]
    chosen = describable[: config.text_describe_max_findings]
    described_all = text_diff.describe([c.bbox_in_original for c in chosen], original, suspect, config, original_layer, suspect_layer)
    # In an element found moved but changed (`moved_to`), its words are also read where the element now is in the
    # suspect -- and whichever reading pairs words that are more alike is the one (a re-placed batch label read in place
    # pairs each line with the one above it: '150.00' -> '2022-03'; lined up, '2022-02' -> '2022-03'). Something in it
    # that moved its own way reads better in place, and is left so. (All read at once, over the suspect with each such
    # part laid back where it was: a reading costs little beside starting the reader.)
    went = [cluster for cluster in chosen if cluster.moved_to is not None]
    lined_up_all = dict(zip(map(id, went), _described_where_they_went(went, original, suspect, config))) if went else {}
    for cluster, described in zip(chosen, described_all):
        if cluster.moved_to is not None:
            lined_up = lined_up_all.get(id(cluster))
            if lined_up is None:
                if described is not None and described.was_text and described.now_text and described.was_text != described.now_text:
                    # the words read the same once lined up: the change read in place was only the element's move
                    described = None
            elif _pairing(lined_up) > (0.0 if described is None else _pairing(described)):
                described = lined_up  # (only a reading that pairs words: words only added or missing prove nothing)
        if described is not None:
            add_member(cluster, described)
    if len(describable) > len(chosen):
        warnings.append(
            f"There are {len(describable)} differences; the words were read in the first {len(chosen)} only, "
            "so the rest are described as artwork changes rather than text changes."
        )


def _pairing(described: DiffCandidate) -> float:
    """How alike the words a text change pairs are: 1 where both sides are the same words (a font, a size, a move), the
    likeness of the two where one was changed to the other ('2022-02' -> '2022-03'), 0 where nothing is paired (words
    only added or only missing)."""
    was, now = described.was_text, described.now_text
    if not (was and now):
        changed = _CHANGED_WORDS.search(described.detail)
        if changed:
            was, now = changed.group(1), changed.group(2)
        elif _QUOTED_WORDS.search(described.detail):
            return 0.0  # (added or missing)
        elif "'" in described.detail:
            return 1.0  # the same words, said of differently
        else:
            return 0.0
    return difflib.SequenceMatcher(None, was, now).ratio()


_CHANGED_WORDS = re.compile(r"'(.*)' -> '(.*)'")


def _described_where_they_went(
    clusters: list[DiffCluster], original: np.ndarray, suspect: np.ndarray, config: DetectionConfig
) -> list[DiffCandidate | None]:
    """The text change in each of `clusters`' boxes with the suspect read where that box's content now is (`moved_to`):
    the suspect's part brought back over the original's, so line meets line ('2019-03' -> '2019-04', not '2019-03'
    against the line above it, which is what lies at the old place). One reading for all: a copy of the suspect with
    each part laid back where it was."""
    height, width = original.shape[:2]
    lined_up = suspect.copy()
    # each element laid back once, its whole area (its parts share how it moved), so no part's surroundings are another's
    elements: dict[tuple, list[BoundingBox]] = {}
    for cluster in clusters:
        box, went = cluster.bbox_in_original, cluster.moved_to
        if box.width <= 0 or box.height <= 0:
            continue
        scale_x, scale_y = went.width / box.width, went.height / box.height
        # (a point x of the original is went.x0 + (x - box.x0) * scale_x in the suspect: x * scale_x + offset_x)
        how = (round(scale_x, 3), round(scale_y, 3), round(went.x0 - box.x0 * scale_x), round(went.y0 - box.y0 * scale_y))
        elements.setdefault(how, []).append(box)
    for (scale_x, scale_y, offset_x, offset_y), boxes in elements.items():
        pad = max(8.0, max(box.height for box in boxes))
        x0, y0 = max(0, int(min(b.x0 for b in boxes) - pad)), max(0, int(min(b.y0 for b in boxes) - pad))
        x1 = min(width, int(math.ceil(max(b.x1 for b in boxes) + pad)))
        y1 = min(height, int(math.ceil(max(b.y1 for b in boxes) + pad)))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        to_suspect = np.float32([[scale_x, 0, x0 * scale_x + offset_x], [0, scale_y, y0 * scale_y + offset_y]])
        lined_up[y0:y1, x0:x1] = cv2.warpAffine(
            suspect, to_suspect, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE
        )
    return text_diff.describe([cluster.bbox_in_original for cluster in clusters], original, lined_up, config)


_MOVE_MIN_PX = 6  # something that is this far from where it was, or further, has moved (nearer is the registration)
_MOVE_MAX_PX = 2400
_MOVE_SIZE_TOLERANCE = 0.12  # the two boxes' sides may differ by this much (the ink softens at an edge)
_MOVE_MIN_SIMILARITY = 0.8  # normalised correlation of what is in them
_MOVE_MIN_STRUCTURE = 10.0
_QUOTED_WORDS = re.compile(r"'(.*)' (?:missing from suspect|added in suspect)")


_INSIDE_SHARE = 0.9  # a finding with this much of its box inside another's lies inside it
_SAME_LINE_HEIGHTS = 2.5  # a text finding no taller than this many times another text finding inside it is one line with it
_ELEMENT_HOST_HEIGHTS = 3.0  # a changed graphic no taller than this many times what lies in it is one element with it
_UPRIGHT_ASPECT = 1.5  # a finding this many times taller than wide can be a piece of a word running up the page...
_UPRIGHT_OVERLAP = 0.6  # ...and two such, overlapping across by this share of the narrower...
_UPRIGHT_GAP = 0.35  # ...no further apart along than this share of their width, are one word's pieces
_RECOLOUR_TOLERANCE = 30.0  # (RGB distance) pixels this far apart differ
_RECOLOUR_MARK = 60.0  # (RGB distance) a pixel this far from its file's area colour is a mark on it
_RECOLOUR_SLACK_PX = 1  # marks this close in the two files are the same marks
_RECOLOUR_SAME_MARKS = 0.9  # a finding inside a recoloured area whose marks agree this well is part of the recolour


_LINE_SCALE = 0.5  # text lines are found on a copy this share of the size
_LINE_EDGE = 40  # (8-bit) a step this big between neighbouring pixels is the edge of a mark (dark on light or light on dark)
_LINE_JOIN_PX = 14  # (full size) marks this close along a line are one line (letters of a word, words of a line)
_LINE_MIN_PX = 8  # a line is this tall at least (full size)...
_LINE_MAX_PX = 140  # ...and no taller (a picture, a panel, a block of paragraphs is not a line)
_LINE_ELONGATION = 1.3  # ...and this many times longer than it is tall
_LINE_INSIDE = 0.45  # a finding with this share of its box in one line is a piece of it (a letter's box reaches
# above and below the line's, kept clear of its neighbours)
_LINE_MAX_FINDING_HEIGHTS = 1.6  # ...unless it is taller than this many times the line (a graphic the line runs through)
_LINE_RUN_GAP_HEIGHTS = 3.0  # pieces this many line-heights apart along the line, or further, are separate edits


def _text_lines(images: tuple[np.ndarray, ...]) -> list[tuple[float, float, float, float]]:
    """The lines of text in either picture (shared frame, full-size (x0, y0, x1, y1)), found from the pictures alone (no
    reading): marks close together along a row are words, line-shaped and the height of type; words side by side, their
    gap small for their height (a heading's words are further apart than body text's), are one line. Type set on its side
    is looked for down the columns only where no row of type is: a paragraph's letters are close down the page too, and
    taken for columns they tied every line of it into one. The same line in the two files (overlapping) is one line."""
    scale = _LINE_SCALE
    shape = (max(1, round(images[0].shape[0] * scale)), max(1, round(images[0].shape[1] * scale)))
    join = max(3, round(_LINE_JOIN_PX * scale)) | 1
    low, high = _LINE_MIN_PX * scale, _LINE_MAX_PX * scale
    found: list[tuple[float, float, float, float]] = []
    for image in images:
        gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        edges = (cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > _LINE_EDGE).astype(np.uint8)
        rows = np.zeros(shape, np.uint8)
        for along_x in (True, False):
            marks = edges if along_x else edges & (1 - cv2.dilate(rows, np.ones((5, 5), np.uint8)))
            kernel = (1, join) if along_x else (join, 1)
            joined = cv2.morphologyEx(marks, cv2.MORPH_CLOSE, np.ones(kernel, np.uint8))
            _count, _labels, stats, _centres = cv2.connectedComponentsWithStats(joined, connectivity=8)
            words = []
            for x, y, w, h, _area in stats[1:]:
                a0, c0, along, across = (x, y, w, h) if along_x else (y, x, h, w)
                if low <= across <= high and along >= _LINE_ELONGATION * across:
                    words.append([a0, c0, a0 + along, c0 + across])
            for a0, c0, a1, c1 in _joined_words(words):
                x0, y0, x1, y1 = (a0, c0, a1, c1) if along_x else (c0, a0, c1, a1)
                if along_x:
                    rows[y0:y1, x0:x1] = 1
                found.append((x0 / scale, y0 / scale, x1 / scale, y1 / scale))
    # the same line in both files (it may sit a little apart, or run a little longer, in one): one line
    merged: list[list[float]] = []
    for box in sorted(found, key=lambda b: (b[1], b[0])):
        for line in merged:
            if _same_line(line, box):
                line[0], line[1], line[2], line[3] = min(line[0], box[0]), min(line[1], box[1]), max(line[2], box[2]), max(line[3], box[3])
                break
        else:
            merged.append(list(box))
    return [tuple(line) for line in merged]


_SAME_LINE_OVERLAP = 0.6  # two lines overlapping this much (of the smaller, each way) are one


def _same_line(a, b) -> bool:
    across = min(a[3], b[3]) - max(a[1], b[1])
    along = min(a[2], b[2]) - max(a[0], b[0])
    if across <= 0 or along <= 0:
        return False
    return (
        across >= _SAME_LINE_OVERLAP * min(a[3] - a[1], b[3] - b[1])
        and along >= _SAME_LINE_OVERLAP * min(a[2] - a[0], b[2] - b[0])
    )


_PICTURE_SCALE = 0.25  # photographs are found on a copy this share of the size
_PICTURE_TONE = (3.0, 40.0)  # a gentle step between neighbours (a photograph's tones), not flat and not a printed edge
# (a panel shaded from one colour to another is gentle too: what is on it is kept out by being text, or too big a piece)
_PICTURE_WINDOW_PX = 60  # (full size) judged over a window this wide...
_PICTURE_TONAL_SHARE = 0.45  # ...where this share of it is such steps
_PICTURE_SHARP = 60.0  # a step this big is a printed edge (type, a rule)...
_PICTURE_MAX_SHARP_SHARE = 0.12  # ...and a window with more than this share of them is type (small print's soft
# letters are gentle steps too, but among sharp ones), not a photograph
_PICTURE_MIN_SIDE_PX = 150  # (full size) a photograph this big each way at least
_PICTURE_INSIDE = 0.6  # a finding with this share of its box in a photograph is a piece of it
_PICTURE_TONE_BIN = 16  # tones are counted in bands of this many grey levels
_PICTURE_TWO_TONES = 0.6  # a region with this much of it in its two commonest bands is not a photograph (measured:
# photographs 26-60% -- more where one lies on a flat background --, panels of type on a tint or a screen 62-91%)
_PICTURE_MIN_PIECES = 3  # ...and this many pieces of one photograph are one finding
_PICTURE_PIECE_SHARE = 0.5  # (a finding as big as this share of the photograph, or more, is not a piece of it)


def _pictures(images: tuple[np.ndarray, ...], tones_of: np.ndarray | None = None) -> np.ndarray:
    """Where the photographs are, in either picture (shared frame): an int32 label per pixel of a copy `_PICTURE_SCALE`
    the size, 0 elsewhere. A photograph is continuous tone -- most of it gentle steps between neighbouring pixels --
    where type, rules and flat colour are either flat or a sharp edge."""
    scale = _PICTURE_SCALE
    shape = (max(1, round(images[0].shape[0] * scale)), max(1, round(images[0].shape[1] * scale)))
    window = max(3, round(_PICTURE_WINDOW_PX * scale)) | 1
    found = np.zeros(shape, np.uint8)
    for image in images:
        gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
        step = np.maximum(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)), np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)))
        tonal = ((step >= _PICTURE_TONE[0]) & (step <= _PICTURE_TONE[1])).astype(np.float32)
        sharp = (step > _PICTURE_SHARP).astype(np.float32)
        share = cv2.boxFilter(tonal, -1, (window, window))
        sharp_share = cv2.boxFilter(sharp, -1, (window, window))
        found |= ((share >= _PICTURE_TONAL_SHARE) & (sharp_share <= _PICTURE_MAX_SHARP_SHARE)).astype(np.uint8)
    found = cv2.morphologyEx(found, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    found = cv2.morphologyEx(found, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))  # (a photograph's flat patches are in it)
    count, labels, stats, _centres = cv2.connectedComponentsWithStats(found, connectivity=8)
    side = _PICTURE_MIN_SIDE_PX * scale
    keep = np.zeros(count, np.int32)
    tones = None
    if tones_of is not None:  # (a panel of type on a tint is two tones, the tint and the ink; a photograph is many)
        tones = cv2.resize(cv2.cvtColor(tones_of, cv2.COLOR_RGB2GRAY), (shape[1], shape[0]), interpolation=cv2.INTER_AREA) // _PICTURE_TONE_BIN
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_WIDTH] < side or stats[label, cv2.CC_STAT_HEIGHT] < side:
            continue
        if tones is not None:
            x, y, w, h = (int(v) for v in stats[label, :4])
            inside = tones[y : y + h, x : x + w][labels[y : y + h, x : x + w] == label]
            counts = np.sort(np.bincount(inside.ravel(), minlength=256 // _PICTURE_TONE_BIN))[::-1]
            if counts[:2].sum() >= _PICTURE_TWO_TONES * max(1, inside.size):
                continue  # two tones: a flat (or screened) panel with type on it, not a photograph
        keep[label] = label
    return keep[labels]


def _outside_pictures(lines: list[tuple[float, float, float, float]], pictures: np.ndarray | None) -> list:
    """The lines of text not in a photograph (mostly)."""
    if pictures is None or not pictures.any():
        return lines
    scale = _PICTURE_SCALE
    kept = []
    for x0, y0, x1, y1 in lines:
        a0, b0 = max(0, int(x0 * scale)), max(0, int(y0 * scale))
        a1, b1 = min(pictures.shape[1], int(math.ceil(x1 * scale)) + 1), min(pictures.shape[0], int(math.ceil(y1 * scale)) + 1)
        window = pictures[b0:b1, a0:a1]
        if not window.size or (window > 0).mean() < _PICTURE_LINE_SHARE:
            kept.append((x0, y0, x1, y1))
    return kept


_PICTURE_LINE_SHARE = 0.5  # a 'line' this much inside a photograph is its own edges, not text
def _by_picture(
    clusters: list[DiffCluster],
    original: np.ndarray,
    suspect: np.ndarray,
    lines: list[tuple[float, float, float, float]] = (),
    labels: np.ndarray | None = None,
) -> list[DiffCluster]:
    """The graphics and colour findings inside one photograph, where there are several, made one finding: that the
    picture changed. Text on it stays its own -- what lies on a line of text is left out, read later as the words it is --
    and so does an area recoloured round the photograph (it is bigger than a piece of it)."""
    boxes = np.array(lines, np.float32).reshape(-1, 4)
    candidates = [
        c
        for c in clusters
        if c.category in ("graphics", "colour")
        and not all(m.printers_note for m in c.members)
        and _line_of(c.bbox_in_original, boxes) is None
    ]
    if len(candidates) < _PICTURE_MIN_PIECES:
        return clusters
    if labels is None:
        labels = _pictures((original, suspect))
    if not labels.any():
        return clusters
    scale = _PICTURE_SCALE
    groups: dict[int, list[DiffCluster]] = {}
    for cluster in candidates:
        b = cluster.bbox_in_original
        x0, y0 = max(0, int(b.x0 * scale)), max(0, int(b.y0 * scale))
        x1, y1 = min(labels.shape[1], int(math.ceil(b.x1 * scale)) + 1), min(labels.shape[0], int(math.ceil(b.y1 * scale)) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        found = labels[y0:y1, x0:x1].ravel()
        found = found[found > 0]
        if not found.size:
            continue
        picture = int(np.bincount(found).argmax())
        size = int((labels == picture).sum())
        if (found == picture).sum() >= _PICTURE_INSIDE * (x1 - x0) * (y1 - y0) and (x1 - x0) * (y1 - y0) <= _PICTURE_PIECE_SHARE * size:
            groups.setdefault(picture, []).append(cluster)
    replaced: dict[int, DiffCluster] = {}
    absorbed: set[int] = set()
    for pieces in groups.values():
        if len(pieces) < _PICTURE_MIN_PIECES:
            continue
        one = combined(pieces)
        one.type = DiffType.GRAPHICS
        one.detail = f"Picture changed ({len(pieces)} areas of it differ)"
        replaced[id(pieces[0])] = one
        absorbed.update(id(c) for c in pieces[1:])
    if not replaced:
        return clusters
    out = [replaced.get(id(c), c) for c in clusters if id(c) not in absorbed]
    for number, cluster in enumerate(out, start=1):
        cluster.id = number
    return out


_WORD_GAP_HEIGHTS = 1.0  # words this close along (a share of their height) are one line...
_WORD_SAME_ROW = 0.6  # ...if they overlap across by this share of the shorter


def _joined_words(words: list[list[int]]) -> list[list[int]]:
    """Word boxes ([along0, across0, along1, across1]) joined into lines: side by side, on one row, the gap between them
    no more than their height."""
    words = sorted(words, key=lambda b: (b[1], b[0]))
    lines: list[list[int]] = []
    for box in sorted(words, key=lambda b: b[0]):
        height = box[3] - box[1]
        for line in lines:
            overlap = min(line[3], box[3]) - max(line[1], box[1])
            shorter = min(line[3] - line[1], height)
            gap = box[0] - line[2]
            if overlap >= _WORD_SAME_ROW * shorter and -height <= gap <= _WORD_GAP_HEIGHTS * max(height, line[3] - line[1]):
                line[0], line[1] = min(line[0], box[0]), min(line[1], box[1])
                line[2], line[3] = max(line[2], box[2]), max(line[3], box[3])
                break
        else:
            lines.append(list(box))
    return lines


def _line_of(b: BoundingBox, boxes: np.ndarray) -> int | None:
    """The line of text (a row of `boxes`) a finding's box is a piece of -- mostly in it, and no taller -- or None."""
    if not len(boxes):
        return None
    area = max(1.0, b.width * b.height)
    overlap = (np.clip(np.minimum(boxes[:, 2], b.x1) - np.maximum(boxes[:, 0], b.x0), 0, None)
               * np.clip(np.minimum(boxes[:, 3], b.y1) - np.maximum(boxes[:, 1], b.y0), 0, None))
    line = int(np.argmax(overlap))
    height = float(boxes[line, 3] - boxes[line, 1])
    if overlap[line] >= _LINE_INSIDE * area and b.height <= _LINE_MAX_FINDING_HEIGHTS * height:
        return line
    return None


def _by_text_line(clusters: list[DiffCluster], lines: list[tuple[float, float, float, float]]) -> list[DiffCluster]:
    """Findings that are pieces of one line of text -- in either file -- made one finding. A line re-set (a new size, a
    little along) differs by every letter, and each letter was a finding of its own ('Graphic moved 613 px left' for the
    'a' of 'Sugammadex'); a line is what a reviewer reads, and what is described once, by its words. Only what lies in
    the line and is no taller than it: a graphic or an area the line runs through is its own finding."""
    if len(clusters) < 2 or not lines:
        return clusters
    boxes = np.array(lines, np.float32)
    groups: dict[int, list[DiffCluster]] = {}
    for cluster in clusters:
        if cluster.category not in ("text", "graphics", "colour") or all(m.region or m.printers_note for m in cluster.members):
            continue
        b = cluster.bbox_in_original
        area = max(1.0, b.width * b.height)
        overlap = (np.clip(np.minimum(boxes[:, 2], b.x1) - np.maximum(boxes[:, 0], b.x0), 0, None)
                   * np.clip(np.minimum(boxes[:, 3], b.y1) - np.maximum(boxes[:, 1], b.y0), 0, None))
        line = _line_of(b, boxes)
        if line is not None:
            groups.setdefault(line, []).append(cluster)
    replaced: dict[int, DiffCluster] = {}
    absorbed: set[int] = set()
    for line, members in groups.items():
        x0, y0, x1, y1 = boxes[line]
        across = min(x1 - x0, y1 - y0)
        along_x = (x1 - x0) >= (y1 - y0)
        # (pieces of one re-set stretch lie letter by letter; two edits far apart on one long line are two findings)
        runs: list[list[DiffCluster]] = []
        for cluster in sorted(members, key=lambda c: c.bbox_in_original.x0 if along_x else c.bbox_in_original.y0):
            b = cluster.bbox_in_original
            if runs:
                last = runs[-1]
                reach = max(c.bbox_in_original.x1 if along_x else c.bbox_in_original.y1 for c in last)
                if (b.x0 if along_x else b.y0) - reach <= _LINE_RUN_GAP_HEIGHTS * across:
                    last.append(cluster)
                    continue
            runs.append([cluster])
        for pieces in runs:
            if len(pieces) < 2:
                continue
            first = min(pieces, key=lambda c: c.id)
            replaced[id(first)] = combined(pieces)
            absorbed.update(id(c) for c in pieces if c is not first)
    if not replaced:
        return clusters
    out = [replaced.get(id(c), c) for c in clusters if id(c) not in absorbed]
    for number, cluster in enumerate(out, start=1):
        cluster.id = number
    return out


_DIE_WARP_SLACK_PX = 4  # the suspect's die lines, brought into the original's frame, are widened by this much more


def _die_lines_to_ignore(
    original_image: np.ndarray,
    suspect_image: np.ndarray,
    original_area: "_Area",
    suspect_area: "_Area",
    original_strokes: list | None,
    suspect_strokes: list | None,
    transform: np.ndarray,
    reference_shape: tuple[int, int],
) -> np.ndarray | None:
    """Where either file's die lines are, in the original's pixels (see `die_lines`): None if neither has any to tell."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        found_original, found_suspect = pool.map(
            lambda job: die_lines.die_line_mask(*job),
            [(original_image, original_area.boundary, original_strokes), (suspect_image, suspect_area.boundary, suspect_strokes)],
        )
    ignore = None
    if found_original is not None:
        ignore = found_original
    if found_suspect is not None:
        moved = _warped(found_suspect, transform, reference_shape)
        size = 2 * _DIE_WARP_SLACK_PX + 1
        moved = cv2.dilate(moved.astype(np.uint8), np.ones((size, size), np.uint8)).astype(bool)
        ignore = moved if ignore is None else ignore | moved
    if ignore is None:
        return None
    # the same line drawn in another colour in one of the files (a solid red outline where the other has its pink die
    # line) runs along the die lines found: it is a die line too
    also = die_lines.strokes_along(original_strokes, ignore)
    if suspect_strokes:
        back = _warped(ignore, np.linalg.inv(transform), suspect_image.shape[:2])
        along_suspect = die_lines.strokes_along(suspect_strokes, back)
        if along_suspect is not None:
            moved = _warped(along_suspect, transform, reference_shape)
            also = moved if also is None else also | moved
    if also is not None:
        ignore = ignore | also
    logger.info("die lines left out", extra={"pixels": int(ignore.sum())})
    return ignore


def _consolidated(clusters: list[DiffCluster], original: np.ndarray, suspect: np.ndarray) -> list[DiffCluster]:
    """One finding for one changed thing, once the findings are described (what they are is known only then):

    * what lies inside a word or line whose text changed, moved or is set differently is part of that change (the hyphen
      of 'eco-pack' gone from inside 'eco-pack' -> 'ecopack'; a piece of a letter inside 'MATRIX' -> 'MATRX');
    * so is what lies inside a changed graphic only a few times its size (one badge, one mark);
    * and what lies inside a recoloured area, where its own pixels changed by that same recolour (a letter whose
      counter shows the panel's new colour is not a letter that changed) -- but not a symbol removed from the panel;
    * two pieces of one word running up the page (a flap's 'MATRIX', found top half and bottom half) are one.
    """
    alive = [cluster for cluster in clusters if cluster.members]
    order = sorted(range(len(alive)), key=lambda k: -alive[k].bbox_in_original.area)
    gone: set[int] = set()
    recolours: dict = {}  # (a recoloured area's old and new colour, worked out once however many findings lie in it)
    for i in order:  # the biggest first: what lies inside it is looked at while it is still whole
        if i in gone:
            continue
        outer = alive[i]
        for j in range(len(alive)):
            if j == i or j in gone:
                continue
            inner = alive[j]
            if inner.bbox_in_original.area >= outer.bbox_in_original.area or not _lies_inside(inner.bbox_in_original, outer.bbox_in_original):
                continue
            if _part_of(outer, inner, original, suspect, recolours):
                _absorb(outer, inner)
                gone.add(j)
    alive = [cluster for k, cluster in enumerate(alive) if k not in gone]
    # pieces of one word running up the page
    changed = True
    while changed:
        changed = False
        for i in range(len(alive)):
            for j in range(i + 1, len(alive)):
                if _one_upright_word(alive[i], alive[j]):
                    first, second = alive[i], alive[j]
                    keep, other = (first, second) if _text_first(first, second) else (second, first)
                    _absorb(keep, other)
                    alive = [cluster for cluster in alive if cluster is not other]
                    changed = True
                    break
            if changed:
                break
    for number, cluster in enumerate(sorted(alive, key=lambda c: (c.bbox_in_original.y0, c.bbox_in_original.x0)), start=1):
        cluster.id = number
    return sorted(alive, key=lambda c: c.id)


def _lies_inside(inner: BoundingBox, outer: BoundingBox) -> bool:
    return inner.intersection_area(outer) >= _INSIDE_SHARE * max(1.0, inner.area)


def _absorb(outer: DiffCluster, inner: DiffCluster) -> None:
    """`inner`'s members into `outer`, which keeps what it says (it is the finding the reviewer reads)."""
    outer.members.extend(inner.members)
    outer.bbox_in_original = outer.bbox_in_original.union(inner.bbox_in_original)
    outer.bbox_in_suspect = outer.bbox_in_suspect.union(inner.bbox_in_suspect)


def _text_first(a: DiffCluster, b: DiffCluster) -> bool:
    """Whether `a` should stand for the two (a finding that says what the words are, before one that does not)."""
    return (a.category == "text", a.bbox_in_original.area) >= (b.category == "text", b.bbox_in_original.area)


def _part_of(outer: DiffCluster, inner: DiffCluster, original: np.ndarray, suspect: np.ndarray, recolours: dict | None = None) -> bool:
    if outer.subtype is DiffSubtype.MOVED and inner.subtype is not DiffSubtype.MOVED and outer.category != "text":
        return False  # a graphic that moved, and something that changed where it now is, are two things
    if outer.category == "text":
        # a piece of the words' own pixels is part of them; another reading of words inside a rewritten paragraph is a
        # change of its own unless it is on the same line (a word of the phrase found twice)
        return inner.category != "text" or outer.bbox_in_original.height <= _SAME_LINE_HEIGHTS * max(1.0, inner.bbox_in_original.height)
    if outer.category == "graphics" and outer.subtype is DiffSubtype.CHANGED:
        return outer.bbox_in_original.height <= _ELEMENT_HOST_HEIGHTS * max(1.0, inner.bbox_in_original.height)
    if outer.category == "colour":
        if any(member.type in (DiffType.TEXT, DiffType.NUMBER) and member.subtype is not DiffSubtype.MOVED
               and "font" not in member.detail.lower() for member in inner.members):
            return False  # words that changed on a recoloured panel are a change of their own, whatever the panel did
        return _same_recolour(outer.bbox_in_original, inner.bbox_in_original, original, suspect, recolours)
    return False


_RECOLOUR_SAMPLE_PX = 250_000  # an area's old and new colour are judged from at most about this many of its pixels


def _same_recolour(
    outer: BoundingBox, inner: BoundingBox, original: np.ndarray, suspect: np.ndarray, recolours: dict | None = None
) -> bool:
    """Whether `inner` holds the same marks in both files, only over the recoloured area's old colour in one and its new
    colour in the other: then nothing of its own changed (a letter in a panel that went from grey to red), and it is part
    of the recolour. A symbol removed from the area, or a digit changed on it, leaves marks in one file only."""
    height, width = original.shape[:2]

    def crop(image: np.ndarray, box: BoundingBox) -> np.ndarray:
        x0, y0 = max(0, int(box.x0)), max(0, int(box.y0))
        x1, y1 = min(width, int(np.ceil(box.x1))), min(height, int(np.ceil(box.y1)))
        return image[y0:y1, x0:x1].astype(np.float32)

    key = (outer.x0, outer.y0, outer.x1, outer.y1)
    if recolours is not None and key in recolours:
        colours = recolours[key]
    else:
        x0, y0 = max(0, int(outer.x0)), max(0, int(outer.y0))
        x1, y1 = min(width, int(np.ceil(outer.x1))), min(height, int(np.ceil(outer.y1)))
        step = max(1, int(math.ceil(math.sqrt(max(1, (x1 - x0) * (y1 - y0)) / _RECOLOUR_SAMPLE_PX))))
        a = original[y0:y1:step, x0:x1:step].reshape(-1, 3).astype(np.float32)
        b = suspect[y0:y1:step, x0:x1:step].reshape(-1, 3).astype(np.float32)
        differing = np.linalg.norm(a - b, axis=1) > _RECOLOUR_TOLERANCE if len(a) else np.zeros(0, bool)
        colours = None
        if differing.sum() * step * step >= 50:
            colours = np.median(a[differing], axis=0), np.median(b[differing], axis=0)
        if recolours is not None:
            recolours[key] = colours
    if colours is None:
        return False
    old_colour, new_colour = colours
    a, b = crop(original, inner), crop(suspect, inner)
    if a.size == 0:
        return False
    # each file's marks: what stands off its own background (the area's old colour here, its new colour there)
    marks_a = np.linalg.norm(a - old_colour, axis=2) > _RECOLOUR_MARK
    marks_b = np.linalg.norm(b - new_colour, axis=2) > _RECOLOUR_MARK
    if not marks_a.any() and not marks_b.any():
        return True  # bare area in both: the recolour and nothing else
    if not marks_a.any() or not marks_b.any():
        return False  # marks in one file only: something added or removed there
    near = np.ones((2 * _RECOLOUR_SLACK_PX + 1, 2 * _RECOLOUR_SLACK_PX + 1), np.uint8)
    near_a = cv2.dilate(marks_a.astype(np.uint8), near).astype(bool)
    near_b = cv2.dilate(marks_b.astype(np.uint8), near).astype(bool)
    agreement = min((marks_a & near_b).sum() / marks_a.sum(), (marks_b & near_a).sum() / marks_b.sum())
    return float(agreement) >= _RECOLOUR_SAME_MARKS


def _one_upright_word(a: DiffCluster, b: DiffCluster) -> bool:
    """Two findings that are pieces of one word running up (or down) the page: both taller than wide, side by side
    across (their widths overlapping), end to end along, and neither a move while the other is not."""
    first, second = a.bbox_in_original, b.bbox_in_original
    for box in (first, second):
        if box.height < _UPRIGHT_ASPECT * max(1.0, box.width):
            return False
    if (a.subtype is DiffSubtype.MOVED) != (b.subtype is DiffSubtype.MOVED):
        return False
    across = min(first.x1, second.x1) - max(first.x0, second.x0)
    narrow = min(first.width, second.width)
    if across < _UPRIGHT_OVERLAP * narrow:
        return False
    if max(first.width, second.width) > 1.6 * max(1.0, narrow):
        return False  # a line of type is one width all along
    gap = max(first.y0 - second.y1, second.y0 - first.y1, 0.0)
    return gap <= _UPRIGHT_GAP * max(first.width, second.width)


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
            moved_to=new.bbox_in_original,
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


_PAGE_FOR_PAGE = 0.85  # a scan whose page, laid over the digital file's, coincides with it this much is of the document


def _glass_beyond_page(suspect_shape: tuple[int, int], transform: np.ndarray, reference_shape: tuple[int, int]) -> bool:
    """Whether the scan is of the scanner's glass with an item laid on it, rather than of the document itself page for
    page: its page, laid over the digital file's, is not that page (a page-for-page scan coincides with it)."""
    height, width = suspect_shape
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
    laid = cv2.perspectiveTransform(corners, transform.astype(np.float64)).reshape(-1, 2).astype(np.float32)
    area = abs(cv2.contourArea(laid))
    page = np.float32([[0, 0], [reference_shape[1], 0], [reference_shape[1], reference_shape[0]], [0, reference_shape[0]]])
    page_area = float(reference_shape[0] * reference_shape[1])
    if area <= 0:
        return False
    overlap, _points = cv2.intersectConvexConvex(laid, page)
    coincide = overlap / (area + page_area - overlap)
    logger.info("scan page laid over the digital page", extra={"coincide": round(coincide, 3)})
    return coincide < _PAGE_FOR_PAGE


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
