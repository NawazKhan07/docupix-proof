"""Tunable runtime configuration.

Values QA needs to change without a code edit live here rather than as
constants inside the pipeline (Build Prompt Phase 3: "Make the threshold a
config value, not a hard-coded constant"). A `config.json` beside the
application overrides any subset of these defaults.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from app import paths

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "config.json"


@dataclass(frozen=True)
class AlignmentConfig:
    # PENDING QA SIGN-OFF (SRS §7): the numeric acceptance threshold is
    # explicitly left open in the SRS, to be set with QA during validation.
    # 2.0 px at the 300 DPI working resolution is ~0.17 mm.
    residual_threshold_px: float = 2.0

    feature_detector: str = "ORB"  # "ORB" | "AKAZE"
    orb_features: int = 5000
    ratio_test: float = 0.75
    # Keep this above `residual_threshold_px`: a keypoint-basis residual is
    # measured over RANSAC inliers, so it can never exceed this value, and a
    # gate threshold above it could never fire.
    ransac_reproj_threshold_px: float = 3.0
    min_match_count: int = 12
    # A homography needs only 4 points, so RANSAC can reach a "confident"
    # consensus on a handful of coincidental matches between two unrelated
    # files. Measured on synthetic pairs: genuine registrations produced
    # 124-768 inliers, two different products produced 8.
    min_inlier_count: int = 25
    # Rejects transforms that mirror or collapse the artwork; the bound is
    # generous enough for real resolution differences between a digital file
    # and a scan (scale factor up to 5x either way).
    max_scale_factor: float = 5.0
    polygon_resample_points: int = 64


@dataclass(frozen=True)
class DetectionConfig:
    # Architecture §6.1: run detection at a working resolution rather than
    # full print resolution. Measured on an A4 300 DPI pair, dropping from
    # 3508 to 2000 px took the SSIM pass from 1.48s to 0.54s. Symbol and
    # Braille detection deliberately ignore this and use full resolution,
    # since small symbols and dots do not survive downscaling.
    detection_working_long_edge_px: int = 2000

    # Coarse-to-fine (Architecture §6.3). Most of two artwork files is usually
    # identical, so a cheap full-resolution pixel diff finds where they differ
    # at all, and the expensive detectors (CIEDE2000, SSIM) then run only inside
    # those regions -- at full resolution, so thin text strokes survive instead
    # of being averaged away by a downscale. Measured on a real 310 x 420 mm
    # carton pair: colour detection alone was 1.1 s of a 2.2 s comparison.
    #
    # Per-channel difference (0-255) at which a pixel counts as changed.
    region_diff_threshold: int = 16
    # Context kept around each changed area, in working pixels (~4 mm at 300 DPI).
    region_margin_px: int = 48
    # Changed pixels are grouped on a grid this coarse, which keeps grouping cheap.
    region_block_px: int = 8
    # If the regions cover more than this share of the page (a scan, or a pair
    # that is not registered), region-limited work stops paying off and the
    # whole-page pass is used instead.
    region_max_area_fraction: float = 0.30

    # When alignment is exact (both files digital and already registered) there
    # is no misregistration noise to filter, so every noise filter is switched
    # off and sensitivity is at its maximum. The filters below exist for
    # rotated/scanned pairs; on an exact pair they only erase real, thin
    # differences -- the opening removed every stroke under ~5 px, which turned
    # a missing sentence into four tiny boxes on stray letters.
    exact_alignment_residual_px: float = 0.25
    # Smallest difference reported when alignment is exact, in raw pixels
    # (~0.5 mm across at 300 DPI): small enough to catch a decimal point.
    exact_min_area_px: int = 6
    # Smallest per-channel change worth reporting when alignment is exact.
    exact_intensity_tolerance: int = 3
    # "Exact" is not "zero": a residual of a few hundredths of a pixel still moves a black-on-white
    # edge by that share of its contrast (255 grey levels, ~100 delta-E), and at full sensitivity
    # that shows up as hairline 'differences' along every strong edge. So the tolerances are never
    # tighter than the residual measured for THIS pair explains (1.5x, in grey levels; 3x, in
    # delta-E), up to these caps -- a real change is far larger.
    exact_levels_per_px: float = 380.0
    exact_delta_e_per_px: float = 300.0
    exact_max_intensity_tolerance: int = 24
    exact_max_delta_e: float = 8.0

    # Registration tolerance, in detection-resolution pixels. Any rotation
    # between the two files means the suspect is resampled, so thin strokes
    # land a fraction of a pixel off and every edge becomes a hairline
    # mismatch -- measured on a 1.5-degree rotation, that traced the whole
    # dieline as one page-sized "difference", with 57% of the differing
    # pixels lying on that 3 px border line.
    #
    # A difference is only kept where no similar value exists within this
    # radius in the other image, which tolerates a sub-pixel shift without
    # blurring away genuinely changed content.
    #
    # THESE THREE ARE NOT YET VALIDATED AGAINST REAL ARTWORK. They trade
    # false positives against sensitivity, and the trade is sharp: measured
    # on synthetic pairs, a 3 px radius silenced a genuine 500mg -> 250mg
    # change (only 8 of its 270 differing pixels survived) because the slack
    # spans the gaps between glyph strokes. Tolerance must stay smaller than
    # the finest feature that has to be told apart. Tune against real
    # original/suspect pairs in tests/golden_pairs/ before trusting them.
    #
    # Defaults to no positional slack, biasing towards sensitivity: even 1 px
    # measurably hid a genuine text change in testing, and for artwork review
    # a false positive costs a reviewer a glance whereas a miss ships. Raise
    # it only if misregistration noise proves worse than that on real files.
    registration_tolerance_px: int = 0
    # How different two values must be to count as different at all, on a
    # 0-255 scale, once position has been tolerated. Above roughly 20 this
    # starts swallowing whole-page colour casts.
    registration_intensity_tolerance: int = 12
    # A second, smaller pass that drops any stragglers thinner than this.
    edge_tolerance_px: int = 2
    # Drop regions whose pixels fill less than this share of their own
    # bounding box: a hairline tracing a shape's outline has a box the size
    # of the shape and almost nothing in it. Set low deliberately, since a
    # genuinely changed frame or rule is also outline-shaped.
    min_fill_ratio: float = 0.10

    # Structural comparison of "ink" (Scan-vs-Digital, SRS FR-6.3). Ink is anything
    # that departs from its LOCAL background, so orange text on grey counts exactly
    # like black text on white (a global black/white threshold loses it).
    #
    # Two levels give hysteresis: a difference must be STRONG ink in one file and
    # not even WEAK ink in the other, so faint content flickering across a single
    # threshold between two files is never reported. Each is a share of the local
    # tone with a floor in grey levels. Measured on a scan of identical content,
    # a single 12% threshold flagged up to 104k stray pixels; these leave ~0 on a
    # light scan and a small fraction of that on a harsh one.
    ink_strong_relative: float = 0.22
    ink_strong_floor: float = 36.0
    ink_weak_relative: float = 0.06
    ink_weak_floor: float = 10.0
    # Each image's own measured noise (standard deviation, grey levels) raises
    # the floors to this many sigmas, so a noisy scan is judged against its own
    # noise floor while a clean digital file keeps the sensitive defaults above.
    ink_noise_strong_sigmas: float = 7.0
    ink_noise_weak_sigmas: float = 2.5
    # The local background is a median over each of these window sizes, in
    # pixels at 300 DPI (~3.7 mm and ~17 mm), and the differences found at each
    # scale are unioned. ~4 stroke-widths keeps text a minority of any window yet
    # small enough to follow a small coloured box the text sits in; the larger
    # scale sees whole panels and boxes as ink. One scale alone fails one way or
    # the other (see structure_diff.py).
    structure_background_windows_px: tuple[int, ...] = (44, 200)
    # A scan is softer than the file it was made from, and a soft edge reads as ink
    # a few pixels wider than a sharp one, which registration slack cannot absorb.
    # So the sharper image is blurred to the softness of the other (measured from
    # the pixels, up to this many pixels of Gaussian blur) before they are compared.
    # A match must beat "no blur" by this much correlation to be believed.
    structure_softness_max_px: float = 3.5
    # How graphics are compared when the two files are NOT exactly registered
    # (a digital pair placed slightly differently, or a scan): "structure" asks where
    # ink is, within the measured registration error; "ssim" is the similarity map;
    # "both" runs the two and lets overlapping findings merge. Measured on the
    # real carton pair placed differently: ssim alone found a missing line of small
    # print with 0.07 IoU, structure 0.96, and both together also framed a
    # shifted-lettering change best (0.78 vs 0.31-0.58) for ~0.4 s more.
    # (Scans always use "structure": the similarity map cannot follow a scan.)
    graphics_when_misregistered: str = "both"
    # A scan this much softer than the original (Gaussian sigma, px at 300 DPI) can no
    # longer hold the finest print, so the reviewer is told fine print is unverified.
    structure_soft_warning_sigma_px: float = 2.0
    structure_softness_min_gain: float = 0.01
    # Where one image is much softer than the other -- fine detail below this share
    # of the sharper one's, per ~6 mm cell -- while the coarse LAYOUT (word and line
    # structure) still correlates at least this strongly, the detail was lost to the
    # scan rather than removed from the artwork: the cell is skipped and reported as
    # unverifiable. Blur keeps layout; a real change alters it, so it is never
    # skipped. (An earlier test on ink mass alone mistook a box added around
    # identical text for blur, and skipped a real change.)
    structure_detail_ratio: float = 0.4
    structure_layout_correlation: float = 0.85
    structure_layout_variance_floor: float = 0.5
    # Below this much fine detail in the sharper image there is nothing to lose.
    structure_detail_floor: float = 2.5
    # The scan's own boundary is kept clear by this many slack-widths.
    structure_edge_margin_factor: int = 3
    # Report the skipped area to the reviewer once it reaches this share of the artwork.
    unverifiable_warning_fraction: float = 0.02
    # A stroke is only a difference if the OTHER file has no ink within this
    # radius, so registration error of up to this much cannot flag it. The
    # radius is the alignment residual times this factor, within the bounds below.
    structure_slack_factor: float = 1.5
    structure_min_slack_px: int = 1
    structure_max_slack_px: int = 4
    # Differing ink this close together is one object (~1 mm at 300 DPI); the
    # aggregator then joins words into a sentence.
    structure_group_px: int = 12
    # Fewest differing ink pixels worth reporting (~0.4 mm across at 300 DPI).
    structure_min_area_px: int = 25
    # ...and never less than the footprint of the blur the two files were matched at
    # (a disc this many sigmas in radius): a mark smaller than the softness of the
    # scan cannot be told from JPEG ringing or noise at a corner.
    structure_footprint_sigmas: float = 2.0

    # The ink picture of a file (Scan vs Digital; see scan_normalizer.ink_image): background colours
    # removed to white and every ink black. A tint is estimated as the median over a window this share
    # of the page's long edge (wider than any stroke, narrower than a panel); an ink is that many grey
    # levels (0-255) darker than its tint to begin to show, and that many to be fully black; and a
    # surround at least `ink_solid_level` dark is a solid graphic (a dark bar, a reversed-out block),
    # not a tint, so it stays black.
    ink_window_fraction: float = 0.02
    ink_contrast_low: float = 48.0
    ink_contrast_high: float = 108.0
    ink_solid_level: float = 200.0
    # Text is read from the ink pictures at no more than this long edge (px): Tesseract reads best at
    # about 300 DPI, so an A4 page (3508 px) is read as it is and a larger sheet is reduced to this.
    ocr_long_edge_px: int = 3600

    # Pharmacode (see pharmacode.py): bars of one length, at least this long and this
    # many times longer than thick (px at 300 DPI: ~1.4 mm up to ~50 mm), in a row of
    # 4 to 16 (the symbology's own limit is 2 to 16; 2-3 bars are indistinguishable
    # from ordinary marks).
    pharmacode_min_bars: int = 4
    pharmacode_max_bars: int = 16
    pharmacode_min_bar_length_px: int = 16
    pharmacode_max_bar_length_px: int = 600
    pharmacode_min_aspect: float = 3.0

    # Colour (CIEDE2000 in Lab). 2.3 is the classic "just noticeable
    # difference" figure; print tolerances are usually looser, so this is a
    # sensitive default QA will likely want to raise.
    color_delta_e_threshold: float = 2.3
    color_min_area_px: int = 40
    # Colour gets its own positional slack, separate from the graphics
    # channel's (which must stay at 0). Anti-aliased glyph edges shift by a
    # fraction of a pixel under any rotation, and on a 1.2 degree pair that
    # made three tiny colour findings appear on small text, each ~dE 25 and so
    # rated CRITICAL -- noise in the most severe tier. 1 px removed all three
    # while both real changes and a genuine black->red recolour of a thin
    # text line were still found (2 px gave the same result). Unlike the
    # graphics channel, colour has no need to catch a text *edit* (the text
    # detector and SSIM do), so it can afford the slack.
    color_registration_tolerance_px: int = 1
    # CIEDE2000 over a full 300 DPI page is far too slow for NFR-1, so the
    # colour pass runs on a long edge no larger than this and its boxes are
    # scaled back up (Architecture §6 allows a per-superpixel colour delta).
    color_working_long_edge_px: int = 1200

    # Graphics (SSIM).
    ssim_threshold: float = 0.80
    ssim_min_area_px: int = 40

    # Text (OCR).
    ocr_min_confidence: float = 40.0
    text_move_tolerance_px: float = 6.0

    # Symbols (barcode / pharmacode / QR).
    symbol_match_tolerance_px: float = 60.0

    # Braille. PENDING SIGN-OFF (SRS §7): the dot colour convention used in
    # our source artwork is an open item. These HSV bounds cover the
    # magenta/pink spot colour dots are commonly marked in; they must be
    # confirmed against real files before this detector is trusted.
    braille_hsv_lower: tuple[int, int, int] = (140, 80, 80)
    braille_hsv_upper: tuple[int, int, int] = (175, 255, 255)
    braille_dot_min_area_px: int = 6
    braille_match_tolerance_px: float = 10.0


@dataclass(frozen=True)
class ClusteringConfig:
    # PENDING QA SIGN-OFF (SRS §7): the proximity rule for merging individual
    # word / number / barcode-line differences into one cluster is not
    # numerically defined. Two same-type differences whose boxes are within
    # this many pixels of each other (in the original's render resolution,
    # 300 DPI for a PDF) become one finding. 30 px is ~2.5 mm: enough to join
    # the words of a moved line and the lines of a paragraph, small enough
    # not to fuse unrelated changes on opposite sides of a carton panel.
    distance_px: float = 30.0
    # Two findings whose boxes overlap by at least this share of the SMALLER
    # one are the same physical change seen by different detectors (a new logo
    # is both a colour and a graphics difference) and are reported once.
    merge_overlap_ratio: float = 0.2
    # ...unless one box is vastly larger than the other, which is a stray
    # outline artefact swallowing genuine findings rather than a duplicate.
    merge_max_area_ratio: float = 60.0


@dataclass(frozen=True)
class SubjectConfig:
    """Lifting the artwork off its background (`app/engine/isolation/subject_lifter.py`)."""

    # Segmentation runs on a reduced copy of the page, then the outline is refined at full
    # resolution along its edge. A carton at 300 DPI is ~5000 px tall; 800 is what keeps a
    # lift to a fraction of a second.
    working_long_edge_px: int = 720
    # The outer strip of the page is where the background is looked for.
    border_band_fraction: float = 0.03
    # The press point marks a disc of definite subject this big (share of the page's short side).
    seed_radius_fraction: float = 0.02
    grabcut_iterations: int = 4
    # A pixel this close (CIELAB) to the background's colours, and closer to it than to the
    # subject at the press point, starts out as background -- if it is joined to the page
    # edge through such pixels, so a background-coloured panel INSIDE the artwork is not
    # mistaken for background. "This close" follows how much the page edge itself varies
    # (spread_factor x its 90th-percentile distance from its own colours), between the two limits.
    background_tolerance: float = 12.0
    background_tolerance_min: float = 4.0
    background_spread_factor: float = 1.6
    # A cast shadow is the background darkened, hue unchanged: a pixel that is the page
    # edge's colour at this fraction of its brightness (between the two limits), to within
    # this relative colour error, counts as background too. Without it a photographed
    # carton's shadow is lifted along with the carton.
    shadow_brightness_range: tuple[float, float] = (0.45, 0.97)
    shadow_hue_tolerance: float = 0.08
    # Full-resolution edge refinement reaches this many working-resolution pixels either side
    # of the outline.
    refine_band_working_px: float = 1.6
    # The outline is simplified to within this many pixels (full resolution)...
    outline_epsilon_px: float = 1.5
    max_outline_points: int = 400
    # ...and an outline that is nearly a rectangle is drawn as one (share of its rotated box).
    rectangle_snap_fill: float = 0.985
    # Lifting the area inside a closed border line (border_lifter.py), tried before colour:
    # the lines are found at this size (long edge, px); a line is a pixel darker than this
    # (or than half the sheet's own brightness); the sheet must be at least this light for
    # lines to be told from the surface it is on.
    border_working_long_edge_px: int = 2400
    border_dark_level: float = 115.0
    # A thin line that is not dark (a grey hairline, or a black one softened by a screenshot or a
    # scan's resampling) still counts if it is this many grey levels darker than its surroundings.
    # "Thin" is narrower than this share of the long edge (at least 5 px).
    border_faint_line_contrast: float = 30.0
    border_line_width_fraction: float = 0.006
    border_min_page_brightness: float = 170.0
    # Automatic mode accepts an enclosure only if it is a sizeable part of the sheet, so a text
    # box is not taken for the artwork; "Border line" mode accepts any of at least this share.
    border_min_area_fraction: float = 0.08
    border_min_area_fraction_forced: float = 0.005
    # Thin protrusions (a leader line, a dimension line and its arrowheads, a label hung on one)
    # narrower than twice this share of the long edge are shaved off; what is left of the outline
    # is then grown back by the same amount, so real corners keep their points.
    border_shave_fraction: float = 0.0016
    # Comparing starts this many pixels inside a lifted outline's own edge, beyond the width of the
    # drawn line: the die line itself is not printed artwork (a scan of the printed sheet has none),
    # and the fringe of a colour-lifted edge is where a scan's blur and shadow are.
    border_line_margin_px: float = 1.5
    colour_edge_inset_px: float = 3.0
    # An enclosure this much of the sheet is a frame around it, not the artwork.
    border_frame_fraction: float = 0.92
    # How long a press must be held to lift, and how far it may drift while held (ms, screen px).
    hold_ms: int = 350
    hold_slop_px: int = 6


@dataclass(frozen=True)
class RuntimeConfig:
    """Guard rails and worker settings (Architecture §5 and §7).

    Files are checked against these BEFORE anything is allocated for them, so an
    oversized input produces a clear message instead of an out-of-memory crash.
    """

    # Real packaging artwork is tens of MB; a file far beyond this is not artwork.
    max_file_size_mb: int = 250
    # A carton or leaflet file is a handful of pages; hundreds is a catalogue or a mistake.
    max_pdf_pages: int = 100
    # The largest page compared, in millions of pixels at the working resolution. A
    # 1 m x 1 m sheet at 300 DPI is ~140 MP (about 420 MB as RGB, and several
    # times that while detecting), so this is the point past which memory, not
    # the artwork, decides the outcome. The 310 x 420 mm sample carton is 18 MP.
    max_render_megapixels: int = 160

    # One comparison at a time in a separate process (Architecture §5).
    # Recycled after this many jobs so native-library memory cannot creep.
    worker_recycle_after_jobs: int = 20
    # A comparison still running after this long is stopped and reported.
    job_timeout_s: int = 600


@dataclass(frozen=True)
class AppConfig:
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    subject: SubjectConfig = field(default_factory=SubjectConfig)


def load_config(path: Path | str | None = None) -> AppConfig:
    """Load config from JSON, falling back to defaults for anything absent."""
    if path is not None:
        config_path = Path(path)
    else:  # beside the application, or in the user's folder
        config_path = next((c for c in paths.config_candidates(CONFIG_FILENAME) if c.is_file()), Path(CONFIG_FILENAME))
    if not config_path.is_file():
        return AppConfig()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("ignoring unreadable config %s: %s", config_path, exc)
        return AppConfig()

    return AppConfig(
        alignment=_merge(AlignmentConfig(), raw.get("alignment", {})),
        detection=_merge(DetectionConfig(), raw.get("detection", {})),
        clustering=_merge(ClusteringConfig(), raw.get("clustering", {})),
        runtime=_merge(RuntimeConfig(), raw.get("runtime", {})),
        subject=_merge(SubjectConfig(), raw.get("subject", {})),
    )


def _merge(defaults, overrides: dict):
    known = {f.name for f in fields(defaults)}
    unknown = set(overrides) - known
    if unknown:
        logger.warning("ignoring unknown config keys: %s", sorted(unknown))
    return replace(defaults, **{k: v for k, v in overrides.items() if k in known})
