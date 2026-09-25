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
    # Where two layouts differ, differing ink is everywhere and the grouping above joins it, line to line,
    # into one 'object' the size of the sheet (one 2278 x 2556 px 'graphic changed' on a real pair). An
    # object bigger than this whose differing ink fills under `structure_split_fill` of its box is a chain:
    # it is grouped again with a shorter reach (halved, down to 1 px) until it falls into its pieces.
    structure_split_min_area_px: int = 250000
    structure_split_fill: float = 0.08
    # Fewest differing ink pixels worth reporting (~0.4 mm across at 300 DPI).
    structure_min_area_px: int = 25
    # ...and never less than the footprint of the blur the two files were matched at
    # (a disc this many sigmas in radius): a mark smaller than the softness of the
    # scan cannot be told from JPEG ringing or noise at a corner.
    structure_footprint_sigmas: float = 2.0

    # The ink picture of a file (Scan vs Digital; see scan_normalizer.ink_image): background colours
    # removed to white and every ink black. A tint is estimated as the median over a window this share
    # of the page's long edge (wider than any stroke, narrower than a panel); an ink is that many grey
    # levels (0-255) darker than its tint to begin to show, and that many to be fully black -- but a scan's
    # strokes are lighter than a digital file's, so those two follow the picture's own ink strength (this
    # share of its 90th percentile, never below the minimum). And how SOLID a surround is (a dark bar, a
    # reversed-out block, which stays black) is a smooth function of its luminance darkness, 0 below `low`
    # and 1 above `high`: never a cut-off that a scan's slightly lighter colour can fall the wrong side of.
    ink_window_fraction: float = 0.02
    ink_contrast_low: float = 48.0
    ink_contrast_high: float = 108.0
    ink_ramp_share: float = 0.5
    ink_ramp_low_share: float = 0.45
    ink_ramp_min_high: float = 60.0
    ink_ramp_min_low: float = 20.0
    ink_solid_low: float = 100.0
    ink_solid_high: float = 160.0
    # Scan vs Digital compares TEXT (and barcode bars as marks). Its graphics are compared only if this is set: a scan
    # changes every colour, blurs every edge and never registers exactly, and comparing the two pictures gave dozens
    # of false 'graphics' differences (a fold line, the scanner's own edge, every dark block) round the few real ones.
    scan_compare_graphics: bool = False
    # What each file is READ from in Scan vs Digital (see scan_normalizer.reading_source): 'ink' (the ink picture: background
    # removed, ink black), 'gray' (the file's own grey), 'flat' (grey with the paper divided out), 'darkest' (the darkest
    # channel, paper divided out), 'clahe' (flat, local contrast stretched) or 'adaptive' (flat, thresholded locally).
    scan_reading_digital: str = "ink"
    scan_reading_scan: str = "ink"
    # A line that differs is read again, from its own window, in the picture above AND in this one ("" for none): a word
    # that one picture misreads and the other reads is not a difference, and a real change is different in both.
    scan_reading_alternate: str = ""
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
    # The reader's language (a Tesseract code, or several joined with "+", e.g. "eng+fra"): only what has that
    # language's tessdata installed can be read as more than shapes. A finding a word cannot be read for is still
    # found and located from its pixels (OCR only DESCRIBES a finding, never decides one) -- it is just not
    # captioned with the words. See `app.engine.ocr_locator.available_languages`.
    ocr_language: str = "eng"
    text_move_tolerance_px: float = 6.0
    # Two digital files are read the same where their pixels are the same: a word read differently over
    # marks that agree this closely (within 2 px, each way) is the reader's inconsistency (reading a page
    # groups its lines differently when anything else on it differs), not a change.
    text_identical_marks_agreement: float = 0.985
    # Reading a SCAN is noisier than reading the digital file it came from. Compared page against page,
    # a changed digit reshuffled the reading order and one identical layout gave 23 false 'changes'
    # (a case or comma, an l read as i, a line grouped differently, rotated text read as junk). So
    # Scan vs Digital anchors on the DIGITAL file's own lines: each line's window is read again, alone, in
    # both files; the words are compared without case, punctuation or the l/i/1 and o/0 confusions (and
    # nothing looser: one letter is also what a real change to a word is), and words read less surely
    # than these floors are ignored. A line the scan yields nothing for is REMOVED only if
    # the ink is gone too (its marks agree with the digital's under `text_present_ink_agreement`, judged where
    # the scan actually sits); with the ink still there the scan is simply
    # too soft to read it, which is reported as unreadable, not as a change. Text that is only in the scan
    # is found by reading the whole scan and keeping confidently read words outside every digital line
    # where the digital has no ink. A moved line is a change of its words' place, not of the words.
    ocr_min_confidence_scan: float = 55.0
    ocr_short_word_confidence: float = 75.0
    text_line_pad_px: int = 20
    # Ink this close to where a word is counts as being there: a scan is never registered exactly (a two-sheet scan
    # of a carton was 10-20 px out in places, so lines the scan HAD were 'removed' and words it had 'added').
    text_ink_reach_px: int = 14
    text_present_ink_agreement: float = 0.5
    text_added_confidence: float = 70.0
    text_added_ink_share: float = 0.25
    # A word read differently in the scan whose ink lies, at least this much of it each way (within 2 px), on the
    # digital word's ink was very probably only misread: it is kept, ranked Minor and worded as a likely misread.
    # (A real one-letter change also matches this closely, so it cannot be dropped -- only ranked.)
    text_misread_ink_agreement: float = 0.96
    # ...where 'on the same ink' is judged to this many pixels: a scan's dot gain makes a hairline of lettering several
    # times as thick, so its ink lies further from the digital's than a couple of pixels.
    text_misread_slack_px: int = 3
    # Where the two files' INK PICTURES are, in the difference's box, the same picture -- correlate (normalised, softened a little,
    # each window about a letter wide re-settling within a few pixels of where the scan sits, so a line's warp along a
    # scan is not counted against it) at least `text_same_ink_loose_correlation` as a whole (one shift for the box; only
    # rules out an accident) and `text_same_ink_worst` in their worst window -- the reading, not the artwork, is what
    # differs, and the finding is ranked down (a changed letter leaves a window that does not correlate, however well
    # the rest of the word does).
    text_same_ink_loose_correlation: float = 0.6
    text_same_ink_worst: float = 0.85

    # Automatic mode finds differences in the PIXELS. Two files that register to a pixel or two, or where one is a
    # little softer than the other (a picture of the type against live type), differ along every edge -- 13 of 19
    # 'critical colour changes' on one real carton pair were hairlines round letters that are the same in both. So
    # each finding is judged by what is INSIDE its box (plus this margin, to take in the whole edge). It is the same
    # artwork drawn slightly differently if the two files' marks (ink) coincide to at least `residue_min_agreement`
    # and their AVERAGE colour there differs by less than `residue_max_ink_colour_delta` (CIEDE2000: a thin line
    # drawn softer or a shade lighter) while the background between the marks differs by less than
    # `residue_max_colour_delta` (a tint that changed is a change). Where there are no marks to compare (a flat
    # area), the average colour must differ by less than the latter. On the carton pairs every such hairline
    # measured under 3 (average) and over 0.97 (marks), and every real change was over 4 or under 0.94.
    residue_margin_px: int = 6
    # 'Coincide' means within this many pixels: the registration error and the softness to be forgiven. An exactly
    # registered pair forgives one pixel only (`effective_detection_config`): with two, a 0 -> 1 in six-pixel type was
    # 'the same marks' and was missed (a 0 and a 1 are within two pixels of each other's strokes).
    residue_slack_px: int = 2
    residue_max_colour_delta: float = 2.0
    # Two exports of the same artwork often render a solid ink a few dE apart (a colour profile: one pink or black a
    # shade off across the whole page), which is not a change to a piece of it: 4.6 on a wordmark, 4.0 on a black word.
    residue_max_ink_colour_delta: float = 5.0
    # A shape finding (the similarity map) whose marks coincide has no shape difference: what colour it has is for the
    # colour comparison to say, and reports where it is a change. So it is only kept for a colour difference this big.
    residue_max_shape_colour_delta: float = 8.0
    residue_edge_shift_px: float = 2.0  # ...or what an edge moved by this much shifts it (contrast x this / the window's width), if more
    residue_min_agreement: float = 0.97
    # The deep local comparison (`local_diff`): where the two files differ at all (a full-size overlay, per-channel
    # difference above `local_coarse_threshold` levels), each 128 px tile is registered on its own (a whole-pixel shift
    # within `local_shift_radius_px`, taken only if it clearly helps), softened by `local_blur_px`, and compared with
    # `local_forgive_px` forgiven each way; what still differs by more than `local_tolerance` levels is a difference. Groups
    # of such pixels within `local_group_px` are one, if at least `local_min_pixels` strong.
    local_enabled: bool = True
    # True: the older reduced-size detectors (colour, similarity map, structure) run beside the local comparison.
    # False: the local comparison alone finds the pixel differences (exact boxes, and faster).
    legacy_pixel_detectors: bool = False
    # Automatic mode with no crop or lift finds the artwork on each sheet itself and compares only that.
    auto_artwork: bool = True
    # Something missing where it was and added where it is, looking alike, is one finding: that it moved. And a finding whose
    # crop of the original fits the suspect a few pixels away is that picture moved.
    pair_moves: bool = True
    find_moves: bool = True
    find_moves_max_findings: int = 300
    local_coarse_threshold: int = 24
    local_tolerance: int = 32
    local_blur_px: float = 0.8
    local_shift_radius_px: int = 2
    local_forgive_px: int = 1
    local_group_px: int = 3
    local_min_pixels: int = 8
    # Colour regions: where two files differ in colour over a FLAT area (no edge within a pixel; a recoloured panel, a
    # tint, the fill of a letter), the difference is found at full size as CIE76 delta-E of at least
    # `local_colour_delta_e` over at least `local_colour_min_area_px` (after closing gaps and opening specks), and its box is
    # the region's own. A region of at least `local_offset_min_area_px` is a recoloured AREA: what differs by no more
    # than its colour offset is not a difference, so the digits changed inside it are found on their own.
    local_colour_delta_e: float = 4.0
    local_colour_min_area_px: int = 60
    local_offset_min_area_px: int = 1500
    local_faint_threshold: int = 8
    local_flat_gradient: int = 12
    # A difference that is one long thin stroke (at least this long) is line-work: a die line or fold line.
    local_line_min_length_px: int = 60
    # What a finding says is then READ from its own box (never the whole page: two readings of one page group
    # their lines differently and report areas that did not change). At most this many findings are read.
    text_describe_max_findings: int = 80
    # A word is claimed changed, missing or new only if it was read at least this surely, and the pixels back it: over
    # the same marks (to `residue_min_agreement`) a reading that differs is the reader's; a word said to be
    # missing or new whose marks are still at least this much the same in the other file is unread there, not gone.
    text_describe_min_confidence: float = 75.0
    text_describe_present_agreement: float = 0.5

    # Symbols (barcode / pharmacode / QR).
    symbol_match_tolerance_px: float = 60.0

    # Braille. PENDING SIGN-OFF (SRS §7): the dot colour convention is still an open item, but the
    # only real Braille files so far (the B-Carton / Metformin cartons) mark their dots RED
    # (RGB 239, 65, 53: hue 2 on OpenCV's 0-179 scale), where the first assumption was magenta/pink,
    # which found no dot at all. Hue wraps round 0/180, so red is two arcs: these cover red and orange-red
    # (not orange, whose hue is ~15) and magenta/pink. A dot is a small, roundish blob: text, logos and
    # panels of the same colour are not dots.
    braille_hsv_ranges: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = (
        ((0, 110, 110), (9, 255, 255)),
        ((140, 80, 80), (179, 255, 255)),
    )
    braille_dot_min_area_px: int = 30
    braille_dot_max_area_px: int = 1500
    # Real dots are all one size (~270 px at 300 DPI on the sample cartons); the specks of a red letter
    # (the dot of an i, a colon) are round and small enough to pass the shape test but are a fraction of
    # that. With enough dots to judge by, a blob must be within this factor of the median dot's area.
    braille_dot_size_factor: float = 2.5
    braille_dot_min_count_for_size: int = 8
    braille_dot_min_fill: float = 0.5  # share of its bounding box a round dot fills (a disc is ~0.79)
    braille_dot_max_aspect: float = 1.8  # a dot is about as wide as it is tall
    braille_dot_min_circularity: float = 0.6  # 4 pi area / perimeter^2: a disc is ~1, the strokes of a letter far less
    braille_match_tolerance_px: float = 10.0


@dataclass(frozen=True)
class ClusteringConfig:
    """How raw differences become findings. Each difference is its own finding, exactly where it is, and is joined to
    another only where the two are really one thing:

    * the bars of a barcode (and the dots of Braille) are one symbol: within `distance_px` of each other they are one
      finding;
    * the letters and words of a changed line of type are one finding when they adjoin on the same line (below);
    * two boxes that are the same place found by two detectors (a new logo is both a colour and a graphics difference)
      are one finding;
    * nothing else is joined: a small difference inside a big one (a digit changed in a recoloured panel) is a finding of
      its own, and two differences a few millimetres apart are two.
    """

    # Barcode bars and Braille dots within this many pixels (300 DPI: ~2.5 mm) are one symbol, one finding.
    distance_px: float = 30.0
    # Letters/words are one line of type if they are on the same line -- their boxes overlap by at least this share of the
    # smaller height -- and no further apart than `line_gap_heights` times the taller one (at most `line_max_gap_px`). A box
    # taller than `line_max_height_px` is not a letter or a word and is not joined along a line.
    line_overlap: float = 0.5
    line_gap_heights: float = 1.2
    line_max_height_px: float = 400.0
    line_max_gap_px: float = 60.0  # (however tall the letters: a heading's letters are a few millimetres apart, not a hand)
    # Boxes of one category that touch (a gap of at most this many pixels) and lie side by side are one thing cut in two by
    # a tile edge; boxes of different categories that cover the same place (this much of their union) are one change.
    touch_px: float = 2.0
    duplicate_iou: float = 0.6
    # Words that MOVED are one move only if they moved alike: this much difference in displacement (px) at
    # most, or this share of the larger one, whichever is more.
    moved_displacement_tolerance_px: float = 6.0
    moved_displacement_tolerance_share: float = 0.3


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
    # With no press point (finding the artwork on its own) an enclosure is accepted only if it is a sizeable part of
    # the sheet, so a text box is not taken for the artwork. Where the reviewer POINTED, there is no size rule: any
    # closed border round the spot is lifted, however small (the floor, 0.02% of the sheet, is only there so that the counter of a
    # letter is not taken for the artwork: a 2 x 1 cm label on a carton sheet is 0.15%).
    border_min_area_fraction: float = 0.03
    border_min_area_fraction_forced: float = 0.0002
    # The colour method's press marks a disc of definite subject: this share of the page's short side, then smaller
    # (each in turn) where the area pressed on is smaller than the disc.
    seed_radius_steps: tuple[float, ...] = (1.0, 0.3, 0.1)
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
class SeriesConfig:
    """Finding every repeat of one artwork on a sheet printed with several copies of it (`app/engine/series.py`)."""

    # The scale is settled once by the strongest single match anywhere on the sheet, tried at steps of this ratio...
    scale_step: float = 1.06
    scale_steps_each_way: int = 14  # ...covering a factor of about scale_step ** steps_each_way either way
    # ...around 1.0 (the same size as the original crop) unless the two files' resolutions say otherwise.
    default_expected_scale: float = 1.0
    min_match_score: float = 0.4  # normalised correlation of the two pictures' edges
    min_match_peak: float = 5.0  # ...standing out from the correlation's own average by this many standard deviations
    max_repeats: int = 60
    search_long_edge_px: int = 1600  # the sheet is searched on a copy no bigger than this (long edge)
    instance_margin_px: int = 24  # each repeat is cropped out with this much extra around it, for its own alignment to refine
    # After a repeat is found, another is not looked for within this fraction of the template's width/height of it (a
    # real neighbour, printed with a gap, sits further off; this only stops the same instance being found twice).
    suppression_fraction: float = 0.3


@dataclass(frozen=True)
class LeafletConfig:
    """Comparing a multi-page document (a leaflet, an insert) page by page, however it was made -- outlined to a
    picture, live text, closed or open, one language or several (`app/engine/leaflet.py`).

    Every page's pixels are compared exactly as Automatic mode compares a single pair (alignment, every detector,
    clustering): a page needs no border or die-line, only its two renders. OCR only ever DESCRIBES a difference the
    pixels already found (never decides one), so a page in a language nobody has installed tessdata for is still
    compared correctly -- its findings are just not captioned with the words.
    """

    # A Tesseract language code, or several joined with "+" (e.g. "eng+fra+deu"): whichever of them has its tessdata
    # installed is used; none installed falls back to English and says so once, rather than failing page by page.
    language: str = "eng"
    max_page_workers: int = 4  # pages are compared this many at a time (each comparison already threads its own work)
    page_gap_px: int = 28  # blank strip between one page and the next in the stacked view, with a label drawn on it
    # When the two files do not have the same number of pages, each page is paired with its most alike page in the
    # other file (a thumbnail correlation) rather than assumed to be at the same position; below this, a page is left
    # unpaired (a whole page added or missing) rather than compared with something it is not.
    min_page_match_score: float = 0.3
    thumbnail_long_edge_px: int = 300  # pages are paired on copies this small (fast; only which page matches which)


@dataclass(frozen=True)
class RuntimeConfig:
    """Worker settings and memory guards (Architecture §5 and §7).

    Nothing here refuses a file. A page too big to hold at the working resolution is rendered at a lower one
    (`working_megapixels`), and the comparison says so; the size and page limits are off (0) unless someone sets them.
    """

    # A file bigger than this many MB is refused (0 = never). Real artwork runs to hundreds of MB, so no default.
    max_file_size_mb: int = 0
    # A PDF with more pages than this is refused (0 = never): a leaflet or a carton file is a few pages, a catalogue is many.
    max_pdf_pages: int = 0
    # The most pixels (millions) a page is held at. A 300 DPI render of a 310 x 420 mm carton is 18 MP; a 1 m x 1 m sheet is
    # ~140 MP (420 MB as RGB, and several times that while detecting). A page that would be bigger is rendered at whatever
    # lower DPI fits (a 2.2 m shipper at ~180 DPI), so memory, not a rule, decides the resolution -- and nothing is turned away.
    max_render_megapixels: int = 100

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
    series: SeriesConfig = field(default_factory=SeriesConfig)
    leaflet: LeafletConfig = field(default_factory=LeafletConfig)


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
        series=_merge(SeriesConfig(), raw.get("series", {})),
        leaflet=_merge(LeafletConfig(), raw.get("leaflet", {})),
    )


def _merge(defaults, overrides: dict):
    known = {f.name for f in fields(defaults)}
    unknown = set(overrides) - known
    if unknown:
        logger.warning("ignoring unknown config keys: %s", sorted(unknown))
    return replace(defaults, **{k: v for k, v in overrides.items() if k in known})
