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


@dataclass(frozen=True)
class AppConfig:
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)


def load_config(path: Path | str | None = None) -> AppConfig:
    """Load config from JSON, falling back to defaults for anything absent."""
    config_path = Path(path or CONFIG_FILENAME)
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
    )


def _merge(defaults, overrides: dict):
    known = {f.name for f in fields(defaults)}
    unknown = set(overrides) - known
    if unknown:
        logger.warning("ignoring unknown config keys: %s", sorted(unknown))
    return replace(defaults, **{k: v for k, v in overrides.items() if k in known})
