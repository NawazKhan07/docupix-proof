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
class AppConfig:
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)


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

    return AppConfig(alignment=_merge(AlignmentConfig(), raw.get("alignment", {})))


def _merge(defaults, overrides: dict):
    known = {f.name for f in fields(defaults)}
    unknown = set(overrides) - known
    if unknown:
        logger.warning("ignoring unknown config keys: %s", sorted(unknown))
    return replace(defaults, **{k: v for k, v in overrides.items() if k in known})
