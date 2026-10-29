"""Keypoint detection and descriptor matching for alignment (Architecture §3.3.3)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import AlignmentConfig

logger = logging.getLogger(__name__)


@dataclass
class FeatureMatches:
    src_points: np.ndarray  # (N, 2) float32 in the source image
    dst_points: np.ndarray  # (N, 2) float32 in the destination image

    def __len__(self) -> int:
        return len(self.src_points)


def _empty() -> FeatureMatches:
    empty = np.empty((0, 2), dtype=np.float32)
    return FeatureMatches(src_points=empty, dst_points=empty.copy())


def _create_detector(config: AlignmentConfig):
    if config.feature_detector.upper() == "AKAZE":
        return cv2.AKAZE_create()
    return cv2.ORB_create(nfeatures=config.orb_features)


def _to_detector_mask(keep_mask: np.ndarray | None) -> np.ndarray | None:
    """Restrict *where keypoints are found* rather than blanking pixels.

    Painting outside the artwork boundary white would draw a hard artificial
    edge along the boundary and generate spurious corners right where the
    alignment is most sensitive; a detector mask has no such side effect.
    """
    if keep_mask is None:
        return None
    return keep_mask.astype(np.uint8) * 255


def match_features(
    src_image: np.ndarray,
    dst_image: np.ndarray,
    config: AlignmentConfig,
    src_mask: np.ndarray | None = None,
    dst_mask: np.ndarray | None = None,
) -> FeatureMatches:
    """Ratio-tested descriptor matches between two RGB images."""
    detector = _create_detector(config)
    src_kp, src_desc = detector.detectAndCompute(
        cv2.cvtColor(src_image, cv2.COLOR_RGB2GRAY), _to_detector_mask(src_mask)
    )
    dst_kp, dst_desc = detector.detectAndCompute(
        cv2.cvtColor(dst_image, cv2.COLOR_RGB2GRAY), _to_detector_mask(dst_mask)
    )
    if src_desc is None or dst_desc is None or len(src_kp) < 2 or len(dst_kp) < 2:
        logger.info(
            "too few keypoints to match",
            extra={"src_keypoints": len(src_kp or []), "dst_keypoints": len(dst_kp or [])},
        )
        return _empty()

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = [
        pair[0]
        for pair in matcher.knnMatch(src_desc, dst_desc, k=2)
        if len(pair) == 2 and pair[0].distance < config.ratio_test * pair[1].distance
    ]
    if not good:
        return _empty()

    logger.info(
        "matched features",
        extra={
            "src_keypoints": len(src_kp),
            "dst_keypoints": len(dst_kp),
            "good_matches": len(good),
        },
    )
    return FeatureMatches(
        src_points=np.array([src_kp[m.queryIdx].pt for m in good], dtype=np.float32),
        dst_points=np.array([dst_kp[m.trainIdx].pt for m in good], dtype=np.float32),
    )

