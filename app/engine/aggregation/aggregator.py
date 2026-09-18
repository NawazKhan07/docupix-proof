"""The Diff Aggregator (Architecture §3.3.6): raw candidates -> findings.

Composes the three aggregation steps -- cluster, assign severity, bind
coordinates -- into the `DiffCluster` list the UI presents.
"""

from __future__ import annotations

import logging

import numpy as np

from app.config import ClusteringConfig
from app.engine.aggregation.clustering import cluster_candidates, union_box
from app.engine.aggregation.coordinate_projection import original_to_suspect
from app.engine.aggregation.severity import DEFAULT_RULES, SeverityRule, worst_severity
from app.logging_config import StageTimer
from app.models.diff import DiffCandidate, DiffCluster, DiffType

logger = logging.getLogger(__name__)

# When one finding is seen by several detectors, it is labelled by the most
# specific description that applies: a changed word is a text change before
# it is a graphics change, and a graphics change before it is merely a colour
# one (a new logo differs in colour too, but "graphic added" is what it is).
TYPE_PRIORITY = (
    DiffType.TEXT,
    DiffType.NUMBER,
    DiffType.BARCODE,
    DiffType.BRAILLE,
    DiffType.GRAPHICS,
    DiffType.COLOUR,
)


def build_clusters(
    candidates: list[DiffCandidate],
    alignment_transform: np.ndarray,
    config: ClusteringConfig,
    rules: tuple[SeverityRule, ...] = DEFAULT_RULES,
) -> list[DiffCluster]:
    """Cluster `candidates`, rate each cluster, and bind it to both files' coordinates."""
    with StageTimer(logger, "aggregation"):
        clusters = []
        for members in cluster_candidates(
            candidates,
            config.distance_px,
            config.merge_overlap_ratio,
            config.merge_max_area_ratio,
        ):
            box = union_box(members)
            clusters.append(
                DiffCluster(
                    id=0,  # numbered below, once the reading order is known
                    bbox_in_original=box,
                    bbox_in_suspect=original_to_suspect(box, alignment_transform),
                    type=_dominant_type(members),
                    severity=worst_severity(members, rules),
                    members=members,
                    detail=_summarise(members),
                )
            )

        # `cluster_candidates` already returns reading order; numbering here
        # gives every finding a stable id that matches what the reviewer sees.
        for number, cluster in enumerate(clusters, start=1):
            cluster.id = number

    logger.info(
        "aggregation finished",
        extra={"candidates": len(candidates), "clusters": len(clusters)},
    )
    return clusters


def _dominant_type(members: list[DiffCandidate]) -> DiffType:
    present = {member.type for member in members}
    return next(diff_type for diff_type in TYPE_PRIORITY if diff_type in present)


def _summarise(members: list[DiffCandidate]) -> str:
    """One line a reviewer can read without opening the finding.

    Each kind of difference that went into the finding contributes its
    strongest description, with a count when there were several.
    """
    parts = []
    for diff_type in TYPE_PRIORITY:
        group = [member for member in members if member.type is diff_type]
        if not group:
            continue
        strongest = max(group, key=lambda member: (member.magnitude, member.confidence))
        parts.append(strongest.detail if len(group) == 1 else f"{strongest.detail} \u00d7{len(group)}")
    return "; ".join(parts)
