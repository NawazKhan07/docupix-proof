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
from app.models.diff import DiffCandidate, DiffCluster

logger = logging.getLogger(__name__)


def build_clusters(
    candidates: list[DiffCandidate],
    alignment_transform: np.ndarray,
    config: ClusteringConfig,
    rules: tuple[SeverityRule, ...] = DEFAULT_RULES,
) -> list[DiffCluster]:
    """Cluster `candidates`, rate each cluster, and bind it to both files' coordinates."""
    with StageTimer(logger, "aggregation"):
        clusters = []
        for members in cluster_candidates(candidates, config.distance_px):
            box = union_box(members)
            clusters.append(
                DiffCluster(
                    id=0,  # numbered below, once the reading order is known
                    bbox_in_original=box,
                    bbox_in_suspect=original_to_suspect(box, alignment_transform),
                    type=members[0].type,
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


def _summarise(members: list[DiffCandidate]) -> str:
    """One line a reviewer can read without opening the finding."""
    first = members[0].detail
    if len(members) == 1:
        return first
    return f"{len(members)} {members[0].type.value} differences, e.g. {first}"
