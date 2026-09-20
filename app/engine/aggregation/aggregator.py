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
            (config.moved_displacement_tolerance_px, config.moved_displacement_tolerance_share),
            config.min_fill_ratio,
            config.split_min_area_px,
            config.min_link_px,
            config.huge_candidate_area_px,
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


def add_member(
    cluster: DiffCluster, member: DiffCandidate, rules: tuple[SeverityRule, ...] = DEFAULT_RULES
) -> None:
    """Add what a detector found about a finding after it was clustered (the words its pixels show, say):
    its type, severity and one-line description are worked out again with the new member in."""
    cluster.members.append(member)
    cluster.type = _dominant_type(cluster.members)
    cluster.severity = worst_severity(cluster.members, rules)
    cluster.detail = _summarise(cluster.members)


def merge_clusters(clusters: list[DiffCluster], rules: tuple[SeverityRule, ...] = DEFAULT_RULES) -> DiffCluster:
    """One finding from several the reviewer chose to treat as one (the bars of a barcode, a cluster of small changes).

    It has the members of all of them, a box that covers them all in both files, the type the most specific of them
    gives it and the severity of the worst; it keeps the lowest number, so the numbering the reviewer has been reading
    does not shift under them.
    """
    if len(clusters) < 2:
        raise ValueError("merging takes at least two findings")
    ordered = sorted(clusters, key=lambda cluster: cluster.id)
    members = [member for cluster in ordered for member in cluster.members]
    original_box, suspect_box = ordered[0].bbox_in_original, ordered[0].bbox_in_suspect
    for cluster in ordered[1:]:
        original_box = original_box.union(cluster.bbox_in_original)
        suspect_box = suspect_box.union(cluster.bbox_in_suspect)
    return DiffCluster(
        id=ordered[0].id,
        bbox_in_original=original_box,
        bbox_in_suspect=suspect_box,
        type=_dominant_type(members),
        severity=worst_severity(members, rules),
        members=members,
        detail=f"{len(ordered)} findings merged: {_summarise(members)}",
    )


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
