"""The Diff Aggregator (Architecture §3.3.6): raw candidates -> findings.

Composes the two aggregation steps -- cluster, bind coordinates -- into the `DiffCluster`
list the UI presents. A finding is described (its category, what happened to it), never ranked.
"""

from __future__ import annotations

import logging

import numpy as np

from app.config import ClusteringConfig
from app.engine.aggregation.clustering import cluster_candidates, union_box
from app.engine.aggregation.coordinate_projection import original_to_suspect
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
    panel_area: float | None = None,
) -> list[DiffCluster]:
    """Cluster `candidates` and bind each cluster to both files' coordinates. `panel_area`: see `cluster_candidates`."""
    with StageTimer(logger, "aggregation"):
        clusters = []
        for members in cluster_candidates(
            candidates,
            config.distance_px,
            config.line_overlap,
            config.line_gap_heights,
            config.line_max_height_px,
            config.touch_px,
            config.duplicate_iou,
            (config.moved_displacement_tolerance_px, config.moved_displacement_tolerance_share),
            config.line_max_gap_px,
            panel_area,
        ):
            box = union_box(members)
            # (a thing that moved is shown where it is in the suspect, not where it was)
            moved_to = [member.moved_to for member in members if member.moved_to is not None]
            there = moved_to[0] if moved_to else box
            for other in moved_to[1:]:
                there = there.union(other)
            clusters.append(
                DiffCluster(
                    id=0,  # numbered below, once the reading order is known
                    bbox_in_original=box,
                    bbox_in_suspect=original_to_suspect(there, alignment_transform),
                    type=_dominant_type(members),
                    members=members,
                    detail=_summarise(members),
                    moved_to=there if moved_to else None,
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


def add_member(cluster: DiffCluster, member: DiffCandidate) -> None:
    """Add what a detector found about a finding after it was clustered (the words its pixels show, say):
    its type and one-line description are worked out again with the new member in."""
    cluster.members.append(member)
    cluster.type = _dominant_type(cluster.members)
    cluster.detail = _summarise(cluster.members)


def merge_clusters(clusters: list[DiffCluster]) -> DiffCluster:
    """One finding from several the reviewer chose to treat as one (the bars of a barcode, a cluster of small changes).

    It has the members of all of them, a box that covers them all in both files and the type the most specific of them
    gives it; it keeps the lowest number, so the numbering the reviewer has been reading does not shift under them.
    """
    if len(clusters) < 2:
        raise ValueError("merging takes at least two findings")
    ordered = sorted(clusters, key=lambda cluster: cluster.id)
    members = [member for cluster in ordered for member in cluster.members]
    original_box, suspect_box = ordered[0].bbox_in_original, ordered[0].bbox_in_suspect
    shown = ordered[0].shown_in_suspect
    for cluster in ordered[1:]:
        original_box = original_box.union(cluster.bbox_in_original)
        suspect_box = suspect_box.union(cluster.bbox_in_suspect)
        shown = shown.union(cluster.shown_in_suspect)
    moved = any(cluster.moved_to is not None for cluster in ordered)
    return DiffCluster(
        id=ordered[0].id,
        bbox_in_original=original_box,
        bbox_in_suspect=suspect_box,
        type=_dominant_type(members),
        members=members,
        detail=f"{len(ordered)} findings merged: {_summarise(members)}",
        moved_to=shown if moved else None,
    )


def combined(clusters: list[DiffCluster]) -> DiffCluster:
    """One finding from several that are pieces of one thing (the letters of one line of text): all their members, a box
    round them in both files, the lowest number, and said the way one finding is (not as a merge the reviewer made)."""
    merged = merge_clusters(clusters)
    merged.detail = _summarise(merged.members)
    return merged


def _dominant_type(members: list[DiffCandidate]) -> DiffType:
    present = {member.type for member in members}
    return next(diff_type for diff_type in TYPE_PRIORITY if diff_type in present)


def _summarise(members: list[DiffCandidate]) -> str:
    """One line a reviewer can read without opening the finding: what the most specific kind of difference in it says (a
    changed word before the graphic it is on, a graphic before the colour it also differs in)."""
    for diff_type in TYPE_PRIORITY:
        group = [member for member in members if member.type is diff_type]
        if group:
            return max(group, key=lambda member: (member.magnitude, member.confidence)).detail
    return ""
