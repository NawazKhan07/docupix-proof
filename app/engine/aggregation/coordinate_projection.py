"""Projecting a difference into both files' native coordinates (SRS FR-7.4,
Architecture §3.3.6).

The shared aligned space is the *original's* pixel frame, and the alignment
transform maps suspect pixels into it. So a box found in the shared space is
already the box in the original; the suspect's own box is that box carried
back through the inverse transform. Both come from the one transform the
alignment stage measured -- nothing here re-aligns anything, which is what
stops the two views ever disagreeing about where a difference is.

A box is not generally preserved by a rotation: its four corners map to a
tilted quadrilateral, and the box returned is the axis-aligned box that
encloses it. For a pure translation that is exact; under rotation it is
slightly larger than the true region (never smaller), which is the safe
direction for a marker that must not miss what it points at.
"""

from __future__ import annotations

import numpy as np

from app.engine.alignment.homography import apply_transform
from app.models.diff import BoundingBox


def project_box(box: BoundingBox, transform: np.ndarray) -> BoundingBox:
    """The axis-aligned box enclosing `box` after projecting it through `transform`."""
    corners = np.array(
        [[box.x0, box.y0], [box.x1, box.y0], [box.x1, box.y1], [box.x0, box.y1]],
        dtype=np.float64,
    )
    projected = apply_transform(transform, corners)
    return BoundingBox(
        float(projected[:, 0].min()),
        float(projected[:, 1].min()),
        float(projected[:, 0].max()),
        float(projected[:, 1].max()),
    )


def original_to_suspect(box_in_original: BoundingBox, alignment_transform: np.ndarray) -> BoundingBox:
    """Where a shared-space (original-frame) box sits in the suspect's own pixels."""
    return project_box(box_in_original, np.linalg.inv(alignment_transform))


def suspect_to_original(box_in_suspect: BoundingBox, alignment_transform: np.ndarray) -> BoundingBox:
    """Where a box in the suspect's own pixels sits in the shared (original) frame."""
    return project_box(box_in_suspect, alignment_transform)
