"""Click-to-trace dieline polygon extraction (SRS FR-4.1/FR-4.2, Architecture §3.3.2).

Given a single click near a dieline, this walks the connected boundary --
vector path segments for a digital PDF, connected contour pixels for a
raster-only source -- back to the start point, and returns the enclosed
polygon. This is a graph-topology traversal ("follow the border and loop
back"), not a segmentation heuristic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import fitz
import numpy as np

Point = tuple[float, float]

_SNAP_DECIMALS = 2  # merge segment endpoints within ~0.01pt as the same graph node
_CURVE_SAMPLES = 8  # straight-line segments used to approximate one bezier curve
_MAX_WALK_STEPS = 20_000  # safety cap against a malformed/self-crossing path set
_MAX_RASTER_CLICK_DISTANCE_PX = 25  # how far a click may be from a contour and still hit it


class NoDielineFoundError(Exception):
    """No closed boundary loop could be traced from the given click point."""


@dataclass
class TracedPolygon:
    points: list[Point]  # ordered closed loop, in the same coordinate space as the input


# ---------------------------------------------------------------------------
# Vector (PDF) tracing
# ---------------------------------------------------------------------------


def _snap(point: Point) -> Point:
    return (round(point[0], _SNAP_DECIMALS), round(point[1], _SNAP_DECIMALS))


def _bezier_sample(p0: Point, p1: Point, p2: Point, p3: Point, n: int = _CURVE_SAMPLES) -> list[Point]:
    points = []
    for i in range(n + 1):
        t = i / n
        mt = 1 - t
        x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
        points.append((x, y))
    return points


def _flatten_drawings_to_segments(drawings: list[dict]) -> list[tuple[Point, Point]]:
    """Turn PyMuPDF's get_drawings() output into flat straight-line segments.

    Curves are approximated by short chords so the graph walk below only
    ever deals with straight edges; a smooth curved dieline corner becomes
    several degree-2 nodes instead of one, which the walk handles the same
    way as a polyline corner.
    """
    segments: list[tuple[Point, Point]] = []
    for path in drawings:
        for item in path.get("items", []):
            kind = item[0]
            if kind == "l":
                p1, p2 = item[1], item[2]
                segments.append(((p1.x, p1.y), (p2.x, p2.y)))
            elif kind == "c":
                p1, p2, p3, p4 = item[1], item[2], item[3], item[4]
                pts = _bezier_sample((p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y), (p4.x, p4.y))
                segments.extend(zip(pts, pts[1:]))
            elif kind == "re":
                rect = item[1]
                corners = [
                    (rect.x0, rect.y0),
                    (rect.x1, rect.y0),
                    (rect.x1, rect.y1),
                    (rect.x0, rect.y1),
                ]
                segments.extend(zip(corners, corners[1:] + corners[:1]))
            elif kind == "qu":
                quad = item[1]
                corners = [
                    (quad.ul.x, quad.ul.y),
                    (quad.ur.x, quad.ur.y),
                    (quad.lr.x, quad.lr.y),
                    (quad.ll.x, quad.ll.y),
                ]
                segments.extend(zip(corners, corners[1:] + corners[:1]))
    return segments


def _build_graph(segments: list[tuple[Point, Point]]) -> dict[Point, list[Point]]:
    graph: dict[Point, list[Point]] = {}
    for a, b in segments:
        a, b = _snap(a), _snap(b)
        if a == b:
            continue
        graph.setdefault(a, []).append(b)
        graph.setdefault(b, []).append(a)
    return graph


def _point_to_segment_distance(p: Point, a: Point, b: Point) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def _nearest_segment(
    segments: list[tuple[Point, Point]], click: Point
) -> tuple[Point, Point] | None:
    best, best_dist = None, math.inf
    for a, b in segments:
        dist = _point_to_segment_distance(click, a, b)
        if dist < best_dist:
            best, best_dist = (a, b), dist
    return best


def _straightest_continuation(prev: Point, current: Point, candidates: list[Point]) -> Point:
    in_angle = math.atan2(current[1] - prev[1], current[0] - prev[0])

    def turn(candidate: Point) -> float:
        out_angle = math.atan2(candidate[1] - current[1], candidate[0] - current[0])
        return abs(((out_angle - in_angle) + math.pi) % (2 * math.pi) - math.pi)

    return min(candidates, key=turn)


def _walk_loop(
    graph: dict[Point, list[Point]], start: Point, first_next: Point
) -> list[Point] | None:
    """Follow the boundary from `start` through `first_next`, always continuing
    in the straightest available direction, until back at `start`."""
    loop = [start]
    prev, current = start, first_next
    for _ in range(_MAX_WALK_STEPS):
        loop.append(current)
        if current == start:
            return loop
        # Deliberately exclude `prev` even as a fallback: a dead end (no
        # forward neighbor) must fail the walk, not bounce back and forth
        # and falsely report an open path as a closed loop.
        candidates = [n for n in graph.get(current, []) if n != prev]
        if not candidates:
            return None
        nxt = _straightest_continuation(prev, current, candidates)
        prev, current = current, nxt
    return None


def trace_vector_dieline(page: fitz.Page, click_point: Point) -> TracedPolygon:
    """Trace the closed dieline loop nearest `click_point` (PDF point-space) on `page`."""
    segments = _flatten_drawings_to_segments(page.get_drawings())
    if not segments:
        raise NoDielineFoundError("No vector paths found on this page.")
    nearest = _nearest_segment(segments, click_point)
    if nearest is None:
        raise NoDielineFoundError("No vector path found near the clicked point.")
    a, b = _snap(nearest[0]), _snap(nearest[1])
    graph = _build_graph(segments)

    loop = _walk_loop(graph, a, b) or _walk_loop(graph, b, a)
    if loop is None:
        raise NoDielineFoundError("The path near the clicked point does not form a closed loop.")
    return TracedPolygon(points=loop[:-1])  # drop the duplicated closing point


# ---------------------------------------------------------------------------
# Raster (image) tracing
# ---------------------------------------------------------------------------


def trace_raster_dieline(image: np.ndarray, click_point: tuple[float, float]) -> TracedPolygon:
    """Contour-based trace for raster-only sources.

    `cv2.findContours` already returns an ordered walk of connected boundary
    pixels, so tracing here is "find the contour nearest the click" rather
    than a separate walk implementation.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise NoDielineFoundError("No closed contours found in the image.")

    click = (float(click_point[0]), float(click_point[1]))

    def distance_to_contour(contour: np.ndarray) -> float:
        return abs(cv2.pointPolygonTest(contour, click, True))

    nearest = min(contours, key=distance_to_contour)
    if distance_to_contour(nearest) > _MAX_RASTER_CLICK_DISTANCE_PX:
        raise NoDielineFoundError("No closed contour found near the clicked point.")
    points = [(float(p[0][0]), float(p[0][1])) for p in nearest]
    return TracedPolygon(points=points)
