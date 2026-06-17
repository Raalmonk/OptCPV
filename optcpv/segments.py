"""Wire segment helpers shared by renderers and critics."""

from __future__ import annotations

from .models import LayoutPlan, Point


EPSILON = 1e-6


def merged_axis_aligned_segments(points: list[Point]) -> list[tuple[Point, Point]]:
    """Return drawable segments with same-line overlaps collapsed.

    Layout routes are often stored as traversal paths through a net graph. That
    representation can visit the same bus repeatedly, which is useful for
    topology but ugly when rendered as visible wires. This helper turns those
    traversals into the visual set of line segments.
    """

    horizontal: dict[float, list[tuple[float, float]]] = {}
    vertical: dict[float, list[tuple[float, float]]] = {}
    other: list[tuple[Point, Point]] = []
    seen: set[tuple[tuple[float, float], tuple[float, float]]] = set()

    for start, end in zip(points, points[1:]):
        if start == end:
            continue
        start_key = _point_key(start)
        end_key = _point_key(end)
        key = (start_key, end_key) if start_key <= end_key else (end_key, start_key)
        if key in seen:
            continue
        seen.add(key)

        if _same(start.y, end.y):
            y = round(start.y, 4)
            horizontal.setdefault(y, []).append((start.x, end.x))
        elif _same(start.x, end.x):
            x = round(start.x, 4)
            vertical.setdefault(x, []).append((start.y, end.y))
        else:
            other.append((start, end))

    merged: list[tuple[Point, Point]] = []
    for y, intervals in sorted(horizontal.items()):
        for start, end in _merge_intervals(intervals):
            merged.append((Point(start, y), Point(end, y)))
    for x, intervals in sorted(vertical.items()):
        for start, end in _merge_intervals(intervals):
            merged.append((Point(x, start), Point(x, end)))
    merged.extend(other)
    return merged


def layout_wire_segments(layout: LayoutPlan) -> list[tuple[str, Point, Point]]:
    return [
        (wire.net, start, end)
        for wire in layout.wires
        for start, end in merged_axis_aligned_segments(wire.points)
    ]


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    normalized = sorted((min(start, end), max(start, end)) for start, end in intervals)
    if not normalized:
        return []
    merged: list[tuple[float, float]] = []
    current_start, current_end = normalized[0]
    for start, end in normalized[1:]:
        if start <= current_end + EPSILON:
            current_end = max(current_end, end)
            continue
        merged.append((current_start, current_end))
        current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


def _point_key(point: Point) -> tuple[float, float]:
    return (round(point.x, 4), round(point.y, 4))


def _same(left: float, right: float) -> bool:
    return abs(left - right) < EPSILON
