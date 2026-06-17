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


def junction_points(points: list[Point]) -> list[Point]:
    """Return true branch junctions from a routed net traversal.

    A route traversal can revisit waypoints while walking a tree-shaped net.
    After visual segment merging, those waypoints may sit inside a longer line,
    but they are not necessarily electrical junctions. Count the incident
    visual directions at each original waypoint so bends and entry stubs do not
    get false junction dots.
    """

    segments = merged_axis_aligned_segments(points)
    candidates: dict[tuple[float, float], Point] = {}
    for point in points:
        candidates[_point_key(point)] = point
    for start, end in segments:
        candidates[_point_key(start)] = start
        candidates[_point_key(end)] = end

    junctions = []
    for point in candidates.values():
        if _incident_direction_count(point, segments) >= 3:
            junctions.append(point)
    return sorted(junctions, key=lambda point: (point.y, point.x))


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


def _incident_direction_count(point: Point, segments: list[tuple[Point, Point]]) -> int:
    directions: set[tuple[int, int]] = set()
    for start, end in segments:
        if not _point_on_segment(point, start, end):
            continue
        if _same(start.x, end.x):
            if point.y > min(start.y, end.y) + EPSILON:
                directions.add((0, -1))
            if point.y < max(start.y, end.y) - EPSILON:
                directions.add((0, 1))
        elif _same(start.y, end.y):
            if point.x > min(start.x, end.x) + EPSILON:
                directions.add((-1, 0))
            if point.x < max(start.x, end.x) - EPSILON:
                directions.add((1, 0))
        else:
            directions.add(_diagonal_direction(point, start))
            directions.add(_diagonal_direction(point, end))
    return len(directions)


def _point_on_segment(point: Point, start: Point, end: Point) -> bool:
    if _same(start.x, end.x):
        return _same(point.x, start.x) and _between(point.y, start.y, end.y)
    if _same(start.y, end.y):
        return _same(point.y, start.y) and _between(point.x, start.x, end.x)
    cross = (point.x - start.x) * (end.y - start.y) - (point.y - start.y) * (end.x - start.x)
    if abs(cross) > EPSILON:
        return False
    return _between(point.x, start.x, end.x) and _between(point.y, start.y, end.y)


def _between(value: float, left: float, right: float) -> bool:
    return min(left, right) - EPSILON <= value <= max(left, right) + EPSILON


def _diagonal_direction(point: Point, other: Point) -> tuple[int, int]:
    dx = 0 if _same(point.x, other.x) else 1 if other.x > point.x else -1
    dy = 0 if _same(point.y, other.y) else 1 if other.y > point.y else -1
    return (dx, dy)


def _same(left: float, right: float) -> bool:
    return abs(left - right) < EPSILON
