"""Hard routing invariants for OptCPV layout wires."""

from __future__ import annotations

from .models import BBox, LayoutPlan, Point
from .segments import is_axis_aligned


def orthogonalize_route(
    points: list[Point],
    keepouts: list[BBox] | tuple[BBox, ...] = (),
    preferred_policy: str | None = None,
) -> list[Point]:
    """Return a Manhattan route while preferring elbows outside keepouts."""

    if len(points) < 2:
        return list(points)
    result = [points[0]]
    for end in points[1:]:
        start = result[-1]
        if start == end:
            continue
        if is_axis_aligned(start, end):
            detour = _axis_aligned_keepout_detour(start, end, keepouts, preferred_policy)
            if detour:
                result.extend(detour)
            else:
                result.append(end)
            continue
        first = Point(end.x, start.y)
        second = Point(start.x, end.y)
        elbow = min(
            (first, second),
            key=lambda candidate: _elbow_score(start, end, candidate, keepouts, preferred_policy),
        )
        if elbow != start:
            result.append(elbow)
        result.append(end)
    return _dedupe(result)


def assert_no_diagonal_wires(layout: LayoutPlan) -> None:
    for wire in layout.wires:
        for start, end in zip(wire.points, wire.points[1:]):
            if not is_axis_aligned(start, end):
                raise ValueError(f"LayoutWire {wire.net} has a non-Manhattan segment.")


def route_crosses_keepout(points: list[Point], keepout: BBox, *, padding: float = 0.0) -> bool:
    box = keepout.expanded(padding) if padding else keepout
    for start, end in zip(points, points[1:]):
        if box.contains_point(start) or box.contains_point(end):
            return True
        if not is_axis_aligned(start, end):
            elbow = Point(end.x, start.y)
            if route_crosses_keepout([start, elbow, end], box):
                return True
            continue
        if _axis_segment_intersects_bbox(start, end, box):
            return True
    return False


def _elbow_score(
    start: Point,
    end: Point,
    elbow: Point,
    keepouts: list[BBox] | tuple[BBox, ...],
    preferred_policy: str | None,
) -> float:
    segments = [(start, elbow), (elbow, end)]
    score = sum(abs(a.x - b.x) + abs(a.y - b.y) for a, b in segments) * 0.01
    for keepout in keepouts:
        if any(_axis_segment_intersects_bbox(a, b, keepout) for a, b in segments):
            score += 100.0
    if preferred_policy in {"top_feedback_corridor", "top"}:
        score += elbow.y * 0.001
    elif preferred_policy in {"bottom_feedback_corridor", "bottom_auxiliary_corridor", "bottom"}:
        score -= elbow.y * 0.001
    return score


def _axis_aligned_keepout_detour(
    start: Point,
    end: Point,
    keepouts: list[BBox] | tuple[BBox, ...],
    preferred_policy: str | None,
) -> list[Point] | None:
    blockers = [bbox for bbox in keepouts if _blocking_keepout(start, end, bbox)]
    if not blockers:
        return None
    if abs(start.y - end.y) < 1e-6:
        return _horizontal_keepout_detour(start, end, blockers, keepouts, preferred_policy)
    if abs(start.x - end.x) < 1e-6:
        return _vertical_keepout_detour(start, end, blockers, keepouts, preferred_policy)
    return None


def _horizontal_keepout_detour(
    start: Point,
    end: Point,
    blockers: list[BBox],
    keepouts: list[BBox] | tuple[BBox, ...],
    preferred_policy: str | None,
) -> list[Point]:
    margin = 0.28
    top_y = min(bbox.y for bbox in blockers) - margin
    bottom_y = max(bbox.bottom for bbox in blockers) + margin
    candidates = [
        [Point(start.x, top_y), Point(end.x, top_y), end],
        [Point(start.x, bottom_y), Point(end.x, bottom_y), end],
    ]
    if preferred_policy in {"top_feedback_corridor", "top"}:
        candidates = candidates[:1] + candidates[1:]
    elif preferred_policy in {"bottom_feedback_corridor", "bottom_auxiliary_corridor", "bottom"}:
        candidates = candidates[1:] + candidates[:1]
    return min(candidates, key=lambda candidate: _detour_score(start, candidate, keepouts, preferred_policy))


def _vertical_keepout_detour(
    start: Point,
    end: Point,
    blockers: list[BBox],
    keepouts: list[BBox] | tuple[BBox, ...],
    preferred_policy: str | None,
) -> list[Point]:
    margin = 0.28
    left_x = min(bbox.x for bbox in blockers) - margin
    right_x = max(bbox.right for bbox in blockers) + margin
    candidates = [
        [Point(left_x, start.y), Point(left_x, end.y), end],
        [Point(right_x, start.y), Point(right_x, end.y), end],
    ]
    return min(candidates, key=lambda candidate: _detour_score(start, candidate, keepouts, preferred_policy))


def _blocking_keepout(start: Point, end: Point, bbox: BBox) -> bool:
    if bbox.width <= 0 or bbox.height <= 0:
        return False
    if not _axis_segment_intersects_bbox(start, end, bbox):
        return False
    return not (bbox.contains_point(start) or bbox.contains_point(end))


def _detour_score(
    start: Point,
    candidate: list[Point],
    keepouts: list[BBox] | tuple[BBox, ...],
    preferred_policy: str | None = None,
) -> float:
    path = [start, *candidate]
    segments = list(zip(path, path[1:]))
    score = sum(abs(a.x - b.x) + abs(a.y - b.y) for a, b in segments) * 0.01
    for keepout in keepouts:
        if keepout.width <= 0 or keepout.height <= 0:
            continue
        for a, b in segments:
            if keepout.contains_point(a) or keepout.contains_point(b):
                score += 25.0
            if _axis_segment_intersects_bbox(a, b, keepout):
                score += 100.0
    if preferred_policy in {"top_feedback_corridor", "top"}:
        score += sum(point.y for point in candidate) * 0.001
    elif preferred_policy in {"bottom_feedback_corridor", "bottom_auxiliary_corridor", "bottom"}:
        score -= sum(point.y for point in candidate) * 0.001
    return score


def _axis_segment_intersects_bbox(start: Point, end: Point, bbox: BBox) -> bool:
    if abs(start.x - end.x) < 1e-6:
        return bbox.x <= start.x <= bbox.right and _intervals_overlap(start.y, end.y, bbox.y, bbox.bottom)
    if abs(start.y - end.y) < 1e-6:
        return bbox.y <= start.y <= bbox.bottom and _intervals_overlap(start.x, end.x, bbox.x, bbox.right)
    return False


def _intervals_overlap(a: float, b: float, c: float, d: float) -> bool:
    return min(max(a, b), max(c, d)) >= max(min(a, b), min(c, d))


def _dedupe(points: list[Point]) -> list[Point]:
    result: list[Point] = []
    for point in points:
        if not result or point != result[-1]:
            result.append(point)
    return result
