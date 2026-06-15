"""Geometry-based layout criticism."""

from __future__ import annotations

from itertools import combinations
from math import hypot

from .models import BBox, CriticReport, CriticViolation, LayoutPlan, Point


MAX_VIEWBOX_AREA = 1_200_000


def critique_layout(layout: LayoutPlan) -> CriticReport:
    violations: list[CriticViolation] = []
    component_bbox = _union([component.bbox for component in layout.components])
    fill_ratio = component_bbox.area() / max(1.0, (layout.width / layout.grid) * (layout.height / layout.grid))
    wire_length = _total_wire_length(layout)
    avg_distance = _average_component_distance(layout)

    _component_overlaps(layout, violations)
    _label_violations(layout, violations)
    _wire_violations(layout, violations)
    _convention_violations(layout, violations)

    canvas_area = layout.width * layout.height
    if canvas_area > MAX_VIEWBOX_AREA:
        violations.append(CriticViolation("viewbox_too_large", "Canvas area is above the bounded layout limit.", 30, True))
    if fill_ratio < 0.025:
        violations.append(CriticViolation("fill_ratio_low", "Drawing occupies too little of the fixed evaluation frame.", 18))
    if avg_distance > 7.5:
        violations.append(CriticViolation("spread_excessive", "Average component distance is too large.", 12))
    if wire_length > max(18.0, len(layout.components) * 4.8):
        violations.append(CriticViolation("wire_length_high", "Total wire length is excessive.", 8))

    metrics = {
        "component_fill_ratio": fill_ratio,
        "total_wire_length": wire_length,
        "average_component_distance": avg_distance,
        "viewbox_area": canvas_area,
        "component_bbox_area": component_bbox.area(),
    }
    score = sum(violation.severity for violation in violations)
    return CriticReport(score=score, violations=violations, metrics=metrics, hard_fail=any(v.hard for v in violations))


def _component_overlaps(layout: LayoutPlan, violations: list[CriticViolation]) -> None:
    for left, right in combinations(layout.components, 2):
        if left.bbox.intersects(right.bbox, padding=0.05):
            violations.append(
                CriticViolation(
                    "component_overlap",
                    f"{left.id} overlaps {right.id}.",
                    50,
                    True,
                    subject=f"{left.id},{right.id}",
                )
            )


def _label_violations(layout: LayoutPlan, violations: list[CriticViolation]) -> None:
    for label in layout.labels:
        if label.bbox.x < 0 or label.bbox.y < 0 or label.bbox.right > layout.width / layout.grid or label.bbox.bottom > layout.height / layout.grid:
            violations.append(CriticViolation("label_outside_canvas", f"{label.id} is outside canvas.", 12, subject=label.id))
        for component in layout.components:
            if component.id != label.owner_id and label.bbox.intersects(component.bbox, padding=0.05):
                violations.append(
                    CriticViolation("label_component_overlap", f"{label.id} overlaps {component.id}.", 16, subject=label.id)
                )
        for wire in layout.wires:
            if _polyline_intersects_bbox(wire.points, label.bbox.expanded(0.03)):
                violations.append(CriticViolation("label_wire_overlap", f"{label.id} sits on net {wire.net}.", 9, subject=label.id))


def _wire_violations(layout: LayoutPlan, violations: list[CriticViolation]) -> None:
    for wire in layout.wires:
        for component in layout.components:
            connected = {component_id for component_id, _ in wire.connected_pins}
            if component.id in connected:
                continue
            if _polyline_intersects_bbox(wire.points, component.bbox.expanded(-0.08)):
                hard = "op" in component.type.lower()
                violations.append(
                    CriticViolation(
                        "wire_through_component",
                        f"Net {wire.net} crosses {component.id} body.",
                        35 if hard else 18,
                        hard,
                        subject=f"{wire.net}:{component.id}",
                    )
                )
    crossings = _wire_crossings(layout)
    if crossings:
        violations.append(CriticViolation("wire_crossings", f"{crossings} wire crossings detected.", crossings * 3))


def _convention_violations(layout: LayoutPlan, violations: list[CriticViolation]) -> None:
    inputs = [component for component in layout.components if _key(component.type) in {"input", "source", "voltage_source"}]
    outputs = [component for component in layout.components if _key(component.type) == "output"]
    if inputs and outputs and min(output.x for output in outputs) <= max(item.x for item in inputs):
        violations.append(CriticViolation("output_not_right", "Output is not to the right of input.", 8))
    for component in layout.components:
        key = _key(component.type)
        if ("op_amp" in key or "opamp" in key) and component.orientation != "right":
            violations.append(CriticViolation("opamp_orientation", f"{component.id} is not right-facing.", 10, subject=component.id))
        if key in {"ground", "gnd"}:
            median_y = sorted(item.y for item in layout.components)[len(layout.components) // 2]
            if component.y < median_y:
                violations.append(CriticViolation("ground_not_low", f"{component.id} is not below signal path.", 7, subject=component.id))
        if "resistor" in key and "feedback" in _key(component.role) and component.y > _opamp_y(layout):
            violations.append(CriticViolation("feedback_not_above", f"{component.id} feedback resistor is not above op amp.", 8, subject=component.id))


def _wire_crossings(layout: LayoutPlan) -> int:
    segments: list[tuple[str, Point, Point]] = []
    for wire in layout.wires:
        segments.extend((wire.net, a, b) for a, b in zip(wire.points, wire.points[1:]))
    count = 0
    for left, right in combinations(segments, 2):
        if left[0] == right[0]:
            continue
        if _segments_intersect(left[1], left[2], right[1], right[2]):
            count += 1
    return count


def _polyline_intersects_bbox(points: list[Point], bbox: BBox) -> bool:
    for start, end in zip(points, points[1:]):
        if bbox.contains_point(start) or bbox.contains_point(end):
            return True
        edges = [
            (Point(bbox.x, bbox.y), Point(bbox.right, bbox.y)),
            (Point(bbox.right, bbox.y), Point(bbox.right, bbox.bottom)),
            (Point(bbox.right, bbox.bottom), Point(bbox.x, bbox.bottom)),
            (Point(bbox.x, bbox.bottom), Point(bbox.x, bbox.y)),
        ]
        if any(_segments_intersect(start, end, edge_start, edge_end) for edge_start, edge_end in edges):
            return True
    return False


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    def orient(p: Point, q: Point, r: Point) -> float:
        return (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)

    return orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0


def _total_wire_length(layout: LayoutPlan) -> float:
    return sum(hypot(b.x - a.x, b.y - a.y) for wire in layout.wires for a, b in zip(wire.points, wire.points[1:]))


def _average_component_distance(layout: LayoutPlan) -> float:
    pairs = list(combinations(layout.components, 2))
    if not pairs:
        return 0.0
    return sum(hypot(left.x - right.x, left.y - right.y) for left, right in pairs) / len(pairs)


def _union(boxes: list[BBox]) -> BBox:
    if not boxes:
        return BBox(0, 0, 0, 0)
    x = min(box.x for box in boxes)
    y = min(box.y for box in boxes)
    right = max(box.right for box in boxes)
    bottom = max(box.bottom for box in boxes)
    return BBox(x, y, right - x, bottom - y)


def _opamp_y(layout: LayoutPlan) -> float:
    for component in layout.components:
        if "op" in component.type.lower():
            return component.y
    return 0.0


def _key(value: str | None) -> str:
    return (value or "").lower().replace("-", "_").replace(" ", "_")
