"""Rendered-geometry critic for schematic layout quality."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import hypot
from typing import Any

from .models import BBox, ComponentLayout, LayoutPlan, Point, RenderGeometry, RenderResult, WireSegment


@dataclass
class CriticViolation:
    code: str
    message: str
    penalty: int
    severity: str
    entities: list[str] = field(default_factory=list)


@dataclass
class CriticReport:
    total_score: int
    violations: list[CriticViolation]
    fatal_count: int
    warning_count: int
    suggestions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_score": self.total_score,
            "violations": [asdict(violation) for violation in self.violations],
            "fatal_count": self.fatal_count,
            "warning_count": self.warning_count,
            "suggestions": list(self.suggestions),
        }


def _type_key(component: ComponentLayout) -> str:
    return component.type.lower().replace("-", "_").replace(" ", "_")


def _role_key(component: ComponentLayout) -> str:
    return (component.role or "").lower().replace("-", "_").replace(" ", "_")


def _is_opamp(component: ComponentLayout) -> bool:
    key = _type_key(component)
    return "op_amp" in key or "opamp" in key or "operational_amplifier" in key


def _is_ground(component: ComponentLayout) -> bool:
    role = _role_key(component)
    return "ground" in _type_key(component) or role in {"ground", "gnd", "ground_symbol"}


def _is_feedback_resistor(component: ComponentLayout) -> bool:
    key = _type_key(component)
    return ("resistor" in key or component.type.lower().startswith("r")) and "feedback" in _role_key(component)


def _is_input(component: ComponentLayout) -> bool:
    role = _role_key(component)
    component_type = _type_key(component)
    return role in {"input", "input_source", "input_terminal", "sensor"} or component_type in {
        "input",
        "input_terminal",
    }


def _is_output(component: ComponentLayout) -> bool:
    return "output" in _role_key(component) or _type_key(component) == "output"


def _component_centers(layout_plan: LayoutPlan) -> dict[str, Point]:
    return {
        component.id: Point(
            component.grid_x * layout_plan.grid_size,
            component.grid_y * layout_plan.grid_size,
        )
        for component in layout_plan.components
    }


def _same_point(a: Point, b: Point, epsilon: float = 1.0) -> bool:
    return abs(a.x - b.x) <= epsilon and abs(a.y - b.y) <= epsilon


def _segment_hits_bbox(segment: WireSegment, bbox: BBox) -> bool:
    rect = bbox.inset(1.0)
    start = segment.start
    end = segment.end
    if rect.width <= 0 or rect.height <= 0:
        return False
    if rect.contains_point(start, strict=True) or rect.contains_point(end, strict=True):
        return True
    if abs(start.y - end.y) <= 0.001:
        y = start.y
        if not (rect.y < y < rect.bottom):
            return False
        left = min(start.x, end.x)
        right = max(start.x, end.x)
        return max(left, rect.x) < min(right, rect.right)
    if abs(start.x - end.x) <= 0.001:
        x = start.x
        if not (rect.x < x < rect.right):
            return False
        top = min(start.y, end.y)
        bottom = max(start.y, end.y)
        return max(top, rect.y) < min(bottom, rect.bottom)

    rect_edges = [
        (Point(rect.x, rect.y), Point(rect.right, rect.y)),
        (Point(rect.right, rect.y), Point(rect.right, rect.bottom)),
        (Point(rect.right, rect.bottom), Point(rect.x, rect.bottom)),
        (Point(rect.x, rect.bottom), Point(rect.x, rect.y)),
    ]
    return any(_segments_intersect(start, end, edge_start, edge_end) for edge_start, edge_end in rect_edges)


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y)


def _on_segment(a: Point, b: Point, c: Point) -> bool:
    return (
        min(a.x, c.x) - 0.001 <= b.x <= max(a.x, c.x) + 0.001
        and min(a.y, c.y) - 0.001 <= b.y <= max(a.y, c.y) + 0.001
    )


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)

    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    epsilon = 0.001
    if abs(o1) <= epsilon and _on_segment(a, c, b):
        return True
    if abs(o2) <= epsilon and _on_segment(a, d, b):
        return True
    if abs(o3) <= epsilon and _on_segment(c, a, d):
        return True
    if abs(o4) <= epsilon and _on_segment(c, b, d):
        return True
    return False


def _segments_share_endpoint(a: WireSegment, b: WireSegment) -> bool:
    return (
        _same_point(a.start, b.start)
        or _same_point(a.start, b.end)
        or _same_point(a.end, b.start)
        or _same_point(a.end, b.end)
    )


def _bbox_distance(a: BBox, b: BBox) -> float:
    dx = max(a.x - b.right, b.x - a.right, 0.0)
    dy = max(a.y - b.bottom, b.y - a.bottom, 0.0)
    return hypot(dx, dy)


def _pin_kind(pin_name: str) -> str:
    compact = pin_name.lower().replace("_", "").replace("-", "")
    if pin_name == "-" or compact in {"minus", "inminus", "inv", "inverting", "inn", "vn"}:
        return "inverting"
    if pin_name == "+" or compact in {"plus", "inplus", "noninv", "noninverting", "inp", "vp"}:
        return "non_inverting"
    if compact in {"out", "output", "vo", "vout"}:
        return "output"
    return compact


def _component_pin_nets(component: ComponentLayout) -> dict[str, str]:
    return {pin.pin_name: pin.net_name for pin in component.pins}


def _add(
    violations: list[CriticViolation],
    suggestions: list[str],
    code: str,
    message: str,
    penalty: int,
    entities: list[str],
    suggestion: str | None = None,
) -> None:
    severity = "fatal" if penalty >= 800 else "warning"
    violations.append(
        CriticViolation(
            code=code,
            message=message,
            penalty=penalty,
            severity=severity,
            entities=entities,
        )
    )
    if suggestion and suggestion not in suggestions:
        suggestions.append(suggestion)


def critique_layout(
    layout_plan: LayoutPlan,
    rendered: RenderResult | RenderGeometry,
) -> CriticReport:
    """Score a layout using rendered pixel geometry."""

    geometry = rendered.geometry if isinstance(rendered, RenderResult) else rendered
    components_by_id = {component.id: component for component in layout_plan.components}
    centers = _component_centers(layout_plan)
    violations: list[CriticViolation] = []
    suggestions: list[str] = []

    component_items = sorted(geometry.component_bboxes.items())
    for index, (left_id, left_bbox) in enumerate(component_items):
        for right_id, right_bbox in component_items[index + 1 :]:
            if left_bbox.intersects(right_bbox):
                _add(
                    violations,
                    suggestions,
                    "component_overlap",
                    f"Components {left_id} and {right_id} overlap.",
                    1000,
                    [left_id, right_id],
                    "Move overlapping components to separate grid slots.",
                )
            else:
                distance = _bbox_distance(left_bbox, right_bbox)
                if 0 < distance < 10:
                    _add(
                        violations,
                        suggestions,
                        "too_small_spacing",
                        f"Components {left_id} and {right_id} are too close.",
                        50,
                        [left_id, right_id],
                        "Increase spacing between neighboring component bodies.",
                    )

    for segment in geometry.wire_segments:
        if abs(segment.start.x - segment.end.x) > 0.001 and abs(segment.start.y - segment.end.y) > 0.001:
            _add(
                violations,
                suggestions,
                "diagonal_wire",
                f"Wire on net {segment.net_name} is diagonal.",
                100,
                [segment.net_name],
                "Use orthogonal wire waypoints.",
            )

        for component_id, bbox in geometry.component_bboxes.items():
            component = components_by_id[component_id]
            pin_nets = _component_pin_nets(component)
            if _segment_hits_bbox(segment, bbox):
                own_pin_refs = [
                    f"{component_id}.{pin_name}"
                    for pin_name, net_name in pin_nets.items()
                    if net_name == segment.net_name
                ]
                own_pin_contact = any(
                    pin_ref in geometry.pin_points
                    and (
                        _same_point(segment.start, geometry.pin_points[pin_ref])
                        or _same_point(segment.end, geometry.pin_points[pin_ref])
                    )
                    for pin_ref in own_pin_refs
                )
                if own_pin_contact:
                    continue
                _add(
                    violations,
                    suggestions,
                    "wire_crosses_component_body",
                    f"Wire on net {segment.net_name} crosses body of {component_id}.",
                    800,
                    [segment.net_name, component_id],
                    "Route wires around component bodies except at their pin anchors.",
                )

    for label_id, label_bbox in sorted(geometry.label_bboxes.items()):
        for component_id, component_bbox in sorted(geometry.component_bboxes.items()):
            if label_bbox.intersects(component_bbox):
                _add(
                    violations,
                    suggestions,
                    "label_overlaps_component",
                    f"Label {label_id} overlaps component {component_id}.",
                    300,
                    [label_id, component_id],
                    "Move labels outside component bodies.",
                )
        for segment in geometry.wire_segments:
            if _segment_hits_bbox(segment, label_bbox):
                _add(
                    violations,
                    suggestions,
                    "label_overlaps_wire",
                    f"Label {label_id} overlaps wire on net {segment.net_name}.",
                    250,
                    [label_id, segment.net_name],
                    "Offset labels away from routed wires.",
                )

    for component in layout_plan.components:
        if _is_opamp(component) and component.orientation != "right":
            _add(
                violations,
                suggestions,
                "opamp_not_facing_right",
                f"Op-amp {component.id} is not facing right.",
                200,
                [component.id],
                "Orient op-amps to the right for textbook signal flow.",
            )
        if _is_ground(component):
            if component.orientation != "down":
                _add(
                    violations,
                    suggestions,
                    "ground_not_down",
                    f"Ground {component.id} is not oriented down.",
                    200,
                    [component.id],
                    "Orient ground symbols down.",
                )
            for pin in component.pins:
                pin_ref = f"{component.id}.{pin.pin_name}"
                ground_point = geometry.pin_points.get(pin_ref)
                other_refs = [
                    ref for ref in layout_plan.net_to_pins.get(pin.net_name, []) if ref != pin_ref
                ]
                other_points = [
                    geometry.pin_points[ref] for ref in other_refs if ref in geometry.pin_points
                ]
                if ground_point and other_points and ground_point.y <= min(point.y for point in other_points):
                    _add(
                        violations,
                        suggestions,
                        "ground_not_below_node",
                        f"Ground {component.id} is not below its connected node.",
                        200,
                        [component.id, pin.net_name],
                        "Place ground symbols below the node they reference.",
                    )

    input_centers = [centers[item.id] for item in layout_plan.components if _is_input(item)]
    output_centers = [centers[item.id] for item in layout_plan.components if _is_output(item)]
    if input_centers and output_centers:
        if min(point.x for point in output_centers) <= max(point.x for point in input_centers):
            _add(
                violations,
                suggestions,
                "output_not_right_of_input",
                "Output is not generally to the right of inputs.",
                150,
                ["inputs", "outputs"],
                "Place input sources on the left and output terminals on the right.",
            )

    opamps = [component for component in layout_plan.components if _is_opamp(component)]
    for resistor in [component for component in layout_plan.components if _is_feedback_resistor(component)]:
        resistor_nets = set(_component_pin_nets(resistor).values())
        for opamp in opamps:
            opamp_nets = _component_pin_nets(opamp)
            inv_net = next(
                (net for pin_name, net in opamp_nets.items() if _pin_kind(pin_name) == "inverting"),
                None,
            )
            out_net = next(
                (net for pin_name, net in opamp_nets.items() if _pin_kind(pin_name) == "output"),
                None,
            )
            if inv_net in resistor_nets and out_net in resistor_nets:
                if centers[resistor.id].y >= centers[opamp.id].y:
                    _add(
                        violations,
                        suggestions,
                        "feedback_resistor_not_above_opamp",
                        f"Feedback resistor {resistor.id} is not above op-amp {opamp.id}.",
                        150,
                        [resistor.id, opamp.id],
                        "Place feedback resistors above their op-amps.",
                    )

    wire_segments = geometry.wire_segments
    for index, left in enumerate(wire_segments):
        for right in wire_segments[index + 1 :]:
            if left.net_name == right.net_name or _segments_share_endpoint(left, right):
                continue
            if _segments_intersect(left.start, left.end, right.start, right.end):
                _add(
                    violations,
                    suggestions,
                    "wire_crossing",
                    f"Wire on {left.net_name} crosses wire on {right.net_name}.",
                    100,
                    [left.net_name, right.net_name],
                    "Separate unrelated nets with additional orthogonal waypoints.",
                )

    total_score = sum(violation.penalty for violation in violations)
    fatal_count = sum(1 for violation in violations if violation.severity == "fatal")
    warning_count = len(violations) - fatal_count

    if total_score == 0:
        suggestions.append("Layout follows the zero-penalty schematic conventions.")
        if len(opamps) >= 3:
            suggestions.append("Instrumentation amplifier symmetry is preserved.")

    return CriticReport(
        total_score=total_score,
        violations=violations,
        fatal_count=fatal_count,
        warning_count=warning_count,
        suggestions=suggestions,
    )
