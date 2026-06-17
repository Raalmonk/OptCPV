"""Geometry-based layout criticism."""

from __future__ import annotations

from itertools import combinations
from math import hypot

from .models import BBox, CriticReport, CriticViolation, LayoutComponent, LayoutPlan, NetClass, Point
from .segments import layout_wire_segments, merged_axis_aligned_segments
from .semantics import classify_net, is_local_terminal_net
from .symbols import OPAMP_INPUT_LEAD_X, OPAMP_INPUT_LEAD_Y, OPAMP_OUTPUT_LEAD_X


MAX_VIEWBOX_AREA = 1_200_000


def critique_layout(layout: LayoutPlan) -> CriticReport:
    violations: list[CriticViolation] = []
    component_bbox = _union([component.bbox for component in layout.components])
    fill_ratio = component_bbox.area() / max(1.0, (layout.width / layout.grid) * (layout.height / layout.grid))
    wire_length = _total_wire_length(layout)
    avg_distance = _average_component_distance(layout)
    avg_distance_limit = _average_distance_limit(layout)
    wire_length_limit = _wire_length_limit(layout)

    _component_overlaps(layout, violations)
    _label_violations(layout, violations)
    _wire_violations(layout, violations)
    _pin_contract_violations(layout, violations)
    _convention_violations(layout, violations)
    _semantic_violations(layout, violations)

    canvas_area = layout.width * layout.height
    if canvas_area > MAX_VIEWBOX_AREA:
        violations.append(CriticViolation("viewbox_too_large", "Canvas area is above the bounded layout limit.", 30, True))
    if fill_ratio < 0.025:
        violations.append(CriticViolation("fill_ratio_low", "Drawing occupies too little of the fixed evaluation frame.", 18))
    if avg_distance > avg_distance_limit + 0.05:
        violations.append(CriticViolation("spread_excessive", "Average component distance is too large.", 12))
    if wire_length > wire_length_limit:
        violations.append(CriticViolation("wire_length_high", "Total wire length is excessive.", 8))

    metrics = {
        "component_fill_ratio": fill_ratio,
        "total_wire_length": wire_length,
        "average_component_distance": avg_distance,
        "average_component_distance_limit": avg_distance_limit,
        "viewbox_area": canvas_area,
        "component_bbox_area": component_bbox.area(),
        "wire_length_limit": wire_length_limit,
        "semantic_local_terminal_count": len(layout.semantic.local_terminals),
        "semantic_terminal_net_count": sum(1 for net_class in layout.semantic.net_classes.values() if _is_terminal_net_class(net_class)),
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
    overlaps = _wire_net_overlaps(layout)
    if overlaps:
        violations.append(
            CriticViolation(
                "wire_net_overlap",
                f"{overlaps} different-net wire overlaps detected.",
                min(80, overlaps * 20),
                True,
            )
        )
    disconnected = _wire_pin_disconnects(layout)
    if disconnected:
        violations.append(
            CriticViolation(
                "wire_pin_disconnected",
                f"{disconnected} connected pins are not touched by their routed wires.",
                min(90, disconnected * 30),
                True,
            )
        )


def _pin_contract_violations(layout: LayoutPlan, violations: list[CriticViolation]) -> None:
    for component in layout.components:
        component_key = _key(component.type)
        if "op_amp" not in component_key and "opamp" not in component_key and "operational_amplifier" not in component_key:
            continue
        flip = _is_flipped_opamp(component)
        expected = {
            "-": Point(
                component.x + OPAMP_INPUT_LEAD_X,
                component.y + OPAMP_INPUT_LEAD_Y if flip else component.y - OPAMP_INPUT_LEAD_Y,
            ),
            "+": Point(
                component.x + OPAMP_INPUT_LEAD_X,
                component.y - OPAMP_INPUT_LEAD_Y if flip else component.y + OPAMP_INPUT_LEAD_Y,
            ),
            "out": Point(component.x + OPAMP_OUTPUT_LEAD_X, component.y),
        }
        for pin_name in component.pins:
            kind = _pin_kind(pin_name)
            expected_key = kind if kind in {"+", "-"} else "out" if _is_opamp_output_pin(pin_name) else ""
            expected_point = expected.get(expected_key)
            if expected_point is None:
                continue
            actual = layout.pin_map.get((component.id, pin_name))
            if actual is None:
                continue
            if hypot(actual.x - expected_point.x, actual.y - expected_point.y) > 0.03:
                violations.append(
                    CriticViolation(
                        "pin_renderer_mismatch",
                        f"{component.id}.{pin_name} does not match the renderer's op amp lead endpoint.",
                        80,
                        True,
                        subject=f"{component.id}:{pin_name}",
                    )
                )


def _convention_violations(layout: LayoutPlan, violations: list[CriticViolation]) -> None:
    inputs = [component for component in layout.components if _key(component.type) in {"input", "source", "voltage_source"}]
    outputs = [component for component in layout.components if _key(component.type) == "output"]
    if inputs and outputs and min(output.x for output in outputs) <= max(item.x for item in inputs):
        violations.append(CriticViolation("output_not_right", "Output is not to the right of input.", 8))
    for component in layout.components:
        key = _key(component.type)
        if ("op_amp" in key or "opamp" in key) and component.orientation not in {"right", "right_flip"}:
            violations.append(CriticViolation("opamp_orientation", f"{component.id} is not right-facing.", 10, subject=component.id))
        if key in {"ground", "gnd"}:
            median_y = sorted(item.y for item in layout.components)[len(layout.components) // 2]
            if component.y < median_y:
                violations.append(CriticViolation("ground_not_low", f"{component.id} is not below signal path.", 7, subject=component.id))
        if "resistor" in key and "feedback" in _key(component.role):
            owner = _feedback_owner(layout, component)
            owner_y = owner.y if owner is not None else _opamp_y(layout)
            if owner is not None and _is_flipped_opamp(owner):
                if component.y < owner_y:
                    violations.append(
                        CriticViolation(
                            "feedback_not_below",
                            f"{component.id} feedback resistor is not below flipped op amp.",
                            8,
                            subject=component.id,
                        )
                    )
            elif component.y > owner_y:
                violations.append(CriticViolation("feedback_not_above", f"{component.id} feedback resistor is not above op amp.", 8, subject=component.id))


def _semantic_violations(layout: LayoutPlan, violations: list[CriticViolation]) -> None:
    _terminal_wire_violations(layout, violations)
    _local_terminal_violations(layout, violations)
    _feedback_semantic_violations(layout, violations)
    _signal_flow_violations(layout, violations)
    _parallel_lane_violations(layout, violations)
    _filter_block_violations(layout, violations)


def _terminal_wire_violations(layout: LayoutPlan, violations: list[CriticViolation]) -> None:
    terminal_classes = {NetClass.GROUND, NetClass.POSITIVE_SUPPLY, NetClass.NEGATIVE_SUPPLY, NetClass.REFERENCE}
    stage_span_limit = max(2.4, (layout.width / layout.grid) * 0.18)
    for wire in layout.wires:
        net_class = layout.semantic.net_classes.get(wire.net, classify_net(wire.net))
        if net_class not in terminal_classes:
            continue
        span_x = max(point.x for point in wire.points) - min(point.x for point in wire.points)
        span_y = max(point.y for point in wire.points) - min(point.y for point in wire.points)
        length = sum(hypot(end.x - start.x, end.y - start.y) for start, end in zip(wire.points, wire.points[1:]))
        if span_x > stage_span_limit or length > stage_span_limit * 1.4:
            violations.append(
                CriticViolation(
                    "long_global_terminal_wire",
                    f"{wire.net} is routed as a long shared physical wire instead of local terminal symbols.",
                    85,
                    True,
                    subject=wire.net,
                )
            )
        elif span_x > 1.8 or span_y > 1.8:
            violations.append(
                CriticViolation(
                    "terminal_net_routed",
                    f"{wire.net} is routed as a physical wire; terminal nets should stay local by default.",
                    30,
                    False,
                    subject=wire.net,
                )
            )


def _local_terminal_violations(layout: LayoutPlan, violations: list[CriticViolation]) -> None:
    terminal_keys = {(terminal.component_id, terminal.pin_name) for terminal in layout.semantic.local_terminals}
    for key, pin in layout.pin_map.items():
        net_class = layout.semantic.net_classes.get(pin.net, classify_net(pin.net))
        if not _is_terminal_net_class(net_class):
            continue
        owner = _component_by_id(layout, key[0])
        if owner is not None and _is_explicit_terminal_component(owner):
            continue
        if key not in terminal_keys:
            violations.append(
                CriticViolation(
                    "missing_local_terminal",
                    f"{key[0]}.{key[1]} on {pin.net} has no local terminal intent.",
                    45,
                    True,
                    subject=f"{key[0]}:{key[1]}",
                )
            )
    for terminal in layout.semantic.local_terminals:
        pin = layout.pin_map.get((terminal.component_id, terminal.pin_name))
        if pin is None:
            violations.append(
                CriticViolation(
                    "missing_local_terminal_pin",
                    f"Local terminal {terminal.component_id}.{terminal.pin_name} does not map to a pin.",
                    45,
                    True,
                    subject=f"{terminal.component_id}:{terminal.pin_name}",
                )
            )
            continue
        if terminal.preferred_direction == "up" and pin.side == "bottom":
            violations.append(
                CriticViolation(
                    "supply_terminal_wrong_side",
                    f"{terminal.net} local supply terminal is attached below its owning pin.",
                    10,
                    subject=f"{terminal.component_id}:{terminal.pin_name}",
                )
            )


def _feedback_semantic_violations(layout: LayoutPlan, violations: list[CriticViolation]) -> None:
    for opamp in [component for component in layout.components if _is_opamp(component)]:
        output_net = _opamp_output_net(opamp)
        input_nets = _opamp_input_nets(opamp)
        if output_net is None:
            continue
        feedback_nets = {output_net, *input_nets}
        for wire in layout.wires:
            if wire.net not in feedback_nets:
                continue
            if not _polyline_intersects_bbox_interior(wire.points, _opamp_body_bbox(opamp)):
                continue
            violations.append(
                CriticViolation(
                    "feedback_crosses_opamp_body",
                    f"Feedback-related net {wire.net} crosses {opamp.id} body.",
                    45,
                    True,
                    subject=f"{wire.net}:{opamp.id}",
                )
            )


def _signal_flow_violations(layout: LayoutPlan, violations: list[CriticViolation]) -> None:
    component_by_id = {component.id: component for component in layout.components}
    for route in layout.semantic.routes:
        source = component_by_id.get(route.source[0])
        target = component_by_id.get(route.target[0])
        if source is None or target is None or source.id == target.id:
            continue
        if target.x + 0.2 < source.x and not _is_feedback_route(layout, route.net, source, target):
            violations.append(
                CriticViolation(
                    "signal_path_not_left_to_right",
                    f"Signal net {route.net} routes backward from {source.id} to {target.id}.",
                    16,
                    subject=route.net,
                )
            )


def _parallel_lane_violations(layout: LayoutPlan, violations: list[CriticViolation]) -> None:
    input_ports = [component for component in layout.components if _is_input_component(component)]
    if len(input_ports) >= 2:
        spread = max(component.x for component in input_ports) - min(component.x for component in input_ports)
        if spread > 0.55:
            violations.append(CriticViolation("parallel_inputs_not_aligned", "Parallel input ports are not vertically aligned.", 12))
    buffer_motifs = [motif for motif in layout.semantic.motifs if motif.motif_type == "opamp_buffer"]
    buffers = [_component_by_id(layout, motif.component_ids[0]) for motif in buffer_motifs if motif.component_ids]
    buffers = [component for component in buffers if component is not None]
    if len(buffers) >= 2:
        spread = max(component.x for component in buffers) - min(component.x for component in buffers)
        unique_lanes = {round(component.y, 1) for component in buffers}
        if spread > 0.65:
            violations.append(CriticViolation("parallel_buffers_not_same_stage", "Parallel input buffers do not share the same x-stage.", 14))
        if len(unique_lanes) != len(buffers):
            violations.append(CriticViolation("parallel_inputs_collapsed", "Parallel input buffers collapsed onto the same lane.", 18, True))


def _filter_block_violations(layout: LayoutPlan, violations: list[CriticViolation]) -> None:
    for component in layout.components:
        if not _is_filter_block(component):
            continue
        input_pins = [pin for key, pin in layout.pin_map.items() if key[0] == component.id and _pin_kind(key[1]) in {"in", "input", "a"}]
        output_pins = [pin for key, pin in layout.pin_map.items() if key[0] == component.id and _pin_kind(key[1]) in {"out", "output", "b"}]
        if input_pins and any(pin.x > component.x for pin in input_pins):
            violations.append(CriticViolation("filter_input_not_left", f"{component.id} filter input is not on the left.", 12, subject=component.id))
        if output_pins and any(pin.x < component.x for pin in output_pins):
            violations.append(CriticViolation("filter_output_not_right", f"{component.id} filter output is not on the right.", 12, subject=component.id))


def _wire_crossings(layout: LayoutPlan) -> int:
    segments = _wire_segments(layout)
    count = 0
    for left, right in combinations(segments, 2):
        if left[0] == right[0]:
            continue
        if _segments_intersect(left[1], left[2], right[1], right[2]):
            count += 1
    return count


def _wire_net_overlaps(layout: LayoutPlan) -> int:
    segments = _wire_segments(layout)
    count = 0
    for left, right in combinations(segments, 2):
        if left[0] == right[0]:
            continue
        if _collinear_overlap_length(left[1], left[2], right[1], right[2]) > 0.08:
            count += 1
    return count


def _wire_pin_disconnects(layout: LayoutPlan) -> int:
    disconnected = 0
    for wire in layout.wires:
        segments = merged_axis_aligned_segments(wire.points)
        for key in wire.connected_pins:
            pin = layout.pin_map.get(key)
            if pin is None:
                continue
            point = Point(pin.x, pin.y)
            if not any(_point_on_segment(point, start, end, tolerance=0.015) for start, end in segments):
                disconnected += 1
    return disconnected


def _point_on_segment(point: Point, start: Point, end: Point, *, tolerance: float) -> bool:
    if _same(start.x, end.x):
        return abs(point.x - start.x) <= tolerance and _between(point.y, start.y, end.y, tolerance)
    if _same(start.y, end.y):
        return abs(point.y - start.y) <= tolerance and _between(point.x, start.x, end.x, tolerance)
    dx = end.x - start.x
    dy = end.y - start.y
    length = hypot(dx, dy)
    if length <= 1e-9:
        return hypot(point.x - start.x, point.y - start.y) <= tolerance
    distance = abs(dy * point.x - dx * point.y + end.x * start.y - end.y * start.x) / length
    projection = ((point.x - start.x) * dx + (point.y - start.y) * dy) / (length * length)
    return distance <= tolerance and -tolerance <= projection <= 1.0 + tolerance


def _between(value: float, first: float, second: float, tolerance: float) -> bool:
    return min(first, second) - tolerance <= value <= max(first, second) + tolerance


def _wire_segments(layout: LayoutPlan) -> list[tuple[str, Point, Point]]:
    return layout_wire_segments(layout)


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


def _collinear_overlap_length(a: Point, b: Point, c: Point, d: Point) -> float:
    if _same(a.y, b.y) and _same(c.y, d.y) and _same(a.y, c.y):
        return _interval_overlap(a.x, b.x, c.x, d.x)
    if _same(a.x, b.x) and _same(c.x, d.x) and _same(a.x, c.x):
        return _interval_overlap(a.y, b.y, c.y, d.y)
    return 0.0


def _interval_overlap(a1: float, a2: float, b1: float, b2: float) -> float:
    left = max(min(a1, a2), min(b1, b2))
    right = min(max(a1, a2), max(b1, b2))
    return max(0.0, right - left)


def _same(left: float, right: float) -> bool:
    return abs(left - right) < 1e-6


def _total_wire_length(layout: LayoutPlan) -> float:
    seen: set[tuple[str, tuple[float, float], tuple[float, float]]] = set()
    total = 0.0
    for net, a, b in _wire_segments(layout):
        left = (round(a.x, 4), round(a.y, 4))
        right = (round(b.x, 4), round(b.y, 4))
        start, end = (left, right) if left <= right else (right, left)
        key = (net, start, end)
        if key in seen:
            continue
        seen.add(key)
        total += hypot(b.x - a.x, b.y - a.y)
    return total


def _wire_length_limit(layout: LayoutPlan) -> float:
    if _opamp_network_layout(layout):
        per_component = 11.2
    else:
        per_component = 6.2 if _complex_opamp_layout(layout) else 4.8
    return max(18.0, len(layout.components) * per_component)


def _average_component_distance(layout: LayoutPlan) -> float:
    pairs = list(combinations(layout.components, 2))
    if not pairs:
        return 0.0
    return sum(hypot(left.x - right.x, left.y - right.y) for left, right in pairs) / len(pairs)


def _average_distance_limit(layout: LayoutPlan) -> float:
    if _opamp_network_layout(layout):
        if layout.width > 1400 and layout.height <= 720:
            return 14.0
        return 11.2
    return 8.2 if _complex_opamp_layout(layout) else 7.5


def _opamp_network_layout(layout: LayoutPlan) -> bool:
    return any(warning.strip() == "motif: op_amp_network" for warning in layout.warnings)


def _large_opamp_network(layout: LayoutPlan) -> bool:
    return len(layout.components) >= 20 and sum(1 for component in layout.components if "op" in _key(component.type)) >= 6


def _complex_opamp_layout(layout: LayoutPlan) -> bool:
    return len(layout.components) >= 12 and sum(1 for component in layout.components if "op" in _key(component.type)) >= 3


def _is_terminal_net_class(net_class: NetClass) -> bool:
    return net_class in {NetClass.GROUND, NetClass.POSITIVE_SUPPLY, NetClass.NEGATIVE_SUPPLY, NetClass.REFERENCE}


def _component_by_id(layout: LayoutPlan, component_id: str) -> LayoutComponent | None:
    return next((component for component in layout.components if component.id == component_id), None)


def _is_explicit_terminal_component(component: LayoutComponent) -> bool:
    key = _key(component.type)
    return key in {"input", "output", "input_terminal", "voltage_source", "source", "ground", "gnd", "supply", "power"} or "source" in key


def _is_opamp(component: LayoutComponent) -> bool:
    key = _key(component.type)
    return "op_amp" in key or "opamp" in key or "operational_amplifier" in key


def _is_flipped_opamp(component: LayoutComponent) -> bool:
    return _is_opamp(component) and "flip" in _key(component.orientation)


def _is_input_component(component: LayoutComponent) -> bool:
    key = _key(component.type)
    role = _key(component.role)
    return key in {"input", "input_terminal", "voltage_source", "source"} or "input" in role or "source" in key


def _is_filter_block(component: LayoutComponent) -> bool:
    key = _key(component.type)
    label = _key(component.label)
    value = _key(component.value)
    return any(_is_filter_text(text) for text in [key, label, value])


def _is_filter_text(value: str) -> bool:
    return any(token in value for token in ["filter", "lpf", "hpf", "bpf", "bessel", "butterworth", "chebyshev"])


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


def _feedback_owner_y(layout: LayoutPlan, feedback) -> float:
    owner = _feedback_owner(layout, feedback)
    return owner.y if owner is not None else _opamp_y(layout)


def _feedback_owner(layout: LayoutPlan, feedback) -> LayoutComponent | None:
    feedback_nets = set(feedback.pins.values())
    for component in layout.components:
        if "op" not in _key(component.type):
            continue
        output_net = None
        input_nets: set[str] = set()
        for pin_name, net in component.pins.items():
            if _is_opamp_output_pin(pin_name):
                output_net = net
            else:
                input_nets.add(net)
        if output_net in feedback_nets and feedback_nets & input_nets:
            return component
    return None


def _opamp_output_net(opamp: LayoutComponent) -> str | None:
    for pin_name, net in opamp.pins.items():
        if _is_opamp_output_pin(pin_name):
            return net
    return None


def _opamp_input_nets(opamp: LayoutComponent) -> set[str]:
    result: set[str] = set()
    for pin_name, net in opamp.pins.items():
        if _is_opamp_output_pin(pin_name) or is_local_terminal_net(net):
            continue
        kind = _pin_kind(pin_name)
        if kind in {"vcc", "vdd", "vee", "vss", "v+", "v-", "vp", "vn"}:
            continue
        result.add(net)
    return result


def _polyline_intersects_bbox_interior(points: list[Point], bbox: BBox) -> bool:
    if bbox.width <= 0 or bbox.height <= 0:
        return False
    for start, end in zip(points, points[1:]):
        if bbox.contains_point(_midpoint(start, end)):
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


def _midpoint(start: Point, end: Point) -> Point:
    return Point((start.x + end.x) / 2.0, (start.y + end.y) / 2.0)


def _is_feedback_route(layout: LayoutPlan, net: str, source: LayoutComponent, target: LayoutComponent) -> bool:
    if source.id == target.id:
        return True
    return any(net in motif.feedback_nets for motif in layout.semantic.motifs)


def _opamp_body_bbox(opamp: LayoutComponent) -> BBox:
    return BBox(
        opamp.bbox.x + 0.82,
        opamp.bbox.y + 0.12,
        max(0.1, opamp.bbox.width - 1.02),
        max(0.1, opamp.bbox.height - 0.24),
    )


def _is_opamp_output_pin(pin_name: str) -> bool:
    return _pin_kind(pin_name) in {"out", "output", "o", "vout"}


def _pin_kind(pin_name: str) -> str:
    compact = _key(pin_name).replace("_", "")
    if pin_name in {"+", "-"}:
        return pin_name
    if compact in {"plus", "noninverting", "noninv", "inp", "vp"}:
        return "+"
    if compact in {"minus", "inverting", "inv", "inn", "vn"}:
        return "-"
    return compact


def _key(value: str | None) -> str:
    return (value or "").lower().replace("-", "_").replace(" ", "_")
