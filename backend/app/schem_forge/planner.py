"""Canonical schematic planners for known analog circuit motifs."""

from __future__ import annotations

from statistics import median
from typing import Any

from .models import (
    ComponentLayout,
    LabelLayout,
    LayoutPlan,
    PinLayout,
    Point,
    WireRoute,
)
from .verifier import normalize_circuit_ir, topology_signature_from_circuit


DEFAULT_CANVAS_WIDTH = 960
DEFAULT_CANVAS_HEIGHT = 660
DEFAULT_GRID_SIZE = 28


def _key(value: str | None) -> str:
    return (value or "").lower().replace("-", "_").replace(" ", "_")


def _compact(value: str | None) -> str:
    return _key(value).replace("_", "")


def _is_type(component: dict[str, Any], *needles: str) -> bool:
    haystack = _key(component["type"])
    return any(needle in haystack for needle in needles)


def _has_role(component: dict[str, Any], *needles: str) -> bool:
    haystack = _key(component.get("role"))
    return any(needle in haystack for needle in needles)


def _is_opamp(component: dict[str, Any]) -> bool:
    return _is_type(component, "op_amp", "opamp", "operational_amplifier")


def _is_resistor(component: dict[str, Any]) -> bool:
    return _is_type(component, "resistor") or component["type"].lower().startswith("r")


def _is_capacitor(component: dict[str, Any]) -> bool:
    return _is_type(component, "capacitor") or component["type"].lower().startswith("c")


def _is_ground(component: dict[str, Any]) -> bool:
    role = _key(component.get("role"))
    return _is_type(component, "ground", "gnd") or role in {"ground", "gnd", "ground_symbol"}


def _is_output(component: dict[str, Any]) -> bool:
    return _is_type(component, "output", "terminal") and _has_role(component, "output")


def _is_input(component: dict[str, Any]) -> bool:
    role = _key(component.get("role"))
    component_type = _key(component["type"])
    return role in {"input", "input_source", "input_terminal", "sensor"} or component_type in {
        "input",
        "input_terminal",
    }


def _is_source(component: dict[str, Any]) -> bool:
    return _is_type(component, "source", "voltage_source", "current_source")


def _pin_kind(component: dict[str, Any], pin_name: str) -> str:
    compact = _compact(pin_name)
    if _is_opamp(component):
        if pin_name == "+" or compact in {
            "plus",
            "inplus",
            "noninv",
            "noninverting",
            "noninvertinginput",
            "inp",
            "vp",
            "positive",
        }:
            return "non_inverting"
        if pin_name == "-" or compact in {
            "minus",
            "inminus",
            "inv",
            "inverting",
            "invertinginput",
            "inn",
            "vn",
            "negative",
        }:
            return "inverting"
        if compact in {"out", "output", "vo", "vout"}:
            return "output"
        if compact in {"vcc", "vplus", "vdd", "supplyplus"}:
            return "positive_supply"
        if compact in {"vee", "vminus", "vss", "supplyminus"}:
            return "negative_supply"
    return compact


def _pin_name_by_kind(component: dict[str, Any], kind: str) -> str | None:
    for pin_name in component["pins"]:
        if _pin_kind(component, pin_name) == kind:
            return pin_name
    return None


def _pin_ref(component: dict[str, Any], pin_name: str) -> str:
    return f"{component['id']}.{pin_name}"


def _pin_ref_by_kind(component: dict[str, Any] | None, kind: str) -> str | None:
    if component is None:
        return None
    pin_name = _pin_name_by_kind(component, kind)
    return _pin_ref(component, pin_name) if pin_name else None


def _pin_ref_for_net(component: dict[str, Any] | None, net_name: str | None) -> str | None:
    if component is None or net_name is None:
        return None
    for pin_name, candidate_net in component["pins"].items():
        if candidate_net == net_name:
            return _pin_ref(component, pin_name)
    return None


def _two_pin_names(component: dict[str, Any]) -> list[str]:
    return sorted(component["pins"])


def pin_layouts_for_component(
    component: dict[str, Any],
    orientation: str,
) -> list[PinLayout]:
    """Create visual pin anchors without changing electrical pin names or nets."""

    pin_names = _two_pin_names(component)
    layouts: list[PinLayout] = []
    for index, pin_name in enumerate(pin_names):
        side = "right"
        offset_x = 0.6
        offset_y = 0.0

        if _is_opamp(component):
            kind = _pin_kind(component, pin_name)
            if orientation == "left":
                table = {
                    "inverting": ("right", 2.0, -1.0),
                    "non_inverting": ("right", 2.0, 1.0),
                    "output": ("left", -2.0, 0.0),
                    "positive_supply": ("top", 0.0, -1.5),
                    "negative_supply": ("bottom", 0.0, 1.5),
                }
            elif orientation == "up":
                table = {
                    "inverting": ("bottom", -1.0, 2.0),
                    "non_inverting": ("bottom", 1.0, 2.0),
                    "output": ("top", 0.0, -2.0),
                    "positive_supply": ("left", -1.5, 0.0),
                    "negative_supply": ("right", 1.5, 0.0),
                }
            elif orientation == "down":
                table = {
                    "inverting": ("top", -1.0, -2.0),
                    "non_inverting": ("top", 1.0, -2.0),
                    "output": ("bottom", 0.0, 2.0),
                    "positive_supply": ("right", 1.5, 0.0),
                    "negative_supply": ("left", -1.5, 0.0),
                }
            else:
                table = {
                    "inverting": ("left", -2.0, -1.0),
                    "non_inverting": ("left", -2.0, 1.0),
                    "output": ("right", 2.0, 0.0),
                    "positive_supply": ("top", 0.0, -1.5),
                    "negative_supply": ("bottom", 0.0, 1.5),
                }
            side, offset_x, offset_y = table.get(kind, ("left", -2.0, 0.0))
        elif _is_ground(component):
            side, offset_x, offset_y = "top", 0.0, -0.8
        elif _is_output(component):
            side, offset_x, offset_y = "left", -0.6, 0.0
        elif _is_input(component) and len(pin_names) == 1:
            side, offset_x, offset_y = "right", 0.6, 0.0
        elif _is_resistor(component) or _is_capacitor(component):
            first_pin = index == 0
            if orientation in {"up", "down"}:
                side = "top" if first_pin else "bottom"
                offset_x = 0.0
                offset_y = -1.0 if first_pin else 1.0
            else:
                side = "left" if first_pin else "right"
                offset_x = -1.0 if first_pin else 1.0
                offset_y = 0.0
        elif _is_source(component) and len(pin_names) >= 2:
            first_pin = index == 0
            if orientation in {"left", "right"}:
                if orientation == "right":
                    side = "right" if first_pin else "left"
                    offset_x = 1.0 if first_pin else -1.0
                else:
                    side = "left" if first_pin else "right"
                    offset_x = -1.0 if first_pin else 1.0
                offset_y = 0.0
            else:
                side = "top" if first_pin else "bottom"
                offset_x = 0.0
                offset_y = -1.0 if first_pin else 1.0

        layouts.append(
            PinLayout(
                component_id=component["id"],
                pin_name=pin_name,
                net_name=component["pins"][pin_name],
                side=side,  # type: ignore[arg-type]
                offset_x=offset_x,
                offset_y=offset_y,
            )
        )
    return layouts


def _make_component_layout(
    component: dict[str, Any],
    grid_x: float,
    grid_y: float,
    orientation: str,
) -> ComponentLayout:
    return ComponentLayout(
        id=component["id"],
        type=component["type"],
        role=component.get("role"),
        grid_x=grid_x,
        grid_y=grid_y,
        orientation=orientation,  # type: ignore[arg-type]
        value_label=component.get("value_label"),
        display_label=component.get("display_label") or component["id"],
        pins=pin_layouts_for_component(component, orientation),
        bbox=None,
    )


def _pin_points_grid(components: list[ComponentLayout]) -> dict[str, Point]:
    return {
        f"{component.id}.{pin.pin_name}": Point(
            component.grid_x + pin.offset_x,
            component.grid_y + pin.offset_y,
        )
        for component in components
        for pin in component.pins
    }


def _dedupe_consecutive(points: list[Point]) -> list[Point]:
    deduped: list[Point] = []
    for point in points:
        if not deduped or deduped[-1].x != point.x or deduped[-1].y != point.y:
            deduped.append(point)
    return deduped


def _route_from_sequence(
    net_name: str,
    connected_pins: list[str],
    pin_points: dict[str, Point],
    sequence: list[str | Point],
) -> WireRoute:
    points: list[Point] = []
    for item in sequence:
        if isinstance(item, Point):
            points.append(item)
        elif item in pin_points:
            points.append(pin_points[item])
    return WireRoute(
        net_name=net_name,
        connected_pins=sorted(connected_pins),
        waypoints=_dedupe_consecutive(points),
        segments=None,
    )


def _orthogonal_route_between(start: Point, end: Point) -> list[Point]:
    if start.x == end.x or start.y == end.y:
        return [start, end]
    mid_x = (start.x + end.x) / 2.0
    return [start, Point(mid_x, start.y), Point(mid_x, end.y), end]


def _build_generic_routes(
    net_to_pins: dict[str, list[str]],
    pin_points: dict[str, Point],
) -> list[WireRoute]:
    routes: list[WireRoute] = []
    for net_name, pin_refs in sorted(net_to_pins.items()):
        available = [pin_ref for pin_ref in pin_refs if pin_ref in pin_points]
        if len(available) == 2:
            waypoints = _orthogonal_route_between(pin_points[available[0]], pin_points[available[1]])
        elif len(available) > 2:
            xs = [pin_points[pin_ref].x for pin_ref in available]
            ys = [pin_points[pin_ref].y for pin_ref in available]
            hub = Point(float(median(xs)), float(median(ys)))
            waypoints = []
            for index, pin_ref in enumerate(available):
                if index:
                    waypoints.append(hub)
                waypoints.extend(_orthogonal_route_between(hub, pin_points[pin_ref])[::-1])
        else:
            waypoints = [pin_points[available[0]]] if available else []
        routes.append(
            WireRoute(
                net_name=net_name,
                connected_pins=sorted(pin_refs),
                waypoints=_dedupe_consecutive(waypoints),
                segments=None,
            )
        )
    return routes


def _add_default_labels(
    normalized: dict[str, Any],
    placements: dict[str, tuple[float, float, str]],
    special_positions: dict[str, tuple[float, float]] | None = None,
) -> list[LabelLayout]:
    special_positions = special_positions or {}
    labels: list[LabelLayout] = []
    for component in normalized["components"]:
        text = component.get("value_label")
        if not text or component["id"] not in placements:
            continue
        grid_x, grid_y, _orientation = placements[component["id"]]
        label_x, label_y = special_positions.get(component["id"], (grid_x, grid_y - 0.85))
        labels.append(
            LabelLayout(
                id=f"label_{component['id']}",
                text=str(text),
                owner_id=component["id"],
                grid_x=label_x,
                grid_y=label_y,
                anchor="middle",
                bbox=None,
            )
        )
    return labels


def _build_plan(
    circuit_ir: Any,
    placements: dict[str, tuple[float, float, str]],
    labels: list[LabelLayout],
    wires: list[WireRoute] | None,
    warnings: list[str],
    canvas_width: int = DEFAULT_CANVAS_WIDTH,
    canvas_height: int = DEFAULT_CANVAS_HEIGHT,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> LayoutPlan:
    normalized = normalize_circuit_ir(circuit_ir)
    fallback_index = 0
    for component in normalized["components"]:
        if component["id"] not in placements:
            column = fallback_index % 4
            row = fallback_index // 4
            placements[component["id"]] = (4.0 + column * 4.0, 4.0 + row * 4.0, "right")
            fallback_index += 1

    components = [
        _make_component_layout(
            component,
            placements[component["id"]][0],
            placements[component["id"]][1],
            placements[component["id"]][2],
        )
        for component in normalized["components"]
    ]
    pin_points = _pin_points_grid(components)
    if wires is None:
        wires = _build_generic_routes(normalized["net_to_pins"], pin_points)

    return LayoutPlan(
        circuit_id=normalized["circuit_id"],
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        grid_size=grid_size,
        components=components,
        wires=wires,
        labels=labels,
        net_to_pins=normalized["net_to_pins"],
        component_pin_nets=normalized["component_pin_nets"],
        topology_signature=topology_signature_from_circuit(circuit_ir),
        renderer="schem_forge.svg.v1",
        warnings=warnings,
    )


def _component_by_id(normalized: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {component["id"]: component for component in normalized["components"]}


def _first_component(
    normalized: dict[str, Any],
    predicate,
) -> dict[str, Any] | None:
    for component in normalized["components"]:
        if predicate(component):
            return component
    return None


def _components_matching(normalized: dict[str, Any], predicate) -> list[dict[str, Any]]:
    return [component for component in normalized["components"] if predicate(component)]


def _resistor_between(
    normalized: dict[str, Any],
    net_a: str | None,
    net_b: str | None,
    role_hint: str | None = None,
) -> dict[str, Any] | None:
    if not net_a or not net_b:
        return None
    for component in normalized["components"]:
        if not _is_resistor(component):
            continue
        if role_hint and role_hint not in _key(component.get("role")):
            continue
        nets = set(component["pins"].values())
        if net_a in nets and net_b in nets:
            return component
    for component in normalized["components"]:
        if not _is_resistor(component):
            continue
        nets = set(component["pins"].values())
        if net_a in nets and net_b in nets:
            return component
    return None


def _ground_net(normalized: dict[str, Any]) -> str | None:
    ground = _first_component(normalized, _is_ground)
    if ground:
        return next(iter(ground["pins"].values()))
    for net_name in normalized["net_to_pins"]:
        if _compact(net_name) in {"gnd", "ground", "0"}:
            return net_name
    return None


def _source_for_net(normalized: dict[str, Any], net_name: str | None) -> dict[str, Any] | None:
    if not net_name:
        return None
    return _first_component(
        normalized,
        lambda item: _is_source(item) and net_name in set(item["pins"].values()),
    )


def _finalize_with_generic_routes(
    circuit_ir: Any,
    placements: dict[str, tuple[float, float, str]],
    labels: list[LabelLayout],
    warnings: list[str],
) -> LayoutPlan:
    return _build_plan(circuit_ir, placements, labels, wires=None, warnings=warnings)


def plan_non_inverting_op_amp(circuit_ir: Any) -> LayoutPlan:
    normalized = normalize_circuit_ir(circuit_ir)
    opamp = _first_component(normalized, _is_opamp)
    input_component = _first_component(normalized, _is_input)
    output_component = _first_component(normalized, _is_output)
    ground_component = _first_component(normalized, _is_ground)

    placements: dict[str, tuple[float, float, str]] = {}
    if input_component:
        placements[input_component["id"]] = (3.0, 10.0, "right")
    if opamp:
        placements[opamp["id"]] = (10.0, 9.0, "right")
    if output_component:
        placements[output_component["id"]] = (17.0, 9.0, "right")
    if ground_component:
        placements[ground_component["id"]] = (7.0, 15.0, "down")

    if opamp:
        out_net = opamp["pins"].get(_pin_name_by_kind(opamp, "output") or "")
        inv_net = opamp["pins"].get(_pin_name_by_kind(opamp, "inverting") or "")
        input_net = opamp["pins"].get(_pin_name_by_kind(opamp, "non_inverting") or "")
        ground_net = _ground_net(normalized)
        input_source = _source_for_net(normalized, input_net)
        feedback = _resistor_between(normalized, out_net, inv_net, "feedback")
        gain = _resistor_between(normalized, inv_net, ground_net, "gain")
        if input_source and input_source["id"] not in placements:
            placements[input_source["id"]] = (1.2 if input_component else 3.0, 10.0, "right")
        if feedback:
            placements[feedback["id"]] = (10.0, 5.0, "right")
        if gain:
            placements[gain["id"]] = (7.0, 12.0, "down")

    special_labels: dict[str, tuple[float, float]] = {}
    if opamp:
        out_net = opamp["pins"].get(_pin_name_by_kind(opamp, "output") or "")
        inv_net = opamp["pins"].get(_pin_name_by_kind(opamp, "inverting") or "")
        ground_net = _ground_net(normalized)
        feedback = _resistor_between(normalized, out_net, inv_net, "feedback")
        gain = _resistor_between(normalized, inv_net, ground_net, "gain")
        if feedback:
            special_labels[feedback["id"]] = (10.0, 3.8)
        if gain:
            special_labels[gain["id"]] = (5.7, 12.0)

    labels = _add_default_labels(normalized, placements, special_labels)
    provisional = _build_plan(
        circuit_ir,
        dict(placements),
        labels,
        wires=[],
        warnings=["motif: non_inverting_op_amp"],
    )
    pin_points = _pin_points_grid(provisional.components)
    routes_by_net: dict[str, WireRoute] = {}

    def add_route(net_name: str | None, sequence: list[str | Point]) -> None:
        if not net_name or net_name not in normalized["net_to_pins"]:
            return
        routes_by_net[net_name] = _route_from_sequence(
            net_name,
            normalized["net_to_pins"][net_name],
            pin_points,
            sequence,
        )

    if opamp and (input_component or _source_for_net(normalized, opamp["pins"].get(_pin_name_by_kind(opamp, "non_inverting") or ""))):
        plus_ref = _pin_ref_by_kind(opamp, "non_inverting") or ""
        plus_point = pin_points.get(plus_ref)
        input_net = opamp["pins"].get(_pin_name_by_kind(opamp, "non_inverting") or "")
        input_source = _source_for_net(normalized, input_net)
        input_sequence: list[str | Point] = []
        if input_source:
            input_sequence.append(_pin_ref_for_net(input_source, input_net) or "")
        if input_component:
            input_sequence.append(_pin_ref_for_net(input_component, input_net) or "")
        input_sequence.extend(
            [
                Point(3.6, 16.5),
                Point(plus_point.x if plus_point else 8.0, 16.5),
                plus_ref,
            ]
        )
        add_route(
            input_net,
            input_sequence,
        )

    if opamp:
        out_net = opamp["pins"].get(_pin_name_by_kind(opamp, "output") or "")
        inv_net = opamp["pins"].get(_pin_name_by_kind(opamp, "inverting") or "")
        ground_net = _ground_net(normalized)
        feedback = _resistor_between(normalized, out_net, inv_net, "feedback")
        gain = _resistor_between(normalized, inv_net, ground_net, "gain")
        if feedback and gain:
            add_route(
                inv_net,
                [
                    _pin_ref_for_net(feedback, inv_net) or "",
                    Point(8.0, 5.0),
                    _pin_ref_by_kind(opamp, "inverting") or "",
                    Point(7.0, 8.0),
                    _pin_ref_for_net(gain, inv_net) or "",
                ],
            )
        if feedback and output_component:
            add_route(
                out_net,
                [
                    _pin_ref_for_net(feedback, out_net) or "",
                    Point(12.0, 5.0),
                    _pin_ref_by_kind(opamp, "output") or "",
                    _pin_ref_for_net(output_component, out_net) or "",
                ],
            )
        if gain and ground_component:
            add_route(
                ground_net,
                [
                    _pin_ref_for_net(gain, ground_net) or "",
                    _pin_ref_for_net(ground_component, ground_net) or "",
                ],
            )

    for route in _build_generic_routes(normalized["net_to_pins"], pin_points):
        routes_by_net.setdefault(route.net_name, route)

    return _build_plan(
        circuit_ir,
        placements,
        labels,
        wires=[routes_by_net[net] for net in sorted(routes_by_net)],
        warnings=["motif: non_inverting_op_amp"],
    )


def plan_instrumentation_amplifier(circuit_ir: Any) -> LayoutPlan:
    normalized = normalize_circuit_ir(circuit_ir)
    opamps = _components_matching(normalized, _is_opamp)
    buffers = _components_matching(
        normalized,
        lambda item: _is_opamp(item) and _has_role(item, "input_buffer", "buffer"),
    )
    diff_stage = _first_component(
        normalized,
        lambda item: _is_opamp(item) and _has_role(item, "differential_stage", "diff_stage"),
    )
    if len(buffers) < 2:
        buffers = [item for item in opamps if item != diff_stage][:2]
    if diff_stage is None and len(opamps) >= 3:
        diff_stage = opamps[2]

    top_buffer = buffers[0] if buffers else None
    bottom_buffer = buffers[1] if len(buffers) > 1 else None
    inputs = _components_matching(normalized, _is_input)
    output_component = _first_component(normalized, _is_output)
    ground_component = _first_component(normalized, _is_ground)
    top_plus_net = (
        top_buffer["pins"].get(_pin_name_by_kind(top_buffer, "non_inverting") or "")
        if top_buffer
        else None
    )
    bottom_plus_net = (
        bottom_buffer["pins"].get(_pin_name_by_kind(bottom_buffer, "non_inverting") or "")
        if bottom_buffer
        else None
    )

    def input_for_net(net_name: str | None) -> dict[str, Any] | None:
        if net_name is None:
            return None
        for input_component in inputs:
            if net_name in set(input_component["pins"].values()):
                return input_component
        return None

    top_input = input_for_net(top_plus_net) or (inputs[0] if inputs else None)
    bottom_input = input_for_net(bottom_plus_net) or (
        next((item for item in inputs if item != top_input), None)
    )
    top_source = _source_for_net(normalized, top_plus_net)
    bottom_source = _source_for_net(normalized, bottom_plus_net)

    placements: dict[str, tuple[float, float, str]] = {}
    if top_buffer:
        placements[top_buffer["id"]] = (9.0, 6.0, "right")
    if bottom_buffer:
        placements[bottom_buffer["id"]] = (9.0, 16.0, "right")
    if diff_stage:
        placements[diff_stage["id"]] = (23.0, 11.0, "right")
    if top_input:
        placements[top_input["id"]] = (3.0, 7.0, "right")
    if bottom_input:
        placements[bottom_input["id"]] = (3.0, 17.0, "right")
    if top_source and top_source["id"] not in placements:
        placements[top_source["id"]] = (1.2 if top_input else 3.0, 7.0, "right")
    if bottom_source and bottom_source["id"] not in placements:
        placements[bottom_source["id"]] = (1.2 if bottom_input else 3.0, 17.0, "right")
    if output_component:
        placements[output_component["id"]] = (30.0, 11.0, "right")
    if ground_component:
        placements[ground_component["id"]] = (20.0, 20.0, "down")

    top_inv_net = (
        top_buffer["pins"].get(_pin_name_by_kind(top_buffer, "inverting") or "")
        if top_buffer
        else None
    )
    top_out_net = (
        top_buffer["pins"].get(_pin_name_by_kind(top_buffer, "output") or "")
        if top_buffer
        else None
    )
    bottom_inv_net = (
        bottom_buffer["pins"].get(_pin_name_by_kind(bottom_buffer, "inverting") or "")
        if bottom_buffer
        else None
    )
    bottom_out_net = (
        bottom_buffer["pins"].get(_pin_name_by_kind(bottom_buffer, "output") or "")
        if bottom_buffer
        else None
    )
    diff_inv_net = (
        diff_stage["pins"].get(_pin_name_by_kind(diff_stage, "inverting") or "")
        if diff_stage
        else None
    )
    diff_non_inv_net = (
        diff_stage["pins"].get(_pin_name_by_kind(diff_stage, "non_inverting") or "")
        if diff_stage
        else None
    )
    diff_out_net = (
        diff_stage["pins"].get(_pin_name_by_kind(diff_stage, "output") or "")
        if diff_stage
        else None
    )

    rf_top = _resistor_between(normalized, top_inv_net, top_out_net, "feedback")
    rf_bottom = _resistor_between(normalized, bottom_inv_net, bottom_out_net, "feedback")
    rf_diff = _resistor_between(normalized, diff_inv_net, diff_out_net, "feedback")
    gain = _resistor_between(normalized, top_inv_net, bottom_inv_net, "gain")
    r_top = _resistor_between(normalized, top_out_net, diff_inv_net, "input")
    r_bottom = _resistor_between(normalized, bottom_out_net, diff_non_inv_net, "input")
    r_ref = _resistor_between(normalized, diff_non_inv_net, _ground_net(normalized), "ground")

    if rf_top:
        placements[rf_top["id"]] = (9.0, 3.0, "right")
    if rf_bottom:
        placements[rf_bottom["id"]] = (9.0, 13.0, "right")
    if rf_diff:
        placements[rf_diff["id"]] = (23.0, 6.0, "right")
    if gain:
        placements[gain["id"]] = (5.0, 10.0, "down")
    if r_top:
        placements[r_top["id"]] = (18.0, 8.0, "right")
    if r_bottom:
        placements[r_bottom["id"]] = (18.0, 14.0, "right")
    if r_ref:
        placements[r_ref["id"]] = (20.0, 16.0, "down")

    special_labels: dict[str, tuple[float, float]] = {}
    if rf_top:
        special_labels[rf_top["id"]] = (9.0, 1.9)
    if rf_bottom:
        special_labels[rf_bottom["id"]] = (9.0, 11.9)
    if rf_diff:
        special_labels[rf_diff["id"]] = (23.0, 4.8)
    if r_top:
        special_labels[r_top["id"]] = (18.0, 6.9)
    if r_bottom:
        special_labels[r_bottom["id"]] = (18.0, 15.2)
    if r_ref:
        special_labels[r_ref["id"]] = (22.1, 16.0)
    if gain:
        special_labels[gain["id"]] = (3.2, 10.0)
    labels = _add_default_labels(normalized, placements, special_labels)

    provisional = _build_plan(
        circuit_ir,
        dict(placements),
        labels,
        wires=[],
        warnings=["motif: instrumentation_amplifier"],
    )
    pin_points = _pin_points_grid(provisional.components)
    routes_by_net: dict[str, WireRoute] = {}

    def add_route(net_name: str | None, sequence: list[str | Point]) -> None:
        if not net_name or net_name not in normalized["net_to_pins"]:
            return
        route = _route_from_sequence(
            net_name,
            normalized["net_to_pins"][net_name],
            pin_points,
            sequence,
        )
        if len(route.waypoints) >= 1:
            routes_by_net[net_name] = route

    if top_buffer and (top_input or top_source):
        top_plus = _pin_ref_by_kind(top_buffer, "non_inverting") or ""
        top_plus_net = top_buffer["pins"].get(_pin_name_by_kind(top_buffer, "non_inverting") or "")
        top_sequence: list[str | Point] = []
        if top_source:
            top_sequence.append(_pin_ref_for_net(top_source, top_plus_net) or "")
        if top_input:
            top_sequence.append(_pin_ref_for_net(top_input, top_plus_net) or "")
        top_sequence.append(top_plus)
        add_route(
            top_plus_net,
            top_sequence,
        )
    if bottom_buffer and (bottom_input or bottom_source):
        bottom_plus_net = bottom_buffer["pins"].get(_pin_name_by_kind(bottom_buffer, "non_inverting") or "")
        bottom_sequence: list[str | Point] = []
        if bottom_source:
            bottom_sequence.append(_pin_ref_for_net(bottom_source, bottom_plus_net) or "")
        if bottom_input:
            bottom_sequence.append(_pin_ref_for_net(bottom_input, bottom_plus_net) or "")
        bottom_sequence.append(_pin_ref_by_kind(bottom_buffer, "non_inverting") or "")
        add_route(
            bottom_plus_net,
            bottom_sequence,
        )
    if top_buffer and rf_top and gain and top_inv_net:
        add_route(
            top_inv_net,
            [
                _pin_ref_for_net(rf_top, top_inv_net) or "",
                Point(7.0, 3.0),
                _pin_ref_by_kind(top_buffer, "inverting") or "",
                Point(2.0, 5.0),
                Point(2.0, 9.0),
                _pin_ref_for_net(gain, top_inv_net) or "",
            ],
        )
    if top_buffer and rf_top and r_top and top_out_net:
        add_route(
            top_out_net,
            [
                _pin_ref_for_net(rf_top, top_out_net) or "",
                Point(11.0, 3.0),
                _pin_ref_by_kind(top_buffer, "output") or "",
                Point(13.0, 6.0),
                Point(13.0, 8.0),
                _pin_ref_for_net(r_top, top_out_net) or "",
            ],
        )
    if bottom_buffer and rf_bottom and gain and bottom_inv_net:
        add_route(
            bottom_inv_net,
            [
                _pin_ref_for_net(gain, bottom_inv_net) or "",
                Point(5.0, 15.0),
                _pin_ref_by_kind(bottom_buffer, "inverting") or "",
                Point(7.0, 13.0),
                _pin_ref_for_net(rf_bottom, bottom_inv_net) or "",
            ],
        )
    if bottom_buffer and rf_bottom and r_bottom and bottom_out_net:
        add_route(
            bottom_out_net,
            [
                _pin_ref_for_net(rf_bottom, bottom_out_net) or "",
                Point(11.0, 13.0),
                _pin_ref_by_kind(bottom_buffer, "output") or "",
                Point(13.0, 16.0),
                Point(13.0, 14.0),
                _pin_ref_for_net(r_bottom, bottom_out_net) or "",
            ],
        )
    if diff_stage and r_top and rf_diff and diff_inv_net:
        add_route(
            diff_inv_net,
            [
                _pin_ref_for_net(r_top, diff_inv_net) or "",
                Point(21.0, 8.0),
                _pin_ref_by_kind(diff_stage, "inverting") or "",
                Point(21.0, 6.0),
                _pin_ref_for_net(rf_diff, diff_inv_net) or "",
            ],
        )
    if diff_stage and r_bottom and r_ref and diff_non_inv_net:
        add_route(
            diff_non_inv_net,
            [
                _pin_ref_for_net(r_bottom, diff_non_inv_net) or "",
                Point(21.0, 14.0),
                _pin_ref_by_kind(diff_stage, "non_inverting") or "",
                Point(20.0, 12.0),
                _pin_ref_for_net(r_ref, diff_non_inv_net) or "",
            ],
        )
    if diff_stage and rf_diff and output_component and diff_out_net:
        add_route(
            diff_out_net,
            [
                _pin_ref_for_net(rf_diff, diff_out_net) or "",
                Point(25.0, 6.0),
                _pin_ref_by_kind(diff_stage, "output") or "",
                _pin_ref_for_net(output_component, diff_out_net) or "",
            ],
        )
    if r_ref and ground_component:
        ground_net = _ground_net(normalized)
        add_route(
            ground_net,
            [
                _pin_ref_for_net(r_ref, ground_net) or "",
                _pin_ref_for_net(ground_component, ground_net) or "",
            ],
        )

    generic_routes = _build_generic_routes(normalized["net_to_pins"], pin_points)
    for route in generic_routes:
        routes_by_net.setdefault(route.net_name, route)

    return _build_plan(
        circuit_ir,
        placements,
        labels,
        wires=[routes_by_net[net] for net in sorted(routes_by_net)],
        warnings=["motif: instrumentation_amplifier"],
    )


def plan_rc_low_pass(circuit_ir: Any) -> LayoutPlan:
    normalized = normalize_circuit_ir(circuit_ir)
    source = _first_component(normalized, _is_input) or _first_component(normalized, _is_source)
    resistor = _first_component(normalized, _is_resistor)
    capacitor = _first_component(normalized, _is_capacitor)
    ground = _first_component(normalized, _is_ground)
    output = _first_component(normalized, _is_output)
    placements: dict[str, tuple[float, float, str]] = {}
    if source:
        placements[source["id"]] = (3.0, 9.0, "right")
    if source:
        source_net = next(iter(source["pins"].values()))
        input_source = _source_for_net(normalized, source_net)
        if input_source and input_source["id"] not in placements:
            placements[input_source["id"]] = (1.2, 9.0, "right")
    if resistor:
        placements[resistor["id"]] = (8.0, 9.0, "right")
    if capacitor:
        placements[capacitor["id"]] = (12.0, 12.0, "down")
    if ground:
        placements[ground["id"]] = (12.0, 16.0, "down")
    if output:
        placements[output["id"]] = (17.0, 9.0, "right")
    special_labels: dict[str, tuple[float, float]] = {}
    if resistor:
        special_labels[resistor["id"]] = (8.0, 7.8)
    if capacitor:
        special_labels[capacitor["id"]] = (13.4, 12.0)
    labels = _add_default_labels(normalized, placements, special_labels)
    return _finalize_with_generic_routes(
        circuit_ir,
        placements,
        labels,
        ["motif: rc_low_pass"],
    )


def plan_voltage_divider(circuit_ir: Any) -> LayoutPlan:
    normalized = normalize_circuit_ir(circuit_ir)
    resistors = _components_matching(normalized, _is_resistor)
    source = _first_component(normalized, _is_input) or _first_component(normalized, _is_source)
    ground = _first_component(normalized, _is_ground)
    output = _first_component(normalized, _is_output)
    placements: dict[str, tuple[float, float, str]] = {}
    if source:
        placements[source["id"]] = (3.0, 7.0, "right")
    if source:
        source_net = next(iter(source["pins"].values()))
        input_source = _source_for_net(normalized, source_net)
        if input_source and input_source["id"] not in placements:
            placements[input_source["id"]] = (1.2, 7.0, "right")
    if resistors:
        placements[resistors[0]["id"]] = (9.0, 7.0, "down")
    if len(resistors) > 1:
        placements[resistors[1]["id"]] = (9.0, 12.0, "down")
    if output:
        placements[output["id"]] = (15.0, 9.5, "right")
    if ground:
        placements[ground["id"]] = (9.0, 16.0, "down")
    special_labels: dict[str, tuple[float, float]] = {}
    if resistors:
        special_labels[resistors[0]["id"]] = (10.3, 7.0)
    if len(resistors) > 1:
        special_labels[resistors[1]["id"]] = (10.3, 12.0)
    labels = _add_default_labels(normalized, placements, special_labels)
    return _finalize_with_generic_routes(
        circuit_ir,
        placements,
        labels,
        ["motif: voltage_divider"],
    )


def plan_bridge_or_wheatstone(circuit_ir: Any) -> LayoutPlan:
    normalized = normalize_circuit_ir(circuit_ir)
    resistors = _components_matching(normalized, _is_resistor)
    inputs = _components_matching(normalized, lambda item: _is_input(item) or _is_source(item))
    outputs = _components_matching(normalized, _is_output)
    ground = _first_component(normalized, _is_ground)
    placements: dict[str, tuple[float, float, str]] = {}
    resistor_positions = [(8.0, 6.0), (8.0, 12.0), (14.0, 6.0), (14.0, 12.0)]
    for component, (grid_x, grid_y) in zip(resistors, resistor_positions):
        placements[component["id"]] = (grid_x, grid_y, "down")
    if inputs:
        placements[inputs[0]["id"]] = (3.0, 6.0, "right")
    if len(inputs) > 1:
        placements[inputs[1]["id"]] = (3.0, 14.0, "right")
    for index, output in enumerate(outputs[:2]):
        placements[output["id"]] = (19.0, 8.0 + index * 4.0, "right")
    if ground:
        placements[ground["id"]] = (11.0, 17.0, "down")
    labels = _add_default_labels(normalized, placements)
    return _finalize_with_generic_routes(
        circuit_ir,
        placements,
        labels,
        ["motif: bridge_or_wheatstone"],
    )


def plan_grid_fallback(circuit_ir: Any) -> LayoutPlan:
    normalized = normalize_circuit_ir(circuit_ir)
    placements: dict[str, tuple[float, float, str]] = {}
    columns = 5
    for index, component in enumerate(normalized["components"]):
        placements[component["id"]] = (
            4.0 + (index % columns) * 5.0,
            5.0 + (index // columns) * 5.0,
            "right",
        )
    labels = _add_default_labels(normalized, placements)
    return _finalize_with_generic_routes(
        circuit_ir,
        placements,
        labels,
        ["motif: grid_fallback"],
    )


def plan_circuit(circuit_ir: Any) -> LayoutPlan:
    normalized = normalize_circuit_ir(circuit_ir)
    motif = _key(normalized.get("motif"))
    opamp_count = len(_components_matching(normalized, _is_opamp))
    resistor_count = len(_components_matching(normalized, _is_resistor))
    capacitor_count = len(_components_matching(normalized, _is_capacitor))

    if "instrumentation" in motif or opamp_count >= 3:
        return plan_instrumentation_amplifier(circuit_ir)
    if "non_inverting" in motif or ("op_amp" in motif and opamp_count == 1):
        return plan_non_inverting_op_amp(circuit_ir)
    if "low_pass" in motif or (resistor_count >= 1 and capacitor_count >= 1):
        return plan_rc_low_pass(circuit_ir)
    if "divider" in motif or resistor_count == 2:
        return plan_voltage_divider(circuit_ir)
    if "bridge" in motif or "wheatstone" in motif or resistor_count >= 4:
        return plan_bridge_or_wheatstone(circuit_ir)
    return plan_grid_fallback(circuit_ir)
