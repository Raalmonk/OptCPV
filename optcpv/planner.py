"""Deterministic initial layout planners for OptCPV."""

from __future__ import annotations

from dataclasses import replace
from math import ceil
from statistics import median
from typing import Callable

from .labels import component_display_label, label_bbox_size
from .models import (
    BBox,
    Circuit,
    Component,
    LayoutComponent,
    LayoutLabel,
    LayoutPin,
    LayoutPlan,
    LayoutWire,
    Point,
    circuit_from_any,
)
from .verifier import topology_signature


GRID = 48
DEFAULT_WIDTH = 1100
DEFAULT_HEIGHT = 800


def plan_layout(circuit: Circuit | dict) -> LayoutPlan:
    native = circuit_from_any(circuit)
    motif = _validated_motif(native) or _infer_motif(native.components)
    planners: dict[str, Callable[[Circuit], LayoutPlan]] = {
        "voltage_divider": _plan_voltage_divider,
        "rc_low_pass": _plan_rc_low_pass,
        "non_inverting_op_amp": _plan_non_inverting_op_amp,
        "instrumentation_amplifier": _plan_instrumentation_amplifier,
        "bridge_or_wheatstone": _plan_bridge,
        "op_amp_network": _plan_op_amp_network,
    }
    planner = planners.get(motif)
    if planner:
        return planner(native)
    return _plan_diagnostic(native)


def rebuild_layout_geometry(layout: LayoutPlan) -> LayoutPlan:
    circuit = Circuit(
        id=layout.circuit_id,
        components=[
            Component(
                id=component.id,
                type=component.type,
                pins=dict(component.pins),
                label=component.label,
                role=component.role,
                value=component.value,
            )
            for component in layout.components
        ],
    )
    placements = {component.id: (component.x, component.y, component.orientation) for component in layout.components}
    label_offsets = {
        label.owner_id: (label.x - _component_by_id(layout, label.owner_id).x, label.y - _component_by_id(layout, label.owner_id).y)
        for label in layout.labels
        if _has_component(layout, label.owner_id)
    }
    return _build_layout(
        circuit,
        placements,
        list(layout.warnings),
        width=layout.width,
        height=layout.height,
        grid=layout.grid,
        label_offsets=label_offsets,
    )


def _plan_voltage_divider(circuit: Circuit) -> LayoutPlan:
    placements: dict[str, tuple[float, float, str]] = {}
    source = _first(circuit, _is_input_or_source)
    output = _first(circuit, _is_output)
    ground = _first(circuit, _is_ground)
    resistors = [component for component in circuit.components if _is_type(component, "resistor")]
    top, bottom = (resistors + [None, None])[:2]
    if source:
        placements[source.id] = (2.0, 4.05, "right")
    if top:
        placements[top.id] = (10.0, 5.0, "down")
    if bottom:
        placements[bottom.id] = (10.0, 8.2, "down")
    if output:
        placements[output.id] = (15.0, 6.6, "right")
    if ground:
        placements[ground.id] = (10.0, 10.9, "right")
    return _build_layout(circuit, placements, ["motif: voltage_divider"])


def _plan_rc_low_pass(circuit: Circuit) -> LayoutPlan:
    placements: dict[str, tuple[float, float, str]] = {}
    source = _first(circuit, _is_input_or_source)
    resistor = _first(circuit, lambda item: _is_type(item, "resistor"))
    capacitor = _first(circuit, lambda item: _is_type(item, "capacitor"))
    output = _first(circuit, _is_output)
    ground = _first(circuit, _is_ground)
    if source:
        placements[source.id] = (5.0, 5.0, "right")
    if resistor:
        placements[resistor.id] = (8.6, 5.0, "right")
    if capacitor:
        placements[capacitor.id] = (12.2, 7.2, "down")
    if output:
        placements[output.id] = (15.8, 5.0, "right")
    if ground:
        placements[ground.id] = (12.2, 10.3, "right")
    return _build_layout(circuit, placements, ["motif: rc_low_pass"])


def _plan_non_inverting_op_amp(circuit: Circuit) -> LayoutPlan:
    placements: dict[str, tuple[float, float, str]] = {}
    opamp = _first(circuit, _is_opamp)
    source = _first(circuit, _is_input_or_source)
    output = _first(circuit, _is_output)
    ground = _first(circuit, _is_ground)
    resistors = [component for component in circuit.components if _is_type(component, "resistor")]
    feedback = _first(circuit, lambda item: _has_role(item, "feedback")) or (resistors[0] if resistors else None)
    gain = _first(circuit, lambda item: _has_role(item, "gain")) or (
        next((item for item in resistors if item != feedback), None)
    )
    if source:
        placements[source.id] = (4.5, 7.125, "right")
    if opamp:
        placements[opamp.id] = (10.0, 6.5, "right")
    if feedback:
        placements[feedback.id] = (10.2, 3.3, "left")
    if gain:
        placements[gain.id] = (11.05, 8.85, "down")
    if output:
        placements[output.id] = (15.2, 6.5, "right")
    if ground:
        placements[ground.id] = (11.05, 11.1, "right")
    return _build_layout(circuit, placements, ["motif: non_inverting_op_amp"])


def _plan_instrumentation_amplifier(circuit: Circuit) -> LayoutPlan:
    placements: dict[str, tuple[float, float, str]] = {}
    opamps = [component for component in circuit.components if _is_opamp(component)]
    inputs = [component for component in circuit.components if _is_input_or_source(component)]
    output = _first(circuit, _is_output)
    ground = _first(circuit, _is_ground)
    resistors = [component for component in circuit.components if _is_type(component, "resistor")]

    # The initial layout is intentionally conservative and a little wide; the
    # optimizer is expected to compact it without changing topology.
    slots = {
        "input_top": (2.7, 4.625, "right"),
        "input_bottom": (2.7, 11.125, "right"),
        "u1": (6.2, 4.0, "right"),
        "u2": (6.2, 10.5, "right"),
        "u3": (14.0, 7.25, "right"),
        "output": (18.8, 7.25, "right"),
        "ground": (12.95, 12.0, "right"),
    }
    for component, key in zip(inputs[:2], ["input_top", "input_bottom"]):
        placements[component.id] = slots[key]
    for component, key in zip(opamps[:3], ["u1", "u2", "u3"]):
        placements[component.id] = slots[key]
    if output:
        placements[output.id] = slots["output"]
    if ground:
        placements[ground.id] = slots["ground"]

    resistor_slots = [
        (7.2, 1.8, "left"),
        (7.2, 12.7, "left"),
        (2.0, 6.625, "down"),
        (12.0, 7.875, "right"),
        (12.95, 9.4, "down"),
        (12.0, 6.625, "right"),
        (15.1, 4.8, "right"),
    ]
    for component, slot in zip(resistors, resistor_slots):
        placements[component.id] = slot
    return _build_layout(circuit, placements, ["motif: instrumentation_amplifier"])


def _plan_bridge(circuit: Circuit) -> LayoutPlan:
    placements: dict[str, tuple[float, float, str]] = {}
    resistors = [component for component in circuit.components if _is_type(component, "resistor")]
    for component, slot in zip(resistors, [(8, 4.3, "down"), (8, 8.0, "down"), (14, 4.3, "down"), (14, 8.0, "down")]):
        placements[component.id] = slot
    for component, slot in zip(
        [item for item in circuit.components if item.id not in placements],
        [(5, 3.35, "right"), (12.4, 6.15, "right"), (11, 10.8, "right")],
    ):
        placements[component.id] = slot
    return _build_layout(circuit, placements, ["motif: bridge_or_wheatstone"])


def _plan_op_amp_network(circuit: Circuit) -> LayoutPlan:
    placements: dict[str, tuple[float, float, str]] = {}
    opamps = _ordered_opamps(circuit, [component for component in circuit.components if _is_opamp(component)])
    width = 1400 if len(opamps) >= 3 else DEFAULT_WIDTH
    height = 850 if len(opamps) >= 3 else DEFAULT_HEIGHT
    span_x = width / GRID
    span_y = height / GRID
    columns = min(4, max(1, len(opamps)))
    rows = max(1, ceil(len(opamps) / columns))
    x0 = 3.2 if width > DEFAULT_WIDTH else 3.0
    dx = 6.0 if width > DEFAULT_WIDTH else 5.3
    long_label_mode = any(_long_label(component) for component in circuit.components)
    y0 = 4.15 if long_label_mode else 3.3
    dy = min(6.4, max(4.8, (span_y - 5.4) / max(1, rows - 1))) if rows > 1 else 0.0

    for index, opamp in enumerate(opamps):
        row, col = divmod(index, columns)
        placements[opamp.id] = (x0 + col * dx, y0 + row * dy, "right")

    output_net_to_opamp = _output_net_to_opamp(opamps)
    input_net_to_opamps = _input_net_to_opamps(opamps)
    fallback_inputs = 0
    fallback_outputs = 0
    output_counts: dict[str, int] = {}

    for terminal in [component for component in circuit.components if _is_input_or_source(component)]:
        target = _first_net_opamp(terminal.pins.values(), input_net_to_opamps)
        if target and target.id in placements:
            tx, ty, _ = placements[target.id]
            placements[terminal.id] = (max(1.2, tx - 2.4), _nearest_input_y(target, terminal.pins.values(), ty), "right")
        else:
            placements[terminal.id] = (1.5, 2.6 + fallback_inputs * 1.5, "right")
            fallback_inputs += 1

    for terminal in [component for component in circuit.components if _is_output(component)]:
        driver = _first_net_driver(terminal.pins.values(), output_net_to_opamp)
        if driver and driver.id in placements:
            dx0, dy0, _ = placements[driver.id]
            output_index = output_counts.get(driver.id, 0)
            output_counts[driver.id] = output_index + 1
            placements[terminal.id] = (
                min(span_x - 1.3, dx0 + 4.9),
                min(span_y - 2.1, max(1.2, dy0 + _spread_offset(output_index, 1.05))),
                "right",
            )
        else:
            placements[terminal.id] = (span_x - 1.5, 3.0 + fallback_outputs * 1.5, "right")
            fallback_outputs += 1

    grounds = [component for component in circuit.components if _is_ground(component)]
    ground_spacing = 3.4 if any(_long_label(ground) for ground in grounds) else 2.4
    ground_start_x = span_x * 0.5 - ground_spacing * max(0, len(grounds) - 1) / 2.0
    for index, ground in enumerate(grounds):
        placements[ground.id] = (
            min(span_x - 1.4, max(1.4, ground_start_x + index * ground_spacing)),
            span_y - 1.7,
            "right",
        )

    passive_index = 0
    for component in circuit.components:
        if component.id in placements or _is_opamp(component):
            continue
        slot = _op_amp_passive_slot(component, placements, opamps, output_net_to_opamp, input_net_to_opamps, grounds)
        if slot is None:
            col = passive_index % max(1, columns)
            row = passive_index // max(1, columns)
            slot = (2.7 + col * dx, span_y - 3.3 - row * 1.25, "right")
            passive_index += 1
        placements[component.id] = slot

    label_offsets: dict[str, tuple[float, float]] = {}
    for opamp in opamps:
        label_offsets[opamp.id] = (4.25, -0.85) if long_label_mode else (1.9, -1.25)
    row_split_y = y0 + max(1.0, dy * 0.5)
    for component in circuit.components:
        if "feedback" in _key(component.role) and component.id in placements:
            if long_label_mode:
                label_offsets[component.id] = (2.05, -0.75)
            elif placements[component.id][1] > row_split_y:
                label_offsets[component.id] = (0.0, -1.5)
        if _is_output(component) and _is_monitor_output(component):
            label_offsets[component.id] = (0.0, -1.45)

    routed_circuit = Circuit(id=circuit.id, motif="op_amp_network", title=circuit.title, components=circuit.components)
    return _build_layout(
        routed_circuit,
        placements,
        ["motif: op_amp_network"],
        width=width,
        height=height,
        label_offsets=label_offsets,
    )


def _ordered_opamps(circuit: Circuit, opamps: list[Component]) -> list[Component]:
    by_id = {component.id: component for component in opamps}
    original_index = {component.id: index for index, component in enumerate(opamps)}
    output_driver = _output_net_to_opamp(opamps)
    dependencies: dict[str, set[str]] = {component.id: set() for component in opamps}
    dependents: dict[str, set[str]] = {component.id: set() for component in opamps}
    for opamp in opamps:
        for net in _opamp_input_nets(opamp):
            driver = output_driver.get(net)
            if driver is None or driver.id == opamp.id:
                continue
            dependencies[opamp.id].add(driver.id)
            dependents[driver.id].add(opamp.id)

    ready = sorted([component_id for component_id, deps in dependencies.items() if not deps], key=original_index.get)
    ordered: list[Component] = []
    while ready:
        component_id = ready.pop(0)
        ordered.append(by_id[component_id])
        for dependent_id in sorted(dependents[component_id], key=original_index.get):
            dependencies[dependent_id].discard(component_id)
            if not dependencies[dependent_id] and dependent_id not in ready and by_id[dependent_id] not in ordered:
                ready.append(dependent_id)
        ready.sort(key=original_index.get)

    ordered_ids = {component.id for component in ordered}
    ordered.extend(component for component in opamps if component.id not in ordered_ids)
    return ordered


def _op_amp_passive_slot(
    component: Component,
    placements: dict[str, tuple[float, float, str]],
    opamps: list[Component],
    output_net_to_opamp: dict[str, Component],
    input_net_to_opamps: dict[str, list[Component]],
    grounds: list[Component],
) -> tuple[float, float, str] | None:
    nets = list(component.pins.values())
    if len(nets) < 2:
        return None
    ground_nets = {net for ground in grounds for net in ground.pins.values()} | {"0", "gnd", "ground"}

    for opamp in opamps:
        if opamp.id not in placements:
            continue
        output_net = _opamp_output_net(opamp)
        if output_net and output_net in nets and any(net in nets for net in _opamp_input_nets(opamp)):
            x, y, _ = placements[opamp.id]
            return (x + 1.75, max(1.2, y - 1.95), "right")

    active_nets = [net for net in nets if net not in ground_nets]
    if any(net in ground_nets for net in nets):
        owner = _first_net_opamp(active_nets, input_net_to_opamps) or _first_net_driver(active_nets, output_net_to_opamp)
        if owner and owner.id in placements:
            x, y, _ = placements[owner.id]
            return (x + 0.7, y + 2.45, "down")

    driver = _first_net_driver(nets, output_net_to_opamp)
    receiver = _first_net_opamp(nets, input_net_to_opamps, exclude=driver.id if driver else None)
    if driver and receiver and driver.id in placements and receiver.id in placements:
        sx, sy, _ = placements[driver.id]
        tx, ty, _ = placements[receiver.id]
        return ((sx + tx) / 2.0 + 1.6, (sy + ty) / 2.0, "right" if sx <= tx else "left")

    return None


def _spread_offset(index: int, step: float) -> float:
    if index == 0:
        return 0.0
    rank = (index + 1) // 2
    sign = 1 if index % 2 else -1
    return sign * rank * step


def _long_label(component: Component) -> bool:
    text = component.label or component.value or component.id
    return len(text) > 18


def _display_label(component: LayoutComponent) -> str:
    return component_display_label(component.id, component.type, component.label, component.value)


def _is_monitor_output(component: Component) -> bool:
    text = f"{component.id} {component.label or ''} {component.role or ''}"
    return "monitor" in _key(text) or _key(component.id).startswith("vmon")


def _output_net_to_opamp(opamps: list[Component]) -> dict[str, Component]:
    result: dict[str, Component] = {}
    for opamp in opamps:
        output_net = _opamp_output_net(opamp)
        if output_net:
            result.setdefault(output_net, opamp)
    return result


def _input_net_to_opamps(opamps: list[Component]) -> dict[str, list[Component]]:
    result: dict[str, list[Component]] = {}
    for opamp in opamps:
        for net in _opamp_input_nets(opamp):
            result.setdefault(net, []).append(opamp)
    return result


def _opamp_output_net(opamp: Component) -> str | None:
    for pin_name, net in opamp.pins.items():
        if _is_opamp_output_pin(pin_name):
            return net
    return None


def _opamp_input_nets(opamp: Component) -> list[str]:
    return [net for pin_name, net in opamp.pins.items() if not _is_opamp_output_pin(pin_name)]


def _is_opamp_output_pin(pin_name: str) -> bool:
    return _pin_kind(pin_name) in {"out", "output", "o", "vout"}


def _first_net_opamp(
    nets,
    input_net_to_opamps: dict[str, list[Component]],
    *,
    exclude: str | None = None,
) -> Component | None:
    for net in nets:
        for opamp in input_net_to_opamps.get(net, []):
            if opamp.id != exclude:
                return opamp
    return None


def _first_net_driver(nets, output_net_to_opamp: dict[str, Component]) -> Component | None:
    for net in nets:
        driver = output_net_to_opamp.get(net)
        if driver is not None:
            return driver
    return None


def _nearest_input_y(opamp: Component, nets, opamp_y: float) -> float:
    net_set = set(nets)
    for pin_name, net in opamp.pins.items():
        if net not in net_set:
            continue
        kind = _pin_kind(pin_name)
        if kind in {"-", "minus", "inverting"}:
            return opamp_y - 0.625
        if kind in {"+", "plus", "non_inverting"}:
            return opamp_y + 0.625
    return opamp_y


def _plan_diagnostic(circuit: Circuit) -> LayoutPlan:
    placements = {
        component.id: (2.0 + index * 2.6, 5.0, "right")
        for index, component in enumerate(circuit.components)
    }
    return _build_layout(circuit, placements, ["diagnostic: generic layout"])


def _build_layout(
    circuit: Circuit,
    placements: dict[str, tuple[float, float, str]],
    warnings: list[str],
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    grid: int = GRID,
    label_offsets: dict[str, tuple[float, float]] | None = None,
) -> LayoutPlan:
    fallback_index = 0
    components: list[LayoutComponent] = []
    for component in circuit.components:
        if component.id not in placements:
            placements[component.id] = (2.0 + fallback_index * 2.8, 15.0, "right")
            fallback_index += 1
        x, y, orientation = placements[component.id]
        components.append(_layout_component(component, x, y, orientation))

    pin_map = _pin_map(components)
    net_to_pins: dict[str, list[tuple[str, str]]] = {}
    for key, pin in pin_map.items():
        net_to_pins.setdefault(pin.net, []).append(key)
    net_to_pins = {net: sorted(pins) for net, pins in sorted(net_to_pins.items())}

    labels = _labels_for_components(components, label_offsets or {})
    return LayoutPlan(
        circuit_id=circuit.id,
        width=width,
        height=height,
        grid=grid,
        components=components,
        wires=_route_wires(pin_map, net_to_pins, motif=_canonical_motif(circuit.motif) or _key(circuit.id)),
        labels=labels,
        pin_map=pin_map,
        net_to_pins=net_to_pins,
        topology_signature=topology_signature(circuit),
        warnings=warnings,
    )


def _layout_component(component: Component, x: float, y: float, orientation: str) -> LayoutComponent:
    return LayoutComponent(
        id=component.id,
        type=component.type,
        role=component.role,
        label=component.label,
        value=component.value,
        x=x,
        y=y,
        orientation=orientation,
        pins=dict(component.pins),
        bbox=_component_bbox(component.type, x, y, orientation),
    )


def _component_bbox(component_type: str, x: float, y: float, orientation: str) -> BBox:
    key = _key(component_type)
    if _is_opamp_type(key):
        return BBox(x - 0.1, y - 1.05, 3.65, 2.1)
    if key in {"input", "output", "input_terminal", "voltage_source", "source"} or "source" in key:
        return BBox(x - 0.55, y - 0.55, 1.1, 1.1)
    if key in {"ground", "gnd"}:
        return BBox(x - 0.65, y - 0.75, 1.3, 1.0)
    if orientation in {"up", "down"}:
        return BBox(x - 0.35, y - 1.0, 0.7, 2.0)
    return BBox(x - 1.0, y - 0.35, 2.0, 0.7)


def _labels_for_components(
    components: list[LayoutComponent],
    label_offsets: dict[str, tuple[float, float]],
) -> list[LayoutLabel]:
    labels: list[LayoutLabel] = []
    for component in components:
        text = _display_label(component)
        dx, dy = label_offsets.get(component.id, _default_label_offset(component))
        x, y = component.x + dx, component.y + dy
        width, height = label_bbox_size(text)
        labels.append(
            LayoutLabel(
                id=f"label:{component.id}",
                text=text,
                owner_id=component.id,
                x=x,
                y=y,
                anchor="middle",
                bbox=BBox(x - width / 2.0, y - height / 2.0, width, height),
            )
        )
    return labels


def _default_label_offset(component: LayoutComponent) -> tuple[float, float]:
    key = _key(component.type)
    if _is_terminal_layout(component):
        return (0.0, -0.72)
    if key in {"ground", "gnd"}:
        return (0.0, 1.05)
    if _is_opamp_layout(component):
        return (1.45, -1.45)
    if component.orientation in {"up", "down"}:
        return (0.62, 0.0)
    return (0.0, -0.9)


def _pin_map(components: list[LayoutComponent]) -> dict[tuple[str, str], LayoutPin]:
    pins: dict[tuple[str, str], LayoutPin] = {}
    for component in components:
        for pin_name, net in component.pins.items():
            x, y, side = _pin_point(component, pin_name)
            pins[(component.id, pin_name)] = LayoutPin(component.id, pin_name, net, x, y, side)
    return pins


def _route_wires(
    pin_map: dict[tuple[str, str], LayoutPin],
    net_to_pins: dict[str, list[tuple[str, str]]],
    *,
    motif: str | None = None,
) -> list[LayoutWire]:
    motif_routes = _route_known_motif(pin_map, net_to_pins, motif)
    if motif_routes is not None:
        return motif_routes

    wires: list[LayoutWire] = []
    for net, connected in sorted(net_to_pins.items()):
        route = _generic_route([Point(pin_map[key].x, pin_map[key].y) for key in connected])
        if not route:
            continue
        wires.append(LayoutWire(net=net, points=_dedupe(route), connected_pins=connected))
    return wires


def _generic_route(points: list[Point]) -> list[Point]:
    if len(points) < 2:
        return []
    if len(points) == 2:
        start, end = points
        if start.x == end.x or start.y == end.y:
            return [start, end]
        mid_x = (start.x + end.x) / 2.0
        return [start, Point(mid_x, start.y), Point(mid_x, end.y), end]
    hub = Point(float(median(point.x for point in points)), float(median(point.y for point in points)))
    route = [hub]
    for point in points:
        elbow = Point(hub.x, point.y)
        route.extend([elbow, point, elbow, hub])
    return route


def _route_known_motif(
    pin_map: dict[tuple[str, str], LayoutPin],
    net_to_pins: dict[str, list[tuple[str, str]]],
    motif: str | None,
) -> list[LayoutWire] | None:
    motif_key = _key(motif)
    builders: dict[str, Callable[[dict[tuple[str, str], LayoutPin], dict[str, list[tuple[str, str]]]], dict[str, list[Point]] | None]] = {
        "voltage_divider": _voltage_divider_routes,
        "rc_low_pass": _rc_low_pass_routes,
        "non_inverting_op_amp": _non_inverting_op_amp_routes,
        "instrumentation_amplifier": _instrumentation_amplifier_routes,
        "bridge_or_wheatstone": _bridge_routes,
        "op_amp_network": _op_amp_network_routes,
    }
    builder = builders.get(motif_key)
    if builder is None:
        return None
    routes = builder(pin_map, net_to_pins)
    if routes is None:
        return None
    wires: list[LayoutWire] = []
    for net, connected in sorted(net_to_pins.items()):
        route = routes.get(net)
        if route is None:
            route = _generic_route([_point(pin_map, key) for key in connected])
        if route:
            wires.append(LayoutWire(net=net, points=_dedupe(route), connected_pins=connected))
    return wires


def _voltage_divider_routes(
    pin_map: dict[tuple[str, str], LayoutPin],
    net_to_pins: dict[str, list[tuple[str, str]]],
) -> dict[str, list[Point]] | None:
    required = [("VIN", "out"), ("R1", "a"), ("R1", "b"), ("R2", "a"), ("R2", "b"), ("VOUT", "in"), ("GND", "gnd")]
    if not _has_pins(pin_map, required):
        return None
    vhub = Point(_p(pin_map, "R1", "b").x, _p(pin_map, "VOUT", "in").y)
    return {
        "vin": [_p(pin_map, "VIN", "out"), _p(pin_map, "R1", "a")],
        "vout": [_p(pin_map, "R1", "b"), vhub, _p(pin_map, "VOUT", "in"), vhub, _p(pin_map, "R2", "a")],
        "gnd": [_p(pin_map, "R2", "b"), _p(pin_map, "GND", "gnd")],
    }


def _rc_low_pass_routes(
    pin_map: dict[tuple[str, str], LayoutPin],
    net_to_pins: dict[str, list[tuple[str, str]]],
) -> dict[str, list[Point]] | None:
    required = [("VIN", "out"), ("R1", "a"), ("R1", "b"), ("C1", "a"), ("C1", "b"), ("VOUT", "in"), ("GND", "gnd")]
    if not _has_pins(pin_map, required):
        return None
    hub = Point(_p(pin_map, "C1", "a").x, _p(pin_map, "R1", "b").y)
    return {
        "vin": [_p(pin_map, "VIN", "out"), _p(pin_map, "R1", "a")],
        "vout": [_p(pin_map, "R1", "b"), hub, _p(pin_map, "VOUT", "in"), hub, _p(pin_map, "C1", "a")],
        "gnd": [_p(pin_map, "C1", "b"), _p(pin_map, "GND", "gnd")],
    }


def _non_inverting_op_amp_routes(
    pin_map: dict[tuple[str, str], LayoutPin],
    net_to_pins: dict[str, list[tuple[str, str]]],
) -> dict[str, list[Point]] | None:
    required = [
        ("VIN", "out"),
        ("U1", "+"),
        ("U1", "-"),
        ("U1", "out"),
        ("Rf", "a"),
        ("Rf", "b"),
        ("Rg", "a"),
        ("Rg", "b"),
        ("VOUT", "in"),
        ("GND", "gnd"),
    ]
    if not _has_pins(pin_map, required):
        return None
    out = _p(pin_map, "U1", "out")
    minus = _p(pin_map, "U1", "-")
    vm_bus = Point(_p(pin_map, "Rg", "a").x, minus.y)
    return {
        "vin": [_p(pin_map, "VIN", "out"), _p(pin_map, "U1", "+")],
        "vout": [out, _p(pin_map, "VOUT", "in"), out, Point(out.x, _p(pin_map, "Rf", "a").y), _p(pin_map, "Rf", "a")],
        "vm": [_p(pin_map, "Rf", "b"), Point(_p(pin_map, "Rf", "b").x, minus.y), minus, vm_bus, _p(pin_map, "Rg", "a")],
        "gnd": [_p(pin_map, "Rg", "b"), _p(pin_map, "GND", "gnd")],
    }


def _instrumentation_amplifier_routes(
    pin_map: dict[tuple[str, str], LayoutPin],
    net_to_pins: dict[str, list[tuple[str, str]]],
) -> dict[str, list[Point]] | None:
    required = [
        ("INP", "out"),
        ("INN", "out"),
        ("U1", "+"),
        ("U1", "-"),
        ("U1", "out"),
        ("U2", "+"),
        ("U2", "-"),
        ("U2", "out"),
        ("U3", "+"),
        ("U3", "-"),
        ("U3", "out"),
        ("R1", "a"),
        ("R1", "b"),
        ("R2", "a"),
        ("R2", "b"),
        ("Rg", "a"),
        ("Rg", "b"),
        ("R3", "a"),
        ("R3", "b"),
        ("R4", "a"),
        ("R4", "b"),
        ("R5", "a"),
        ("R5", "b"),
        ("R6", "a"),
        ("R6", "b"),
        ("VOUT", "in"),
        ("GND", "gnd"),
    ]
    if not _has_pins(pin_map, required):
        return None

    u1_out = _p(pin_map, "U1", "out")
    u2_out = _p(pin_map, "U2", "out")
    u3_out = _p(pin_map, "U3", "out")
    u1_minus = _p(pin_map, "U1", "-")
    u2_minus = _p(pin_map, "U2", "-")
    u3_minus = _p(pin_map, "U3", "-")
    rg_x = _p(pin_map, "Rg", "a").x
    o1_bus_x = u1_out.x + 0.45
    o2_bus_x = u2_out.x + 0.3
    return {
        "vinp": [_p(pin_map, "INP", "out"), _p(pin_map, "U1", "+")],
        "vinn": [_p(pin_map, "INN", "out"), _p(pin_map, "U2", "+")],
        "n1": [
            _p(pin_map, "R1", "b"),
            Point(rg_x, _p(pin_map, "R1", "b").y),
            Point(rg_x, u1_minus.y),
            u1_minus,
            Point(rg_x, u1_minus.y),
            _p(pin_map, "Rg", "a"),
        ],
        "n2": [
            _p(pin_map, "R2", "b"),
            Point(rg_x, _p(pin_map, "R2", "b").y),
            Point(rg_x, u2_minus.y),
            u2_minus,
            Point(rg_x, u2_minus.y),
            _p(pin_map, "Rg", "b"),
        ],
        "o1": [
            u1_out,
            Point(o1_bus_x, u1_out.y),
            Point(o1_bus_x, _p(pin_map, "R1", "a").y),
            _p(pin_map, "R1", "a"),
            Point(o1_bus_x, _p(pin_map, "R1", "a").y),
            Point(o1_bus_x, _p(pin_map, "R3", "a").y),
            _p(pin_map, "R3", "a"),
        ],
        "o2": [
            u2_out,
            Point(o2_bus_x, u2_out.y),
            Point(o2_bus_x, _p(pin_map, "R2", "a").y),
            _p(pin_map, "R2", "a"),
            Point(o2_bus_x, _p(pin_map, "R2", "a").y),
            Point(o2_bus_x, _p(pin_map, "R5", "a").y),
            _p(pin_map, "R5", "a"),
        ],
        "n3": [_p(pin_map, "R3", "b"), _p(pin_map, "U3", "+")],
        "gnd": [_p(pin_map, "R4", "b"), _p(pin_map, "GND", "gnd")],
        "n4": [
            _p(pin_map, "R5", "b"),
            u3_minus,
            Point(_p(pin_map, "R6", "a").x, u3_minus.y),
            _p(pin_map, "R6", "a"),
        ],
        "vout": [u3_out, _p(pin_map, "VOUT", "in"), u3_out, Point(u3_out.x, _p(pin_map, "R6", "b").y), _p(pin_map, "R6", "b")],
    }


def _bridge_routes(
    pin_map: dict[tuple[str, str], LayoutPin],
    net_to_pins: dict[str, list[tuple[str, str]]],
) -> dict[str, list[Point]] | None:
    required = [("VIN", "out"), ("R1", "a"), ("R1", "b"), ("R2", "a"), ("R2", "b"), ("R3", "a"), ("R3", "b"), ("R4", "a"), ("R4", "b"), ("VOUT", "in"), ("GND", "gnd")]
    if not _has_pins(pin_map, required):
        return None
    vl_hub = Point(_p(pin_map, "R1", "b").x, _p(pin_map, "VOUT", "in").y)
    gnd_hub = Point((_p(pin_map, "R2", "b").x + _p(pin_map, "R4", "b").x) / 2.0, _p(pin_map, "R2", "b").y)
    return {
        "vin": [_p(pin_map, "VIN", "out"), _p(pin_map, "R1", "a"), _p(pin_map, "R3", "a"), _p(pin_map, "R1", "a")],
        "vl": [_p(pin_map, "R1", "b"), vl_hub, _p(pin_map, "VOUT", "in"), vl_hub, _p(pin_map, "R2", "a")],
        "vr": [_p(pin_map, "R3", "b"), _p(pin_map, "R4", "a")],
        "gnd": [_p(pin_map, "R2", "b"), gnd_hub, _p(pin_map, "R4", "b"), gnd_hub, _p(pin_map, "GND", "gnd")],
    }


def _op_amp_network_routes(
    pin_map: dict[tuple[str, str], LayoutPin],
    net_to_pins: dict[str, list[tuple[str, str]]],
) -> dict[str, list[Point]] | None:
    routes: dict[str, list[Point]] = {}
    right_gutter = max(pin.x for pin in pin_map.values()) + 1.25
    for net, connected in net_to_pins.items():
        pins = [pin_map[key] for key in connected]
        if _net_has_ground_pin(pins):
            routes[net] = _op_amp_ground_route(pins, right_gutter=right_gutter)
            continue
        driver = next((pin for pin in pins if _is_opamp_output_pin(pin.pin_name) and pin.side == "right"), None)
        receivers = [
            pin
            for pin in pins
            if pin is not driver and _pin_kind(pin.pin_name) in {"+", "-", "plus", "minus", "non_inverting", "inverting"}
        ]
        if driver and receivers:
            routes[net] = _op_amp_output_route(
                driver,
                receivers,
                [pin for pin in pins if pin is not driver and pin not in receivers],
                right_gutter=right_gutter,
            )
    return routes


def _op_amp_ground_route(pins: list[LayoutPin], *, right_gutter: float) -> list[Point]:
    points = [Point(pin.x, pin.y) for pin in pins]
    bus_y = max(point.y for point in points) - 0.55
    gutter_x = right_gutter
    hub = Point(gutter_x, bus_y)
    route = [hub]
    for point in points:
        if point.y < bus_y - 2.0:
            elbow = Point(gutter_x, point.y)
            route.extend([elbow, point, elbow, hub])
        else:
            tap = Point(point.x, bus_y)
            route.extend([tap, point, tap, hub])
    return route


def _op_amp_output_route(
    driver: LayoutPin,
    receivers: list[LayoutPin],
    branches: list[LayoutPin],
    *,
    right_gutter: float,
) -> list[Point]:
    driver_point = Point(driver.x, driver.y)
    route = [driver_point]
    for branch in sorted(branches, key=lambda pin: (abs(pin.y - driver.y), pin.x)):
        elbow = Point(driver.x, branch.y)
        branch_point = Point(branch.x, branch.y)
        route.extend([elbow, branch_point, elbow, driver_point])

    for receiver in sorted(receivers, key=lambda pin: (pin.y, pin.x)):
        receiver_point = Point(receiver.x, receiver.y)
        if receiver.x < driver.x - 0.5 or abs(receiver.y - driver.y) > 2.0:
            bend_y = (driver.y + receiver.y) / 2.0 + (0.55 if receiver.y > driver.y else -0.55)
            right_x = right_gutter
            left_x = min(driver.x, receiver.x) - 1.4
            route.extend(
                [
                    Point(right_x, driver.y),
                    Point(right_x, bend_y),
                    Point(left_x, bend_y),
                    Point(left_x, receiver.y),
                    receiver_point,
                    Point(left_x, receiver.y),
                    Point(left_x, bend_y),
                    Point(right_x, bend_y),
                    Point(right_x, driver.y),
                    driver_point,
                ]
            )
        else:
            mid_x = (driver.x + receiver.x) / 2.0
            route.extend([Point(mid_x, driver.y), Point(mid_x, receiver.y), receiver_point, Point(mid_x, receiver.y), Point(mid_x, driver.y), driver_point])
    return route


def _net_has_ground_pin(pins: list[LayoutPin]) -> bool:
    return any(_key(pin.pin_name) == "gnd" or _key(pin.net) in {"0", "gnd", "ground"} for pin in pins)


def _has_pins(pin_map: dict[tuple[str, str], LayoutPin], pins: list[tuple[str, str]]) -> bool:
    return all(pin in pin_map for pin in pins)


def _p(pin_map: dict[tuple[str, str], LayoutPin], component_id: str, pin_name: str) -> Point:
    return _point(pin_map, (component_id, pin_name))


def _point(pin_map: dict[tuple[str, str], LayoutPin], key: tuple[str, str]) -> Point:
    pin = pin_map[key]
    return Point(pin.x, pin.y)


def _pin_point(component: LayoutComponent, pin_name: str) -> tuple[float, float, str]:
    kind = _pin_kind(pin_name)
    x, y = component.x, component.y
    if _is_opamp_layout(component):
        if kind in {"-", "minus", "inverting"}:
            return (x - 0.12, y - 0.625, "left")
        if kind in {"+", "plus", "non_inverting"}:
            return (x - 0.12, y + 0.625, "left")
        return (x + 3.57, y, "right")
    if _is_ground_layout(component):
        return (x, y, "top")
    if _is_terminal_layout(component):
        return (x, y, "center")
    if component.orientation in {"up", "down"}:
        return (x, y - 0.95, "top") if _is_first_pin(component, pin_name) else (x, y + 0.95, "bottom")
    if component.orientation in {"left", "west"}:
        return (x + 0.95, y, "right") if _is_first_pin(component, pin_name) else (x - 0.95, y, "left")
    return (x - 0.95, y, "left") if _is_first_pin(component, pin_name) else (x + 0.95, y, "right")


def shifted_layout(layout: LayoutPlan, moves: dict[str, tuple[float, float]]) -> LayoutPlan:
    components = [
        replace(component, x=moves.get(component.id, (component.x, component.y))[0], y=moves.get(component.id, (component.x, component.y))[1])
        if component.id in moves
        else component
        for component in layout.components
    ]
    updated = replace(layout, components=components)
    return rebuild_layout_geometry(updated)


def _dedupe(points: list[Point]) -> list[Point]:
    result: list[Point] = []
    for point in points:
        if not result or point != result[-1]:
            result.append(point)
    return result


def _first(circuit: Circuit, predicate: Callable[[Component], bool]) -> Component | None:
    return next((component for component in circuit.components if predicate(component)), None)


def _infer_motif(components: list[Component]) -> str | None:
    opamps = sum(1 for component in components if _is_opamp(component))
    resistors = sum(1 for component in components if _is_type(component, "resistor"))
    capacitors = sum(1 for component in components if _is_type(component, "capacitor"))
    if opamps >= 4:
        return "op_amp_network"
    if opamps == 3 and resistors >= 7:
        return "instrumentation_amplifier"
    if opamps >= 2:
        return "op_amp_network"
    if opamps == 1:
        return "non_inverting_op_amp"
    if resistors >= 1 and capacitors >= 1:
        return "rc_low_pass"
    if resistors >= 4:
        return "bridge_or_wheatstone"
    if resistors == 2:
        return "voltage_divider"
    return None


def _validated_motif(circuit: Circuit) -> str | None:
    motif = _canonical_motif(circuit.motif)
    if motif is None:
        return None
    opamps = sum(1 for component in circuit.components if _is_opamp(component))
    resistors = sum(1 for component in circuit.components if _is_type(component, "resistor"))
    capacitors = sum(1 for component in circuit.components if _is_type(component, "capacitor"))
    if motif in {"voltage_divider", "rc_low_pass", "bridge_or_wheatstone"} and opamps:
        return None
    if motif == "voltage_divider" and (resistors != 2 or capacitors):
        return None
    if motif == "rc_low_pass" and (resistors < 1 or capacitors < 1):
        return None
    if motif == "bridge_or_wheatstone" and resistors < 4:
        return None
    if motif == "instrumentation_amplifier" and (opamps != 3 or resistors < 7):
        return None
    if motif == "non_inverting_op_amp" and (opamps != 1 or resistors < 2):
        return None
    if motif == "op_amp_network" and opamps < 2:
        return None
    return motif


def _canonical_motif(value: str | None) -> str | None:
    key = _key(value)
    aliases = {
        "divider": "voltage_divider",
        "potential_divider": "voltage_divider",
        "resistive_divider": "voltage_divider",
        "voltage_divider": "voltage_divider",
        "rc_filter": "rc_low_pass",
        "rc_low_pass": "rc_low_pass",
        "rc_low_pass_filter": "rc_low_pass",
        "low_pass": "rc_low_pass",
        "lowpass": "rc_low_pass",
        "non_inverting_amplifier": "non_inverting_op_amp",
        "non_inverting_op_amp": "non_inverting_op_amp",
        "non_inverting_opamp": "non_inverting_op_amp",
        "noninverting_amplifier": "non_inverting_op_amp",
        "noninverting_op_amp": "non_inverting_op_amp",
        "noninverting_opamp": "non_inverting_op_amp",
        "noninv": "non_inverting_op_amp",
        "instrumentation_amplifier": "instrumentation_amplifier",
        "instrumentation_amp": "instrumentation_amplifier",
        "in_amp": "instrumentation_amplifier",
        "ina": "instrumentation_amplifier",
        "bridge": "bridge_or_wheatstone",
        "bridge_or_wheatstone": "bridge_or_wheatstone",
        "wheatstone": "bridge_or_wheatstone",
        "wheatstone_bridge": "bridge_or_wheatstone",
        "analog_front_end": "op_amp_network",
        "multi_op_amp": "op_amp_network",
        "multi_opamp": "op_amp_network",
        "op_amp_chain": "op_amp_network",
        "op_amp_network": "op_amp_network",
        "opamp_chain": "op_amp_network",
        "opamp_network": "op_amp_network",
    }
    return aliases.get(key)


def _is_type(component: Component, needle: str) -> bool:
    key = _key(component.type)
    return needle in key or key.startswith(needle[0])


def _has_role(component: Component, needle: str) -> bool:
    return needle in _key(component.role)


def _is_opamp(component: Component) -> bool:
    return _is_opamp_type(_key(component.type))


def _is_opamp_type(key: str) -> bool:
    return "op_amp" in key or "opamp" in key or "operational_amplifier" in key or "ideal_op_amp" in key


def _is_input_or_source(component: Component) -> bool:
    key = _key(component.type)
    role = _key(component.role)
    return key in {"input", "input_terminal", "voltage_source", "source"} or "input" in role or "source" in key


def _is_output(component: Component) -> bool:
    return _key(component.type) == "output" or "output" in _key(component.role)


def _is_ground(component: Component) -> bool:
    key = _key(component.type)
    return key in {"ground", "gnd"} or "ground" in _key(component.role)


def _is_first_pin(component: LayoutComponent, pin_name: str) -> bool:
    return list(component.pins).index(pin_name) == 0


def _is_opamp_layout(component: LayoutComponent) -> bool:
    return _is_opamp_type(_key(component.type))


def _is_ground_layout(component: LayoutComponent) -> bool:
    key = _key(component.type)
    return key in {"ground", "gnd"} or "ground" in _key(component.role)


def _is_output_layout(component: LayoutComponent) -> bool:
    return _key(component.type) == "output" or "output" in _key(component.role)


def _is_terminal_layout(component: LayoutComponent) -> bool:
    key = _key(component.type)
    return key in {"input", "output", "input_terminal", "voltage_source", "source"} or "source" in key


def _pin_kind(pin_name: str) -> str:
    compact = _key(pin_name).replace("_", "")
    if pin_name in {"+", "-"}:
        return pin_name
    if compact in {"plus", "noninverting", "noninv", "inp", "vp"}:
        return "+"
    if compact in {"minus", "inverting", "inv", "inn", "vn"}:
        return "-"
    return compact


def _component_by_id(layout: LayoutPlan, component_id: str) -> LayoutComponent:
    return next(component for component in layout.components if component.id == component_id)


def _has_component(layout: LayoutPlan, component_id: str) -> bool:
    return any(component.id == component_id for component in layout.components)


def _key(value: str | None) -> str:
    return (value or "").lower().replace("-", "_").replace(" ", "_")
