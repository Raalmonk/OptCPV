"""Deterministic initial layout planners for OptCPV."""

from __future__ import annotations

from dataclasses import replace
from statistics import median
from typing import Callable

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
    motif = _key(native.motif) or _infer_motif(native.components)
    planners: dict[str, Callable[[Circuit], LayoutPlan]] = {
        "voltage_divider": _plan_voltage_divider,
        "rc_low_pass": _plan_rc_low_pass,
        "non_inverting_op_amp": _plan_non_inverting_op_amp,
        "instrumentation_amplifier": _plan_instrumentation_amplifier,
        "bridge_or_wheatstone": _plan_bridge,
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
        placements[source.id] = (2.0, 4.0, "right")
    if top:
        placements[top.id] = (5.0, 4.0, "down")
    if bottom:
        placements[bottom.id] = (5.0, 7.2, "down")
    if output:
        placements[output.id] = (8.8, 5.6, "right")
    if ground:
        placements[ground.id] = (5.0, 10.0, "down")
    return _build_layout(circuit, placements, ["motif: voltage_divider"])


def _plan_rc_low_pass(circuit: Circuit) -> LayoutPlan:
    placements: dict[str, tuple[float, float, str]] = {}
    source = _first(circuit, _is_input_or_source)
    resistor = _first(circuit, lambda item: _is_type(item, "resistor"))
    capacitor = _first(circuit, lambda item: _is_type(item, "capacitor"))
    output = _first(circuit, _is_output)
    ground = _first(circuit, _is_ground)
    if source:
        placements[source.id] = (2.0, 5.0, "right")
    if resistor:
        placements[resistor.id] = (5.0, 5.0, "right")
    if capacitor:
        placements[capacitor.id] = (8.0, 7.5, "down")
    if output:
        placements[output.id] = (10.8, 5.0, "right")
    if ground:
        placements[ground.id] = (8.0, 10.2, "down")
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
        placements[source.id] = (2.0, 7.0, "right")
    if opamp:
        placements[opamp.id] = (7.0, 6.5, "right")
    if feedback:
        placements[feedback.id] = (7.0, 2.7, "right")
    if gain:
        placements[gain.id] = (4.1, 9.2, "down")
    if output:
        placements[output.id] = (11.6, 6.5, "right")
    if ground:
        placements[ground.id] = (4.1, 11.8, "down")
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
        "input_top": (1.5, 4.0, "right"),
        "input_bottom": (1.5, 11.0, "right"),
        "u1": (6.5, 4.0, "right"),
        "u2": (6.5, 11.0, "right"),
        "u3": (15.7, 7.5, "right"),
        "output": (21.0, 7.5, "right"),
        "ground": (13.5, 14.0, "down"),
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
        (6.5, 1.2, "right"),
        (6.5, 13.8, "right"),
        (4.2, 7.5, "down"),
        (10.8, 5.1, "right"),
        (10.8, 9.9, "right"),
        (15.7, 3.6, "right"),
        (13.5, 11.5, "down"),
    ]
    for component, slot in zip(resistors, resistor_slots):
        placements[component.id] = slot
    return _build_layout(circuit, placements, ["motif: instrumentation_amplifier"])


def _plan_bridge(circuit: Circuit) -> LayoutPlan:
    placements: dict[str, tuple[float, float, str]] = {}
    resistors = [component for component in circuit.components if _is_type(component, "resistor")]
    for component, slot in zip(resistors, [(5, 4, "down"), (5, 8, "down"), (9, 4, "down"), (9, 8, "down")]):
        placements[component.id] = slot
    for component, slot in zip(
        [item for item in circuit.components if item.id not in placements],
        [(2, 4, "right"), (12, 6.0, "right"), (7, 11, "down")],
    ):
        placements[component.id] = slot
    return _build_layout(circuit, placements, ["motif: bridge_or_wheatstone"])


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
        wires=_route_wires(pin_map, net_to_pins),
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
        return BBox(x - 1.25, y - 1.05, 2.7, 2.1)
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
        text = component.label or component.value or component.id
        dx, dy = label_offsets.get(component.id, (0.0, -0.95 if not _is_opamp_layout(component) else -1.35))
        x, y = component.x + dx, component.y + dy
        width = max(0.7, 0.16 * len(text) + 0.35)
        labels.append(
            LayoutLabel(
                id=f"label:{component.id}",
                text=text,
                owner_id=component.id,
                x=x,
                y=y,
                anchor="middle",
                bbox=BBox(x - width / 2.0, y - 0.22, width, 0.38),
            )
        )
    return labels


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
) -> list[LayoutWire]:
    wires: list[LayoutWire] = []
    for net, connected in sorted(net_to_pins.items()):
        points = [Point(pin_map[key].x, pin_map[key].y) for key in connected]
        if len(points) < 2:
            continue
        if len(points) == 2:
            start, end = points
            if start.x == end.x or start.y == end.y:
                route = [start, end]
            else:
                mid_x = (start.x + end.x) / 2.0
                route = [start, Point(mid_x, start.y), Point(mid_x, end.y), end]
        else:
            hub = Point(float(median(point.x for point in points)), float(median(point.y for point in points)))
            route = [hub]
            for point in points:
                elbow = Point(hub.x, point.y)
                route.extend([elbow, point, elbow, hub])
        wires.append(LayoutWire(net=net, points=_dedupe(route), connected_pins=connected))
    return wires


def _pin_point(component: LayoutComponent, pin_name: str) -> tuple[float, float, str]:
    kind = _pin_kind(pin_name)
    x, y = component.x, component.y
    if _is_opamp_layout(component):
        if kind in {"-", "minus", "inverting"}:
            return (x - 1.4, y - 0.7, "left")
        if kind in {"+", "plus", "non_inverting"}:
            return (x - 1.4, y + 0.7, "left")
        return (x + 1.7, y, "right")
    if _is_ground_layout(component):
        return (x, y - 0.7, "top")
    if _is_terminal_layout(component):
        return (x - 0.5, y, "left") if _is_output_layout(component) else (x + 0.5, y, "right")
    if component.orientation in {"up", "down"}:
        return (x, y - 0.95, "top") if _is_first_pin(component, pin_name) else (x, y + 0.95, "bottom")
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
    if opamps >= 3:
        return "instrumentation_amplifier"
    if opamps == 1:
        return "non_inverting_op_amp"
    if resistors >= 1 and capacitors >= 1:
        return "rc_low_pass"
    if resistors >= 4:
        return "bridge_or_wheatstone"
    if resistors == 2:
        return "voltage_divider"
    return None


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
