"""Small deterministic planners for common circuit motifs."""

from __future__ import annotations

from statistics import median
from typing import Callable

from .models import Circuit, Component, Layout, LayoutComponent, LayoutWire, circuit_from_any


GRID = 48


def plan_layout(circuit: Circuit | dict) -> Layout:
    native = circuit_from_any(circuit)
    motif = _key(native.motif) or _infer_motif(native.components)
    planners: dict[str, Callable[[Circuit], Layout]] = {
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


def _plan_voltage_divider(circuit: Circuit) -> Layout:
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
        placements[bottom.id] = (5.0, 8.0, "down")
    if output:
        placements[output.id] = (9.0, 6.0, "right")
    if ground:
        placements[ground.id] = (5.0, 11.0, "down")
    return _build_layout(circuit, placements, ["motif: voltage_divider"])


def _plan_rc_low_pass(circuit: Circuit) -> Layout:
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
        placements[capacitor.id] = (8.0, 8.0, "down")
    if output:
        placements[output.id] = (11.0, 5.0, "right")
    if ground:
        placements[ground.id] = (8.0, 11.0, "down")
    return _build_layout(circuit, placements, ["motif: rc_low_pass"])


def _plan_non_inverting_op_amp(circuit: Circuit) -> Layout:
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
        placements[feedback.id] = (7.0, 2.5, "right")
    if gain:
        placements[gain.id] = (4.0, 9.5, "down")
    if output:
        placements[output.id] = (12.0, 6.5, "right")
    if ground:
        placements[ground.id] = (4.0, 12.5, "down")
    return _build_layout(circuit, placements, ["motif: non_inverting_op_amp"])


def _plan_instrumentation_amplifier(circuit: Circuit) -> Layout:
    placements: dict[str, tuple[float, float, str]] = {}
    opamps = [component for component in circuit.components if _is_opamp(component)]
    inputs = [component for component in circuit.components if _is_input_or_source(component)]
    output = _first(circuit, _is_output)
    ground = _first(circuit, _is_ground)
    resistors = [component for component in circuit.components if _is_type(component, "resistor")]

    slots = {
        "input_top": (2.0, 4.0, "right"),
        "input_bottom": (2.0, 11.0, "right"),
        "u1": (7.0, 4.0, "right"),
        "u2": (7.0, 11.0, "right"),
        "u3": (15.0, 7.5, "right"),
        "output": (20.0, 7.5, "right"),
        "ground": (13.0, 14.0, "down"),
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
        (7.0, 1.2, "right"),
        (7.0, 13.8, "right"),
        (4.4, 7.5, "down"),
        (11.0, 5.3, "right"),
        (11.0, 9.7, "right"),
        (15.0, 3.8, "right"),
        (13.0, 11.5, "down"),
    ]
    for component, slot in zip(resistors, resistor_slots):
        placements[component.id] = slot
    return _build_layout(circuit, placements, ["motif: instrumentation_amplifier"])


def _plan_bridge(circuit: Circuit) -> Layout:
    placements: dict[str, tuple[float, float, str]] = {}
    resistors = [component for component in circuit.components if _is_type(component, "resistor")]
    for component, slot in zip(resistors, [(5, 4, "down"), (5, 9, "down"), (9, 4, "down"), (9, 9, "down")]):
        placements[component.id] = slot
    for component, slot in zip(
        [item for item in circuit.components if item.id not in placements],
        [(2, 4, "right"), (12, 6.5, "right"), (7, 12, "down")],
    ):
        placements[component.id] = slot
    return _build_layout(circuit, placements, ["motif: bridge_or_wheatstone"])


def _plan_diagnostic(circuit: Circuit) -> Layout:
    placements = {
        component.id: (2.0 + index * 3.0, 5.0, "right")
        for index, component in enumerate(circuit.components)
    }
    return _build_layout(circuit, placements, ["diagnostic: generic layout"])


def _build_layout(
    circuit: Circuit,
    placements: dict[str, tuple[float, float, str]],
    warnings: list[str],
) -> Layout:
    fallback_index = 0
    components: list[LayoutComponent] = []
    for component in circuit.components:
        if component.id not in placements:
            placements[component.id] = (2.0 + fallback_index * 3.0, 16.0, "right")
            fallback_index += 1
        x, y, orientation = placements[component.id]
        components.append(
            LayoutComponent(
                id=component.id,
                x=x,
                y=y,
                orientation=orientation,
                type=component.type,
                role=component.role,
                label=component.label,
                value=component.value,
                pins=dict(component.pins),
            )
        )
    return Layout(
        circuit_id=circuit.id,
        width=1100,
        height=800,
        components=components,
        wires=_route_wires(components),
        warnings=warnings,
    )


def _route_wires(components: list[LayoutComponent]) -> list[LayoutWire]:
    net_points: dict[str, list[tuple[float, float]]] = {}
    for component in components:
        for pin_name, net in component.pins.items():
            net_points.setdefault(net, []).append(_pin_point(component, pin_name))

    wires: list[LayoutWire] = []
    for net, points in sorted(net_points.items()):
        if len(points) < 2:
            continue
        if len(points) == 2:
            start, end = points
            if start[0] == end[0] or start[1] == end[1]:
                route = [start, end]
            else:
                mid_x = (start[0] + end[0]) / 2.0
                route = [start, (mid_x, start[1]), (mid_x, end[1]), end]
        else:
            hub = (float(median(point[0] for point in points)), float(median(point[1] for point in points)))
            route = []
            for index, point in enumerate(points):
                if index:
                    route.append(hub)
                route.extend([point, (hub[0], point[1]), hub])
        wires.append(LayoutWire(net=net, points=_dedupe(route)))
    return wires


def _pin_point(component: LayoutComponent, pin_name: str) -> tuple[float, float]:
    kind = _pin_kind(pin_name)
    x, y = component.x, component.y
    if _is_opamp_layout(component):
        if kind in {"-", "minus", "inverting"}:
            return (x - 1.4, y - 0.7)
        if kind in {"+", "plus", "non_inverting"}:
            return (x - 1.4, y + 0.7)
        return (x + 1.7, y)
    if _is_ground_layout(component):
        return (x, y - 0.7)
    if _is_terminal_layout(component):
        return (x + 0.5 if not _is_output_layout(component) else x - 0.5, y)
    if component.orientation in {"up", "down"}:
        return (x, y - 0.9 if _is_first_pin(component, pin_name) else y + 0.9)
    return (x - 0.9 if _is_first_pin(component, pin_name) else x + 0.9, y)


def _dedupe(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
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
    if resistors == 2:
        return "voltage_divider"
    if resistors >= 4:
        return "bridge_or_wheatstone"
    return None


def _is_type(component: Component, needle: str) -> bool:
    key = _key(component.type)
    return needle in key or key.startswith(needle[0])


def _has_role(component: Component, needle: str) -> bool:
    return needle in _key(component.role)


def _is_opamp(component: Component) -> bool:
    key = _key(component.type)
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
    return _is_opamp(Component(component.id, component.type, component.pins, role=component.role))


def _is_ground_layout(component: LayoutComponent) -> bool:
    return _is_ground(Component(component.id, component.type, component.pins, role=component.role))


def _is_output_layout(component: LayoutComponent) -> bool:
    return _is_output(Component(component.id, component.type, component.pins, role=component.role))


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


def _key(value: str | None) -> str:
    return (value or "").lower().replace("-", "_").replace(" ", "_")
