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
    LayoutSupport,
    LayoutWire,
    Lane,
    LocalTerminalIntent,
    Motif,
    NetClass,
    Point,
    RouteIntent,
    Stage,
    TopologySemanticPlan,
    circuit_from_any,
)
from .hint_legalizer import legalize_planning_hints
from .planning_agent import SemanticPlanningClient
from .planning_hints import RoutePolicyHint, SchematicLayoutHints
from .route_contract import assert_no_diagonal_wires, orthogonalize_route
from .semantics import (
    classify_net,
    is_local_terminal_net,
    is_negative_supply_pin,
    is_positive_supply_pin,
    is_reference_pin,
    preferred_terminal_direction,
    terminal_label,
    terminal_type_for_net,
)
from .segments import is_axis_aligned
from .symbols import OPAMP_HALF_HEIGHT, OPAMP_INPUT_LEAD_X, OPAMP_INPUT_LEAD_Y, OPAMP_OUTPUT_LEAD_X
from .verifier import topology_signature


GRID = 48
DEFAULT_WIDTH = 1100
DEFAULT_HEIGHT = 800
CANVAS_EDGE_PADDING = 0.55
NATIVE_MOTIFS = {
    "voltage_divider",
    "rc_low_pass",
    "non_inverting_op_amp",
    "instrumentation_amplifier",
    "bridge_or_wheatstone",
}


def plan_layout(
    circuit: Circuit | dict,
    *,
    planning_hints: SchematicLayoutHints | dict | None = None,
    planning_client: SemanticPlanningClient | None = None,
    reference_image: bytes | None = None,
) -> LayoutPlan:
    native = circuit_from_any(circuit)
    base_layout = _plan_layout_deterministic(native)
    hints = _resolve_planning_hints(
        native,
        planning_hints=planning_hints,
        planning_client=planning_client,
        reference_image=reference_image,
    )
    if hints is None:
        return base_layout
    candidate = _layout_from_planning_hints(native, base_layout, hints)
    if candidate is None:
        return base_layout
    return candidate


def _plan_layout_deterministic(native: Circuit) -> LayoutPlan:
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
        support=layout.support,
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
    return _build_layout(circuit, placements, ["motif: voltage_divider"], support=_native_motif_support("voltage_divider"))


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
    return _build_layout(circuit, placements, ["motif: rc_low_pass"], support=_native_motif_support("rc_low_pass"))


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
    return _build_layout(
        circuit,
        placements,
        ["motif: non_inverting_op_amp"],
        support=_native_motif_support("non_inverting_op_amp"),
    )


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
    return _build_layout(
        circuit,
        placements,
        ["motif: instrumentation_amplifier"],
        support=_native_motif_support("instrumentation_amplifier"),
    )


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
    return _build_layout(circuit, placements, ["motif: bridge_or_wheatstone"], support=_native_motif_support("bridge_or_wheatstone"))


def _plan_op_amp_network(circuit: Circuit) -> LayoutPlan:
    placements: dict[str, tuple[float, float, str]] = {}
    opamps = _ordered_opamps(circuit, [component for component in circuit.components if _is_opamp(component)])
    semantic_chain = _plan_parallel_summing_signal_chain(circuit, opamps)
    if semantic_chain is not None:
        return semantic_chain
    single_row_network = 5 <= len(opamps) <= 7
    if single_row_network:
        width = 1680
        height = 700
    else:
        width = 1400 if len(opamps) >= 3 else DEFAULT_WIDTH
        height = 850 if len(opamps) >= 3 else DEFAULT_HEIGHT
    span_x = width / GRID
    span_y = height / GRID
    columns = len(opamps) if single_row_network else min(4, max(1, len(opamps)))
    rows = max(1, ceil(len(opamps) / columns))
    x0 = 2.4 if single_row_network else (3.2 if width > DEFAULT_WIDTH else 3.0)
    dx = (span_x - 8.2) / max(1, columns - 1) if single_row_network else (6.0 if width > DEFAULT_WIDTH else 5.3)
    long_label_mode = any(_long_label(component) for component in circuit.components)
    y0 = 4.0 if single_row_network else (4.15 if long_label_mode else 3.3)
    dy = min(6.4, max(4.8, (span_y - 5.4) / max(1, rows - 1))) if rows > 1 else 0.0

    for index, opamp in enumerate(opamps):
        row, col = divmod(index, columns)
        placements[opamp.id] = (x0 + col * dx, y0 + row * dy, _opamp_orientation(opamp, circuit.components))

    output_net_to_opamp = _output_net_to_opamp(opamps)
    input_net_to_opamps = _input_net_to_opamps(opamps)
    fallback_inputs = 0
    fallback_outputs = 0
    output_counts: dict[str, int] = {}

    for terminal in [component for component in circuit.components if _is_input_or_source(component)]:
        target = _first_net_opamp(terminal.pins.values(), input_net_to_opamps)
        if target and target.id in placements:
            tx, ty, _ = placements[target.id]
            placements[terminal.id] = (
                max(1.2, tx - 2.4),
                _nearest_input_y(target, terminal.pins.values(), ty, placements[target.id][2]),
                "right",
            )
        else:
            placements[terminal.id] = (1.5, 2.6 + fallback_inputs * 1.5, "right")
            fallback_inputs += 1

    for terminal in [component for component in circuit.components if _is_output(component)]:
        driver = _first_net_driver(terminal.pins.values(), output_net_to_opamp)
        if driver and driver.id in placements:
            dx0, dy0, _ = placements[driver.id]
            output_index = output_counts.get(driver.id, 0)
            output_counts[driver.id] = output_index + 1
            if single_row_network and _is_monitor_output(terminal):
                placements[terminal.id] = (
                    min(span_x - 1.3, dx0 + 3.57),
                    max(1.35, dy0 - 2.45 - output_index * 0.8),
                    "right",
                )
            else:
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
    if single_row_network:
        for component in circuit.components:
            if _is_input_or_source(component):
                label_offsets[component.id] = (0.0, -1.65)
            elif _is_output(component):
                label_offsets[component.id] = (0.0, -1.2)
    row_split_y = y0 + max(1.0, dy * 0.5)
    for component in circuit.components:
        if "feedback" in _key(component.role) and component.id in placements:
            if long_label_mode:
                label_offsets[component.id] = (2.05, -0.75)
            elif placements[component.id][1] > row_split_y:
                label_offsets[component.id] = (0.0, -1.5)
        if _is_output(component) and _is_monitor_output(component):
            label_offsets[component.id] = (0.0, -0.82 if single_row_network else -1.45)
        elif single_row_network and _is_output(component):
            label_offsets[component.id] = (0.95, -0.62)
        elif single_row_network and _is_input_or_source(component):
            label_offsets[component.id] = (-0.72, -0.62)
        if single_row_network and _is_opamp(component) and component.id in placements:
            label_offsets[component.id] = (1.7, -2.08)
        if single_row_network and _is_type(component, "resistor") and "feedback" not in _key(component.role):
            label_offsets[component.id] = (1.0, 0.0)

    routed_circuit = Circuit(id=circuit.id, motif="op_amp_network", title=circuit.title, components=circuit.components)
    return _build_layout(
        routed_circuit,
        placements,
        ["motif: op_amp_network"],
        width=width,
        height=height,
        label_offsets=label_offsets,
        support=_op_amp_network_support(),
    )


def _plan_parallel_summing_signal_chain(circuit: Circuit, opamps: list[Component]) -> LayoutPlan | None:
    if len(opamps) < 2:
        return None
    resistors = [component for component in circuit.components if _is_type(component, "resistor")]
    filters = [component for component in circuit.components if _is_filter_block_component(component)]
    summing = _find_summing_opamp(opamps, resistors)
    if summing is None:
        return None

    input_resistors = _summing_input_resistors(summing, resistors)
    if len(input_resistors) < 2:
        return None

    output_to_opamp = _output_net_to_opamp(opamps)
    input_buffers: list[Component] = []
    for resistor in input_resistors:
        source_net = next((net for net in resistor.pins.values() if net not in _opamp_input_nets(summing)), None)
        source = output_to_opamp.get(source_net or "")
        if source is not None and source.id != summing.id and source not in input_buffers:
            input_buffers.append(source)
    if len(input_buffers) < 2:
        return None

    width = 1600
    height = 740
    span_x = width / GRID
    lanes = _lane_y_values(len(input_buffers), center=6.4, spacing=2.8)
    lane_by_source = {opamp.id: lanes[index] for index, opamp in enumerate(input_buffers)}
    center_y = sum(lanes) / len(lanes)
    placements: dict[str, tuple[float, float, str]] = {}

    for opamp in input_buffers:
        placements[opamp.id] = (4.0, lane_by_source[opamp.id], _opamp_orientation(opamp, circuit.components))

    for component in [item for item in circuit.components if _is_input_or_source(item)]:
        target = _first_net_opamp(component.pins.values(), _input_net_to_opamps(input_buffers))
        y = lane_by_source.get(target.id, center_y) if target else center_y
        placements[component.id] = (1.4, y, "right")

    sum_input_nets = set(_opamp_input_nets(summing))
    for resistor in input_resistors:
        source_net = next((net for net in resistor.pins.values() if net not in sum_input_nets), None)
        source = output_to_opamp.get(source_net or "")
        y = lane_by_source.get(source.id, center_y) if source else center_y
        placements[resistor.id] = (8.6, y, "right")

    placements[summing.id] = (12.4, center_y, _opamp_orientation(summing, circuit.components))
    for resistor in _feedback_resistors_for_component_opamp(summing, resistors):
        placements[resistor.id] = (13.5, max(1.25, center_y - 2.4), "left")

    x_cursor = 17.6
    downstream_opamps = [
        opamp
        for opamp in opamps
        if opamp.id not in {summing.id, *(buffer.id for buffer in input_buffers)}
    ]
    remaining_filters = list(filters)
    if remaining_filters:
        placements[remaining_filters.pop(0).id] = (x_cursor, center_y, "right")
        x_cursor += 5.4
    if downstream_opamps:
        gain = downstream_opamps[0]
        placements[gain.id] = (x_cursor, center_y, _opamp_orientation(gain, circuit.components))
        for resistor in _feedback_resistors_for_component_opamp(gain, resistors):
            if resistor.id in placements:
                continue
            placements[resistor.id] = (x_cursor + 1.05, max(1.25, center_y - 2.4), "left")
        for resistor in _ground_leg_resistors(gain, resistors):
            if resistor.id in placements:
                continue
            placements[resistor.id] = (x_cursor + 0.85, center_y + 2.55, "down")
        x_cursor += 5.4
    for filter_block in remaining_filters:
        placements[filter_block.id] = (x_cursor, center_y, "right")
        x_cursor += 4.5

    for terminal in [component for component in circuit.components if _is_output(component)]:
        placements[terminal.id] = (min(span_x - 1.4, x_cursor), center_y, "right")

    grounds = [component for component in circuit.components if _is_ground(component)]
    for index, ground in enumerate(grounds):
        placements[ground.id] = (2.1 + index * 2.6, max(lanes) + 2.75, "right")

    passive_index = 0
    for component in circuit.components:
        if component.id in placements:
            continue
        if _is_type(component, "resistor"):
            slot = _semantic_resistor_slot(component, placements, opamps, resistors)
            if slot is not None:
                placements[component.id] = slot
                continue
        if _is_filter_block_component(component):
            placements[component.id] = (x_cursor, center_y, "right")
            x_cursor += 4.5
            continue
        col = passive_index % 4
        row = passive_index // 4
        placements[component.id] = (7.0 + col * 2.8, max(lanes) + 2.2 + row * 1.25, "right")
        passive_index += 1

    label_offsets: dict[str, tuple[float, float]] = {}
    for opamp in opamps:
        if opamp.id in placements:
            label_offsets[opamp.id] = (1.75, -1.28)
    for filter_block in filters:
        label_offsets[filter_block.id] = (0.0, -1.22)
    for component in circuit.components:
        if _is_input_or_source(component) or _is_output(component):
            label_offsets[component.id] = (0.0, -0.85)

    routed_circuit = Circuit(id=circuit.id, motif="op_amp_network", title=circuit.title, components=circuit.components)
    return _build_layout(
        routed_circuit,
        placements,
        ["motif: op_amp_network", "semantic: parallel_summing_signal_chain"],
        width=width,
        height=height,
        label_offsets=label_offsets,
        support=_op_amp_network_support(),
    )


def _lane_y_values(count: int, *, center: float, spacing: float) -> list[float]:
    if count <= 1:
        return [center]
    start = center - spacing * (count - 1) / 2.0
    return [start + index * spacing for index in range(count)]


def _find_summing_opamp(opamps: list[Component], resistors: list[Component]) -> Component | None:
    candidates: list[tuple[int, Component]] = []
    for opamp in opamps:
        count = len(_summing_input_resistors(opamp, resistors))
        if count >= 2:
            candidates.append((count, opamp))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _summing_input_resistors(opamp: Component, resistors: list[Component]) -> list[Component]:
    input_nets = set(_opamp_input_nets(opamp))
    output_net = _opamp_output_net(opamp)
    result: list[Component] = []
    for resistor in resistors:
        nets = set(resistor.pins.values())
        if output_net in nets:
            continue
        if input_nets & nets and not any(is_local_terminal_net(net) for net in nets):
            result.append(resistor)
    return result


def _feedback_resistors_for_component_opamp(opamp: Component, resistors: list[Component]) -> list[Component]:
    output_net = _opamp_output_net(opamp)
    input_nets = set(_opamp_input_nets(opamp))
    if output_net is None:
        return []
    if output_net in input_nets:
        return [
            resistor
            for resistor in resistors
            if output_net in set(resistor.pins.values()) and "feedback" in _key(resistor.role)
        ]
    return [
        resistor
        for resistor in resistors
        if output_net in set(resistor.pins.values())
        and (input_nets & set(resistor.pins.values()) or "feedback" in _key(resistor.role))
    ]


def _ground_leg_resistors(opamp: Component, resistors: list[Component]) -> list[Component]:
    input_nets = set(_opamp_input_nets(opamp))
    result: list[Component] = []
    for resistor in resistors:
        nets = set(resistor.pins.values())
        if input_nets & nets and any(is_local_terminal_net(net) for net in nets):
            result.append(resistor)
    return result


def _semantic_resistor_slot(
    component: Component,
    placements: dict[str, tuple[float, float, str]],
    opamps: list[Component],
    resistors: list[Component],
) -> tuple[float, float, str] | None:
    for opamp in opamps:
        if opamp.id not in placements:
            continue
        if component in _feedback_resistors_for_component_opamp(opamp, resistors):
            x, y, _ = placements[opamp.id]
            return (x + 1.05, max(1.35, y - 2.55), "left")
        if component in _ground_leg_resistors(opamp, resistors):
            x, y, _ = placements[opamp.id]
            return (x + 0.85, y + 2.7, "down")
    return None


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
            if _is_flipped_opamp_orientation(placements[opamp.id][2]):
                return (x + 2.25, y + 1.75, "left")
            return (x + 1.75, max(1.2, y - 1.95), "left")

    active_nets = [net for net in nets if net not in ground_nets]
    if any(net in ground_nets for net in nets):
        owner = _first_net_opamp(active_nets, input_net_to_opamps) or _first_net_driver(active_nets, output_net_to_opamp)
        if owner and owner.id in placements:
            x, y, _ = placements[owner.id]
            if _is_flipped_opamp_orientation(placements[owner.id][2]):
                return (x + 0.7, y + 2.75, "down")
            return (x + 0.7, y + 2.45, "down")

    driver = _first_net_driver(nets, output_net_to_opamp)
    receiver = _first_net_opamp(nets, input_net_to_opamps, exclude=driver.id if driver else None)
    if driver and receiver and driver.id in placements and receiver.id in placements:
        sx, sy, _ = placements[driver.id]
        tx, ty, _ = placements[receiver.id]
        return ((sx + tx) / 2.0 + 1.6, (sy + ty) / 2.0, "right" if sx <= tx else "left")

    return None


def _opamp_orientation(opamp: Component, components: list[Component]) -> str:
    return "right_flip" if _should_flip_non_inverting_stage(opamp, components) else "right"


def _should_flip_non_inverting_stage(opamp: Component, components: list[Component]) -> bool:
    plus_net = _opamp_named_input_net(opamp, "+")
    minus_net = _opamp_named_input_net(opamp, "-")
    output_net = _opamp_output_net(opamp)
    if plus_net is None or minus_net is None or output_net is None:
        return False
    if is_local_terminal_net(plus_net):
        return False
    if minus_net == output_net:
        return True
    attached = [
        component
        for component in components
        if component.id != opamp.id and minus_net in component.pins.values()
    ]
    has_feedback = any(output_net in component.pins.values() for component in attached)
    has_ground_leg = any(
        any(is_local_terminal_net(net) for net in component.pins.values() if net != minus_net)
        for component in attached
    )
    return has_feedback and has_ground_leg


def _opamp_named_input_net(opamp: Component, target_kind: str) -> str | None:
    for pin_name, net in opamp.pins.items():
        if _pin_kind(pin_name) == target_kind:
            return net
    return None


def _is_flipped_opamp_orientation(orientation: str) -> bool:
    return "flip" in _key(orientation)


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
    return [
        net
        for pin_name, net in opamp.pins.items()
        if not _is_opamp_output_pin(pin_name)
        and not is_positive_supply_pin(pin_name, net)
        and not is_negative_supply_pin(pin_name, net)
        and not is_reference_pin(pin_name, net)
    ]


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


def _nearest_input_y(opamp: Component, nets, opamp_y: float, orientation: str = "right") -> float:
    net_set = set(nets)
    flip = _is_flipped_opamp_orientation(orientation)
    for pin_name, net in opamp.pins.items():
        if net not in net_set:
            continue
        kind = _pin_kind(pin_name)
        if kind in {"-", "minus", "inverting"}:
            return opamp_y + 0.625 if flip else opamp_y - 0.625
        if kind in {"+", "plus", "non_inverting"}:
            return opamp_y - 0.625 if flip else opamp_y + 0.625
    return opamp_y


def _plan_diagnostic(circuit: Circuit) -> LayoutPlan:
    placements = {
        component.id: (2.0 + index * 2.6, 5.0, "right")
        for index, component in enumerate(circuit.components)
    }
    return _build_layout(circuit, placements, ["diagnostic: generic layout"], support=_diagnostic_support())


def _native_motif_support(motif: str) -> LayoutSupport:
    return LayoutSupport(
        layout_mode="native_motif",
        layout_confidence=0.95,
        matched_motifs=(motif,),
        fallback_used=False,
        notes=("schemdraw native motif renderer available",),
    )


def _op_amp_network_support() -> LayoutSupport:
    return LayoutSupport(
        layout_mode="motif_network",
        layout_confidence=0.72,
        matched_motifs=("op_amp_network",),
        fallback_used=False,
        notes=("heuristic multi-op-amp placement; not a textbook guarantee",),
    )


def _diagnostic_support() -> LayoutSupport:
    return LayoutSupport(
        layout_mode="diagnostic_fallback",
        layout_confidence=0.25,
        matched_motifs=(),
        fallback_used=True,
        unsupported_regions=("circuit:unknown_topology",),
        notes=("generic diagnostic layout; no known motif matched",),
    )


def _support_from_warnings(warnings: list[str]) -> LayoutSupport:
    motif = _warning_motif(warnings)
    if motif in NATIVE_MOTIFS:
        return _native_motif_support(motif)
    if motif == "op_amp_network":
        return _op_amp_network_support()
    if any(warning.startswith("diagnostic:") for warning in warnings):
        return _diagnostic_support()
    return LayoutSupport(notes=("layout support metadata was not supplied by planner",))


def _support_with_fallback_components(support: LayoutSupport, component_ids: list[str]) -> LayoutSupport:
    if not component_ids:
        return support
    unsupported = _unique((*support.unsupported_regions, *(f"component:{component_id}" for component_id in component_ids)))
    notes = _unique((*support.notes, "one or more components used generic fallback placement"))
    mode = "partial_motif" if support.matched_motifs else support.layout_mode
    return replace(
        support,
        layout_mode=mode,
        layout_confidence=min(support.layout_confidence, 0.55),
        fallback_used=True,
        unsupported_regions=unsupported,
        notes=notes,
    )


def _warning_motif(warnings: list[str]) -> str | None:
    for warning in warnings:
        if warning.startswith("motif:"):
            return _key(warning.split(":", 1)[1].strip())
    return None


def _unique(values) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _route_motif(circuit: Circuit, support: LayoutSupport) -> str:
    if support.matched_motifs:
        return support.matched_motifs[0]
    return _canonical_motif(circuit.motif) or _key(circuit.id)


def _resolve_planning_hints(
    circuit: Circuit,
    *,
    planning_hints: SchematicLayoutHints | dict | None,
    planning_client: SemanticPlanningClient | None,
    reference_image: bytes | None,
) -> SchematicLayoutHints | None:
    if planning_hints is not None and planning_client is not None:
        raise ValueError("Pass either planning_hints or planning_client, not both.")
    if isinstance(planning_hints, dict):
        return SchematicLayoutHints.from_dict(planning_hints)
    if planning_hints is not None:
        return planning_hints
    if planning_client is None:
        return None
    try:
        return planning_client.propose_hints(circuit, reference_image=reference_image)
    except TypeError:
        return planning_client.propose_hints(circuit)


def _layout_from_planning_hints(circuit: Circuit, base_layout: LayoutPlan, hints: SchematicLayoutHints) -> LayoutPlan | None:
    legal_hints = legalize_planning_hints(circuit, hints)
    if legal_hints is None:
        return None
    try:
        placements = _hint_placements(circuit, base_layout, legal_hints)
    except ValueError:
        return None
    if placements is None:
        return None

    support = replace(
        base_layout.support,
        layout_confidence=max(base_layout.support.layout_confidence, legal_hints.confidence),
        fallback_used=False,
        notes=_unique((*base_layout.support.notes, "external semantic planning hints applied")),
        planning_hints=legal_hints.to_dict(),
        tutor_explanation=legal_hints.tutor_explanation or base_layout.support.tutor_explanation,
    )
    candidate = _build_layout(
        circuit,
        placements,
        [*base_layout.warnings, "planning_hints: accepted"],
        width=base_layout.width,
        height=base_layout.height,
        grid=base_layout.grid,
        support=support,
    )
    candidate = _improve_hint_opamp_orientations(circuit, candidate)
    try:
        from .vector_critic import critique_layout
        from .verifier import verify_layout_topology

        verify_layout_topology(circuit, candidate)
        base_report = critique_layout(base_layout)
        candidate_report = critique_layout(candidate)
        assert_no_diagonal_wires(candidate)
    except Exception:
        return None
    base_hard = sum(1 for violation in base_report.violations if violation.hard)
    candidate_hard = sum(1 for violation in candidate_report.violations if violation.hard)
    if any(violation.code == "component_overlap" for violation in candidate_report.violations):
        return None
    if candidate_hard > base_hard or candidate_report.score > base_report.score:
        return None
    return candidate


def _improve_hint_opamp_orientations(circuit: Circuit, layout: LayoutPlan) -> LayoutPlan:
    if not layout.support.planning_hints:
        return layout
    try:
        from .vector_critic import critique_layout
        from .verifier import verify_layout_topology

        verify_layout_topology(circuit, layout)
        best = layout
        best_report = critique_layout(layout)
        for _ in range(2):
            improved = False
            for component in sorted(best.components, key=lambda item: item.id):
                if not _is_opamp_layout(component):
                    continue
                placements = {
                    item.id: (item.x, item.y, item.orientation)
                    for item in best.components
                }
                flipped = "right_flip" if component.orientation == "right" else "right"
                placements[component.id] = (component.x, component.y, flipped)
                candidate = _build_layout(
                    circuit,
                    placements,
                    list(best.warnings),
                    width=best.width,
                    height=best.height,
                    grid=best.grid,
                    support=best.support,
                )
                verify_layout_topology(circuit, candidate)
                candidate_report = critique_layout(candidate)
                if _orientation_candidate_better(candidate_report, best_report):
                    best = candidate
                    best_report = candidate_report
                    improved = True
            if not improved:
                break
        return best
    except Exception:
        return layout


def _orientation_candidate_better(candidate_report, best_report) -> bool:
    candidate_hard = sum(1 for violation in candidate_report.violations if violation.hard)
    best_hard = sum(1 for violation in best_report.violations if violation.hard)
    if any(violation.code == "component_overlap" for violation in candidate_report.violations):
        return False
    if candidate_hard > best_hard:
        return False
    if candidate_hard < best_hard:
        return True
    return candidate_report.score < best_report.score - 0.1


def _hint_placements(
    circuit: Circuit,
    base_layout: LayoutPlan,
    hints: SchematicLayoutHints,
) -> dict[str, tuple[float, float, str]] | None:
    base_components = {component.id: component for component in base_layout.components}
    placements = {component.id: (component.x, component.y, component.orientation) for component in base_layout.components}
    span_x = base_layout.width / base_layout.grid
    span_y = base_layout.height / base_layout.grid
    orientations = {
        hint.component_id: _legalized_hint_orientation(base_components[hint.component_id], hint.orientation)
        for hint in hints.placements
    }
    stage_positions = _hint_stage_positions(hints, base_components, orientations, span_x)
    lane_positions = _hint_lane_positions(hints, base_components, orientations, span_y)

    for hint in hints.placements:
        component_id = hint.component_id
        component = base_components[component_id]
        orientation = orientations[component_id]
        x = _clamp(stage_positions[hint.stage_x], 0.6, span_x - 0.6)
        y = _clamp(lane_positions[hint.lane_y], 0.6, span_y - 0.6)
        x, y = _clamp_anchor_to_canvas(component, x, y, orientation, span_x, span_y)
        placements[component_id] = (x, y, orientation)
    placements = _anchor_hint_ground_legs(circuit, base_components, placements, span_x, span_y)
    placements = _orient_hint_feedback_resistors(circuit, base_components, placements, span_x, span_y)
    return _relax_hint_component_overlaps(base_components, placements, span_x, span_y)


def _anchor_hint_ground_legs(
    circuit: Circuit,
    components: dict[str, LayoutComponent],
    placements: dict[str, tuple[float, float, str]],
    span_x: float,
    span_y: float,
) -> dict[str, tuple[float, float, str]]:
    result = dict(placements)
    opamps = [component for component in circuit.components if _is_opamp(component) and component.id in result]
    for component in circuit.components:
        if component.id not in result or not _is_type(component, "resistor"):
            continue
        nets = set(component.pins.values())
        if not any(is_local_terminal_net(net) for net in nets):
            continue
        active_nets = [net for net in nets if not is_local_terminal_net(net)]
        if len(active_nets) != 1:
            continue
        owner = next((opamp for opamp in opamps if active_nets[0] in _opamp_input_nets(opamp)), None)
        if owner is None:
            continue
        owner_x, owner_y, owner_orientation = result[owner.id]
        x = owner_x + 1.35
        y = owner_y + (2.75 if _is_flipped_opamp_orientation(owner_orientation) else 2.45)
        x, y = _clamp_anchor_to_canvas(components[component.id], x, y, "right", span_x, span_y)
        result[component.id] = (x, y, "right")
    return result


def _orient_hint_feedback_resistors(
    circuit: Circuit,
    components: dict[str, LayoutComponent],
    placements: dict[str, tuple[float, float, str]],
    span_x: float,
    span_y: float,
) -> dict[str, tuple[float, float, str]]:
    result = dict(placements)
    opamps = [component for component in circuit.components if _is_opamp(component) and component.id in result]
    for component in circuit.components:
        if component.id not in result or component.id not in components:
            continue
        if not _is_type(component, "resistor") or "feedback" not in _key(component.role):
            continue
        orientation = _feedback_resistor_orientation(component, opamps)
        if orientation is None:
            continue
        x, y, _ = result[component.id]
        x, y = _clamp_anchor_to_canvas(components[component.id], x, y, orientation, span_x, span_y)
        result[component.id] = (x, y, orientation)
    return result


def _feedback_resistor_orientation(resistor: Component, opamps: list[Component]) -> str | None:
    resistor_pins = resistor.pins
    for opamp in opamps:
        output_net = _opamp_output_net(opamp)
        input_nets = set(_opamp_input_nets(opamp))
        if output_net is None:
            continue
        output_pin = next((pin_name for pin_name, net in resistor_pins.items() if net == output_net), None)
        input_pin = next((pin_name for pin_name, net in resistor_pins.items() if net in input_nets), None)
        if output_pin is None or input_pin is None:
            continue
        if _pin_kind(output_pin) == "a" and _pin_kind(input_pin) == "b":
            return "left"
        if _pin_kind(output_pin) == "b" and _pin_kind(input_pin) == "a":
            return "right"
    return None


def _hint_stage_positions(
    hints: SchematicLayoutHints,
    components: dict[str, LayoutComponent],
    orientations: dict[str, str],
    span_x: float,
) -> dict[int, float]:
    stages = sorted({hint.stage_x for hint in hints.placements})
    if not stages:
        return {}
    extents: dict[int, tuple[float, float]] = {}
    for stage in stages:
        left = right = 0.0
        for hint in hints.placements:
            if hint.stage_x != stage:
                continue
            component = components[hint.component_id]
            component_left, component_right, _, _ = _hint_bbox_extents(component, orientations[component.id])
            left = max(left, component_left)
            right = max(right, component_right)
        extents[stage] = (left, right)

    minimum_stage = min(stages)
    maximum_stage = max(stages)
    naive_gap = min(4.35, (span_x - 2.4) / max(1, maximum_stage - minimum_stage))
    positions: dict[int, float] = {}
    cursor_right = 0.55
    for stage in stages:
        left, right = extents[stage]
        semantic_target = 1.2 + (stage - minimum_stage) * naive_gap
        x = max(semantic_target, cursor_right + left + 0.32)
        positions[stage] = x
        cursor_right = x + right

    overflow = cursor_right - (span_x - 0.55)
    if overflow > 0 and len(stages) > 1:
        first = positions[stages[0]]
        last = positions[stages[-1]]
        usable = max(1.0, last - first)
        scale = max(0.72, (usable - overflow) / usable)
        for stage in stages:
            positions[stage] = first + (positions[stage] - first) * scale
    return positions


def _hint_lane_positions(
    hints: SchematicLayoutHints,
    components: dict[str, LayoutComponent],
    orientations: dict[str, str],
    span_y: float,
) -> dict[int, float]:
    lanes = sorted({hint.lane_y for hint in hints.placements})
    if not lanes:
        return {}
    extents: dict[int, tuple[float, float]] = {}
    for lane in lanes:
        top = bottom = 0.0
        for hint in hints.placements:
            if hint.lane_y != lane:
                continue
            component = components[hint.component_id]
            _, _, component_top, component_bottom = _hint_bbox_extents(component, orientations[component.id])
            top = max(top, component_top)
            bottom = max(bottom, component_bottom)
        extents[lane] = (top, bottom)

    minimum_lane = min(lanes)
    maximum_lane = max(lanes)
    naive_gap = (span_y - 1.6) / max(1, maximum_lane - minimum_lane)
    positions: dict[int, float] = {}
    cursor_bottom = 0.45
    for lane in lanes:
        top, bottom = extents[lane]
        semantic_target = 0.8 + (lane - minimum_lane) * naive_gap
        y = max(semantic_target, cursor_bottom + top + 0.18)
        positions[lane] = y
        cursor_bottom = y + bottom

    overflow = cursor_bottom - (span_y - 0.45)
    if overflow > 0 and len(lanes) > 1:
        first = positions[lanes[0]]
        last = positions[lanes[-1]]
        usable = max(1.0, last - first)
        scale = max(0.78, (usable - overflow) / usable)
        for lane in lanes:
            positions[lane] = first + (positions[lane] - first) * scale
    return positions


def _hint_bbox_extents(component: LayoutComponent, orientation: str) -> tuple[float, float, float, float]:
    bbox = _component_bbox(component.type, 0.0, 0.0, orientation)
    return (-bbox.x, bbox.right, -bbox.y, bbox.bottom)


def _clamp_anchor_to_canvas(
    component: LayoutComponent,
    x: float,
    y: float,
    orientation: str,
    span_x: float,
    span_y: float,
) -> tuple[float, float]:
    bbox = _component_bbox(component.type, x, y, orientation)
    if bbox.x < CANVAS_EDGE_PADDING:
        x += CANVAS_EDGE_PADDING - bbox.x
    if bbox.right > span_x - CANVAS_EDGE_PADDING:
        x -= bbox.right - (span_x - CANVAS_EDGE_PADDING)
    bbox = _component_bbox(component.type, x, y, orientation)
    if bbox.y < CANVAS_EDGE_PADDING:
        y += CANVAS_EDGE_PADDING - bbox.y
    if bbox.bottom > span_y - CANVAS_EDGE_PADDING:
        y -= bbox.bottom - (span_y - CANVAS_EDGE_PADDING)
    return (x, y)


def _relax_hint_component_overlaps(
    components: dict[str, LayoutComponent],
    placements: dict[str, tuple[float, float, str]],
    span_x: float,
    span_y: float,
) -> dict[str, tuple[float, float, str]]:
    result = dict(placements)
    hinted_ids = [component_id for component_id in result if component_id in components]
    for _ in range(80):
        boxes = {
            component_id: _component_bbox(components[component_id].type, *result[component_id])
            for component_id in hinted_ids
        }
        overlap = _first_component_overlap(boxes)
        if overlap is None:
            return result
        left_id, right_id = overlap
        moving_id, fixed_id = _overlap_move_order(components[left_id], components[right_id])
        moving_box = boxes[moving_id]
        fixed_box = boxes[fixed_id]
        x, y, orientation = result[moving_id]
        overlap_x = min(moving_box.right, fixed_box.right) - max(moving_box.x, fixed_box.x)
        overlap_y = min(moving_box.bottom, fixed_box.bottom) - max(moving_box.y, fixed_box.y)
        if overlap_y <= overlap_x + 0.08:
            direction = 1.0 if moving_box.center.y >= fixed_box.center.y else -1.0
            y = _try_shift_anchor_y(components[moving_id], x, y, orientation, direction * (overlap_y + 0.22), span_y)
        else:
            direction = 1.0 if moving_box.center.x >= fixed_box.center.x else -1.0
            x = _try_shift_anchor_x(components[moving_id], x, y, orientation, direction * (overlap_x + 0.22), span_x)
        result[moving_id] = _clamp_anchor_to_canvas(components[moving_id], x, y, orientation, span_x, span_y) + (orientation,)
    return result


def _first_component_overlap(boxes: dict[str, BBox]) -> tuple[str, str] | None:
    ids = sorted(boxes)
    for index, left_id in enumerate(ids):
        for right_id in ids[index + 1 :]:
            if boxes[left_id].intersects(boxes[right_id], padding=0.05):
                return (left_id, right_id)
    return None


def _overlap_move_order(left: LayoutComponent, right: LayoutComponent) -> tuple[str, str]:
    if _component_layout_priority(left) < _component_layout_priority(right):
        return (left.id, right.id)
    if _component_layout_priority(right) < _component_layout_priority(left):
        return (right.id, left.id)
    return (right.id, left.id) if (right.x, right.y) >= (left.x, left.y) else (left.id, right.id)


def _component_layout_priority(component: LayoutComponent) -> int:
    if _is_opamp_layout(component):
        return 100
    if _is_input_or_source_layout(component) or _is_output_layout(component):
        return 80
    if _is_ground_layout(component):
        return 25
    return 50


def _try_shift_anchor_y(component: LayoutComponent, x: float, y: float, orientation: str, delta: float, span_y: float) -> float:
    for candidate in (y + delta, y - delta):
        bbox = _component_bbox(component.type, x, candidate, orientation)
        if bbox.y >= CANVAS_EDGE_PADDING and bbox.bottom <= span_y - CANVAS_EDGE_PADDING:
            return candidate
    return y + delta


def _try_shift_anchor_x(component: LayoutComponent, x: float, y: float, orientation: str, delta: float, span_x: float) -> float:
    for candidate in (x + delta, x - delta):
        bbox = _component_bbox(component.type, candidate, y, orientation)
        if bbox.x >= CANVAS_EDGE_PADDING and bbox.right <= span_x - CANVAS_EDGE_PADDING:
            return candidate
    return x + delta


def _legalized_hint_orientation(component: LayoutComponent, orientation: str | None) -> str:
    value = _key(orientation)
    current = component.orientation
    if _is_opamp_layout(component):
        return orientation if value in {"right", "right_flip"} and orientation is not None else current if current in {"right", "right_flip"} else "right"
    if _is_input_or_source_layout(component) or _is_output_layout(component) or _is_filter_block_layout(component):
        return "right"
    allowed = {"right", "left", "up", "down", "east", "west", "north", "south"}
    return orientation if value in allowed and orientation is not None else current


def _median_component_y(components: list[LayoutComponent]) -> float:
    values = sorted(component.y for component in components)
    return values[len(values) // 2] if values else 5.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _build_layout(
    circuit: Circuit,
    placements: dict[str, tuple[float, float, str]],
    warnings: list[str],
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    grid: int = GRID,
    label_offsets: dict[str, tuple[float, float]] | None = None,
    support: LayoutSupport | None = None,
) -> LayoutPlan:
    fallback_index = 0
    fallback_component_ids: list[str] = []
    components: list[LayoutComponent] = []
    for component in circuit.components:
        if component.id not in placements:
            placements[component.id] = (2.0 + fallback_index * 2.8, 15.0, "right")
            fallback_component_ids.append(component.id)
            fallback_index += 1
        x, y, orientation = placements[component.id]
        components.append(_layout_component(component, x, y, orientation))

    pin_map = _pin_map(components)
    net_to_pins: dict[str, list[tuple[str, str]]] = {}
    for key, pin in pin_map.items():
        net_to_pins.setdefault(pin.net, []).append(key)
    net_to_pins = {net: sorted(pins) for net, pins in sorted(net_to_pins.items())}

    labels = _labels_for_components(components, label_offsets or {})
    layout_support = _support_with_fallback_components(
        support or _support_from_warnings(warnings),
        fallback_component_ids,
    )
    semantic = _build_semantic_plan(circuit, components, pin_map, net_to_pins)
    wires = _route_wires(
        pin_map,
        net_to_pins,
        motif=_route_motif(circuit, layout_support),
        semantic=semantic,
        components=components,
        labels=labels,
        route_policies=_route_policy_hints(layout_support),
        canvas_bottom=height / grid,
    )
    layout = LayoutPlan(
        circuit_id=circuit.id,
        width=width,
        height=height,
        grid=grid,
        components=components,
        wires=wires,
        labels=labels,
        pin_map=pin_map,
        net_to_pins=net_to_pins,
        topology_signature=topology_signature(circuit),
        warnings=warnings,
        support=layout_support,
        semantic=semantic,
    )
    assert_no_diagonal_wires(layout)
    return layout


def _build_semantic_plan(
    circuit: Circuit,
    components: list[LayoutComponent],
    pin_map: dict[tuple[str, str], LayoutPin],
    net_to_pins: dict[str, list[tuple[str, str]]],
) -> TopologySemanticPlan:
    component_by_id = {component.id: component for component in components}
    net_classes = {net: classify_net(net) for net in net_to_pins}
    local_terminals = _local_terminal_intents(component_by_id, pin_map, net_classes)
    return TopologySemanticPlan(
        net_classes=net_classes,
        local_terminals=tuple(local_terminals),
        stages=tuple(_semantic_stages(components)),
        lanes=tuple(_semantic_lanes(components)),
        motifs=tuple(_semantic_motifs(circuit, components, net_to_pins, net_classes)),
        routes=tuple(_semantic_routes(pin_map, net_to_pins, net_classes, components)),
    )


def _local_terminal_intents(
    component_by_id: dict[str, LayoutComponent],
    pin_map: dict[tuple[str, str], LayoutPin],
    net_classes: dict[str, NetClass],
) -> list[LocalTerminalIntent]:
    terminals: list[LocalTerminalIntent] = []
    for (component_id, pin_name), pin in sorted(pin_map.items()):
        net_class = net_classes.get(pin.net, classify_net(pin.net))
        if net_class not in {NetClass.GROUND, NetClass.POSITIVE_SUPPLY, NetClass.NEGATIVE_SUPPLY, NetClass.REFERENCE}:
            continue
        component = component_by_id.get(component_id)
        if component is not None and _is_explicit_terminal_component(component):
            continue
        terminals.append(
            LocalTerminalIntent(
                component_id=component_id,
                pin_name=pin_name,
                net=pin.net,
                terminal_type=terminal_type_for_net(pin.net),
                label=terminal_label(pin.net, net_class),
                preferred_direction=preferred_terminal_direction(pin.net),
            )
        )
    return terminals


def _semantic_stages(components: list[LayoutComponent]) -> list[Stage]:
    buckets: dict[tuple[int, str], list[str]] = {}
    for component in components:
        stage_type = _semantic_stage_type(component)
        x_key = round(component.x * 2)
        buckets.setdefault((x_key, stage_type), []).append(component.id)
    ordered = sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1]))
    return [
        Stage(
            stage_id=f"stage:{index}:{stage_type}",
            stage_type=stage_type,
            component_ids=tuple(sorted(component_ids)),
            x_order=index,
        )
        for index, ((_, stage_type), component_ids) in enumerate(ordered)
    ]


def _semantic_lanes(components: list[LayoutComponent]) -> list[Lane]:
    signal_components = [
        component
        for component in components
        if not _is_explicit_terminal_component(component) and _semantic_stage_type(component) != "local_ground"
    ]
    y_values = sorted({round(component.y * 2) / 2.0 for component in signal_components})
    lanes: list[Lane] = []
    for index, y_value in enumerate(y_values):
        members = tuple(
            component.id
            for component in sorted(signal_components, key=lambda item: item.x)
            if abs(component.y - y_value) <= 0.45
        )
        if members:
            lanes.append(Lane(lane_id=f"lane:{index}", source=members[0], y_order=index, component_ids=members))
    return lanes


def _semantic_motifs(
    circuit: Circuit,
    components: list[LayoutComponent],
    net_to_pins: dict[str, list[tuple[str, str]]],
    net_classes: dict[str, NetClass],
) -> list[Motif]:
    motifs: list[Motif] = []
    component_by_id = {component.id: component for component in components}
    opamps = [component for component in components if _is_opamp_layout(component)]
    resistors = [component for component in components if _is_resistor_layout(component)]

    for component in components:
        stage_type = _semantic_stage_type(component)
        if stage_type == "input_port":
            motifs.append(
                Motif(
                    motif_id=f"input:{component.id}",
                    motif_type="input_port",
                    component_ids=(component.id,),
                    output_nets=tuple(component.pins.values()),
                )
            )
        elif stage_type == "output_port":
            motifs.append(
                Motif(
                    motif_id=f"output:{component.id}",
                    motif_type="output_port",
                    component_ids=(component.id,),
                    input_nets=tuple(component.pins.values()),
                )
            )
        elif stage_type == "filter_block":
            motifs.append(
                Motif(
                    motif_id=f"filter:{component.id}",
                    motif_type="functional_filter_block",
                    component_ids=(component.id,),
                    input_nets=tuple(net for pin, net in component.pins.items() if _pin_kind(pin) in {"in", "input", "a"}),
                    output_nets=tuple(net for pin, net in component.pins.items() if _is_opamp_output_pin(pin) or _pin_kind(pin) in {"out", "output", "b"}),
                )
            )

    for component in components:
        local_nets = tuple(net for net in component.pins.values() if net_classes.get(net) in {NetClass.GROUND, NetClass.REFERENCE})
        supply_nets = tuple(
            net
            for net in component.pins.values()
            if net_classes.get(net) in {NetClass.POSITIVE_SUPPLY, NetClass.NEGATIVE_SUPPLY}
        )
        if local_nets and not _is_explicit_terminal_component(component):
            motifs.append(
                Motif(
                    motif_id=f"local-reference:{component.id}",
                    motif_type="local_ground",
                    component_ids=(component.id,),
                    local_reference_nets=local_nets,
                )
            )
        if supply_nets:
            motifs.append(
                Motif(
                    motif_id=f"local-supply:{component.id}",
                    motif_type="local_supply_terminal",
                    component_ids=(component.id,),
                    local_reference_nets=supply_nets,
                )
            )

    for opamp in opamps:
        output_net = _opamp_output_net_layout(opamp)
        input_nets = _opamp_input_nets_layout(opamp)
        feedback = _feedback_resistors_for_opamp(opamp, resistors)
        feedback_nets = tuple(sorted({net for resistor in feedback for net in resistor.pins.values()}))
        motif_type = "op_amp_feedback_stage" if feedback else "generic_functional_block"
        if output_net and output_net in input_nets:
            motif_type = "opamp_buffer"
        elif _is_summing_opamp(opamp, resistors):
            motif_type = "summing_opamp"
        motifs.append(
            Motif(
                motif_id=f"opamp:{opamp.id}",
                motif_type=motif_type,
                component_ids=tuple([opamp.id, *(resistor.id for resistor in feedback)]),
                input_nets=tuple(input_nets),
                output_nets=(output_net,) if output_net else (),
                local_reference_nets=tuple(
                    net for net in opamp.pins.values() if net_classes.get(net) in {NetClass.GROUND, NetClass.REFERENCE}
                ),
                feedback_nets=feedback_nets,
            )
        )

    for resistor in resistors:
        classes = [net_classes.get(net, classify_net(net)) for net in resistor.pins.values()]
        if any(net_class in {NetClass.GROUND, NetClass.REFERENCE} for net_class in classes):
            motifs.append(
                Motif(
                    motif_id=f"ground-leg:{resistor.id}",
                    motif_type="resistor_to_ground_reference_leg",
                    component_ids=(resistor.id,),
                    input_nets=tuple(net for net in resistor.pins.values() if not is_local_terminal_net(net)),
                    local_reference_nets=tuple(net for net in resistor.pins.values() if is_local_terminal_net(net)),
                )
            )

    motifs.extend(_passive_section_motifs(components, net_to_pins, net_classes))
    return motifs


def _passive_section_motifs(
    components: list[LayoutComponent],
    net_to_pins: dict[str, list[tuple[str, str]]],
    net_classes: dict[str, NetClass],
) -> list[Motif]:
    motifs: list[Motif] = []
    component_by_id = {component.id: component for component in components}
    for net, pins in net_to_pins.items():
        if net_classes.get(net) != NetClass.SIGNAL:
            continue
        attached = [component_by_id[component_id] for component_id, _ in pins if component_id in component_by_id]
        resistors = [component for component in attached if _is_resistor_layout(component)]
        capacitors = [component for component in attached if _is_capacitor_layout(component)]
        if resistors and capacitors:
            motifs.append(
                Motif(
                    motif_id=f"rc-section:{net}",
                    motif_type="rc_low_pass_or_high_pass_section",
                    component_ids=tuple(sorted({component.id for component in [*resistors, *capacitors]})),
                    input_nets=(net,),
                )
            )
    return motifs


def _semantic_routes(
    pin_map: dict[tuple[str, str], LayoutPin],
    net_to_pins: dict[str, list[tuple[str, str]]],
    net_classes: dict[str, NetClass],
    components: list[LayoutComponent],
) -> list[RouteIntent]:
    routes: list[RouteIntent] = []
    component_by_id = {component.id: component for component in components}
    for net, pins in net_to_pins.items():
        if net_classes.get(net) != NetClass.SIGNAL or len(pins) < 2:
            continue
        drivers = [key for key in pins if _is_driver_pin(key, component_by_id)]
        targets = [key for key in pins if key not in drivers]
        if not drivers:
            drivers = [min(pins, key=lambda key: pin_map[key].x)]
            targets = [key for key in pins if key not in drivers]
        for source in drivers:
            for target in targets:
                routes.append(
                    RouteIntent(
                        route_type="left_to_right_signal",
                        source=source,
                        target=target,
                        net=net,
                        preferred_side="right",
                        avoid_component_ids=tuple(
                            sorted(
                                component.id
                                for component in components
                                if component.id not in {source[0], target[0]} and _is_opamp_layout(component)
                            )
                        ),
                    )
                )
    return routes


def _semantic_stage_type(component: LayoutComponent) -> str:
    key = _key(component.type)
    if _is_input_or_source_layout(component):
        return "input_port"
    if _is_output_layout(component):
        return "output_port"
    if _is_filter_block_layout(component):
        return "filter_block"
    if _is_opamp_layout(component):
        return "op_amp_stage"
    if _is_ground_layout(component):
        return "local_ground"
    if _is_resistor_layout(component) or _is_capacitor_layout(component):
        return "passive"
    return "generic_functional_block"


def _is_driver_pin(key: tuple[str, str], component_by_id: dict[str, LayoutComponent]) -> bool:
    component_id, pin_name = key
    component = component_by_id.get(component_id)
    if component is None:
        return False
    return _is_output_layout(component) is False and (
        _is_opamp_output_pin(pin_name)
        or (_is_input_or_source_layout(component) and _pin_kind(pin_name) in {"out", "output", "o"})
        or (_is_filter_block_layout(component) and _pin_kind(pin_name) in {"out", "output", "b"})
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
        return BBox(x, y - OPAMP_HALF_HEIGHT, OPAMP_OUTPUT_LEAD_X, OPAMP_HALF_HEIGHT * 2.0)
    if _is_filter_block_type(key):
        return BBox(x - 1.6, y - 0.8, 3.2, 1.6)
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
        if _is_output_layout(component):
            return (0.95, -0.62)
        return (-0.72, -0.62)
        return (0.0, -0.72)
    if key in {"ground", "gnd"}:
        return (0.0, 1.05)
    if _is_opamp_layout(component):
        return (1.55, -1.7)
    if component.orientation in {"up", "down"}:
        return (1.0, 0.0)
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
    semantic: TopologySemanticPlan | None = None,
    components: list[LayoutComponent] | None = None,
    labels: list[LayoutLabel] | None = None,
    route_policies: tuple[RoutePolicyHint, ...] = (),
    canvas_bottom: float | None = None,
) -> list[LayoutWire]:
    motif_routes = _route_known_motif(
        pin_map,
        net_to_pins,
        motif,
        semantic=semantic,
        components=components,
        labels=labels,
        route_policies=route_policies,
        canvas_bottom=canvas_bottom,
    )
    if motif_routes is not None:
        return motif_routes

    wires: list[LayoutWire] = []
    for net, connected in sorted(net_to_pins.items()):
        if _net_is_terminalized(net, semantic):
            continue
        policy = _policy_for_net_with_context(
            net,
            route_policies,
            semantic=semantic,
            pins=[pin_map[key] for key in connected],
            components=components or [],
        )
        if policy == "bottom_auxiliary_corridor":
            route = _bottom_auxiliary_corridor_route(
                [pin_map[key] for key in connected],
                components or [],
                net=net,
                canvas_bottom=canvas_bottom,
            )
        elif policy in {"top_feedback_corridor", "bottom_feedback_corridor"}:
            route = _feedback_corridor_route([pin_map[key] for key in connected], components or [], policy=policy)
        else:
            route = _generic_route([Point(pin_map[key].x, pin_map[key].y) for key in connected])
        if not route:
            continue
        wires.append(
            LayoutWire(
                net=net,
                points=_finalize_route(
                    net,
                    connected,
                    route,
                    pin_map,
                    semantic=semantic,
                    components=components,
                    labels=labels,
                    preferred_policy=policy,
                ),
                connected_pins=connected,
            )
        )
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


def _route_policy_hints(support: LayoutSupport) -> tuple[RoutePolicyHint, ...]:
    raw = (support.planning_hints or {}).get("route_policies", [])
    policies: list[RoutePolicyHint] = []
    if not isinstance(raw, list):
        return ()
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            policies.append(RoutePolicyHint.from_dict(item))
        except ValueError:
            continue
    return tuple(policies)


def _policy_for_net(net: str, policies: tuple[RoutePolicyHint, ...]) -> str | None:
    for policy in policies:
        if policy.net == net:
            return policy.policy
    net_key = _key(net)
    for policy in policies:
        role_key = _key(policy.net_role)
        if policy.net is not None:
            continue
        if policy.policy == "bottom_auxiliary_corridor" and (
            any(token in role_key for token in ("right_leg", "rld", "aux", "common_mode", "cmfb"))
            or any(token in net_key for token in ("right_leg", "rld", "aux", "cmfb"))
        ):
            return policy.policy
    if any(token in net_key for token in ("rld", "right_leg", "driven_right_leg", "cmfb")):
        return "bottom_auxiliary_corridor"
    return None


def _policy_for_net_with_context(
    net: str,
    policies: tuple[RoutePolicyHint, ...],
    *,
    semantic: TopologySemanticPlan | None,
    pins: list[LayoutPin],
    components: list[LayoutComponent],
) -> str | None:
    policy = _policy_for_net(net, policies)
    if (
        policy == "bottom_auxiliary_corridor"
        and _has_opamp_input_pin(pins, components)
        and _has_explicit_feedback_passive(net, pins, components)
        and not any(_is_opamp_output_pin(pin.pin_name) for pin in pins)
    ):
        return _feedback_corridor_policy_for_pins(pins, components)
    if policy is not None:
        return policy

    for hint in policies:
        if hint.net is not None or hint.policy not in {"top_feedback_corridor", "bottom_feedback_corridor"}:
            continue
        if "feedback" in _key(hint.net_role) and _is_feedback_related_net(net, semantic):
            return hint.policy

    if (
        policies
        and _is_feedback_related_net(net, semantic)
        and _has_explicit_feedback_passive(net, pins, components)
        and not any(_is_opamp_output_pin(pin.pin_name) for pin in pins)
    ):
        return _feedback_corridor_policy_for_pins(pins, components)
    return None


def _has_explicit_feedback_passive(net: str, pins: list[LayoutPin], components: list[LayoutComponent]) -> bool:
    component_by_id = {component.id: component for component in components}
    return any(
        net in component.pins.values()
        and _is_resistor_layout(component)
        and "feedback" in _key(component.role)
        for component in (component_by_id.get(pin.component_id) for pin in pins)
        if component is not None
    )


def _has_opamp_input_pin(pins: list[LayoutPin], components: list[LayoutComponent]) -> bool:
    component_by_id = {component.id: component for component in components}
    return any(
        _is_opamp_input_pin(pin)
        and (component := component_by_id.get(pin.component_id)) is not None
        and _is_opamp_layout(component)
        for pin in pins
    )


def _feedback_corridor_policy_for_pins(pins: list[LayoutPin], components: list[LayoutComponent]) -> str:
    component_by_id = {component.id: component for component in components}
    opamp_receivers = [
        pin
        for pin in pins
        if _is_opamp_input_pin(pin) and _is_opamp_layout(component_by_id.get(pin.component_id, _null_component()))
    ]
    driver = next((pin for pin in pins if _is_opamp_output_pin(pin.pin_name)), None)
    if driver is not None and opamp_receivers:
        receiver = sorted(opamp_receivers, key=lambda pin: abs(pin.y - driver.y))[0]
        return "bottom_feedback_corridor" if receiver.y > driver.y else "top_feedback_corridor"
    return "top_feedback_corridor"


def _bottom_auxiliary_corridor_route(
    pins: list[LayoutPin],
    components: list[LayoutComponent],
    *,
    net: str | None = None,
    canvas_bottom: float | None = None,
) -> list[Point]:
    if len(pins) < 2:
        return []
    component_by_id = {component.id: component for component in components}
    if len(pins) == 2:
        return _two_pin_auxiliary_route(pins[0], pins[1], component_by_id)
    driver = _bottom_auxiliary_opamp_driver(pins, component_by_id)
    if driver is not None:
        local_branches = _local_feedback_branches_for_output_net(driver, pins, component_by_id)
        if local_branches:
            return _opamp_output_bottom_auxiliary_route(
                driver,
                [pin for pin in pins if pin is not driver and pin not in local_branches],
                local_branches,
                components,
                net=net,
                canvas_bottom=canvas_bottom,
            )
    ordered = sorted(
        pins,
        key=lambda pin: (
            0 if _is_opamp_output_pin(pin.pin_name) else 1,
            pin.x,
            pin.y,
            pin.component_id,
            pin.pin_name,
        ),
    )
    attached = [
        component_by_id[pin.component_id]
        for pin in pins
        if pin.component_id in component_by_id and not _is_ground_layout(component_by_id[pin.component_id])
    ]
    bottom = max([pin.y for pin in pins] + [component.bbox.bottom for component in attached])
    corridor_y = bottom + _bottom_auxiliary_corridor_offset(net)
    corridor_y = _avoid_bottom_corridor_blockers(corridor_y, pins, components)
    corridor_y = _fit_bottom_corridor_to_canvas(corridor_y, net, canvas_bottom)
    start = ordered[0]
    start_point = Point(start.x, start.y)
    if _is_opamp_output_pin(start.pin_name):
        trunk_x = start.x + 0.54
        route = [start_point, Point(trunk_x, start.y), Point(trunk_x, corridor_y)]
    else:
        route = [start_point, Point(start.x, corridor_y)]
    for pin in ordered[1:]:
        route.extend(_bottom_corridor_branch_route(pin, corridor_y, components))
    return route


def _two_pin_auxiliary_route(
    first: LayoutPin,
    second: LayoutPin,
    component_by_id: dict[str, LayoutComponent],
) -> list[Point]:
    first_component = component_by_id.get(first.component_id)
    second_component = component_by_id.get(second.component_id)
    if first_component is not None and _is_passive_layout(first_component):
        leaf, source = first, second
    elif second_component is not None and _is_passive_layout(second_component):
        leaf, source = second, first
    else:
        return _generic_route([Point(first.x, first.y), Point(second.x, second.y)])

    source_point = Point(source.x, source.y)
    leaf_point = Point(leaf.x, leaf.y)
    approach = _passive_leaf_approach(leaf, component_by_id.get(leaf.component_id))
    if abs(source.y - approach.y) < 0.15:
        return [source_point, approach, leaf_point]
    return [
        source_point,
        Point(approach.x, source.y),
        approach,
        leaf_point,
    ]


def _is_passive_layout(component: LayoutComponent) -> bool:
    return _is_resistor_layout(component) or _is_capacitor_layout(component)


def _passive_leaf_approach(pin: LayoutPin, component: LayoutComponent | None) -> Point:
    if component is None:
        return Point(pin.x, pin.y)
    if pin.side == "right":
        return Point(component.bbox.right + 0.42, pin.y)
    if pin.side == "left":
        return Point(component.bbox.x - 0.42, pin.y)
    if pin.side == "bottom":
        return Point(pin.x, component.bbox.bottom + 0.42)
    if pin.side == "top":
        return Point(pin.x, component.bbox.y - 0.42)
    return Point(pin.x, pin.y)


def _bottom_auxiliary_opamp_driver(
    pins: list[LayoutPin],
    component_by_id: dict[str, LayoutComponent],
) -> LayoutPin | None:
    return next(
        (
            pin
            for pin in pins
            if _is_opamp_output_pin(pin.pin_name)
            and _is_opamp_layout(component_by_id.get(pin.component_id, _null_component()))
        ),
        None,
    )


def _local_feedback_branches_for_output_net(
    driver: LayoutPin,
    pins: list[LayoutPin],
    component_by_id: dict[str, LayoutComponent],
) -> list[LayoutPin]:
    driver_component = component_by_id.get(driver.component_id)
    if driver_component is None:
        return []
    window_left = driver_component.bbox.x - 1.4
    window_right = driver_component.bbox.right + 2.8
    window_top = driver_component.bbox.y - 2.2
    window_bottom = driver_component.bbox.bottom + 2.2
    branches: list[LayoutPin] = []
    for pin in pins:
        if pin is driver:
            continue
        component = component_by_id.get(pin.component_id)
        if component is None or not _is_resistor_layout(component):
            continue
        if "feedback" not in _key(component.role):
            continue
        if window_left <= pin.x <= window_right and window_top <= pin.y <= window_bottom:
            branches.append(pin)
    return branches


def _opamp_output_bottom_auxiliary_route(
    driver: LayoutPin,
    corridor_branches: list[LayoutPin],
    local_branches: list[LayoutPin],
    components: list[LayoutComponent],
    *,
    net: str | None,
    canvas_bottom: float | None,
) -> list[Point]:
    driver_point = Point(driver.x, driver.y)
    route = [driver_point]
    for branch in sorted(local_branches, key=lambda pin: (abs(pin.y - driver.y), pin.x, pin.component_id, pin.pin_name)):
        route.extend(_local_output_feedback_branch(driver, branch))
    if not corridor_branches:
        return route

    corridor_pins = [driver, *corridor_branches]
    component_by_id = {component.id: component for component in components}
    attached = [
        component_by_id[pin.component_id]
        for pin in corridor_pins
        if pin.component_id in component_by_id and not _is_ground_layout(component_by_id[pin.component_id])
    ]
    bottom = max([pin.y for pin in corridor_pins] + [component.bbox.bottom for component in attached])
    corridor_y = bottom + _bottom_auxiliary_corridor_offset(net)
    corridor_y = _avoid_bottom_corridor_blockers(corridor_y, corridor_pins, components)
    corridor_y = _fit_bottom_corridor_to_canvas(corridor_y, net, canvas_bottom)
    trunk_x = driver.x + 0.54
    route.extend([Point(trunk_x, driver.y), Point(trunk_x, corridor_y)])
    for pin in sorted(corridor_branches, key=lambda pin: (pin.x, pin.y, pin.component_id, pin.pin_name)):
        route.extend(_bottom_corridor_branch_route(pin, corridor_y, components))
    return route


def _local_output_feedback_branch(driver: LayoutPin, branch: LayoutPin) -> list[Point]:
    driver_point = Point(driver.x, driver.y)
    branch_point = Point(branch.x, branch.y)
    if abs(branch.y - driver.y) < 0.25:
        return [branch_point, driver_point]
    bus_x = max(driver.x + 0.54, branch.x + 0.56)
    return [
        Point(bus_x, driver.y),
        Point(bus_x, branch.y),
        branch_point,
        Point(bus_x, branch.y),
        Point(bus_x, driver.y),
        driver_point,
    ]


def _bottom_auxiliary_corridor_offset(net: str | None) -> float:
    key = _key(net)
    if any(token in key for token in ("fb", "feedback", "sense")):
        return 0.52
    if any(token in key for token in ("drive", "output", "out")):
        return 1.16
    return 0.84


def _fit_bottom_corridor_to_canvas(corridor_y: float, net: str | None, canvas_bottom: float | None) -> float:
    if canvas_bottom is None:
        return corridor_y
    limit = canvas_bottom - 0.58
    if corridor_y <= limit:
        return corridor_y
    return max(0.45, limit - _bottom_auxiliary_overflow_slot(net) * 0.34)


def _bottom_auxiliary_overflow_slot(net: str | None) -> int:
    key = _key(net)
    if any(token in key for token in ("drive", "output", "out")):
        return 0
    if any(token in key for token in ("fb", "feedback", "sense")):
        return 2
    return 1


def _avoid_bottom_corridor_blockers(
    corridor_y: float,
    pins: list[LayoutPin],
    components: list[LayoutComponent],
) -> float:
    if len(pins) < 2:
        return corridor_y
    attached_ids = {pin.component_id for pin in pins}
    left = min(pin.x for pin in pins)
    right = max(pin.x for pin in pins)
    adjusted = corridor_y
    for component in components:
        if component.id in attached_ids:
            continue
        if component.bbox.right < left or component.bbox.x > right:
            continue
        if component.bbox.y - 0.06 <= adjusted <= component.bbox.bottom + 0.06:
            adjusted = component.bbox.bottom + 0.32
    return adjusted


def _bottom_corridor_branch_route(
    pin: LayoutPin,
    corridor_y: float,
    components: list[LayoutComponent],
) -> list[Point]:
    pin_point = Point(pin.x, pin.y)
    blocker = _vertical_branch_blocker(pin, corridor_y, components)
    if blocker is None:
        tap = Point(pin.x, corridor_y)
        return [tap, pin_point, tap]
    via_x = _branch_dogleg_x(pin, blocker)
    corridor_tap = Point(via_x, corridor_y)
    side_tap = Point(via_x, pin.y)
    return [corridor_tap, side_tap, pin_point, side_tap, corridor_tap]


def _branch_dogleg_x(pin: LayoutPin, blocker: LayoutComponent) -> float:
    if pin.side == "right":
        return blocker.bbox.right + 0.56
    if pin.side == "left":
        return blocker.bbox.x - 0.56
    return blocker.bbox.x - 0.56 if pin.x <= blocker.bbox.center.x else blocker.bbox.right + 0.56


def _vertical_branch_blocker(
    pin: LayoutPin,
    corridor_y: float,
    components: list[LayoutComponent],
) -> LayoutComponent | None:
    top = min(pin.y, corridor_y)
    bottom = max(pin.y, corridor_y)
    for component in components:
        if component.id == pin.component_id or not _is_opamp_layout(component):
            continue
        if component.bbox.x - 0.04 <= pin.x <= component.bbox.right + 0.04 and _intervals_overlap(
            top,
            bottom,
            component.bbox.y + 0.08,
            component.bbox.bottom - 0.08,
        ):
            return component
    return None


def _feedback_corridor_route(
    pins: list[LayoutPin],
    components: list[LayoutComponent],
    *,
    policy: str,
) -> list[Point]:
    if len(pins) < 2:
        return []
    component_by_id = {component.id: component for component in components}
    driver = next(
        (
            pin
            for pin in pins
            if _is_opamp_output_pin(pin.pin_name)
            and _is_opamp_layout(component_by_id.get(pin.component_id, _null_component()))
        ),
        None,
    )
    receivers = [
        pin
        for pin in pins
        if pin is not driver
        and _is_opamp_input_pin(pin)
        and _is_opamp_layout(component_by_id.get(pin.component_id, _null_component()))
    ]
    branches = [pin for pin in pins if pin is not driver and pin not in receivers]
    if driver is None:
        if receivers:
            return _input_feedback_corridor_route(
                receivers[0],
                [pin for pin in pins if pin not in receivers],
                components,
                policy=policy,
            )
        return _generic_route([Point(pin.x, pin.y) for pin in pins])

    driver_point = Point(driver.x, driver.y)
    corridor_y = _feedback_corridor_y(pins, components, policy)
    min_x = min([pin.x for pin in pins] + [component_by_id[pin.component_id].bbox.x for pin in pins if pin.component_id in component_by_id])
    max_x = max([pin.x for pin in pins] + [component_by_id[pin.component_id].bbox.right for pin in pins if pin.component_id in component_by_id])
    right_x = max(driver.x + 0.64, max_x + 0.45)
    left_x = min(min_x - 0.45, min((_opamp_input_approach(pin).x for pin in receivers), default=min_x) - 0.2)
    route = [driver_point, Point(right_x, driver.y), Point(right_x, corridor_y)]

    for branch in sorted(branches, key=lambda pin: (pin.x, pin.y, pin.component_id, pin.pin_name)):
        branch_point = Point(branch.x, branch.y)
        if _is_output_terminal_pin(branch, component_by_id) and abs(branch.y - driver.y) < 0.3:
            route.extend([Point(branch.x, corridor_y), branch_point, Point(branch.x, corridor_y)])
            continue
        tap = Point(branch.x, corridor_y)
        route.extend([tap, branch_point, tap])

    for receiver in sorted(receivers, key=lambda pin: (pin.y, pin.x, pin.component_id, pin.pin_name)):
        receiver_point = Point(receiver.x, receiver.y)
        approach = _opamp_input_approach(receiver)
        entry_x = min(left_x, approach.x)
        route.extend(
            [
                Point(entry_x, corridor_y),
                Point(entry_x, receiver.y),
                approach,
                receiver_point,
                approach,
                Point(entry_x, receiver.y),
                Point(entry_x, corridor_y),
            ]
        )
    return route


def _input_feedback_corridor_route(
    receiver: LayoutPin,
    branches: list[LayoutPin],
    components: list[LayoutComponent],
    *,
    policy: str,
) -> list[Point]:
    receiver_point = Point(receiver.x, receiver.y)
    approach = _opamp_input_approach(receiver)
    if len(branches) == 1:
        return _single_feedback_input_branch_route(receiver, branches[0], components, policy=policy)
    corridor_y = _feedback_corridor_y([receiver, *branches], components, policy)
    route = [receiver_point, approach, Point(approach.x, corridor_y)]
    for branch in sorted(branches, key=lambda pin: (pin.x, pin.y, pin.component_id, pin.pin_name)):
        branch_point = Point(branch.x, branch.y)
        tap = Point(branch.x, corridor_y)
        route.extend([tap, branch_point, tap])
    return route


def _single_feedback_input_branch_route(
    receiver: LayoutPin,
    branch: LayoutPin,
    components: list[LayoutComponent],
    *,
    policy: str,
) -> list[Point]:
    receiver_point = Point(receiver.x, receiver.y)
    branch_point = Point(branch.x, branch.y)
    approach = _opamp_input_approach(receiver)
    owner = next((component for component in components if component.id == receiver.component_id), None)
    if owner is not None and _axis_segment_intersects_bbox(approach, Point(branch.x, approach.y), owner.bbox.expanded(-0.05)):
        outer_y = _feedback_corridor_y([receiver, branch], components, policy)
        return [
            receiver_point,
            approach,
            Point(approach.x, outer_y),
            Point(branch.x, outer_y),
            branch_point,
        ]
    if abs(branch.y - approach.y) < 0.15:
        return [receiver_point, approach, branch_point]
    return [
        receiver_point,
        approach,
        Point(approach.x, branch.y),
        branch_point,
    ]


def _feedback_corridor_y(pins: list[LayoutPin], components: list[LayoutComponent], policy: str) -> float:
    component_by_id = {component.id: component for component in components}
    attached = [component_by_id[pin.component_id] for pin in pins if pin.component_id in component_by_id]
    if policy == "bottom_feedback_corridor":
        bottom = max([pin.y for pin in pins] + [component.bbox.bottom for component in attached])
        return bottom + 0.68
    top = min([pin.y for pin in pins] + [component.bbox.y for component in attached])
    return max(0.45, top - 0.68)


def _is_opamp_input_pin(pin: LayoutPin) -> bool:
    return _pin_kind(pin.pin_name) in {"+", "-", "plus", "minus", "non_inverting", "inverting"}


def _is_output_terminal_pin(pin: LayoutPin, component_by_id: dict[str, LayoutComponent]) -> bool:
    component = component_by_id.get(pin.component_id)
    return component is not None and _is_output_layout(component)


def _null_component() -> LayoutComponent:
    return LayoutComponent("", "", None, None, None, 0.0, 0.0, "right", {}, BBox(0.0, 0.0, 0.0, 0.0))


def _finalize_route(
    net: str,
    connected: list[tuple[str, str]],
    route: list[Point],
    pin_map: dict[tuple[str, str], LayoutPin],
    *,
    semantic: TopologySemanticPlan | None,
    components: list[LayoutComponent] | None,
    labels: list[LayoutLabel] | None,
    preferred_policy: str | None = None,
) -> list[Point]:
    route = _dedupe(route)
    return _dedupe(
        _orthogonalize_route(
            route,
            net=net,
            connected=connected,
            pin_map=pin_map,
            semantic=semantic,
            components=components or [],
            labels=labels or [],
            preferred_policy=preferred_policy,
        )
    )


def _orthogonalize_route(
    points: list[Point],
    *,
    net: str,
    connected: list[tuple[str, str]],
    pin_map: dict[tuple[str, str], LayoutPin],
    semantic: TopologySemanticPlan | None,
    components: list[LayoutComponent],
    labels: list[LayoutLabel],
    preferred_policy: str | None = None,
) -> list[Point]:
    if len(points) < 2:
        return points
    component_keepouts = [component.bbox.expanded(-0.08) for component in components]
    label_keepouts = [label.bbox.expanded(0.03) for label in labels]
    feedback_keepouts = [
        _opamp_route_body_bbox(component)
        for component in components
        if _is_feedback_related_net(net, semantic) and _is_opamp_layout(component)
    ]
    endpoint_kinds = _route_endpoint_kinds(pin_map, components)
    points = orthogonalize_route(points, [*component_keepouts, *feedback_keepouts], preferred_policy=preferred_policy)

    fixed = [points[0]]
    for end in points[1:]:
        start = fixed[-1]
        if start == end:
            continue
        if is_axis_aligned(start, end):
            fixed.append(end)
            continue

        elbow_a = Point(end.x, start.y)
        elbow_b = Point(start.x, end.y)
        elbow = min(
            [elbow_a, elbow_b],
            key=lambda candidate: _elbow_score(
                start,
                end,
                candidate,
                component_keepouts=component_keepouts,
                label_keepouts=label_keepouts,
                feedback_keepouts=feedback_keepouts,
                endpoint_kinds=endpoint_kinds,
            ),
        )
        fixed.extend([elbow, end])
    return fixed


def _elbow_score(
    start: Point,
    end: Point,
    elbow: Point,
    *,
    component_keepouts: list[BBox],
    label_keepouts: list[BBox],
    feedback_keepouts: list[BBox],
    endpoint_kinds: dict[tuple[float, float], str],
) -> float:
    segments = [(start, elbow), (elbow, end)]
    score = sum(abs(a.x - b.x) + abs(a.y - b.y) for a, b in segments) * 0.01
    for bbox in component_keepouts:
        if bbox.width > 0 and bbox.height > 0 and any(_axis_segment_intersects_bbox(a, b, bbox) for a, b in segments):
            score += 100.0
    for bbox in label_keepouts:
        if any(_axis_segment_intersects_bbox(a, b, bbox) for a, b in segments):
            score += 70.0
    for bbox in feedback_keepouts:
        if any(_axis_segment_intersects_bbox(a, b, bbox) for a, b in segments):
            score += 140.0

    if endpoint_kinds.get(_point_key(end)) == "opamp_input" and not _same(elbow.y, end.y):
        score += 25.0
    if endpoint_kinds.get(_point_key(start)) == "opamp_input" and not _same(start.y, elbow.y):
        score += 25.0
    if _same(elbow.x, end.x) and _same(elbow.y, start.y):
        score -= 0.02
    return score


def _route_endpoint_kinds(
    pin_map: dict[tuple[str, str], LayoutPin],
    components: list[LayoutComponent],
) -> dict[tuple[float, float], str]:
    component_by_id = {component.id: component for component in components}
    endpoints: dict[tuple[float, float], str] = {}
    for (component_id, pin_name), pin in pin_map.items():
        component = component_by_id.get(component_id)
        if component is None:
            continue
        if _is_opamp_layout(component) and _pin_kind(pin_name) in {"+", "-"}:
            endpoints[_point_key(Point(pin.x, pin.y))] = "opamp_input"
    return endpoints


def _is_feedback_related_net(net: str, semantic: TopologySemanticPlan | None) -> bool:
    if semantic is None:
        return False
    return any(net in motif.feedback_nets for motif in semantic.motifs)


def _opamp_route_body_bbox(component: LayoutComponent) -> BBox:
    return BBox(
        component.bbox.x + 0.16,
        component.bbox.y + 0.14,
        max(0.1, component.bbox.width - 0.36),
        max(0.1, component.bbox.height - 0.28),
    )


def _axis_segment_intersects_bbox(start: Point, end: Point, bbox: BBox) -> bool:
    if _same(start.x, end.x):
        return bbox.x <= start.x <= bbox.right and _intervals_overlap(start.y, end.y, bbox.y, bbox.bottom)
    if _same(start.y, end.y):
        return bbox.y <= start.y <= bbox.bottom and _intervals_overlap(start.x, end.x, bbox.x, bbox.right)
    elbow = Point(end.x, start.y)
    return _axis_segment_intersects_bbox(start, elbow, bbox) or _axis_segment_intersects_bbox(elbow, end, bbox)


def _intervals_overlap(a: float, b: float, c: float, d: float) -> bool:
    return min(max(a, b), max(c, d)) >= max(min(a, b), min(c, d))


def _point_key(point: Point) -> tuple[float, float]:
    return (round(point.x, 4), round(point.y, 4))


def _route_known_motif(
    pin_map: dict[tuple[str, str], LayoutPin],
    net_to_pins: dict[str, list[tuple[str, str]]],
    motif: str | None,
    *,
    semantic: TopologySemanticPlan | None = None,
    components: list[LayoutComponent] | None = None,
    labels: list[LayoutLabel] | None = None,
    route_policies: tuple[RoutePolicyHint, ...] = (),
    canvas_bottom: float | None = None,
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
        if _net_is_terminalized(net, semantic):
            continue
        policy = _policy_for_net_with_context(
            net,
            route_policies,
            semantic=semantic,
            pins=[pin_map[key] for key in connected],
            components=components or [],
        )
        if policy == "bottom_auxiliary_corridor":
            route = _bottom_auxiliary_corridor_route(
                [pin_map[key] for key in connected],
                components or [],
                net=net,
                canvas_bottom=canvas_bottom,
            )
        elif policy in {"top_feedback_corridor", "bottom_feedback_corridor"}:
            route = _feedback_corridor_route([pin_map[key] for key in connected], components or [], policy=policy)
        else:
            route = routes.get(net)
        if route is None:
            route = _generic_route([_point(pin_map, key) for key in connected])
        if route:
            wires.append(
                LayoutWire(
                    net=net,
                    points=_finalize_route(
                        net,
                        connected,
                        route,
                        pin_map,
                        semantic=semantic,
                        components=components,
                        labels=labels,
                        preferred_policy=policy,
                    ),
                    connected_pins=connected,
                )
            )
    return wires


def _net_is_terminalized(net: str, semantic: TopologySemanticPlan | None) -> bool:
    if semantic is not None:
        net_class = semantic.net_classes.get(net)
        if net_class is not None:
            return net_class in {NetClass.GROUND, NetClass.POSITIVE_SUPPLY, NetClass.NEGATIVE_SUPPLY, NetClass.REFERENCE}
    return is_local_terminal_net(net)


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
    vm_left = Point(_p(pin_map, "Rf", "b").x, minus.y)
    rg_drop = Point(vm_left.x, _p(pin_map, "Rg", "a").y)
    return {
        "vin": [_p(pin_map, "VIN", "out"), _p(pin_map, "U1", "+")],
        "vout": [out, _p(pin_map, "VOUT", "in"), out, Point(out.x, _p(pin_map, "Rf", "a").y), _p(pin_map, "Rf", "a")],
        "vm": [_p(pin_map, "Rf", "b"), vm_left, minus, vm_left, rg_drop, _p(pin_map, "Rg", "a")],
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
        "n3": [
            _p(pin_map, "R3", "b"),
            _p(pin_map, "U3", "+"),
            _p(pin_map, "R3", "b"),
            _p(pin_map, "R4", "a"),
        ],
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
    max_pin_x = max(pin.x for pin in pin_map.values())
    min_pin_y = min(pin.y for pin in pin_map.values())
    max_pin_y = max(pin.y for pin in pin_map.values())
    ground_gutter = max_pin_x + 1.05
    signal_gutter = max_pin_x + 1.85
    top_gutter = max(0.75, min_pin_y - 0.6)
    bottom_gutter = max_pin_y + 0.6
    for net, connected in net_to_pins.items():
        pins = [pin_map[key] for key in connected]
        if _net_has_ground_pin(pins):
            routes[net] = _op_amp_ground_route(pins, right_gutter=ground_gutter)
            continue
        driver = next((pin for pin in pins if _is_opamp_output_pin(pin.pin_name) and pin.side == "right"), None)
        receivers = [
            pin
            for pin in pins
            if pin is not driver and _pin_kind(pin.pin_name) in {"+", "-", "plus", "minus", "non_inverting", "inverting"}
        ]
        if len(receivers) == 1 and driver is None and len(pins) >= 3:
            routes[net] = _opamp_passive_input_route(
                receivers[0],
                [pin for pin in pins if pin not in receivers],
            )
            continue
        if receivers and driver is None and len(pins) >= 4:
            routes[net] = _passive_input_bus_route(
                receivers,
                [pin for pin in pins if pin not in receivers],
            )
            continue
        if receivers and driver is None and len(pins) == 2:
            routes[net] = _single_input_route(
                receivers[0],
                next(pin for pin in pins if pin not in receivers),
            )
            continue
        if driver and receivers:
            routes[net] = _op_amp_output_route(
                driver,
                receivers,
                [pin for pin in pins if pin is not driver and pin not in receivers],
                right_gutter=signal_gutter,
                top_gutter=top_gutter,
                bottom_gutter=bottom_gutter,
            )
    return routes


def _single_input_route(receiver: LayoutPin, source: LayoutPin) -> list[Point]:
    source_point = Point(source.x, source.y)
    receiver_point = Point(receiver.x, receiver.y)
    approach = _opamp_input_approach(receiver)
    if abs(source.y - approach.y) < 1e-6:
        return [source_point, approach, receiver_point]
    mid_x = _single_input_mid_x(source, receiver, approach)
    return [
        source_point,
        Point(mid_x, source.y),
        Point(mid_x, approach.y),
        approach,
        receiver_point,
    ]


def _single_input_mid_x(source: LayoutPin, receiver: LayoutPin, approach: Point) -> float:
    left = min(source.x, approach.x)
    right = max(source.x, approach.x)
    span = max(0.1, right - left)
    ratio = 0.4 if receiver.y > source.y else 0.62
    if _pin_kind(receiver.pin_name) in {"-", "minus", "inverting"}:
        ratio = 0.62 if receiver.y < source.y else 0.4
    mid_x = left + span * ratio
    return max(left + 0.42, min(right - 0.42, mid_x))


def _op_amp_ground_route(pins: list[LayoutPin], *, right_gutter: float) -> list[Point]:
    simple_bus = _single_row_ground_bus_route(pins)
    if simple_bus is not None:
        return simple_bus

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


def _single_row_ground_bus_route(pins: list[LayoutPin]) -> list[Point] | None:
    ground_symbols = [pin for pin in pins if _key(pin.pin_name) == "gnd"]
    taps = [pin for pin in pins if pin not in ground_symbols]
    if not ground_symbols or len(taps) < 2:
        return None
    tap_y_values = [pin.y for pin in taps]
    if max(tap_y_values) - min(tap_y_values) > 0.35:
        return None

    bus_y = sum(tap_y_values) / len(tap_y_values)
    ordered_taps = sorted(taps, key=lambda pin: pin.x)
    ground = sorted(ground_symbols, key=lambda pin: abs(pin.x - ordered_taps[len(ordered_taps) // 2].x))[0]
    route = [Point(ordered_taps[0].x, bus_y)]
    route.extend(Point(pin.x, bus_y) for pin in ordered_taps[1:])
    route.extend([Point(ground.x, bus_y), Point(ground.x, ground.y)])
    return route


def _passive_input_bus_route(receivers: list[LayoutPin], branches: list[LayoutPin]) -> list[Point]:
    receiver = sorted(receivers, key=lambda pin: pin.x)[-1]
    receiver_point = Point(receiver.x, receiver.y)
    bus_x = receiver.x - 0.72
    bus = Point(bus_x, receiver.y)
    route = [receiver_point, bus]
    for branch in sorted(branches, key=lambda pin: (pin.y, pin.x)):
        branch_point = Point(branch.x, branch.y)
        tap = Point(bus_x, branch.y)
        route.extend([tap, branch_point, tap, bus])
    for extra in [pin for pin in receivers if pin is not receiver]:
        extra_point = Point(extra.x, extra.y)
        tap = Point(bus_x, extra.y)
        route.extend([tap, extra_point, tap, bus])
    return route


def _opamp_passive_input_route(receiver: LayoutPin, branches: list[LayoutPin]) -> list[Point]:
    receiver_point = Point(receiver.x, receiver.y)
    bus_x = _opamp_passive_input_bus_x(receiver, branches)
    hub = Point(bus_x, receiver.y)
    route = [hub, receiver_point, hub]
    for branch in sorted(branches, key=lambda pin: (pin.y, pin.x)):
        branch_point = Point(branch.x, branch.y)
        if branch.y < receiver.y - 0.2:
            lane_y = branch.y - 0.42
            lane_bus = Point(bus_x, lane_y)
            lane_branch = Point(branch.x, lane_y)
            route.extend([lane_bus, lane_branch, branch_point, lane_branch, lane_bus, hub])
        else:
            tap = Point(bus_x, branch.y)
            route.extend([tap, branch_point, tap, hub])
    return route


def _opamp_passive_input_bus_x(receiver: LayoutPin, branches: list[LayoutPin]) -> float:
    return receiver.x - 0.62


def _op_amp_output_route(
    driver: LayoutPin,
    receivers: list[LayoutPin],
    branches: list[LayoutPin],
    *,
    right_gutter: float,
    top_gutter: float,
    bottom_gutter: float,
) -> list[Point]:
    driver_point = Point(driver.x, driver.y)
    route = [driver_point]
    for branch in sorted(branches, key=lambda pin: (abs(pin.y - driver.y), pin.x)):
        elbow = Point(driver.x, branch.y)
        branch_point = Point(branch.x, branch.y)
        route.extend([elbow, branch_point, elbow, driver_point])

    for receiver in sorted(receivers, key=lambda pin: (pin.y, pin.x)):
        receiver_point = Point(receiver.x, receiver.y)
        if receiver.component_id == driver.component_id:
            route.extend(_local_feedback_detour(driver, receiver, top_gutter=top_gutter, bottom_gutter=bottom_gutter))
        elif receiver.x > driver.x + 0.45:
            route.extend(_forward_opamp_input_detour(driver, receiver, branches))
        elif receiver.x < driver.x - 0.5 or abs(receiver.y - driver.y) > 2.0:
            if receiver.y > driver.y + 2.0:
                bend_y = bottom_gutter
            elif receiver.y < driver.y - 2.0:
                bend_y = top_gutter
            else:
                bend_y = (driver.y + receiver.y) / 2.0 + (0.55 if receiver.y > driver.y else -0.55)
            right_x = right_gutter
            left_x = min(driver.x, receiver.x) - 1.4
            approach = _opamp_input_approach(receiver)
            route.extend(
                [
                    Point(right_x, driver.y),
                    Point(right_x, bend_y),
                    Point(left_x, bend_y),
                    Point(left_x, receiver.y),
                    approach,
                    receiver_point,
                    approach,
                    Point(left_x, receiver.y),
                    Point(left_x, bend_y),
                    Point(right_x, bend_y),
                    Point(right_x, driver.y),
                    driver_point,
                ]
            )
        else:
            approach = _opamp_input_approach(receiver)
            route.extend(
                [
                    Point(driver.x, approach.y),
                    approach,
                    receiver_point,
                    approach,
                    Point(driver.x, approach.y),
                    driver_point,
                ]
            )
    return route


def _forward_opamp_input_detour(driver: LayoutPin, receiver: LayoutPin, branches: list[LayoutPin]) -> list[Point]:
    driver_point = Point(driver.x, driver.y)
    receiver_point = Point(receiver.x, receiver.y)
    approach = _opamp_input_approach(receiver)
    available = max(0.85, approach.x - driver.x)
    trunk_x = min(approach.x - 0.58, driver.x + min(3.2, available * 0.48))
    same_lane_branches = [
        branch.x
        for branch in branches
        if branch.x > driver.x + 0.25 and abs(branch.y - driver.y) < 0.35
    ]
    if same_lane_branches:
        trunk_x = min(trunk_x, min(same_lane_branches) - 0.34)
    if trunk_x <= driver.x + 0.2:
        trunk_x = min(approach.x - 0.58, driver.x + 0.22)
    return [
        Point(trunk_x, driver.y),
        Point(trunk_x, approach.y),
        approach,
        receiver_point,
        approach,
        Point(trunk_x, approach.y),
        Point(trunk_x, driver.y),
        driver_point,
    ]


def _local_feedback_detour(driver: LayoutPin, receiver: LayoutPin, *, top_gutter: float, bottom_gutter: float) -> list[Point]:
    driver_point = Point(driver.x, driver.y)
    receiver_point = Point(receiver.x, receiver.y)
    if receiver.y <= driver.y:
        bend_y = max(0.55, driver.y - OPAMP_HALF_HEIGHT - 0.18)
    else:
        bend_y = driver.y + OPAMP_HALF_HEIGHT + 0.18
    right_x = driver.x + 0.58
    left_x = receiver.x - 0.58
    return [
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


def _opamp_input_approach(receiver: LayoutPin) -> Point:
    return Point(receiver.x - 0.62, receiver.y)


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
        net = component.pins.get(pin_name)
        flip = _is_flipped_opamp_orientation(component.orientation)
        if kind in {"-", "minus", "inverting"}:
            return (x + OPAMP_INPUT_LEAD_X, y + OPAMP_INPUT_LEAD_Y if flip else y - OPAMP_INPUT_LEAD_Y, "left")
        if kind in {"+", "plus", "non_inverting"}:
            return (x + OPAMP_INPUT_LEAD_X, y - OPAMP_INPUT_LEAD_Y if flip else y + OPAMP_INPUT_LEAD_Y, "left")
        if is_positive_supply_pin(pin_name, net):
            return (x + OPAMP_OUTPUT_LEAD_X * 0.45, y - OPAMP_HALF_HEIGHT, "top")
        if is_negative_supply_pin(pin_name, net) or is_reference_pin(pin_name, net):
            return (x + OPAMP_OUTPUT_LEAD_X * 0.45, y + OPAMP_HALF_HEIGHT, "bottom")
        return (x + OPAMP_OUTPUT_LEAD_X, y, "right")
    if _is_filter_block_layout(component):
        if kind in {"out", "output", "o", "b", "right"}:
            return (component.bbox.right, y, "right")
        if kind in {"gnd", "ground", "ref"} or is_local_terminal_net(component.pins.get(pin_name)):
            return (x, component.bbox.bottom, "bottom")
        return (component.bbox.x, y, "left")
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


def _is_filter_block_component(component: Component) -> bool:
    return _is_filter_block_type(_key(component.type)) or _is_filter_block_type(_key(component.label)) or _is_filter_block_type(_key(component.value))


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


def _is_resistor_layout(component: LayoutComponent) -> bool:
    key = _key(component.type)
    return "resistor" in key or key.startswith("r")


def _is_capacitor_layout(component: LayoutComponent) -> bool:
    key = _key(component.type)
    return "capacitor" in key or key.startswith("c")


def _is_ground_layout(component: LayoutComponent) -> bool:
    key = _key(component.type)
    return key in {"ground", "gnd"} or "ground" in _key(component.role)


def _is_input_or_source_layout(component: LayoutComponent) -> bool:
    key = _key(component.type)
    role = _key(component.role)
    return key in {"input", "input_terminal", "voltage_source", "source"} or "input" in role or "source" in key


def _is_output_layout(component: LayoutComponent) -> bool:
    return _key(component.type) == "output" or "output" in _key(component.role)


def _is_terminal_layout(component: LayoutComponent) -> bool:
    key = _key(component.type)
    return key in {"input", "output", "input_terminal", "voltage_source", "source"} or "source" in key


def _is_explicit_terminal_component(component: LayoutComponent) -> bool:
    return _is_terminal_layout(component) or _is_ground_layout(component) or _key(component.type) in {
        "supply",
        "power",
        "vcc",
        "vdd",
        "vee",
        "vss",
    }


def _is_filter_block_layout(component: LayoutComponent) -> bool:
    return _is_filter_block_type(_key(component.type)) or _is_filter_block_type(_key(component.label)) or _is_filter_block_type(_key(component.value))


def _is_filter_block_type(key: str) -> bool:
    return any(token in key for token in ["filter", "lpf", "hpf", "bpf", "bessel", "butterworth", "chebyshev"])


def _opamp_output_net_layout(opamp: LayoutComponent) -> str | None:
    for pin_name, net in opamp.pins.items():
        if _is_opamp_output_pin(pin_name):
            return net
    return None


def _opamp_input_nets_layout(opamp: LayoutComponent) -> list[str]:
    nets: list[str] = []
    for pin_name, net in opamp.pins.items():
        kind = _pin_kind(pin_name)
        if kind in {"+", "-", "plus", "minus", "non_inverting", "inverting"}:
            nets.append(net)
    return nets


def _feedback_resistors_for_opamp(opamp: LayoutComponent, resistors: list[LayoutComponent]) -> list[LayoutComponent]:
    output_net = _opamp_output_net_layout(opamp)
    input_nets = set(_opamp_input_nets_layout(opamp))
    if output_net is None:
        return []
    if output_net in input_nets:
        return [
            resistor
            for resistor in resistors
            if output_net in set(resistor.pins.values()) and "feedback" in _key(resistor.role)
        ]
    return [
        resistor
        for resistor in resistors
        if output_net in set(resistor.pins.values())
        and (input_nets & set(resistor.pins.values()) or "feedback" in _key(resistor.role))
    ]


def _is_summing_opamp(opamp: LayoutComponent, resistors: list[LayoutComponent]) -> bool:
    input_nets = set(_opamp_input_nets_layout(opamp))
    for input_net in input_nets:
        if is_local_terminal_net(input_net):
            continue
        incoming = [resistor for resistor in resistors if input_net in set(resistor.pins.values())]
        if len(incoming) >= 3:
            return True
    return False


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


def _same(left: float, right: float) -> bool:
    return abs(left - right) < 1e-6
