"""Deterministic schematic grammar inference.

This module extracts reusable schematic intent from topology: roles, local
motiflets, ordering constraints, and route intents.  It is not a template
renderer; the output is a soft semantic layer consumed by the existing planner.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .intent import FunctionalBlock, LayoutConstraint, RouteIntentRule, SchematicIntent
from .models import Circuit, Component, NetClass
from .planning_hints import (
    BlockHint,
    GridPlacementHint,
    LaneHint,
    RoutePolicyHint,
    SchematicLayoutHints,
    StageHint,
)
from .semantics import (
    classify_net,
    is_local_terminal_net,
    is_negative_supply_pin,
    is_positive_supply_pin,
    is_reference_pin,
)


def infer_schematic_intent(circuit: Circuit) -> SchematicIntent:
    """Infer circuit semantics that general schematic layout can solve."""

    net_to_pins = _net_to_pins(circuit.components)
    component_roles: dict[str, set[str]] = {}
    pin_roles: dict[str, set[str]] = {}
    net_roles: dict[str, set[str]] = {}
    blocks: list[FunctionalBlock] = []
    constraints: list[LayoutConstraint] = []
    route_intents: list[RouteIntentRule] = []

    for component in circuit.components:
        _classify_component(component, component_roles)
        for pin_name, net in component.pins.items():
            _classify_pin(component, pin_name, net, pin_roles)
            _classify_net_role(net, net_to_pins.get(net, []), net_roles)

    feedback_paths = _detect_opamp_feedback_paths(circuit)
    for feedback in feedback_paths:
        _mark_feedback_path(feedback, component_roles, pin_roles, net_roles, blocks, constraints, route_intents)

    _mark_summing_inputs(circuit, feedback_paths, component_roles, pin_roles)
    _mark_passive_roles(circuit, feedback_paths, component_roles)
    _mark_signal_constraints(circuit, feedback_paths, constraints)
    _mark_high_fanout_nets(net_to_pins, net_roles, route_intents)
    _mark_local_terminal_constraints(net_to_pins, net_roles, constraints, route_intents)
    blocks.extend(_source_and_output_blocks(circuit))
    blocks.extend(_passive_filter_blocks(circuit, net_to_pins))

    topology = _recognized_topology(circuit, component_roles, blocks)
    confidence = _intent_confidence(circuit, component_roles, blocks, constraints)
    return SchematicIntent(
        recognized_topology=topology,
        component_roles=_freeze_roles(component_roles),
        pin_roles=_freeze_roles(pin_roles),
        net_roles=_freeze_roles(net_roles),
        blocks=tuple(_dedupe_blocks(blocks)),
        constraints=tuple(_dedupe_constraints(constraints)),
        route_intents=tuple(_dedupe_routes(route_intents)),
        unsupported_reasons=(),
        confidence=confidence,
        source="deterministic",
    )


def planning_hints_from_intent(circuit: Circuit, intent: SchematicIntent) -> SchematicLayoutHints | None:
    """Translate grammar intent into conservative stage/lane hints."""

    if not circuit.components:
        return None
    ranks = _component_ranks(circuit, intent)
    lanes = _component_lanes(circuit, intent, ranks)
    placements = []
    for component in circuit.components:
        roles = intent.component_roles.get(component.id, ())
        placements.append(
            GridPlacementHint(
                component_id=component.id,
                stage_x=ranks.get(component.id, 1),
                lane_y=lanes.get(component.id, 0),
                orientation=_orientation_for_component(component, roles, intent),
                role=",".join(roles) if roles else None,
                confidence=max(0.45, intent.confidence),
            )
        )

    route_policies = []
    for route in intent.route_intents:
        try:
            route_policies.append(RoutePolicyHint(net=route.net, net_role=route.net_role, policy=route.policy))
        except ValueError:
            continue

    local_terminal_policy = {
        net: "local_symbol_only"
        for net, roles in intent.net_roles.items()
        if "local_reference" in roles or "supply" in roles or "ground" in roles
    }
    blocks = tuple(_block_hint_from_intent(block, ranks, lanes) for block in intent.blocks if block.members)
    stage_hints = tuple(
        StageHint(stage_x=stage, stage_type=_stage_type_for_rank(stage, ranks, intent), members=tuple(_rank_members(stage, ranks)))
        for stage in sorted(set(ranks.values()))
    )
    lane_hints = tuple(
        LaneHint(lane_y=lane, lane_type=_lane_type_for_rank(lane, lanes, intent), members=tuple(_lane_members(lane, lanes)))
        for lane in sorted(set(lanes.values()))
    )
    return SchematicLayoutHints(
        recognized_topology=intent.recognized_topology,
        confidence=intent.confidence,
        tutor_explanation="Deterministic schematic grammar inferred roles, blocks, constraints, and route intents.",
        source="deterministic",
        stages=stage_hints,
        lanes=lane_hints,
        placements=tuple(placements),
        route_policies=tuple(route_policies),
        local_terminal_policy=local_terminal_policy,
        blocks=blocks,
        intent=intent,
    )


@dataclass(frozen=True)
class FeedbackPath:
    opamp_id: str
    input_pin: str
    input_net: str
    output_net: str
    member_ids: tuple[str, ...]
    path_nets: tuple[str, ...]


def _classify_component(component: Component, roles: dict[str, set[str]]) -> None:
    if _is_input_or_source(component):
        _add_role(roles, component.id, "input_port")
        _add_role(roles, component.id, "source_port")
    if _is_output(component):
        _add_role(roles, component.id, "output_port")
        _add_role(roles, component.id, "load")
    if _is_ground(component):
        _add_role(roles, component.id, "local_reference_symbol")
    if _is_opamp(component):
        _add_role(roles, component.id, "op_amp")
        if "comparator" in _identity(component):
            _add_role(roles, component.id, "op_amp_comparator")
    if _is_transistor(component):
        _add_role(roles, component.id, "active_device")
        _add_role(roles, component.id, "transistor_gain_stage")
    if _is_resistor(component):
        _add_role(roles, component.id, "passive")
        _add_role(roles, component.id, "resistor")
    if _is_capacitor(component):
        _add_role(roles, component.id, "passive")
        _add_role(roles, component.id, "capacitor")
    if _is_filter_block(component):
        _add_role(roles, component.id, "filter_block")


def _classify_pin(component: Component, pin_name: str, net: str, roles: dict[str, set[str]]) -> None:
    key = _pin_role_key(component.id, pin_name)
    pin_kind = _pin_kind(pin_name)
    if _is_input_or_source(component) and pin_kind in {"out", "output", "o"}:
        _add_role(roles, key, "source_output")
    elif _is_output(component):
        _add_role(roles, key, "load_input")
    elif _is_opamp(component):
        if _is_opamp_output_pin(pin_name):
            _add_role(roles, key, "stage_output")
        elif is_positive_supply_pin(pin_name, net) or is_negative_supply_pin(pin_name, net):
            _add_role(roles, key, "supply_pin")
        elif is_reference_pin(pin_name, net):
            _add_role(roles, key, "reference_input")
        else:
            _add_role(roles, key, "signal_input")
    elif is_local_terminal_net(net):
        _add_role(roles, key, "reference_side")
    else:
        _add_role(roles, key, "signal_side")


def _classify_net_role(net: str, pins: list[tuple[Component, str]], roles: dict[str, set[str]]) -> None:
    net_class = classify_net(net)
    if net_class == NetClass.GROUND:
        _add_role(roles, net, "ground")
        _add_role(roles, net, "local_reference")
    elif net_class == NetClass.REFERENCE:
        _add_role(roles, net, "reference")
        _add_role(roles, net, "local_reference")
    elif net_class in {NetClass.POSITIVE_SUPPLY, NetClass.NEGATIVE_SUPPLY}:
        _add_role(roles, net, "supply")
        _add_role(roles, net, "local_reference")
    else:
        _add_role(roles, net, "signal")
    if any(_is_input_or_source(component) for component, _ in pins):
        _add_role(roles, net, "input_signal")
        _add_role(roles, net, "forward_signal")
    if any(_is_output(component) for component, _ in pins):
        _add_role(roles, net, "output_signal")
    if any(_is_opamp(component) and _is_opamp_output_pin(pin_name) for component, pin_name in pins):
        _add_role(roles, net, "stage_output")
        _add_role(roles, net, "forward_signal")


def _detect_opamp_feedback_paths(circuit: Circuit) -> list[FeedbackPath]:
    passives = [component for component in circuit.components if _is_passive(component)]
    paths: list[FeedbackPath] = []
    for opamp in [component for component in circuit.components if _is_opamp(component)]:
        output_net = _opamp_output_net(opamp)
        if output_net is None:
            continue
        for input_pin, input_net in _opamp_input_pins(opamp):
            if input_net == output_net:
                paths.append(
                    FeedbackPath(
                        opamp_id=opamp.id,
                        input_pin=input_pin,
                        input_net=input_net,
                        output_net=output_net,
                        member_ids=(),
                        path_nets=(output_net,),
                    )
                )
                continue
            passive_path = _passive_path_between(output_net, input_net, passives)
            if passive_path is None:
                continue
            member_ids, path_nets = passive_path
            paths.append(
                FeedbackPath(
                    opamp_id=opamp.id,
                    input_pin=input_pin,
                    input_net=input_net,
                    output_net=output_net,
                    member_ids=member_ids,
                    path_nets=path_nets,
                )
            )
    return _dedupe_feedback_paths(paths)


def _passive_path_between(
    start_net: str,
    end_net: str,
    passives: list[Component],
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    graph: dict[str, list[tuple[str, str]]] = {}
    for component in passives:
        nets = list(dict.fromkeys(component.pins.values()))
        if len(nets) < 2:
            continue
        if all(is_local_terminal_net(net) for net in nets):
            continue
        for left in nets:
            for right in nets:
                if left != right:
                    graph.setdefault(left, []).append((right, component.id))

    queue: deque[tuple[str, tuple[str, ...], tuple[str, ...]]] = deque([(start_net, (), (start_net,))])
    seen = {start_net}
    while queue:
        net, member_ids, path_nets = queue.popleft()
        if len(member_ids) > 4:
            continue
        for next_net, component_id in graph.get(net, []):
            if is_local_terminal_net(next_net):
                continue
            next_members = (*member_ids, component_id)
            next_path_nets = (*path_nets, next_net)
            if next_net == end_net:
                return (tuple(dict.fromkeys(next_members)), tuple(dict.fromkeys(next_path_nets)))
            if next_net in seen:
                continue
            seen.add(next_net)
            queue.append((next_net, next_members, next_path_nets))
    return None


def _mark_feedback_path(
    feedback: FeedbackPath,
    component_roles: dict[str, set[str]],
    pin_roles: dict[str, set[str]],
    net_roles: dict[str, set[str]],
    blocks: list[FunctionalBlock],
    constraints: list[LayoutConstraint],
    route_intents: list[RouteIntentRule],
) -> None:
    _add_role(component_roles, feedback.opamp_id, "op_amp_feedback_stage")
    if not feedback.member_ids:
        _add_role(component_roles, feedback.opamp_id, "op_amp_buffer")
    for component_id in feedback.member_ids:
        _add_role(component_roles, component_id, "feedback_element")
    _add_role(pin_roles, _pin_role_key(feedback.opamp_id, feedback.input_pin), "feedback_input")
    _add_role(net_roles, feedback.output_net, "feedback_signal")
    _add_role(net_roles, feedback.input_net, "feedback_node")
    for net in feedback.path_nets:
        _add_role(net_roles, net, "feedback_signal")

    side = _feedback_side_for_pin(feedback.input_pin)
    policy = "bottom_feedback_corridor" if side == "bottom" else "top_feedback_corridor"
    route_intents.append(RouteIntentRule(net=feedback.output_net, net_role="feedback_signal", policy=policy))
    if feedback.input_net != feedback.output_net:
        route_intents.append(RouteIntentRule(net=feedback.input_net, net_role="feedback_node", policy=policy))
    blocks.append(
        FunctionalBlock(
            block_id=f"feedback:{feedback.opamp_id}",
            block_type="single_opamp_feedback_stage",
            members=(feedback.opamp_id, *feedback.member_ids),
            input_nets=(feedback.input_net,),
            output_nets=(feedback.output_net,),
            feedback_nets=feedback.path_nets,
        )
    )
    constraints.append(
        LayoutConstraint(
            constraint_type="feedback_outside_body",
            subject=feedback.opamp_id,
            net=feedback.output_net,
            strength="hard",
            preferred_side=side,
            reason="op-amp output-to-input feedback should route outside the active body",
        )
    )


def _mark_summing_inputs(
    circuit: Circuit,
    feedback_paths: list[FeedbackPath],
    component_roles: dict[str, set[str]],
    pin_roles: dict[str, set[str]],
) -> None:
    feedback_members = {member_id for feedback in feedback_paths for member_id in feedback.member_ids}
    for opamp in [component for component in circuit.components if _is_opamp(component)]:
        for pin_name, net in _opamp_input_pins(opamp):
            attached_passives = [
                component
                for component in circuit.components
                if component.id not in feedback_members and _is_passive(component) and net in component.pins.values()
            ]
            signal_inputs = [
                component
                for component in attached_passives
                if any(other_net != net and not is_local_terminal_net(other_net) for other_net in component.pins.values())
            ]
            if len(signal_inputs) >= 2:
                _add_role(component_roles, opamp.id, "op_amp_summing_stage")
                _add_role(pin_roles, _pin_role_key(opamp.id, pin_name), "summing_input")
                for component in signal_inputs:
                    _add_role(component_roles, component.id, "series_input_element")


def _mark_passive_roles(
    circuit: Circuit,
    feedback_paths: list[FeedbackPath],
    component_roles: dict[str, set[str]],
) -> None:
    feedback_members = {member_id for feedback in feedback_paths for member_id in feedback.member_ids}
    input_nets = {net for component in circuit.components if _is_input_or_source(component) for net in component.pins.values()}
    opamp_input_nets = {net for component in circuit.components if _is_opamp(component) for _, net in _opamp_input_pins(component)}
    for component in circuit.components:
        if not _is_passive(component):
            continue
        nets = set(component.pins.values())
        if component.id in feedback_members or "feedback" in _identity(component):
            _add_role(component_roles, component.id, "feedback_element")
        if any(is_local_terminal_net(net) for net in nets):
            _add_role(component_roles, component.id, "shunt_reference_element")
        if _is_capacitor(component) and any(is_local_terminal_net(net) for net in nets):
            _add_role(component_roles, component.id, "shunt_filter_element")
        if len(nets - {net for net in nets if is_local_terminal_net(net)}) >= 2:
            _add_role(component_roles, component.id, "series_element")
        if nets & input_nets and nets & opamp_input_nets:
            _add_role(component_roles, component.id, "series_input_element")
        if "bias" in _identity(component):
            _add_role(component_roles, component.id, "bias_element")
        if _is_capacitor(component) and not any(is_local_terminal_net(net) for net in nets):
            _add_role(component_roles, component.id, "coupling_element")


def _mark_signal_constraints(
    circuit: Circuit,
    feedback_paths: list[FeedbackPath],
    constraints: list[LayoutConstraint],
) -> None:
    feedback_self_edges = {
        (feedback.opamp_id, feedback.input_pin, feedback.output_net)
        for feedback in feedback_paths
        if feedback.output_net == feedback.input_net
    }
    net_to_pins = _net_to_pins(circuit.components)
    for net, pins in sorted(net_to_pins.items()):
        if is_local_terminal_net(net):
            continue
        drivers = [(component, pin_name) for component, pin_name in pins if _is_driver_pin(component, pin_name)]
        if not drivers:
            drivers = []
        for driver, _ in drivers:
            _add_left_of_constraints_from_driver(driver, pins, net, feedback_self_edges, constraints)
        _add_series_pin_order_constraints(pins, net, constraints)


def _add_left_of_constraints_from_driver(
    driver: Component,
    pins: list[tuple[Component, str]],
    net: str,
    feedback_self_edges: set[tuple[str, str, str]],
    constraints: list[LayoutConstraint],
) -> None:
    for target, target_pin in pins:
        if target.id == driver.id or _is_driver_pin(target, target_pin):
            continue
        if (target.id, target_pin, net) in feedback_self_edges:
            continue
        constraints.append(
            LayoutConstraint(
                constraint_type="left_of",
                subject=driver.id,
                object=target.id,
                net=net,
                strength="soft",
                reason="forward signal flow normally increases x-rank",
            )
        )


def _add_series_pin_order_constraints(
    pins: list[tuple[Component, str]],
    net: str,
    constraints: list[LayoutConstraint],
) -> None:
    series_pins = [(component, pin_name) for component, pin_name in pins if _is_series_path_component(component)]
    for source, source_pin in series_pins:
        if _series_pin_direction(source, source_pin) != "output":
            continue
        for target, target_pin in series_pins:
            if target.id == source.id or _series_pin_direction(target, target_pin) != "input":
                continue
            constraints.append(
                LayoutConstraint(
                    constraint_type="left_of",
                    subject=source.id,
                    object=target.id,
                    net=net,
                    strength="soft",
                    reason="two-terminal series chain follows input-to-output pin order",
                )
            )


def _series_pin_direction(component: Component, pin_name: str) -> str:
    ordered = list(component.pins)
    kind = _pin_kind(pin_name)
    if kind in {"out", "output", "o", "b", "right", "to"}:
        return "output"
    if kind in {"in", "input", "i", "a", "left", "from"}:
        return "input"
    if len(ordered) >= 2 and pin_name == ordered[-1]:
        return "output"
    return "input"


def _component_drives_net(component: Component, net: str) -> bool:
    return any(pin_net == net and _is_driver_pin(component, pin_name) for pin_name, pin_net in component.pins.items())


def _series_component_has_input_net(component: Component, net: str) -> bool:
    return any(
        pin_net == net and _series_pin_direction(component, pin_name) == "input"
        for pin_name, pin_net in component.pins.items()
    )


def _series_component_input_nets(component: Component) -> list[str]:
    return [
        net
        for pin_name, net in component.pins.items()
        if not is_local_terminal_net(net) and _series_pin_direction(component, pin_name) == "input"
    ]


def _series_component_output_nets(component: Component) -> list[str]:
    return [
        net
        for pin_name, net in component.pins.items()
        if not is_local_terminal_net(net) and _series_pin_direction(component, pin_name) == "output"
    ]


def _mark_high_fanout_nets(
    net_to_pins: dict[str, list[tuple[Component, str]]],
    roles: dict[str, set[str]],
    route_intents: list[RouteIntentRule],
) -> None:
    for net, pins in net_to_pins.items():
        if len(pins) >= 4 and not is_local_terminal_net(net):
            _add_role(roles, net, "high_fanout_signal")
            route_intents.append(RouteIntentRule(net=net, net_role="high_fanout_signal", policy="named_net_label"))


def _mark_local_terminal_constraints(
    net_to_pins: dict[str, list[tuple[Component, str]]],
    net_roles: dict[str, set[str]],
    constraints: list[LayoutConstraint],
    route_intents: list[RouteIntentRule],
) -> None:
    for net in sorted(net_to_pins):
        if not is_local_terminal_net(net):
            continue
        _add_role(net_roles, net, "local_reference")
        constraints.append(
            LayoutConstraint(
                constraint_type="local_terminal_only",
                net=net,
                strength="hard",
                reason="reference and supply nets should render as local terminals by default",
            )
        )
        route_intents.append(RouteIntentRule(net=net, net_role="local_reference", policy="local_terminal_only"))


def _source_and_output_blocks(circuit: Circuit) -> list[FunctionalBlock]:
    blocks: list[FunctionalBlock] = []
    for component in circuit.components:
        if _is_input_or_source(component):
            blocks.append(
                FunctionalBlock(
                    block_id=f"source:{component.id}",
                    block_type="source_block",
                    members=(component.id,),
                    output_nets=tuple(component.pins.values()),
                )
            )
        elif _is_output(component):
            blocks.append(
                FunctionalBlock(
                    block_id=f"output:{component.id}",
                    block_type="load_output_block",
                    members=(component.id,),
                    input_nets=tuple(component.pins.values()),
                )
            )
    return blocks


def _passive_filter_blocks(circuit: Circuit, net_to_pins: dict[str, list[tuple[Component, str]]]) -> list[FunctionalBlock]:
    component_by_id = {component.id: component for component in circuit.components}
    blocks: list[FunctionalBlock] = []
    for net, pins in sorted(net_to_pins.items()):
        if is_local_terminal_net(net):
            continue
        attached = [component for component, _ in pins]
        capacitors = [component for component in attached if _is_capacitor(component)]
        if not capacitors:
            continue
        local_members = [
            component
            for component in capacitors
            if any(is_local_terminal_net(other_net) for other_net in component.pins.values())
        ]
        series_members = [
            component
            for component in attached
            if _is_resistor(component)
            and any(
                other_component.id != component.id and other_component.id in component_by_id
                for other_component, _ in pins
            )
        ]
        members = tuple(sorted({component.id for component in [*local_members, *series_members]}))
        if not members:
            continue
        reference_nets = tuple(
            sorted(
                {
                    pin_net
                    for component_id in members
                    for pin_net in component_by_id[component_id].pins.values()
                    if is_local_terminal_net(pin_net)
                }
            )
        )
        blocks.append(
            FunctionalBlock(
                block_id=f"filter:{net}",
                block_type="rc_filter_block",
                members=members,
                input_nets=(net,),
                reference_nets=reference_nets,
            )
        )
    return blocks


def _component_ranks(circuit: Circuit, intent: SchematicIntent) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for component in circuit.components:
        roles = set(intent.component_roles.get(component.id, ()))
        if "input_port" in roles or "source_port" in roles:
            ranks[component.id] = 0
        elif "local_reference_symbol" in roles:
            ranks[component.id] = 1

    dependencies = [
        (constraint.subject, constraint.object)
        for constraint in intent.constraints
        if constraint.constraint_type == "left_of" and constraint.subject and constraint.object
    ]
    for component in circuit.components:
        ranks.setdefault(component.id, 1)

    changed = True
    for _ in range(len(circuit.components) + 2):
        if not changed:
            break
        changed = False
        for left, right in dependencies:
            if left not in ranks or right not in ranks:
                continue
            if ranks[right] <= ranks[left]:
                ranks[right] = ranks[left] + 1
                changed = True

    ranks = _propagate_series_component_ranks(circuit, ranks)
    owner_by_passive = _passive_owner_by_role(circuit, intent)
    for component_id, owner_id in owner_by_passive.items():
        if owner_id in ranks:
            ranks[component_id] = ranks[owner_id]

    max_signal_rank = max(ranks.values(), default=1)
    for component in circuit.components:
        roles = set(intent.component_roles.get(component.id, ()))
        if "output_port" in roles:
            driver_rank = _driver_rank_for_output(component, circuit, ranks)
            ranks[component.id] = max(max_signal_rank + 1, driver_rank + 1)
    return _compact_ranks(ranks)


def _propagate_series_component_ranks(circuit: Circuit, ranks: dict[str, int]) -> dict[str, int]:
    result = dict(ranks)
    components_by_id = {component.id: component for component in circuit.components}
    net_to_components: dict[str, set[str]] = {}
    for component in circuit.components:
        for net in component.pins.values():
            if not is_local_terminal_net(net):
                net_to_components.setdefault(net, set()).add(component.id)

    net_ranks: dict[str, int] = {}
    for component in circuit.components:
        rank = result.get(component.id)
        if rank is None:
            continue
        for pin_name, net in component.pins.items():
            if is_local_terminal_net(net):
                continue
            if _is_driver_pin(component, pin_name) or _is_input_or_source(component):
                net_ranks[net] = max(net_ranks.get(net, -1), rank)

    for _ in range(len(circuit.components) + 2):
        changed = False
        for component in circuit.components:
            if not _is_series_path_component(component):
                continue
            nets = [net for net in dict.fromkeys(component.pins.values()) if not is_local_terminal_net(net)]
            if len(nets) < 2:
                continue
            input_nets = _series_component_input_nets(component)
            output_nets = _series_component_output_nets(component)
            known = [net_ranks[net] for net in input_nets if net in net_ranks]
            if known:
                proposed = max(known) + 1
                if proposed > result.get(component.id, 0):
                    result[component.id] = proposed
                    changed = True
            if component.id in result:
                for net in output_nets:
                    proposed_net = result[component.id]
                    if proposed_net > net_ranks.get(net, -1):
                        net_ranks[net] = proposed_net
                        changed = True

        for net, component_ids in net_to_components.items():
            if net not in net_ranks:
                continue
            for component_id in component_ids:
                component = components_by_id[component_id]
                if _is_input_or_source(component) or _is_ground(component):
                    continue
                if _is_opamp(component):
                    continue
                if _component_drives_net(component, net):
                    continue
                if _is_series_path_component(component):
                    if not _series_component_has_input_net(component, net):
                        continue
                    increment = 1
                elif _is_output(component):
                    increment = 1
                else:
                    increment = 0
                proposed = net_ranks[net] + increment
                if proposed > result.get(component_id, 0):
                    result[component_id] = proposed
                    changed = True
        if not changed:
            break
    return result


def _component_lanes(circuit: Circuit, intent: SchematicIntent, ranks: dict[str, int]) -> dict[str, int]:
    lanes: dict[str, int] = {}
    sources = [component for component in circuit.components if "input_port" in intent.component_roles.get(component.id, ())]
    if len(sources) <= 1:
        for source in sources:
            lanes[source.id] = 0
    else:
        start = -len(sources) + 1
        for index, source in enumerate(sorted(sources, key=lambda item: item.id)):
            lanes[source.id] = start + index * 2

    by_id = {component.id: component for component in circuit.components}
    output_to_owner = _output_net_to_component(circuit.components)
    for component in sorted(circuit.components, key=lambda item: (ranks.get(item.id, 0), item.id)):
        if component.id in lanes:
            continue
        if _is_opamp(component):
            input_lanes = []
            for _, net in _opamp_input_pins(component):
                drivers = [output_to_owner.get(net)]
                drivers.extend(source.id for source in sources if net in source.pins.values())
                input_lanes.extend(lanes[driver] for driver in drivers if driver in lanes)
            lanes[component.id] = _median_int(input_lanes) if input_lanes else 0
        elif _is_output(component):
            lanes[component.id] = _driver_lane_for_output(component, output_to_owner, lanes)
        elif _is_ground(component):
            lanes[component.id] = max([*lanes.values(), 0]) + 3

    lanes = _separate_helper_opamp_lanes(circuit, intent, lanes)
    feedback_by_member = {
        member_id: feedback
        for block in intent.blocks
        if "feedback" in block.block_type
        for feedback in _feedback_blocks_for_members(intent, block)
        for member_id in feedback.member_ids
    }
    for component in circuit.components:
        if component.id in lanes:
            continue
        owner_id = _passive_owner_by_role(circuit, intent).get(component.id)
        owner_lane = lanes.get(owner_id or "", 0)
        roles = set(intent.component_roles.get(component.id, ()))
        if component.id in feedback_by_member or "feedback_element" in roles:
            lanes[component.id] = owner_lane - 2 if owner_lane >= 0 else owner_lane + 2
        elif "shunt_reference_element" in roles:
            lanes[component.id] = owner_lane + 2
        elif "series_input_element" in roles or "series_element" in roles:
            lanes[component.id] = owner_lane
        else:
            lanes[component.id] = owner_lane

    for component in circuit.components:
        lanes.setdefault(component.id, 0)
    return _spread_duplicate_cells(ranks, lanes)


def _separate_helper_opamp_lanes(
    circuit: Circuit,
    intent: SchematicIntent,
    lanes: dict[str, int],
) -> dict[str, int]:
    result = dict(lanes)
    opamps = [component for component in circuit.components if _is_opamp(component)]
    opamp_by_id = {component.id: component for component in opamps}
    for driver in opamps:
        output_net = _opamp_output_net(driver)
        if output_net is None or not _is_helper_opamp(driver, intent):
            continue
        receivers = [
            receiver
            for receiver in opamps
            if receiver.id != driver.id and any(net == output_net for _, net in _opamp_input_pins(receiver))
        ]
        if not receivers:
            continue
        receiver = sorted(receivers, key=lambda component: (result.get(component.id, 0), component.id))[0]
        receiver_lane = result.get(receiver.id, 0)
        result[driver.id] = receiver_lane - 2
    return result


def _is_helper_opamp(opamp: Component, intent: SchematicIntent) -> bool:
    roles = set(intent.component_roles.get(opamp.id, ()))
    if roles & {"op_amp_buffer", "op_amp_summing_stage"}:
        return "op_amp_buffer" in roles
    output_net = _opamp_output_net(opamp)
    return output_net is not None and any(net == output_net for _, net in _opamp_input_pins(opamp))


def _passive_owner_by_role(circuit: Circuit, intent: SchematicIntent) -> dict[str, str]:
    owners: dict[str, str] = {}
    opamps = [component for component in circuit.components if _is_opamp(component)]
    for block in intent.blocks:
        if "feedback" in block.block_type:
            opamp_id = next((member for member in block.members if member in {opamp.id for opamp in opamps}), None)
            if opamp_id:
                for member in block.members:
                    if member != opamp_id:
                        owners[member] = opamp_id
    for component in circuit.components:
        if not _is_passive(component) or component.id in owners:
            continue
        roles = set(intent.component_roles.get(component.id, ()))
        if not (roles & {"feedback_element", "shunt_reference_element"} or any(is_local_terminal_net(net) for net in component.pins.values())):
            continue
        nets = set(component.pins.values())
        for opamp in opamps:
            if nets & {net for _, net in _opamp_input_pins(opamp)}:
                owners[component.id] = opamp.id
                break
    return owners


def _orientation_for_component(component: Component, roles: tuple[str, ...], intent: SchematicIntent) -> str:
    role_set = set(roles)
    if _is_opamp(component):
        if _feedback_pin_for_opamp(component.id, intent) in {"+", "plus", "non_inverting"}:
            return "RIGHT"
        return "RIGHT"
    if "shunt_reference_element" in role_set:
        return "DOWN"
    if "feedback_element" in role_set:
        return "LEFT"
    if _is_ground(component) or _is_input_or_source(component) or _is_output(component):
        return "RIGHT"
    return "RIGHT"


def _block_hint_from_intent(block: FunctionalBlock, ranks: dict[str, int], lanes: dict[str, int]) -> BlockHint:
    member_ranks = [ranks[member] for member in block.members if member in ranks]
    member_lanes = [lanes[member] for member in block.members if member in lanes]
    ports = {
        **{f"in{index}": net for index, net in enumerate(block.input_nets)},
        **{f"out{index}": net for index, net in enumerate(block.output_nets)},
        **{f"fb{index}": net for index, net in enumerate(block.feedback_nets)},
        **{f"ref{index}": net for index, net in enumerate(block.reference_nets)},
    }
    return BlockHint(
        block_id=block.block_id,
        block_type=block.block_type,
        members=block.members,
        stage_x=_median_int(member_ranks) if member_ranks else 0,
        lane_y=_median_int(member_lanes) if member_lanes else 0,
        ports=ports,
    )


def _stage_type_for_rank(stage: int, ranks: dict[str, int], intent: SchematicIntent) -> str:
    members = _rank_members(stage, ranks)
    member_roles = {role for member in members for role in intent.component_roles.get(member, ())}
    if "input_port" in member_roles:
        return "source_stage"
    if "output_port" in member_roles:
        return "output_stage"
    if "op_amp_feedback_stage" in member_roles:
        return "opamp_feedback_stage"
    if "filter_block" in member_roles or "shunt_filter_element" in member_roles:
        return "filter_stage"
    return "generic_stage"


def _lane_type_for_rank(lane: int, lanes: dict[str, int], intent: SchematicIntent) -> str:
    members = _lane_members(lane, lanes)
    member_roles = {role for member in members for role in intent.component_roles.get(member, ())}
    if "feedback_element" in member_roles:
        return "feedback_lane"
    if "shunt_reference_element" in member_roles or "local_reference_symbol" in member_roles:
        return "reference_lane"
    if "input_port" in member_roles:
        return "input_lane"
    return "signal_lane"


def _feedback_blocks_for_members(intent: SchematicIntent, block: FunctionalBlock) -> list[FeedbackPath]:
    opamp_id = next((member for member in block.members if "op_amp" in intent.component_roles.get(member, ())), "")
    return [
        FeedbackPath(
            opamp_id=opamp_id,
            input_pin="",
            input_net=block.input_nets[0] if block.input_nets else "",
            output_net=block.output_nets[0] if block.output_nets else "",
            member_ids=tuple(member for member in block.members if member != opamp_id),
            path_nets=block.feedback_nets,
        )
    ]


def _driver_rank_for_output(component: Component, circuit: Circuit, ranks: dict[str, int]) -> int:
    output_nets = set(component.pins.values())
    best = 0
    for candidate in circuit.components:
        if candidate.id == component.id:
            continue
        if output_nets & set(candidate.pins.values()):
            best = max(best, ranks.get(candidate.id, 0))
    return best


def _driver_lane_for_output(component: Component, output_to_owner: dict[str, str], lanes: dict[str, int]) -> int:
    for net in component.pins.values():
        owner = output_to_owner.get(net)
        if owner in lanes:
            return lanes[owner]
    return 0


def _output_net_to_component(components: list[Component]) -> dict[str, str]:
    result: dict[str, str] = {}
    for component in components:
        for pin_name, net in component.pins.items():
            if _is_driver_pin(component, pin_name):
                result.setdefault(net, component.id)
    return result


def _compact_ranks(ranks: dict[str, int]) -> dict[str, int]:
    ordered = {rank: index for index, rank in enumerate(sorted(set(ranks.values())))}
    return {component_id: ordered[rank] for component_id, rank in ranks.items()}


def _spread_duplicate_cells(ranks: dict[str, int], lanes: dict[str, int]) -> dict[str, int]:
    occupied: set[tuple[int, int]] = set()
    result: dict[str, int] = {}
    for component_id in sorted(lanes, key=lambda item: (ranks.get(item, 0), lanes[item], item)):
        lane = lanes[component_id]
        step = 1
        while (ranks.get(component_id, 0), lane) in occupied:
            lane += step
            step = -step if step > 0 else -step + 1
        occupied.add((ranks.get(component_id, 0), lane))
        result[component_id] = lane
    return result


def _recognized_topology(
    circuit: Circuit,
    component_roles: dict[str, set[str]],
    blocks: list[FunctionalBlock],
) -> str:
    opamp_count = sum(1 for component in circuit.components if _is_opamp(component))
    if opamp_count and any(block.block_type == "single_opamp_feedback_stage" for block in blocks):
        return "grammar_opamp_feedback_network" if opamp_count > 1 else "grammar_single_opamp_feedback_stage"
    if any("filter" in block.block_type for block in blocks):
        return "grammar_passive_filter_network"
    if opamp_count:
        return "grammar_opamp_network"
    if component_roles:
        return "grammar_generic_signal_network"
    return "unknown"


def _intent_confidence(
    circuit: Circuit,
    component_roles: dict[str, set[str]],
    blocks: list[FunctionalBlock],
    constraints: list[LayoutConstraint],
) -> float:
    if not circuit.components:
        return 0.0
    coverage = sum(1 for component in circuit.components if component_roles.get(component.id)) / len(circuit.components)
    block_bonus = 0.12 if blocks else 0.0
    constraint_bonus = 0.08 if constraints else 0.0
    return min(0.88, 0.42 + coverage * 0.28 + block_bonus + constraint_bonus)


def _dedupe_blocks(blocks: list[FunctionalBlock]) -> list[FunctionalBlock]:
    result: dict[str, FunctionalBlock] = {}
    for block in blocks:
        result.setdefault(block.block_id, block)
    return list(result.values())


def _dedupe_constraints(constraints: list[LayoutConstraint]) -> list[LayoutConstraint]:
    result: dict[tuple[str, str | None, str | None, str | None, tuple[str, ...]], LayoutConstraint] = {}
    for constraint in constraints:
        key = (constraint.constraint_type, constraint.subject, constraint.object, constraint.net, constraint.members)
        result.setdefault(key, constraint)
    return list(result.values())


def _dedupe_routes(routes: list[RouteIntentRule]) -> list[RouteIntentRule]:
    result: dict[tuple[str | None, str, str], RouteIntentRule] = {}
    for route in routes:
        result.setdefault((route.net, route.net_role, route.policy), route)
    return list(result.values())


def _dedupe_feedback_paths(paths: list[FeedbackPath]) -> list[FeedbackPath]:
    result: dict[tuple[str, str, str], FeedbackPath] = {}
    for path in paths:
        result.setdefault((path.opamp_id, path.input_pin, path.output_net), path)
    return list(result.values())


def _freeze_roles(raw: dict[str, set[str]]) -> dict[str, tuple[str, ...]]:
    return {key: tuple(sorted(values)) for key, values in sorted(raw.items())}


def _rank_members(stage: int, ranks: dict[str, int]) -> list[str]:
    return sorted(component_id for component_id, rank in ranks.items() if rank == stage)


def _lane_members(lane: int, lanes: dict[str, int]) -> list[str]:
    return sorted(component_id for component_id, value in lanes.items() if value == lane)


def _median_int(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _feedback_pin_for_opamp(opamp_id: str, intent: SchematicIntent) -> str | None:
    prefix = f"{opamp_id}."
    for key, roles in intent.pin_roles.items():
        if key.startswith(prefix) and "feedback_input" in roles:
            return key[len(prefix) :]
    return None


def _feedback_side_for_pin(pin_name: str) -> str:
    return "bottom" if _pin_kind(pin_name) in {"+", "plus", "non_inverting"} else "top"


def _net_to_pins(components: list[Component]) -> dict[str, list[tuple[Component, str]]]:
    result: dict[str, list[tuple[Component, str]]] = {}
    for component in components:
        for pin_name, net in component.pins.items():
            result.setdefault(net, []).append((component, pin_name))
    return result


def _add_role(roles: dict[str, set[str]], key: str, role: str) -> None:
    roles.setdefault(key, set()).add(role)


def _pin_role_key(component_id: str, pin_name: str) -> str:
    return f"{component_id}.{pin_name}"


def _opamp_output_net(opamp: Component) -> str | None:
    for pin_name, net in opamp.pins.items():
        if _is_opamp_output_pin(pin_name):
            return net
    return None


def _opamp_input_pins(opamp: Component) -> list[tuple[str, str]]:
    return [
        (pin_name, net)
        for pin_name, net in opamp.pins.items()
        if not _is_opamp_output_pin(pin_name)
        and not is_positive_supply_pin(pin_name, net)
        and not is_negative_supply_pin(pin_name, net)
        and not is_reference_pin(pin_name, net)
    ]


def _is_driver_pin(component: Component, pin_name: str) -> bool:
    kind = _pin_kind(pin_name)
    if _is_input_or_source(component) and kind in {"out", "output", "o"}:
        return True
    if _is_output(component):
        return False
    if _is_filter_block(component) and kind in {"out", "output", "o", "b"}:
        return True
    if _is_opamp(component) and _is_opamp_output_pin(pin_name):
        return True
    return kind in {"out", "output", "o", "collector", "drain"}


def _is_opamp_output_pin(pin_name: str) -> bool:
    return _pin_kind(pin_name) in {"out", "output", "o", "vout"}


def _is_input_or_source(component: Component) -> bool:
    key = _key(component.type)
    role = _key(component.role)
    return key in {"input", "input_terminal", "voltage_source", "source"} or "input" in role or "source" in key


def _is_output(component: Component) -> bool:
    return _key(component.type) == "output" or _role_is_output_port(component.role)


def _is_ground(component: Component) -> bool:
    key = _key(component.type)
    return key in {"ground", "gnd"} or "ground" in _key(component.role)


def _is_opamp(component: Component) -> bool:
    key = _key(component.type)
    return "op_amp" in key or "opamp" in key or "operational_amplifier" in key or "ideal_op_amp" in key


def _is_transistor(component: Component) -> bool:
    key = _key(component.type)
    return any(token in key for token in ("transistor", "bjt", "mosfet", "jfet", "fet"))


def _is_resistor(component: Component) -> bool:
    key = _key(component.type)
    return "resistor" in key or key.startswith("r")


def _is_capacitor(component: Component) -> bool:
    key = _key(component.type)
    return "capacitor" in key or key.startswith("c")


def _is_passive(component: Component) -> bool:
    return _is_resistor(component) or _is_capacitor(component)


def _is_series_path_component(component: Component) -> bool:
    if _is_input_or_source(component) or _is_output(component) or _is_ground(component) or _is_opamp(component):
        return False
    nets = [net for net in dict.fromkeys(component.pins.values()) if not is_local_terminal_net(net)]
    return len(nets) >= 2


def _is_filter_block(component: Component) -> bool:
    key = _key(" ".join(filter(None, [component.type, component.label, component.value])))
    return "filter" in key or "lpf" in key or "hpf" in key or "bpf" in key


def _pin_kind(pin_name: str) -> str:
    compact = _key(pin_name).replace("_", "")
    if pin_name in {"+", "-"}:
        return pin_name
    aliases = {
        "plus": "+",
        "noninverting": "+",
        "noninv": "+",
        "inp": "+",
        "vp": "+",
        "minus": "-",
        "inverting": "-",
        "inv": "-",
        "inn": "-",
        "vn": "-",
    }
    return aliases.get(compact, compact)


def _role_is_output_port(role: str | None) -> bool:
    key = _key(role)
    return key in {"output", "output_port", "output_terminal", "load_output", "monitor_output", "final_output"} or key.endswith("_output")


def _identity(component: Component) -> str:
    return _key(" ".join(filter(None, [component.id, component.type, component.role, component.label, component.value])))


def _key(value: str | None) -> str:
    return (value or "").strip().lower().replace("-", "_").replace(" ", "_")


__all__ = ["infer_schematic_intent", "planning_hints_from_intent"]
