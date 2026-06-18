"""Local validation and repair for external semantic planning hints."""

from __future__ import annotations

from .models import Circuit, Component, NetClass
from .planning_hints import GridPlacementHint, RoutePolicyHint, SchematicLayoutHints
from .semantics import classify_net, is_local_terminal_net, is_negative_supply_pin, is_positive_supply_pin, is_reference_pin


TERMINAL_NET_CLASSES = {
    NetClass.GROUND,
    NetClass.POSITIVE_SUPPLY,
    NetClass.NEGATIVE_SUPPLY,
    NetClass.REFERENCE,
}


def legalize_planning_hints(circuit: Circuit, hints: SchematicLayoutHints) -> SchematicLayoutHints | None:
    """Return topology-safe hints, or None when hints must be ignored."""

    component_ids = {component.id for component in circuit.components}
    net_names = {net for component in circuit.components for net in component.pins.values()}
    if not hints.placements:
        return None
    if any(placement.component_id not in component_ids for placement in hints.placements):
        return None
    if not _members_are_known(hints, component_ids, net_names):
        return None
    if not _route_policies_are_legal(hints, net_names):
        return None

    placements = _dedupe_component_placements(hints.placements)
    if placements is None:
        return None
    placements = _force_terminal_stage_conventions(circuit, placements)
    placements = _force_opamp_orientation(circuit, placements)
    placements = _enforce_monotonic_signal_order(circuit, placements)
    placements = _resolve_duplicate_cells(placements)

    confidence = hints.confidence
    if len(placements) < len(circuit.components):
        confidence = min(confidence or 0.5, 0.45)

    return hints.with_updates(placements=tuple(placements.values()), confidence=confidence)


def _members_are_known(hints: SchematicLayoutHints, component_ids: set[str], net_names: set[str]) -> bool:
    for stage in hints.stages:
        if set(stage.members) - component_ids:
            return False
    for lane in hints.lanes:
        if set(lane.members) - component_ids:
            return False
    for motif in hints.motifs:
        if set(motif.members) - component_ids:
            return False
    for motif in hints.block_internal_motifs:
        if set(motif.members) - component_ids:
            return False
    for block in hints.blocks:
        if set(block.members) - component_ids:
            return False
        if set(block.ports.values()) - net_names:
            return False
    for route in hints.inter_block_routes:
        if route.net not in net_names:
            return False
    for loop in hints.auxiliary_loops:
        if set(loop.members) - component_ids or set(loop.nets) - net_names:
            return False
    for override in hints.orientation_overrides:
        if override.component_id not in component_ids:
            return False
    return True


def _route_policies_are_legal(hints: SchematicLayoutHints, net_names: set[str]) -> bool:
    terminal_nets = {net for net in net_names if classify_net(net) in TERMINAL_NET_CLASSES}
    if set(hints.local_terminal_policy) - net_names:
        return False
    for net, policy in hints.local_terminal_policy.items():
        if classify_net(net) in TERMINAL_NET_CLASSES and policy != "local_symbol_only":
            return False

    for route_policy in hints.route_policies:
        if route_policy.net is not None:
            if route_policy.net not in net_names:
                return False
            if classify_net(route_policy.net) in TERMINAL_NET_CLASSES and route_policy.policy != "local_terminal_only":
                return False
        if _terminal_role(route_policy.net_role) and route_policy.policy != "local_terminal_only":
            return False

    for rule in hints.routing_rules:
        lowered = _key(rule)
        if any(_key(net) in lowered for net in terminal_nets) and any(
            token in lowered for token in ("route", "global", "rail", "bus", "corridor")
        ):
            return False
    return True


def _dedupe_component_placements(
    placements: tuple[GridPlacementHint, ...],
) -> dict[str, GridPlacementHint] | None:
    by_id: dict[str, GridPlacementHint] = {}
    for placement in placements:
        existing = by_id.get(placement.component_id)
        if existing is None:
            by_id[placement.component_id] = placement
            continue
        if (
            existing.stage_x != placement.stage_x
            or existing.lane_y != placement.lane_y
            or existing.orientation != placement.orientation
        ):
            return None
        if placement.confidence > existing.confidence:
            by_id[placement.component_id] = placement
    return by_id


def _force_terminal_stage_conventions(
    circuit: Circuit,
    placements: dict[str, GridPlacementHint],
) -> dict[str, GridPlacementHint]:
    if not placements:
        return placements
    minimum = min(placement.stage_x for placement in placements.values())
    maximum = max(placement.stage_x for placement in placements.values())
    result = dict(placements)
    for component in circuit.components:
        placement = result.get(component.id)
        if placement is None:
            continue
        if _is_input_or_source(component):
            result[component.id] = placement.with_updates(stage_x=min(placement.stage_x, minimum), orientation="RIGHT")
        elif _is_output(component):
            result[component.id] = placement.with_updates(stage_x=max(placement.stage_x, maximum), orientation="RIGHT")
    return result


def _force_opamp_orientation(
    circuit: Circuit,
    placements: dict[str, GridPlacementHint],
) -> dict[str, GridPlacementHint]:
    result = dict(placements)
    for component in circuit.components:
        if not _is_opamp(component) or component.id not in result:
            continue
        placement = result[component.id]
        if placement.orientation not in {"right", "right_flip"}:
            result[component.id] = placement.with_updates(orientation="RIGHT")
    return result


def _resolve_duplicate_cells(
    placements: dict[str, GridPlacementHint],
) -> dict[str, GridPlacementHint]:
    occupied: set[tuple[int, int]] = set()
    result: dict[str, GridPlacementHint] = {}
    ordered = sorted(placements.values(), key=lambda item: (item.stage_x, item.lane_y, item.component_id))
    for placement in ordered:
        lane_y = placement.lane_y
        cell = (placement.stage_x, lane_y)
        while cell in occupied:
            lane_y += 1 if lane_y >= 0 else -1
            cell = (placement.stage_x, lane_y)
        occupied.add(cell)
        result[placement.component_id] = placement.with_updates(lane_y=lane_y)
    return result


def _enforce_monotonic_signal_order(
    circuit: Circuit,
    placements: dict[str, GridPlacementHint],
) -> dict[str, GridPlacementHint]:
    result = dict(placements)
    dependencies = _signal_dependencies(circuit)
    changed = True
    passes = 0
    while changed and passes < len(result) + 2:
        changed = False
        passes += 1
        for source_id, target_id, net in dependencies:
            if _feedback_or_auxiliary_net(net) or source_id == target_id:
                continue
            source = result.get(source_id)
            target = result.get(target_id)
            if source is None or target is None:
                continue
            if target.stage_x < source.stage_x:
                result[target_id] = target.with_updates(stage_x=source.stage_x + 1)
                changed = True
    return _force_terminal_stage_conventions(circuit, result)


def _signal_dependencies(circuit: Circuit) -> list[tuple[str, str, str]]:
    by_net: dict[str, list[tuple[Component, str]]] = {}
    for component in circuit.components:
        for pin_name, net in component.pins.items():
            if is_local_terminal_net(net):
                continue
            by_net.setdefault(net, []).append((component, pin_name))

    dependencies: list[tuple[str, str, str]] = []
    for net, pins in by_net.items():
        drivers = [(component, pin_name) for component, pin_name in pins if _is_driver_pin(component, pin_name)]
        if not drivers:
            continue
        for driver, _ in drivers:
            for target, target_pin in pins:
                if target.id == driver.id or _is_driver_pin(target, target_pin):
                    continue
                dependencies.append((driver.id, target.id, net))
    return dependencies


def _is_driver_pin(component: Component, pin_name: str) -> bool:
    kind = _pin_kind(pin_name)
    if _is_input_or_source(component) and kind in {"out", "output", "o"}:
        return True
    if _is_output(component):
        return False
    return kind in {"out", "output", "o", "vout"}


def _pin_sort_key(component: Component, pin_name: str) -> tuple[int, str, str]:
    if _is_input_or_source(component):
        return (0, component.id, pin_name)
    if _is_output(component):
        return (2, component.id, pin_name)
    return (1, component.id, pin_name)


def _terminal_role(role: str) -> bool:
    key = _key(role)
    return any(token in key for token in ("ground", "gnd", "supply", "vcc", "vdd", "vee", "vss", "reference", "ref"))


def _feedback_or_auxiliary_net(net: str) -> bool:
    key = _key(net)
    return any(token in key for token in ("fb", "feedback", "rld", "right_leg", "driven_right_leg", "common_mode", "cmfb", "aux"))


def _is_input_or_source(component: Component) -> bool:
    key = _key(component.type)
    return key in {"input", "input_terminal", "voltage_source", "source"} or "source" in key


def _is_output(component: Component) -> bool:
    return _key(component.type) == "output" or "output" in _key(component.role)


def _is_opamp(component: Component) -> bool:
    key = _key(component.type)
    return "op_amp" in key or "opamp" in key or "operational_amplifier" in key or "ideal_op_amp" in key


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
