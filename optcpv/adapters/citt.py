"""Adapter for small CiTT-style component payloads."""

from __future__ import annotations

from typing import Any

from ..models import Circuit, Component


def from_citt_payload(payload: dict[str, Any]) -> Circuit:
    """Convert a simple payload with components[].nodes into native OptCPV IR."""

    raw_components = payload.get("components", [])
    if not isinstance(raw_components, list) or not raw_components:
        raise ValueError("CiTT payload requires a non-empty components list.")

    components = [_convert_component(raw, index) for index, raw in enumerate(raw_components)]
    ground_node = _optional_str(payload.get("ground_node") or payload.get("ground"))
    output_node = _goal_output_node(payload)
    input_node = _first_source_node(raw_components, ground_node) or _first_non_ground_node(components, ground_node)

    voltage_clamp_terminals_added = False
    if _is_two_electrode_voltage_clamp(payload):
        components = _canonicalize_voltage_clamp_components(
            components,
            raw_components,
            ground_node=ground_node,
            output_node=output_node,
        )
        ground_node = "0"
        output_node = "Vm"
        voltage_clamp_terminals_added = _add_canonical_voltage_clamp_terminals(components)

    if not voltage_clamp_terminals_added and input_node and not _has_terminal(components, input_node, "input"):
        components.insert(0, Component(id=_unique_id(components, "VIN"), type="input", pins={"out": input_node}, label="VIN"))
    if not voltage_clamp_terminals_added and output_node and not _has_terminal(components, output_node, "output"):
        components.append(
            Component(id=_unique_id(components, "VOUT"), type="output", pins={"in": output_node}, label="VOUT")
        )
    if ground_node and not _has_ground(components, ground_node):
        components.append(Component(id=_unique_id(components, "GND"), type="ground", pins={"gnd": ground_node}, label="GND"))

    return Circuit(
        id=str(payload.get("id") or payload.get("problem_id") or "citt_payload"),
        motif=_optional_str(payload.get("motif") or payload.get("topology")),
        title=_optional_str(payload.get("title")),
        components=components,
    )


def _convert_component(raw: dict[str, Any], index: int) -> Component:
    if not isinstance(raw, dict):
        raise ValueError("Each CiTT component must be a dictionary.")

    component_type = str(raw.get("type") or raw.get("kind") or "component")
    component_id = str(raw.get("id") or raw.get("name") or f"X{index + 1}")
    nodes = raw.get("nodes")
    if isinstance(nodes, dict):
        pins = {str(pin): str(net) for pin, net in nodes.items()}
    elif isinstance(nodes, list):
        pins = _pins_from_node_list(component_type, nodes)
    else:
        raise ValueError(f"Component {component_id} requires nodes.")

    return Component(
        id=component_id,
        type=component_type,
        pins=pins,
        label=_optional_str(raw.get("label") or raw.get("display_label") or raw.get("value_label")),
        role=_optional_str(raw.get("role")),
        value=_optional_str(raw.get("value")),
    )


def _pins_from_node_list(component_type: str, nodes: list[Any]) -> dict[str, str]:
    nets = [str(node) for node in nodes]
    key = _key(component_type)
    if key in {"ground", "gnd"}:
        return {"gnd": nets[0]}
    if "op_amp" in key or "opamp" in key:
        names = ["+", "-", "out"]
    elif len(nets) == 1:
        names = ["out"]
    else:
        names = ["a", "b", "c", "d"]
    return {name: net for name, net in zip(names, nets)}


def _goal_output_node(payload: dict[str, Any]) -> str | None:
    goals = payload.get("goals") or payload.get("goal")
    if isinstance(goals, dict):
        return _optional_str(goals.get("output_node") or goals.get("target_node") or goals.get("node"))
    if isinstance(goals, list):
        for goal in goals:
            if isinstance(goal, dict):
                node = _optional_str(goal.get("output_node") or goal.get("target_node") or goal.get("node"))
                if node:
                    return node
            elif isinstance(goal, str):
                return goal
    return _optional_str(payload.get("output_node"))


def _first_source_node(raw_components: list[Any], ground_node: str | None) -> str | None:
    for raw in raw_components:
        if not isinstance(raw, dict):
            continue
        component_type = _key(str(raw.get("type") or raw.get("kind") or ""))
        if component_type not in {"source", "voltage_source", "input"} and "source" not in component_type:
            continue
        nodes = raw.get("nodes")
        values = list(nodes.values()) if isinstance(nodes, dict) else nodes if isinstance(nodes, list) else []
        for node in values:
            text = str(node)
            if text != ground_node:
                return text
    return None


def _first_non_ground_node(components: list[Component], ground_node: str | None) -> str | None:
    for component in components:
        for net in component.pins.values():
            if net != ground_node:
                return net
    return None


def _is_two_electrode_voltage_clamp(payload: dict[str, Any]) -> bool:
    motif = _key(_optional_str(payload.get("motif") or payload.get("topology") or payload.get("topology_id")))
    return motif in {"two_electrode_voltage_clamp", "twoelectrodevoltageclamp", "tevc"}


def _add_canonical_voltage_clamp_terminals(components: list[Component]) -> bool:
    if not _has_terminal(components, "Vc", "input"):
        components.insert(0, Component(id=_unique_id(components, "VC"), type="input", pins={"out": "Vc"}, label="V_c"))
    if not _has_terminal(components, "Vm", "output"):
        components.append(Component(id=_unique_id(components, "VM"), type="output", pins={"in": "Vm"}, label="V_m"))
    if not _has_terminal(components, "Vo", "output"):
        components.append(Component(id=_unique_id(components, "VO"), type="output", pins={"in": "Vo"}, label="V_o"))
    return True


def _canonicalize_voltage_clamp_components(
    components: list[Component],
    raw_components: list[Any],
    *,
    ground_node: str | None,
    output_node: str | None,
) -> list[Component]:
    command_node, membrane_node, amplifier_output_node = _voltage_clamp_nets(
        raw_components,
        ground_node=ground_node,
        output_node=output_node,
    )
    if not command_node or not membrane_node or not amplifier_output_node:
        return components

    aliases = {
        command_node: "Vc",
        membrane_node: "Vm",
        amplifier_output_node: "Vo",
    }
    if ground_node:
        aliases[ground_node] = "0"

    return [
        Component(
            id=_canonical_voltage_clamp_component_id(component),
            type=component.type,
            pins={pin: aliases.get(net, net) for pin, net in component.pins.items()},
            label=component.label,
            role=component.role,
            value=component.value,
        )
        for component in components
    ]


def _canonical_voltage_clamp_component_id(component: Component) -> str:
    key = _key(component.id)
    if key in {"vc", "vcommand", "vcmd", "commandvoltage"} and _key(component.type) == "input":
        return "VC"
    if key in {"vm", "vmembrane", "membrane"} and _key(component.type) == "output":
        return "VM"
    if key in {"vo", "vout", "ampout", "amplifieroutput"} and _key(component.type) == "output":
        return "VO"
    if key in {"gnd", "ground"} and _key(component.type) in {"ground", "gnd"}:
        return "GND"
    return component.id


def _voltage_clamp_nets(
    raw_components: list[Any],
    *,
    ground_node: str | None,
    output_node: str | None,
) -> tuple[str | None, str | None, str | None]:
    all_nets: list[str] = []
    for raw in raw_components:
        if not isinstance(raw, dict):
            continue
        nodes = raw.get("nodes")
        values = list(nodes.values()) if isinstance(nodes, dict) else nodes if isinstance(nodes, list) else []
        all_nets.extend(str(node) for node in values)

        component_type = _key(str(raw.get("type") or raw.get("kind") or ""))
        if "op_amp" not in component_type and "opamp" not in component_type:
            continue
        if isinstance(nodes, dict):
            command = _node_for_pin(nodes, ["+", "plus", "non_inverting", "noninverting", "vp", "vplus"])
            membrane = _node_for_pin(nodes, ["-", "minus", "inverting", "vm", "vminus"])
            output = _node_for_pin(nodes, ["out", "output", "vo"])
            if command and membrane and output:
                return command, membrane, output
        elif isinstance(nodes, list) and len(nodes) >= 3:
            return str(nodes[0]), str(nodes[1]), str(nodes[2])

    return (
        _find_named_net(all_nets, {"vc", "vcommand", "vcmd", "commandvoltage"}),
        output_node or _find_named_net(all_nets, {"vm", "vmembrane", "membrane"}),
        _find_named_net(all_nets, {"vo", "vout", "ampout", "amplifieroutput"}),
    )


def _node_for_pin(nodes: dict[Any, Any], keys: list[str]) -> str | None:
    normalized = {_key(str(pin)): str(net) for pin, net in nodes.items()}
    for key in keys:
        value = normalized.get(_key(key))
        if value:
            return value
    return None


def _find_named_net(nets: list[str], names: set[str]) -> str | None:
    for net in nets:
        if net == "0":
            continue
        if _key(net) in names:
            return net
    return None


def _has_terminal(components: list[Component], net: str, terminal_type: str) -> bool:
    return any(_key(component.type) == terminal_type and net in component.pins.values() for component in components)


def _has_ground(components: list[Component], net: str) -> bool:
    return any(_key(component.type) in {"ground", "gnd"} and net in component.pins.values() for component in components)


def _unique_id(components: list[Component], preferred: str) -> str:
    existing = {component.id for component in components}
    if preferred not in existing:
        return preferred
    index = 2
    while f"{preferred}{index}" in existing:
        index += 1
    return f"{preferred}{index}"


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _key(value: str | None) -> str:
    return (value or "").lower().replace("-", "_").replace(" ", "_")
