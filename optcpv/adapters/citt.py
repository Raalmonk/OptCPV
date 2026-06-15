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

    if input_node and not _has_terminal(components, input_node, "input"):
        components.insert(0, Component(id=_unique_id(components, "VIN"), type="input", pins={"out": input_node}, label="VIN"))
    if output_node and not _has_terminal(components, output_node, "output"):
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
