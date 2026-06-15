"""Adapters from CiTT-style circuit payloads into schem_forge IR."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _read(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _key(value: Any) -> str:
    return str(value or "").lower().replace("-", "_").replace(" ", "_")


def _nodes(raw_component: Any) -> list[str]:
    nodes = _read(raw_component, "nodes")
    if nodes is None:
        nodes = _read(raw_component, "node_ids")
    if nodes is None:
        return []
    return [str(node) for node in nodes]


def _pins(raw_component: Any) -> dict[str, str] | None:
    raw_pins = _read(raw_component, "pins")
    if raw_pins is None:
        return None
    if isinstance(raw_pins, dict):
        return {str(pin): str(net) for pin, net in raw_pins.items()}
    pin_map: dict[str, str] = {}
    for item in raw_pins:
        name = _read(item, "name", _read(item, "pin_name"))
        net = _read(item, "net", _read(item, "net_name", _read(item, "node")))
        if name is not None and net is not None:
            pin_map[str(name)] = str(net)
    return pin_map or None


def _pin_names_for_type(component_type: str, node_count: int) -> list[str]:
    key = _key(component_type)
    if key in {"resistor", "capacitor", "inductor"}:
        return ["a", "b"]
    if key == "voltage_source":
        return ["+", "-"]
    if key == "current_source":
        return ["from", "to"]
    if key in {"ideal_op_amp", "op_amp_ideal", "op_amp", "opamp"}:
        return ["+", "-", "out", "ref"][:node_count]
    return [f"p{index + 1}" for index in range(node_count)]


def _component_id(raw_component: Any, index: int) -> str:
    return str(
        _read(
            raw_component,
            "id",
            _read(raw_component, "component_id", _read(raw_component, "name", f"X{index + 1}")),
        )
    )


def _component_type(raw_component: Any) -> str:
    return str(_read(raw_component, "type", _read(raw_component, "component_type", "component")))


def _value_label(raw_component: Any) -> str | None:
    value = _read(raw_component, "value_label", _read(raw_component, "value"))
    return None if value is None else str(value)


def _display_label(raw_component: Any) -> str | None:
    label = _read(raw_component, "display_label", _read(raw_component, "label"))
    return None if label is None else str(label)


def _goal_nodes(circuit_problem: Any) -> set[str]:
    goals = _read(circuit_problem, "goals", None)
    if goals is None:
        goal = _read(circuit_problem, "goal", None)
        goals = [goal] if goal is not None else []
    nodes: set[str] = set()
    for goal in goals:
        goal_type = _key(_read(goal, "type", _read(goal, "kind", "")))
        node = _read(goal, "node", _read(goal, "node_name", _read(goal, "net")))
        if node is not None and ("voltage" in goal_type or "node" in goal_type or goal_type == ""):
            nodes.add(str(node))
    return nodes


def _ground_node(circuit_problem: Any, all_nets: set[str]) -> str | None:
    explicit = _read(circuit_problem, "ground", _read(circuit_problem, "ground_node"))
    if explicit is not None:
        return str(explicit)
    for net_name in sorted(all_nets):
        if _key(net_name) in {"0", "gnd", "ground"}:
            return net_name
    return None


def _unique_id(base: str, used_ids: set[str]) -> str:
    if base not in used_ids:
        used_ids.add(base)
        return base
    index = 2
    while f"{base}_{index}" in used_ids:
        index += 1
    candidate = f"{base}_{index}"
    used_ids.add(candidate)
    return candidate


def _infer_motif(components: list[dict[str, Any]], fallback: str | None) -> str | None:
    if fallback:
        return fallback
    type_keys = [_key(component["type"]) for component in components]
    opamps = sum(1 for key in type_keys if key in {"ideal_op_amp", "op_amp_ideal", "op_amp", "opamp"})
    resistors = sum(1 for key in type_keys if key == "resistor")
    capacitors = sum(1 for key in type_keys if key == "capacitor")
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


def circuit_problem_to_schem_forge_ir(circuit_problem: Any) -> dict:
    """Convert object-like or dict-like CiTT CircuitProblem data into schem_forge IR."""

    raw_problem = deepcopy(circuit_problem)
    raw_components = _read(raw_problem, "components", []) or []
    components: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    all_nets: set[str] = set()
    source_positive_nets: set[str] = set()

    for index, raw_component in enumerate(raw_components):
        component_id = _unique_id(_component_id(raw_component, index), used_ids)
        component_type = _component_type(raw_component)
        pins = _pins(raw_component)
        if pins is None:
            nodes = _nodes(raw_component)
            pin_names = _pin_names_for_type(component_type, len(nodes))
            pins = {pin_name: node for pin_name, node in zip(pin_names, nodes)}
        if not pins:
            continue

        all_nets.update(pins.values())
        key = _key(component_type)
        if key == "voltage_source" and "+" in pins:
            source_positive_nets.add(pins["+"])
        elif key == "current_source" and "from" in pins:
            source_positive_nets.add(pins["from"])

        components.append(
            {
                "id": component_id,
                "type": component_type,
                "role": _read(raw_component, "role"),
                "value_label": _value_label(raw_component),
                "display_label": _display_label(raw_component) or component_id,
                "pins": pins,
            }
        )

    ground_net = _ground_node(raw_problem, all_nets)
    has_ground_component = any(
        _key(component["type"]) in {"ground", "gnd"} or _key(component.get("role")) == "ground"
        for component in components
    )
    if ground_net and not has_ground_component:
        components.append(
            {
                "id": _unique_id("GND", used_ids),
                "type": "ground",
                "role": "ground",
                "display_label": "GND",
                "pins": {"gnd": ground_net},
            }
        )

    existing_output_nets = {
        next(iter(component["pins"].values()))
        for component in components
        if _key(component["type"]) == "output" or _key(component.get("role")) == "output"
    }
    for node in sorted(_goal_nodes(raw_problem)):
        if node in all_nets and node not in existing_output_nets:
            components.append(
                {
                    "id": _unique_id("VOUT", used_ids),
                    "type": "output",
                    "role": "output",
                    "display_label": node,
                    "pins": {"in": node},
                }
            )
            existing_output_nets.add(node)

    existing_input_nets = {
        next(iter(component["pins"].values()))
        for component in components
        if _key(component["type"]) == "input" or _key(component.get("role")) in {"input", "sensor", "input_source"}
    }
    for node in sorted(source_positive_nets):
        if node in all_nets and node not in existing_input_nets:
            components.append(
                {
                    "id": _unique_id("VIN", used_ids),
                    "type": "input",
                    "role": "input_source",
                    "display_label": node,
                    "pins": {"out": node},
                }
            )
            existing_input_nets.add(node)

    circuit_id = _read(raw_problem, "id", _read(raw_problem, "circuit_id", "citt_circuit"))
    motif = _infer_motif(components, _read(raw_problem, "motif", _read(raw_problem, "kind")))
    return {
        "id": str(circuit_id),
        "motif": motif,
        "components": components,
    }
