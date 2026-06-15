"""Electrical topology normalization and equivalence checks."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .models import LayoutPlan


class ElectricalTopologyError(ValueError):
    """Raised when a layout no longer matches the source circuit topology."""


def _read(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _pins_to_map(raw_pins: Any) -> dict[str, str]:
    if raw_pins is None:
        return {}
    if isinstance(raw_pins, dict):
        return {str(pin): str(net) for pin, net in raw_pins.items()}

    pin_map: dict[str, str] = {}
    for item in raw_pins:
        if isinstance(item, str):
            raise ElectricalTopologyError(
                f"Pin entry {item!r} is missing a net assignment."
            )
        name = _read(item, "name", _read(item, "pin_name"))
        net = _read(item, "net", _read(item, "net_name"))
        if name is None or net is None:
            raise ElectricalTopologyError(f"Invalid pin entry: {item!r}")
        pin_map[str(name)] = str(net)
    return pin_map


def normalize_circuit_ir(circuit_ir: Any) -> dict[str, Any]:
    """Return a small canonical circuit shape from dict-like or object-like IR."""

    raw_components = _read(circuit_ir, "components", [])
    if raw_components is None:
        raw_components = []

    components: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_component in raw_components:
        component_id = _read(raw_component, "id", _read(raw_component, "component_id"))
        component_type = _read(raw_component, "type", _read(raw_component, "component_type"))
        if component_id is None or component_type is None:
            raise ElectricalTopologyError(
                f"Every component must have id and type fields: {raw_component!r}"
            )
        component_id = str(component_id)
        if component_id in seen_ids:
            raise ElectricalTopologyError(f"Duplicate component id: {component_id}")
        seen_ids.add(component_id)

        pin_map = _pins_to_map(_read(raw_component, "pins", {}))
        if not pin_map:
            raise ElectricalTopologyError(f"Component {component_id} has no pins.")

        components.append(
            {
                "id": component_id,
                "type": str(component_type),
                "role": _read(raw_component, "role"),
                "value_label": _read(raw_component, "value_label", _read(raw_component, "value")),
                "display_label": _read(raw_component, "display_label", _read(raw_component, "label")),
                "pins": pin_map,
            }
        )

    component_pin_nets = {
        component["id"]: dict(sorted(component["pins"].items()))
        for component in sorted(components, key=lambda item: item["id"])
    }

    net_to_pins: dict[str, list[str]] = {}
    for component_id, pin_map in component_pin_nets.items():
        for pin_name, net_name in pin_map.items():
            if not net_name:
                raise ElectricalTopologyError(
                    f"Pin {component_id}.{pin_name} has an empty net name."
                )
            net_to_pins.setdefault(net_name, []).append(f"{component_id}.{pin_name}")

    net_to_pins = {
        net_name: sorted(pins)
        for net_name, pins in sorted(net_to_pins.items(), key=lambda item: item[0])
    }

    return {
        "circuit_id": str(_read(circuit_ir, "id", _read(circuit_ir, "circuit_id", "circuit"))),
        "motif": _read(circuit_ir, "motif", _read(circuit_ir, "kind")),
        "components": sorted(components, key=lambda item: item["id"]),
        "component_pin_nets": component_pin_nets,
        "net_to_pins": net_to_pins,
    }


def _topology_payload_from_components(
    components: list[dict[str, Any]],
    component_pin_nets: dict[str, dict[str, str]],
    net_to_pins: dict[str, list[str]],
) -> dict[str, Any]:
    return {
        "components": [
            {"id": item["id"], "type": item["type"]}
            for item in sorted(components, key=lambda component: component["id"])
        ],
        "component_pin_nets": {
            component_id: dict(sorted(pin_map.items()))
            for component_id, pin_map in sorted(component_pin_nets.items())
        },
        "net_to_pins": {
            net_name: sorted(pins) for net_name, pins in sorted(net_to_pins.items())
        },
    }


def _signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def topology_signature_from_circuit(circuit_ir: Any) -> str:
    normalized = normalize_circuit_ir(circuit_ir)
    payload = _topology_payload_from_components(
        normalized["components"],
        normalized["component_pin_nets"],
        normalized["net_to_pins"],
    )
    return _signature(payload)


def topology_signature_from_layout(layout_plan: LayoutPlan) -> str:
    components = [
        {"id": component.id, "type": component.type}
        for component in sorted(layout_plan.components, key=lambda item: item.id)
    ]
    payload = _topology_payload_from_components(
        components,
        layout_plan.component_pin_nets,
        layout_plan.net_to_pins,
    )
    return _signature(payload)


def verify_equivalence(original_circuit: Any, layout_plan: LayoutPlan) -> bool:
    """Raise ElectricalTopologyError unless the layout matches the source circuit."""

    normalized = normalize_circuit_ir(original_circuit)
    expected_components = {
        component["id"]: component for component in normalized["components"]
    }
    actual_components = {component.id: component for component in layout_plan.components}

    if set(expected_components) != set(actual_components):
        missing = sorted(set(expected_components) - set(actual_components))
        extra = sorted(set(actual_components) - set(expected_components))
        raise ElectricalTopologyError(
            f"Component ids changed. Missing={missing}, extra={extra}"
        )

    for component_id, expected_component in expected_components.items():
        actual_component = actual_components[component_id]
        if actual_component.type != expected_component["type"]:
            raise ElectricalTopologyError(
                f"Component {component_id} type changed from "
                f"{expected_component['type']} to {actual_component.type}."
            )

        expected_pin_map = expected_component["pins"]
        layout_pin_map = layout_plan.component_pin_nets.get(component_id)
        if layout_pin_map != expected_pin_map:
            raise ElectricalTopologyError(
                f"Pin-to-net map changed for {component_id}: "
                f"expected {expected_pin_map}, got {layout_pin_map}."
            )

        component_pin_map = {
            pin.pin_name: pin.net_name for pin in actual_component.pins
        }
        if component_pin_map != expected_pin_map:
            raise ElectricalTopologyError(
                f"Pin layouts changed electrical nets for {component_id}: "
                f"expected {expected_pin_map}, got {component_pin_map}."
            )

    expected_net_to_pins = normalized["net_to_pins"]
    actual_net_to_pins = {
        net: sorted(pins) for net, pins in sorted(layout_plan.net_to_pins.items())
    }
    if actual_net_to_pins != expected_net_to_pins:
        raise ElectricalTopologyError(
            f"Net membership changed. Expected {expected_net_to_pins}, "
            f"got {actual_net_to_pins}."
        )

    all_layout_pin_refs = {
        f"{component.id}.{pin.pin_name}"
        for component in layout_plan.components
        for pin in component.pins
    }
    all_expected_pin_refs = {
        pin_ref for pins in expected_net_to_pins.values() for pin_ref in pins
    }
    if all_layout_pin_refs != all_expected_pin_refs:
        missing = sorted(all_expected_pin_refs - all_layout_pin_refs)
        extra = sorted(all_layout_pin_refs - all_expected_pin_refs)
        raise ElectricalTopologyError(
            f"Disconnected or fake pins detected. Missing={missing}, extra={extra}"
        )

    disconnected_nets = [
        net_name for net_name, pins in expected_net_to_pins.items() if len(pins) < 2
    ]
    if disconnected_nets:
        raise ElectricalTopologyError(
            f"Disconnected nets with fewer than two pins: {disconnected_nets}"
        )

    expected_signature = topology_signature_from_circuit(original_circuit)
    actual_signature = topology_signature_from_layout(layout_plan)
    if layout_plan.topology_signature != expected_signature:
        raise ElectricalTopologyError(
            "Stored topology signature changed or does not match original circuit."
        )
    if actual_signature != expected_signature:
        raise ElectricalTopologyError(
            "Layout topology signature does not match original circuit."
        )

    return True
