"""Topology guards for layout optimization."""

from __future__ import annotations

import hashlib
import json

from .models import Circuit, LayoutPlan, circuit_from_any


class TopologyVerificationError(Exception):
    """Raised when a layout no longer represents the input circuit topology."""


def topology_signature(circuit: Circuit | dict) -> str:
    native = circuit_from_any(circuit)
    payload = {
        "components": [
            {
                "id": component.id,
                "type": component.type,
                "pins": sorted(component.pins.items()),
            }
            for component in sorted(native.components, key=lambda item: item.id)
        ]
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_layout_topology(circuit: Circuit | dict, layout: LayoutPlan) -> None:
    native = circuit_from_any(circuit)
    expected_components = {component.id: component for component in native.components}
    actual_components = {component.id: component for component in layout.components}

    if set(actual_components) != set(expected_components):
        missing = sorted(set(expected_components) - set(actual_components))
        fake = sorted(set(actual_components) - set(expected_components))
        raise TopologyVerificationError(f"Component set changed; missing={missing}, fake={fake}.")

    for component_id, expected in expected_components.items():
        actual = actual_components[component_id]
        if actual.type != expected.type:
            raise TopologyVerificationError(f"Component {component_id} type changed.")
        if set(actual.pins) != set(expected.pins):
            raise TopologyVerificationError(f"Component {component_id} pin names changed.")
        if actual.pins != expected.pins:
            raise TopologyVerificationError(f"Component {component_id} pin-to-net mapping changed.")

    expected_pin_map = {
        (component.id, pin_name): net
        for component in native.components
        for pin_name, net in component.pins.items()
    }
    actual_pin_map = {
        key: pin.net
        for key, pin in layout.pin_map.items()
    }
    if actual_pin_map != expected_pin_map:
        raise TopologyVerificationError("Layout pin map does not match circuit pins.")

    expected_net_to_pins: dict[str, list[tuple[str, str]]] = {}
    for key, net in expected_pin_map.items():
        expected_net_to_pins.setdefault(net, []).append(key)
    expected_net_to_pins = {net: sorted(pins) for net, pins in sorted(expected_net_to_pins.items())}
    if layout.net_to_pins != expected_net_to_pins:
        raise TopologyVerificationError("Net membership changed.")

    expected_signature = topology_signature(native)
    if layout.topology_signature != expected_signature:
        raise TopologyVerificationError("Topology signature changed.")
