from dataclasses import replace

import pytest

from optcpv.examples import voltage_divider
from optcpv.patch import LayoutPatch, MoveComponent, MoveLabel, apply_patch
from optcpv.planner import plan_layout
from optcpv.verifier import TopologyVerificationError, verify_layout_topology


def test_topology_verifier_rejects_electrical_changes() -> None:
    circuit = voltage_divider()
    layout = plan_layout(circuit)

    changed_net = replace(layout.components[1], pins={"a": "vin", "b": "bad"})
    with pytest.raises(TopologyVerificationError):
        verify_layout_topology(circuit, replace(layout, components=[layout.components[0], changed_net, *layout.components[2:]]))

    with pytest.raises(TopologyVerificationError):
        verify_layout_topology(circuit, replace(layout, components=layout.components[:-1]))

    fake = replace(layout.components[0], id="FAKE")
    with pytest.raises(TopologyVerificationError):
        verify_layout_topology(circuit, replace(layout, components=[fake, *layout.components]))

    changed_pin = replace(layout.components[1], pins={"x": "vin", "b": "vout"})
    with pytest.raises(TopologyVerificationError):
        verify_layout_topology(circuit, replace(layout, components=[layout.components[0], changed_pin, *layout.components[2:]]))


def test_topology_verifier_allows_geometric_changes() -> None:
    circuit = voltage_divider()
    layout = plan_layout(circuit)
    moved = apply_patch(circuit, layout, LayoutPatch(move_component=[MoveComponent("R1", 5.3, 4.2)]))
    verify_layout_topology(circuit, moved)

    label = moved.labels[0]
    moved_label = apply_patch(circuit, moved, LayoutPatch(move_label=[MoveLabel(label.id, label.x, label.y - 0.2)]))
    verify_layout_topology(circuit, moved_label)
