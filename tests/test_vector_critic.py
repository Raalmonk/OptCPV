from dataclasses import replace

from optcpv.examples import voltage_divider
from optcpv.models import LayoutWire, Point
from optcpv.patch import LayoutPatch, MoveComponent, apply_patch
from optcpv.planner import plan_layout
from optcpv.segments import merged_axis_aligned_segments
from optcpv.vector_critic import critique_layout


def test_vector_critic_catches_component_overlap() -> None:
    circuit = voltage_divider()
    layout = plan_layout(circuit)
    overlapped = apply_patch(circuit, layout, LayoutPatch(move_component=[MoveComponent("R1", 2.0, 4.0)]))
    report = critique_layout(overlapped)

    assert any(violation.code == "component_overlap" for violation in report.violations)
    assert report.hard_fail


def test_vector_critic_catches_different_net_wire_overlap() -> None:
    layout = plan_layout(voltage_divider())
    overlapped = replace(
        layout,
        wires=[
            LayoutWire(net="a", points=[Point(2.0, 2.0), Point(8.0, 2.0)], connected_pins=[]),
            LayoutWire(net="b", points=[Point(4.0, 2.0), Point(10.0, 2.0)], connected_pins=[]),
        ],
    )
    report = critique_layout(overlapped)

    assert any(violation.code == "wire_net_overlap" for violation in report.violations)
    assert report.hard_fail


def test_same_net_nested_bus_segments_are_merged_for_display() -> None:
    segments = merged_axis_aligned_segments(
        [
            Point(10.0, 2.0),
            Point(2.0, 2.0),
            Point(10.0, 2.0),
            Point(6.0, 2.0),
            Point(10.0, 2.0),
        ]
    )

    assert segments == [(Point(2.0, 2.0), Point(10.0, 2.0))]
