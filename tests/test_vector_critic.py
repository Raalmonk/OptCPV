from dataclasses import replace

from optcpv import Component, Circuit
from optcpv.examples import instrumentation_amplifier, non_inverting_op_amp, voltage_divider
from optcpv.models import LayoutWire, Point
from optcpv.patch import LayoutPatch, MoveComponent, apply_patch
from optcpv.planner import plan_layout
from optcpv.segments import merged_axis_aligned_segments
from optcpv.symbols import OPAMP_OUTPUT_LEAD_X
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


def test_vector_critic_rejects_diagonal_wire() -> None:
    layout = plan_layout(voltage_divider())
    diagonal = replace(
        layout,
        wires=[
            LayoutWire(net="diag", points=[Point(1.0, 1.0), Point(3.0, 2.0)], connected_pins=[]),
            *layout.wires,
        ],
    )
    report = critique_layout(diagonal)

    assert any(violation.code == "diagonal_wire" and violation.hard for violation in report.violations)
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


def test_vector_critic_catches_wire_that_misses_connected_pin() -> None:
    layout = plan_layout(voltage_divider())
    wire = layout.wires[0]
    disconnected = replace(wire, points=[Point(0.0, 0.0), Point(1.0, 0.0)])
    report = critique_layout(replace(layout, wires=[disconnected, *layout.wires[1:]]))

    assert any(violation.code == "wire_pin_disconnected" for violation in report.violations)
    assert report.hard_fail


def test_instrumentation_amplifier_routes_touch_all_connected_pins() -> None:
    report = critique_layout(plan_layout(instrumentation_amplifier()))

    assert not any(violation.code == "wire_pin_disconnected" for violation in report.violations)


def test_non_inverting_feedback_routes_around_opamp_body() -> None:
    report = critique_layout(plan_layout(non_inverting_op_amp()))

    assert not any(violation.code == "feedback_crosses_opamp_body" for violation in report.violations)


def test_opamp_pin_contract_matches_renderer_leads() -> None:
    layout = plan_layout(_two_stage_opamp_chain())
    u1 = next(component for component in layout.components if component.id == "U1")

    assert layout.pin_map[("U1", "+")].x == u1.x
    assert layout.pin_map[("U1", "-")].x == u1.x
    assert layout.pin_map[("U1", "+")].y < u1.y < layout.pin_map[("U1", "-")].y
    assert layout.pin_map[("U1", "out")].x == u1.x + OPAMP_OUTPUT_LEAD_X


def test_vector_critic_catches_opamp_pin_renderer_mismatch() -> None:
    layout = plan_layout(_two_stage_opamp_chain())
    bad_pin_map = dict(layout.pin_map)
    pin = bad_pin_map[("U1", "out")]
    bad_pin_map[("U1", "out")] = replace(pin, x=pin.x + 0.2)
    report = critique_layout(replace(layout, pin_map=bad_pin_map))

    assert any(violation.code == "pin_renderer_mismatch" for violation in report.violations)
    assert report.hard_fail


def _two_stage_opamp_chain() -> Circuit:
    return Circuit(
        id="two_stage_opamp_chain",
        motif="op_amp_network",
        components=[
            Component(id="VIN", type="input", pins={"out": "vin"}),
            Component(id="VOUT", type="output", pins={"in": "o2"}),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}),
            Component(id="U1", type="op_amp", pins={"+": "vin", "-": "fb1", "out": "o1"}),
            Component(id="Rf1", type="resistor", pins={"a": "o1", "b": "fb1"}, role="feedback"),
            Component(id="Rg1", type="resistor", pins={"a": "fb1", "b": "gnd"}, role="gain"),
            Component(id="U2", type="op_amp", pins={"+": "o1", "-": "fb2", "out": "o2"}),
            Component(id="Rf2", type="resistor", pins={"a": "o2", "b": "fb2"}, role="feedback"),
            Component(id="Rg2", type="resistor", pins={"a": "fb2", "b": "gnd"}, role="gain"),
        ],
    )
