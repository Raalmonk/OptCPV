from dataclasses import replace

from optcpv import Circuit, Component, draw_artifact
from optcpv.models import LayoutWire, NetClass, Point
from optcpv.planner import plan_layout
from optcpv.renderers.schemdraw_backend import _junction_points
from optcpv.segments import merged_axis_aligned_segments
from optcpv.symbols import OPAMP_LEAD_LENGTH
from optcpv.vector_critic import critique_layout


def test_supply_and_reference_nets_are_terminalized_locally() -> None:
    layout = plan_layout(analog_signal_chain())

    assert layout.semantic.net_classes["GND"] == NetClass.GROUND
    assert layout.semantic.net_classes["VCC"] == NetClass.POSITIVE_SUPPLY
    assert layout.semantic.net_classes["VEE"] == NetClass.NEGATIVE_SUPPLY
    assert {"GND", "VCC", "VEE"}.isdisjoint({wire.net for wire in layout.wires})

    terminal_keys = {(terminal.component_id, terminal.pin_name, terminal.net) for terminal in layout.semantic.local_terminals}
    assert ("R8", "b", "GND") in terminal_keys
    assert ("U3A", "+", "GND") in terminal_keys
    assert ("U5A", "v+", "VCC") in terminal_keys
    assert ("U5A", "v-", "VEE") in terminal_keys


def test_parallel_summing_signal_chain_uses_stages_lanes_and_motifs() -> None:
    layout = plan_layout(analog_signal_chain())
    components = {component.id: component for component in layout.components}

    assert "semantic: parallel_summing_signal_chain" in layout.warnings
    assert components["U1A"].x == components["U2A"].x == components["U4A"].x
    assert len({components["U1A"].y, components["U2A"].y, components["U4A"].y}) == 3
    assert components["U1A"].x < components["R1"].x < components["U3A"].x < components["F1"].x < components["U5A"].x < components["F2"].x
    assert components["VOUT"].x == max(component.x for component in layout.components)

    motif_types = {motif.motif_type for motif in layout.semantic.motifs}
    assert {"opamp_buffer", "summing_opamp", "functional_filter_block", "resistor_to_ground_reference_leg"} <= motif_types

    assert layout.pin_map[("F1", "in")].x < components["F1"].x
    assert layout.pin_map[("F1", "out")].x > components["F1"].x
    assert layout.pin_map[("F2", "in")].x < components["F2"].x
    assert layout.pin_map[("F2", "out")].x > components["F2"].x


def test_semantic_critic_rejects_old_global_ground_bus() -> None:
    layout = plan_layout(analog_signal_chain())
    bad = replace(
        layout,
        wires=[
            *layout.wires,
            LayoutWire(
                net="GND",
                points=[Point(1.0, 12.0), Point(30.0, 12.0)],
                connected_pins=layout.net_to_pins["GND"],
            ),
        ],
    )

    report = critique_layout(bad)

    assert any(violation.code == "long_global_terminal_wire" for violation in report.violations)
    assert report.hard_fail


def test_feedback_and_local_terminal_semantics_have_no_hard_vector_failure() -> None:
    layout = plan_layout(analog_signal_chain())
    report = critique_layout(layout)
    codes = {violation.code for violation in report.violations}

    assert "feedback_crosses_opamp_body" not in codes
    assert "long_global_terminal_wire" not in codes
    assert "missing_local_terminal" not in codes
    assert report.hard_fail is False


def test_renderer_and_artifact_expose_semantic_metadata() -> None:
    artifact = draw_artifact(analog_signal_chain())

    assert 'data-local-terminal="true"' in artifact.svg
    assert 'data-terminal-type="ground"' in artifact.svg
    assert 'data-motif-type="filter_block"' in artifact.svg
    assert artifact.nets["GND"]["class"] == "ground"
    assert artifact.nets["GND"]["routed"] is False
    assert artifact.nets["VCC"]["routed"] is False
    assert artifact.semantic_plan["net_classes"]["VEE"] == "negative_supply"
    assert artifact.semantic_plan["local_terminals"]


def test_opamp_input_route_uses_horizontal_entry_stub_before_pin() -> None:
    layout = plan_layout(two_stage_local_ground_chain())
    pin = layout.pin_map[("U1", "+")]
    wire = next(wire for wire in layout.wires if wire.net == "vin")

    assert wire.points[-1] == Point(pin.x, pin.y)
    assert wire.points[-2].x < pin.x
    assert wire.points[-2].y == pin.y


def test_opamp_input_keepout_has_no_false_junction_dot() -> None:
    layout = plan_layout(two_stage_local_ground_chain())
    pin = layout.pin_map[("U2", "+")]
    wire = next(wire for wire in layout.wires if wire.net == "o1")

    assert not any(pin.x - 0.85 <= point.x < pin.x and abs(point.y - pin.y) <= 0.35 for point in _junction_points(wire))


def test_feedback_gain_leg_does_not_overlap_opamp_input_edge() -> None:
    layout = plan_layout(two_stage_local_ground_chain())
    minus_pin = layout.pin_map[("U1", "-")]
    plus_pin = layout.pin_map[("U1", "+")]
    edge_x = minus_pin.x + OPAMP_LEAD_LENGTH
    wire = next(wire for wire in layout.wires if wire.net == "fb1")
    input_span_segments = [
        (start, end)
        for start, end in merged_axis_aligned_segments(wire.points)
        if start.x == end.x and min(start.y, end.y) < plus_pin.y and max(start.y, end.y) > minus_pin.y
    ]

    assert input_span_segments
    assert all(abs(start.x - edge_x) > 0.16 for start, _ in input_span_segments)
    assert all(minus_pin.x < start.x < edge_x - 0.16 for start, _ in input_span_segments)


def test_redundant_global_ground_label_is_hidden_when_local_terminals_exist() -> None:
    artifact = draw_artifact(two_stage_local_ground_chain())

    assert artifact.nets["gnd"]["routed"] is False
    assert artifact.nets["gnd"]["local_terminals"]
    assert 'data-local-terminal="true"' in artifact.svg
    assert 'data-label-id="label:GND"' not in artifact.svg


def analog_signal_chain() -> Circuit:
    return Circuit(
        id="analog_signal_chain",
        motif="op_amp_network",
        components=[
            Component(id="INA", type="input", pins={"out": "ina"}, label="A"),
            Component(id="INB", type="input", pins={"out": "inb"}, label="B"),
            Component(id="INC", type="input", pins={"out": "inc"}, label="C"),
            Component(
                id="U1A",
                type="op_amp",
                pins={"+": "ina", "-": "a_buf", "out": "a_buf", "v+": "VCC", "v-": "VEE"},
                role="buffer",
            ),
            Component(
                id="U2A",
                type="op_amp",
                pins={"+": "inb", "-": "b_buf", "out": "b_buf", "v+": "VCC", "v-": "VEE"},
                role="buffer",
            ),
            Component(
                id="U4A",
                type="op_amp",
                pins={"+": "inc", "-": "c_buf", "out": "c_buf", "v+": "VCC", "v-": "VEE"},
                role="buffer",
            ),
            Component(id="R1", type="resistor", pins={"a": "a_buf", "b": "sum"}, label="R1"),
            Component(id="R2", type="resistor", pins={"a": "b_buf", "b": "sum"}, label="R2"),
            Component(id="R3", type="resistor", pins={"a": "c_buf", "b": "sum"}, label="R3"),
            Component(
                id="U3A",
                type="op_amp",
                pins={"+": "GND", "-": "sum", "out": "mix", "v+": "VCC", "v-": "VEE"},
                role="summing",
            ),
            Component(id="RF3", type="resistor", pins={"a": "mix", "b": "sum"}, role="feedback"),
            Component(id="F1", type="filter_block", pins={"in": "mix", "out": "f1"}, label="8th-order Bessel LPF"),
            Component(
                id="U5A",
                type="op_amp",
                pins={"+": "f1", "-": "gain_fb", "out": "gain", "v+": "VCC", "v-": "VEE"},
                role="gain",
            ),
            Component(id="R7", type="resistor", pins={"a": "gain", "b": "gain_fb"}, role="feedback"),
            Component(id="R8", type="resistor", pins={"a": "gain_fb", "b": "GND"}, role="reference"),
            Component(id="F2", type="filter_block", pins={"in": "gain", "out": "final"}, label="8th-order Bessel LPF"),
            Component(id="VOUT", type="output", pins={"in": "final"}, label="OUT"),
            Component(id="GND", type="ground", pins={"gnd": "GND"}, label="GND"),
        ],
    )


def two_stage_local_ground_chain() -> Circuit:
    return Circuit(
        id="two_stage_local_ground_chain",
        motif="op_amp_network",
        components=[
            Component(id="VIN", type="input", pins={"out": "vin"}, label="EEG"),
            Component(id="VOUT", type="output", pins={"in": "o2"}, label="ADC"),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}, label="GND"),
            Component(id="U1", type="op_amp", pins={"+": "vin", "-": "fb1", "out": "o1"}),
            Component(id="Rf1", type="resistor", pins={"a": "o1", "b": "fb1"}, role="feedback"),
            Component(id="Rg1", type="resistor", pins={"a": "fb1", "b": "gnd"}, role="gain"),
            Component(id="U2", type="op_amp", pins={"+": "o1", "-": "fb2", "out": "o2"}),
            Component(id="Rf2", type="resistor", pins={"a": "o2", "b": "fb2"}, role="feedback"),
            Component(id="Rg2", type="resistor", pins={"a": "fb2", "b": "gnd"}, role="gain"),
        ],
    )
