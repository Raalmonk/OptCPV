import re

from optcpv import Circuit, Component, draw_optimized_artifact, infer_schematic_intent, planning_hints_from_intent
from optcpv.hint_legalizer import legalize_planning_hints
from optcpv.planner import plan_layout
from optcpv.vector_critic import critique_layout
from optcpv.visual_review import VisualPatch, VisualReview, layout_patch_from_visual_review


def test_schematic_intent_detects_opamp_feedback_grammar() -> None:
    circuit = feedback_stage()

    intent = infer_schematic_intent(circuit)
    hints = planning_hints_from_intent(circuit, intent)

    assert "op_amp_feedback_stage" in intent.component_roles["U1"]
    assert "feedback_element" in intent.component_roles["Rf"]
    assert "shunt_reference_element" in intent.component_roles["Rg"]
    assert "feedback_input" in intent.pin_roles["U1.-"]
    assert {"ground", "local_reference"} <= set(intent.net_roles["gnd"])
    assert any(
        constraint.constraint_type == "feedback_outside_body"
        and constraint.subject == "U1"
        and constraint.net == "vout"
        for constraint in intent.constraints
    )
    assert any(route.net == "vout" and "feedback" in route.net_role for route in intent.route_intents)
    assert hints is not None
    assert hints.intent == intent
    assert legalize_planning_hints(circuit, hints) is not None


def test_generic_signal_network_uses_schematic_grammar_layout() -> None:
    layout = plan_layout(transistor_stage())
    components = {component.id: component for component in layout.components}

    assert "planning_hints: accepted" in layout.warnings
    assert components["VIN"].x < components["Q1"].x < components["VOUT"].x
    assert components["GND"].y > components["Q1"].y


def test_two_opamp_series_network_uses_generic_grammar_not_whole_circuit_template() -> None:
    circuit = two_opamp_series_sense_network()

    intent = infer_schematic_intent(circuit)
    hints = planning_hints_from_intent(circuit, intent)
    placements = {placement.component_id: placement for placement in hints.placements}

    assert "output_port" not in intent.component_roles["R_o"]
    assert "series_element" in intent.component_roles["R_o"]
    assert any(
        constraint.constraint_type == "left_of"
        and constraint.subject == "A"
        and constraint.object == "R_o"
        and constraint.net == "icl"
        for constraint in intent.constraints
    )
    assert any(route.net == "vm" and route.policy == "named_net_label" for route in intent.route_intents)
    assert placements["BUF"].stage_x < placements["DiffAmp"].stage_x < placements["A"].stage_x
    assert placements["A"].stage_x < placements["R_o"].stage_x < placements["VM"].stage_x
    assert placements["BUF"].lane_y < placements["DiffAmp"].lane_y

    layout = plan_layout(circuit)
    components = {component.id: component for component in layout.components}
    report = critique_layout(layout)

    assert "planning_hints: accepted" in layout.warnings
    assert layout.width == 980
    assert layout.height == 600
    assert components["BUF"].x < components["DiffAmp"].x < components["A"].x
    assert components["A"].x < components["R_o"].x < components["VM"].x
    assert components["BUF"].y < components["DiffAmp"].y
    assert components["R_m"].orientation == "down"
    assert "vm" not in {wire.net for wire in layout.wires}
    assert any(terminal.net == "vm" and terminal.terminal_type == "signal_label" for terminal in layout.semantic.local_terminals)
    assert not report.hard_fail


def test_two_opamp_series_network_renders_without_redundant_output_terminal() -> None:
    artifact = draw_optimized_artifact(two_opamp_series_sense_network(), max_iterations=3)

    assert artifact.critic_report["score"] == 0
    assert not re.search(r'<g class="optcpv-local-terminal"[^>]*data-component-id="VM"', artifact.svg)
    assert re.search(r'<g class="optcpv-local-terminal"[^>]*data-component-id="R_o"', artifact.svg)


def test_visual_review_add_constraint_becomes_route_policy_patch() -> None:
    layout = plan_layout(feedback_stage())
    review = VisualReview(
        passed=False,
        score=40,
        recognized_topology="feedback_stage",
        patches=[
            VisualPatch(
                action="add_constraint",
                net="vout",
                constraint={
                    "type": "feedback_outside_body",
                    "subject": "U1",
                    "net": "vout",
                    "preferred_side": "bottom",
                },
            )
        ],
    )

    patch = layout_patch_from_visual_review(review, layout)

    assert patch.set_route_policy
    assert patch.set_route_policy[0].net == "vout"
    assert patch.set_route_policy[0].policy == "bottom_feedback_corridor"


def feedback_stage() -> Circuit:
    return Circuit(
        id="feedback_stage",
        components=[
            Component(id="VIN", type="input", pins={"out": "vin"}),
            Component(id="VOUT", type="output", pins={"in": "vout"}),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}),
            Component(id="U1", type="op_amp", pins={"+": "vin", "-": "vm", "out": "vout"}),
            Component(id="Rf", type="resistor", pins={"a": "vout", "b": "vm"}),
            Component(id="Rg", type="resistor", pins={"a": "vm", "b": "gnd"}),
        ],
    )


def transistor_stage() -> Circuit:
    return Circuit(
        id="generic_transistor_stage",
        components=[
            Component(id="VIN", type="input", pins={"out": "base"}),
            Component(id="Q1", type="transistor", pins={"base": "base", "collector": "collector", "emitter": "gnd"}),
            Component(id="VOUT", type="output", pins={"in": "collector"}),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}),
        ],
    )


def two_opamp_series_sense_network() -> Circuit:
    return Circuit(
        id="generic_two_opamp_series_sense_network",
        components=[
            Component(id="VC", type="input", pins={"out": "vc"}, label="Vc"),
            Component(id="BUF", type="op_amp", pins={"+": "vm", "-": "sense", "out": "sense"}, label="Buffer Amp"),
            Component(id="DiffAmp", type="op_amp", pins={"+": "vc", "-": "sense", "out": "drive"}, label="Diff Amp"),
            Component(id="A", type="ammeter", pins={"a": "drive", "b": "icl"}, label="A"),
            Component(
                id="R_o",
                type="resistor",
                pins={"a": "icl", "b": "vm"},
                label="Ro",
                role="current_electrode_output_resistance",
            ),
            Component(id="R_m", type="resistor", pins={"a": "vm", "b": "gnd"}, label="Rm"),
            Component(id="VM", type="output", pins={"in": "vm"}, label="Vm"),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}, label="GND"),
        ],
    )
