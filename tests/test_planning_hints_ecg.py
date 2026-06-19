from dataclasses import replace
from itertools import combinations

from optcpv import (
    Circuit,
    BlockHint,
    Component,
    FakePlanningClient,
    FakeVisualReviewClient,
    GridPlacementHint,
    OrientationOverrideHint,
    PlanningHints,
    RoutePolicyHint,
    SchematicLayoutHints,
    SchematicPlanningRequest,
    VisualPatch,
    VisualReview,
    draw_artifact,
    draw_optimized_artifact,
)
from optcpv.hint_legalizer import legalize_planning_hints
from optcpv.models import LayoutWire, Point
from optcpv.optimizer import propose_local_patch
from optcpv.patch import LayoutPatch, SetOrientation, apply_patch
from optcpv.planner import plan_layout
from optcpv.route_contract import assert_no_diagonal_wires, route_crosses_keepout
from optcpv.vector_critic import critique_layout
from optcpv.visual_review import layout_patch_from_visual_review


def test_fake_planning_client_places_ecg_rld_loop_in_bottom_auxiliary_lane() -> None:
    circuit = ecg_rld_frontend()
    layout = plan_layout(circuit, planning_client=FakePlanningClient(ecg_rld_hints()))
    components = {component.id: component for component in layout.components}
    rld = next(wire for wire in layout.wires if wire.net == "rld")
    report = critique_layout(layout)
    hard_codes = {violation.code for violation in report.violations if violation.hard}

    assert "planning_hints: accepted" in layout.warnings
    assert components["A1"].x == components["A2"].x
    assert components["A1"].y != components["A2"].y
    assert components["A1"].x < components["A3"].x < components["A4"].x
    assert components["Aau"].y > components["A4"].y
    assert max(point.y for point in rld.points) > components["Aau"].bbox.bottom
    assert not any(route_crosses_keepout(rld.points, components[item].bbox.expanded(-0.06)) for item in ["Aau", "A3", "A4"])
    assert "wire_net_overlap" not in hard_codes


def test_rld_feedback_stays_local_while_drive_uses_bottom_corridor() -> None:
    layout = plan_layout(ecg_rld_frontend(), planning_client=FakePlanningClient(ecg_rld_hints()))
    components = {component.id: component for component in layout.components}
    rld_fb = next(wire for wire in layout.wires if wire.net == "rld_fb")
    rld = next(wire for wire in layout.wires if wire.net == "rld")
    rld_drive = next(wire for wire in layout.wires if wire.net == "rld_drive")

    assert max(point.y for point in rld_fb.points) < components["Aau"].bbox.bottom + 0.2
    assert max(point.y for point in rld.points) > components["Aau"].bbox.bottom
    assert max(point.y for point in rld_drive.points) <= layout.pin_map[("Ro", "b")].y + 1e-6


def test_single_feedback_input_branch_does_not_backtrack_to_fake_corridor() -> None:
    layout = plan_layout(ecg_rld_frontend(), planning_client=FakePlanningClient(ecg_rld_hints()))
    components = {component.id: component for component in layout.components}
    rld_fb = next(wire for wire in layout.wires if wire.net == "rld_fb")
    feedback_pin = layout.pin_map[("Rf", "b")]

    assert rld_fb.points[-1] == Point(feedback_pin.x, feedback_pin.y)
    assert rld_fb.points[-1] != rld_fb.points[-3]
    assert not route_crosses_keepout(rld_fb.points, components["Aau"].bbox.expanded(-0.06))


def test_hint_router_flips_crossed_opamp_inputs_and_reanchors_ground_leg() -> None:
    layout = plan_layout(ecg_rld_frontend(), planning_client=FakePlanningClient(ecg_rld_hints()))
    components = {component.id: component for component in layout.components}

    assert components["A3"].orientation == "right_flip"
    assert components["R6"].orientation == "right"
    assert components["A4"].x < components["R6"].x
    assert components["A4"].y < components["R6"].y


def test_feedback_policy_uses_outer_corridor_instead_of_opamp_body() -> None:
    hints = ecg_rld_hints().with_updates(
        route_policies=(
            *ecg_rld_hints().route_policies,
            RoutePolicyHint(net="ecg_out", net_role="feedback", policy="top_feedback_corridor"),
        )
    )
    layout = plan_layout(ecg_rld_frontend(), planning_client=FakePlanningClient(hints))
    components = {component.id: component for component in layout.components}
    ecg_out = next(wire for wire in layout.wires if wire.net == "ecg_out")

    assert min(point.y for point in ecg_out.points) < components["R5"].bbox.y
    assert not route_crosses_keepout(ecg_out.points, components["A4"].bbox.expanded(-0.06))


def test_bottom_auxiliary_output_keeps_feedback_resistor_as_local_branch() -> None:
    layout = plan_layout(ecg_rld_frontend(), planning_client=FakePlanningClient(ecg_rld_hints()))
    components = {component.id: component for component in layout.components}
    rld = next(wire for wire in layout.wires if wire.net == "rld")
    driver = layout.pin_map[("Aau", "out")]
    feedback_pin = layout.pin_map[("Rf", "a")]
    bottom_entry = next(
        index
        for index, point in enumerate(rld.points)
        if point.y > components["Aau"].bbox.bottom + 0.25
    )
    feedback_index = next(
        index
        for index, point in enumerate(rld.points)
        if abs(point.x - feedback_pin.x) < 1e-6 and abs(point.y - feedback_pin.y) < 1e-6
    )

    assert rld.points[0] == Point(driver.x, driver.y)
    assert feedback_index < bottom_entry


def test_two_pin_auxiliary_leaf_stops_at_passive_pin_without_overshoot() -> None:
    layout = plan_layout(ecg_rld_frontend(), planning_client=FakePlanningClient(ecg_rld_hints()))
    components = {component.id: component for component in layout.components}
    rld_drive = next(wire for wire in layout.wires if wire.net == "rld_drive")
    ro_pin = layout.pin_map[("Ro", "b")]

    assert rld_drive.points[-1] == Point(ro_pin.x, ro_pin.y)
    assert max(point.y for point in rld_drive.points) <= ro_pin.y + 1e-6
    assert not route_crosses_keepout(rld_drive.points, components["Ro"].bbox.expanded(-0.06))


def test_opamp_supply_symbols_are_not_visible_in_main_svg() -> None:
    artifact = draw_artifact(ecg_rld_frontend(), planning_client=FakePlanningClient(ecg_rld_hints()))

    assert ">VEE<" not in artifact.svg
    assert ">VCC<" not in artifact.svg
    assert 'data-terminal-type="negative_supply"' in artifact.svg


def test_invalid_planning_hints_with_unknown_component_are_rejected() -> None:
    hints = ecg_rld_hints().with_updates(
        placements=(*ecg_rld_hints().placements, GridPlacementHint("HEART_ART", 1, 1, "RIGHT"))
    )

    assert legalize_planning_hints(ecg_rld_frontend(), hints) is None


def test_route_policy_cannot_turn_ground_or_supplies_into_global_wires() -> None:
    bad = ecg_rld_hints().with_updates(
        route_policies=(RoutePolicyHint(net="GND", net_role="ground", policy="bottom_auxiliary_corridor"),)
    )

    assert legalize_planning_hints(ecg_rld_frontend(), bad) is None


def test_no_planning_client_keeps_deterministic_behavior_unchanged() -> None:
    circuit = ecg_rld_frontend()
    first = plan_layout(circuit)
    second = plan_layout(circuit, planning_client=None)

    assert [(component.id, component.x, component.y) for component in first.components] == [
        (component.id, component.x, component.y) for component in second.components
    ]
    assert [wire.points for wire in first.wires] == [wire.points for wire in second.wires]


def test_hinted_layout_falls_back_if_hard_failures_increase() -> None:
    circuit = ecg_rld_frontend()
    base = plan_layout(circuit)
    bad = ecg_rld_hints().with_updates(
        placements=tuple(GridPlacementHint(component.id, 0, 0, "LEFT") for component in circuit.components)
    )
    planned = plan_layout(circuit, planning_hints=bad)

    assert [(component.id, component.x, component.y) for component in planned.components] == [
        (component.id, component.x, component.y) for component in base.components
    ]
    assert "planning_hints: accepted" not in planned.warnings


def test_route_contract_rejects_injected_diagonal_wire() -> None:
    layout = plan_layout(ecg_rld_frontend(), planning_client=FakePlanningClient(ecg_rld_hints()))
    assert_no_diagonal_wires(layout)
    bad = replace(layout, wires=[LayoutWire("diag", [Point(0, 0), Point(1, 1)], []), *layout.wires])

    report = critique_layout(bad)

    assert any(violation.code == "diagonal_wire" and violation.hard for violation in report.violations)


def test_visual_review_patch_cannot_change_topology() -> None:
    layout = plan_layout(ecg_rld_frontend(), planning_client=FakePlanningClient(ecg_rld_hints()))
    review = VisualReview(
        passed=False,
        score=30,
        recognized_topology="ecg",
        patches=[VisualPatch(action="move_component", component_id="NEW_FAKE", x=2, y=2)],
    )

    patch = layout_patch_from_visual_review(review, layout)

    assert not patch.move_component
    assert not patch.move_label


def test_visual_review_request_reroute_becomes_route_policy_patch() -> None:
    circuit = ecg_rld_frontend()
    layout = plan_layout(circuit, planning_client=FakePlanningClient(ecg_rld_hints()))
    review = VisualReview(
        passed=False,
        score=42,
        recognized_topology="ecg",
        patches=[VisualPatch(action="request_reroute", net="ecg_out", corridor="top_feedback_corridor")],
    )

    patch = layout_patch_from_visual_review(review, layout)
    rerouted = apply_patch(circuit, layout, patch)

    assert patch.set_route_policy[0].net == "ecg_out"
    assert patch.set_route_policy[0].policy == "top_feedback_corridor"
    assert any(item["net"] == "ecg_out" for item in rerouted.support.planning_hints["route_policies"])


def test_local_optimizer_searches_opamp_flip_candidates() -> None:
    circuit = ecg_rld_frontend()
    layout = plan_layout(circuit, planning_client=FakePlanningClient(ecg_rld_hints()))
    forced_right = apply_patch(circuit, layout, LayoutPatch(set_orientation=[SetOrientation("A3", "right")]))

    patch = propose_local_patch(forced_right, critique_layout(forced_right))

    assert any(item.component_id == "A3" and item.orientation == "right_flip" for item in patch.set_orientation)


def test_block_decomposition_hints_round_trip_and_legalize() -> None:
    raw = ecg_rld_hints().to_dict()
    raw["blocks"] = [
        BlockHint(
            block_id="first_stage_buffers",
            block_type="parallel_opamp_buffers",
            members=("A1", "A2", "R1", "R2"),
            stage_x=1,
            lane_y=0,
            ports={"in_plus": "e_p", "in_minus": "e_n", "out_plus": "buf_p", "out_minus": "buf_n"},
        ).to_dict()
    ]
    raw["block_internal_motifs"] = [{"motif_type": "parallel_input_buffer_pair", "members": ["A1", "A2"]}]
    raw["inter_block_routes"] = [
        {"net": "buf_p", "from": "first_stage_buffers.out_plus", "to": "rld_aux_loop.sense", "policy": "bottom_auxiliary_corridor"}
    ]
    raw["auxiliary_loops"] = [
        {"loop_id": "rld_aux_loop", "loop_type": "right_leg_drive", "members": ["Aau", "Rf", "Ro"], "nets": ["rld", "rld_drive"]}
    ]
    raw["orientation_overrides"] = [OrientationOverrideHint("A3", "RIGHT_FLIP", "feedback should stay outside body").to_dict()]

    hints = SchematicLayoutHints.from_dict(raw)
    legalized = legalize_planning_hints(ecg_rld_frontend(), hints)

    assert hints.blocks[0].ports["out_plus"] == "buf_p"
    assert any(route.net == "buf_p" for route in hints.inter_block_routes)
    assert any(policy.net == "buf_p" and policy.policy == "bottom_auxiliary_corridor" for policy in hints.route_policies)
    assert legalized is not None


def test_block_internal_placement_shapes_auxiliary_and_feedback_blocks() -> None:
    raw = ecg_rld_hints().to_dict()
    raw["blocks"] = [
        {
            "block_id": "input_buffers",
            "block_type": "parallel_opamp_buffers",
            "members": ["INP", "INN", "A1", "A2", "R1", "R2"],
            "stage_x": 1,
            "lane_y": 0,
            "ports": {"out_plus": "buf_p", "out_minus": "buf_n"},
        },
        {
            "block_id": "final_gain",
            "block_type": "opamp_feedback_stage",
            "members": ["A4", "R5", "R6"],
            "stage_x": 5,
            "lane_y": 0,
            "ports": {"in": "ac", "out": "ecg_out"},
        },
        {
            "block_id": "rld_aux",
            "block_type": "auxiliary_feedback_loop",
            "members": ["Aau", "Rf", "Ro", "RL"],
            "stage_x": 3,
            "lane_y": 4,
            "ports": {"sense": "buf_p", "drive": "rld_drive"},
        },
    ]

    layout = plan_layout(ecg_rld_frontend(), planning_hints=SchematicLayoutHints.from_dict(raw))
    components = {component.id: component for component in layout.components}

    assert "planning_hints: accepted" in layout.warnings
    assert components["A1"].x == components["A2"].x
    assert components["R1"].x > components["A1"].bbox.right
    assert components["R2"].x > components["A2"].bbox.right
    assert components["R5"].y < components["A4"].y
    assert components["R6"].y > components["A4"].y
    assert components["R6"].x < components["A4"].x
    assert components["Aau"].y > components["A4"].y
    assert components["Rf"].y < components["Aau"].y
    assert components["Ro"].y > components["Aau"].y
    assert components["RL"].x > components["Ro"].x
    buf_p = next(wire for wire in layout.wires if wire.net == "buf_p")
    local_terminals = [
        terminal
        for terminal in layout.semantic.local_terminals
        if terminal.net == "buf_p" and terminal.terminal_type == "signal_label"
    ]
    violation_codes = {violation.code for violation in critique_layout(layout).violations}

    assert ("Aau", "+") not in buf_p.connected_pins
    assert any(terminal.component_id == "Aau" and terminal.pin_name == "+" for terminal in local_terminals)
    assert min(point.x for point in buf_p.points) > components["INP"].bbox.x
    assert "wire_crossings" not in violation_codes
    assert "wire_through_component" not in violation_codes
    assert "wire_net_overlap" not in violation_codes


def test_auxiliary_sense_signal_label_is_rendered_instead_of_long_wire() -> None:
    raw = ecg_rld_hints().to_dict()
    raw["blocks"] = [
        {
            "block_id": "input_buffers",
            "block_type": "parallel_opamp_buffers",
            "members": ["INP", "INN", "A1", "A2", "R1", "R2"],
            "stage_x": 1,
            "lane_y": 0,
            "ports": {"out_plus": "buf_p", "out_minus": "buf_n"},
        },
        {
            "block_id": "rld_aux",
            "block_type": "auxiliary_feedback_loop",
            "members": ["Aau", "Rf", "Ro", "RL"],
            "stage_x": 3,
            "lane_y": 4,
            "ports": {"sense": "buf_p", "drive": "rld_drive"},
        },
    ]

    artifact = draw_artifact(ecg_rld_frontend(), planning_client=FakePlanningClient(SchematicLayoutHints.from_dict(raw)))

    assert 'data-terminal-type="signal_label"' in artifact.svg
    assert ">buf_p<" in artifact.svg


def test_monitor_outputs_become_signal_labels_not_physical_wire_targets() -> None:
    layout = plan_layout(monitor_output_fixture())
    monitor_terminals = [
        terminal
        for terminal in layout.semantic.local_terminals
        if terminal.component_id == "VMON" and terminal.terminal_type == "signal_label"
    ]
    vout = next(wire for wire in layout.wires if wire.net == "vout")

    assert monitor_terminals
    assert ("VMON", "in") not in vout.connected_pins
    assert ("VOUT", "in") in vout.connected_pins


def test_chapter6_ecg_monitor_labels_attach_to_source_nodes() -> None:
    layout = plan_layout(chapter6_ecg_frontend(), planning_hints=chapter6_ecg_hints())
    components = {component.id: component for component in layout.components}
    v3_wire = next(wire for wire in layout.wires if wire.net == "v3_cm")
    v3_terminals = {
        (terminal.component_id, terminal.pin_name, terminal.label)
        for terminal in layout.semantic.local_terminals
        if terminal.net == "v3_cm" and terminal.terminal_type == "signal_label"
    }

    assert "planning_hints: accepted" in layout.warnings
    assert components["V2_MON"].x < components["A3"].x
    assert components["V4_MON"].x < components["A4"].x
    assert components["V3_MON"].x < components["A3"].x
    assert components["AUX_OUT_MON"].x < components["RL_ELECTRODE"].x
    assert set(v3_wire.connected_pins) == {("RA_TOP", "b"), ("RA_BOT", "a")}
    assert (layout.pin_map[("RF_AUX", "a")].x, layout.pin_map[("RF_AUX", "a")].y) in {point.as_tuple() for point in v3_wire.points}
    assert (layout.pin_map[("A_AUX", "-")].x, layout.pin_map[("A_AUX", "-")].y) in {point.as_tuple() for point in v3_wire.points}
    assert ("A_AUX", "-", "V3") in v3_terminals
    assert ("RF_AUX", "a", "V3") in v3_terminals


def test_chapter6_ecg_auto_structural_fallback_without_external_hints() -> None:
    layout = plan_layout(chapter6_ecg_frontend())
    components = {component.id: component for component in layout.components}
    report = critique_layout(layout)
    v4_ac = next(wire for wire in layout.wires if wire.net == "v4_ac")
    a4_feedback_node = next(wire for wire in layout.wires if wire.net == "a4_feedback_node")
    a4_ground_pins = [
        layout.pin_map[(terminal.component_id, terminal.pin_name)]
        for terminal in layout.semantic.local_terminals
        if terminal.terminal_type == "ground" and terminal.component_id in {"S1", "R7_BIAS_A4", "R5"}
    ]
    gain_spine = [components["R2_TOP"], components["R1_GAIN"], components["R2_BOT"]]

    assert "planning_hints: accepted" in layout.warnings
    assert layout.support.planning_hints["source"] == "deterministic"
    assert layout.support.planning_hints["recognized_topology"] == "ecg_frontend_right_leg_drive"
    assert not any(violation.hard for violation in report.violations)
    assert components["A1"].x < components["A3"].x < components["A4"].x
    assert components["V3_MON"].x < components["A3"].x
    assert all(component.orientation == "down" for component in gain_spine)
    assert max(component.x for component in gain_spine) - min(component.x for component in gain_spine) <= 0.05
    assert components["R2_TOP"].y < components["R1_GAIN"].y < components["R2_BOT"].y
    assert components["R5"].orientation == "right"
    assert components["R5"].y < components["A4"].y
    assert max(point.y for point in a4_feedback_node.points) < min(point.y for point in v4_ac.points)
    assert v4_ac.points == sorted(v4_ac.points, key=lambda point: (point.x, point.y))
    assert len({round(point.y, 3) for point in v4_ac.points}) == 1
    assert len(a4_ground_pins) == 3
    assert all(abs(left.x - right.x) >= 0.75 or abs(left.y - right.y) >= 0.9 for left, right in combinations(a4_ground_pins, 2))


def test_chapter6_ecg_planned_crossing_renders_as_jump_bridge() -> None:
    artifact = draw_artifact(chapter6_ecg_frontend())

    assert artifact.critic_report is not None
    assert not artifact.critic_report["hard_fail"]
    assert " Q " in artifact.svg


def test_tutor_artifact_exposes_bounding_boxes_hints_and_explanation() -> None:
    artifact = draw_artifact(ecg_rld_frontend(), planning_client=FakePlanningClient(ecg_rld_hints()))

    assert artifact.components["Aau"]["bbox"]["height"] > 0
    assert artifact.labels["label:Aau"]["bbox"]["width"] > 0
    assert artifact.semantic_plan["stages"]
    assert artifact.planning_hints_used
    assert "right-leg-drive" in artifact.tutor_explanation
    assert artifact.layout_confidence >= 0.9
    assert artifact.fallback_used is False


def test_visual_review_result_is_exposed_on_optimized_artifact(monkeypatch) -> None:
    review = VisualReview(passed=True, score=96, recognized_topology="ecg", visual_errors=[], patches=[])
    monkeypatch.setattr("optcpv.optimizer.propose_local_patch", lambda layout, report: __import__("optcpv.patch").patch.LayoutPatch())

    artifact = draw_optimized_artifact(
        ecg_rld_frontend(),
        max_iterations=1,
        planning_client=FakePlanningClient(ecg_rld_hints()),
        visual_review_client=FakeVisualReviewClient(review),
    )

    assert artifact.visual_review_result["passed"] is True
    assert artifact.visual_review_result["recognized_topology"] == "ecg"


def test_image_guided_and_model_guided_requests_share_hint_schema() -> None:
    circuit = ecg_rld_frontend()
    image_request = SchematicPlanningRequest.from_circuit(circuit, input_mode="image_guided", reference_image={"mime": "image/png"})
    model_request = SchematicPlanningRequest.from_circuit(circuit, input_mode="model_guided")
    image_hints = SchematicLayoutHints.from_dict(ecg_rld_hints().with_updates(source="gemini").to_dict())
    model_hints = SchematicLayoutHints.from_dict(ecg_rld_hints().with_updates(source="deterministic").to_dict())

    assert image_request.input_mode == "image_guided"
    assert model_request.input_mode == "model_guided"
    assert set(image_hints.to_dict()) == set(model_hints.to_dict())
    assert len(image_hints.placements) == len(model_hints.placements)


def test_reference_image_annotations_are_not_required_components() -> None:
    raw = ecg_rld_hints().to_dict()
    raw["placements"] = [*raw["placements"], {"component_id": "torso_annotation", "stage_x": 0, "lane_y": 0, "orientation": "RIGHT"}]

    assert legalize_planning_hints(ecg_rld_frontend(), SchematicLayoutHints.from_dict(raw)) is None


def ecg_rld_frontend() -> Circuit:
    return Circuit(
        id="ecg_rld_frontend",
        motif="op_amp_network",
        components=[
            Component(id="INP", type="input", pins={"out": "e_p"}, label="RA"),
            Component(id="INN", type="input", pins={"out": "e_n"}, label="LA"),
            Component(id="RL", type="output", pins={"in": "rld_drive"}, label="RL"),
            Component(id="VOUT", type="output", pins={"in": "ecg_out"}, label="OUT"),
            Component(id="A1", type="op_amp", pins={"+": "e_p", "-": "buf_p", "out": "buf_p", "v+": "VCC", "v-": "VEE"}, role="buffer"),
            Component(id="A2", type="op_amp", pins={"+": "e_n", "-": "buf_n", "out": "buf_n", "v+": "VCC", "v-": "VEE"}, role="buffer"),
            Component(id="R1", type="resistor", pins={"a": "buf_p", "b": "diff_p"}),
            Component(id="R2", type="resistor", pins={"a": "buf_n", "b": "diff_n"}),
            Component(id="A3", type="op_amp", pins={"+": "diff_p", "-": "diff_n", "out": "diff_out", "v+": "VCC", "v-": "VEE"}, role="differential gain"),
            Component(id="C1", type="capacitor", pins={"a": "diff_out", "b": "ac"}, role="ac coupling"),
            Component(id="A4", type="op_amp", pins={"+": "ac", "-": "gain_fb", "out": "ecg_out", "v+": "VCC", "v-": "VEE"}, role="final gain filter"),
            Component(id="R5", type="resistor", pins={"a": "ecg_out", "b": "gain_fb"}, role="feedback"),
            Component(id="R6", type="resistor", pins={"a": "gain_fb", "b": "GND"}, role="gain"),
            Component(id="Aau", type="op_amp", pins={"+": "buf_p", "-": "rld_fb", "out": "rld", "v+": "VCC", "v-": "VEE"}, role="right-leg-drive auxiliary feedback"),
            Component(id="Rf", type="resistor", pins={"a": "rld", "b": "rld_fb"}, role="feedback"),
            Component(id="Ro", type="resistor", pins={"a": "rld", "b": "rld_drive"}, role="output resistor"),
            Component(id="GND", type="ground", pins={"gnd": "GND"}, label="GND"),
        ],
    )


def monitor_output_fixture() -> Circuit:
    return Circuit(
        id="monitor_output_fixture",
        motif="non_inverting_op_amp",
        components=[
            Component(id="VIN", type="input", pins={"out": "vin"}),
            Component(id="VOUT", type="output", pins={"in": "vout"}, label="OUT"),
            Component(id="VMON", type="output", pins={"in": "vout"}, label="Vmon", role="monitor_output"),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}),
            Component(id="U1", type="op_amp", pins={"+": "vin", "-": "vm", "out": "vout"}),
            Component(id="Rf", type="resistor", pins={"a": "vout", "b": "vm"}, role="feedback"),
            Component(id="Rg", type="resistor", pins={"a": "vm", "b": "gnd"}, role="gain"),
        ],
    )


def chapter6_ecg_frontend() -> Circuit:
    def r(cid: str, a: str, b: str, label: str, role: str | None = None) -> Component:
        return Component(id=cid, type="resistor", pins={"a": a, "b": b}, label=label, role=role)

    def c(cid: str, a: str, b: str, label: str, role: str | None = None) -> Component:
        return Component(id=cid, type="capacitor", pins={"a": a, "b": b}, label=label, role=role)

    def op(cid: str, plus: str, minus: str, out: str, label: str) -> Component:
        return Component(id=cid, type="op_amp", pins={"+": plus, "-": minus, "out": out}, label=label)

    def output(cid: str, net: str, label: str, role: str = "monitor_output") -> Component:
        return Component(id=cid, type="output", pins={"in": net}, label=label, role=role)

    return Circuit(
        id="chapter6_ecg_frontend_test",
        motif="op_amp_network",
        components=[
            Component(id="E_PLUS", type="input", pins={"out": "lead_plus"}, label="E+"),
            Component(id="E_MINUS", type="input", pins={"out": "lead_minus"}, label="E-"),
            op("A1", "lead_plus", "a1_inv", "v2", "A1"),
            op("A2", "lead_minus", "a2_inv", "a2_out", "A2"),
            r("R2_TOP", "v2", "a1_inv", "R2"),
            r("R1_GAIN", "a1_inv", "a2_inv", "R1", role="gain"),
            r("R2_BOT", "a2_inv", "a2_out", "R2"),
            r("RA_TOP", "v2", "v3_cm", "Ra", role="common_mode_sense"),
            r("RA_BOT", "v3_cm", "a2_out", "Ra", role="common_mode_sense"),
            op("A3", "a3_plus", "a3_minus", "v4_raw", "A3"),
            r("R3_MINUS", "v2", "a3_minus", "R3"),
            r("R4_FEEDBACK_A3", "v4_raw", "a3_minus", "R4", role="feedback"),
            r("R3_PLUS", "a2_out", "a3_plus", "R3"),
            r("R4_PLUS_GND", "a3_plus", "gnd", "R4"),
            c("C1", "v4_raw", "v4_ac", "C1"),
            Component(id="S1", type="switch", pins={"a": "v4_ac", "b": "gnd"}, label="S1"),
            r("R7_BIAS_A4", "v4_ac", "gnd", "R7"),
            op("A4", "v4_ac", "a4_minus", "v5", "A4"),
            r("R5", "gnd", "a4_feedback_node", "R5"),
            r("R7_A4_MINUS", "a4_feedback_node", "a4_minus", "R7"),
            r("R6", "a4_feedback_node", "v5", "R6", role="feedback"),
            c("C2", "a4_feedback_node", "v5", "C2", role="feedback"),
            op("A_AUX", "gnd", "v3_cm", "aux_out", "Aau"),
            r("RF_AUX", "v3_cm", "right_leg_drive", "Rf", role="feedback"),
            r("RO_AUX", "aux_out", "right_leg_drive", "Ro"),
            r("R_RL", "right_leg_drive", "right_leg_electrode", "R_RL"),
            output("V2_MON", "v2", "V2"),
            output("V3_MON", "v3_cm", "V3"),
            output("V4_MON", "v4_raw", "V4"),
            output("V5_OUT", "v5", "V5", role="final_output"),
            output("AUX_OUT_MON", "aux_out", "v_out"),
            output("VO_MON", "right_leg_drive", "Vo"),
            output("RL_ELECTRODE", "right_leg_electrode", "RL", role="body_terminal"),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}, label="GND"),
        ],
    )


def chapter6_ecg_hints() -> SchematicLayoutHints:
    placements = [
        {"component_id": component.id, "stage_x": 0, "lane_y": 0, "orientation": "RIGHT"}
        for component in chapter6_ecg_frontend().components
    ]
    return SchematicLayoutHints.from_dict(
        {
            "recognized_topology": "ECG_frontend_with_instrumentation_amplifier_and_right_leg_drive",
            "confidence": 0.95,
            "placements": placements,
            "blocks": [
                {
                    "block_id": "IA_FRONT_END",
                    "block_type": "differential_input_pair",
                    "members": ["E_PLUS", "E_MINUS", "A1", "A2", "R2_TOP", "R1_GAIN", "R2_BOT", "RA_TOP", "RA_BOT"],
                    "stage_x": 1,
                    "lane_y": 0,
                    "ports": {"out_v2": "v2", "out_a2_out": "a2_out", "out_v3_cm": "v3_cm"},
                },
                {
                    "block_id": "DIFF_AMP",
                    "block_type": "opamp_feedback_stage",
                    "members": ["A3", "R3_MINUS", "R4_FEEDBACK_A3", "R3_PLUS", "R4_PLUS_GND"],
                    "stage_x": 2,
                    "lane_y": 0,
                    "ports": {"out_v4_raw": "v4_raw"},
                },
                {
                    "block_id": "AC_COUPLE",
                    "block_type": "rc_filter_or_passive_filter",
                    "members": ["C1", "S1", "R7_BIAS_A4"],
                    "stage_x": 3,
                    "lane_y": 0,
                    "ports": {"out_v4_ac": "v4_ac"},
                },
                {
                    "block_id": "ACTIVE_FILTER",
                    "block_type": "opamp_feedback_stage",
                    "members": ["A4", "R5", "R7_A4_MINUS", "R6", "C2"],
                    "stage_x": 4,
                    "lane_y": 0,
                    "ports": {"out_v5": "v5"},
                },
                {
                    "block_id": "RLD_LOOP",
                    "block_type": "auxiliary_feedback_loop",
                    "members": ["A_AUX", "RF_AUX", "RO_AUX", "R_RL", "RL_ELECTRODE"],
                    "stage_x": 2,
                    "lane_y": 2,
                    "ports": {"in_v3_cm": "v3_cm", "out_rld_drive_net": "right_leg_drive"},
                    "route_policy": "bottom_auxiliary_corridor",
                },
            ],
            "route_policies": [{"net": "gnd", "net_role": "ground", "policy": "local_terminal_only"}],
        }
    )


def ecg_rld_hints() -> PlanningHints:
    placements = {
        "INP": (0, -2),
        "INN": (0, 1),
        "A1": (1, -2),
        "A2": (1, 1),
        "R1": (2, -2),
        "R2": (2, 1),
        "A3": (3, 0),
        "C1": (4, 0),
        "A4": (5, 0),
        "R5": (5, -2),
        "R6": (5, 2),
        "VOUT": (6, 0),
        "Aau": (3, 4),
        "Rf": (4, 4),
        "Ro": (5, 4),
        "RL": (6, 4),
        "GND": (4, 6),
    }
    return PlanningHints(
        recognized_topology="ECG analog front end with right-leg-drive loop",
        confidence=0.9,
        tutor_explanation="Differential ECG inputs flow left to right while the right-leg-drive loop is isolated in the lower auxiliary lane.",
        placements=[GridPlacementHint(component_id, stage_x, lane_y, "RIGHT") for component_id, (stage_x, lane_y) in placements.items()],
        route_policies=[RoutePolicyHint(net="rld", net_role="right_leg_drive", policy="bottom_auxiliary_corridor")],
        local_terminal_policy={"GND": "local_symbol_only", "VCC": "local_symbol_only", "VEE": "local_symbol_only"},
    )
