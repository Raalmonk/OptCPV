from dataclasses import replace

from optcpv import (
    Circuit,
    Component,
    FakePlanningClient,
    FakeVisualReviewClient,
    GridPlacementHint,
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
from optcpv.planner import plan_layout
from optcpv.route_contract import assert_no_diagonal_wires, route_crosses_keepout
from optcpv.vector_critic import critique_layout
from optcpv.visual_review import layout_patch_from_visual_review


def test_fake_planning_client_places_ecg_rld_loop_in_bottom_auxiliary_lane() -> None:
    circuit = ecg_rld_frontend()
    layout = plan_layout(circuit, planning_client=FakePlanningClient(ecg_rld_hints()))
    components = {component.id: component for component in layout.components}
    rld = next(wire for wire in layout.wires if wire.net == "rld")

    assert "planning_hints: accepted" in layout.warnings
    assert components["A1"].x == components["A2"].x
    assert components["A1"].y != components["A2"].y
    assert components["A1"].x < components["A3"].x < components["A4"].x
    assert components["Aau"].y > components["A4"].y
    assert max(point.y for point in rld.points) > components["Aau"].bbox.bottom
    assert not any(route_crosses_keepout(rld.points, components[item].bbox.expanded(-0.06)) for item in ["Aau", "A3", "A4"])


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
