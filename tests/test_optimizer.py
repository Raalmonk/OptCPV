from dataclasses import replace
import xml.etree.ElementTree as ET

from optcpv import Circuit, Component, draw_optimized_artifact, draw_optimized_svg
from optcpv.critic import critique
from optcpv.examples import instrumentation_amplifier
from optcpv.models import LayoutWire, Point
from optcpv.optimizer import propose_local_patch
from optcpv.patch import LayoutPatch, MoveComponent
from optcpv.planning_agent import SemanticPlanningClient
from optcpv.planner import plan_layout
from optcpv.renderer import render_svg
from optcpv.vision_agent import VisionLayoutClient
from optcpv.vector_critic import critique_layout


def test_optimizer_improves_instrumentation_amplifier_without_canvas_growth() -> None:
    circuit = instrumentation_amplifier()
    raw_layout = plan_layout(circuit)
    raw_report = critique(circuit, raw_layout, render_svg(raw_layout))
    optimized = draw_optimized_artifact(circuit)

    assert optimized.viewbox == {"x": 0, "y": 0, "width": raw_layout.width, "height": raw_layout.height}
    assert optimized.critic_report["score"] <= raw_report.score - 20 or optimized.critic_report["score"] <= 20
    assert optimized.optimization_log


def test_optimized_output_has_no_component_overlaps_and_healthy_fill() -> None:
    optimized = draw_optimized_artifact(instrumentation_amplifier())

    assert not any(v["code"] == "component_overlap" for v in optimized.critic_report["violations"])
    assert not any(v["code"] == "label_visual_collision" for v in optimized.critic_report["violations"])
    assert not any(v["code"] == "wire_component_visual_collision" for v in optimized.critic_report["violations"])
    assert optimized.critic_report["metrics"]["vector.component_fill_ratio"] >= 0.025
    assert optimized.cv_report["metrics"]["label_layer_collision_count"] == 0
    assert optimized.cv_report["metrics"]["wire_component_mask_hits"] == 0


def test_draw_optimized_svg_returns_valid_svg_for_instrumentation_amp() -> None:
    svg = draw_optimized_svg(instrumentation_amplifier())
    root = ET.fromstring(svg)

    assert root.tag.endswith("svg")
    assert 'data-renderer="optcpv.schemdraw"' in svg
    assert "data-component-id" in svg
    assert "data-net-name" in svg
    assert "data-pin-name" in svg


def test_artifact_reports_are_split() -> None:
    artifact = draw_optimized_artifact(instrumentation_amplifier())

    assert artifact.critic_report == artifact.combined_report
    assert artifact.vector_report is not None
    assert artifact.cv_report is not None
    assert "component_fill_ratio" in artifact.vector_report["metrics"]
    assert "label_layer_collision_count" in artifact.cv_report["metrics"]
    assert "cv.label_layer_collision_count" in artifact.combined_report["metrics"]
    assert artifact.cv_report != artifact.combined_report


def test_optimizer_asks_vision_client_after_local_patch_fails(monkeypatch) -> None:
    class RecordingVisionClient(VisionLayoutClient):
        def __init__(self) -> None:
            self.calls = 0

        def propose_patch(self, circuit, layout, svg, raster, critic_report):
            self.calls += 1
            return LayoutPatch()

    client = RecordingVisionClient()
    monkeypatch.setattr("optcpv.optimizer.propose_local_patch", lambda layout, report: LayoutPatch())
    artifact = draw_optimized_artifact(instrumentation_amplifier(), max_iterations=1, vision_client=client)

    assert client.calls == 1
    assert any(item["source"] == "vision" and item["reason"] == "empty_patch" for item in artifact.optimization_log)


def test_optimizer_asks_planning_client_for_refinement_after_local_patch_fails(monkeypatch) -> None:
    class RecordingPlanningClient(SemanticPlanningClient):
        def __init__(self) -> None:
            self.refine_calls = 0

        def propose_hints(self, circuit, reference_image=None):
            return None

        def refine_hints(self, circuit, layout, svg, critic_report, reference_image=None):
            self.refine_calls += 1
            assert svg.startswith("<svg") or "<svg" in svg
            assert critic_report.to_dict()["score"] >= 0
            return None

    client = RecordingPlanningClient()
    monkeypatch.setattr("optcpv.optimizer.propose_local_patch", lambda layout, report: LayoutPatch())
    monkeypatch.setattr("optcpv.optimizer._should_request_refinement", lambda report: True)
    artifact = draw_optimized_artifact(instrumentation_amplifier(), max_iterations=1, planning_client=client)

    assert client.refine_calls == 1
    assert any(
        item["source"] == "planning_refinement" and item["reason"] == "no_refinement_hints"
        for item in artifact.optimization_log
    )


def test_optimizer_rejects_illegal_vision_patch(monkeypatch) -> None:
    class BadVisionClient(VisionLayoutClient):
        def propose_patch(self, circuit, layout, svg, raster, critic_report):
            return LayoutPatch(move_component=[MoveComponent("U1", 100.0, 100.0)])

    monkeypatch.setattr("optcpv.optimizer.propose_local_patch", lambda layout, report: LayoutPatch())
    artifact = draw_optimized_artifact(instrumentation_amplifier(), max_iterations=1, vision_client=BadVisionClient())

    assert any(item["source"] == "vision" and item["accepted"] is False for item in artifact.optimization_log)


def test_local_optimizer_requests_reroute_for_wire_through_component() -> None:
    circuit = Circuit(
        id="two_stage_reroute_repair",
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
    layout = plan_layout(circuit)
    blocker = next(component for component in layout.components if component.id == "U1")
    bad_wire = LayoutWire(
        "o2",
        [Point(blocker.bbox.x - 1.0, blocker.y), Point(blocker.bbox.right + 2.0, blocker.y)],
        layout.net_to_pins["o2"],
    )
    bad_layout = replace(layout, wires=[bad_wire, *(wire for wire in layout.wires if wire.net != "o2")])
    report = critique_layout(bad_layout)

    patch = propose_local_patch(bad_layout, report)

    assert any(violation.code == "wire_through_component" for violation in report.violations)
    assert any(policy.net == "o2" and policy.policy == "top_feedback_corridor" for policy in patch.set_route_policy)
